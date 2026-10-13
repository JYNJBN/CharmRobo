"""设备千问协议回归：模拟上游/数据库；验证提交、RAG 顺序、取消及级联回退。"""

import asyncio
import base64
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import SecretStr
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session as SqlSession

from app.api.v1 import hardware_voice as gateway
from app.integrations import hardware_qwen_realtime as qwen
from app.integrations.hardware_mp3_bridge import PcmToCiMp3Bridge
from app.models.conversation import Conversation
from app.models.conversation_message import ConversationMessage
from app.services import conversation_service
from app.services import hardware_realtime_context as context_service
from app.utils.tools import build_current_time_context


class FakeUpstream:
    def __init__(
        self, status="completed", transcribe=True, transcript="酒店早餐几点？"
    ):
        self.sent = []
        self.queue = asyncio.Queue()
        self.closed = False
        self.status = status
        self.transcribe = transcribe
        self.transcript = transcript
        self.reply = "早餐七点开始。"
        self.turn_number = 0
        self.queue.put_nowait({"type": "session.created"})

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        kind = event["type"]
        if kind == "session.update":
            session = event.get("session") or {}
            if session.get(
                "enable_search"
            ) is False and qwen.ASR_EMPTY_REPEAT_PROMPT in session.get(
                "instructions", ""
            ):
                self.reply = qwen.ASR_EMPTY_REPEAT_PROMPT
            self.queue.put_nowait({"type": "session.updated"})
        elif kind == "input_audio_buffer.commit" and self.transcribe:
            self.turn_number += 1
            self.queue.put_nowait(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": self.transcript,
                    "item_id": f"user-{self.turn_number}",
                }
            )
        elif kind == "response.create":
            for reply in (
                {"type": "response.audio_transcript.delta", "delta": self.reply},
                {"type": "response.text.delta", "delta": "重复来源不可重复保存"},
                {
                    "type": "response.audio.delta",
                    "delta": base64.b64encode(b"\0" * 100).decode(),
                },
                {
                    "type": "response.done",
                    "response": {
                        "status": self.status,
                        "output": [{"id": f"assistant-{self.turn_number}", "role": "assistant"}],
                        "usage": {"plugins": {"search": {"count": 1}}},
                    },
                },
            ):
                self.queue.put_nowait(reply)
        elif kind == "input_audio_buffer.clear":
            self.queue.put_nowait({"type": "input_audio_buffer.cleared"})
        elif kind == "conversation.item.delete":
            self.queue.put_nowait({"type": "conversation.item.deleted", "item_id": event["item_id"]})

    async def recv(self):
        return json.dumps(await self.queue.get(), ensure_ascii=False)

    async def close(self):
        self.closed = True


class FakeBridge:
    def __init__(self, **kwargs):
        self.callback = kwargs["on_mp3"]
        self.closed = False

    def write(self, pcm):
        self.callback(b"mock-mp3")  # write 在 asyncio.to_thread 中运行。
        return True

    def close(self, **kwargs):
        self.closed = True
        return 0, ""


class DelayedAudioUpstream(FakeUpstream):
    """文字先到、音频迟到或只返回部分语音的上游。"""

    def __init__(self, delay=0.05, finish=True):
        super().__init__()
        self.delay = delay
        self.finish = finish
        self.delivery = None

    async def send(self, raw):
        event = json.loads(raw)
        if event["type"] != "response.create":
            return await super().send(raw)
        self.sent.append(event)
        self.queue.put_nowait(
            {"type": "response.created", "response": {"id": "test-response-id"}}
        )
        self.queue.put_nowait(
            {"type": "response.audio_transcript.delta", "delta": "延迟返回的语音。"}
        )

        async def deliver():
            await asyncio.sleep(self.delay)
            self.queue.put_nowait(
                {
                    "type": "response.audio.delta",
                    "delta": base64.b64encode(b"\0" * 100).decode(),
                }
            )
            if self.finish:
                self.queue.put_nowait(
                    {"type": "response.done", "response": {"status": "completed"}}
                )

        self.delivery = asyncio.create_task(deliver())

    async def close(self):
        if self.delivery is not None and not self.delivery.done():
            self.delivery.cancel()
            await asyncio.gather(self.delivery, return_exceptions=True)
        await super().close()


class FakeDevice:
    def __init__(
        self, route=None, turns=1, cancel=False, size=None, audio_format="pcm",
        device_sn="test-device",
    ):
        self.query_params = {}
        self.headers = {}
        self.messages = []
        self.audio = []
        self.queue = asyncio.Queue()
        self.route, self.turns, self.cancel = route, turns, cancel
        self.size, self.audio_format = size, audio_format
        self.completed = 0
        self.device_sn = device_sn
        self.start()

    def start(self):
        message = {
            "type": "dialogue.start",
            "device_sn": self.device_sn,
            "audio_format": self.audio_format,
        }
        if self.route:
            message["route"] = self.route
        if self.size is not None:
            message["audio_bytes"] = self.size
        self.control(message)

    def control(self, message):
        self.queue.put_nowait(
            {"type": "websocket.receive", "text": json.dumps(message)}
        )

    async def accept(self):
        pass

    async def receive(self):
        return await self.queue.get()

    async def send_json(self, message):
        self.messages.append(message)
        if message["type"] == "dialogue.ready":
            self.queue.put_nowait({"type": "websocket.receive", "bytes": b"\0" * 4})
            self.control(
                {"type": "dialogue.cancel" if self.cancel else "dialogue.commit"}
            )
        elif message["type"] == "dialogue.done":
            self.completed += 1
            if self.completed < self.turns:
                self.start()
            else:
                self.queue.put_nowait({"type": "websocket.disconnect"})
        elif message["type"] in {"dialogue.error", "dialogue.cancelled"}:
            self.queue.put_nowait({"type": "websocket.disconnect"})

    async def send_bytes(self, data):
        self.audio.append(data)


def business_session(agent_id=9):
    return SimpleNamespace(
        device_sn="test-device",
        user_id=1,
        device_id=2,
        agent_id=agent_id,
        agent_name=f"Agent{agent_id}",
        agent_system_prompt="酒店前台",
        history_messages=[{"role": "user", "content": "历史问题"}],
        conversation_id=agent_id,
        model={},
        voice="other-voice",
    )


class HardwareQwenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patchers = [
            patch.object(
                qwen.settings, "hardware_qwen_realtime_api_key", SecretStr("fake-key")
            ),
            patch.object(qwen.settings, "hardware_voice_route", "realtime"),
            patch.object(qwen, "PcmToCiMp3Bridge", FakeBridge),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_audio_streams_before_commit_and_rag_precedes_response(self):
        upstream = FakeUpstream()
        retrieved = asyncio.Event()
        allow_retrieval = asyncio.Event()

        async def retrieve(query):
            self.assertEqual(query, "酒店早餐几点？")
            retrieved.set()
            await allow_retrieval.wait()
            return "知识库：早餐七点开始。"

        output = []

        async def audio(chunk):
            output.append(chunk)

        session = qwen.HardwareQwenRealtimeSession(
            instructions="人设和历史",
            trace_id="test",
            send_audio=audio,
            retrieve_references=retrieve,
        )
        with patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)):
            await session.start()
            try:
                # PCM 样本跨帧拆分仍按顺序传输。
                self.assertTrue(session.append_audio(b"abc"))
                self.assertTrue(session.append_audio(b"d"))
                for _ in range(30):
                    await asyncio.sleep(0)
                    if len(upstream.sent) >= 3:
                        break
                self.assertEqual(
                    b"".join(
                        base64.b64decode(x["audio"])
                        for x in upstream.sent
                        if x["type"] == "input_audio_buffer.append"
                    ),
                    b"abcd",
                )
                session.commit()
                session.commit()
                self.assertFalse(session.append_audio(b"late"))
                await asyncio.wait_for(retrieved.wait(), 1)
                self.assertNotIn("response.create", [x["type"] for x in upstream.sent])
                allow_retrieval.set()
                result = await asyncio.wait_for(session.wait(), 2)
                self.assertEqual(result.answer_text, "早餐七点开始。")
                self.assertEqual(result.search_count, 1)
                for name in (
                    "asr_ms",
                    "retrieval_ms",
                    "context_update_ms",
                    "response_to_first_text_ms",
                    "response_to_first_pcm_ms",
                    "pcm_to_first_mp3_ms",
                    "end_to_first_audio_ms",
                    "mp3_drain_ms",
                ):
                    self.assertIn(name, result.timings_ms)
                    self.assertGreaterEqual(result.timings_ms[name], 0)
                self.assertEqual(result.endpoint_silence_bytes, 6400)
                self.assertEqual(output, [b"mock-mp3"])
                updates = [x for x in upstream.sent if x["type"] == "session.update"]
                self.assertEqual(updates[0]["session"]["voice"], "Tina")
                self.assertIsNone(updates[0]["session"]["turn_detection"])
                self.assertIn(
                    "知识库：早餐七点开始。", updates[-1]["session"]["instructions"]
                )
                self.assertEqual(
                    sum(
                        x["type"] == "input_audio_buffer.commit" for x in upstream.sent
                    ),
                    1,
                )
            finally:
                await session.aclose()
        self.assertTrue(upstream.closed)

    async def test_qwen_audio_flash_uses_builtin_asr_and_compatible_voice(self):
        class StrictAudio(FakeUpstream):
            async def send(self, raw):
                event_data = json.loads(raw)
                if (
                    event_data["type"] == "session.update"
                    and "voice" in event_data["session"]
                ):
                    config = event_data["session"]
                    if (
                        config["voice"] != "longanqian"
                        or "input_audio_transcription" in config
                    ):
                        raise ValueError(
                            "Audio 3.0 received incompatible Omni parameters"
                        )
                await super().send(raw)

        upstream = StrictAudio()
        with (
            patch.object(
                qwen.settings,
                "hardware_qwen_realtime_model",
                "qwen-audio-3.0-realtime-flash",
            ),
            patch.object(qwen.settings, "hardware_qwen_realtime_voice", "Tina"),
            patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)),
        ):
            retrieve = AsyncMock(return_value="知识库资料")
            audio = AsyncMock()
            session = qwen.HardwareQwenRealtimeSession(
                instructions="persona",
                trace_id="audio-flash",
                send_audio=audio,
                retrieve_references=retrieve,
            )
            await session.start()
            try:
                session.append_audio(b"\0\0")
                session.commit()
                result = await asyncio.wait_for(session.wait(), 1)
                self.assertEqual(result.answer_text, "早餐七点开始。")
                self.assertEqual(session._voice, "longanqian")
                retrieve.assert_awaited_once()
                audio.assert_awaited_once_with(b"mock-mp3")
                config = upstream.sent[0]["session"]
                self.assertNotIn("input_audio_transcription", config)
                self.assertEqual(config["input_audio_format"], "pcm")
                self.assertEqual(config["output_audio_format"], "pcm")
            finally:
                await session.aclose()

    async def test_date_is_refreshed_after_capture_crosses_beijing_midnight(self):
        upstream = FakeUpstream()
        session = qwen.HardwareQwenRealtimeSession(
            instructions="历史对话：今天是五月。",
            trace_id="date-test",
            send_audio=AsyncMock(),
            retrieve_references=AsyncMock(return_value="旧知识库日期：五月"),
        )
        with (
            patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)),
            patch.object(
                qwen,
                "local_now",
                side_effect=[
                    datetime(2026, 10, 9, 15, 59, 59, tzinfo=timezone.utc),
                    datetime(2026, 10, 9, 16, 0, 1, tzinfo=timezone.utc),
                ],
            ),
        ):
            await session.start()
            try:
                session.append_audio(b"\0\0")
                session.commit()
                await asyncio.wait_for(session.wait(), 2)
                updates = [
                    x["session"]["instructions"]
                    for x in upstream.sent
                    if x["type"] == "session.update"
                ]
                self.assertIn("今天是2026年10月9日", updates[0])
                self.assertIn("今天是2026年10月10日，星期六", updates[1])
                self.assertNotIn("今天是2026年10月9日", updates[1])
                self.assertEqual(updates[1].count("本轮当前时间"), 1)
                self.assertGreater(
                    updates[1].index("本轮当前时间"), updates[1].index("旧知识库日期")
                )
            finally:
                await session.aclose()

    def test_latency_metrics_use_separate_capture_and_response_reference_points(self):
        session = qwen.HardwareQwenRealtimeSession(
            instructions="persona",
            trace_id="metrics-test",
            send_audio=AsyncMock(),
            retrieve_references=AsyncMock(),
            started_at=10,
        )
        session._marks.update(
            {
                "capture_end": 14,
                "response_requested": 15,
                "first_pcm": 15.5,
                "first_mp3_generated": 15.75,
                "first_mp3_send_start": 15.8,
                "first_mp3_sent": 16,
            }
        )
        session._update_timings()
        self.assertEqual(session.result.timings_ms["end_to_first_audio_ms"], 2000)
        self.assertEqual(session.result.timings_ms["response_to_first_audio_ms"], 1000)
        self.assertEqual(session.result.timings_ms["pcm_to_first_mp3_ms"], 250)
        self.assertNotIn("response_to_first_text_ms", session.result.timings_ms)

    async def test_failed_response_is_not_success(self):
        upstream = FakeUpstream(status="failed")
        session = qwen.HardwareQwenRealtimeSession(
            instructions="persona",
            trace_id="failure",
            send_audio=AsyncMock(),
            retrieve_references=AsyncMock(return_value=""),
        )
        with patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)):
            await session.start()
            try:
                session.append_audio(b"\0\0")
                session.commit()
                with self.assertRaisesRegex(qwen.HardwareQwenError, "未成功完成"):
                    await asyncio.wait_for(session.wait(), 2)
            finally:
                await session.aclose()

    async def test_empty_asr_requests_one_voice_hint_without_retrieval(self):
        class DuplicateEmptyAsr(FakeUpstream):
            async def send(self, raw):
                await super().send(raw)
                if json.loads(raw)["type"] == "input_audio_buffer.commit":
                    self.queue.put_nowait(
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "transcript": "",
                        }
                    )

        upstream = DuplicateEmptyAsr(transcript="   ")
        retrieve = AsyncMock()
        audio = AsyncMock()
        session = qwen.HardwareQwenRealtimeSession(
            instructions="酒店前台，历史问题",
            trace_id="repeat",
            send_audio=audio,
            retrieve_references=retrieve,
        )
        with patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)):
            await session.start()
            try:
                session.append_audio(b"\0\0")
                session.commit()
                result = await asyncio.wait_for(session.wait(), 1)
                self.assertTrue(result.needs_repeat)
                self.assertEqual(result.asr_text, "")
                self.assertEqual(result.answer_text, qwen.ASR_EMPTY_REPEAT_PROMPT)
                audio.assert_awaited_once_with(b"mock-mp3")
                retrieve.assert_not_awaited()
                self.assertEqual(
                    sum(x["type"] == "response.create" for x in upstream.sent), 1
                )
                fallback = [x for x in upstream.sent if x["type"] == "session.update"][
                    -1
                ]["session"]
                self.assertIs(fallback["enable_search"], False)
                self.assertIn(qwen.ASR_EMPTY_REPEAT_PROMPT, fallback["instructions"])
                self.assertNotIn("历史问题", fallback["instructions"])
            finally:
                await session.aclose()

    async def test_asr_service_failure_is_not_disguised_as_repeat_prompt(self):
        class FailedAsr(FakeUpstream):
            async def send(self, raw):
                await super().send(raw)
                if json.loads(raw)["type"] == "input_audio_buffer.commit":
                    self.queue.put_nowait(
                        {"type": "conversation.item.input_audio_transcription.failed"}
                    )

        upstream = FailedAsr(transcribe=False)
        retrieve = AsyncMock()
        audio = AsyncMock()
        session = qwen.HardwareQwenRealtimeSession(
            instructions="persona",
            trace_id="asr-failed",
            send_audio=audio,
            retrieve_references=retrieve,
        )
        with patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)):
            await session.start()
            try:
                session.append_audio(b"\0\0")
                session.commit()
                with self.assertRaisesRegex(qwen.HardwareQwenError, "输入转写失败"):
                    await asyncio.wait_for(session.wait(), 1)
                self.assertFalse(session.result.needs_repeat)
                audio.assert_not_awaited()
                retrieve.assert_not_awaited()
                self.assertNotIn("response.create", [x["type"] for x in upstream.sent])
            finally:
                await session.aclose()

    async def test_text_first_and_delayed_audio_can_complete_within_limit(self):
        upstream = DelayedAudioUpstream(delay=0.05)
        session = qwen.HardwareQwenRealtimeSession(
            instructions="persona",
            trace_id="delayed",
            send_audio=AsyncMock(),
            retrieve_references=AsyncMock(return_value=""),
        )
        with (
            patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)),
            patch.object(
                qwen.settings, "hardware_qwen_first_response_timeout_seconds", 0.3
            ),
        ):
            await session.start()
            try:
                session.append_audio(b"\0\0")
                session.commit()
                result = await asyncio.wait_for(session.wait(), 1)
                self.assertEqual(result.answer_text, "延迟返回的语音。")
                self.assertGreaterEqual(
                    result.timings_ms["response_to_first_pcm_ms"], 40
                )
                self.assertGreater(
                    result.timings_ms["response_to_first_pcm_ms"],
                    result.timings_ms["response_to_first_text_ms"],
                )
            finally:
                await session.aclose()

    async def test_first_audio_timeout_is_bounded_and_distinct_from_network_error(self):
        upstream = DelayedAudioUpstream(delay=1)
        session = qwen.HardwareQwenRealtimeSession(
            instructions="persona",
            trace_id="bounded",
            send_audio=AsyncMock(),
            retrieve_references=AsyncMock(return_value=""),
        )
        with (
            patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)),
            patch.object(
                qwen.settings, "hardware_qwen_first_response_timeout_seconds", 0.025
            ),
        ):
            await session.start()
            try:
                session.append_audio(b"\0\0")
                session.commit()
                with self.assertRaises(qwen.HardwareQwenTimeout) as caught:
                    await asyncio.wait_for(session.wait(), 1)
                self.assertEqual(
                    caught.exception.error_code, "upstream_first_audio_timeout"
                )
                self.assertTrue(caught.exception.retryable)
                self.assertIn("test-response-id", session._upstream_response_id)
                self.assertEqual(session.result.audio_bytes, 0)
            finally:
                await session.aclose()

    async def test_timeout_after_audio_does_not_mark_round_safe_to_replay(self):
        upstream = DelayedAudioUpstream(delay=0, finish=False)
        session = qwen.HardwareQwenRealtimeSession(
            instructions="persona",
            trace_id="partial",
            send_audio=AsyncMock(),
            retrieve_references=AsyncMock(return_value=""),
        )
        with (
            patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)),
            patch.object(
                qwen.settings, "hardware_qwen_output_idle_timeout_seconds", 0.05
            ),
        ):
            await session.start()
            try:
                session.append_audio(b"\0\0")
                session.commit()
                with self.assertRaises(qwen.HardwareQwenTimeout) as caught:
                    await asyncio.wait_for(session.wait(), 1)
                self.assertEqual(caught.exception.error_code, "upstream_wait_timeout")
                self.assertFalse(caught.exception.retryable)
                self.assertGreater(session.result.audio_bytes, 0)
            finally:
                await session.aclose()

    async def test_commit_changes_receive_timeout(self):
        upstream = FakeUpstream(transcribe=False)
        session = qwen.HardwareQwenRealtimeSession(
            instructions="persona",
            trace_id="timeout",
            send_audio=AsyncMock(),
            retrieve_references=AsyncMock(),
        )
        with (
            patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)),
            patch.object(qwen.settings, "hardware_qwen_asr_timeout_seconds", 0.01),
        ):
            await session.start()
            try:
                session.append_audio(b"\0\0")
                session.commit()
                with self.assertRaisesRegex(qwen.HardwareQwenError, "超时"):
                    await asyncio.wait_for(session.wait(), 2)
            finally:
                await session.aclose()

    async def run_gateway(
        self, device, upstream_status="completed", upstream_factory=None, sessions=None
    ):
        connections = []

        async def connect(*args, **kwargs):
            upstream = (
                upstream_factory()
                if upstream_factory
                else FakeUpstream(status=upstream_status)
            )
            connections.append(upstream)
            return upstream

        load = AsyncMock(side_effect=sessions or [business_session(9), business_session(10)])
        persist = AsyncMock()
        refs = AsyncMock(return_value="知识库资料")
        with (
            patch.object(qwen.websockets, "connect", connect),
            patch.object(gateway, "_load_hardware_session", load),
            patch.object(gateway, "persist_hardware_realtime_turn", persist),
            patch.object(gateway, "retrieve_hardware_realtime_references", refs),
            patch.object(
                gateway, "summarize_conversation_in_background", AsyncMock()
            ) as summary,
        ):
            await asyncio.wait_for(gateway.hardware_dialogue_websocket(device), 3)
            self.summary_await_count = summary.await_count
        return connections, load, persist, refs

    async def test_empty_asr_hint_completes_without_writing_conversation(self):
        device = FakeDevice()
        connections, _, persist, refs = await self.run_gateway(
            device, upstream_factory=lambda: FakeUpstream(transcript="")
        )
        self.assertEqual(device.completed, 1)
        self.assertEqual(device.audio, [b"mock-mp3"])
        self.assertFalse(any(x["type"] == "dialogue.error" for x in device.messages))
        persist.assert_not_awaited()
        refs.assert_not_awaited()
        self.assertEqual(self.summary_await_count, 0)
        self.assertTrue(connections[0].closed)

    async def test_normal_turn_after_repeat_hint_restores_search_and_memory(self):
        device = FakeDevice(turns=2)
        transcripts = iter(["", "酒店早餐几点？"])
        connections, _, persist, refs = await self.run_gateway(
            device, upstream_factory=lambda: FakeUpstream(transcript=next(transcripts))
        )
        self.assertEqual(device.completed, 2)
        persist.assert_awaited_once()
        refs.assert_awaited_once()
        self.assertEqual(persist.await_args.args[0].agent_id, 10)
        self.assertTrue(connections[1].sent[0]["session"]["enable_search"])

    async def test_gateway_reports_timeout_code_without_success_or_persist(self):
        device = FakeDevice()
        with patch.object(
            qwen.settings, "hardware_qwen_first_response_timeout_seconds", 0.025
        ):
            connections, _, persist, _ = await self.run_gateway(
                device, upstream_factory=lambda: DelayedAudioUpstream(delay=1)
            )
        persist.assert_not_awaited()
        self.assertEqual(
            device.messages[-1]["error_code"], "upstream_first_audio_timeout"
        )
        self.assertTrue(device.messages[-1]["retryable"])
        self.assertEqual(device.completed, 0)
        self.assertFalse(device.audio)
        self.assertTrue(connections[0].closed)

    async def test_initial_configuration_failure_is_logged_before_device_disconnects(
        self,
    ):
        class RejectedConfig(FakeUpstream):
            async def send(self, raw):
                data = json.loads(raw)
                if data["type"] == "session.update":
                    self.sent.append(data)
                    self.queue.put_nowait(
                        {"type": "error", "error": {"message": "unsupported voice"}}
                    )
                else:
                    await super().send(raw)

        device = FakeDevice()
        with self.assertLogs(gateway.logger, level="ERROR") as captured:
            connections, _, persist, _ = await self.run_gateway(
                device, upstream_factory=RejectedConfig
            )
        self.assertTrue(
            any(
                "初始化失败" in line and "unsupported voice" in line
                for line in captured.output
            )
        )
        self.assertEqual(device.messages[-1]["type"], "dialogue.error")
        self.assertIn("unsupported voice", device.messages[-1]["error"])
        self.assertTrue(connections[0].closed)
        persist.assert_not_awaited()

    async def test_device_two_turns_reload_agent_and_keep_memory(self):
        device = FakeDevice(turns=2)
        connections, load, persist, refs = await self.run_gateway(device)
        self.assertEqual(load.await_count, 2)
        self.assertEqual(persist.await_count, 2)
        self.assertEqual(refs.await_args_list[1].args[0].agent_id, 10)
        self.assertEqual(device.completed, 2)
        self.assertEqual(device.audio, [b"mock-mp3", b"mock-mp3"])
        # 保持同事版本 dialogue.done 的全部字段，便于固件继续读取统计。
        expected_fields = {
            "type",
            "req",
            "client_req",
            "route",
            "provider",
            "finish_reason",
            "endpoint_silence_bytes",
            "endpoint_silence_chunks",
            "output_pcm_bytes",
            "output_pcm_chunks",
            "audio_bytes",
            "audio_chunks",
            "post_endpoint_input_bytes",
            "post_endpoint_input_chunks",
            "provider_audio_max_gap_ms",
            "mp3_output_max_gap_ms",
            "first_provider_audio_ms",
            "first_chunk_ms",
            "pcm_to_first_mp3_ms",
            "total_ms",
        }
        for message in device.messages:
            if message["type"] == "dialogue.done":
                self.assertTrue(expected_fields.issubset(message))
                self.assertEqual(message["output_pcm_chunks"], 1)
        self.assertTrue(all(x.closed for x in connections))

    async def test_same_agent_three_turns_reuse_and_delete_only_previous_items(self):
        device = FakeDevice(turns=3)
        sessions = [business_session(9) for _ in range(3)]
        sessions[1].history_messages = [{"role": "assistant", "content": "第一轮历史"}]
        sessions[2].history_messages = [{"role": "assistant", "content": "第二轮历史"}]
        connections, load, persist, refs = await self.run_gateway(device, sessions=sessions)
        self.assertEqual(len(connections), 1)
        self.assertEqual((load.await_count, persist.await_count, refs.await_count), (3, 3, 3))
        self.assertEqual(device.audio, [b"mock-mp3"] * 3)
        done = [x for x in device.messages if x["type"] == "dialogue.done"]
        self.assertEqual([x["upstream_reused"] for x in done], [False, True, True])
        self.assertEqual([x["output_pcm_chunks"] for x in done], [1, 1, 1])
        sent = connections[0].sent
        deleted = [x["item_id"] for x in sent if x["type"] == "conversation.item.delete"]
        self.assertEqual(deleted, ["assistant-1", "user-1", "assistant-2", "user-2"])
        configs = [x["session"]["instructions"] for x in sent if x["type"] == "session.update" and "modalities" in x["session"]]
        self.assertIn("第一轮历史", configs[1])
        self.assertIn("第二轮历史", configs[2])
        self.assertTrue(connections[0].closed)  # 设备断线才关闭。

    async def test_reuse_restores_search_after_empty_transcription(self):
        class RepeatThenNormal(FakeUpstream):
            async def send(self, raw):
                data = json.loads(raw)
                if data["type"] == "input_audio_buffer.commit":
                    self.transcript = "" if self.turn_number == 0 else "早餐几点？"
                    self.reply = "早餐七点开始。"
                await super().send(raw)

        with patch.object(qwen.settings, "hardware_qwen_realtime_enable_search", True):
            device = FakeDevice(turns=2)
            connections, _, persist, refs = await self.run_gateway(
                device, upstream_factory=RepeatThenNormal,
                sessions=[business_session(9), business_session(9)],
            )
        self.assertEqual(len(connections), 1)
        self.assertEqual(device.completed, 2)
        persist.assert_awaited_once()
        refs.assert_awaited_once()
        updates = [x["session"] for x in connections[0].sent if x["type"] == "session.update"]
        self.assertFalse(updates[1]["enable_search"])
        self.assertTrue(updates[2]["enable_search"])

    async def test_failed_reuse_reconnects_before_new_audio(self):
        class DisconnectedBetweenTurns(FakeUpstream):
            async def send(self, raw):
                if json.loads(raw)["type"] == "input_audio_buffer.clear":
                    raise ConnectionError("upstream idle connection closed")
                await super().send(raw)

        device = FakeDevice(turns=2)
        with self.assertLogs(qwen.logger, level="WARNING"):
            connections, _, persist, _ = await self.run_gateway(
                device, upstream_factory=DisconnectedBetweenTurns,
                sessions=[business_session(9), business_session(9)],
            )
        self.assertEqual(len(connections), 2)
        self.assertEqual(device.completed, 2)
        self.assertEqual(persist.await_count, 2)
        self.assertTrue(all(x.closed for x in connections))
        self.assertEqual(sum(x["type"] == "response.create" for c in connections for x in c.sent), 2)

    async def test_unknown_upstream_items_fall_back_to_new_connection(self):
        class MissingItemIds(FakeUpstream):
            async def recv(self):
                event = json.loads(await super().recv())
                event.pop("item_id", None)
                if event["type"] == "response.done":
                    event["response"].pop("output", None)
                return json.dumps(event)

        device = FakeDevice(turns=2)
        connections, _, _, _ = await self.run_gateway(
            device, upstream_factory=MissingItemIds,
            sessions=[business_session(9), business_session(9)],
        )
        self.assertEqual(len(connections), 2)
        self.assertEqual(device.completed, 2)
        self.assertTrue(all(x.closed for x in connections))

    async def test_cancel_second_turn_closes_reused_connection(self):
        class CancelSecondTurn(FakeDevice):
            async def send_json(self, message):
                if message["type"] == "dialogue.done":
                    self.cancel = True
                await super().send_json(message)

        connections, _, persist, _ = await self.run_gateway(
            CancelSecondTurn(turns=2),
            sessions=[business_session(9), business_session(9)],
        )
        self.assertEqual(len(connections), 1)
        persist.assert_awaited_once()
        self.assertTrue(connections[0].closed)
        self.assertEqual(sum(x["type"] == "response.create" for x in connections[0].sent), 1)

    async def test_cancel_during_capture_cleans_up_even_before_worker_started(self):
        connections, _, persist, _ = await self.run_gateway(FakeDevice(cancel=True))
        persist.assert_not_awaited()
        self.assertTrue(connections[0].closed)
        self.assertNotIn("response.create", [x["type"] for x in connections[0].sent])

    async def test_size_mismatch_cancels_without_persist(self):
        device = FakeDevice(size=6)
        connections, _, persist, _ = await self.run_gateway(device)
        self.assertIn("audio size mismatch", device.messages[-1]["error"])
        persist.assert_not_awaited()
        self.assertTrue(connections[0].closed)

    async def test_failed_gateway_does_not_report_done_or_persist(self):
        device = FakeDevice()
        connections, _, persist, _ = await self.run_gateway(device, "failed")
        self.assertEqual(device.messages[-1]["type"], "dialogue.error")
        self.assertEqual(device.completed, 0)
        persist.assert_not_awaited()
        self.assertTrue(connections[0].closed)

    async def test_speex_requires_cascade(self):
        device = FakeDevice(audio_format="speex")
        connections, load, persist, _ = await self.run_gateway(device)
        self.assertFalse(connections)
        load.assert_not_awaited()
        persist.assert_not_awaited()
        self.assertIn("requires", device.messages[-1]["error"])

    async def test_cascade_still_uses_existing_pipeline(self):
        device = FakeDevice(route="cascade")
        pipeline = AsyncMock(
            return_value=gateway.HardwareTurnResult("用户", "回答", 10, 1, 5)
        )
        with (
            patch.object(
                gateway,
                "_load_hardware_session",
                AsyncMock(return_value=business_session()),
            ),
            patch.object(gateway, "_run_hardware_turn", pipeline),
            patch.object(qwen.websockets, "connect", AsyncMock()) as connect,
        ):
            await asyncio.wait_for(gateway.hardware_dialogue_websocket(device), 2)
        pipeline.assert_awaited_once()
        self.assertEqual(pipeline.await_args.kwargs["audio_data"], b"\0" * 4)
        connect.assert_not_awaited()
        self.assertEqual(device.completed, 1)


class ReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_knowledge_does_not_discard_completed_memory(self):
        manager = AsyncMock()

        async def slow_knowledge(*args, **kwargs):
            await asyncio.sleep(5)

        with (
            patch.object(context_service, "AsyncSessionLocal", return_value=manager),
            patch.object(
                context_service.settings, "hardware_qwen_retrieval_timeout_seconds", 0.1
            ),
            patch.object(context_service, "embed_text", AsyncMock(return_value=[0.1])),
            patch.object(
                context_service,
                "search_summaries",
                return_value=[
                    [{"distance": 0.9, "entity": {"summary_text": "偏好清淡"}}]
                ],
            ),
            patch.object(context_service, "retrieve_agent_knowledge", slow_knowledge),
            self.assertLogs(context_service.logger, level="WARNING"),
        ):
            text = await asyncio.wait_for(
                context_service.retrieve_hardware_realtime_references(
                    business_session(), "问题", "test"
                ),
                1,
            )
        self.assertIn("偏好清淡", text)

    async def test_shared_embedding_and_user_agent_filters(self):
        db = AsyncMock()
        manager = AsyncMock()
        manager.__aenter__.return_value = db
        hit = SimpleNamespace(title="早餐", content="七点")
        vectors = [0.1, 0.2]
        with (
            patch.object(context_service, "AsyncSessionLocal", return_value=manager),
            patch.object(
                context_service, "embed_text", AsyncMock(return_value=vectors)
            ) as embed,
            patch.object(
                context_service,
                "search_summaries",
                return_value=[
                    [{"distance": 0.9, "entity": {"summary_text": "偏好清淡"}}]
                ],
            ) as memory,
            patch.object(
                context_service,
                "retrieve_agent_knowledge",
                AsyncMock(return_value=[hit]),
            ) as knowledge,
        ):
            text = await context_service.retrieve_hardware_realtime_references(
                business_session(), "问题", "test"
            )
        embed.assert_awaited_once()
        self.assertIs(
            memory.call_args.kwargs["query_vector"],
            knowledge.await_args.kwargs["query_vector"],
        )
        self.assertEqual(knowledge.await_args.kwargs["agent_id"], 9)
        self.assertEqual(knowledge.await_args.kwargs["user_id"], 1)
        self.assertIn("偏好清淡", text)
        self.assertIn("七点", text)

    async def test_only_real_question_answer_are_saved_atomically(self):
        db = AsyncMock()
        db.add = unittest.mock.Mock()
        conversation = SimpleNamespace(message_count=4, last_message_at=None)
        db.scalar.return_value = conversation
        manager = AsyncMock()
        manager.__aenter__.return_value = db
        with patch.object(context_service, "AsyncSessionLocal", return_value=manager):
            await context_service.persist_hardware_realtime_turn(
                business_session(), "真实问题", "真实回答"
            )
        self.assertEqual(
            [call.args[0].content for call in db.add.call_args_list],
            ["真实问题", "真实回答"],
        )
        self.assertEqual(conversation.message_count, 6)
        db.commit.assert_awaited_once()


class BeijingClockTests(unittest.TestCase):
    def test_utc_date_is_converted_to_beijing_date(self):
        text = build_current_time_context(
            datetime(2026, 10, 9, 16, 30, tzinfo=timezone.utc)
        )
        self.assertIn("2026-10-10 00:30:00", text)
        self.assertIn("今天是2026年10月10日，星期六", text)
        self.assertIn("UTC+08:00", text)


class QwenSessionFamilyTests(unittest.TestCase):
    def test_audio_family_uses_model_specific_default_voice_without_omni_asr(self):
        for model, voice in (
            ("qwen-audio-3.0-realtime-plus", "longanqian"),
            ("qwen-audio-3.0-realtime-flash", "longanqian"),
            ("qwen-audio-3.1-realtime-plus", "longanqian_v3.1"),
        ):
            config = qwen.build_hardware_qwen_session_config(
                model=model, voice="Tina", instructions="prompt"
            )
            self.assertEqual(config["voice"], voice)
            self.assertNotIn("input_audio_transcription", config)
            self.assertIsNone(config["turn_detection"])

    def test_omni_family_keeps_tina_and_configured_transcription(self):
        with patch.object(
            qwen.settings,
            "hardware_qwen_realtime_transcription_model",
            "qwen3-asr-flash-realtime",
        ):
            config = qwen.build_hardware_qwen_session_config(
                model="qwen3.8-omni-flash-realtime", voice="Tina", instructions="prompt"
            )
        self.assertEqual(config["voice"], "Tina")
        self.assertEqual(
            config["input_audio_transcription"], {"model": "qwen3-asr-flash-realtime"}
        )

    def test_custom_audio_voice_is_not_overwritten(self):
        config = qwen.build_hardware_qwen_session_config(
            model="qwen-audio-3.0-realtime-flash",
            voice="my-cloned-voice",
            instructions="prompt",
        )
        self.assertEqual(config["voice"], "my-cloned-voice")


class ConversationOwnershipTests(unittest.IsolatedAsyncioTestCase):
    """用内存 SQL 表验证会话复用和保存条件；不连接线上 PostgreSQL。"""

    def setUp(self):
        self.engine = create_engine("sqlite://")

        @event.listens_for(self.engine, "connect")
        def register_advisory_lock(connection, _record):
            connection.create_function("pg_advisory_xact_lock", 1, lambda _value: None)

        Conversation.__table__.create(self.engine)
        ConversationMessage.__table__.create(self.engine)
        self.sql = SqlSession(self.engine, expire_on_commit=False)
        self.db = self.AsyncSqlDb(self.sql)

    def tearDown(self):
        self.sql.close()
        self.engine.dispose()

    class AsyncSqlDb:
        def __init__(self, sql):
            self.sql = sql

        async def execute(self, statement, parameters=None):
            return self.sql.execute(statement, parameters)

        async def scalar(self, statement):
            return self.sql.scalar(statement)

        def add(self, obj):
            # SQLite 不对 BigInteger 主键自动递增，测试中显式分配 ID。
            if obj.id is None:
                obj.id = (self.sql.scalar(select(func.max(type(obj).id))) or 0) + 1
            self.sql.add(obj)

        async def commit(self):
            self.sql.commit()

        async def refresh(self, obj):
            self.sql.refresh(obj)

        async def __aenter__(self):
            return self

        async def __aexit__(self, kind, value, traceback):
            if kind is not None:
                self.sql.rollback()

    def seed(self, identity=1, user_id=1, device_id=2, agent_id=9):
        row = Conversation(
            id=identity,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
            session_id=f"test-session-{identity}",
            message_count=0,
        )
        self.sql.add(row)
        self.sql.commit()
        return row

    async def claim(self):
        return await conversation_service.get_or_create_conversation(
            self.db,
            user_id=1,
            device_id=2,
            agent_id=9,
        )

    async def test_legacy_null_device_is_reused_and_can_save_turn(self):
        legacy = self.seed(device_id=None)
        self.sql.add(
            ConversationMessage(
                id=1,
                conversation_id=legacy.id,
                role="user",
                content="历史问题",
                source="voice",
            )
        )
        legacy.message_count = 1
        self.sql.commit()
        row = await self.claim()
        self.assertEqual(row.id, legacy.id)
        self.assertEqual(row.device_id, 2)
        with patch.object(context_service, "AsyncSessionLocal", return_value=self.db):
            session = business_session()
            session.conversation_id = row.id
            await context_service.persist_hardware_realtime_turn(
                session, "新问题", "新回答"
            )
        self.assertEqual(row.message_count, 3)
        self.assertEqual(
            self.sql.scalars(
                select(ConversationMessage.content).order_by(ConversationMessage.id)
            ).all(),
            ["历史问题", "新问题", "新回答"],
        )

    async def test_legacy_null_user_is_filled_without_new_conversation(self):
        legacy = self.seed(user_id=None)
        row = await self.claim()
        self.assertEqual(row.id, legacy.id)
        self.assertEqual(row.user_id, 1)

    async def test_old_owner_conversation_is_not_reused(self):
        old = self.seed(user_id=99)
        row = await self.claim()
        self.assertNotEqual(row.id, old.id)
        self.assertEqual(row.user_id, 1)
        self.assertEqual(old.user_id, 99)

    async def test_other_device_conversation_is_not_reused(self):
        old = self.seed(device_id=99)
        row = await self.claim()
        self.assertNotEqual(row.id, old.id)
        self.assertEqual(row.device_id, 2)
        self.assertEqual(old.device_id, 99)

    async def test_exact_match_wins_over_newer_legacy_row(self):
        exact = self.seed()
        self.seed(identity=2, device_id=None)
        self.assertEqual((await self.claim()).id, exact.id)

    async def test_persistence_still_rejects_wrong_owner(self):
        old = self.seed(user_id=99)
        session = business_session()
        session.conversation_id = old.id
        with (
            patch.object(context_service, "AsyncSessionLocal", return_value=self.db),
            self.assertRaisesRegex(ValueError, "归属"),
        ):
            await context_service.persist_hardware_realtime_turn(
                session, "问题", "回答"
            )
        self.assertEqual(self.sql.scalars(select(ConversationMessage)).all(), [])


class CiPrefixTests(unittest.TestCase):
    def test_first_mpeg_prefix_patch_handles_short_chunks(self):
        bridge = PcmToCiMp3Bridge.__new__(PcmToCiMp3Bridge)
        bridge._ci_first_mp3_pending = True
        bridge._ci_first_mp3_prefix = bytearray()
        output = []
        bridge._on_mp3 = output.append
        bridge._forward_ci_compatible_mp3(b"\xff\xfb123")
        self.assertFalse(output)
        bridge._forward_ci_compatible_mp3(b"45678more")
        self.assertEqual(output[0][6:10], b"\0" * 4)
        bridge._forward_ci_compatible_mp3(b"next")
        self.assertEqual(output[1], b"next")


if __name__ == "__main__":
    unittest.main()
