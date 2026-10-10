"""验证千问失败事件不会被保存为成功对话；不连接数据库或外部服务。"""

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from app.api.v1 import qwen_audio_voice as voice
from app.integrations.qwen_audio_realtime import qwen_audio_error_info


class FakeClient:
    def __init__(self, turns=1):
        self.messages = []
        self.close_code = None
        self.turns = turns
        self.queue = asyncio.Queue()
        for command in ("start", "stop"):
            self.queue.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": command})})

    async def accept(self):
        pass

    async def receive_json(self):
        return {"type": "session_start", "device_sn": "test-device"}

    async def receive(self):
        return await self.queue.get()

    async def send_json(self, message):
        self.messages.append(message)
        if message["type"] == "ready" and self.ready_count > 1:
            if self.ready_count <= self.turns:
                for command in ("start", "stop"):
                    self.queue.put_nowait({"type": "websocket.receive", "text": json.dumps({"type": command})})
            else:
                self.queue.put_nowait({"type": "websocket.disconnect"})

    async def send_bytes(self, data):
        pass

    async def close(self, code):
        self.close_code = code

    @property
    def ready_count(self):
        return sum(message["type"] == "ready" for message in self.messages)


class FakeUpstream:
    def __init__(self, response_events, fail_operation=None):
        self.queue = asyncio.Queue()
        self.response_events = response_events
        self.closed = False
        self.sent_events = []
        self.items = []
        self.response_contexts = []
        self.fail_operation = fail_operation
        self.system_create_count = 0

    async def recv(self):
        event = await self.queue.get()
        if isinstance(event, Exception):
            raise event
        return json.dumps(event)

    async def send(self, raw):
        event = json.loads(raw)
        self.sent_events.append(event)
        event_type = event["type"]
        if event_type == "session.update":
            self.queue.put_nowait({"type": "session.updated"})
        if event_type == "conversation.item.create":
            item = dict(event["item"])
            item.setdefault("id", f"server_item_{len(self.sent_events)}")
            if item["role"] == "system":
                self.system_create_count += 1
                if self.fail_operation == "create" and self.system_create_count > 1:
                    self.queue.put_nowait({"type": "error", "error": {"type": "invalid_request_error", "message": "系统资料创建失败"}})
                    return
            previous = event.get("previous_item_id")
            if previous is None:
                self.items.append(item)
            else:
                index = next(index for index, existing in enumerate(self.items) if existing["id"] == previous)
                self.items.insert(index + 1, item)
            self.queue.put_nowait({"type": "conversation.item.created", "item": item})
        if event_type == "conversation.item.delete":
            if self.fail_operation == "delete":
                self.queue.put_nowait({"type": "error", "error": {"type": "invalid_request_error", "message": "旧系统资料删除失败"}})
                return
            self.items = [item for item in self.items if item["id"] != event["item_id"]]
            self.queue.put_nowait({"type": "conversation.item.deleted", "item_id": event["item_id"]})
        if event_type == "input_audio_buffer.commit":
            self.items.append({
                "id": f"audio_{len(self.sent_events)}", "type": "message", "role": "user",
                "content": [{"type": "input_audio", "transcript": "酒店附近有什么吃的吗？"}],
            })
            self.queue.put_nowait({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "酒店附近有什么吃的吗？",
            })
        if event_type == "response.create":
            self.response_contexts.append(list(self.items))
            for event in self.response_events:
                self.queue.put_nowait(event)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.recv()

    async def close(self):
        self.closed = True


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class QwenAudioVoiceTest(unittest.IsolatedAsyncioTestCase):
    async def run_events(self, events, *, memories=None, hits=None, turns=1,
                         memory_batches=None, knowledge_batches=None, fail_operation=None):
        client = FakeClient(turns=turns)
        upstream = FakeUpstream(events, fail_operation=fail_operation)
        context = SimpleNamespace(
            user_id=1, agent_id=2, device_id=3, conversation_id=4,
            agent_system_prompt="你是一位酒店助手。", agent_name="酒店",
            agent_voice=None, history_messages=[],
        )
        persist = AsyncMock(return_value=False)
        with (
            patch.object(voice, "load_e2e_context", AsyncMock(return_value=context)),
            patch.object(voice, "connect_qwen_audio", AsyncMock(return_value=upstream)),
            patch.object(voice, "retrieve_e2e_long_term_memories", AsyncMock(return_value=memories or [], side_effect=memory_batches)),
            patch.object(voice, "AsyncSessionLocal", return_value=FakeSession()),
            patch.object(voice, "retrieve_agent_knowledge", AsyncMock(return_value=hits or [], side_effect=knowledge_batches)),
            patch.object(voice, "persist_e2e_turn", persist),
            self.assertLogs(voice.logger, level="INFO"),
        ):
            await asyncio.wait_for(voice.stream_voice_qwen(client), timeout=2)
        self.assertTrue(upstream.closed)
        client.upstream_events = upstream.sent_events
        client.response_contexts = upstream.response_contexts
        return client, persist

    async def test_failed_response_does_not_persist_or_report_success(self):
        error = {"type": "server_error", "code": "llm_error", "message": "Internal service error"}
        event = {
            "type": "response.done", "event_id": "server-event",
            "response": {"id": "response-id", "status": "failed", "status_details": {"error": error}},
        }
        client, persist = await self.run_events([
            {"type": "response.audio.delta", "delta": base64.b64encode(b"pcm").decode()},
            event,
            {"type": "error", "error": error},
        ])
        persist.assert_not_awaited()
        self.assertEqual(client.ready_count, 1)
        self.assertEqual(client.close_code, 1011)
        self.assertNotIn("reply_end", [item["type"] for item in client.messages])
        self.assertIn("tts_end", [item["type"] for item in client.messages])
        info = qwen_audio_error_info(event)
        self.assertEqual(info["error_code"], "llm_error")
        self.assertEqual(info["event_id"], "server-event")
        self.assertEqual(info["response_id"], "response-id")

    async def test_completed_response_is_persisted(self):
        client, persist = await self.run_events([
            {"type": "response.audio_transcript.delta", "delta": "附近有餐馆。"},
            {"type": "response.done", "response": {"status": "completed"}},
        ])
        persist.assert_awaited_once()
        self.assertEqual(client.ready_count, 2)
        self.assertIsNone(client.close_code)
        self.assertIn("reply_end", [item["type"] for item in client.messages])

    async def test_cancelled_response_does_not_persist(self):
        client, persist = await self.run_events([
            {"type": "response.done", "response": {"status": "cancelled"}},
        ])
        persist.assert_not_awaited()
        self.assertEqual(client.ready_count, 2)
        self.assertIsNone(client.close_code)

    async def test_empty_or_missing_status_response_does_not_persist(self):
        for response in ({"status": "completed"}, {}):
            with self.subTest(response=response):
                client, persist = await self.run_events([
                    {"type": "response.done", "response": response},
                ])
                persist.assert_not_awaited()
                self.assertEqual(client.ready_count, 1)
                self.assertEqual(client.close_code, 1011)

    async def test_server_error_closes_session_without_ready(self):
        for error in (
            {"type": "server_error", "message": "生成失败"},
            {"message": "Internal service error: Please try again later."},
        ):
            with self.subTest(error=error):
                client, persist = await self.run_events([{"type": "error", "error": error}])
                persist.assert_not_awaited()
                self.assertEqual(client.ready_count, 1)
                self.assertEqual(client.close_code, 1011)

    async def test_request_error_keeps_session_available(self):
        client, persist = await self.run_events([{
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "音频太短"},
        }])
        persist.assert_not_awaited()
        self.assertEqual(client.ready_count, 2)
        self.assertIsNone(client.close_code)

    async def test_memory_and_knowledge_share_one_system_at_conversation_start(self):
        client, persist = await self.run_events([
            {"type": "response.audio_transcript.delta", "delta": "附近有餐馆。"},
            {"type": "response.done", "response": {"status": "completed"}},
        ], memories=["用户曾咨询过附近餐馆。"], hits=[SimpleNamespace(title="酒店", content="附近有餐馆。")])
        persist.assert_awaited_once()
        items = client.response_contexts[0]
        self.assertEqual([item["role"] for item in items], ["system", "user"])
        system_text = items[0]["content"][0]["text"]
        self.assertIn("酒店助手", system_text)
        self.assertIn("本轮长期记忆参考资料", system_text)
        self.assertIn("用户曾咨询过附近餐馆", system_text)
        self.assertIn("本轮知识库参考资料", system_text)
        session = client.upstream_events[0]["session"]
        self.assertEqual(session["instructions"], "")
        self.assertEqual(client.upstream_events[-1]["type"], "response.create")

    async def test_next_turn_without_hits_clears_previous_system_references(self):
        client, persist = await self.run_events([
            {"type": "response.audio_transcript.delta", "delta": "回答。"},
            {"type": "response.done", "response": {"status": "completed"}},
        ], turns=2, memory_batches=[["上一轮记忆事实"], []],
            knowledge_batches=[[SimpleNamespace(title="文档", content="上一轮知识库事实")], []])
        self.assertEqual(persist.await_count, 2)
        first, second = client.response_contexts
        for items in (first, second):
            self.assertEqual(items[0]["role"], "system")
            self.assertEqual(sum(item["role"] == "system" for item in items), 1)
        self.assertIn("上一轮记忆事实", first[0]["content"][0]["text"])
        self.assertIn("上一轮知识库事实", first[0]["content"][0]["text"])
        second_text = second[0]["content"][0]["text"]
        self.assertIn("酒店助手", second_text)
        self.assertNotIn("上一轮记忆事实", second_text)
        self.assertNotIn("上一轮知识库事实", second_text)
        self.assertNotEqual(first[0]["id"], second[0]["id"])

    async def test_context_create_or_delete_failure_never_starts_response(self):
        for operation in ("create", "delete"):
            with self.subTest(operation=operation):
                client, persist = await self.run_events([], fail_operation=operation)
                persist.assert_not_awaited()
                self.assertEqual(client.ready_count, 1)
                self.assertEqual(client.close_code, 1011)
                self.assertNotIn("response.create", [event["type"] for event in client.upstream_events])

    async def test_upstream_1007_closes_client_without_success(self):
        close = Close(1007, "Internal service error")
        client, persist = await self.run_events([ConnectionClosedError(close, close, True)])
        persist.assert_not_awaited()
        self.assertEqual(client.ready_count, 1)
        self.assertEqual(client.close_code, 1011)


if __name__ == "__main__":
    unittest.main()
