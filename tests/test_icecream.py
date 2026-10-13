"""无真实下单：模拟 HTTP 与语音上游，验证 SN、重复执行和状态播报。"""

import asyncio
import json
import unittest
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from pydantic import SecretStr

from app.api.v1 import hardware_voice as gateway
from app.integrations import hardware_qwen_realtime as qwen
from app.services.icecream_service import (
    IcecreamService,
    iter_icecream_status_updates,
    match_icecream_intent,
)
from tests.test_hardware_qwen_realtime import (
    FakeBridge,
    FakeDevice,
    FakeUpstream,
    business_session,
)

TARGET = "VI-ESP32S3-68EE8FC60B38"
REAL_CLIENT = httpx.AsyncClient


def config(**kwargs):
    values = {
        "icecream_enabled": True,
        "icecream_device_sn": TARGET,
        "icecream_ip": "192.168.1.11",
        "icecream_port": 8080,
        "icecream_username": "test-account",
        "icecream_password": SecretStr("test-password"),
        "icecream_query_timeout_seconds": 5,
        "icecream_execute_timeout_seconds": 10,
        "icecream_poll_interval_seconds": 0.001,
        "icecream_monitor_timeout_seconds": 0.1,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def instance(status, started="2026-10-13T12:00:00", workflow="make-icecream"):
    return {
        "workflowId": workflow,
        "status": status,
        "startedAt": started,
        "endedAt": None if status == "RUNNING" else started,
        "error": None,
    }


class Machine:
    def __init__(self, states=None):
        self.requests = []
        self.states = list(
            states or [{}, {"new": instance("RUNNING")}, {"new": instance("COMPLETED")}]
        )
        self.query_count = 0
        self.execute_count = 0
        self.login_count = 0
        self.execute_error = None
        self.execute_payload = {"success": True, "workflowId": "make-icecream"}
        self.execute_status = 200
        self.expire_query = False
        self.expire_execute = False
        self.post_entered = None

    async def handle(self, request):
        self.requests.append(request)
        if request.url.path == "/api/auth/login":
            self.login_count += 1
            body = json.loads(request.content)
            assert body == {"username": "test-account", "password": "test-password"}
            return httpx.Response(
                200, json={"success": True, "token": f"token-{self.login_count}"}
            )
        assert request.headers["authorization"].startswith("Bearer token-")
        if request.method == "GET":
            if self.expire_query:
                self.expire_query = False
                return httpx.Response(401, json={"success": False})
            value = self.states[min(self.query_count, len(self.states) - 1)]
            self.query_count += 1
            return httpx.Response(200, json={"success": True, "instances": value})
        assert request.url.path == "/api/workflows-json/make-icecream/execute"
        assert json.loads(request.content) == {}
        self.execute_count += 1
        if self.expire_execute:
            self.expire_execute = False
            return httpx.Response(401, json={"success": False})
        if self.post_entered is not None:
            self.post_entered.set()
            await asyncio.Event().wait()
        if self.execute_error:
            raise self.execute_error("execute interrupted", request=request)
        return httpx.Response(self.execute_status, json=self.execute_payload)

    def client(self, **kwargs):
        return REAL_CLIENT(transport=httpx.MockTransport(self.handle), **kwargs)


class IntentTests(unittest.TestCase):
    def test_explicit_one_cup_commands(self):
        for text in [
            "帮我下单一杯冰淇淋",
            "我要一杯冰淇淋",
            "来一杯冰淇淋",
            "给我一杯冰激凌",
            "帮我做一个冰淇淋",
            "我想买一杯冰淇淋",
            "帮我订一份冰淇淋",
            "请帮我点一杯冰淇淋",
            "能不能帮我下单1杯冰淇淋？",
            "麻烦你给我来一杯冰淇淋，谢谢",
        ]:
            with self.subTest(text=text):
                self.assertEqual(match_icecream_intent(text), "order")

    def test_negation_questions_quotes_and_multiple_cups_never_order(self):
        for text in [
            "不要给我一杯冰淇淋",
            "我不要一杯冰淇淋",
            "取消冰淇淋",
            "如果我要一杯冰淇淋",
            "他说我要一杯冰淇淋",
            "冰淇淋多少钱",
            "我要一杯冰淇淋多少钱",
            "教我说我要一杯冰淇淋",
            "我要一杯冰淇淋但是先别下单",
            "冰淇淋",
            "一杯冰淇淋",
            "不要下单，我要一杯冰淇淋",
        ]:
            with self.subTest(text=text):
                self.assertIsNone(match_icecream_intent(text))
        for text in ["我要两杯冰淇淋", "帮我下单10杯冰淇淋", "给我零杯冰淇淋"]:
            self.assertEqual(match_icecream_intent(text), "quantity")


class IcecreamTests(unittest.IsolatedAsyncioTestCase):
    async def use(self, machine, **settings):
        service = IcecreamService(config(**settings))
        self.addAsyncCleanup(service.aclose)
        client_patch = patch(
            "app.services.icecream_service.httpx.AsyncClient", machine.client
        )
        client_patch.start()
        self.addCleanup(client_patch.stop)
        return service

    async def test_wrong_sn_and_non_order_do_not_touch_network(self):
        machine = Machine()
        service = await self.use(machine)
        for sn in ["another-device", TARGET + "-other", "", TARGET.lower()]:
            self.assertIsNone(await service.resolve("我要一杯冰淇淋", sn, "test"))
        self.assertIsNone(await service.resolve("不要给我一杯冰淇淋", TARGET, "test"))
        self.assertEqual(machine.requests, [])

    async def test_empty_ip_disabled_and_wrong_ip_never_call_http(self):
        machine = Machine()
        for kwargs in [
            {"icecream_ip": ""},
            {"icecream_enabled": False},
            {"icecream_ip": "http://wrong"},
            {"icecream_password": None},
        ]:
            service = IcecreamService(config(**kwargs))
            result = await service.resolve("我要一杯冰淇淋", TARGET, "test")
            self.assertEqual(result.action_payload["status"], "not_configured")
        self.assertEqual(machine.requests, [])

    async def test_accept_then_follow_new_instance_to_completion(self):
        old = {"old": instance("COMPLETED", "2026-10-12T12:00:00")}
        machine = Machine(
            [
                old,
                old,
                {**old, "new": instance("RUNNING")},
                {**old, "new": instance("COMPLETED")},
            ]
        )
        service = await self.use(machine)
        result = await service.resolve("帮我下单一杯冰淇淋", TARGET, "test")
        self.assertEqual(result.action_payload["status"], "accepted")
        updates = [update async for update in iter_icecream_status_updates(result)]
        self.assertEqual(
            [x.action_payload["status"] for x in updates], ["running", "completed"]
        )
        self.assertEqual(machine.execute_count, 1)
        self.assertEqual(service._instance_id, "new")
        self.assertEqual(machine.requests[0].url.host, "192.168.1.11")
        self.assertEqual(machine.requests[0].url.port, 8080)

    async def test_parallel_command_and_running_machine_do_not_execute(self):
        machine = Machine([{"other": instance("RUNNING", workflow="other-workflow")}])
        service = await self.use(machine)
        result = await service.resolve("我要一杯冰淇淋", TARGET, "test")
        self.assertEqual(result.action_payload["status"], "busy")
        async with service._lock:
            result = await service.resolve("我要一杯冰淇淋", TARGET, "second")
        self.assertEqual(result.action_payload["status"], "busy")
        self.assertEqual(machine.execute_count, 0)

    async def test_execute_timeout_never_reposts_and_keeps_unknown(self):
        machine = Machine([{}])
        machine.execute_error = httpx.ReadTimeout
        service = await self.use(machine, icecream_monitor_timeout_seconds=0.01)
        result = await service.resolve("我要一杯冰淇淋", TARGET, "first")
        self.assertEqual(result.action_payload["status"], "unknown")
        updates = [update async for update in iter_icecream_status_updates(result)]
        self.assertEqual(updates[-1].action_payload["status"], "unknown")
        second = await service.resolve("我要一杯冰淇淋", TARGET, "second")
        self.assertEqual(second.action_payload["status"], "busy")
        self.assertEqual(machine.execute_count, 1)

    async def test_401_renews_token_for_query_and_execute(self):
        machine = Machine()
        machine.expire_query = machine.expire_execute = True
        service = await self.use(machine)
        result = await service.resolve("我要一杯冰淇淋", TARGET, "test")
        self.assertEqual(result.action_payload["status"], "accepted")
        self.assertEqual(machine.login_count, 3)
        self.assertEqual(machine.execute_count, 2)  # 第一次明确 401，未受理。
        await service._monitor

    async def test_rejection_is_not_success_and_failure_is_announced(self):
        machine = Machine()
        machine.execute_status = 400
        machine.execute_payload = {"success": False, "failReason": "PARALLELISM_REJECT"}
        service = await self.use(machine)
        result = await service.resolve("我要一杯冰淇淋", TARGET, "reject")
        self.assertEqual(result.action_payload["status"], "busy")
        self.assertIsNone(result.status_updates)
        machine.execute_status = 200
        machine.execute_payload = {"success": True, "workflowId": "make-icecream"}
        machine.states = [{}, {"new": instance("FAILED")}]
        machine.query_count = 0
        result = await service.resolve("我要一杯冰淇淋", TARGET, "fail")
        updates = [update async for update in iter_icecream_status_updates(result)]
        self.assertEqual(updates[-1].action_payload["status"], "failed")

    async def test_malformed_execute_result_is_uncertain_and_never_retried(self):
        machine = Machine([{}])
        machine.execute_payload = []
        service = await self.use(machine, icecream_monitor_timeout_seconds=0.01)
        result = await service.resolve("我要一杯冰淇淋", TARGET, "test")
        self.assertEqual(result.action_payload["status"], "unknown")
        await service._monitor
        await service.resolve("我要一杯冰淇淋", TARGET, "again")
        self.assertEqual(machine.execute_count, 1)

    async def test_cancelled_voice_does_not_cancel_status_confirmation(self):
        machine = Machine([{}])
        machine.post_entered = asyncio.Event()
        service = await self.use(machine, icecream_monitor_timeout_seconds=0.01)
        task = asyncio.create_task(service.resolve("我要一杯冰淇淋", TARGET, "test"))
        await machine.post_entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(service._state, "unknown")
        await service._monitor
        await service.resolve("我要一杯冰淇淋", TARGET, "again")
        self.assertEqual(machine.execute_count, 1)

    async def test_status_question_reports_latest_without_execute(self):
        machine = Machine(
            [
                {
                    "old": instance("FAILED", "2026-10-12T12:00:00"),
                    "new": instance("COMPLETED"),
                }
            ]
        )
        service = await self.use(machine)
        result = await service.resolve("我的冰淇淋做好了吗", TARGET, "test")
        self.assertEqual(result.action_payload["status"], "completed")
        self.assertEqual(machine.execute_count, 0)


class SpeechUpstream(FakeUpstream):
    async def send(self, raw):
        event = json.loads(raw)
        instructions = (event.get("session") or {}).get("instructions", "")
        if "服务器根据设备接口实际返回结果" in instructions:
            self.reply = instructions.split("：\n", 1)[-1]
        await super().send(raw)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_after_status_broadcast_next_question_restores_search_and_references(
        self,
    ):
        machine = Machine()
        service = IcecreamService(config())
        self.addAsyncCleanup(service.aclose)
        business = business_session()
        business.device_sn = TARGET
        upstream = SpeechUpstream(transcript="我要一杯冰淇淋")

        class NextQuestionDevice(FakeDevice):
            async def send_json(self, message):
                if message["type"] == "dialogue.done":
                    upstream.transcript = "酒店早餐几点"
                    upstream.reply = "早餐七点开始。"
                await super().send_json(message)

        device = NextQuestionDevice(turns=2, device_sn=TARGET)
        with (
            patch("app.services.icecream_service.httpx.AsyncClient", machine.client),
            patch.object(gateway, "icecream_service", service),
            patch.object(
                gateway, "_load_hardware_session", AsyncMock(return_value=business)
            ),
            patch.object(
                gateway,
                "retrieve_hardware_realtime_references",
                AsyncMock(return_value="早餐七点"),
            ) as refs,
            patch.object(
                gateway, "persist_hardware_realtime_turn", AsyncMock()
            ) as persist,
            patch.object(gateway, "summarize_conversation_in_background", AsyncMock()),
            patch.object(
                qwen.settings, "hardware_qwen_realtime_api_key", SecretStr("test-key")
            ),
            patch.object(qwen.settings, "hardware_voice_route", "realtime"),
            patch.object(qwen.settings, "hardware_qwen_realtime_enable_search", True),
            patch.object(qwen, "PcmToCiMp3Bridge", FakeBridge),
            patch.object(
                qwen.websockets, "connect", AsyncMock(return_value=upstream)
            ) as connect,
        ):
            await asyncio.wait_for(gateway.hardware_dialogue_websocket(device), 3)
        self.assertEqual(device.completed, 2)
        self.assertEqual(machine.execute_count, 1)
        self.assertEqual(persist.await_count, 2)
        refs.assert_awaited_once_with(business, "酒店早餐几点", unittest.mock.ANY)
        connect.assert_awaited_once()
        self.assertTrue(device.messages[-1]["upstream_reused"])
        configs = [e["session"] for e in upstream.sent if e["type"] == "session.update"]
        self.assertTrue(configs[-2]["enable_search"])
        self.assertIn("早餐七点", configs[-1]["instructions"])

    async def test_realtime_order_and_statuses_use_current_voice_before_done(self):
        machine = Machine()
        service = IcecreamService(config())
        self.addAsyncCleanup(service.aclose)
        device = FakeDevice(device_sn=TARGET)
        business = business_session()
        business.device_sn = TARGET
        upstream = SpeechUpstream(transcript="帮我下单一杯冰淇淋")
        refs, persist = AsyncMock(), AsyncMock()
        with (
            patch("app.services.icecream_service.httpx.AsyncClient", machine.client),
            patch.object(gateway, "icecream_service", service),
            patch.object(
                gateway, "_load_hardware_session", AsyncMock(return_value=business)
            ),
            patch.object(gateway, "retrieve_hardware_realtime_references", refs),
            patch.object(gateway, "persist_hardware_realtime_turn", persist),
            patch.object(gateway, "summarize_conversation_in_background", AsyncMock()),
            patch.object(
                qwen.settings, "hardware_qwen_realtime_api_key", SecretStr("test-key")
            ),
            patch.object(qwen.settings, "hardware_voice_route", "realtime"),
            patch.object(qwen, "PcmToCiMp3Bridge", FakeBridge),
            patch.object(qwen.websockets, "connect", AsyncMock(return_value=upstream)),
        ):
            await asyncio.wait_for(gateway.hardware_dialogue_websocket(device), 3)
        self.assertEqual(machine.execute_count, 1)
        self.assertEqual(device.audio, [b"mock-mp3"] * 3)
        self.assertEqual(
            [m["status"] for m in device.messages if m["type"] == "dialogue.action"],
            ["accepted", "running", "completed"],
        )
        self.assertEqual(device.messages[-1]["type"], "dialogue.done")
        self.assertEqual(device.messages[-1]["action"], "icecream")
        self.assertEqual(device.messages[-1]["audio_chunks"], 3)
        refs.assert_not_awaited()
        self.assertIn("已制作完成", persist.await_args.args[2])
        self.assertEqual(
            len([e for e in upstream.sent if e["type"] == "input_audio_buffer.commit"]),
            1,
        )
        configs = [e["session"] for e in upstream.sent if e["type"] == "session.update"]
        self.assertTrue(all(c["enable_search"] is False for c in configs[1:]))

    async def test_cascade_order_bypasses_other_tools_and_speaks_statuses(self):
        machine = Machine()
        service = IcecreamService(config())
        self.addAsyncCleanup(service.aclose)
        business = business_session()
        business.device_sn = TARGET
        spoken = []

        async def tts(text, **kwargs):
            spoken.append(text)
            yield b"mock-mp3"

        with (
            patch("app.services.icecream_service.httpx.AsyncClient", machine.client),
            patch.object(gateway, "icecream_service", service),
            patch.object(
                gateway.baidu_speech_client,
                "recognize_pcm",
                AsyncMock(return_value="我要一杯冰淇淋"),
            ),
            patch.object(gateway.baidu_speech_client, "stream_synthesize_mp3", tts),
            patch.object(gateway, "build_hardware_tools", return_value=[{}]),
            patch.object(gateway, "_save_message", AsyncMock()),
            patch.object(
                gateway.hardware_action_service, "resolve", AsyncMock()
            ) as other,
        ):
            result = await asyncio.wait_for(
                gateway._run_hardware_turn(
                    session=business,
                    audio_data=b"\0" * 4,
                    audio_format="pcm",
                    request_id="test",
                    turn_started_at=perf_counter(),
                    send_bytes=AsyncMock(),
                    send_json=AsyncMock(),
                    cancel_event=asyncio.Event(),
                ),
                3,
            )
        self.assertEqual(result.action, "icecream")
        self.assertEqual(result.audio_chunks, 3)
        self.assertIn("已制作完成", spoken[-1])
        self.assertEqual(machine.execute_count, 1)
        other.assert_not_awaited()
