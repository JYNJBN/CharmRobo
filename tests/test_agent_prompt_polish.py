"""提示词美化 POST 路由与服务验证；不调用外部模型或数据库。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.api.dependencies import get_current_user_id
from app.api.v1 import agent as agent_api
from app.api.v1.router import api_router
from app.core.database import get_db
from app.core.errors import BizError
from app.core.exception_handlers import biz_error_handler
from app.integrations import ai
from app.services import agent_prompt_service as service


class PolishRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(api_router)
        app.add_exception_handler(BizError, biz_error_handler)
        app.dependency_overrides[get_current_user_id] = lambda: 1
        app.dependency_overrides[get_db] = lambda: None
        self.app = app
        self.client = TestClient(app)

    def test_post_is_registered_and_returns_miniapp_response(self):
        polish = AsyncMock(return_value="你是一名酒店前台，回答简洁清晰。")
        with patch.object(agent_api, "polish_agent_prompt", polish):
            response = self.client.post(
                "/api/v1/agent/polish-prompt",
                json={"name": "酒店", "system_prompt": "酒店前台"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["data"]["system_prompt"], "你是一名酒店前台，回答简洁清晰。"
        )
        polish.assert_awaited_once_with(name="酒店", system_prompt="酒店前台")

    def test_post_requires_login(self):
        self.app.dependency_overrides.pop(get_current_user_id)
        with patch.object(agent_api, "polish_agent_prompt", AsyncMock()) as polish:
            response = self.client.post(
                "/api/v1/agent/polish-prompt",
                json={"system_prompt": "酒店前台"},
            )
        self.assertEqual(response.status_code, 401)
        polish.assert_not_awaited()

    def test_invalid_input_does_not_call_model(self):
        with patch.object(agent_api, "polish_agent_prompt", AsyncMock()) as polish:
            for text in ("", "字" * 501):
                response = self.client.post(
                    "/api/v1/agent/polish-prompt",
                    json={"system_prompt": text},
                )
                self.assertEqual(response.status_code, 422)
        polish.assert_not_awaited()


class PolishServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_chat_request_uses_qwen_thinking_parameter(self):
        key = SecretStr("fake-qwen-key")
        completion = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="美化后的提示词"))
                ],
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
        )
        with (
            patch.object(service.settings, "qwen_api_key", key),
            patch.object(
                service.settings, "agent_prompt_polish_model", "qwen3.7-flash"
            ),
            patch.object(ai, "get_model_client", return_value=client) as factory,
        ):
            result = await service.polish_agent_prompt(system_prompt="酒店前台")
        self.assertEqual(result, "美化后的提示词")
        self.assertIs(factory.call_args.args[0]["api_key"], key)
        self.assertEqual(completion.await_args.kwargs["model"], "qwen3.7-flash")
        self.assertEqual(
            completion.await_args.kwargs["extra_body"], {"enable_thinking": False}
        )
        self.assertEqual(
            ai._thinking_extra_body({"provider": "ark", "thinking_type": "disabled"}),
            {"thinking": {"type": "disabled"}},
        )

    async def test_polish_uses_qwen_key_url_and_independent_model(self):
        qwen_key = SecretStr("fake-qwen-key")
        with (
            patch.object(service.settings, "qwen_api_key", qwen_key),
            patch.object(
                service.settings, "qwen_base_url", "https://example.invalid/qwen"
            ),
            patch.object(service.settings, "qwen_model", None),
            patch.object(service.settings, "agent_prompt_polish_model", "polish-model"),
            patch.object(
                service, "chat", AsyncMock(return_value="  美化后的提示词  ")
            ) as chat,
        ):
            result = await service.polish_agent_prompt(
                name="酒店", system_prompt="  酒店前台  "
            )
        self.assertEqual(result, "美化后的提示词")
        model = chat.await_args.kwargs["model"]
        self.assertEqual(model["provider"], "qwen")
        self.assertEqual(model["model_id"], "polish-model")
        self.assertIs(model["api_key"], qwen_key)
        self.assertEqual(model["base_url"], "https://example.invalid/qwen")
        self.assertEqual(model["thinking_type"], "disabled")

    async def test_missing_qwen_key_does_not_fall_back_to_ark(self):
        for key in (None, SecretStr("")):
            with (
                patch.object(service.settings, "qwen_api_key", key),
                patch.object(service, "chat", AsyncMock()) as chat,
                self.assertRaises(BizError) as caught,
            ):
                await service.polish_agent_prompt(system_prompt="酒店前台")
            self.assertEqual(caught.exception.code, 503)
            self.assertIn("QWEN_API_KEY", caught.exception.message)
            chat.assert_not_awaited()

    async def test_blank_input_is_rejected_before_calling_model(self):
        with (
            patch.object(service, "chat", AsyncMock()) as chat,
            self.assertRaises(BizError) as caught,
        ):
            await service.polish_agent_prompt(system_prompt="   ")
        self.assertEqual(caught.exception.code, 400)
        chat.assert_not_awaited()

    async def test_empty_or_oversized_model_output_is_rejected(self):
        for text in ("   ", "字" * 501):
            with (
                patch.object(service, "chat", AsyncMock(return_value=text)),
                self.assertRaises(BizError) as caught,
            ):
                await service.polish_agent_prompt(system_prompt="酒店前台")
            self.assertEqual(caught.exception.code, 502)


if __name__ == "__main__":
    unittest.main()
