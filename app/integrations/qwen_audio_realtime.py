"""阿里云 Qwen Audio 实时语音协议适配。

本模块只处理后端到阿里 Qwen Audio 的 WebSocket 握手、事件序列化和事件
解析。小程序仍然使用本项目自己的 start/stop/PCM 协议，记忆检索和消息落库
由 API 路由负责。测试链路采用 Qwen 的 push-to-talk 手动模式：提交音频后
不会自动生成回复，路由可以先完成本地记忆检索，再发送 response.create。
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets

from app.core.config import settings

logger = logging.getLogger(__name__)


class QwenAudioRealtimeError(RuntimeError):
    """Qwen Audio 实时会话配置、协议或网络错误。"""


def get_qwen_audio_api_key() -> str:
    """读取 Qwen Audio Key；开发测试时允许复用已有 Qwen 文本 Key。"""

    secret = settings.qwen_audio_realtime_api_key or settings.qwen_api_key
    if settings.qwen_audio_realtime_api_key is None and settings.qwen_api_key:
        logger.warning(
            "[QWEN][鉴权] 未单独配置 QWEN_AUDIO_REALTIME_API_KEY，"
            "回退复用 QWEN_API_KEY"
        )
    if secret is None:
        raise QwenAudioRealtimeError(
            "未配置 QWEN_AUDIO_REALTIME_API_KEY 或 QWEN_API_KEY"
        )
    value = secret.get_secret_value().strip()
    if not value:
        raise QwenAudioRealtimeError("Qwen Audio API Key 为空")
    return value


def get_qwen_audio_ws_url() -> str:
    """拼接阿里实时语音地址和模型参数。"""

    if not settings.qwen_audio_realtime_ws_url:
        raise QwenAudioRealtimeError(
            "未配置 QWEN_AUDIO_REALTIME_WS_URL，"
            "请填写阿里文档中对应工作空间的实时 WebSocket 地址"
        )
    base_url = settings.qwen_audio_realtime_ws_url.strip()
    parsed = urlsplit(base_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["model"] = settings.qwen_audio_realtime_model
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )


def resolve_qwen_audio_voice(voice_key: str | None) -> str:
    """把本地 Agent 音色别名映射为 Qwen Audio 音色。

    目前测试链路固定使用一个阿里已配置音色，避免把火山音色 ID 误传给
    阿里。后续可在 Agent 音色配置中增加厂商维度再扩展映射。
    """

    _ = voice_key
    return settings.qwen_audio_realtime_voice


def build_session_update_event(
        *,
        instructions: str,
        voice: str,
) -> dict[str, object]:
    """构造 Qwen Audio 手动 push-to-talk 会话配置。

    ``turn_detection=None`` 是关键：音频 commit 后不自动触发回答，后端
    收到 ASR completed 后可以先做 Embedding/Milvus，再显式 response.create。
    ``max_history_turns`` 由配置控制，限制 Qwen 参与推理的历史 QA 轮数。
    """

    return {
        "type": "session.update",
        "event_id": "session_update_qwen_audio",
        "session": {
            "modalities": ["text", "audio"],
            "voice": voice,
            "instructions": instructions,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "turn_detection": None,
            "max_history_turns": settings.qwen_audio_realtime_max_history_turns,
        },
    }


def build_history_item(
        *,
        role: str,
        text: str,
) -> dict[str, object]:
    """把本地最近消息转为 Qwen 上下文管理事件。"""

    # Qwen 对不同角色的文本内容类型有严格要求：用户消息使用
    # input_text，助手历史消息使用 output_text。若助手也使用 input_text，
    # 服务端会在 conversation.item.create 时直接拒绝整条会话。
    content_type = "output_text" if role == "assistant" else "input_text"

    return {
        "type": "conversation.item.create",
        "event_id": f"history_{uuid.uuid4().hex[:12]}",
        "item": {
            "type": "message",
            "role": role,
            "content": [
                {
                    "type": content_type,
                    "text": text,
                }
            ],
        },
    }


def build_context_item(text: str) -> dict[str, object]:
    """构造本轮检索结果的 system 上下文项。"""

    return {
        "type": "conversation.item.create",
        "event_id": f"memory_{uuid.uuid4().hex[:12]}",
        "item": {
            "type": "message",
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": text,
                }
            ],
        },
    }


def build_response_create_event() -> dict[str, object]:
    """在检索/上下文注入结束后请求 Qwen 开始本轮回答。"""

    return {
        "type": "response.create",
        "event_id": f"response_create_{uuid.uuid4().hex[:12]}",
        "response": {
            "modalities": ["audio", "text"],
        },
    }


async def connect_qwen_audio():
    """建立只存在于后端与阿里之间的实时语音 WebSocket。"""

    url = get_qwen_audio_ws_url()
    logger.info(
        "[QWEN][上游] 正在连接 Qwen Audio 地址=%s 模型=%s",
        url.split("?", 1)[0],
        settings.qwen_audio_realtime_model,
    )
    return await websockets.connect(
        url,
        additional_headers={
            "Authorization": f"Bearer {get_qwen_audio_api_key()}",
            "x-dashscope-dataInspection": "disable",
        },
        open_timeout=15,
        close_timeout=5,
        ping_interval=20,
        ping_timeout=20,
        max_size=16 * 1024 * 1024,
    )


async def send_qwen_audio_event(
        socket: Any,
        event: dict[str, object],
) -> None:
    """发送 Qwen JSON 事件，只记录事件类型，不记录音频 Base64。"""

    event_type = event.get("type")
    logger.debug("[QWEN][上游] 已发送事件=%s", event_type)
    await socket.send(json.dumps(event, ensure_ascii=False, separators=(",", ":")))


def parse_qwen_audio_event(raw: object) -> dict[str, Any]:
    """解析 Qwen 下行 JSON 事件。"""

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        raise QwenAudioRealtimeError("Qwen Audio 返回了无法识别的消息类型")
    try:
        event = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QwenAudioRealtimeError("Qwen Audio 返回了非法 JSON") from exc
    if not isinstance(event, dict):
        raise QwenAudioRealtimeError("Qwen Audio 返回事件不是 JSON 对象")
    return event


def qwen_audio_error_detail(event: dict[str, Any]) -> str:
    """提取 Qwen 错误事件中的可读信息。"""

    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
        if isinstance(message, str):
            return message
    message = event.get("message") or event.get("detail")
    if isinstance(message, str):
        return message
    if message is not None:
        return json.dumps(message, ensure_ascii=False)[:500]
    return "Qwen Audio 返回未知错误"
