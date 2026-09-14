"""豆包实时语音模型 3.0（Seeduplex）协议适配。

本模块只负责第三方 WebSocket 的鉴权、会话配置和事件序列化；小程序协议、
设备校验、对话落库由 API 路由负责。API Key 只在后端握手时发送，不下发给
小程序。
"""

from __future__ import annotations

import json
import logging
from typing import Any

import websockets

from app.core.config import settings

logger = logging.getLogger(__name__)


class VolcengineDuplexError(RuntimeError):
    """Seeduplex 配置、协议或网络错误。"""


# 数据库存的是厂商中立音色别名；Seeduplex 使用自己的 Jupiter 音色 ID。
DUPLEX_VOICE_ALIAS = {
    "female_gentle": "zh_female_xiaohe_jupiter_bigtts",
    "male_steady": "zh_male_yunzhou_jupiter_bigtts",
    "female_sweet": "zh_female_vv_jupiter_bigtts",
}


def get_duplex_api_key() -> str:
    """读取后端 Seeduplex Key；允许开发期回退复用现有豆包语音 Key。"""

    using_fallback_key = settings.volc_duplex_api_key is None
    secret = settings.volc_duplex_api_key or settings.volc_tts_api_key
    if using_fallback_key:
        logger.warning(
            "[E2E][AUTH] VOLC_DUPLEX_API_KEY 未单独配置，回退复用现有 TTS Key"
        )
    if secret is None:
        raise VolcengineDuplexError(
            "未配置 VOLC_DUPLEX_API_KEY，且没有可复用的 VOLC_TTS_API_KEY"
        )
    value = secret.get_secret_value().strip()
    if not value:
        raise VolcengineDuplexError("Seeduplex API Key 为空")
    return value


def resolve_duplex_voice(voice_key: str | None) -> str:
    """把本地 Agent 音色别名转换为 Seeduplex 音色 ID。"""

    if voice_key:
        mapped = DUPLEX_VOICE_ALIAS.get(voice_key)
        if mapped:
            return mapped
    return settings.volc_duplex_voice


def build_session_create_event(
    *,
    instructions: str,
    voice: str,
    dialog_context: list[dict[str, object]],
    tools: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """构造官方 Realtime 协议的 ``session.create`` 事件。

    音频格式在这里统一约定：上行 16k PCM、下行 24k PCM。后续接入真实
    Agent 时，只需要替换 instructions、voice、dialog_context 和 tools，
    音频协议不需要跟着业务层改动。
    """

    # extension.dialog 是模型扩展参数；dialog_context 必须放在这里，不能
    # 直接塞进 session 顶层。历史项的 role/text/timestamp 在 API 路由层组装。
    dialog_extension: dict[str, object] = {}
    if dialog_context:
        dialog_extension["dialog_context"] = dialog_context

    return {
        "type": "session.create",
        # session 只描述本次实时会话的模型、人设和音频规格。当前版本不传
        # session.id，因此断线恢复依赖本地 dialog_context，而不是厂商会话续接。
        "session": {
            "model": settings.volc_duplex_model,
            "instructions": instructions,
            "audio": {
                "input": {
                    "format": {
                        "type": "pcm",
                        "rate": 16000,
                    }
                },
                "output": {
                    "format": {
                        "type": "pcm",
                        "rate": 24000,
                    },
                    "voice": voice,
                    "speed": 0,
                    "loudness": 0,
                },
            },
            # Function Calling 工具在 API 路由层按业务场景注册。此处保留为
            # 空列表时，仍与不启用工具的普通端到端会话完全兼容。
            "tools": tools or [],
        },
        "extension": {
            "asr": {
                "extra": {
                    "enable_asr_twopass": False,
                }
            },
            "dialog": dialog_extension,
            "tts": {
                "extra": {},
            },
        },
    }


async def connect_duplex():
    """建立只存在于后端与火山之间的 Seeduplex WebSocket。"""

    logger.info(
        "[E2E][火山上游] 正在连接 Seeduplex 地址=%s",
        settings.volc_duplex_ws_url,
    )
    return await websockets.connect(
        settings.volc_duplex_ws_url,
        additional_headers={"X-Api-Key": get_duplex_api_key()},
        open_timeout=15,
        close_timeout=5,
        ping_interval=20,
        ping_timeout=20,
        max_size=16 * 1024 * 1024,
    )


async def send_duplex_event(socket: Any, event: dict[str, object]) -> None:
    """发送一条 Seeduplex JSON 文本事件。

    不记录事件完整内容，尤其不记录 Base64 音频和可能包含隐私的文本；
    只记录事件类型，便于按日志还原协议时序。
    """

    event_type = event.get("type")
    if event_type != "input_audio_buffer.append":
        logger.debug("[E2E][火山上游] 已发送事件=%s", event_type)
    await socket.send(json.dumps(event, ensure_ascii=False, separators=(",", ":")))


def parse_duplex_event(raw: object) -> dict[str, Any]:
    """解析服务端 JSON 文本事件，并拒绝非对象消息。"""

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        raise VolcengineDuplexError("Seeduplex 返回了无法识别的消息类型")
    try:
        event = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VolcengineDuplexError("Seeduplex 返回了非法 JSON") from exc
    if not isinstance(event, dict):
        raise VolcengineDuplexError("Seeduplex 返回事件不是 JSON 对象")
    return event
