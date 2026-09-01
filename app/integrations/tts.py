"""火山引擎 TTS V3 集成。

这里专门负责和火山语音服务通信，路由层只负责接收请求和返回结果。
支持两种模式：

1. ``synthesize_speech``：HTTP 调用，等整段 PCM 收完后返回；
2. ``stream_speech_pcm``：WebSocket 调用，边收到 PCM 分片边交给上层。
"""

import asyncio
import base64
import json
import logging
import struct
import uuid
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

import httpx
import websockets
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)

TTS_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
TTS_STREAM_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

# 这些音色需要和你在火山语音控制台开通的资源匹配。
SUPPORTED_VOICES = {
    "zh_female_vv_uranus_bigtts",
    "zh_male_dayi_saturn_bigtts",
    "zh_female_meilinvyou_saturn_bigtts",
}

# TTS V3 WebSocket 事件编号。
TTS_EVENT_START_CONNECTION = 1
TTS_EVENT_FINISH_CONNECTION = 2
TTS_EVENT_START_SESSION = 100
TTS_EVENT_FINISH_SESSION = 102
TTS_EVENT_TASK_REQUEST = 200
TTS_EVENT_CONNECTION_STARTED = 50
TTS_EVENT_SESSION_STARTED = 150
TTS_EVENT_SESSION_FAILED = 153
TTS_EVENT_AUDIO = 352
TTS_EVENT_ENDED = 359
TTS_MESSAGE_AUDIO = 0xB
TTS_MESSAGE_ERROR = 0xF


def get_tts_settings() -> tuple[str, str]:
    """读取 TTS 的 API Key 和资源 ID。"""

    if settings.volc_tts_api_key is None:
        raise HTTPException(
            status_code=500,
            detail="未配置 VOLC_TTS_API_KEY，请在 .env 中填写火山语音 API Key。",
        )

    return (
        settings.volc_tts_api_key.get_secret_value(),
        settings.volc_tts_resource_id,
    )


async def synthesize_speech(
    text: str,
    voice: str,
    trace_id: str = "standalone",
) -> bytes:
    """调用 TTS V3 HTTP 接口，合并返回的 Base64 PCM 分片。"""

    if voice not in SUPPORTED_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音色：{voice}，请使用已开通的 TTS 音色。",
        )

    api_key, resource_id = get_tts_settings()
    payload = {
        "user": {"uid": "cml-websit-backend"},
        "req_params": {
            "text": text[:1000],
            "speaker": voice,
            "audio_params": {
                # 原来的 MP3 方式保留在这里，切换回 MP3 时取消注释即可：
                # "format": "mp3",
                "format": "pcm",
                "sample_rate": 24000,
            },
        },
    }
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }

    pieces: list[bytes] = []
    messages: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    pending = ""

    started_at = perf_counter()
    first_chunk_received = False
    try:
        async with httpx.AsyncClient(timeout=90) as client, client.stream(
            "POST",
            TTS_URL,
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                log_id = response.headers.get("X-Tt-Logid", "")
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "TTS 请求失败："
                        f"{body.decode(errors='replace')}（logid: {log_id}）"
                    ),
                )

            # 火山返回的是连续 JSON 文本，不保证每个网络 chunk 正好是一条 JSON。
            async for chunk in response.aiter_text(chunk_size=1024):
                if not first_chunk_received:
                    first_chunk_received = True
                    logger.info(
                        "[VOICE-TIMING][%s][TTS-HTTP] 收到首个响应块 "
                        "ttfb=%.3fs",
                        trace_id,
                        perf_counter() - started_at,
                    )
                pending += chunk
                while pending.strip():
                    pending = pending.lstrip()
                    try:
                        message, offset = decoder.raw_decode(pending)
                    except json.JSONDecodeError:
                        # 当前 chunk 不完整，留到下一个 chunk 继续拼接。
                        break

                    pending = pending[offset:]
                    if not isinstance(message, dict):
                        continue

                    messages.append(message)
                    encoded_audio = message.get("data")
                    if isinstance(encoded_audio, str):
                        pieces.append(base64.b64decode(encoded_audio))
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"TTS 网络请求失败：{exc}",
        ) from exc

    if not pieces:
        raise HTTPException(
            status_code=502,
            detail=f"TTS 没有返回音频：{messages}",
        )

    audio = b"".join(pieces)
    logger.info(
        "[VOICE-TIMING][%s][TTS-HTTP] 完整合成结束 total=%.3fs bytes=%d",
        trace_id,
        perf_counter() - started_at,
        len(audio),
    )
    return audio


def build_tts_frame(
    event: int,
    session_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> bytes:
    """把 TTS JSON 数据封装成火山规定的二进制协议包。

    一个协议包可以理解成：固定包头 + event + session_id + JSON 载荷。
    火山根据 event 判断这是一条连接、会话、文本还是结束指令。
    """

    # 0x11：4 字节包头；0x14：完整消息并携带 event/session 信息。
    frame = bytearray((0x11, 0x14, 0x10, 0x00))
    frame.extend(struct.pack(">i", event))

    if session_id:
        session_bytes = session_id.encode("utf-8")
        frame.extend(struct.pack(">I", len(session_bytes)))
        frame.extend(session_bytes)

    payload_bytes = json.dumps(
        payload or {},
        ensure_ascii=False,
    ).encode("utf-8")
    frame.extend(struct.pack(">I", len(payload_bytes)))
    frame.extend(payload_bytes)
    return bytes(frame)


def parse_tts_frame(frame: bytes) -> tuple[int, int, int, bytes]:
    """拆开 TTS V3 返回包，返回消息类型、事件、错误码和载荷。

    这个函数只负责“拆信封”，不负责播放；调用方根据 message_type/event
    判断载荷是错误 JSON、状态 JSON 还是 PCM 音频。
    """

    if len(frame) < 4:
        raise ValueError("TTS 返回了不完整的数据包")

    message_type = (frame[1] >> 4) & 0x0F
    flags = frame[1] & 0x0F
    offset = (frame[0] & 0x0F) * 4
    if offset < 4 or len(frame) < offset + 4:
        raise ValueError("TTS 返回了无效数据包")

    if message_type == TTS_MESSAGE_ERROR:
        error_code = struct.unpack(">I", frame[offset : offset + 4])[0]
        offset += 4
        if len(frame) < offset + 4:
            raise ValueError("TTS 错误包不完整")
        payload_size = struct.unpack(">I", frame[offset : offset + 4])[0]
        offset += 4
        return (
            message_type,
            0,
            error_code,
            frame[offset : offset + payload_size],
        )

    event = 0
    if flags & 0x04:
        event = struct.unpack(">i", frame[offset : offset + 4])[0]
        offset += 4
        session_size = struct.unpack(">I", frame[offset : offset + 4])[0]
        offset += 4 + session_size

    if len(frame) < offset + 4:
        raise ValueError("TTS 数据包缺少载荷长度")

    payload_size = struct.unpack(">I", frame[offset : offset + 4])[0]
    offset += 4
    return message_type, event, 0, frame[offset : offset + payload_size]


def tts_error_detail(payload: bytes, fallback: str) -> str:
    """把火山错误载荷转换为可读文本。"""

    try:
        data = json.loads(payload.decode("utf-8"))
        if isinstance(data, dict):
            return str(data.get("message") or data.get("error") or data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return fallback


# 新版tts
async def stream_speech_pcm(
    text: str,
    voice: str,
    trace_id: str = "standalone",
    sentence_no: int | None = None,
) -> AsyncIterator[bytes]:
    """通过 TTS V3 WebSocket 持续产出 24kHz/16bit/单声道 PCM 分片。"""

    if voice not in SUPPORTED_VOICES:
        raise HTTPException(status_code=400, detail=f"不支持的音色：{voice}")

    api_key, resource_id = get_tts_settings()
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    session_id = str(uuid.uuid4())
    received_audio = False
    started_at = perf_counter()
    first_audio_at: float | None = None
    audio_bytes = 0
    chunk_count = 0
    sentence_label = sentence_no if sentence_no is not None else 0

    try:
        # 这里是 FastAPI 到火山 TTS 的第二条 WebSocket，
        # 不是小程序到 FastAPI 的那条 WebSocket。
        connect_started_at = perf_counter()
        async with websockets.connect(
            TTS_STREAM_URL,
            additional_headers=headers,
            open_timeout=15,
            close_timeout=5,
        ) as tts_ws:
            logger.info(
                "[VOICE-TIMING][%s][TTS#%d] 火山 WebSocket 已连接 "
                "elapsed=%.3fs",
                trace_id,
                sentence_label,
                perf_counter() - connect_started_at,
            )

            # 第一步：建立火山 TTS 连接。
            handshake_started_at = perf_counter()
            await tts_ws.send(build_tts_frame(TTS_EVENT_START_CONNECTION))
            first_frame = await asyncio.wait_for(tts_ws.recv(), timeout=15)
            if not isinstance(first_frame, bytes):
                raise HTTPException(status_code=502, detail="TTS 握手返回格式错误")

            _, event, error_code, payload = parse_tts_frame(first_frame)
            if error_code:
                raise HTTPException(
                    status_code=502,
                    detail=tts_error_detail(payload, f"TTS 连接失败（{error_code}）"),
                )
            if event != TTS_EVENT_CONNECTION_STARTED:
                raise HTTPException(
                    status_code=502,
                    detail=f"TTS 连接握手异常（事件 {event}）",
                )
            logger.info(
                "[VOICE-TIMING][%s][TTS#%d] 连接握手完成 elapsed=%.3fs",
                trace_id,
                sentence_label,
                perf_counter() - handshake_started_at,
            )

            # 第二步：在同一条火山连接中创建一个独立的合成会话。
            # session_id 用来区分本次文本合成任务。
            session_started_at = perf_counter()
            await tts_ws.send(
                build_tts_frame(
                    TTS_EVENT_START_SESSION,
                    session_id,
                    {
                        "req_params": {
                            "speaker": voice,
                            "audio_params": {
                                # 原来的 MP3 流式方式：
                                # "format": "mp3",
                                "format": "pcm",
                                "sample_rate": 24000,
                            },
                        },
                    },
                )
            )
            session_frame = await asyncio.wait_for(tts_ws.recv(), timeout=15)
            if not isinstance(session_frame, bytes):
                raise HTTPException(status_code=502, detail="TTS 会话建立失败")

            _, event, error_code, payload = parse_tts_frame(session_frame)
            if error_code or event == TTS_EVENT_SESSION_FAILED:
                raise HTTPException(
                    status_code=502,
                    detail=tts_error_detail(
                        payload,
                        f"TTS 会话创建失败（{error_code or event}）",
                    ),
                )
            if event != TTS_EVENT_SESSION_STARTED:
                raise HTTPException(
                    status_code=502,
                    detail=f"TTS 会话握手异常（事件 {event}）",
                )
            logger.info(
                "[VOICE-TIMING][%s][TTS#%d] 合成会话建立完成 elapsed=%.3fs",
                trace_id,
                sentence_label,
                perf_counter() - session_started_at,
            )

            # 第三步：提交完整文本。
            # 当前 TTS 是“文本一次提交、音频分片返回”，不是逐字提交文本。
            task_sent_at = perf_counter()
            await tts_ws.send(
                build_tts_frame(
                    TTS_EVENT_TASK_REQUEST,
                    session_id,
                    {"req_params": {"text": text[:1000]}},
                )
            )
            await tts_ws.send(
                build_tts_frame(TTS_EVENT_FINISH_SESSION, session_id)
            )

            # 第四步：持续读取火山返回的二进制事件包。
            while True:
                frame = await asyncio.wait_for(tts_ws.recv(), timeout=30)
                if not isinstance(frame, bytes):
                    raise HTTPException(status_code=502, detail="TTS 音频消息格式错误")

                message_type, event, error_code, payload = parse_tts_frame(frame)
                if (
                    error_code
                    or message_type == TTS_MESSAGE_ERROR
                    or event == TTS_EVENT_SESSION_FAILED
                ):
                    raise HTTPException(
                        status_code=502,
                        detail=tts_error_detail(
                            payload,
                            f"TTS 流式合成失败（{error_code or event}）",
                        ),
                    )

                if (
                    message_type == TTS_MESSAGE_AUDIO
                    and event == TTS_EVENT_AUDIO
                    and payload
                ):
                    # payload 就是一小段裸 PCM，交给上层路由转发给小程序。
                    received_audio = True
                    chunk_count += 1
                    audio_bytes += len(payload)
                    if first_audio_at is None:
                        first_audio_at = perf_counter()
                        logger.info(
                            "[VOICE-TIMING][%s][TTS#%d] 收到首个 PCM "
                            "first_audio=%.3fs text_chars=%d",
                            trace_id,
                            sentence_label,
                            first_audio_at - task_sent_at,
                            len(text),
                        )
                    yield payload

                if event in {TTS_EVENT_ENDED, TTS_EVENT_FINISH_SESSION, 152}:
                    # 收到结束事件，说明本次合成的所有音频已经返回完毕。
                    break

            # 第五步：通知火山关闭底层 TTS 连接。
            await tts_ws.send(build_tts_frame(TTS_EVENT_FINISH_CONNECTION))
    except HTTPException:
        logger.info(
            "[VOICE-TIMING][%s][TTS#%d] 合成失败 elapsed=%.3fs",
            trace_id,
            sentence_label,
            perf_counter() - started_at,
        )
        raise
    except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as exc:
        logger.info(
            "[VOICE-TIMING][%s][TTS#%d] 连接失败 elapsed=%.3fs error=%s",
            trace_id,
            sentence_label,
            perf_counter() - started_at,
            exc,
        )
        raise HTTPException(status_code=502, detail=f"TTS 流式连接失败：{exc}") from exc

    if not received_audio:
        raise HTTPException(status_code=502, detail="TTS 没有返回 PCM 音频")

    # 24kHz、16bit、单声道 PCM：每秒约 24000 * 2 字节。
    audio_duration = audio_bytes / (24000 * 2)
    logger.info(
        "[VOICE-TIMING][%s][TTS#%d] 流式合成结束 total=%.3fs "
        "chunks=%d bytes=%d audio_duration=%.3fs",
        trace_id,
        sentence_label,
        perf_counter() - started_at,
        chunk_count,
        audio_bytes,
        audio_duration,
    )
