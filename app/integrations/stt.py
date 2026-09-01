"""火山引擎 STT V3 集成。

本文件负责两件事：

1. 普通 HTTP 录音文件识别；
2. 流式 ASR 的请求头、二进制协议封装和返回包解析。

WebSocket 和浏览器之间的转发流程放在 ``app.api.v1.voice``，这样路由层只编排
连接，不把火山协议细节和 HTTP 路由混在一起。
"""

import base64
import gzip
import json
import logging
import struct
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)

ASR_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
ASR_STREAM_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"

# 火山流式 ASR V3 二进制协议常量。
ASR_MSG_FULL_CLIENT = 0x1
ASR_MSG_AUDIO_ONLY = 0x2
ASR_MSG_ERROR = 0xF
ASR_FLAG_NO_SEQUENCE = 0x0
ASR_FLAG_FINAL = 0x2
ASR_FLAG_NEG_SEQUENCE = 0x3


def asr_headers() -> dict[str, str]:
    """构造普通 ASR 和流式 ASR 都需要的基础请求头。"""

    if settings.volc_asr_api_key is None:
        raise HTTPException(
            status_code=500,
            detail="未配置 VOLC_ASR_API_KEY，请在 .env 中填写火山语音 API Key。",
        )

    return {
        "X-Api-Key": settings.volc_asr_api_key.get_secret_value(),
        "X-Api-Resource-Id": settings.volc_asr_resource_id,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Api-Sequence": "-1",
    }


def stream_asr_headers() -> dict[str, str]:
    """构造流式 ASR WebSocket 请求头。"""

    headers = asr_headers()
    headers["X-Api-Resource-Id"] = settings.volc_stream_asr_resource_id
    headers["X-Api-Request-Id"] = str(uuid.uuid4())
    headers["X-Api-Connect-Id"] = str(uuid.uuid4())
    return headers


def build_asr_header(
    message_type: int,
    flags: int,
    serialization: int,
    compression: int,
) -> bytes:
    """生成火山流式 ASR 协议的 4 字节包头。"""

    return bytes(
        [
            0x11,
            (message_type << 4) | flags,
            (serialization << 4) | compression,
            0,
        ]
    )


def build_asr_config_frame() -> bytes:
    """告诉火山后续音频是 16kHz、16bit、单声道裸 PCM。"""

    payload = {
        "user": {"uid": "cml-websit-backend"},
        "audio": {
            "format": "pcm",
            "codec": "raw",
            "rate": 16000,
            "bits": 16,
            "channel": 1,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_nonstream": True,
            "enable_itn": True,
            "enable_punc": True,
            "show_utterances": True,
            "result_type": "full",
            "end_window_size": 600,
            "force_to_speech_time": 800,
        },
    }
    compressed = gzip.compress(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )
    return (
        build_asr_header(ASR_MSG_FULL_CLIENT, ASR_FLAG_NO_SEQUENCE, 1, 1)
        + struct.pack(">I", len(compressed))
        + compressed
    )


def build_asr_audio_frame(pcm: bytes, final: bool = False) -> bytes:
    """封装一块 PCM 音频；final=True 表示本轮录音结束。"""

    compressed = gzip.compress(pcm)
    flags = ASR_FLAG_FINAL if final else ASR_FLAG_NO_SEQUENCE
    return (
        build_asr_header(ASR_MSG_AUDIO_ONLY, flags, 0, 1)
        + struct.pack(">I", len(compressed))
        + compressed
    )


def parse_asr_server_frame(
    frame: bytes,
) -> tuple[int, int, int, dict[str, Any]]:
    """解析火山返回的 ASR 二进制包。"""

    if len(frame) < 8:
        raise ValueError("流式 ASR 返回了不完整的数据包")

    header_size = (frame[0] & 0x0F) * 4
    if header_size < 4 or len(frame) < header_size + 4:
        raise ValueError("流式 ASR 返回了无效包头")

    message_type = frame[1] >> 4
    flags = frame[1] & 0x0F
    compression = frame[2] & 0x0F
    offset = header_size
    error_code = 0

    if message_type == ASR_MSG_ERROR:
        error_code = struct.unpack(">I", frame[offset : offset + 4])[0]
        offset += 4
    elif flags in {1, ASR_FLAG_NEG_SEQUENCE}:
        # 带序号的服务端包会多出 4 字节 sequence。
        offset += 4

    if len(frame) < offset + 4:
        raise ValueError("流式 ASR 返回包缺少长度字段")

    size = struct.unpack(">I", frame[offset : offset + 4])[0]
    offset += 4
    payload = frame[offset : offset + size]
    if compression == 1:
        payload = gzip.decompress(payload)

    data = json.loads(payload.decode("utf-8")) if payload else {}
    return message_type, flags, error_code, data


def extract_stream_text(data: dict[str, Any]) -> str:
    """从不同的流式 ASR 返回结构中提取当前识别文字。"""

    result = data.get("result") or {}
    if not isinstance(result, dict):
        return ""

    text = str(result.get("text") or "").strip()
    if text:
        return text

    utterances = result.get("utterances") or []
    if isinstance(utterances, list):
        return "".join(
            str(item.get("text") or "")
            for item in utterances
            if isinstance(item, dict)
        ).strip()
    return ""


def extract_text(data: dict[str, Any]) -> str:
    """从普通 ASR 返回 JSON 中提取最终文字。"""

    result = data.get("result", {})
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    return str(data.get("text", "")).strip()


async def transcribe_wav_legacy(audio: bytes) -> str:
    """旧版 STT：Base64 直传 WAV，保留备用，不再作为主链路。"""

    # 普通 STT 的请求体不是二进制直传，而是把完整音频 Base64 放进 JSON。
    payload = {
        "user": {"uid": "cml-websit-backend"},
        "audio": {
            "format": "wav",
            "data": base64.b64encode(audio).decode("utf-8"),
        },
        "request": {
            "model_name": "bigmodel",
            "show_utterances": True,
            "enable_itn": True,
        },
    }

    try:
        # 这是一次性 HTTP 请求：等完整 WAV 上传并识别结束后才返回结果。
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                ASR_URL,
                headers=asr_headers(),
                json=payload,
            )

    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"STT 网络请求失败：{exc}") from exc

    try:
        # 正常返回应该是 JSON；如果服务异常返回纯文本，也保留原始内容用于报错。
        logger.debug("stt response: %s", response)
        data = response.json()
        logger.debug("stt response data: %s", data)

    except ValueError:
        data = {"raw": response.text}

    if response.status_code != 200:
        # HTTP 层失败，例如鉴权、资源 ID、请求格式错误。
        raise HTTPException(status_code=502, detail=f"语音识别失败：{data}")

    status_code = response.headers.get("X-Api-Status-Code")
    if status_code and status_code not in {"0", "20000000"}:
        # 有些语音接口 HTTP 是 200，但会在响应头里返回业务错误码。
        message = response.headers.get("X-Api-Message", "未知错误")
        raise HTTPException(
            status_code=502,
            detail=f"语音识别失败：{status_code} {message}",
        )

    text = extract_text(data)
    if not text:
        # 请求成功但没有识别文字，通常是音频为空、声音太小或格式不匹配。
        raise HTTPException(status_code=422, detail=f"没有识别到文字：{data}")
    return text


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:
    """给裸 PCM 补一个 WAV 文件头，供普通录音文件识别接口使用。"""

    # WAV = 文件头 + PCM 数据。
    # 这里不会重新编码声音，只是补充采样率、声道、位深等描述信息。
    data_size = len(pcm)
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width

    # RIFF/WAVE 是 WAV 文件的固定标识；后面的字段按 little-endian 写入。
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH",
            16,  # fmt 区块长度
            1,  # PCM 编码
            channels,
            sample_rate,
            byte_rate,
            block_align,
            sample_width * 8,
        )
        + b"data"
        + struct.pack("<I", data_size)
    )
    return header + pcm


async def transcribe_wav(
    audio: bytes,
    trace_id: str = "standalone",
) -> str:
    """使用新版 2608628 极速 HTTP 接口识别 WAV。

    新接口要求 ``audio.url``，所以这里先把 WAV 写入 uploads/asr，
    通过 FastAPI 的 /static 暴露公网地址，再提交给火山识别。
    """

    if not settings.public_base_url:
        raise HTTPException(
            status_code=500,
            detail=(
                "未配置 PUBLIC_BASE_URL。请填写火山可访问的公网 HTTPS 地址，"
                "例如 https://你的-ngrok-域名。"
            ),
        )

    total_started_at = perf_counter()
    request_id = str(uuid.uuid4())
    relative_path = Path("asr") / f"{request_id}.wav"
    file_path = Path(settings.upload_dir) / relative_path

    write_started_at = perf_counter()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(audio)
    logger.info(
        "[VOICE-TIMING][%s][STT] 临时 WAV 写入完成 elapsed=%.3fs bytes=%d",
        trace_id,
        perf_counter() - write_started_at,
        len(audio),
    )

    # 火山服务会从这个 URL 下载音频，因此不能填写 localhost 或 127.0.0.1。
    audio_url = (
        f"{settings.public_base_url.rstrip('/')}/static/"
        f"{relative_path.as_posix()}"
    )
    payload = {
        "audio": {
            "url": audio_url,
            "format": "wav",
            "codec": "raw",
            "rate": 16000,
            "bits": 16,
            "channel": 1,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "enable_ddc": True,
            "show_utterances": True,
            # 新版接口支持自动识别语种；后续可以从返回 additions 中读取语种。
            "enable_auto_lang": False,
            "enable_lid": True,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": settings.volc_asr_api_key.get_secret_value()
        if settings.volc_asr_api_key
        else "",
        "X-Api-Resource-Id": settings.volc_asr_resource_id,
        "X-Api-Request-Id": request_id,
        "X-Api-Sequence": "-1",
    }

    if not headers["X-Api-Key"]:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="未配置 VOLC_ASR_API_KEY，请在 .env 中填写火山语音 API Key。",
        )

    try:
        request_started_at = perf_counter()
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                ASR_URL,
                headers=headers,
                json=payload,
            )
        logger.info(
            "[VOICE-TIMING][%s][STT] 火山 HTTP 下载并识别完成 "
            "elapsed=%.3fs http_status=%d",
            trace_id,
            perf_counter() - request_started_at,
            response.status_code,
        )

        parse_started_at = perf_counter()
        try:
            data = response.json()
            logger.debug("STT payload: %s", payload)
            logger.debug("sttData: %s", data)
            logger.debug("response.headers: %s", response.headers)
            result = data.get("result") or {}
            for utterance in result.get("utterances") or []:
                additions = utterance.get("additions") or {}
                logger.debug("语种标签：%s", additions.get("lid_lang"))
            logger.debug("result additions：%s", result.get("additions"))
            logger.debug("result additions: %s", data["result"].get("additions"))
            for item in data["result"].get("utterances", []):
                logger.debug("utterance additions: %s", item.get("additions"))
        except ValueError:
            data = {"raw": response.text}

        status_code = response.headers.get("X-Api-Status-Code")
        if response.status_code != 200 or (
            status_code and status_code not in {"0", "20000000"}
        ):
            message = response.headers.get("X-Api-Message", "未知错误")
            log_id = response.headers.get("X-Tt-Logid", "")
            raise HTTPException(
                status_code=502,
                detail=f"新版 STT 失败：{status_code} {message} {data} logid={log_id}",
            )

        text = extract_text(data)
        if not text:
            raise HTTPException(status_code=422, detail=f"没有识别到文字：{data}")
        logger.info(
            "[VOICE-TIMING][%s][STT] 响应解析完成 elapsed=%.3fs chars=%d",
            trace_id,
            perf_counter() - parse_started_at,
            len(text),
        )
        logger.info(
            "[VOICE-TIMING][%s][STT] STT 全部完成 total=%.3fs",
            trace_id,
            perf_counter() - total_started_at,
        )
        return text
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"STT 网络请求失败：{exc}") from exc
    finally:
        # 火山已经同步返回识别结果后，临时音频不再需要，立即删除。
        file_path.unlink(missing_ok=True)


async def transcribe_pcm(
    pcm: bytes,
    sample_rate: int,
    channels: int,
    sample_width: int,
    trace_id: str = "standalone",
) -> str:
    """将长连接收到的一轮裸 PCM 转成 WAV 后进行普通 STT。

    小程序录音管理器给我们的是裸 PCM，而普通 STT 接口一次接收完整音频，
    所以这里承担“裸 PCM -> WAV -> 普通 HTTP STT”的适配工作。
    """

    if not pcm:
        raise HTTPException(status_code=422, detail="没有收到有效音频")

    # 先补 WAV 文件头，再调用新版 2608628 极速 HTTP 识别。
    convert_started_at = perf_counter()
    wav = pcm_to_wav(
        pcm,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )
    logger.info(
        "[VOICE-TIMING][%s][STT] PCM 转 WAV 完成 elapsed=%.3fs "
        "pcm_bytes=%d wav_bytes=%d",
        trace_id,
        perf_counter() - convert_started_at,
        len(pcm),
        len(wav),
    )
    return await transcribe_wav(wav, trace_id=trace_id)
