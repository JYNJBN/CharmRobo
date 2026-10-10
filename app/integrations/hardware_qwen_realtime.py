"""Qwen 设备端到端：同事的 PTT 协议与 MP3 桥，适配 FastAPI asyncio。

音频队列、200ms 尾静音、Tina、24k PCM→16k MP3 均沿用设备方案。
区别是 commit 后等待输入转写、检索业务资料并更新 instructions，再创建回答。
一个实例只处理一轮；设备长连接上的下一轮会重新读取当前智能体。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets
from websockets.exceptions import WebSocketException

from app.core.config import settings
from app.integrations.hardware_mp3_bridge import PcmToCiMp3Bridge
from app.utils.tools import BEIJING_TIMEZONE, build_current_time_context, local_now

logger = logging.getLogger(__name__)

ASR_EMPTY_REPEAT_PROMPT = "我没听清，请再说一遍。"


def build_hardware_qwen_session_config(
    *,
    model: str,
    voice: str,
    instructions: str,
) -> dict[str, Any]:
    """Qwen Audio 与 Omni 使用不同音色和转写配置，不能只替换 model。"""
    session: dict[str, Any] = {
        "modalities": ["text", "audio"],
        "voice": voice,
        "instructions": instructions,
        "turn_detection": None,
        "enable_search": settings.hardware_qwen_realtime_enable_search,
        "input_audio_format": "pcm",
        "output_audio_format": "pcm",
    }
    if model.startswith("qwen-audio-"):
        # Audio 3.x 默认内置转写；Omni 的 input_audio_transcription.model 不适用。
        # 仅转换遗留的 Omni 默认音色，其他用户配置/复刻音色仍原样传给上游。
        if voice == "Tina":
            session["voice"] = (
                "longanqian_v3.1"
                if model.startswith("qwen-audio-3.1-")
                else "longanqian"
            )
    else:
        session["input_audio_transcription"] = {
            "model": settings.hardware_qwen_realtime_transcription_model,
        }
    return session


class HardwareQwenError(RuntimeError):
    """设备千问上游、协议或音频错误。"""

    error_code = "hardware_qwen_error"
    retryable = False


class HardwareQwenTimeout(HardwareQwenError):
    """服务端主动限制等待，不等同于物理网络断开。"""

    def __init__(self, message: str, *, error_code: str, phase: str, retryable: bool):
        super().__init__(message)
        self.error_code = error_code
        self.phase = phase
        # 只有尚未发出 MP3 时才能安全从新一轮重试；服务端不自动重复播放。
        self.retryable = retryable


@dataclass
class HardwareQwenResult:
    started_at: float = field(default_factory=time.perf_counter)
    asr_text: str = ""
    answer_text: str = ""
    input_audio_bytes: int = 0
    input_audio_chunks: int = 0
    endpoint_silence_bytes: int = 0
    endpoint_silence_chunks: int = 0
    output_pcm_bytes: int = 0
    output_pcm_chunks: int = 0
    post_endpoint_input_bytes: int = 0
    post_endpoint_input_chunks: int = 0
    audio_bytes: int = 0
    audio_chunks: int = 0
    first_chunk_ms: int | None = None
    first_provider_audio_ms: int | None = None
    provider_audio_max_gap_ms: int = 0
    mp3_output_max_gap_ms: int = 0
    finish_reason: str = ""
    search_count: int | None = None
    timings_ms: dict[str, int] = field(default_factory=dict)
    needs_repeat: bool = False


class HardwareQwenRealtimeSession:
    def __init__(
        self,
        *,
        instructions: str,
        trace_id: str,
        send_audio: Callable[[bytes], Awaitable[None]],
        retrieve_references: Callable[[str], Awaitable[str]],
        started_at: float | None = None,
    ) -> None:
        self.instructions = instructions
        self.trace_id = trace_id
        self._voice = settings.hardware_qwen_realtime_voice
        self._send_audio = send_audio
        self._retrieve_references = retrieve_references
        self.result = HardwareQwenResult(started_at=started_at or time.perf_counter())
        self._commands: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue(
            settings.hardware_qwen_queue_size
        )
        self._mp3: asyncio.Queue[tuple[bytes, float] | None] = asyncio.Queue(
            settings.hardware_qwen_queue_size
        )
        self._ws: Any = None
        self._bridge: PcmToCiMp3Bridge | None = None
        self._task: asyncio.Task[HardwareQwenResult] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._bridge_error: asyncio.Future[Exception] | None = None
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._commit_enqueued = False
        self._committed = False
        self._response_requested = False
        self._active_search_enabled = settings.hardware_qwen_realtime_enable_search
        self._context_update_event_id: str | None = None
        self._deadline: float = 0
        self._phase = "配置会话"
        self._answer_parts: list[str] = []
        self._text_parts: list[str] = []
        self._asr_parts: list[str] = []
        self._last_pcm_at: float | None = None
        self._last_mp3_at: float | None = None
        self._last_event_type = "未返回"
        self._upstream_session_id = "未返回"
        self._upstream_response_id = "未返回"
        self._next_first_audio_wait_log_seconds = 10
        self._marks: dict[str, float] = {"dialogue_start": self.result.started_at}

    def _mark(self, name: str, at: float | None = None) -> float:
        value = time.perf_counter() if at is None else at
        self._marks[name] = value
        return value

    def _duration(self, start: str, end: str) -> int | None:
        if start not in self._marks or end not in self._marks:
            return None
        return max(0, int((self._marks[end] - self._marks[start]) * 1000))

    def _update_timings(self) -> None:
        pairs = {
            "connect_ms": ("connect_start", "connect_done"),
            "session_config_ms": ("session_config_start", "session_config_done"),
            "bridge_start_ms": ("bridge_start", "ready"),
            "start_to_ready_ms": ("dialogue_start", "ready"),
            "capture_ms": ("first_input", "capture_end"),
            "end_to_commit_ms": ("capture_end", "commit_start"),
            "commit_send_ms": ("commit_start", "commit_sent"),
            "asr_ms": ("commit_start", "asr_done"),
            "retrieval_ms": ("retrieval_start", "retrieval_done"),
            "context_update_ms": ("context_update", "context_confirmed"),
            "response_to_first_text_ms": ("response_requested", "first_text"),
            "end_to_first_text_ms": ("capture_end", "first_text"),
            "response_to_first_pcm_ms": ("response_requested", "first_pcm"),
            "end_to_first_pcm_ms": ("capture_end", "first_pcm"),
            "pcm_to_first_mp3_ms": ("first_pcm", "first_mp3_generated"),
            "first_mp3_queue_ms": ("first_mp3_generated", "first_mp3_send_start"),
            "first_mp3_send_ms": ("first_mp3_send_start", "first_mp3_sent"),
            "end_to_first_audio_ms": ("capture_end", "first_mp3_sent"),
            "response_to_first_audio_ms": ("response_requested", "first_mp3_sent"),
            "response_ms": ("response_requested", "response_done"),
            "mp3_drain_ms": ("response_done", "mp3_done"),
            "start_to_audio_done_ms": ("dialogue_start", "mp3_done"),
        }
        for key, (start, end) in pairs.items():
            duration = self._duration(start, end)
            if duration is not None:
                self.result.timings_ms[key] = duration

    def _turn_instructions(self, references: str = "") -> str:
        # 首次配置和 ASR/RAG 后都刷新，录音跨过午夜时仍使用回答时的日期。
        current = local_now()
        sections = [self.instructions]
        if references:
            sections.append(references)
        sections.append(build_current_time_context(current))
        logger.info(
            "[HARDWARE-QWEN][%s][时间基准] 北京时间=%s",
            self.trace_id,
            current.astimezone(BEIJING_TIMEZONE).isoformat(timespec="seconds"),
        )
        return "\n\n".join(sections)

    def _first_text(self, text: str) -> None:
        if not text or "first_text" in self._marks:
            return
        self._mark("first_text")
        self._update_timings()
        logger.info(
            "[HARDWARE-QWEN][%s][首段文字] 请求回答后=%sms 录音结束后=%sms（不是设备实际发声时间）",
            self.trace_id,
            self._duration("response_requested", "first_text"),
            self._duration("capture_end", "first_text"),
        )

    async def _send(self, event_type: str, **fields: Any) -> str:
        event_id = f"hardware_qwen_{uuid.uuid4().hex}"
        await self._ws.send(
            json.dumps(
                {"type": event_type, "event_id": event_id, **fields},
                ensure_ascii=False,
            )
        )
        return event_id

    async def _recv(self) -> dict[str, Any]:
        # sender 会在 commit 时缩短等待时间，定期重新读取最新 deadline。
        while True:
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                first_audio = (
                    self._response_requested and self.result.output_pcm_bytes == 0
                )
                code = (
                    "upstream_first_audio_timeout"
                    if first_audio
                    else "upstream_wait_timeout"
                )
                message = (
                    f"千问生成首段语音超时（等待上限{settings.hardware_qwen_first_response_timeout_seconds:g}秒），请稍后再试"
                    if first_audio
                    else f"千问会话等待超时：{self._phase}"
                )
                logger.warning(
                    "[HARDWARE-QWEN][%s][等待超时] error_code=%s 阶段=%s 已收到文字=%d PCM字节=%d 已发MP3=%d 最后事件=%s session_id=%s response_id=%s",
                    self.trace_id,
                    code,
                    self._phase,
                    len("".join(self._answer_parts or self._text_parts)),
                    self.result.output_pcm_bytes,
                    self.result.audio_bytes,
                    self._last_event_type,
                    self._upstream_session_id,
                    self._upstream_response_id,
                )
                raise HardwareQwenTimeout(
                    message,
                    error_code=code,
                    phase=self._phase,
                    retryable=self.result.audio_bytes == 0,
                )
            if self._response_requested and self.result.output_pcm_bytes == 0:
                waited = time.perf_counter() - self._marks["response_requested"]
                if waited >= self._next_first_audio_wait_log_seconds:
                    logger.info(
                        "[HARDWARE-QWEN][%s][等待首音频] 已等=%.1f秒 上限=%.1f秒 已收到文字=%d 最后事件=%s",
                        self.trace_id,
                        waited,
                        settings.hardware_qwen_first_response_timeout_seconds,
                        len("".join(self._answer_parts or self._text_parts)),
                        self._last_event_type,
                    )
                    self._next_first_audio_wait_log_seconds += 10
            try:
                raw = await asyncio.wait_for(self._ws.recv(), min(remaining, 1.0))
                break
            except asyncio.TimeoutError:
                continue
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise HardwareQwenError("千问返回的事件不是对象")
        self._last_event_type = str(event.get("type") or "unknown")
        if self._last_event_type == "session.created":
            self._upstream_session_id = str(
                (event.get("session") or {}).get("id") or "未返回"
            )
        if event.get("type") == "error":
            raise HardwareQwenError(
                f"千问上游错误：{json.dumps(event.get('error'), ensure_ascii=False)}"
            )
        return event

    async def start(self) -> None:
        """先等待 session.updated，再准备桥和收发任务，最后才允许设备上传。"""
        if self._closed or self._ws is not None:
            raise HardwareQwenError("会话已启动或已关闭")
        key = settings.hardware_qwen_realtime_api_key
        if key is None or not key.get_secret_value().strip():
            key = settings.qwen_api_key
        if key is None or not key.get_secret_value().strip():
            raise HardwareQwenError(
                "未配置 HARDWARE_QWEN_REALTIME_API_KEY 或 QWEN_API_KEY"
            )
        parsed = urlsplit(settings.hardware_qwen_realtime_ws_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query["model"] = settings.hardware_qwen_realtime_model
        url = urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query),
                parsed.fragment,
            )
        )
        self._loop = asyncio.get_running_loop()
        self._bridge_error = self._loop.create_future()
        try:
            self._mark("connect_start")
            self._ws = await websockets.connect(
                url,
                additional_headers={
                    "Authorization": f"Bearer {key.get_secret_value()}"
                },
                ping_interval=None,
                open_timeout=settings.hardware_qwen_connect_timeout_seconds,
                close_timeout=3,
                max_size=2**24,
            )
            self._mark("connect_done")
            logger.info(
                "[HARDWARE-QWEN][%s][上游连接] 耗时=%sms",
                self.trace_id,
                self._duration("connect_start", "connect_done"),
            )
            self._deadline = (
                time.monotonic() + settings.hardware_qwen_connect_timeout_seconds
            )
            # 与同事一样通过 instructions 传入人设和上下文，不新增多条 system item。
            self._mark("session_config_start")
            session_config = build_hardware_qwen_session_config(
                model=settings.hardware_qwen_realtime_model,
                voice=settings.hardware_qwen_realtime_voice,
                instructions=self._turn_instructions(),
            )
            self._voice = str(session_config["voice"])
            if self._voice != settings.hardware_qwen_realtime_voice:
                logger.warning(
                    "[HARDWARE-QWEN][%s][音色适配] model=%s 不使用 Omni 音色=%s，本轮使用=%s；可通过 HARDWARE_QWEN_REALTIME_VOICE 指定匹配音色",
                    self.trace_id,
                    settings.hardware_qwen_realtime_model,
                    settings.hardware_qwen_realtime_voice,
                    self._voice,
                )
            await self._send(
                "session.update",
                session=session_config,
            )
            while True:
                event = await self._recv()
                if event.get("type") == "session.updated":
                    applied = event.get("session") or {}
                    logger.info(
                        "[HARDWARE-QWEN][%s][联网配置] 请求enable_search=%s 上游确认=%s",
                        self.trace_id,
                        settings.hardware_qwen_realtime_enable_search,
                        applied.get("enable_search", "未返回"),
                    )
                    break
            self._mark("session_config_done")
            logger.info(
                "[HARDWARE-QWEN][%s][初始配置] 确认耗时=%sms",
                self.trace_id,
                self._duration("session_config_start", "session_config_done"),
            )
            self._mark("bridge_start")
            bridge_task = asyncio.create_task(
                asyncio.to_thread(
                    PcmToCiMp3Bridge,
                    input_rate=24000,
                    bitrate=settings.hardware_qwen_mp3_bitrate,
                    queue_size=settings.hardware_qwen_queue_size,
                    put_timeout_sec=0.2,
                    close_timeout_sec=60,
                    on_mp3=self._on_mp3,
                    on_error=self._on_bridge_error,
                    on_trace=self._trace,
                )
            )
            try:
                self._bridge = await asyncio.shield(bridge_task)
            except asyncio.CancelledError:
                # 构造器在线程中，取消 asyncio 等待不会取消已启动的 FFmpeg。
                self._bridge = await bridge_task
                raise
            self._mark("ready")
            logger.info(
                "[HARDWARE-QWEN][%s][转码器启动] 耗时=%sms 从对话开始到就绪=%sms",
                self.trace_id,
                self._duration("bridge_start", "ready"),
                self._duration("dialogue_start", "ready"),
            )
            self._deadline = (
                time.monotonic() + settings.hardware_qwen_capture_timeout_seconds
            )
            self._phase = "设备录音结束"
            self._task = asyncio.create_task(self._run())
            logger.info(
                "[HARDWARE-QWEN][%s] 已就绪 model=%s voice=%s enable_search=%s",
                self.trace_id,
                settings.hardware_qwen_realtime_model,
                self._voice,
                settings.hardware_qwen_realtime_enable_search,
            )
        except BaseException:
            await self.aclose()
            raise

    def append_audio(self, pcm: bytes) -> bool:
        if (
            not pcm
            or self._closed
            or self._commit_enqueued
            or self._task is None
            or self._task.done()
        ):
            return False
        try:
            self._commands.put_nowait(("append", pcm))
        except asyncio.QueueFull:
            return False
        if "first_input" not in self._marks:
            self._mark("first_input")
        self.result.input_audio_bytes += len(pcm)
        self.result.input_audio_chunks += 1
        return True

    def commit(self) -> bool:
        if self._commit_enqueued:
            return True
        if self._closed or self._task is None or self._task.done():
            return False
        try:
            self._commands.put_nowait(("commit", None))
        except asyncio.QueueFull:
            return False
        self._mark("capture_end")
        logger.info(
            "[HARDWARE-QWEN][%s][录音结束] 收音耗时=%sms 从对话开始=%sms 帧数=%d 字节=%d",
            self.trace_id,
            self._duration("first_input", "capture_end"),
            self._duration("dialogue_start", "capture_end"),
            self.result.input_audio_chunks,
            self.result.input_audio_bytes,
        )
        self._commit_enqueued = True
        return True

    async def wait(self) -> HardwareQwenResult:
        if self._task is None:
            raise HardwareQwenError("会话尚未启动")
        return await self._task

    def _trace(self, stage: str, fields: dict[str, object]) -> None:
        logger.debug("[HARDWARE-QWEN][%s] %s %s", self.trace_id, stage, fields)

    def _on_bridge_error(self, error: Exception) -> None:
        def report() -> None:
            if self._bridge_error is not None and not self._bridge_error.done():
                self._bridge_error.set_result(error)

        if self._loop is not None and not self._closed:
            self._loop.call_soon_threadsafe(report)

    def _on_mp3(self, chunk: bytes) -> None:
        # FFmpeg reader 线程等待有界下行队列，避免慢设备造成无限积压。
        if self._closed or self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._mp3.put((chunk, time.perf_counter())), self._loop
        )
        try:
            future.result(timeout=5)
        except BaseException:
            future.cancel()
            raise

    async def _pump_mp3(self) -> None:
        while True:
            item = await self._mp3.get()
            if item is None:
                return
            chunk, generated_at = item
            if self._closed:
                return
            first = self.result.first_chunk_ms is None
            if first:
                self._mark("first_mp3_generated", generated_at)
                self._mark("first_mp3_send_start")
            await asyncio.wait_for(self._send_audio(chunk), timeout=5)
            now = time.perf_counter()
            self.result.audio_bytes += len(chunk)
            self.result.audio_chunks += 1
            if self.result.first_chunk_ms is None:
                self._mark("first_mp3_sent", now)
                self.result.first_chunk_ms = int((now - self.result.started_at) * 1000)
                self._update_timings()
                logger.info(
                    "[HARDWARE-QWEN][%s][首音频发送] 录音结束到首包=%sms 请求回答到首包=%sms PCM转MP3=%sms 排队=%sms 发送=%sms 从对话开始=%sms（服务器发出，不是设备实际发声）",
                    self.trace_id,
                    self.result.timings_ms.get("end_to_first_audio_ms"),
                    self.result.timings_ms.get("response_to_first_audio_ms"),
                    self.result.timings_ms.get("pcm_to_first_mp3_ms"),
                    self.result.timings_ms.get("first_mp3_queue_ms"),
                    self.result.timings_ms.get("first_mp3_send_ms"),
                    self.result.first_chunk_ms,
                )
            if self._last_mp3_at is not None:
                self.result.mp3_output_max_gap_ms = max(
                    self.result.mp3_output_max_gap_ms,
                    int((now - self._last_mp3_at) * 1000),
                )
            self._last_mp3_at = now

    async def _send_commands(self) -> None:
        pending = b""
        while True:
            name, payload = await self._commands.get()
            if name == "append":
                # 一个 PCM 样本可能被设备拆到两个 WS 帧，保持完整 s16le 样本。
                data = pending + (payload or b"")
                even_length = len(data) - len(data) % 2
                pending = data[even_length:]
                if even_length:
                    await self._send(
                        "input_audio_buffer.append",
                        audio=base64.b64encode(data[:even_length]).decode("ascii"),
                    )
            elif name == "commit":
                self._mark("commit_prepare_start")
                if pending:
                    raise HardwareQwenError("PCM 字节数不是完整的 16bit 样本")
                count = math.ceil(settings.hardware_qwen_realtime_tail_silence_ms / 20)
                for _ in range(count):
                    await self._send(
                        "input_audio_buffer.append",
                        audio=base64.b64encode(b"\0" * 640).decode("ascii"),
                    )
                self.result.endpoint_silence_chunks = count
                self.result.endpoint_silence_bytes = count * 640
                self._committed = True
                self._phase = "输入语音转写"
                self._deadline = (
                    time.monotonic() + settings.hardware_qwen_asr_timeout_seconds
                )
                self._mark("commit_start")
                await self._send("input_audio_buffer.commit")
                self._mark("commit_sent")
                logger.info(
                    "[HARDWARE-QWEN][%s][音频提交] 队列及尾静音发送=%sms commit发送=%sms 补静音=%dms",
                    self.trace_id,
                    self._duration("capture_end", "commit_start"),
                    self._duration("commit_start", "commit_sent"),
                    count * 20,
                )
                # 不立即 response.create：必须等转写和本地检索完成。

    async def _receive_events(self) -> None:
        while True:
            event = await self._recv()
            event_type = event.get("type", "")
            if event_type == "conversation.item.input_audio_transcription.delta":
                self._asr_parts.append(str(event.get("delta") or ""))
            elif event_type == "conversation.item.input_audio_transcription.failed":
                raise HardwareQwenError("千问输入转写失败")
            elif event_type == "conversation.item.input_audio_transcription.completed":
                if (
                    not self._committed
                    or self._response_requested
                    or self._context_update_event_id
                ):
                    continue
                self.result.asr_text = str(
                    event.get("transcript")
                    or event.get("text")
                    or "".join(self._asr_parts)
                ).strip()
                self._mark("asr_done")
                logger.info(
                    "[HARDWARE-QWEN][%s][识别完成] commit后=%sms 录音结束后=%sms 内容=%s",
                    self.trace_id,
                    self._duration("commit_start", "asr_done"),
                    self._duration("capture_end", "asr_done"),
                    self.result.asr_text,
                )
                if not self.result.asr_text:
                    self.result.needs_repeat = True
                    self._active_search_enabled = False
                    instructions = (
                        "本轮音频未识别到有效文本。不要猜测用户说了什么，"
                        "不回应之前的对话，不使用业务人设、知识库或联网搜索。"
                        f"只用当前音色清晰地说这一句话：{ASR_EMPTY_REPEAT_PROMPT}"
                        "不要补充其他内容，不要说网络错误。"
                    )
                    logger.info(
                        "[HARDWARE-QWEN][%s][未听清提示] 转写完成但文本为空，跳过检索和记忆保存，准备播报=%s",
                        self.trace_id,
                        ASR_EMPTY_REPEAT_PROMPT,
                    )
                else:
                    self._mark("retrieval_start")
                    self._phase = "本轮资料检索"
                    references = await self._retrieve_references(self.result.asr_text)
                    self._mark("retrieval_done")
                    logger.info(
                        "[HARDWARE-QWEN][%s][资料检索] 总耗时=%sms 参考字数=%d",
                        self.trace_id,
                        self._duration("retrieval_start", "retrieval_done"),
                        len(references),
                    )
                    instructions = self._turn_instructions(references)
                self._mark("context_update")
                updated_session: dict[str, Any] = {"instructions": instructions}
                if self.result.needs_repeat:
                    updated_session["enable_search"] = False
                self._context_update_event_id = await self._send(
                    "session.update", session=updated_session
                )
                self._phase = "本轮参考资料更新确认"
                self._deadline = (
                    time.monotonic() + settings.hardware_qwen_connect_timeout_seconds
                )
            elif (
                event_type == "session.updated"
                and self._context_update_event_id
                and not self._response_requested
            ):
                self._mark("context_confirmed")
                self._response_requested = True
                self._phase = "回答首段语音"
                self._deadline = (
                    time.monotonic()
                    + settings.hardware_qwen_first_response_timeout_seconds
                )
                self._mark("response_requested")
                await self._send("response.create")
                logger.info(
                    "[HARDWARE-QWEN][%s][请求回答] 资料更新确认=%sms 录音结束后=%sms enable_search=%s 首音频上限=%.1f秒",
                    self.trace_id,
                    self._duration("context_update", "context_confirmed"),
                    self._duration("capture_end", "response_requested"),
                    self._active_search_enabled,
                    settings.hardware_qwen_first_response_timeout_seconds,
                )
            elif event_type == "response.created":
                self._mark("response_created")
                self._upstream_response_id = str(
                    (event.get("response") or {}).get("id") or "未返回"
                )
                logger.info(
                    "[HARDWARE-QWEN][%s][生成已开始] 请求确认=%sms",
                    self.trace_id,
                    self._duration("response_requested", "response_created"),
                )
            elif event_type in {"response.audio.delta", "response.output_audio.delta"}:
                if not self._response_requested:
                    raise HardwareQwenError("上游在参考资料注入前提前生成了回答")
                pcm = base64.b64decode(str(event.get("delta") or ""), validate=True)
                if not pcm:
                    continue
                now = time.perf_counter()
                if self._last_pcm_at is not None:
                    self.result.provider_audio_max_gap_ms = max(
                        self.result.provider_audio_max_gap_ms,
                        int((now - self._last_pcm_at) * 1000),
                    )
                self._last_pcm_at = now
                if self.result.first_provider_audio_ms is None:
                    self._mark("first_pcm", now)
                    self.result.first_provider_audio_ms = int(
                        (now - self.result.started_at) * 1000
                    )
                    logger.info(
                        "[HARDWARE-QWEN][%s][首段PCM] 请求回答后=%sms 录音结束后=%sms 字节=%d",
                        self.trace_id,
                        self._duration("response_requested", "first_pcm"),
                        self._duration("capture_end", "first_pcm"),
                        len(pcm),
                    )
                self.result.output_pcm_bytes += len(pcm)
                self.result.output_pcm_chunks += 1
                self._phase = "后续语音或回答结束"
                self._deadline = (
                    time.monotonic()
                    + settings.hardware_qwen_output_idle_timeout_seconds
                )
                if self._bridge is None or not await asyncio.to_thread(
                    self._bridge.write, pcm
                ):
                    raise HardwareQwenError("PCM 到 MP3 转换桥不可用")
            elif event_type in {
                "response.audio_transcript.delta",
                "response.output_audio_transcript.delta",
            }:
                text = str(event.get("delta") or "")
                self._first_text(text)
                self._answer_parts.append(text)
            elif event_type in {
                "response.audio_transcript.done",
                "response.output_audio_transcript.done",
            }:
                if not self._answer_parts:
                    text = str(event.get("transcript") or event.get("text") or "")
                    self._first_text(text)
                    self._answer_parts.append(text)
            elif event_type in {"response.text.delta", "response.output_text.delta"}:
                text = str(event.get("delta") or "")
                self._first_text(text)
                self._text_parts.append(text)
            elif event_type in {"response.text.done", "response.output_text.done"}:
                if not self._text_parts:
                    text = str(event.get("text") or "")
                    self._first_text(text)
                    self._text_parts.append(text)
            elif event_type == "response.done":
                self._mark("response_done")
                response = event.get("response") or {}
                if response.get("status") != "completed":
                    raise HardwareQwenError(
                        f"千问回答未成功完成：{json.dumps(response.get('status_details') or response.get('status'), ensure_ascii=False)}"
                    )
                self.result.answer_text = "".join(
                    self._answer_parts or self._text_parts
                ).strip()
                if not self.result.answer_text:
                    self.result.answer_text = self._final_transcript(response)
                if not self.result.answer_text or not self.result.output_pcm_bytes:
                    raise HardwareQwenError("千问没有返回完整的语音回答")
                self._first_text(self.result.answer_text)
                usage = response.get("usage") or {}
                plugins = usage.get("plugins") or {}
                search = plugins.get("search") or {}
                self.result.search_count = (
                    search.get("count") if isinstance(search, dict) else None
                )
                self.result.finish_reason = "response.done:completed"
                logger.info(
                    "[HARDWARE-QWEN][%s][上游回答结束] 生成耗时=%sms 回复字数=%d enable_search=%s 搜索次数=%s",
                    self.trace_id,
                    self._duration("response_requested", "response_done"),
                    len(self.result.answer_text),
                    self._active_search_enabled,
                    self.result.search_count
                    if self.result.search_count is not None
                    else "未返回",
                )
                return

    @staticmethod
    def _final_transcript(response: dict[str, Any]) -> str:
        parts = []
        for item in response.get("output") or []:
            for content in item.get("content") or []:
                value = content.get("transcript") or content.get("text")
                if value:
                    parts.append(str(value))
        return "".join(parts).strip()

    async def _run(self) -> HardwareQwenResult:
        sender = asyncio.create_task(self._send_commands())
        receiver = asyncio.create_task(self._receive_events())
        pump = asyncio.create_task(self._pump_mp3())
        tasks = (sender, receiver, pump)
        try:
            watched = {sender, receiver, pump, self._bridge_error}
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
            if self._bridge_error in done:
                raise HardwareQwenError(str(self._bridge_error.result()))
            for task in done:
                task.result()
            if receiver not in done:
                raise HardwareQwenError("千问收发任务提前结束")
            # 保持下行 pump 活跃，先排空 FFmpeg，再排空 MP3 队列。
            self._phase = "剩余MP3转码及下发"
            bridge = self._bridge
            if bridge is None:
                raise HardwareQwenError("音频转换桥已关闭")
            drain = asyncio.create_task(
                asyncio.to_thread(bridge.close, report_nonzero=False)
            )
            try:
                done, _ = await asyncio.wait(
                    {drain, pump, self._bridge_error},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._bridge_error in done:
                    raise HardwareQwenError(str(self._bridge_error.result()))
                if pump in done:
                    pump.result()
                    raise HardwareQwenError("设备音频下行提前结束")
                code, detail = drain.result()
                if code not in (0, None) or detail:
                    raise HardwareQwenError(f"MP3 转码未完整结束：{detail}")
            finally:
                # 已开始正常排空时，取消也要等待 close 回收转换线程。
                await asyncio.shield(drain)
            self._bridge = None
            await self._mp3.put(None)
            await pump
            self._mark("mp3_done")
            self._update_timings()
            self._phase = "语音已发完"
            if not self.result.audio_bytes:
                raise HardwareQwenError("没有生成设备可播放的 MP3")
            logger.info(
                "[HARDWARE-QWEN][%s] 回答完成 MP3字节=%d 首包=%sms 上游最大间隔=%dms 下行最大间隔=%dms 搜索次数=%s",
                self.trace_id,
                self.result.audio_bytes,
                self.result.first_chunk_ms,
                self.result.provider_audio_max_gap_ms,
                self.result.mp3_output_max_gap_ms,
                self.result.search_count,
            )
            logger.info(
                "[HARDWARE-QWEN][%s][语音阶段汇总] timings_ms=%s",
                self.trace_id,
                json.dumps(self.result.timings_ms, ensure_ascii=False),
            )
            return self.result
        except BaseException:
            self._update_timings()
            logger.info(
                "[HARDWARE-QWEN][%s][中断阶段汇总] 当前阶段=%s timings_ms=%s",
                self.trace_id,
                self._phase,
                json.dumps(self.result.timings_ms, ensure_ascii=False),
            )
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        self._closed = True  # 先阻止转码线程继续排入下行音频。
        async with self._close_lock:
            if self._task is not None and not self._task.done():
                self._task.cancel()
            if self._task is not None:
                await asyncio.gather(self._task, return_exceptions=True)
            if self._bridge is not None:
                bridge, self._bridge = self._bridge, None
                await asyncio.to_thread(bridge.close, report_nonzero=False, drain=False)
            if self._ws is not None:
                if not self.result.finish_reason:
                    # 同事的取消顺序：取消已开始的响应，结束会话，再关闭 WS。
                    kinds = (
                        ["response.cancel"] if self._response_requested else []
                    ) + ["session.finish"]
                    for kind in kinds:
                        try:
                            await asyncio.wait_for(self._send(kind), timeout=1)
                        except (OSError, asyncio.TimeoutError, WebSocketException):
                            break
                ws, self._ws = self._ws, None
                try:
                    await ws.close()
                except Exception:
                    logger.warning(
                        "[HARDWARE-QWEN][%s] 上游关闭时出错",
                        self.trace_id,
                        exc_info=True,
                    )
