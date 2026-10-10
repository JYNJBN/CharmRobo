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

logger = logging.getLogger(__name__)


class HardwareQwenError(RuntimeError):
    """设备千问上游、协议或音频错误。"""


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
        self._send_audio = send_audio
        self._retrieve_references = retrieve_references
        self.result = HardwareQwenResult(started_at=started_at or time.perf_counter())
        self._commands: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue(
            settings.hardware_qwen_queue_size
        )
        self._mp3: asyncio.Queue[bytes | None] = asyncio.Queue(
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
        self._context_update_event_id: str | None = None
        self._deadline: float = 0
        self._phase = "配置会话"
        self._answer_parts: list[str] = []
        self._text_parts: list[str] = []
        self._asr_parts: list[str] = []
        self._last_pcm_at: float | None = None
        self._last_mp3_at: float | None = None

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
                raise HardwareQwenError(f"千问会话等待超时：{self._phase}")
            try:
                raw = await asyncio.wait_for(self._ws.recv(), min(remaining, 1.0))
                break
            except asyncio.TimeoutError:
                continue
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise HardwareQwenError("千问返回的事件不是对象")
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
            self._deadline = (
                time.monotonic() + settings.hardware_qwen_connect_timeout_seconds
            )
            # 与同事一样通过 instructions 传入人设和上下文，不新增多条 system item。
            await self._send(
                "session.update",
                session={
                    "modalities": ["text", "audio"],
                    "voice": settings.hardware_qwen_realtime_voice,
                    "instructions": self.instructions,
                    "turn_detection": None,
                    "enable_search": settings.hardware_qwen_realtime_enable_search,
                    "input_audio_transcription": {
                        "model": settings.hardware_qwen_realtime_transcription_model
                    },
                },
            )
            while (await self._recv()).get("type") != "session.updated":
                pass
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
            self._deadline = (
                time.monotonic() + settings.hardware_qwen_capture_timeout_seconds
            )
            self._phase = "设备录音结束"
            self._task = asyncio.create_task(self._run())
            logger.info(
                "[HARDWARE-QWEN][%s] 已就绪 model=%s voice=%s enable_search=%s",
                self.trace_id,
                settings.hardware_qwen_realtime_model,
                settings.hardware_qwen_realtime_voice,
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
        future = asyncio.run_coroutine_threadsafe(self._mp3.put(chunk), self._loop)
        try:
            future.result(timeout=5)
        except BaseException:
            future.cancel()
            raise

    async def _pump_mp3(self) -> None:
        while True:
            chunk = await self._mp3.get()
            if chunk is None:
                return
            if self._closed:
                return
            await asyncio.wait_for(self._send_audio(chunk), timeout=5)
            now = time.perf_counter()
            self.result.audio_bytes += len(chunk)
            self.result.audio_chunks += 1
            if self.result.first_chunk_ms is None:
                self.result.first_chunk_ms = int((now - self.result.started_at) * 1000)
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
                await self._send("input_audio_buffer.commit")
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
                if not self.result.asr_text:
                    raise HardwareQwenError("没有识别到有效语音，请重试")
                logger.info(
                    "[HARDWARE-QWEN][%s] ASR 完成，开始本轮检索 内容=%s",
                    self.trace_id,
                    self.result.asr_text,
                )
                references = await self._retrieve_references(self.result.asr_text)
                instructions = self.instructions + (
                    "\n\n" + references if references else ""
                )
                self._context_update_event_id = await self._send(
                    "session.update", session={"instructions": instructions}
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
                self._response_requested = True
                self._phase = "回答首段语音"
                self._deadline = (
                    time.monotonic()
                    + settings.hardware_qwen_first_response_timeout_seconds
                )
                await self._send("response.create")
                logger.info(
                    "[HARDWARE-QWEN][%s] 资料更新已确认，已请求回答", self.trace_id
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
                    self.result.first_provider_audio_ms = int(
                        (now - self.result.started_at) * 1000
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
                self._answer_parts.append(str(event.get("delta") or ""))
            elif event_type in {
                "response.audio_transcript.done",
                "response.output_audio_transcript.done",
            }:
                if not self._answer_parts:
                    self._answer_parts.append(
                        str(event.get("transcript") or event.get("text") or "")
                    )
            elif event_type in {"response.text.delta", "response.output_text.delta"}:
                self._text_parts.append(str(event.get("delta") or ""))
            elif event_type in {"response.text.done", "response.output_text.done"}:
                if not self._text_parts:
                    self._text_parts.append(str(event.get("text") or ""))
            elif event_type == "response.done":
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
                usage = response.get("usage") or {}
                plugins = usage.get("plugins") or {}
                search = plugins.get("search") or {}
                self.result.search_count = (
                    search.get("count") if isinstance(search, dict) else None
                )
                self.result.finish_reason = "response.done:completed"
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
            return self.result
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
