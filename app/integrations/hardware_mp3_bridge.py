"""设备 PCM→MP3 转换桥，迁移自 ASR_LLM_TTS_Server 的 realtime.py。

保留 FFmpeg 参数、输入/输出线程、有界队列及 CI 首帧兼容修正。
只负责音频转换；业务和 Qwen 协议由独立模块处理。
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from collections.abc import Callable

AudioCallback = Callable[[bytes], None]
ErrorCallback = Callable[[Exception], None]
TraceCallback = Callable[[str, dict[str, object]], None]


class ProviderError(RuntimeError):
    """设备音频转换错误。"""


class PcmToCiMp3Bridge:
    """将 24 kHz PCM 持续转换为 CI 可播放的 MP3 数据流。"""

    def __init__(
        self,
        input_rate: int,
        bitrate: str,
        queue_size: int,
        put_timeout_sec: float,
        close_timeout_sec: float,
        on_mp3: AudioCallback,
        on_error: ErrorCallback,
        on_trace: TraceCallback | None = None,
    ) -> None:
        """启动独立输入、输出线程，避免转码阻塞实时模型收包循环。"""

        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise ProviderError(f"无法取得实时音频转码器: {exc}") from exc

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            # 原始 PCM 无需探测；关闭探测缓存，让转码器尽快吐出首个 MP3 帧。
            "-fflags",
            "+nobuffer",
            "-flags",
            "low_delay",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
            "-f",
            "s16le",
            "-ar",
            str(input_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-write_xing",
            "0",
            "-id3v2_version",
            "0",
            "-flush_packets",
            "1",
            "-f",
            "mp3",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                # stdin 使用无缓冲管道，避免 FFmpeg 提前退出时 BufferedWriter 析构再次 flush 并打印噪声。
                bufsize=0,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise ProviderError(f"启动实时音频转码器失败: {exc}") from exc

        self._input_queue_maxsize = max(4, queue_size)
        self._put_timeout_sec = max(0.0, put_timeout_sec)
        self._close_timeout_sec = max(1.0, close_timeout_sec)
        self._input_bytes_per_second = max(1, input_rate * 2)
        self._input_queue: queue.Queue[bytes | None] = queue.Queue(
            maxsize=self._input_queue_maxsize
        )
        self._on_mp3 = on_mp3
        self._on_error = on_error
        self._on_trace = on_trace or (lambda _stage, _fields: None)
        self._closed = threading.Event()
        self._reader_failed = threading.Event()
        self._reader_error: Exception | None = None
        self._input_pcm_bytes = 0
        self._input_pcm_chunks = 0
        self._written_pcm_bytes = 0
        self._written_pcm_chunks = 0
        # CI 简易 MP3 解码器会误把任意 MP3 第 6~9 字节当作 ID3 长度；
        # 首次输出前暂存数据，以便安全修正该兼容字段。
        self._ci_first_mp3_pending = True
        self._ci_first_mp3_prefix = bytearray()
        self._writer = threading.Thread(
            target=self._write_loop, name="realtime-pcm-writer", daemon=True
        )
        self._reader = threading.Thread(
            target=self._read_loop, name="realtime-mp3-reader", daemon=True
        )
        self._writer.start()
        self._reader.start()

    def write(self, pcm: bytes) -> bool:
        """把一段模型 PCM 放入有界转码队列，满载时不阻塞调用线程。"""

        if not pcm or self._closed.is_set():
            return False
        try:
            if self._put_timeout_sec > 0:
                self._input_queue.put(pcm, timeout=self._put_timeout_sec)
            else:
                self._input_queue.put_nowait(pcm)
            self._input_pcm_bytes += len(pcm)
            self._input_pcm_chunks += 1
            return True
        except queue.Full:
            wait_ms = int(self._put_timeout_sec * 1000)
            qsize = self._input_queue.qsize()
            self._on_trace(
                "realtime_mp3_bridge_queue_full",
                {
                    "qsize": qsize,
                    "maxsize": self._input_queue_maxsize,
                    "wait_ms": wait_ms,
                },
            )
            self._on_error(
                ProviderError(
                    "实时 PCM 到 MP3 转码队列已满: "
                    f"qsize={qsize}, maxsize={self._input_queue_maxsize}, wait_ms={wait_ms}"
                )
            )
            return False

    def close(
        self, *, report_nonzero: bool = True, drain: bool = True
    ) -> tuple[int | None, str]:
        """关闭 PCM 输入并等待 FFmpeg 刷出最后的 MP3 帧。

        入参含义：`report_nonzero` 控制 FFmpeg 非 0 退出码是否立刻上报为业务错误。
        返回值说明：返回 `(returncode, detail)`，用于上层区分正常收尾警告和真正转码失败。
        使用注意事项：正常 provider 完成路径会自行决定是否把 close-time 异常降级为 warning。
        """

        if self._closed.is_set():
            return self._process.returncode, ""
        self._closed.set()
        detail_parts: list[str] = []
        close_wait_sec = self._normal_close_timeout_sec() if drain else 1.0
        if drain:
            try:
                self._input_queue.put(None, timeout=close_wait_sec)
            except queue.Full:
                detail_parts.append(
                    "bridge close sentinel enqueue timeout: "
                    f"qsize={self._input_queue.qsize()}, maxsize={self._input_queue_maxsize}, "
                    f"input_pcm_bytes={self._input_pcm_bytes}, written_pcm_bytes={self._written_pcm_bytes}"
                )
                self._close_stdin_quietly()
            else:
                writer_deadline = time.monotonic() + close_wait_sec
                while self._writer.is_alive():
                    remaining = writer_deadline - time.monotonic()
                    if remaining <= 0 or self._reader_failed.wait(
                        timeout=min(0.1, remaining)
                    ):
                        break
        else:
            try:
                self._input_queue.put_nowait(None)
            except queue.Full:
                detail_parts.append(
                    "bridge fast close with pending pcm: "
                    f"qsize={self._input_queue.qsize()}, maxsize={self._input_queue_maxsize}"
                )
            self._writer.join(timeout=0.5)
        reader_failed_during_close = self._reader_failed.is_set()
        if reader_failed_during_close:
            detail_parts.append(f"bridge MP3 downlink stopped: {self._reader_error}")
        if self._writer.is_alive():
            detail_parts.append(
                "bridge writer drain timeout: "
                f"qsize={self._input_queue.qsize()}, input_pcm_bytes={self._input_pcm_bytes}, "
                f"written_pcm_bytes={self._written_pcm_bytes}, wait_sec={close_wait_sec:.1f}"
            )
            self._close_stdin_quietly()
        process_wait_sec = (
            close_wait_sec if drain and not reader_failed_during_close else 1.0
        )
        try:
            self._process.wait(timeout=process_wait_sec)
        except subprocess.TimeoutExpired:
            detail_parts.append("ffmpeg process wait timeout")
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                detail_parts.append("ffmpeg process kill after terminate timeout")
                self._process.kill()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    detail_parts.append("ffmpeg process still alive after kill")
        self._reader.join(timeout=process_wait_sec)
        if self._reader.is_alive():
            detail_parts.append("bridge reader drain timeout")
        detail_text = ""
        if self._process.returncode not in (0, None):
            detail = b""
            if self._process.stderr is not None:
                detail = self._process.stderr.read()
            stderr_text = detail.decode("utf-8", errors="replace").strip()
            detail_parts.insert(0, f"ffmpeg returncode={self._process.returncode}")
            if stderr_text:
                detail_parts.append(f"stderr={stderr_text[:300]}")
        if detail_parts:
            detail_text = "; ".join(detail_parts)
            if report_nonzero:
                self._on_error(ProviderError(f"实时 MP3 转码失败: {detail_text}"))
        return self._process.returncode, detail_text

    def _normal_close_timeout_sec(self) -> float:
        """Return a close timeout scaled to the provider PCM duration."""

        audio_sec = self._input_pcm_bytes / self._input_bytes_per_second
        return min(180.0, max(self._close_timeout_sec, audio_sec + 10.0))

    def _close_stdin_quietly(self) -> None:
        """静默关闭 FFmpeg stdin，避免正常收尾阶段泄露 Python 文件析构警告。

        返回值说明：无返回值。
        使用注意事项：仅用于结束转码输入；关闭失败通常表示 FFmpeg 已经先行退出。
        """

        stdin = self._process.stdin
        if stdin is None:
            return
        try:
            stdin.close()
        except (OSError, ValueError):
            pass
        finally:
            self._process.stdin = None

    def _write_loop(self) -> None:
        """把排队的 PCM 安全写入 FFmpeg 标准输入。"""

        try:
            stdin = self._process.stdin
            if stdin is None:
                raise ProviderError("实时音频转码器没有标准输入")
            while True:
                chunk = self._input_queue.get()
                if chunk is None:
                    self._close_stdin_quietly()
                    return
                view = memoryview(chunk)
                written = 0
                while written < len(view):
                    count = stdin.write(view[written:])
                    if count is None or count <= 0:
                        raise OSError("FFmpeg stdin write returned no progress")
                    written += count
                self._written_pcm_bytes += len(chunk)
                self._written_pcm_chunks += 1
        except Exception as exc:  # noqa: BLE001
            if not self._closed.is_set():
                self._on_error(ProviderError(f"实时 PCM 写入转码器失败: {exc}"))

    def _read_loop(self) -> None:
        """持续读取 FFmpeg 标准输出并交给下行 WebSocket 回调。"""

        try:
            if self._process.stdout is None:
                raise ProviderError("实时音频转码器没有标准输出")
            while True:
                chunk = self._process.stdout.read(1024)
                if not chunk:
                    return
                self._forward_ci_compatible_mp3(chunk)
        except Exception as exc:  # noqa: BLE001
            self._reader_error = exc
            self._reader_failed.set()
            if not self._closed.is_set():
                self._on_error(ProviderError(f"实时 MP3 读取或下行失败: {exc}"))

    def _forward_ci_compatible_mp3(self, chunk: bytes) -> None:
        """将转码 MP3 下发给 CI，并只修正首帧的历史兼容字段。

        入参含义：``chunk`` 是 FFmpeg 连续输出的一段原始 MP3 字节。
        返回值说明：无返回值，完整可下发数据通过构造器传入的回调发送。
        使用注意事项：CI 当前简易播放器未确认 ``ID3`` 就读取第 6~9 字节作为
        ID3 长度。实时原始 MP3 的压缩数据常使该长度非零，导致 CI 错跳过几十 KB
        并停止拉取。这里只在首个 MPEG 帧存在时清零该四字节，后续字节保持原样。
        """

        if not self._ci_first_mp3_pending:
            self._on_mp3(chunk)
            return

        # 首个 stdout 读取理论上为 1024 字节；仍兼容极端情况下不足 10 字节的分段输出。
        self._ci_first_mp3_prefix.extend(chunk)
        if len(self._ci_first_mp3_prefix) < 10:
            return

        if (
            self._ci_first_mp3_prefix[0] == 0xFF
            and (self._ci_first_mp3_prefix[1] & 0xE0) == 0xE0
        ):
            self._ci_first_mp3_prefix[6:10] = b"\x00\x00\x00\x00"

        self._ci_first_mp3_pending = False
        output = bytes(self._ci_first_mp3_prefix)
        self._ci_first_mp3_prefix.clear()
        self._on_mp3(output)
