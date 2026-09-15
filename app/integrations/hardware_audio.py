"""CI 语音芯片音频格式适配。

硬件上传的 Speex 不是 Ogg 文件，而是连续的 CI 帧：
``[1 字节帧长][Speex 帧数据]``。本模块先把帧封装成 Ogg Speex，
再交给 FFmpeg 解码为百度 ASR 需要的 16 kHz/16-bit/单声道 PCM。

当前 CI 固件常见配置为 Speex wideband、16 kHz、每帧 20 ms：一帧包含
1 字节长度头和约 42 字节压缩数据。长度头仍按实际值解析，不把 42 写死，
这样可以明确发现截断数据，也给同类 Speex 参数留出兼容空间。

数据转换路径：

``CI 私有帧流 -> Speex 帧列表 -> 内存中的 Ogg Speex -> FFmpeg stdin``
``-> FFmpeg stdout -> 16k/16bit/mono 裸 PCM``

整个过程不创建临时音频文件。同步函数负责 CPU/子进程工作，异步包装使用
``asyncio.to_thread``，避免阻塞 FastAPI 事件循环。
"""

from __future__ import annotations

import asyncio
import audioop
import logging
import math
import struct
import subprocess

import lameenc


class HardwareAudioError(RuntimeError):
    """硬件音频帧或转码失败。"""


logger = logging.getLogger(__name__)


def looks_like_float32_pcm(data: bytes) -> bool:
    """判断一段裸音频是否更像 little-endian float32 PCM。

    Seeduplex 的标准输出是 s16le，但部分资源/兼容层会返回 float32 PCM。
    这里仅用于端到端下行音频的首段探测；正常 s16le 不会被转换。
    """

    usable = len(data) - (len(data) % 4)
    if usable < 512:
        return False

    sample_count = usable // 4
    finite_count = 0
    plausible_count = 0
    active_count = 0
    for offset in range(0, usable, 4):
        value = struct.unpack_from("<f", data, offset)[0]
        if not math.isfinite(value):
            continue
        finite_count += 1
        if abs(value) <= 1.05:
            plausible_count += 1
        if abs(value) > 1e-4:
            active_count += 1

    detected = (
        finite_count / sample_count >= 0.98
        and plausible_count / sample_count >= 0.95
        and active_count / sample_count >= 0.02
    )
    logger.debug(
        "[音频格式] float32 探测 字节=%d 采样数=%d 是否识别=%s",
        len(data),
        sample_count,
        detected,
    )
    return detected


def convert_float32le_to_s16le(data: bytes) -> bytes:
    """把 little-endian float32 PCM 转成单声道 s16le PCM。"""

    usable = len(data) - (len(data) % 4)
    output = bytearray(usable // 2)
    out_offset = 0
    for offset in range(0, usable, 4):
        value = struct.unpack_from("<f", data, offset)[0]
        if not math.isfinite(value):
            value = 0.0
        value = max(-1.0, min(1.0, value))
        sample = int(value * 32767.0)
        struct.pack_into("<h", output, out_offset, sample)
        out_offset += 2
    return bytes(output)


# 标准 SpeexHeader 固定 80 字节。参数与 CI 端一致：
# - rate=16000；
# - mode=1（wideband）；
# - channels=1；
# - frame_size=320 个采样点，即 16000 × 20 ms。
#
# CI 上传时为了节省带宽没有携带标准容器头，所以服务端需要补上此头，FFmpeg
# 才能知道后续压缩帧采用的 Speex 模式和采样参数。
SPEEX_HEADER = (
    b"Speex   1.2.1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x01\x00\x00\x00P\x00\x00\x00\x80>\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00"
    b"\x01\x00\x00\x00\xa0A\x00\x00@\x01\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
)

# Ogg logical bitstream serial number。这里只在内存中生成单一逻辑流，固定值即可；
# 它不是设备编号，也不参与业务追踪。
_OGG_SERIAL = 0x1234

# imageio-ffmpeg 首次查找打包的 FFmpeg 可执行文件后缓存路径，避免每轮
# 对话重复扫描 Python 包和文件系统。
_ffmpeg_path: str | None = None


def parse_ci_speex_frames(data: bytes) -> list[bytes]:
    """解析 ``[1 字节帧长][帧数据]`` 格式的 CI Speex 字节流。

    返回列表中的每一项只包含 Speex 压缩数据，不包含前面的长度字节。
    WebSocket 消息边界与 Speex 帧边界无关，因此调用方必须先把一轮所有二进制
    消息拼接后再调用本函数。
    """

    frames: list[bytes] = []
    # offset 始终指向下一帧的 1 字节长度头。
    offset = 0
    while offset < len(data):
        # Python bytes 索引直接得到 0~255 的整数，正好对应 CI 的 uint8 长度头。
        frame_length = data[offset]
        offset += 1
        if frame_length == 0:
            # 零长帧没有可解码内容，也可能导致封装器生成语义不明确的 Ogg 包。
            raise HardwareAudioError("CI Speex 帧长度不能为 0")
        if offset + frame_length > len(data):
            # 常见原因是固件录音缓冲被截断、audio_bytes 计算错误，或者服务端收到
            # dialogue.end 时最后一帧尚未完整发送。
            raise HardwareAudioError(
                "CI Speex 帧被截断："
                f"声明 {frame_length} 字节，实际剩余 {len(data) - offset} 字节"
            )
        # 切片产生独立 bytes，后续封装不再依赖原始上传缓冲区的游标。
        frames.append(data[offset : offset + frame_length])
        offset += frame_length

    if not frames:
        raise HardwareAudioError("CI Speex 音频为空")
    return frames


def _ogg_crc(data: bytes) -> int:
    """计算 Ogg 页使用的非反射 CRC-32。

    Ogg 使用多项式 ``0x04C11DB7``，与常见 ZIP/zlib 的反射 CRC 表达不同，不能
    直接使用 ``zlib.crc32``。计算时页头 CRC 字段必须先置零。
    """

    crc = 0
    for byte in data:
        # 当前字节进入 CRC 寄存器最高 8 位，再逐 bit 推进多项式。
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04C11DB7 if crc & 0x80000000 else 0)) & 0xFFFFFFFF
    return crc


def _ogg_page(packet: bytes, sequence: int, granule: int, header_type: int) -> bytes:
    """把一个 Speex packet 封装为一张完整 Ogg page。

    ``sequence`` 是页序号；``granule`` 表示截至本页结束已解码的采样点数量；
    ``header_type`` 的 0x02/0x04 分别表示 BOS（流开始）和 EOS（流结束）。

    为了让结构简单且确定，这里每页只放一个 packet。Speex 头和当前 42 字节
    音频帧都远小于 Ogg 页上限，因此这种封装足够硬件语音场景使用。
    """

    # Ogg lacing table 用一串 0~255 的长度描述 packet。每个 255 表示后面还有
    # 数据；最后一个小于 255 的值表示 packet 在本页结束。
    segments: list[int] = []
    remaining = packet
    while remaining:
        segment_length = min(255, len(remaining))
        segments.append(segment_length)
        remaining = remaining[segment_length:]
    # 当 packet 长度恰好是 255 的整数倍时，需要额外的 0 长 segment 标记结束，
    # 否则解码器会认为 packet 延续到下一页。
    if packet and len(packet) % 255 == 0:
        segments.append(0)

    # 页体由 segment table 和真实 packet 数据组成。
    body = bytes(segments) + packet
    # Ogg 页头字段依次为 capture pattern、版本/类型、granule、serial、页序号、
    # CRC 占位、segment 数。所有多字节整数按 Ogg 规范使用小端序。
    header = (
        b"OggS"
        + bytes([0, header_type])
        + struct.pack("<q", granule)
        + struct.pack("<I", _OGG_SERIAL)
        + struct.pack("<I", sequence)
        + struct.pack("<I", 0)
        + bytes([len(segments)])
    )
    # CRC 必须覆盖“CRC 字段为 0 的页头 + 页体”，算出后写回页头 22~25 字节。
    crc = _ogg_crc(header + body)
    return header[:22] + struct.pack("<I", crc) + header[26:] + body


def build_ogg_speex(frames: list[bytes]) -> bytes:
    """把 CI Speex 帧封装成 FFmpeg 可读取的 Ogg Speex。

    第一页放标准 SpeexHeader 并标记 BOS；后续每个 CI 音频帧占一页；最后一个
    音频页标记 EOS。20 ms 帧包含 320 个 16k 采样点，因此第 N 帧的 granule
    位置为 ``N * 320``。
    """

    if not frames:
        raise HardwareAudioError("没有可封装的 Speex 帧")

    # 序号 0 是头页，不代表任何已解码音频采样，所以 granule=0。
    pages = [_ogg_page(SPEEX_HEADER, 0, 0, 0x02)]
    for index, frame in enumerate(frames):
        # 只有最后一帧设置 EOS；中间页 header_type=0。
        header_type = 0x04 if index == len(frames) - 1 else 0x00
        pages.append(
            _ogg_page(
                frame,
                sequence=index + 1,
                granule=(index + 1) * 320,
                header_type=header_type,
            )
        )
    return b"".join(pages)


def _get_ffmpeg_path() -> str:
    """惰性获取 imageio-ffmpeg 随包提供的 FFmpeg 可执行文件路径。"""

    global _ffmpeg_path
    if _ffmpeg_path is None:
        try:
            # 放在函数内导入，使不使用 Speex 的纯 PCM 链路不必加载该依赖。
            import imageio_ffmpeg
        except ImportError as exc:
            raise HardwareAudioError("Speex 解码需要 imageio-ffmpeg 依赖") from exc
        # imageio-ffmpeg 会按当前操作系统/架构返回对应二进制，无需依赖系统 PATH。
        _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    return _ffmpeg_path


def decode_ci_speex_to_pcm(data: bytes) -> bytes:
    """同步解码 CI Speex，返回 16k/16bit/单声道裸 PCM。

    FFmpeg 从 stdin 读取内存中的 Ogg Speex，从 stdout 返回裸 PCM；stderr 仅用于
    失败诊断。该函数会等待子进程完成，因此 WebSocket 路由应调用下面的异步
    包装，而不要直接在事件循环中执行它。
    """

    # 先做严格的 CI 帧边界校验，再构造标准容器；非法输入不会启动 FFmpeg。
    ogg_speex = build_ogg_speex(parse_ci_speex_frames(data))
    result = subprocess.run(
        [
            _get_ffmpeg_path(),
            # 隐藏版本横幅并只输出错误，避免正常解码时产生大量 stderr 日志。
            "-hide_banner",
            "-loglevel",
            "error",
            # pipe:0 表示从当前子进程 stdin 读取上面构造的 Ogg Speex。
            "-i",
            "pipe:0",
            # 输出格式固定为 signed 16-bit little-endian 裸 PCM。
            "-f",
            "s16le",
            # 即使输入头声明一致，也在输出端显式固定 16k 和单声道，确保百度
            # ASR 收到的格式与 Content-Type 声明完全一致。
            "-ar",
            "16000",
            "-ac",
            "1",
            # pipe:1 表示把最终 PCM 写到 stdout，由 result.stdout 返回。
            "pipe:1",
        ],
        input=ogg_speex,
        # 同时捕获 stdout PCM 和 stderr 错误，不在服务器控制台直接输出二进制。
        capture_output=True,
        # 防止损坏输入或 FFmpeg 异常导致硬件连接永久卡住。
        timeout=30,
        # 手动检查返回码，以便抛出统一的 HardwareAudioError。
        check=False,
    )
    if result.returncode != 0:
        # 只带最后 300 字符，既保留关键 FFmpeg 错误，也控制协议错误消息长度。
        detail = result.stderr.decode("utf-8", errors="replace")[-300:]
        raise HardwareAudioError(f"Speex 解码失败：{detail}")
    if not result.stdout:
        raise HardwareAudioError("Speex 解码结果为空")
    return result.stdout


async def decode_ci_speex_to_pcm_async(data: bytes) -> bytes:
    """在线程中解码，避免 FFmpeg 阻塞 FastAPI 事件循环。

    ``subprocess.run`` 是同步阻塞调用。``asyncio.to_thread`` 将等待过程交给默认
    线程池；事件循环仍能继续服务小程序和其他硬件 WebSocket。返回值和异常会
    自动传回当前协程。
    """

    return await asyncio.to_thread(decode_ci_speex_to_pcm, data)


def decode_audio_container_to_pcm(
    data: bytes,
    *,
    sample_rate: int = 24000,
) -> bytes:
    """把意外返回的 OGG/WAV 音频容器转换为 24k/16bit/mono PCM。

    Seeduplex 配置为 ``pcm`` 时正常不会走这里；该兜底用于防止上游配置被
    服务端按默认 OGG/WAV 处理后，客户端误把容器头和压缩数据当裸 PCM 播放。
    """

    if not data:
        raise HardwareAudioError("音频容器为空")
    logger.info(
        "[音频转换] 开始用 FFmpeg 解码容器 字节=%d 采样率=%d",
        len(data),
        sample_rate,
    )
    result = subprocess.run(
        [
            _get_ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-300:]
        raise HardwareAudioError(f"音频容器解码失败：{detail}")
    if not result.stdout:
        raise HardwareAudioError("音频容器解码结果为空")
    logger.info(
        "[音频转换] FFmpeg 解码完成 输出字节=%d",
        len(result.stdout),
    )
    return result.stdout


async def decode_audio_container_to_pcm_async(
    data: bytes,
    *,
    sample_rate: int = 24000,
) -> bytes:
    """在线程中执行 OGG/WAV 到 PCM 的兜底转换。"""

    return await asyncio.to_thread(
        decode_audio_container_to_pcm,
        data,
        sample_rate=sample_rate,
    )


# ==================== 下行 MP3 编码（阿里链路专用） ====================
# 百度 TTS 直接返回 16 kHz MP3，而阿里 Qwen Audio 的语音输出固定是
# 24 kHz/16-bit/单声道裸 PCM。ESP32 固件现有的解码逻辑只认 16 kHz MP3，
# 所以这条链路必须把 PCM 实时编码成 MP3。
#
# 为什么不用本项目已经在用的 FFmpeg：mp3 封装器在管道输出时不支持中途吐
# 数据，实测无论加 -flush_packets 1 / -avioflags direct / -write_xing 0
# 还是 -write_id3v1 0，输出都会全部堆积到 stdin 关闭之后才出现，首包延迟
# 会从"说完话就开始播"退化成"整段回答生成完才出声"。因此改用 lameenc
# （libmp3lame 的绑定），它提供真正的增量 encode/flush 接口。

DEFAULT_MP3_INPUT_RATE = 24000
DEFAULT_MP3_OUTPUT_RATE = 16000
DEFAULT_MP3_BITRATE = 48
DEFAULT_MP3_QUALITY = 2


class StreamingMp3Encoder:
    """把连续写入的裸 PCM 增量编码成 MP3 分片。

    典型用法::

        encoder = StreamingMp3Encoder()
        for pcm in upstream_pcm_chunks:
            for mp3 in encoder.feed(pcm):
                await send_bytes(mp3)
        for mp3 in encoder.finish():
            await send_bytes(mp3)

    没有子进程也没有管道，所以不存在"缓冲攒够才输出"的问题：

    - ``lameenc`` 每凑满一个 MP3 帧就返回一段字节，不足一帧的尾部留在库
      内部，等下一次 ``feed`` 或 ``finish`` 再交付，首包延迟只剩编码器本身
      约一帧的固有延迟（16 kHz 下约 72 ms）；
    - 24k 降到 16k 用标准库 ``audioop.ratecv``，它通过 ``state`` 参数把上一
      个分片的尾巴带到下一个分片，因此可以逐片处理而不会在分片边界上产生
      不连续（否则听感上会是周期性杂音）。

    ``feed`` / ``finish`` 是同步方法：单片 50 ms 音频的 MP3 编码耗时在亚毫秒
    量级，不值得为它承担线程池切换开销。若将来要把整段长音频一次性塞进来，
    调用方应改用 ``asyncio.to_thread`` 包裹。
    """

    def __init__(
        self,
        *,
        input_rate: int = DEFAULT_MP3_INPUT_RATE,
        output_rate: int = DEFAULT_MP3_OUTPUT_RATE,
        bitrate: int = DEFAULT_MP3_BITRATE,
        quality: int = DEFAULT_MP3_QUALITY,
    ) -> None:
        self._input_rate = input_rate
        self._output_rate = output_rate
        # 用 CBR 而不是 VBR：固件播放器按固定码率推算缓冲更稳，字节流长度
        # 也更好预期。
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate)
        encoder.set_in_sample_rate(output_rate)
        encoder.set_channels(1)
        encoder.set_quality(quality)
        self._encoder = encoder
        # audioop.ratecv 的跨分片状态，必须逐片传递，不能每片都重置。
        self._resample_state: object = None
        self._finished = False
        # 供路由侧统计和排查：累计编码出的 MP3 分片数与字节数。
        self.encoded_bytes = 0
        self.chunk_count = 0

    def feed(self, pcm: bytes) -> list[bytes]:
        """写入一片 24k PCM，返回本次已编码完成的 MP3 分片。

        返回空列表是正常情况：说明还没凑满一个 MP3 帧，数据留在编码器内部，
        会在后续 ``feed`` 或 ``finish`` 中交付，不会丢失。
        """

        if self._finished or not pcm:
            return []
        payload = self._resample(pcm)
        if not payload:
            return []
        data = self._encoder.encode(payload)
        if not data:
            return []
        self.encoded_bytes += len(data)
        self.chunk_count += 1
        return [data]

    def finish(self) -> list[bytes]:
        """冲刷编码器内部残留，返回最后一段 MP3。

        必须调用，否则最后一帧（通常不足一个完整帧）会被丢掉，回答末尾的
        几个字就没有声音。
        """

        if self._finished:
            return []
        self._finished = True
        data = self._encoder.flush()
        if not data:
            return []
        self.encoded_bytes += len(data)
        self.chunk_count += 1
        return [data]

    def _resample(self, pcm: bytes) -> bytes:
        """24k 降到 16k；两侧采样率相同时直接透传。"""

        if self._input_rate == self._output_rate:
            return pcm
        # 参数依次是：数据、样本宽度（2 字节 = 16 bit）、声道数、输入采样率、
        # 输出采样率、上一次调用留下的状态。返回 (转换后数据, 新状态)。
        resampled, self._resample_state = audioop.ratecv(
            pcm,
            2,
            1,
            self._input_rate,
            self._output_rate,
            self._resample_state,
        )
        return resampled

