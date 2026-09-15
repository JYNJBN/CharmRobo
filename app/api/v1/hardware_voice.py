"""ESP32/CI 语音硬件 WebSocket 接口。

本文件是“硬件协议适配层”，只服务 ESP32/CI 设备；原小程序使用的
``/api/v1/voice/stream`` 仍在 ``voice.py`` 中，两条链路彼此独立。

硬件连接地址：``/v1/dialogue/ws``。

一轮对话的上行时序（ESP32 -> 服务端）：

1. 文本帧 ``dialogue.start``，声明设备 SN、音频格式、采样率和字节数；
2. 一个或多个二进制帧，内容是裸 PCM 或 CI 私有格式的 Speex；
3. 文本帧 ``dialogue.end``，表示本轮录音上传完毕。

一轮对话的下行时序（服务端 -> ESP32）：

1. 收到并接受 ``dialogue.start`` 后返回 ``dialogue.ready``；
2. ASR、LLM、TTS 开始工作，百度 TTS 每返回一个 MP3 分片就立即发送一个
   WebSocket 二进制帧；
3. 全部音频发完后返回 ``dialogue.done``；任何阶段失败则返回
   ``dialogue.error``。

WebSocket 是可复用长连接：一次连接可以依次完成多轮 start/end 对话。
当前 ASR 不是流式识别——服务器必须先收完整轮音频；LLM 和 TTS 是流式的，
首个稳定短句生成后即可开始下发 MP3。

FastAPI/ASGI 已负责 WebSocket HTTP Upgrade、掩码、Ping/Pong 和底层帧解析，
本文件只处理应用层 JSON 指令、音频二进制和业务状态机。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter

from fastapi import APIRouter, WebSocket
from sqlalchemy.exc import SQLAlchemyError
from starlette.websockets import WebSocketDisconnect

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.errors import BizError
from app.integrations.baidu_speech import BaiduSpeechError, baidu_speech_client
from app.integrations.hardware_audio import (
    HardwareAudioError,
    decode_ci_speex_to_pcm_async,
)
from app.schemas.Model import ModelRegistryObject
from app.services.agent_service import ensure_default_agent_on_device
from app.services.conversation import run_turn
from app.services.conversation_memory_service import (
    summarize_conversation_in_background,
)
from app.services.conversation_service import (
    add_conversation_message,
    get_or_create_conversation,
    get_recent_messages,
)
from app.services.device_service import get_active_owner_binding, get_device_by_sn
from app.services.hardware_actions import hardware_action_service
from app.services.model_service import describe_model, resolve_model
from app.services.voice_service import resolve_baidu_tts_per
from app.utils.tools import local_now

router = APIRouter(prefix="/v1", tags=["硬件语音"])
logger = logging.getLogger(__name__)

# LLM 通常按很小的 delta（一个词或几个字）返回，不能每个 delta 都请求一次
# TTS，否则连接建立开销和音频碎片会非常多。这里通过标点和长度聚合成短句：
# - 强标点：句意完整，尽快切句；
# - 弱标点：至少积累 8 个字符再切，避免片段过短；
# - 32 字上限：模型长时间不输出标点时也能及时开始播报。
STRONG_PUNCTUATION = "。！？!?"
WEAK_PUNCTUATION = "，,；;、\n"
MIN_WEAK_SEGMENT_CHARS = 8
MAX_SEGMENT_CHARS = 32

# 百度短语音识别标准版单次最多处理 60 秒音频。
# 16 kHz × 16 bit（2 字节）× 单声道 × 60 秒 = 1,920,000 字节。
# 此限制检查的是 Speex 解码后的 PCM 大小，而不是压缩后的上传大小。
MAX_ASR_PCM_BYTES = 60 * 16000 * 2

# 将发送函数声明为类型别名，使流水线只关心“发送 JSON/字节”，不直接依赖
# FastAPI WebSocket 对象，也让内部 TTS worker 可以复用同一个发送出口。
SendJson = Callable[[dict[str, object]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]


class HardwareVoiceError(RuntimeError):
    """硬件协议、设备状态或语音流水线错误。"""


@dataclass
class HardwareSession:
    """一轮硬件对话所需的完整业务上下文。

    这些数据从 device、user_device、agent、conversation 等表中加载。将它们
    固定成一个对象传入流水线，可以保证同一轮 ASR、LLM、落库使用的是同一套
    设备/用户/智能体配置。
    """

    # 对外设备编号，同时作为百度 ASR 的 cuid，便于百度侧区分终端。
    device_sn: str
    # 数据库中的设备、拥有者和当前智能体主键。
    device_id: int
    user_id: int
    agent_id: int
    # 智能体名称和人设提示词，最终注入现有 run_turn。
    agent_name: str
    agent_system_prompt: str | None
    # Agent 保存的是中立 voice key，硬件线路在服务端映射成百度 per。
    voice: str | None
    # 当前智能体对应的会话及其最近若干条短期上下文。
    conversation_id: int
    history_messages: list[dict[str, str]]
    # resolve_model 解析后的供应商、模型 ID、API 地址和凭据。
    model: ModelRegistryObject
    # 给系统提示词和日志使用的人类可读模型名。
    model_label: str


@dataclass
class HardwareTurnResult:
    """一轮成功完成后的统计结果，用于构造 ``dialogue.done``。"""

    asr_text: str
    answer_text: str
    # 下行 MP3 的累计字节数和 WebSocket 二进制帧数量。
    audio_bytes: int
    audio_chunks: int
    # 从收到 dialogue.start 到发送第一段 MP3 的耗时。
    first_chunk_ms: int
    # chat/music/weather 等动作名称，供硬件调整播放策略和日志展示。
    action: str = "chat"


class _HardwareSentenceSplitter:
    """面向语音首包延迟的增量分句器。

    ``feed`` 可以被重复调用；每次把 LLM 新增文本追加到内部缓冲区，并返回
    当前已经稳定、可以交给 TTS 的短句。尚不满足切分条件的尾部继续留在
    ``_buffer``，等下一个 delta。LLM 流结束时必须调用一次 ``flush``，否则
    没有结束标点的最后一段回答会丢失。
    """

    def __init__(self) -> None:
        # 只保存“还没有交给 TTS”的文本；已返回的文本会立即从缓冲区移除。
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        """追加一段 LLM 增量文字，返回零个或多个可合成短句。"""

        self._buffer += delta
        segments: list[str] = []
        while True:
            # 一次 delta 里可能包含多句话，所以循环切分直到没有可切位置。
            cut_at = self._find_cut_index()
            if cut_at <= 0:
                break
            segment = self._buffer[:cut_at].strip()
            self._buffer = self._buffer[cut_at:]
            if segment:
                segments.append(segment)
        return segments

    def flush(self) -> list[str]:
        """LLM 结束后强制取出没有标点的最后一段文字。"""

        remainder = self._buffer.strip()
        self._buffer = ""
        return [remainder] if remainder else []

    def _find_cut_index(self) -> int:
        """按强标点、弱标点、最大长度的优先顺序寻找切分点。"""

        for index, character in enumerate(self._buffer):
            # 至少已有两个字符才按强标点切，避免异常的单个标点调用 TTS。
            if character in STRONG_PUNCTUATION and index >= 1:
                return index + 1
            if character in WEAK_PUNCTUATION and index + 1 >= MIN_WEAK_SEGMENT_CHARS:
                return index + 1
        # 兜底上限保证无标点回答也能在 32 字后开始合成，控制首包延迟。
        if len(self._buffer) >= MAX_SEGMENT_CHARS:
            return MAX_SEGMENT_CHARS
        return 0


async def _load_hardware_session(
    device_sn: str,
    *,
    trace_id: str = "unknown",
) -> HardwareSession:
    """校验硬件并加载当前智能体、模型和对话历史。

    每轮 ``dialogue.start`` 都重新调用，而不是只在 WebSocket 建连时调用。
    这样用户在长连接存活期间切换智能体或模型，下一轮即可生效，不必让设备
    断线重连。

    校验顺序：设备存在 -> 设备启用 -> 已绑定拥有者 -> 有可用智能体。
    任一步失败都会转成 ``dialogue.error``，不会调用收费的语音/模型接口。
    """

    started_at = perf_counter()
    logger.info(
        "[HARDWARE-VOICE][%s][会话] 开始加载 device_sn=%s",
        trace_id,
        device_sn,
    )
    async with AsyncSessionLocal() as db:
        device_started_at = perf_counter()
        # device_sn 是硬件和数据库设备记录之间的唯一关联键。
        device = await get_device_by_sn(db, device_sn)
        logger.info(
            "[HARDWARE-VOICE][%s][会话] 设备查询完成 存在=%s 耗时=%.3f秒",
            trace_id,
            device is not None,
            perf_counter() - device_started_at,
        )
        if device is None:
            raise HardwareVoiceError("设备不存在或已删除")
        if device.status != 1:
            raise HardwareVoiceError("设备已被禁用")

        # 硬件没有用户 JWT，因此通过设备绑定表反查当前 owner 用户。
        owner_started_at = perf_counter()
        owner_binding = await get_active_owner_binding(db, device_id=device.id)
        logger.info(
            "[HARDWARE-VOICE][%s][会话] 设备归属查询完成 已绑定=%s "
            "耗时=%.3f秒",
            trace_id,
            owner_binding is not None,
            perf_counter() - owner_started_at,
        )
        if owner_binding is None:
            raise HardwareVoiceError("设备尚未绑定用户")

        # 老设备可能尚未物化默认智能体；该服务会补齐默认副本并返回当前智能体。
        agent_started_at = perf_counter()
        agent = await ensure_default_agent_on_device(db=db, device=device)
        if agent is None:
            raise HardwareVoiceError("设备没有可用智能体")

        # 收到合法的硬件对话请求即认为设备在线。ensure_default_agent_on_device
        # 内部只 flush，所以在这里和 last_online_time 一起统一提交。
        device.last_online_time = local_now()
        await db.commit()
        logger.info(
            "[HARDWARE-VOICE][%s][会话] Agent/设备状态准备完成 agent_id=%s "
            "耗时=%.3f秒",
            trace_id,
            agent.id,
            perf_counter() - agent_started_at,
        )

        # 将数据库里的 model_key 解析成现有 LLM 层可直接使用的模型配置。
        model = resolve_model(agent.model_key)
        model_label = describe_model(agent.model_key)
        if model["provider"] == "ark":
            # 硬件路线固定使用旧 server 验证过的低延迟豆包模型，并明确关闭思考。
            # 这里复制字典而不是修改 MODEL_REGISTRY，避免影响小程序/其他调用方。
            model = {
                **model,
                "model_id": settings.hardware_doubao_model,
                "thinking_type": "disabled",
            }
            logger.info(
                "[HARDWARE-VOICE][%s][会话] 硬件模型已固定 "
                "model=%s thinking=disabled",
                trace_id,
                settings.hardware_doubao_model,
            )

        # 会话按当前智能体复用。切换智能体后会得到另一条 conversation，历史不会
        # 混到旧智能体中。
        conversation_started_at = perf_counter()
        conversation = await get_or_create_conversation(
            db,
            user_id=owner_binding.user_id,
            device_id=device.id,
            agent_id=agent.id,
        )
        logger.info(
            "[HARDWARE-VOICE][%s][会话] conversation 准备完成 "
            "conversation_id=%s 耗时=%.3f秒",
            trace_id,
            conversation.id,
            perf_counter() - conversation_started_at,
        )
        # 只取配置数量的最近消息，避免提示词随对话无限增长。
        history_started_at = perf_counter()
        history_messages = await get_recent_messages(
            db,
            conversation_id=conversation.id,
            limit=settings.short_term_context_messages,
        )
        logger.info(
            "[HARDWARE-VOICE][%s][会话] 短期记忆读取完成 "
            "conversation_id=%s 条数=%d 耗时=%.3f秒 总耗时=%.3f秒",
            trace_id,
            conversation.id,
            len(history_messages),
            perf_counter() - history_started_at,
            perf_counter() - started_at,
        )

        return HardwareSession(
            device_sn=device.device_sn,
            device_id=device.id,
            user_id=owner_binding.user_id,
            agent_id=agent.id,
            agent_name=agent.name,
            agent_system_prompt=agent.system_prompt,
            voice=agent.voice,
            conversation_id=conversation.id,
            history_messages=history_messages,
            model=model,
            model_label=model_label,
        )


async def _save_message(
    session: HardwareSession,
    *,
    role: str,
    content: str,
) -> None:
    """使用独立数据库会话保存一条硬件语音对话消息。

    ASR 用户文本和 LLM 完整回答分别提交，既保留与小程序相同的会话记录，
    也避免长期持有路由加载阶段的数据库 Session。
    """

    async with AsyncSessionLocal() as db:
        await add_conversation_message(
            db,
            conversation_id=session.conversation_id,
            role=role,
            content=content,
            source="voice",
        )


async def _run_hardware_turn(
    *,
    session: HardwareSession,
    audio_data: bytes,
    audio_format: str,
    request_id: str,
    turn_started_at: float,
    send_bytes: SendBytes,
    send_json: SendJson,
    cancel_event: asyncio.Event,
) -> HardwareTurnResult:
    """执行一轮 ``音频 -> ASR -> LLM -> TTS -> MP3`` 流水线。

    上行音频必须已经完整接收。函数只通过 ``send_bytes`` 下发 MP3，不发送
    ``dialogue.done``；完成消息由外层协议处理函数在所有音频发送成功后统一发出。

    LLM 与 TTS 通过异步队列并行：LLM 继续生成后续文字的同时，TTS worker
    合成已经切好的前一句，从而缩短第一段声音到达硬件的时间。
    """

    def check_cancelled() -> None:
        if cancel_event.is_set():
            raise asyncio.CancelledError()

    processing_started_at = perf_counter()
    logger.info(
        "[HARDWARE-VOICE][%s][时序] 语音处理开始 format=%s "
        "上传字节=%d 距 dialogue.start=%.3f秒",
        request_id,
        audio_format,
        len(audio_data),
        processing_started_at - turn_started_at,
    )

    # ==================== 阶段 1：统一为百度 ASR 所需 PCM ====================
    check_cancelled()
    if audio_format == "speex":
        # CI Speex 解码会启动 FFmpeg 子进程；async 包装将阻塞工作放在线程中，
        # 防止一个设备解码时卡住 FastAPI 事件循环中的其他连接。
        decode_started_at = perf_counter()
        pcm = await decode_ci_speex_to_pcm_async(audio_data)
        logger.info(
            "[HARDWARE-VOICE][%s] Speex 解码完成 elapsed=%.3fs encoded=%d pcm=%d",
            request_id,
            perf_counter() - decode_started_at,
            len(audio_data),
            len(pcm),
        )
    else:
        # PCM 模式已经是 16k/16bit/单声道裸数据，不需要再次转码。
        pcm = audio_data
        logger.info(
            "[HARDWARE-VOICE][%s][音频] PCM 无需解码 字节=%d "
            "耗时=%.3f秒",
            request_id,
            len(pcm),
            perf_counter() - processing_started_at,
        )

    # 百度短语音标准版限制 60 秒。Speex 很小，因此必须在解码后用 PCM 大小
    # 判断真实时长，不能只看网络上传的压缩字节数。
    if len(pcm) > MAX_ASR_PCM_BYTES:
        raise HardwareVoiceError("单轮录音不能超过 60 秒")

    # ==================== 阶段 2：百度 ASR ====================
    # ASR 是完整音频请求：等识别结果返回后才能进入 LLM，不会返回 partial 文本。
    asr_started_at = perf_counter()
    asr_text = await baidu_speech_client.recognize_pcm(
        pcm,
        cuid=session.device_sn,
    )
    logger.info(
        "[HARDWARE-VOICE][%s][百度ASR] 完成 elapsed=%.3fs "
        "距处理开始=%.3f秒 text=%s",
        request_id,
        perf_counter() - asr_started_at,
        perf_counter() - processing_started_at,
        asr_text,
    )
    check_cancelled()

    # ==================== 阶段 3：动作路由与统一音频输出 ====================
    action_started_at = perf_counter()
    logger.info("[HARDWARE-VOICE][%s][动作] 开始判断", request_id)
    action_result = await hardware_action_service.resolve(asr_text, session.model)
    logger.info(
        "[HARDWARE-VOICE][%s][动作] 判断完成 action=%s "
        "耗时=%.3f秒",
        request_id,
        action_result.action if action_result is not None else "chat",
        perf_counter() - action_started_at,
    )
    check_cancelled()
    # 当前 Agent 的音色在每轮 start 时重新读取；切换 Agent 后下一轮立即生效。
    baidu_per = resolve_baidu_tts_per(session.voice)

    audio_bytes = 0
    audio_chunks = 0
    first_chunk_ms: int | None = None

    async def send_audio_chunk(chunk: bytes) -> None:
        nonlocal audio_bytes, audio_chunks, first_chunk_ms
        check_cancelled()
        if not chunk:
            return
        if first_chunk_ms is None:
            first_chunk_ms = int((perf_counter() - turn_started_at) * 1000)
            logger.info(
                "[HARDWARE-VOICE][%s][TTS] 首个下行音频块 "
                "距 dialogue.start=%.3f秒 字节=%d",
                request_id,
                perf_counter() - turn_started_at,
                len(chunk),
            )
        await send_bytes(chunk)
        audio_bytes += len(chunk)
        audio_chunks += 1

    async def stream_tts_text(text: str) -> None:
        if not text.strip():
            return
        tts_started_at = perf_counter()
        logger.info(
            "[HARDWARE-VOICE][%s][百度TTS] 开始 per=%s 文本字数=%d",
            request_id,
            baidu_per,
            len(text),
        )
        async for chunk in baidu_speech_client.stream_synthesize_mp3(
            text,
            per=baidu_per,
        ):
            await send_audio_chunk(chunk)
        logger.info(
            "[HARDWARE-VOICE][%s][百度TTS] 完成 文本字数=%d "
            "耗时=%.3f秒",
            request_id,
            len(text),
            perf_counter() - tts_started_at,
        )

    if action_result is not None:
        # action 帧在第一段音频前发送，让固件知道本轮是音乐、天气还是兜底播报。
        if action_result.action_payload:
            await send_json(
                {
                    "type": "dialogue.action",
                    "req": request_id,
                    **action_result.action_payload,
                }
            )
        if action_result.track is not None:
            music_started_at = perf_counter()
            logger.info(
                "[HARDWARE-VOICE][%s][音乐] 开始发送歌曲 title=%s "
                "文件字节=%d",
                request_id,
                action_result.track.title,
                action_result.track.size,
            )
            async for chunk in hardware_action_service.music.iter_track_bytes(
                action_result.track,
                cancel_event=cancel_event,
            ):
                await send_audio_chunk(chunk)
            logger.info(
                "[HARDWARE-VOICE][%s][音乐] 文件发送完成 title=%s "
                "耗时=%.3f秒",
                request_id,
                action_result.track.title,
                perf_counter() - music_started_at,
            )
            answer_text = action_result.answer_text or action_result.track.title
        else:
            await stream_tts_text(action_result.answer_text)
            answer_text = action_result.answer_text
        if not answer_text or audio_bytes == 0:
            raise HardwareVoiceError("动作没有返回有效音频")
        await _save_message(session, role="user", content=asr_text)
        await _save_message(session, role="assistant", content=answer_text)
        logger.info(
            "[HARDWARE-VOICE][%s][时序] 动作分支完成 action=%s "
            "总耗时=%.3f秒 音频字节=%d 分片=%d",
            request_id,
            action_result.action,
            perf_counter() - turn_started_at,
            audio_bytes,
            audio_chunks,
        )
        return HardwareTurnResult(
            asr_text=asr_text,
            answer_text=answer_text,
            audio_bytes=audio_bytes,
            audio_chunks=audio_chunks,
            first_chunk_ms=first_chunk_ms
            or int((perf_counter() - turn_started_at) * 1000),
            action=action_result.action,
        )

    # ==================== 阶段 4：普通闲聊 LLM/TTS 并行流水线 ====================
    # 普通闲聊才进入现有 Agent/LLM/记忆链路。
    await _save_message(session, role="user", content=asr_text)
    llm_pipeline_started_at = perf_counter()
    first_llm_delta_at: float | None = None
    logger.info(
        "[HARDWARE-VOICE][%s][LLM] 阶段开始（包含前置记忆检索） "
        "model=%s 短期消息数=%d",
        request_id,
        session.model_label,
        len(session.history_messages),
    )
    # 队列中是完整短句；None 是结束哨兵，表示 LLM 不会再产生新句子。
    tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def tts_worker() -> None:
        """按顺序消费短句，将百度返回的 MP3 分片立即转发给 ESP32。"""

        nonlocal audio_bytes, audio_chunks, first_chunk_ms
        while True:
            # 没有句子时挂起等待，不占用 CPU；LLM 切出句子后 put 会唤醒这里。
            sentence = await tts_queue.get()
            try:
                if sentence is None:
                    return
                # 每个 chunk 都是 MP3 字节流的一部分，不做 base64、不缓存完整文件。
                sentence_started_at = perf_counter()
                logger.info(
                    "[HARDWARE-VOICE][%s][百度TTS] 句子开始 文本字数=%d",
                    request_id,
                    len(sentence),
                )
                async for chunk in baidu_speech_client.stream_synthesize_mp3(
                    sentence,
                    per=baidu_per,
                ):
                    await send_audio_chunk(chunk)
                logger.info(
                    "[HARDWARE-VOICE][%s][百度TTS] 句子完成 文本字数=%d "
                    "耗时=%.3f秒",
                    request_id,
                    len(sentence),
                    perf_counter() - sentence_started_at,
                )
            finally:
                # Queue 的每个 get 都必须对应 task_done，包括 None 和失败情况。
                tts_queue.task_done()

    # worker 独立运行，因此下面遍历 LLM 流时不会等待当前句 TTS 完整结束。
    worker_task = asyncio.create_task(tts_worker())
    splitter = _HardwareSentenceSplitter()
    reply_parts: list[str] = []
    try:
        # run_turn 是小程序和硬件共用的文字对话核心：它负责短期历史、长期记忆、
        # 智能体人设和具体模型供应商，逐段 yield LLM 新增文字。
        async for delta in run_turn(
            asr_text,
            session.history_messages,
            session.conversation_id,
            user_id=session.user_id,
            device_id=session.device_id,
            trace_id=request_id,
            agent_id=session.agent_id,
            agent_name=session.agent_name,
            agent_system_prompt=session.agent_system_prompt,
            model=session.model,
            model_label=session.model_label,
        ):
            check_cancelled()
            if first_llm_delta_at is None:
                first_llm_delta_at = perf_counter()
                logger.info(
                    "[HARDWARE-VOICE][%s][LLM] 首个文字增量 "
                    "距LLM阶段开始=%.3f秒 距dialogue.start=%.3f秒",
                    request_id,
                    first_llm_delta_at - llm_pipeline_started_at,
                    first_llm_delta_at - turn_started_at,
                )
            # 一份用于最终落库，一份交给分句器驱动实时 TTS。
            reply_parts.append(delta)
            for sentence in splitter.feed(delta):
                await tts_queue.put(sentence)
        # LLM 结束后处理没有标点的尾句，再发送 None 让 worker 正常退出。
        for sentence in splitter.flush():
            await tts_queue.put(sentence)
        llm_stream_finished_at = perf_counter()
        logger.info(
            "[HARDWARE-VOICE][%s][LLM] 上游文字流结束 回复字数=%d "
            "阶段耗时=%.3f秒 首字耗时=%.3f秒",
            request_id,
            sum(len(part) for part in reply_parts),
            llm_stream_finished_at - llm_pipeline_started_at,
            first_llm_delta_at - llm_pipeline_started_at
            if first_llm_delta_at
            else 0.0,
        )
        await tts_queue.put(None)
        # 必须等 TTS worker 发完所有 MP3，外层才能安全发送 dialogue.done。
        await worker_task
        logger.info(
            "[HARDWARE-VOICE][%s][百度TTS] 所有句子发送完成 "
            "等待TTS耗时=%.3f秒 总阶段耗时=%.3f秒",
            request_id,
            perf_counter() - llm_stream_finished_at,
            perf_counter() - llm_pipeline_started_at,
        )
    except Exception:
        # LLM、TTS 或 WebSocket 发送任一失败，都取消后台 worker，避免任务泄漏、
        # 后续继续向已经失败/关闭的连接发送音频。
        worker_task.cancel()
        await asyncio.gather(worker_task, return_exceptions=True)
        raise

    # ==================== 阶段 4：结果校验、落库和摘要 ====================
    answer_text = "".join(reply_parts).strip()
    if not answer_text:
        raise HardwareVoiceError("大模型没有返回回答")
    if audio_bytes == 0:
        raise HardwareVoiceError("百度 TTS 没有返回 MP3")

    # 只有 LLM 和全部 TTS 都成功后才保存助手回答，避免成功记录与硬件实际播放
    # 状态严重不一致。
    await _save_message(session, role="assistant", content=answer_text)
    # 摘要/向量记忆是后台任务，不阻塞 dialogue.done 和下一轮录音。
    asyncio.create_task(
        summarize_conversation_in_background(
            conversation_id=session.conversation_id,
            device_id=session.device_id,
            agent_id=session.agent_id,
            user_id=session.user_id,
            trace_id=request_id,
        )
    )
    logger.info(
        "[HARDWARE-VOICE][%s][记忆] 已提交摘要/向量后台任务 "
        "当前轮不等待后台写入",
        request_id,
    )
    logger.info(
        "[HARDWARE-VOICE][%s][时序] 语音处理完成 "
        "总耗时=%.3f秒 音频字节=%d 分片=%d",
        request_id,
        perf_counter() - turn_started_at,
        audio_bytes,
        audio_chunks,
    )

    return HardwareTurnResult(
        asr_text=asr_text,
        answer_text=answer_text,
        audio_bytes=audio_bytes,
        audio_chunks=audio_chunks,
        first_chunk_ms=first_chunk_ms or int((perf_counter() - turn_started_at) * 1000),
    )


@router.websocket("/dialogue/ws")
async def hardware_dialogue_websocket(client: WebSocket) -> None:
    """处理 ESP32 的可复用语音对话 WebSocket 长连接。

    应用层状态机只有两个主要状态：

    - 空闲（``collecting=False``）：等待 ``dialogue.start``；
    - 收音（``collecting=True``）：累积二进制音频，等待 ``dialogue.end``。

    一轮处理完成或失败后连接不会主动关闭，ESP32 可以发送下一轮 start。
    ``device_sn`` 优先取 start JSON，也兼容 URL 查询参数 ``device_sn`` 和请求头
    ``X-Device-SN``，但推荐固件明确放在每轮 start 中。
    """

    # WebSocket 握手由 FastAPI 完成；accept 后才能收发应用数据。
    connection_started_at = perf_counter()
    await client.accept()
    connection_id = uuid.uuid4().hex[:8]
    logger.info(
        "[HARDWARE-VOICE][%s][连接] WebSocket 已建立 "
        "wss握手耗时=%.3f秒",
        connection_id,
        perf_counter() - connection_started_at,
    )

    # LLM 主协程和 TTS worker 可能并发发送；Starlette 不保证同一 WebSocket 的
    # 并发 send 安全，所以所有文本/二进制发送都必须经过同一把锁。
    send_lock = asyncio.Lock()

    async def send_json(data: dict[str, object]) -> None:
        """串行发送协议控制帧。"""

        async with send_lock:
            await client.send_json(data)

    async def send_bytes(data: bytes) -> None:
        """串行发送百度 TTS 返回的 MP3 二进制帧。"""

        async with send_lock:
            await client.send_bytes(data)

    # ==================== 连接级/当前轮状态 ====================
    # collecting 决定收到的二进制帧是否属于一个合法录音轮次。
    collecting = False
    # WebSocket 可能把一段录音拆成任意数量的二进制消息，先按片保存，end 时拼接。
    audio_parts: list[bytes] = []
    received_bytes = 0
    received_frame_count = 0
    first_audio_frame_at: float | None = None
    # 客户端可不声明大小；声明后在 end 阶段精确校验是否丢包/固件计数错误。
    expected_audio_bytes: int | None = None
    # 为兼容旧固件，未声明格式时按 PCM；新 CI 压缩固件应显式发 speex。
    audio_format = "pcm"
    # request_id 串联该轮服务日志和返回帧；固件没提供时由服务器生成。
    request_id = uuid.uuid4().hex[:8]
    turn_started_at = perf_counter()
    session: HardwareSession | None = None
    # device_sn 的连接级兜底来源。start JSON 中的值始终优先。
    connection_device_sn = (
        client.query_params.get("device_sn") or client.headers.get("x-device-sn") or ""
    ).strip()
    connection_device_ip = client.headers.get("x-device-ip", "").strip()
    client_req: object = None
    turn_task: asyncio.Task[None] | None = None
    turn_cancel_event: asyncio.Event | None = None

    async def run_turn_worker(
        *,
        worker_request_id: str,
        worker_client_req: object,
        worker_cancel_event: asyncio.Event,
        worker_session: HardwareSession,
        worker_audio_data: bytes,
        worker_audio_format: str,
        worker_started_at: float,
    ) -> None:
        """后台执行一轮软件流水线，让主 WS 协程继续接收 cancel。"""

        nonlocal turn_task, turn_cancel_event
        worker_started_processing_at = perf_counter()
        logger.info(
            "[HARDWARE-VOICE][%s][时序] 后台处理任务开始 format=%s "
            "音频字节=%d 距 dialogue.start=%.3f秒",
            worker_request_id,
            worker_audio_format,
            len(worker_audio_data),
            worker_started_processing_at - worker_started_at,
        )
        try:
            result = await _run_hardware_turn(
                session=worker_session,
                audio_data=worker_audio_data,
                audio_format=worker_audio_format,
                request_id=worker_request_id,
                turn_started_at=worker_started_at,
                send_bytes=send_bytes,
                send_json=send_json,
                cancel_event=worker_cancel_event,
            )
            if worker_cancel_event.is_set():
                return
            total_ms = int((perf_counter() - worker_started_at) * 1000)
            done_started_at = perf_counter()
            await send_json(
                {
                    "type": "dialogue.done",
                    "req": worker_request_id,
                    "client_req": worker_client_req,
                    "audio_bytes": result.audio_bytes,
                    "audio_chunks": result.audio_chunks,
                    "first_chunk_ms": result.first_chunk_ms,
                    "total_ms": total_ms,
                    "action": result.action,
                }
            )
            logger.info(
                "[HARDWARE-VOICE][%s] 本轮完成 action=%s total_ms=%d audio_bytes=%d",
                worker_request_id,
                result.action,
                total_ms,
                result.audio_bytes,
            )
            logger.info(
                "[HARDWARE-VOICE][%s][时序] dialogue.done 已发送 "
                "发送耗时=%.3f秒 总耗时=%.3f秒",
                worker_request_id,
                perf_counter() - done_started_at,
                perf_counter() - worker_started_at,
            )
        except asyncio.CancelledError:
            logger.info(
                "[HARDWARE-VOICE][%s][时序] 本轮被取消 "
                "总耗时=%.3f秒",
                worker_request_id,
                perf_counter() - worker_started_at,
            )
            return
        except (HardwareVoiceError, HardwareAudioError, BaiduSpeechError) as exc:
            logger.warning(
                "[HARDWARE-VOICE][%s][时序] 本轮业务失败 "
                "总耗时=%.3f秒 error=%s",
                worker_request_id,
                perf_counter() - worker_started_at,
                exc,
            )
            if not worker_cancel_event.is_set():
                await send_json(
                    {
                        "type": "dialogue.error",
                        "req": worker_request_id,
                        "client_req": worker_client_req,
                        "error": str(exc),
                    }
                )
        except Exception as exc:
            logger.exception(
                "[HARDWARE-VOICE][%s][时序] 未预期异常 "
                "总耗时=%.3f秒",
                worker_request_id,
                perf_counter() - worker_started_at,
            )
            if not worker_cancel_event.is_set():
                await send_json(
                    {
                        "type": "dialogue.error",
                        "req": worker_request_id,
                        "client_req": worker_client_req,
                        "error": f"server error: {exc}",
                    }
                )
        finally:
            if turn_task is asyncio.current_task():
                turn_task = None
                turn_cancel_event = None

    logger.info("[HARDWARE-VOICE][%s] WebSocket 已连接", request_id)
    try:
        # 一个 while 循环覆盖整个长连接；每轮结束后自然回到这里等待下一条消息。
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                return

            # ==================== 分支 1：硬件音频二进制帧 ====================
            binary_data = message.get("bytes")
            if binary_data is not None:
                # 防止未发 start 就上传音频导致音频归属不明。
                if not collecting:
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "binary frame before dialogue.start",
                        }
                    )
                    continue
                received_bytes += len(binary_data)
                received_frame_count += 1
                if first_audio_frame_at is None:
                    first_audio_frame_at = perf_counter()
                    logger.info(
                        "[HARDWARE-VOICE][%s][收音] 首个音频帧到达 "
                        "距 dialogue.start=%.3f秒 本帧字节=%d",
                        request_id,
                        first_audio_frame_at - turn_started_at,
                        len(binary_data),
                    )
                elif received_frame_count % 100 == 0:
                    logger.debug(
                        "[HARDWARE-VOICE][%s][收音] 已接收音频帧=%d "
                        "累计字节=%d",
                        request_id,
                        received_frame_count,
                        received_bytes,
                    )
                # 连接对端不可信，边接收边限制总大小，避免内存被无限占用。
                if received_bytes > settings.hardware_voice_max_audio_bytes:
                    # 本轮立即作废；必须等待新的 dialogue.start 才能重新收音。
                    collecting = False
                    audio_parts.clear()
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "audio payload too large",
                        }
                    )
                    continue
                # 此处不解析 Speex 帧，因为一个 Speex 帧可能跨多个 WebSocket 消息；
                # 等 dialogue.end 拼成完整字节流后统一解析更可靠。
                audio_parts.append(binary_data)
                continue

            # ==================== 分支 2：JSON 控制文本帧 ====================
            raw_text = message.get("text")
            if not raw_text:
                continue
            try:
                command = json.loads(raw_text)
            except json.JSONDecodeError:
                await send_json(
                    {
                        "type": "dialogue.error",
                        "req": request_id,
                        "error": "invalid JSON control frame",
                    }
                )
                continue
            # JSON 数组、数字、字符串都不是合法控制消息，必须是对象。
            if not isinstance(command, dict):
                await send_json(
                    {
                        "type": "dialogue.error",
                        "req": request_id,
                        "error": "control frame must be a JSON object",
                    }
                )
                continue

            command_type = str(command.get("type") or "")
            # 这是应用层心跳；WebSocket 协议本身的 Ping/Pong 由 ASGI/websockets 处理。
            if command_type == "ping":
                await send_json({"type": "pong"})
                continue

            if command_type == "device.online":
                # 在线事件只做协议确认；真正的设备数据库校验仍在 dialogue.start
                # 时完成，避免设备刚上线但未注册时阻塞长连接。
                online_req = uuid.uuid4().hex[:8]
                connection_device_sn = str(
                    command.get("device_sn") or connection_device_sn
                ).strip()
                connection_device_ip = str(
                    command.get("device_ip") or connection_device_ip
                ).strip()
                await send_json(
                    {
                        "type": "device.online.ack",
                        "req": online_req,
                    }
                )
                logger.info(
                    "[HARDWARE-VOICE][%s] device.online device_sn=%s device_ip=%s",
                    online_req,
                    connection_device_sn,
                    connection_device_ip,
                )
                continue

            if command_type == "dialogue.cancel":
                if turn_cancel_event is not None:
                    turn_cancel_event.set()
                if turn_task is not None and not turn_task.done():
                    turn_task.cancel()
                await send_json(
                    {
                        "type": "dialogue.cancelled",
                        "req": request_id,
                        "client_req": client_req,
                        "reason": str(command.get("reason") or "client_cancel"),
                    }
                )
                continue

            # ==================== 控制消息：dialogue.start ====================
            if command_type == "dialogue.start":
                start_received_at = perf_counter()
                logger.info(
                    "[HARDWARE-VOICE][%s][时序] 收到 dialogue.start",
                    request_id,
                )
                if turn_task is not None and not turn_task.done():
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "previous dialogue is still running",
                        }
                    )
                    continue
                # 固件可传 req 便于两端日志对应；限制长度防止日志字段被滥用。
                request_id = uuid.uuid4().hex[:8]
                client_req = command.get("client_req", command.get("req"))
                device_sn = str(
                    command.get("device_sn") or connection_device_sn
                ).strip()
                connection_device_ip = str(
                    command.get("device_ip") or connection_device_ip
                ).strip()
                if not device_sn:
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "device_sn is required",
                        }
                    )
                    continue

                # audio_format 描述网络上传字节，不是 ASR 最终输入格式：
                # speex 会先解码，pcm 则直接进入百度 ASR。
                audio_format = str(command.get("audio_format") or "pcm").lower()
                if audio_format not in {"pcm", "speex"}:
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "audio_format must be pcm or speex",
                        }
                    )
                    continue
                # 当前硬件协议和百度 ASR 都固定 16k；拒绝其他采样率，避免声音变速、
                # 时长计算错误或识别率异常。
                try:
                    sample_rate = int(command.get("sample_rate") or 16000)
                except (TypeError, ValueError):
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "sample_rate must be an integer",
                        }
                    )
                    continue
                if sample_rate != 16000:
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "sample_rate must be 16000",
                        }
                    )
                    continue

                # 先做数据库校验再回 ready。设备不存在、禁用或未绑定时，不接收
                # 后续音频，也不会产生百度/LLM 调用费用。
                try:
                    session = await _load_hardware_session(
                        device_sn,
                        trace_id=request_id,
                    )
                except (
                    HardwareVoiceError,
                    BizError,
                    SQLAlchemyError,
                    ValueError,
                ) as exc:
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": str(exc),
                        }
                    )
                    continue

                # audio_bytes 表示“实际在 WebSocket 上传的编码字节数”。Speex 模式
                # 下它不是解码后的 pcm_bytes。保留 pcm_bytes 只是兼容早期固件。
                declared_size = command.get("audio_bytes")
                if declared_size is None:
                    # 兼容早期固件使用的 pcm_bytes 字段。
                    declared_size = command.get("pcm_bytes")
                try:
                    expected_audio_bytes = (
                        int(declared_size) if declared_size is not None else None
                    )
                except (TypeError, ValueError):
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "audio_bytes must be an integer",
                        }
                    )
                    continue
                # 在真正上传前检查声明值，可以尽早拒绝明显超限的一轮。
                if expected_audio_bytes is not None and expected_audio_bytes < 0:
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "audio_bytes cannot be negative",
                        }
                    )
                    continue
                if (
                    expected_audio_bytes is not None
                    and expected_audio_bytes > settings.hardware_voice_max_audio_bytes
                ):
                    await send_json(
                        {
                            "type": "dialogue.error",
                            "req": request_id,
                            "error": "declared audio payload too large",
                        }
                    )
                    continue

                # 所有 start 参数和设备上下文验证成功后，重置上一轮残留状态。
                audio_parts.clear()
                received_bytes = 0
                received_frame_count = 0
                first_audio_frame_at = None
                collecting = True
                turn_started_at = perf_counter()
                # ready 表示服务端已经准备好接收本轮二进制音频。
                await send_json(
                    {
                        "type": "dialogue.ready",
                        "req": request_id,
                        "client_req": client_req,
                    }
                )
                logger.info(
                    "[HARDWARE-VOICE][%s][时序] dialogue.start 处理完成，开始收音 "
                    "device_sn=%s format=%s expected=%s 校验耗时=%.3f秒",
                    request_id,
                    device_sn,
                    audio_format,
                    expected_audio_bytes,
                    perf_counter() - start_received_at,
                )
                continue

            # ==================== 控制消息：dialogue.end ====================
            # 未识别的控制消息目前忽略，便于以后扩展 cancel/config 等消息。
            if command_type != "dialogue.end":
                continue
            # end 必须和前面的 start 配对，session 也必须已通过数据库校验。
            if not collecting or session is None:
                await send_json(
                    {
                        "type": "dialogue.error",
                        "req": request_id,
                        "error": "dialogue.end before dialogue.start",
                    }
                )
                continue

            # 从这一刻开始本轮不再接收音频；将分片一次性拼成 Speex/PCM 字节流。
            collecting = False
            audio_data = b"".join(audio_parts)
            # 拼接完成立即释放列表引用，减少大音频在内存中保留的时间。
            audio_parts.clear()
            audio_upload_finished_at = perf_counter()
            logger.info(
                "[HARDWARE-VOICE][%s][收音] 收音结束，开始处理 "
                "帧数=%d 字节=%d 收音耗时=%.3f秒 距 dialogue.start=%.3f秒",
                request_id,
                received_frame_count,
                len(audio_data),
                audio_upload_finished_at - turn_started_at,
                audio_upload_finished_at - turn_started_at,
            )
            if not audio_data:
                await send_json(
                    {
                        "type": "dialogue.error",
                        "req": request_id,
                        "error": "empty audio payload",
                    }
                )
                continue
            # WebSocket/TCP 本身可靠，但大小校验能发现固件声明错误、录音缓冲截断
            # 或错误地把其他二进制数据混入本轮。
            if expected_audio_bytes is not None and expected_audio_bytes != len(
                audio_data
            ):
                await send_json(
                    {
                        "type": "dialogue.error",
                        "req": request_id,
                        "error": (
                            "audio size mismatch: "
                            f"expected {expected_audio_bytes}, got {len(audio_data)}"
                        ),
                    }
                )
                continue

            # ==================== 启动后台软件流水线 ====================
            # 主 WebSocket 协程不能在这里直接 await，否则收到 dialogue.end 后就
            # 无法继续接收 dialogue.cancel。流水线放到 Task 后，主循环继续收帧。
            turn_cancel_event = asyncio.Event()
            turn_task = asyncio.create_task(
                run_turn_worker(
                    worker_request_id=request_id,
                    worker_client_req=client_req,
                    worker_cancel_event=turn_cancel_event,
                    worker_session=session,
                    worker_audio_data=audio_data,
                    worker_audio_format=audio_format,
                    worker_started_at=turn_started_at,
                )
            )
    # 设备主动断线或服务停止取消任务都属于正常连接生命周期，不记异常堆栈。
    except (WebSocketDisconnect, asyncio.CancelledError):
        if turn_cancel_event is not None:
            turn_cancel_event.set()
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
        return
    finally:
        # finally 保证正常断线、异常退出、任务取消三种路径都有离线日志。
        logger.info(
            "[HARDWARE-VOICE][%s][连接] WebSocket 已断开 "
            "连接总时长=%.3f秒 at=%d",
            request_id,
            perf_counter() - connection_started_at,
            time.time_ns(),
        )
