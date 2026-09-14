"""AI 语音接口。

浏览器/小程序只连接本文件暴露的接口，火山 API Key 永远只在后端使用：

* ``POST /api/v1/voice/chat``：文字 -> Ark -> 文字；
* ``POST /api/v1/voice/tts``：文字 -> 火山 TTS -> 完整 PCM；
* ``WS /api/v1/voice/tts-stream``：文字 -> 火山 TTS -> PCM 分片；
* ``WS /api/v1/voice/stream``：PCM -> 2608628 极速 HTTP STT -> Ark 流式回答；
* ``WS /api/v1/voice/stream-realtime``：旧版火山流式 STT 备用接口。
"""

import asyncio
import json
import logging
import re
import uuid
from time import perf_counter

import httpx
import websockets
from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import Response
from starlette.websockets import WebSocketDisconnect

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.integrations.ai import chat, stream_chat
from app.integrations.stt import (
    ASR_FLAG_FINAL,
    ASR_FLAG_NEG_SEQUENCE,
    ASR_MSG_ERROR,
    ASR_STREAM_URL,
    build_asr_audio_frame,
    build_asr_config_frame,
    extract_stream_text,
    parse_asr_server_frame,
    stream_asr_headers,
    transcribe_pcm,
    transcribe_wav,
)
from app.integrations.tts import stream_speech_pcm, synthesize_speech
from app.schemas.common import ApiResponse
from app.schemas.voice import (
    TTSRequest,
    VoiceChatRequest,
    VoiceChatResponse,
)
from app.services.agent_service import (
    ensure_default_agent_on_device,
    get_active_agent_for_device,
)
from app.services.conversation import SentenceSplitter, run_turn
from app.services.conversation_memory_service import (
    summarize_conversation_in_background,
)
from app.services.conversation_service import (
    add_conversation_message,
    get_or_create_conversation,
    get_recent_messages,
)
from app.services.device_service import (
    get_active_owner_binding,
    get_device_by_sn,
)
from app.services.model_service import describe_model, resolve_model
from app.services.voice_service import resolve_voice_id

router = APIRouter(prefix="/voice", tags=["AI 语音"])
logger = logging.getLogger(__name__)


@router.post(
    "/chat",
    response_model=ApiResponse[VoiceChatResponse],
)
async def voice_chat_api(
        data: VoiceChatRequest,
) -> ApiResponse[VoiceChatResponse]:
    """文字聊天接口，先采用非流式返回，方便小程序验证链路。"""

    trace_id = uuid.uuid4().hex[:8]
    reply = await chat(data.text, trace_id=trace_id)
    return ApiResponse(data=VoiceChatResponse(reply=reply))


@router.post("/tts")
async def tts_api(data: TTSRequest) -> Response:
    """文字转完整 PCM；当前小程序主要使用下面的 WebSocket 流式接口。"""

    trace_id = uuid.uuid4().hex[:8]
    audio = await synthesize_speech(data.text, data.voice, trace_id=trace_id)
    return Response(
        content=audio,
        media_type="audio/pcm",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/stt")
async def stt_api(request: Request) -> ApiResponse[dict[str, str]]:
    """完整音频转文字，供调试使用；实时页面使用下面的 WebSocket。"""

    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="没有收到音频数据")

    trace_id = uuid.uuid4().hex[:8]
    text = await transcribe_wav(audio, trace_id=trace_id)
    return ApiResponse(data={"text": text})


async def _send_ws_json(client: WebSocket, data: dict[str, str]) -> None:
    """尽力向前端发送错误/状态，避免客户端已断开时再次抛异常。"""

    try:
        await client.send_json(data)
    except (RuntimeError, WebSocketDisconnect):
        pass


@router.websocket("/tts-stream")
async def tts_stream(client: WebSocket) -> None:
    """文字转流式 PCM。

    小程序先发送一条 JSON：``{"text": "你好", "voice": "..."}``；
    后端随后发送一条 config JSON、多条二进制 PCM 分片，最后发送 end JSON。
    """

    # 第一步：接受小程序的 WebSocket 握手。
    await client.accept()
    try:
        # 第二步：第一条消息必须是文字 JSON，里面包含要朗读的文本和音色。
        request = TTSRequest(**json.loads(await client.receive_text()))

        # 第三步：先告诉小程序后面的二进制消息是什么格式。
        # 小程序拿到这些参数后，才能把 PCM 正确转换成 AudioBuffer。
        await client.send_json(
            {
                "type": "config",
                "format": "pcm",
                "sample_rate": 24000,
                "channels": 1,
                "bits": 16,
            }
        )

        # 第四步：连接火山 TTS，并把火山返回的每个 PCM 分片转发给小程序。
        # 这里的 async for 是“收到一块就发送一块”，不会等待全部音频合成完。
        trace_id = uuid.uuid4().hex[:8]
        async for pcm_chunk in stream_speech_pcm(
                request.text,
                request.voice,
                trace_id=trace_id,
        ):
            await client.send_bytes(pcm_chunk)

        # 第五步：所有 PCM 分片发送完后，发送结束标记。
        await client.send_json({"type": "end"})
    except WebSocketDisconnect:
        # 小程序退出页面或网络断开时，直接结束当前连接处理函数。
        return
    except HTTPException as exc:
        # 业务错误，例如音色不存在、TTS Key 未配置，转换成前端能识别的 JSON。
        await _send_ws_json(client, {"type": "error", "detail": str(exc.detail)})
    except httpx.TimeoutException:
        # 火山 TTS 长时间没有返回首个音频分片。
        await _send_ws_json(
            client,
            {"type": "error", "detail": "TTS 首个音频分片超时，请稍后重试。"},
        )
    except Exception as exc:  # noqa: BLE001 - 将异常转成 WebSocket 可读错误
        await _send_ws_json(client, {"type": "error", "detail": str(exc)})


@router.websocket("/stream-realtime")
async def stream_voice_realtime(client: WebSocket) -> None:
    """旧版实时主链路：小程序 PCM -> 火山流式 STT -> Ark 流式回答。

    这个版本保留给后续对比测试，当前页面使用下面的长连接 + 普通 STT 版本。
    """

    await client.accept()
    final_text = ""
    audio_ended = False

    try:
        # 旧版流式 ASR：FastAPI 再连接火山的流式 ASR WebSocket。
        async with websockets.connect(
                ASR_STREAM_URL,
                additional_headers=stream_asr_headers(),
                open_timeout=15,
                close_timeout=5,
        ) as volc_ws:
            # 先发送一次音频配置，告诉火山后面收到的是 16kHz PCM。
            await volc_ws.send(build_asr_config_frame())

            async def upload_audio() -> None:
                """上行任务：接收小程序 PCM，并立即转发给火山 ASR。"""

                nonlocal audio_ended
                while True:
                    message = await client.receive()
                    if message["type"] == "websocket.disconnect":
                        if not audio_ended:
                            await volc_ws.send(build_asr_audio_frame(b"", final=True))
                            audio_ended = True
                        return

                    pcm = message.get("bytes")
                    if pcm:
                        # 音频分片不等待，收到后马上封装成火山协议包并转发。
                        await volc_ws.send(build_asr_audio_frame(pcm))
                        continue

                    raw_text = message.get("text")
                    if not raw_text:
                        continue

                    command = json.loads(raw_text)
                    if command.get("type") == "stop" and not audio_ended:
                        # 空的 final 包不是音频，而是告诉火山“这一句说完了”。
                        await volc_ws.send(build_asr_audio_frame(b"", final=True))
                        audio_ended = True
                        return

            async def receive_results() -> None:
                """下行任务：接收 ASR partial/final，并在 final 后启动 Ark。"""

                nonlocal final_text
                async for raw_frame in volc_ws:
                    if not isinstance(raw_frame, bytes):
                        continue

                    message_type, flags, error_code, data = parse_asr_server_frame(
                        raw_frame
                    )
                    if message_type == ASR_MSG_ERROR:
                        # 火山协议级错误，不能继续使用当前 ASR 会话。
                        detail = data.get("message") or data.get("error") or data
                        raise RuntimeError(f"流式 STT 失败（{error_code}）：{detail}")

                    code = int(data.get("code") or data.get("status_code") or 0)
                    if code not in {0, 1000, 20000000}:
                        raise RuntimeError(
                            f"流式 STT 失败（{code}）：{data.get('message', data)}"
                        )

                    text = extract_stream_text(data)
                    is_final = flags in {ASR_FLAG_FINAL, ASR_FLAG_NEG_SEQUENCE}
                    if text:
                        # partial 可以变化，final 是本轮稳定的最终文字。
                        await client.send_json(
                            {
                                "type": "final" if is_final else "partial",
                                "text": text,
                            }
                        )
                        if is_final:
                            final_text = text

                    if not is_final:
                        continue

                    if not final_text:
                        await _send_ws_json(
                            client,
                            {"type": "error", "detail": "没有识别到文字，请再说一次。"},
                        )
                        return

                    # 只有最终文字才交给 LLM，避免把还会变化的 partial 重复发送给模型。
                    await client.send_json({"type": "reply_start", "text": final_text})
                    full_reply = ""
                    sentence_buffer = ""
                    async for delta in stream_chat(final_text):
                        full_reply += delta
                        sentence_buffer += delta
                        await client.send_json({"type": "reply_delta", "delta": delta})
                        # 一个 delta 里可能包含多个句子，所以要循环切分
                        while True:
                            match = re.search(
                                r"(.+?[。！？；!?;])",
                                sentence_buffer,
                                re.DOTALL,
                            )
                            if not match:
                                break
                            sentence = match.group(1).strip()
                            sentence_buffer = sentence_buffer[match.end():]

                            # 发送给小程序，进入 TTS 队列
                            if sentence:
                                await client.send_json(
                                    {
                                        "type": "tts_text",
                                        "text": sentence,
                                    }
                                )
                            if sentence_buffer.strip():
                                await client.send_json(
                                    {
                                        "type": "tts_text",
                                        "text": sentence_buffer.strip(),
                                    }
                                )
                    await client.send_json(
                        {
                            "type": "reply_end",
                            "text": final_text,
                            "reply": full_reply,
                        }
                    )
                    return

            # 上行和下行必须并发：一边上传声音，一边接收识别结果。
            upload_task = asyncio.create_task(upload_audio())
            receive_task = asyncio.create_task(receive_results())
            done, pending = await asyncio.wait(
                {upload_task, receive_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if upload_task in done:
                await upload_task
                await receive_task
            else:
                await receive_task
                upload_task.cancel()
                await asyncio.gather(upload_task, return_exceptions=True)

            for task in pending:
                if not task.done():
                    task.cancel()
    except (WebSocketDisconnect, asyncio.CancelledError):
        return
    except Exception as exc:  # noqa: BLE001 - 将异常转成 WebSocket 可读错误
        await _send_ws_json(client, {"type": "error", "detail": str(exc)})


@router.websocket("/stream-simple")
async def stream_voice_simple(client: WebSocket) -> None:
    """长连接语音主链路：PCM 收集 -> 普通 STT -> Ark 流式回答。

    连接生命周期由小程序页面控制：

    * 页面进入时建立一次 WebSocket；
    * 每轮录音通过 start/二进制 PCM/stop 完成一次问答；
    * 问答结束后连接继续保持；
    * 页面退出时客户端断开连接。

    和旧版 ``/stream-realtime`` 不同，这里不会再连接火山流式 STT。
    后端会先收完整一轮 PCM，再调用普通录音文件识别接口。
    """

    # 连接建立后不退出函数，while True 会持续等待下一轮 start/stop。
    await client.accept()
    # ready 表示：连接已建立，可以开始发送一轮音频。
    await client.send_json({"type": "ready"})

    is_recording = False
    pcm_buffer = bytearray()

    try:
        while True:
            # receive() 返回一个字典：二进制消息放在 bytes，文字消息放在 text。
            message = await client.receive()

            if message["type"] == "websocket.disconnect":
                # 只有小程序主动退出页面或网络断开，才离开长连接循环。
                return
            logger.debug("message: %s", message)
            # 录音中的二进制消息就是一小段 16kHz/16bit/单声道 PCM。
            pcm = message.get("bytes")
            if pcm:
                if is_recording:
                    # 只有 start 之后的 PCM 才属于当前这一轮录音。
                    pcm_buffer.extend(pcm)
                continue

            raw_text = message.get("text")
            logger.debug("raw_text: %s", raw_text)
            if not raw_text:
                continue

            command = json.loads(raw_text)
            logger.debug("command: %s", command)
            command_type = command.get("type")
            logger.debug("command_type: %s", command_type)
            if command_type == "ping":
                # 可选的心跳消息，用于确认长连接仍然存活。
                await client.send_json({"type": "pong"})
                continue

            if command_type == "start":
                if is_recording:
                    await _send_ws_json(
                        client,
                        {"type": "error", "detail": "当前已经在录音中。"},
                    )
                    continue

                # 新一轮开始前清空上一轮缓存。
                pcm_buffer.clear()
                is_recording = True
                # 这只是状态通知，不包含音频数据。
                await client.send_json({"type": "recording_started"})
                continue

            if command_type != "stop":
                continue

            if not is_recording:
                continue

            # stop 到达：停止接收本轮音频，并复制出完整 PCM。
            is_recording = False
            pcm = bytes(pcm_buffer)
            pcm_buffer.clear()

            if not pcm:
                await _send_ws_json(
                    client,
                    {"type": "error", "detail": "本轮没有收到音频。"},
                )
                await client.send_json({"type": "ready"})
                continue

            # 一轮录音结束后，才调用普通 STT，避免把每个 PCM 分片都单独识别。
            await client.send_json({"type": "stt_start"})
            try:
                # transcribe_pcm 会给裸 PCM 补 WAV 文件头，然后调用普通 HTTP STT。
                final_text = await transcribe_pcm(
                    pcm,
                    sample_rate=16000,
                    channels=1,
                    sample_width=2,
                )
                await client.send_json({"type": "final", "text": final_text})

                # STT 最终文字确定后，再让 Responses API 流式生成回答。
                await client.send_json({"type": "reply_start", "text": final_text})
                full_reply = ""
                async for delta in stream_chat(final_text):
                    # 每个 delta 都是回答的一小段文字，立即推给小程序显示。
                    full_reply += delta
                    await client.send_json({"type": "reply_delta", "delta": delta})

                await client.send_json(
                    {
                        "type": "reply_end",
                        "text": final_text,
                        "reply": full_reply,
                    }
                )
            except HTTPException as exc:
                await _send_ws_json(
                    client,
                    {"type": "error", "detail": str(exc.detail)},
                )
            except Exception as exc:  # noqa: BLE001 - 单轮失败不关闭长连接
                await _send_ws_json(
                    client,
                    {"type": "error", "detail": str(exc)},
                )

            # 本轮失败或成功后，连接都可以继续接收下一轮 start。
            await client.send_json({"type": "ready"})
    except (WebSocketDisconnect, asyncio.CancelledError):
        return
    except Exception as exc:  # noqa: BLE001 - 连接级错误才退出
        await _send_ws_json(client, {"type": "error", "detail": str(exc)})


@router.websocket("/stream")
async def stream_voice(client: WebSocket) -> None:
    """主语音长连接：同一条 WebSocket 同时传输文字 JSON 和 TTS PCM。

    消息约定：

    * 客户端发送 ``start``、PCM 二进制、``stop``；
    * 服务端发送 STT/LLM 状态 JSON；
    * 服务端按句发送 ``tts_start`` JSON、PCM 二进制、``tts_end`` JSON；
    * 页面退出时客户端断开连接。

    旧的“流式 STT”实现保留在 ``/stream-realtime``，本路由使用普通 STT。
    """

    # 接受客户端的 WebSocket 握手，正式建立长连接。
    await client.accept()

    # 创建发送锁，统一保护主 WebSocket 的所有发送操作。
    # LLM 文字任务和 TTS worker 会并发发送消息，必须保证一个消息发送完后再发下一个。
    send_lock = asyncio.Lock()

    # 封装 JSON 发送方法，确保发送过程经过发送锁。
    async def send_json(data: dict[str, object]) -> None:
        async with send_lock:
            await client.send_json(data)

    # 封装二进制发送方法，确保 TTS PCM 分片不会与其他消息交叉发送。
    async def send_bytes(data: bytes) -> None:
        async with send_lock:
            await client.send_bytes(data)

    # 初始化本条长连接和当前录音轮次的状态。
    is_recording = False
    pcm_buffer = bytearray()
    conversation_id: int | None = None
    agent_id: int | None = None
    agent_name: str | None = None
    agent_system_prompt: str | None = None
    device_id: int | None = None
    user_id: int | None = None
    history_messages: list[dict[str, str]] = []
    MAX_HISTORY_MESSAGES = settings.short_term_context_messages
    recording_started_at: float | None = None
    session_command = await client.receive_json()
    # 第一条消息要是
    # {"type": "session_start", "device_sn": "你的设备SN", "conversation_id": null} 格式
    if session_command.get("type") != "session_start":
        await send_json(
            {
                "type": "error",
                "detail": "第一条消息必须是 session_start",
            }
        )
        await client.close()
        return
    device_sn = session_command.get("device_sn")
    # requested_conversation_id = session_command.get("conversation_id")
    if not device_sn:
        await send_json(
            {
                "type": "error",
                "detail": "缺少 device_sn",
            }
        )
        await client.close()
        return
    # 获取deviceid
    async with AsyncSessionLocal() as db:
        device = await get_device_by_sn(db, device_sn)

        if device is None:
            await send_json(
                {
                    "type": "error",
                    "detail": "设备不存在或已删除",
                }
            )
            await client.close()
            return

        if device.status != 1:
            await send_json(
                {
                    "type": "error",
                    "detail": "设备已被禁用",
                }
            )
            await client.close()
            return

        device_id = device.id
        # 获取用户id
        owner_binding = await get_active_owner_binding(
            db,
            device_id=device_id,
        )
        # 老设备兜底：激活指针为空或失效时，物化默认智能体副本（手册 6-Step1）。
        agent = await ensure_default_agent_on_device(db=db, device=device)
        if agent is None:
            await send_json({"type": "error", "detail": "设备没有可用智能体"})
            await client.close()
            return
        # ensure 内部只 flush 不 commit，物化结果在这里统一提交。
        await db.commit()

        if owner_binding is None:
            await send_json(
                {
                    "type": "error",
                    "detail": "设备尚未绑定用户",
                }
            )
            await client.close()
            return
        user_id = owner_binding.user_id
        # agent配置
        agent_id = agent.id
        agent_name = agent.name
        agent_system_prompt = agent.system_prompt
        model_config = resolve_model(agent.model_key)
        model_label = describe_model(agent.model_key)
        voice_id = resolve_voice_id(agent.voice)
    async with AsyncSessionLocal() as db:
        conversation = await get_or_create_conversation(
            db,
            user_id=user_id,
            device_id=device_id,
            agent_id=agent_id,
        )
        conversation_id = conversation.id
        history_messages = await get_recent_messages(
            db,
            conversation_id=conversation_id,
            limit=MAX_HISTORY_MESSAGES,
        )

    await send_json(
        {
            "type": "session_ready",
            "device_sn": device_sn,
            "history_count": len(history_messages),
        }
    )
    # 通知客户端连接已经准备好，可以开始发送录音。
    await send_json({"type": "ready"})
    try:
        # 持续接收客户端消息，一条连接可以处理多轮语音交互。
        while True:
            message = await client.receive()

            # 客户端主动断开时，结束当前 WebSocket 处理函数。
            if message["type"] == "websocket.disconnect":
                return

            # 录音中的二进制消息是 16kHz/16bit/单声道 PCM。
            pcm = message.get("bytes")
            if pcm:
                # 录音状态下缓存客户端上传的 PCM 音频数据。
                if is_recording:
                    pcm_buffer.extend(pcm)
                continue

            # 读取客户端发送的文本消息，并忽略空消息。
            raw_text = message.get("text")
            if not raw_text:
                continue

            # 解析 JSON 指令，获取客户端要求执行的操作类型。
            command = json.loads(raw_text)
            command_type = command.get("type")

            # 处理心跳请求，保持连接活跃。
            if command_type == "ping":
                await send_json({"type": "pong"})
                continue

            # 处理开始录音指令。
            if command_type == "start":
                if is_recording:
                    # 防止客户端重复开始同一轮录音。
                    await send_json({"type": "error", "detail": "当前已经在录音中。"})
                    continue

                # 清空上一轮缓存，开始记录新一轮录音的 PCM 数据。
                pcm_buffer.clear()
                is_recording = True
                recording_started_at = perf_counter()
                await send_json({"type": "recording_started"})
                continue

            # 只处理录音状态下的 stop 指令，其他指令直接忽略。
            if command_type != "stop" or not is_recording:
                continue

            # 结束录音并固定本轮音频内容，后续处理不再修改 pcm_buffer。
            is_recording = False
            pcm = bytes(pcm_buffer)
            pcm_buffer.clear()
            trace_id = uuid.uuid4().hex[:8]
            round_started_at = perf_counter()

            # 计算录音耗时、音频时长和 PCM 大小，便于排查链路延迟。
            recording_elapsed = (
                round_started_at - recording_started_at
                if recording_started_at is not None
                else 0.0
            )
            recording_started_at = None
            audio_duration = len(pcm) / (16000 * 1 * 2) if pcm else 0.0
            logger.info(
                "[VOICE-TIMING][%s][PIPELINE] 录音接收结束 "
                "recording_elapsed=%.3fs audio_duration=%.3fs pcm_bytes=%d",
                trace_id,
                recording_elapsed,
                audio_duration,
                len(pcm),
            )

            # 没有收到音频时直接返回错误，并准备下一轮录音。
            if not pcm:
                await send_json({"type": "error", "detail": "本轮没有收到音频。"})
                await send_json({"type": "ready"})
                continue

            # 通知客户端开始进行语音识别。
            await send_json({"type": "stt_start"})

            try:
                # 普通 STT：完整一轮 PCM 收完后，转换成 WAV 并请求一次。
                # 调用 STT 服务，将完整 PCM 音频转换为用户文本。
                stt_started_at = perf_counter()
                final_text = await transcribe_pcm(
                    pcm,
                    sample_rate=16000,
                    channels=1,
                    sample_width=2,
                    trace_id=trace_id,
                )
                # 删除stt录音
                del pcm
                logger.info(
                    "[VOICE-TIMING][%s][PIPELINE] STT 阶段结束 elapsed=%.3fs",
                    trace_id,
                    perf_counter() - stt_started_at,
                )
                # 把识别出的最终文本发送给客户端，并通知客户端开始等待回答。
                if conversation_id is None:
                    raise RuntimeError("会话初始化失败，无法保存消息")

                async with AsyncSessionLocal() as db:
                    await add_conversation_message(
                        db,
                        conversation_id=conversation_id,
                        role="user",
                        content=final_text,
                        source="voice",
                    )

                await send_json({"type": "final", "text": final_text})
                await send_json({"type": "reply_start", "text": final_text})

                # 每轮重新读一次设备当前配置的模型：
                # 长连接建连时只算一次的话，中途在小程序里切换模型不会生效，
                # 必须等重连才变化。这里按轮刷新，切完立刻生效。
                try:
                    async with AsyncSessionLocal() as db:
                        fresh_device = await get_device_by_sn(db, device_sn)
                        if fresh_device is not None:
                            fresh_agent = await get_active_agent_for_device(db, fresh_device)
                            model_config = resolve_model(fresh_agent.model_key)
                            model_label = describe_model(fresh_agent.model_key)
                            voice_id = resolve_voice_id(fresh_agent.voice)
                            agent_name = fresh_agent.name
                            agent_system_prompt = fresh_agent.system_prompt
                            device_id = fresh_device.id
                            # 智能体被切换了：会话和历史跟着切到新智能体名
                            if fresh_agent.id != agent_id:
                                agent_id = fresh_agent.id
                                conversation = await get_or_create_conversation(
                                    db,
                                    user_id=user_id,
                                    device_id=device_id,
                                    agent_id=agent_id,
                                )
                                conversation_id = conversation.id
                                history_messages = await get_recent_messages(
                                    db,
                                    conversation_id=conversation_id,
                                    limit=MAX_HISTORY_MESSAGES,
                                )
                except ValueError as exc:
                    # 例如切到一个 MODEL_REGISTRY 里没有、或 .env 未配置模型 id 的 key：
                    # 保持上一轮的模型继续服务，不要把整轮对话打断。
                    logger.warning(
                        "[VOICE][%s] 模型解析失败，沿用上一轮模型: %s",
                        trace_id,
                        exc,
                    )
                logger.info(
                    "[VOICE][%s] 本轮使用模型 label=%s model=%s",
                    trace_id,
                    model_label,
                    model_config["model_id"],
                )

                # TTS worker 与 LLM 生成并行：LLM 继续产出文字，worker 负责合成已经完成的句子。
                # 创建 TTS 队列和计时状态，用于异步处理已完成的句子。
                tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
                timing_state: dict[str, float | int | None] = {
                    "first_tts_audio_at": None,
                    "tts_sentence_count": 0,
                    "tts_worker_started_at": None,
                    "tts_active_started_at": None,
                }

                async def tts_worker(
                        queue: asyncio.Queue[str | None] = tts_queue,
                        state: dict[str, float | int | None] = timing_state,
                        current_trace_id: str = trace_id,
                        current_voice_id: str = voice_id,
                        current_round_started_at: float = round_started_at,
                ) -> None:
                    """按顺序消费句子，并把 TTS PCM 复用主 WebSocket 发给小程序。"""

                    # 启动 TTS worker，持续从队列中取出句子进行语音合成。
                    state["tts_worker_started_at"] = perf_counter()
                    while True:
                        sentence = await queue.get()
                        try:
                            if sentence is None:
                                # 收到结束标记，说明 LLM 已经没有更多句子需要合成。
                                return

                            # worker 创建后会先等 LLM 产出完整句子；这里才是 TTS 真正开始。
                            if state["tts_active_started_at"] is None:
                                state["tts_active_started_at"] = perf_counter()

                            # 记录当前句子序号，并通知客户端即将发送该句的音频。
                            sentence_no = int(state["tts_sentence_count"] or 0) + 1
                            state["tts_sentence_count"] = sentence_no

                            await send_json(
                                {
                                    "type": "tts_start",
                                    "format": "pcm",
                                    "sample_rate": 24000,
                                    "channels": 1,
                                    "bits": 16,
                                    "text": sentence,
                                }
                            )

                            # stream_speech_pcm 是 FastAPI 到火山 TTS 的内部连接，逐片返回 PCM 后经主 WebSocket 转发给小程序。
                            async for pcm_chunk in stream_speech_pcm(
                                    sentence,
                                    current_voice_id,
                                    trace_id=current_trace_id,
                                    sentence_no=sentence_no,
                            ):
                                if state["first_tts_audio_at"] is None:
                                    first_tts_audio_at = perf_counter()
                                    state["first_tts_audio_at"] = first_tts_audio_at
                                    logger.info(
                                        "[VOICE-TIMING][%s][PIPELINE] 首段 TTS 音频开始转发 "
                                        "elapsed_from_stop=%.3fs",
                                        current_trace_id,
                                        first_tts_audio_at - current_round_started_at,
                                    )
                                await send_bytes(pcm_chunk)

                            # 当前句子的所有音频发送完成，通知客户端结束本句播放。
                            await send_json({"type": "tts_end"})
                        except Exception as exc:  # noqa: BLE001 - 单句 TTS 失败不影响文字
                            # 单句 TTS 失败只通知客户端，不中断剩余文字和后续句子的处理。
                            await send_json({"type": "tts_error", "detail": str(exc)})
                        finally:
                            # 无论成功还是失败，都标记当前队列任务已经处理完毕。
                            queue.task_done()

                # 后台启动 TTS worker，使 LLM 生成文字和 TTS 合成可以并行进行。
                tts_task = asyncio.create_task(tts_worker())
                full_reply = ""
                splitter = SentenceSplitter()
                llm_started_at = perf_counter()
                first_reply_delta_received = False
                # 调用共享对话层 run_turn，逐段接收 LLM 回答文本。
                # logger.info('所有消息%s',history_messages)
                async for delta in run_turn(
                        final_text,
                        history_messages,
                        conversation_id,
                        user_id=user_id,
                        device_id=device_id,
                        trace_id=trace_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        agent_system_prompt=agent_system_prompt,
                        model=model_config,
                        # 每轮刷新的模型名（豆包/DeepSeek），注入系统提示词用
                        model_label=model_label
                ):
                    if not first_reply_delta_received:
                        first_reply_delta_received = True
                        logger.info(
                            "[VOICE-TIMING][%s][PIPELINE] 首个回答文字开始转发 "
                            "elapsed_from_stop=%.3fs elapsed_from_llm=%.3fs",
                            trace_id,
                            perf_counter() - round_started_at,
                            perf_counter() - llm_started_at,
                        )
                    full_reply += delta

                    # 立即转发每个文字增量，让前端边生成边显示回答。
                    await send_json({"type": "reply_delta", "delta": delta})

                    # 从累计文本中提取完整句子，交给 TTS worker 排队合成。
                    # 分句逻辑已抽到共享层 SentenceSplitter，保证 /stream 与
                    # 未来 /dialogue/ws 复用同一套切句规则。
                    for sentence in splitter.feed(delta):
                        # put 不会调用 TTS，只是把句子交给后台 worker 排队。
                        await tts_queue.put(sentence)

                # LLM 生成结束，记录阶段耗时。
                logger.info(
                    "[VOICE-TIMING][%s][PIPELINE] LLM 文字阶段结束 elapsed=%.3fs",
                    trace_id,
                    perf_counter() - llm_started_at,
                )

                # 最后一段可能没有标点，回答结束时也要送进 TTS 队列。
                # 把没有标点结尾的剩余文本也加入 TTS 队列。
                for sentence in splitter.flush():
                    await tts_queue.put(sentence)
                #     添加历史聊天记录
                async with AsyncSessionLocal() as db:
                    await add_conversation_message(
                        db,
                        conversation_id=conversation_id,
                        role="assistant",
                        content=full_reply,
                        source="voice",
                    )
                asyncio.create_task(
                    summarize_conversation_in_background(
                        conversation_id=conversation_id,
                        device_id=device_id,
                        agent_id=agent_id,
                        user_id=user_id,
                        trace_id=trace_id,
                    )
                )
                history_messages.extend(
                    [
                        {
                            "role": "user",
                            "content": final_text,
                        },
                        {
                            "role": "assistant",
                            "content": full_reply,
                        },
                    ]
                )
                # 如果聊天记录大于20条对话只取最新20条
                if len(history_messages) > MAX_HISTORY_MESSAGES:
                    del history_messages[:-MAX_HISTORY_MESSAGES]
                # 文字回答已经完整，通知前端文字结束。
                # 发送完整回答文本，通知客户端 LLM 阶段结束。
                await send_json(
                    {
                        "type": "reply_end",
                        "text": final_text,
                        "reply": full_reply,
                    }
                )

                # 等所有句子的 TTS 都发送完成，再结束本轮。
                # 向 worker 发送结束标记，并等待所有 TTS 队列任务完成。
                await tts_queue.put(None)
                # await tts_queue.join()
                # await tts_task 等的是worker这个协程本身结束
                await tts_task
                # 记录 TTS 阶段和整轮语音交互的耗时。
                tts_worker_started_at = timing_state["tts_worker_started_at"]
                tts_active_started_at = timing_state["tts_active_started_at"]
                if isinstance(tts_worker_started_at, float) and isinstance(
                        tts_active_started_at,
                        float,
                ):
                    logger.info(
                        "[VOICE-TIMING][%s][PIPELINE] 全部 TTS 阶段结束 "
                        "active_elapsed=%.3fs including_wait=%.3fs sentences=%d",
                        trace_id,
                        perf_counter() - tts_active_started_at,
                        perf_counter() - tts_worker_started_at,
                        int(timing_state["tts_sentence_count"] or 0),
                    )
                logger.info(
                    "[VOICE-TIMING][%s][PIPELINE] 本轮语音交互全部完成 total=%.3fs",
                    trace_id,
                    perf_counter() - round_started_at,
                )
            except HTTPException as exc:
                # 处理可预期的业务异常，并把错误信息返回给客户端。
                logger.info(
                    "[VOICE-TIMING][%s][PIPELINE] 本轮失败 elapsed=%.3fs error=%s",
                    trace_id,
                    perf_counter() - round_started_at,
                    exc.detail,
                )
                await send_json({"type": "error", "detail": str(exc.detail)})
            except Exception as exc:
                # 记录未预期异常，并返回通用错误信息。
                logger.exception(
                    "[VOICE-TIMING][%s][PIPELINE] 本轮异常 elapsed=%.3fs",
                    trace_id,
                    perf_counter() - round_started_at,
                )
                await send_json({"type": "error", "detail": str(exc)})

            # 本轮成功或失败后都保留连接，通知客户端可以开始下一轮录音。
            await send_json({"type": "ready"})
    except (WebSocketDisconnect, asyncio.CancelledError):
        # 客户端断开或任务被取消时，安静结束连接处理。
        return
    except Exception as exc:  # noqa: BLE001 - 连接级错误才退出
        # 处理连接级异常；此时无法继续当前循环，只尝试发送错误。
        await _send_ws_json(client, {"type": "error", "detail": str(exc)})


@router.get("/stream-info", tags=["AI 语音"])
async def stream_info():
    return {
        "protocol": "WebSocket",
        "url": "/api/v1/voice/stream",
        "message": (
            "页面进入时连接；每轮依次发送 "
            "{type: start}、PCM 二进制、{type: stop}；"
            "文字使用 JSON，TTS 使用 tts_start/PCM/tts_end；退出页面时断开。"
        ),
    }
