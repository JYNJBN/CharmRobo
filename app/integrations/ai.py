"""火山方舟 Responses API 集成。

这一层只负责把业务文字交给 Ark，并把结果转换成项目内部统一的文字格式。
STT、TTS 和小程序 WebSocket 协议不需要知道 Ark 使用的是哪一种 API。
"""

import logging
from collections.abc import AsyncIterator
from time import perf_counter

from fastapi import HTTPException
from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger("uvicorn.error")

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
SYSTEM_INSTRUCTIONS = (
    "你是由深圳市创梦龙公司开发的小梦机器人。回答尽可能简短，优先用两到三句话回答。"
)
MAX_OUTPUT_TOKENS = 300


def get_ark_client() -> AsyncOpenAI:
    """创建一个连接方舟的异步客户端。"""

    if settings.ark_api_key is None:
        raise HTTPException(
            status_code=500,
            detail="未配置 ARK_API_KEY",
        )

    return AsyncOpenAI(
        api_key=settings.ark_api_key.get_secret_value(),
        base_url=ARK_BASE_URL,
    )


async def chat(
    text: str,
    trace_id: str = "standalone",
    instructions: str = SYSTEM_INSTRUCTIONS,
) -> str:
    """使用 Responses API 进行一次性文字对话。"""

    client = get_ark_client()
    print(text,'text',instructions,'instructions')
    started_at = perf_counter()
    try:
        response = await client.responses.create(
            model=settings.ark_model,
            instructions=instructions,
            input=text,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            # 保存响应，后续接入 previous_response_id 时可以继续使用。
            store=True,
        )

        answer = response.output_text or "我暂时没有想好怎么回答。"
        logger.info(
            "[VOICE-TIMING][%s][LLM] 非流式回答完成 total=%.3fs chars=%d",
            trace_id,
            perf_counter() - started_at,
            len(answer),
        )
        return answer
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ark Responses 调用失败：{exc}",
        ) from exc


async def stream_chat(
    text: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    previous_response_id: str | None = None,
    response_state: dict[str, str | None] | None = None,
    trace_id: str = "standalone",
    instructions: str = SYSTEM_INSTRUCTIONS,
) -> AsyncIterator[str]:
    """使用 Responses API 流式返回文字增量。"""

    client = get_ark_client()

    started_at = perf_counter()
    first_delta_received = False
    delta_count = 0
    character_count = 0
    completed = False

    try:
        create_started_at = perf_counter()
        if messages is None and text is None:
            raise ValueError("text 或 messages 不能同时为空")
        # llm_input =  text if messages is None else messages
        llm_input = ""
        if messages is None:
            if text is None:
                raise ValueError("text 或 messages 不能同时为空")
            llm_input = text
        else:
            llm_input = messages
        stream = await client.responses.create(
            model=settings.ark_model,
            instructions=instructions,
            input=llm_input,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            # previous_response_id=previous_response_id,
            store=True,
            stream=True,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )
        logger.info(
            "[VOICE-TIMING][%s][LLM] Responses 流已建立 elapsed=%.3fs model=%s",
            trace_id,
            perf_counter() - create_started_at,
            settings.ark_model,
        )

        async for event in stream:
            event_type = getattr(event, "type", "")
            if event_type in {
                "response.created",
                "response.completed",
            }:
                response = getattr(event, "response", None)
                response_id = getattr(response, "id", None)

                if response_id and response_state is not None:
                    response_state["response_id"] = response_id

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    delta_count += 1
                    character_count += len(delta)
                    if not first_delta_received:
                        first_delta_received = True
                        logger.info(
                            "[VOICE-TIMING][%s][LLM] 收到首个文字增量 ttft=%.3fs",
                            trace_id,
                            perf_counter() - started_at,
                        )
                    yield delta
                continue

            if event_type == "response.failed":
                error = getattr(event, "error", None)
                raise RuntimeError(f"Responses 流式响应失败：{error or event}")

        completed = True

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ark Responses 流式调用失败：{exc}",
        ) from exc
    finally:
        logger.info(
            "[VOICE-TIMING][%s][LLM] 流式回答结束 total=%.3fs "
            "status=%s deltas=%d chars=%d",
            trace_id,
            perf_counter() - started_at,
            "success" if completed else "failed_or_cancelled",
            delta_count,
            character_count,
        )


# 旧版 Chat Completions 调用方式保留作对照，不再执行：
#
# response = await client.chat.completions.create(
#     model=settings.ark_model,
#     messages=[
#         {"role": "system", "content": SYSTEM_INSTRUCTIONS},
#         {"role": "user", "content": text},
#     ],
# )
# answer = response.choices[0].message.content
#
# 旧版流式读取方式：
#
# stream = await client.chat.completions.create(
#     model=settings.ark_model,
#     messages=messages,
#     stream=True,
# )
# async for chunk in stream:
#     delta = chunk.choices[0].delta.content or ""
