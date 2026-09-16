"""统一的大模型 Chat Completions 集成。

上层只关心文字输入和文字增量输出；具体使用哪一家厂商、哪个 API Key、
哪个 Base URL 和哪个模型 ID，由 model_service 中的模型配置决定。
"""

import logging
from collections.abc import AsyncIterator
from time import perf_counter

from fastapi import HTTPException
from openai import AsyncOpenAI

from app.schemas.Model import ModelRegistryObject
from app.services.model_service import resolve_model

logger = logging.getLogger(__name__)
DEFAULT_PERSONA = "你是小梦机器人，由深圳市创梦龙公司开发。"
STYLE_BASE = "回答尽可能简短，适合语音播报，不使用markdown和表情符号。由深圳市创梦龙公司开发。"
SYSTEM_INSTRUCTIONS = (
    "你是小梦机器人，由深圳市创梦龙公司开发。"
    "回答尽可能简短，优先用两到三句话回答，适合语音播报。"
)
MAX_OUTPUT_TOKENS = 300


def _thinking_extra_body(
        model_config: ModelRegistryObject,
) -> dict[str, object] | None:
    """按模型配置生成 OpenAI 兼容客户端的厂商扩展参数。"""

    thinking_type = model_config.get("thinking_type")
    if not thinking_type:
        return None
    return {"thinking": {"type": thinking_type}}


def build_instructions(
        model_label: str | None = None,
        persona: str | None = None,
        agent_name: str | None = None,
) -> str:
    """拼装人设、回答风格、当前智能体名和模型名。"""
    default_identity = (
        f"你是“{agent_name}”智能体，由深圳市创梦龙公司开发。"
        if agent_name
        else DEFAULT_PERSONA
    )
    identity = persona.strip() if persona and persona.strip() else default_identity
    parts = [identity, STYLE_BASE]
    if agent_name:
        parts.append(
            f"当前设备使用的智能体名称是“{agent_name}”。"
            "当用户询问你是什么智能体、当前使用哪个智能体或智能体叫什么时，"
            f"直接回答当前使用的是“{agent_name}”智能体。"
        )
    if model_label:
        parts.append(
            f"当前设备使用的大模型是{model_label}，具体模型由系统自动选择。"
            "当用户询问你现在使用的是什么大模型时，直接回答这个名称。"
        )
    return "\n".join(parts)


def get_model_client(
        model_config: ModelRegistryObject,
        *,
        max_retries: int | None = None,
) -> AsyncOpenAI:
    """根据模型配置创建 OpenAI 兼容客户端。

    max_retries 默认 None，表示沿用 openai SDK 自身的默认值（也就是会自动
    重试两次），因此既有的 chat / stream_chat 行为完全不变。只有「调用失败
    就降级」的场景才会显式传 0，例如工具路由：SDK 默认重试会把一次超时放大
    成两次，用户侧整体延迟翻倍，而路由判断本来就允许失败后回落闲聊，重试
    没有任何收益。超时统一由调用方用 asyncio.wait_for 在外面兜，这样能覆盖
    整个路由步骤（建客户端、发请求、解析 JSON），而不只是单次 HTTP 请求。
    """

    api_key = model_config["api_key"]
    if not api_key.get_secret_value():
        raise HTTPException(
            status_code=500,
            detail=f"未配置 {model_config['provider']} API Key",
        )

    overrides: dict[str, object] = {}
    if max_retries is not None:
        overrides["max_retries"] = max_retries

    return AsyncOpenAI(
        api_key=api_key.get_secret_value(),
        base_url=model_config["base_url"].rstrip("/"),
        **overrides,
    )


def _build_messages(
        *,
        text: str | None,
        messages: list[dict[str, str]] | None,
        instructions: str,
) -> list[dict[str, str]]:
    """把系统提示词、历史消息和本轮文字组装成 Chat 消息。"""

    if messages is None and text is None:
        raise ValueError("text 或 messages 不能同时为空")

    result: list[dict[str, str]] = []
    if instructions:
        result.append({"role": "system", "content": instructions})

    if messages is not None:
        result.extend(messages)
    else:
        result.append({"role": "user", "content": text or ""})

    return result


async def chat(
        text: str,
        trace_id: str = "standalone",
        instructions: str = SYSTEM_INSTRUCTIONS,
        model: ModelRegistryObject | None = None,
        *,
        max_retries: int | None = None,
        temperature: float | None = None,
) -> str:
    """使用 Chat Completions 进行一次性文字对话。

    max_retries / temperature 默认都是 None，沿用 SDK 与厂商的默认行为；它们是
    后加的关键词参数，不影响任何既有调用方。

    - `max_retries=0`：工具路由用。SDK 默认重试两次，一次超时会被放大成两次，
      用户侧延迟翻倍；而路由失败本来就可以降级，重试没有收益。
    - `temperature=0`：工具路由用。路由是**分类任务**（chat / music / weather），
      不设温度时同一句话多次调用会给出不同 action，表现为"时好时坏"。实测
      「给我放橙子」4 次里只有 1 次判 music，其余判 chat。分类任务要可复现。
    """

    model_config = model or resolve_model(None)
    client = get_model_client(model_config, max_retries=max_retries)
    messages = _build_messages(
        text=text,
        messages=None,
        instructions=instructions,
    )
    logger.debug(
        "text=%s instructions=%s provider=%s model=%s",
        text,
        instructions,
        model_config["provider"],
        model_config["model_id"],
    )
    started_at = perf_counter()

    try:
        extra_body = _thinking_extra_body(model_config)
        response = await client.chat.completions.create(
            model=model_config["model_id"],
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            **({"temperature": temperature} if temperature is not None else {}),
            **({"extra_body": extra_body} if extra_body else {}),
        )

        answer = response.choices[0].message.content or (
            "我暂时没有想好怎么回答。"
        )
        logger.info(
            "[LLM][%s] 非流式回答完成 provider=%s model=%s total=%.3fs chars=%d",
            trace_id,
            model_config["provider"],
            model_config["model_id"],
            perf_counter() - started_at,
            len(answer),
        )
        return answer
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"{model_config['provider']} Chat Completions 调用失败：{exc}"
            ),
        ) from exc


async def stream_chat(
        text: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        previous_response_id: str | None = None,
        response_state: dict[str, str | None] | None = None,
        trace_id: str = "standalone",
        instructions: str = SYSTEM_INSTRUCTIONS,
        model: ModelRegistryObject | None = None,
) -> AsyncIterator[str]:
    """使用 Chat Completions 流式返回文字增量。"""

    # Chat Completions 不使用 Responses API 的 previous_response_id。
    # 保留这两个参数是为了兼容当前上层调用，实际上下文由 messages 提供。
    _ = previous_response_id, response_state

    model_config = model or resolve_model(None)
    client = get_model_client(model_config)
    llm_messages = _build_messages(
        text=text,
        messages=messages,
        instructions=instructions,
    )
    started_at = perf_counter()
    first_delta_received = False
    delta_count = 0
    character_count = 0
    completed = False

    try:
        create_started_at = perf_counter()
        extra_body = _thinking_extra_body(model_config)
        stream = await client.chat.completions.create(
            model=model_config["model_id"],
            messages=llm_messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
            **({"extra_body": extra_body} if extra_body else {}),
        )
        logger.info(
            "[LLM][%s] Chat Completions 流已建立 provider=%s model=%s elapsed=%.3fs",
            trace_id,
            model_config["provider"],
            model_config["model_id"],
            perf_counter() - create_started_at,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            delta_count += 1
            character_count += len(delta)
            if not first_delta_received:
                first_delta_received = True
                logger.info(
                    "[LLM][%s] 收到首个文字增量 ttft=%.3fs",
                    trace_id,
                    perf_counter() - started_at,
                )
            yield delta

        completed = True

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"{model_config['provider']} Chat Completions 流式调用失败：{exc}"
            ),
        ) from exc
    finally:
        logger.info(
            "[LLM][%s] 流式回答结束 total=%.3fs status=%s deltas=%d chars=%d",
            trace_id,
            perf_counter() - started_at,
            "success" if completed else "failed_or_cancelled",
            delta_count,
            character_count,
        )
