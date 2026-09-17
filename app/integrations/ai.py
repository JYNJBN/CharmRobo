"""统一的大模型 Chat Completions 集成。

上层只关心文字输入和文字增量输出；具体使用哪一家厂商、哪个 API Key、
哪个 Base URL 和哪个模型 ID，由 model_service 中的模型配置决定。
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from time import perf_counter
from typing import Any

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


async def _close_stream(stream: Any) -> None:
    """显式关闭上游流。

    不关的话，某些 httpcore 版本在异步生成器清理时会抛
    "generator didn't stop after athrow()"，变成一条与业务无关的后台异常日志。
    已经关闭的流再关一次是幂等的，关不掉也不该影响本轮结果。
    """

    if stream is None:
        return
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # noqa: BLE001
        logger.debug("关闭上游 LLM 流失败（不影响本轮结果）", exc_info=True)


def _merge_tool_fragments(
        fragments: dict[int, dict[str, Any]],
        tool_calls: Any,
) -> None:
    """把流式返回的 tool_calls 分片按 index 合并成完整调用。

    两个容易踩的点：
      · arguments 是**逐字符分片**下发的，必须字符串拼接，不能覆盖赋值；
      · name / id 只在第一个分片里出现，后续分片是 None，覆盖写会把它们清空。
    """

    for call in tool_calls:
        index = call.index if call.index is not None else 0
        slot = fragments.setdefault(
            index, {"id": None, "name": None, "arguments": ""}
        )
        if call.id:
            slot["id"] = call.id
        function = getattr(call, "function", None)
        if function is None:
            continue
        if function.name:
            slot["name"] = function.name
        if function.arguments:
            slot["arguments"] += function.arguments


def _first_tool_call(
        fragments: dict[int, dict[str, Any]],
) -> tuple[str | None, str]:
    """取第一个（index 最小的）工具调用，返回 (工具名, arguments 原文)。"""

    if not fragments:
        return None, ""
    slot = fragments[min(fragments)]
    return slot.get("name"), slot.get("arguments") or ""


async def stream_chat_with_tools(
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        execute_tool: Callable[[str, Any], Awaitable[str | None]] | None = None,
        trace_id: str = "standalone",
        instructions: str = SYSTEM_INSTRUCTIONS,
        model: ModelRegistryObject | None = None,
) -> AsyncIterator[str]:
    """带工具的主对话流式调用：一次 LLM 完成「判断 + 执行」。

    与 stream_chat 的唯一区别是请求里多了 tools。对外仍然只 yield 文字增量，
    所以上层（分句器 / TTS 队列）完全不用改。

    三种结果：
      1. 模型直接回答 —— 逐字 yield content，与普通流式完全一致；
      2. 模型发起工具调用 —— 执行工具，把返回的成品文案一次性 yield 出去。
         不做第二次 LLM 整合：天气播报文案本身就是成品，再跑一次只多几秒；
      3. 什么都没有 —— yield 一句兜底，**绝不返回空串**（上层没有文字就没有
         TTS 音频，硬件链路会直接抛「动作没有返回有效音频」）。

    关键实测依据（方舟 deepseek-v4-1-flash-260910，thinking=disabled）：
    模型决定调工具时，流里**只有 tool_calls 分片、一个 content 分片都没有**，
    所以收到第一个分片就能定模式，不存在"先播了半句话再改口"的问题。
    万一某个模型两种都给，以先出现的为准并打 warning —— 已经 yield 出去的文字
    撤不回来，这时只能丢掉工具结果，保证用户不会先后听到两套答案。
    """

    model_config = model or resolve_model(None)
    client = get_model_client(model_config)
    llm_messages = _build_messages(
        text=None,
        messages=messages,
        instructions=instructions,
    )
    started_at = perf_counter()
    # None=还没定模式，"content"=普通回答，"tool_calls"=模型要调工具
    mode: str | None = None
    tool_fragments: dict[int, dict[str, Any]] = {}
    content_chars = 0
    mixed_warned = False

    stream: Any = None
    try:
        extra_body = _thinking_extra_body(model_config)
        stream = await client.chat.completions.create(
            model=model_config["model_id"],
            messages=llm_messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
            **({"tools": tools} if tools else {}),
            **({"extra_body": extra_body} if extra_body else {}),
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                if mode is None:
                    mode = "tool_calls"
                    logger.info(
                        "[LLM][%s] 模型选择调用工具，转入工具模式",
                        trace_id,
                    )
                elif mode == "content":
                    if not mixed_warned:
                        mixed_warned = True
                        logger.warning(
                            "[LLM][%s] 同一轮既有文字又有工具调用，"
                            "保留文字、丢弃工具结果",
                            trace_id,
                        )
                # 这里必须是独立的 if，不能写成上面的 else 分支：**第一个分片
                # 同时也是"决定模式"的那一个**，而且 name 只在这个首片里出现。
                # 写成 elif/else 就会把首片漏掉，最终拼不出工具名，整轮只能
                # 落兜底文案（实测踩过：7 条问句全部"没解析出工具名"）。
                if mode == "tool_calls":
                    _merge_tool_fragments(tool_fragments, tool_calls)

            text = getattr(delta, "content", None)
            if not text:
                continue
            if mode is None:
                mode = "content"
            if mode == "content":
                content_chars += len(text)
                yield text

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"{model_config['provider']} Chat Completions "
                f"工具调用失败：{exc}"
            ),
        ) from exc
    finally:
        await _close_stream(stream)

    if mode != "tool_calls":
        if mode is None:
            logger.warning("[LLM][%s] 带工具请求却没收到任何内容", trace_id)
            yield "我没听清，你再说一遍好吗？"
        return

    name, arguments = _first_tool_call(tool_fragments)
    if not name:
        logger.warning("[LLM][%s] 工具模式但没解析出工具名", trace_id)
        yield "抱歉，我暂时没法处理这个请求。"
        return

    result: str | None = None
    if execute_tool is not None:
        try:
            result = await execute_tool(name, arguments)
        except Exception:
            logger.exception(
                "[LLM][%s] 工具执行抛出异常 name=%s", trace_id, name
            )
    logger.info(
        "[LLM][%s] 工具调用结束 name=%s args=%s chars=%d total=%.3fs",
        trace_id,
        name,
        arguments[:120],
        content_chars,
        perf_counter() - started_at,
    )
    yield result or "抱歉，我暂时没法处理这个请求。"
