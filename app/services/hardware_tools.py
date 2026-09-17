"""硬件语音链路的工具定义与执行器（工具直连模式）。

和 hardware_actions.resolve() 是**互斥的两套方案**：

  resolve()：关键词层命中就 return → 没命中再发一次独立的 LLM 路由判断
  本模块  ：把工具定义挂到**主对话那一次** LLM 请求上，模型在同一次调用里
            决定要不要调，省掉整轮判断开销，也不用维护关键词词表。

工具执行结果**不回灌**给模型做二次整合：weather.answer_intent() 返回的
就是可以直接播报的成品文案，再走一次 LLM 只会白白多几秒延迟。
"""

import json
import logging
from typing import Any

from app.core.config import Settings, settings
from app.services.hardware_actions import WeatherIntent, WeatherService

logger = logging.getLogger(__name__)

WEATHER_TOOL_NAME = "get_weather"

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": WEATHER_TOOL_NAME,
        "description": (
            "查询指定城市的实时天气或未来天气预报。"
            "当用户**明确想查询天气信息**时调用：问气温多少度、会不会下雨、"
            "要不要带伞、该穿什么衣服、紫外线强不强。"
            "用户只是**随口感慨天气**（例如「今天天气好差」「这天气真烦」"
            "「这天气真是服了」）而没问任何具体信息时，不要调用，正常安慰就行。"
            "与天气无关的闲聊不要调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": (
                        "城市名，如 深圳、上海。用户没提到城市时填空字符串，"
                        "系统会用设备所在地补齐。"
                    ),
                },
                "day_offset": {
                    "type": "integer",
                    "description": "今天=0，明天=1，后天=2",
                },
                "forecast": {
                    "type": "boolean",
                    "description": "问未来天气为 true，问此刻为 false",
                },
            },
            "required": ["city"],
        },
    },
}

# 工具拿不到结果时的兜底话术。**绝不能返回空串** —— 上层没有文字就没有 TTS
# 音频，硬件链路会直接抛「动作没有返回有效音频」，用户听到的是断线。
_FALLBACK_REPLY = "抱歉，我暂时查不到这个信息。"


def build_hardware_tools(config: Settings = settings) -> list[dict[str, Any]] | None:
    """返回本轮要挂给主 LLM 的工具列表；没有可用工具时返回 None。

    返回 None 时上层会退回普通流式对话（请求里不带 tools），行为与改造前
    完全一致 —— 这样"开关没开"或者"天气没配 Key"都不会让链路变得更差。
    """

    if not config.hardware_tool_calling_enable:
        return None
    tools: list[dict[str, Any]] = []
    if config.weather_enable and config.qweather_api_key:
        tools.append(WEATHER_TOOL)
    return tools or None


def _safe_int(value: Any, default: int = 0) -> int:
    """模型可能给出 "明天" 这种非数字值，不能让它直接抛异常。"""

    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "是")
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """把模型给的 arguments 归一成 dict。

    流式下拼出来的是 JSON 字符串，部分模型会直接给 dict。JSON 坏掉时返回
    空 dict，让每个字段各自落默认值，而不是整轮调用失败。
    """

    if isinstance(raw, dict):
        return raw
    if not raw or not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("工具参数不是合法 JSON，按空参数处理 raw=%s", raw[:200])
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def execute_hardware_tool(
        name: str,
        arguments: Any,
        config: Settings = settings,
) -> str | None:
    """执行工具并返回可直接播报的成品文案。"""

    if name != WEATHER_TOOL_NAME:
        logger.warning("收到未注册的工具调用，已忽略 name=%s", name)
        return _FALLBACK_REPLY

    args = parse_tool_arguments(arguments)
    raw_city = args.get("city")
    city = raw_city.strip() if isinstance(raw_city, str) else ""
    # 模型没给城市时用设备默认城市，而不是反问"你在哪个城市"：语音场景多问
    # 一轮的代价比直接用默认城市答大，而且实测流式下模型本来就倾向反问。
    if not city:
        city = config.weather_default_city
    # day_offset 只有 0/1/2 三个合法值，越界就夹到边界，不要让它去索引数组。
    day_offset = max(0, min(2, _safe_int(args.get("day_offset"), 0)))
    forecast = _safe_bool(args.get("forecast"), day_offset > 0)

    logger.info(
        "[硬件工具] 执行 name=%s city=%s day_offset=%d forecast=%s",
        name,
        city,
        day_offset,
        forecast,
    )
    try:
        answer = await WeatherService(config).answer_intent(
            WeatherIntent(city, day_offset, forecast)
        )
    except Exception:  # noqa: BLE001
        # 查询失败不能变成静音，也不能让整轮对话报错，给一句能播出去的兜底。
        logger.exception("[硬件工具] 执行失败 name=%s city=%s", name, city)
        return _FALLBACK_REPLY
    return answer or _FALLBACK_REPLY
