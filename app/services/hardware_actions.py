"""硬件语音的软件动作层。

本模块只做“用户意图 -> 软件动作”的判断和动作数据准备，不处理 WebSocket、
ESP32 帧、CI 播放器或硬件超时。调用方可以是当前 FastAPI 硬件路由，也可以是
未来对方 Python WebSocket 服务。

优先级与目标项目保持一致：

1. 音乐规则命中：直接选择本地 MP3；
2. 天气规则命中：调用和风天气生成确定性播报文本；
3. 规则未命中且开启工具路由：让 LLM 只输出 JSON 动作判断；
4. 仍未命中：回到普通聊天 LLM。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from app.core.config import Settings, settings
from app.integrations.ai import chat
from app.schemas.Model import ModelRegistryObject

logger = logging.getLogger(__name__)

ActionName = Literal["chat", "music", "music_not_found", "weather", "weather_error"]

_MUSIC_TRIGGER_WORDS = (
    "播放",
    "放音乐",
    "放歌",
    "放一首",
    "放首",
    "听歌",
    "听音乐",
    "来首",
    "来一首",
    "点歌",
    "点一首",
    "换首",
    "唱歌",
    "唱个歌",
    "唱一首",
    "唱首",
    "我想听",
    "想听",
)
_MUSIC_RANDOM_WORDS = ("随便", "随机", "任意")
_MUSIC_COMMAND_WORDS = (
    "播放一下",
    "播放",
    "帮我",
    "给我",
    "请",
    "我想听",
    "想听",
    "听一听",
    "听听",
    "听一下",
    "听",
    "放一下",
    "放一首",
    "放首",
    "放音乐",
    "放歌",
    "来一首",
    "来首",
    "点一首",
    "点歌",
    "换首",
    "唱个歌",
    "唱一首",
    "唱首",
    "唱歌",
    "你会",
    "会不会",
    "能不能",
    "能",
    "可以",
)
_MUSIC_GENERIC_WORDS = ("", "歌", "歌曲", "音乐", "一首", "首歌", "一首歌", "歌听")

_CITY_IDS = {
    "北京": "101010100",
    "上海": "101020100",
    "广州": "101280101",
    "深圳": "101280601",
    "杭州": "101210101",
    "南京": "101190101",
    "苏州": "101190401",
    "成都": "101270101",
    "重庆": "101040100",
    "武汉": "101200101",
    "长沙": "101250101",
    "西安": "101110101",
    "天津": "101030100",
    "青岛": "101120201",
    "厦门": "101230201",
    "东莞": "101281601",
}
_WEATHER_WORDS = (
    "天气",
    "气温",
    "温度",
    "湿度",
    "下雨",
    "下不下雨",
    "会不会下雨",
    "冷不冷",
    "热不热",
    "冷吗",
    "热吗",
    "降温",
    "升温",
    "刮风",
    "风大",
    "带伞",
    "雨伞",
    "穿什么",
    "穿衣",
)
_WEATHER_CONDITION_WORDS = ("晴天", "晴不晴", "阴天", "多云", "太阳")


@dataclass(frozen=True)
class MusicTrack:
    title: str
    path: Path
    size: int
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class MusicSelection:
    query: str
    random_play: bool
    track: MusicTrack | None
    message: str


@dataclass(frozen=True)
class WeatherIntent:
    city: str
    day_offset: int
    forecast: bool


@dataclass(frozen=True)
class ToolDecision:
    action: Literal["chat", "music", "weather"]
    confidence: float
    city: str = ""
    day_offset: int = 0
    forecast: bool = False
    song_query: str = ""
    random_play: bool = False


@dataclass(frozen=True)
class ActionResult:
    action: ActionName
    answer_text: str = ""
    track: MusicTrack | None = None
    action_payload: dict[str, object] | None = None


class LocalMusicLibrary:
    """扫描本地 MP3 并按 ASR 文本匹配歌曲。"""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.tracks = self._scan(Path(config.music_dir))

    def select(
        self, text: str, *, require_trigger: bool = True
    ) -> MusicSelection | None:
        """按文本匹配本地曲库。

        require_trigger=False 是给工具路由用的：路由已经判定「这是听歌请求」，
        此时用户可能只报了歌名或用了词表外的说法（如「橙子那首歌」「来一下
        太阳之子」），句子里没有「播放/想听」这类触发词，不该再拦。
        关键词路径必须保持 require_trigger=True —— 它是"缓存"，如果连触发词都
        不要求，「外面下雪了吗」这种句子里恰好出现的歌名会被误播。
        """

        if not self.config.music_enable:
            return None
        raw = (text or "").strip()
        compact = re.sub(r"[，。！？?、\s]", "", raw)
        if not compact:
            return None
        if require_trigger and not any(
            word in compact for word in _MUSIC_TRIGGER_WORDS
        ):
            return None
        query = compact
        for word in sorted(_MUSIC_COMMAND_WORDS, key=len, reverse=True):
            query = query.replace(word, "")
        query = (
            query.replace("这首歌", "")
            .replace("这首", "")
            .replace("那首歌", "")
            .replace("那首", "")
            .replace("歌曲", "歌")
            .replace("音乐", "歌")
        )
        query = query.strip("吗呢嘛吧呀啊了")
        random_play = any(
            word in compact for word in _MUSIC_RANDOM_WORDS
        ) or _normalize(query) in {_normalize(word) for word in _MUSIC_GENERIC_WORDS}
        if random_play:
            query = ""
        track = self._match(query) if query and self.tracks else None
        if random_play and self.tracks:
            import random

            track = random.choice(self.tracks)
        if track is None:
            message = (
                "本地曲库里还没有可播放的音乐。"
                if not self.tracks
                else f"我这里还没有《{query}》这首歌。"
            )
            return MusicSelection(query, random_play, None, message)
        return MusicSelection(query, random_play, track, f"播放《{track.title}》。")

    def iter_track_bytes(
        self, track: MusicTrack, cancel_event: object | None = None
    ) -> AsyncIterator[bytes]:
        """异步读取本地 MP3，避免阻塞事件循环。"""

        async def iterator() -> AsyncIterator[bytes]:
            chunk_size = max(512, self.config.music_chunk_bytes)
            with track.path.open("rb") as file:
                while True:
                    if _is_cancelled(cancel_event):
                        return
                    chunk = await asyncio.to_thread(file.read, chunk_size)
                    if not chunk:
                        return
                    yield chunk

        return iterator()

    def list_tracks(self) -> list[dict[str, object]]:
        return [
            {"title": track.title, "file": track.path.name, "bytes": track.size}
            for track in self.tracks
        ]

    def _scan(self, music_dir: Path) -> list[MusicTrack]:
        if not music_dir.is_dir():
            return []
        tracks: list[MusicTrack] = []
        for path in sorted(music_dir.glob("*.mp3")):
            title = path.stem.strip()
            if not title:
                continue
            aliases = {_normalize(title), _normalize(path.name)}
            if " - " in title:
                singer, song = title.split(" - ", 1)
                aliases.update({_normalize(song), _normalize(singer + song)})
            tracks.append(
                MusicTrack(
                    title, path, path.stat().st_size, tuple(filter(None, aliases))
                )
            )
        return tracks

    def _match(self, query: str) -> MusicTrack | None:
        needle = _normalize(query)
        best: tuple[int, MusicTrack] | None = None
        for track in self.tracks:
            score = 0
            for alias in track.aliases:
                if needle == alias:
                    score = max(score, 100)
                elif needle in alias or alias in needle:
                    score = max(score, min(len(needle), len(alias)))
            if score and (best is None or score > best[0]):
                best = score, track
        return best[1] if best else None


class WeatherService:
    """和风天气轻量客户端；不开启或未配置 Key 时不参与路由。"""

    def __init__(self, config: Settings) -> None:
        self.config = config

    async def answer(self, text: str) -> str | None:
        if not self.config.weather_enable or not self.config.qweather_api_key:
            return None
        intent = self._parse(text)
        if intent is None:
            return None
        return await self.answer_intent(intent)

    async def answer_intent(self, intent: WeatherIntent) -> str:
        """按动作路由已经解析出的城市和日期查询天气。"""

        key = self.config.qweather_api_key.get_secret_value()
        location = _CITY_IDS.get(intent.city)
        if not location:
            return f"暂时查不到{intent.city}的天气信息。"
        path = "/v7/weather/3d" if intent.forecast else "/v7/weather/now"
        try:
            async with httpx.AsyncClient(
                timeout=self.config.qweather_timeout_sec
            ) as client:
                response = await client.get(
                    f"{self.config.qweather_api_host.rstrip('/')}{path}",
                    params={"location": location},
                    headers={"X-QW-Api-Key": key},
                )
            payload = response.json()
            if str(payload.get("code")) != "200":
                raise RuntimeError(payload)
            if intent.forecast:
                items = payload.get("daily") or []
                item = (
                    items[min(intent.day_offset, max(len(items) - 1, 0))]
                    if items
                    else {}
                )
                return f"{intent.city}{'明天' if intent.day_offset == 1 else '后天' if intent.day_offset == 2 else '今天'}{item.get('textDay', '')}，气温{item.get('tempMin', '')}到{item.get('tempMax', '')}度。"
            now = payload.get("now") or {}
            return f"{intent.city}现在{now.get('text', '')}，气温{now.get('temp', '')}度，体感{now.get('feelsLike', '')}度。"
        except Exception as exc:
            logger.warning("天气查询失败 city=%s error=%s", intent.city, exc)
            raise

    def _parse(self, text: str) -> WeatherIntent | None:
        raw = text or ""
        if not any(word in raw for word in _WEATHER_WORDS):
            # 没命中天气词时，只有「整句本身就是一个天气现象词」（如单独一句
            # “晴天”）才当作天气查询。
            # 这里曾经写成 compact in CONDITIONS or not any(...) → return None，
            # 布尔逻辑正好相反，导致：单说“晴天”反而被排除，而含现象词的更长
            # 句子（如歌名“太阳之子”命中“太阳”）却被判成天气 —— 曲库里名字带
            # 这些词的音乐因此永远播不出来，用户问歌名得到的是天气播报。
            compact = re.sub(r"[，。！？?、\s]", "", raw)
            if compact not in _WEATHER_CONDITION_WORDS:
                return None
        day_offset = 2 if "后天" in raw else 1 if "明天" in raw else 0
        city = next(
            (name for name in sorted(_CITY_IDS, key=len, reverse=True) if name in raw),
            self.config.weather_default_city,
        )
        return WeatherIntent(
            city, day_offset, day_offset > 0 or "预报" in raw or "未来" in raw
        )


class HardwareActionService:
    """统一动作选择，返回结果但不发送 WebSocket。"""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.music = LocalMusicLibrary(config)
        self.weather = WeatherService(config)

    async def resolve(
        self, text: str, model: ModelRegistryObject
    ) -> ActionResult | None:
        music = self.music.select(text)
        if music is not None:
            if music.track is None:
                return ActionResult("music_not_found", music.message)
            return ActionResult(
                "music",
                music.message,
                music.track,
                {
                    "action": "music",
                    "title": music.track.title,
                    "bytes": music.track.size,
                    "timeout_hint_ms": 600000,
                },
            )
        try:
            weather = await self.weather.answer(text)
        except Exception:  # noqa: BLE001
            return ActionResult("weather_error", "天气查询暂时失败，请稍后再试。")
        if weather is not None:
            return ActionResult("weather", weather)
        if not self.config.tool_router_enable:
            return None
        # 两个能力开关都关着时，路由能判出的三个 action 最终都会回落闲聊
        # （chat → None；music → 下面的 music_enable 守卫；weather → 下面的
        # weather_enable 守卫），所以这次 LLM 调用纯属浪费。提前短路，省掉
        # 每轮约 7 秒的判断开销。行为与不短路完全一致，只是不再白花一次请求。
        if not self.config.music_enable and not self.config.weather_enable:
            logger.info(
                "音乐与天气能力都未开启，工具路由无可用动作，跳过本轮判断 text=%s",
                text,
            )
            return None
        # 路由只是「要不要调工具」的判断，失败必须能降级，绝不能阻塞对话。
        # 这里给它加硬超时：超时或结果不可解析都直接 return None，交给上层走
        # run_turn 闲聊。用户侧最坏也只是少了一次工具调用，而不是干等十几秒。
        try:
            decision = await asyncio.wait_for(
                self._tool_decide(text, model),
                timeout=self.config.tool_router_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "工具路由超时（%.1f 秒），本轮回落闲聊 text=%s",
                self.config.tool_router_timeout_sec,
                text,
            )
            return None
        if decision is None:
            return None
        if decision.action == "music":
            # 音乐开关关掉时 select() 会直接返回 None。这时不能再落到下面的
            # music_not_found 兜底，否则会把「功能未开启」谎报成「曲库里没有
            # 这首歌」——用户听到的是"我这儿没这首歌"，而曲库其实是满的。
            # 直接回落闲聊，让用户得到一个正常的回应。
            if not self.config.music_enable:
                logger.info("音乐功能未开启，路由结果回落闲聊 text=%s", text)
                return None
            if decision.song_query:
                # 路由已经确认这是听歌请求，歌名不再过触发词闸门：用户很可能
                # 只说「橙子那首歌」「来一下太阳之子」，句子里没有「播放/想听」。
                # 不放开的话，曲库里明明有《橙子》也会落到 music_not_found，
                # 又变成谎报"曲库没有"。
                music = self.music.select(
                    decision.song_query, require_trigger=False
                )
            else:
                # 路由判了 music 但没给歌名 → 当作"随便放一首"，沿用原路径。
                music = self.music.select("播放一首歌")
            if music and music.track:
                return ActionResult(
                    "music",
                    music.message,
                    music.track,
                    {
                        "action": "music",
                        "title": music.track.title,
                        "bytes": music.track.size,
                        "timeout_hint_ms": 600000,
                    },
                )
            return ActionResult(
                "music_not_found",
                music.message if music else "本地曲库里还没有可播放的音乐。",
            )
        if decision.action == "weather":
            # 与 music 分支保持一致的语义：功能开关关掉就回落闲聊。
            # 这道守卫必须加在这里 —— answer_intent() 自己不检查 weather_enable
            # （它只是"按已解析好的意图去查"的执行层，另一个调用方 answer() 已经
            # 在入口查过了）。少了这道判断，关掉天气开关后路由路径照样会真查真播，
            # 同一个开关在两条路径上表现不一致，调用方无法预测。
            if not self.config.weather_enable:
                logger.info("天气功能未开启，路由结果回落闲聊 text=%s", text)
                return None
            try:
                answer = await self.weather.answer_intent(
                    WeatherIntent(
                        city=decision.city or self.config.weather_default_city,
                        day_offset=decision.day_offset,
                        forecast=decision.forecast,
                    )
                )
                return ActionResult("weather", answer)
            except Exception:  # noqa: BLE001
                return ActionResult("weather_error", "天气查询暂时失败，请稍后再试。")
        return None

    async def _tool_decide(
        self, text: str, model: ModelRegistryObject
    ) -> ToolDecision | None:
        # 字段声明必须与下面的解析代码一一对应。曾经这里只声明了 action /
        # confidence / song_query / random_play 四个字段，却在解析时读 city /
        # day_offset / forecast，导致模型永远不返回这三个字段，路由判出的天气
        # 问题一律落到默认城市和「今天」。
        prompt = (
            "你是语音助手动作路由器，只输出 JSON，不回答用户。格式："
            '{"action":"chat|weather|music","confidence":0.0,'
            '"city":"城市名，没提到就填空字符串","day_offset":0,'
            '"forecast":false,"song_query":"","random_play":false}. '
            "day_offset：今天=0，明天=1，后天=2；forecast：问未来天气为 true。"
            "用户想听歌/播放音乐选 music；天气问题选 weather；其它选 chat。"
        )
        try:
            # max_retries=0：SDK 默认会重试两次，一次超时被放大成两次，用户侧
            # 延迟直接翻倍。路由判断失败本身可以降级，重试没有收益。
            # temperature=0：路由是分类任务，不设温度同一句话会给出不同 action。
            # 实测「给我放橙子」连跑 4 次只有 1 次判 music，其余判 chat —— 用户
            # 侧表现为"时好时坏"，且无法复现排查。分类任务必须固定温度。
            raw = await chat(
                text,
                instructions=prompt,
                model=model,
                max_retries=0,
                temperature=0,
            )
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            payload = json.loads(match.group(0) if match else raw)
            action = str(payload.get("action") or "chat").lower()
            confidence = min(max(float(payload.get("confidence") or 0), 0), 1)
            threshold = (
                self.config.tool_router_music_min_confidence
                if action == "music"
                else self.config.tool_router_min_confidence
            )
            if action not in {"music", "weather"} or confidence < threshold:
                return None
            return ToolDecision(
                action,
                confidence,
                city=str(payload.get("city") or self.config.weather_default_city),
                day_offset=min(max(int(payload.get("day_offset") or 0), 0), 2),
                forecast=bool(payload.get("forecast"))
                or int(payload.get("day_offset") or 0) > 0,
                song_query=str(payload.get("song_query") or ""),
                random_play=bool(payload.get("random_play")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("工具路由跳过 error=%s", exc)
            return None


def _normalize(value: str) -> str:
    return re.sub(r"[\s\-_.·,，。！？?、（）()《》“”\"']", "", value or "").lower()


def _is_cancelled(cancel_event: object | None) -> bool:
    return bool(
        cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)()
    )


# 辅助函数定义完成后再创建单例；创建时会扫描 music 目录并调用 _normalize。
hardware_action_service = HardwareActionService()
