"""百度短语音识别和流式文本合成客户端。

硬件语音链路使用三个百度接口：

1. OAuth：API Key + Secret Key 换取有有效期的 ``access_token``；
2. 短语音识别标准版：一次性 POST 完整 16k PCM，返回最终文字；
3. 流式文本在线合成：WebSocket 发送短句，持续接收 16k MP3 二进制帧。

本模块只负责“百度供应商协议”，不处理 ESP32 的 start/end 消息，也不操作
数据库。硬件 WebSocket 层只调用 ``recognize_pcm`` 和
``stream_synthesize_mp3``，因此将来更换百度接口或鉴权方式时，不需要改固件
协议。

ASR 和 TTS 分别维护 Token 缓存。项目允许它们使用不同百度应用凭据；即使
实际填写同一套 Key，分开缓存也能避免一个接口的鉴权失败误清除另一个正在
使用的 Token。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

import httpx
import websockets

from app.core.config import Settings, settings

logger = logging.getLogger(__name__)

TokenKind = Literal["asr", "tts"]

# 百度可能通过 HTTP 状态码，也可能通过 HTTP 200 内的业务错误码表示鉴权失败。
# 这些错误与普通识别/合成错误不同：可以清除缓存、重新获取 Token 并重试一次。
# 只允许重试一次，防止凭据或权限确实错误时形成无限请求循环。
AUTH_ERROR_CODES = {110, 111, 3302, 401, 403}


class BaiduSpeechError(RuntimeError):
    """百度语音接口调用失败。"""


class _BaiduAuthError(BaiduSpeechError):
    """百度凭据或 Access Token 失效。"""


@dataclass
class _TokenCache:
    """进程内 Access Token 缓存。

    ``expires_at`` 是根据 OAuth 返回的 ``expires_in`` 换算出的 Unix 时间。
    缓存不写磁盘，服务重启后第一次语音请求会重新获取 Token。
    """

    token: str = ""
    expires_at: float = 0.0

    def valid(self) -> bool:
        """Token 存在且距离官方过期时间大于十分钟时才视为有效。"""

        # 提前十分钟刷新，避免 ASR/TTS 请求建立后 Token 在处理中到期；也给
        # 本机与百度服务器之间可能存在的少量时钟偏差留出余量。
        return bool(self.token) and time.time() < self.expires_at - 600


class BaiduSpeechClient:
    """异步百度 ASR/TTS 客户端，供所有 FastAPI 硬件连接共享。"""

    def __init__(self, config: Settings) -> None:
        # 保存 Settings 对象而不是复制字段，便于测试时注入独立配置实例。
        self.config = config

        # ASR/TTS 可能使用不同应用，因此凭据、缓存和刷新锁完全独立。
        self._asr_token = _TokenCache()
        self._tts_token = _TokenCache()

        # 多台设备可能在 Token 同时过期后一起发起请求。每类 Token 使用一把锁，
        # 保证只有第一个协程访问 OAuth，其余协程等待并复用刷新结果。
        self._asr_token_lock = asyncio.Lock()
        self._tts_token_lock = asyncio.Lock()

    @staticmethod
    def _secret_value(secret: object, label: str) -> str:
        """安全提取 Pydantic SecretStr，不把真实凭据拼进错误信息。"""

        if secret is None:
            raise BaiduSpeechError(f"{label} 未配置")
        # 正常运行时是 SecretStr；保留 str 兼容路径，便于单元测试注入假配置。
        getter = getattr(secret, "get_secret_value", None)
        value = getter() if callable(getter) else str(secret)
        if not value:
            raise BaiduSpeechError(f"{label} 未配置")
        return value

    def _credentials(self, kind: TokenKind) -> tuple[str, str]:
        """根据 ASR/TTS 类型选择对应的 API Key 和 Secret Key。"""

        if kind == "asr":
            return (
                self._secret_value(
                    self.config.baidu_asr_api_key,
                    "BAIDU_ASR_API_KEY",
                ),
                self._secret_value(
                    self.config.baidu_asr_secret_key,
                    "BAIDU_ASR_SECRET_KEY",
                ),
            )
        return (
            self._secret_value(
                self.config.baidu_tts_api_key,
                "BAIDU_TTS_API_KEY",
            ),
            self._secret_value(
                self.config.baidu_tts_secret_key,
                "BAIDU_TTS_SECRET_KEY",
            ),
        )

    def _cache_and_lock(
        self,
        kind: TokenKind,
    ) -> tuple[_TokenCache, asyncio.Lock]:
        """返回指定接口的当前缓存对象和刷新锁。"""

        if kind == "asr":
            return self._asr_token, self._asr_token_lock
        return self._tts_token, self._tts_token_lock

    async def _get_token(self, kind: TokenKind) -> str:
        """读取可用 Token；缺失或临近过期时只刷新一次。

        这里采用“双重检查锁”：加锁前先检查可以避免正常请求都排队；获得锁后
        再检查一次，是因为等待锁期间可能已有另一个协程完成了刷新。
        """

        cache, lock = self._cache_and_lock(kind)
        # 高频路径：Token 有效时不获取锁，直接返回。
        if cache.valid():
            return cache.token

        async with lock:
            # 等锁期间其他协程可能已经刷新，必须重新获取 cache，不能继续使用
            # 加锁前保存的旧对象。
            cache, _ = self._cache_and_lock(kind)
            if cache.valid():
                return cache.token

            # 只有锁内第一个协程真正请求 OAuth。
            api_key, secret_key = self._credentials(kind)
            refreshed = await self._request_token(api_key, secret_key)
            if kind == "asr":
                self._asr_token = refreshed
            else:
                self._tts_token = refreshed
            return refreshed.token

    def _invalidate_token(self, kind: TokenKind, token: str) -> None:
        """仅当失败 Token 仍是当前缓存时才使其失效。

        比较 Token 是为了处理并发竞态：某请求拿着旧 Token 失败时，另一个协程
        可能已经刷新成功；这时不能把刚得到的新 Token 一并清掉。
        """

        cache, _ = self._cache_and_lock(kind)
        if cache.token != token:
            return
        if kind == "asr":
            self._asr_token = _TokenCache()
        else:
            self._tts_token = _TokenCache()

    async def _request_token(self, api_key: str, secret_key: str) -> _TokenCache:
        """通过 OAuth client_credentials 模式获取并缓存 Access Token。

        百度要求 ``grant_type``、``client_id``、``client_secret`` 放在查询参数；
        响应中的 ``expires_in`` 是秒数。日志和异常均不包含真实 Key 或 Token。
        """

        try:
            # OAuth 请求很短且不复用响应流，使用独立 AsyncClient 并设置 15 秒
            # 总超时，避免百度不可达时长期占住硬件连接。
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    self.config.baidu_oauth_url,
                    params={
                        "grant_type": "client_credentials",
                        "client_id": api_key,
                        "client_secret": secret_key,
                    },
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                )
        except httpx.HTTPError as exc:
            # DNS、连接、TLS、读取超时等统一转换成本模块异常。
            raise BaiduSpeechError(f"百度 OAuth 网络请求失败：{exc}") from exc

        # HTTP 非 200 时不尝试从响应中推断 Token，直接按鉴权服务失败处理。
        if response.status_code != 200:
            raise BaiduSpeechError(f"百度 OAuth 返回 HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise BaiduSpeechError("百度 OAuth 返回了无效 JSON") from exc

        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in") or 0)
        if not token or expires_in <= 0:
            error = payload.get("error_description") or payload.get("error")
            raise BaiduSpeechError(
                f"百度 OAuth 未返回有效 Token：{error or '未知错误'}"
            )
        # 使用本机绝对时间保存到期点；valid() 会在这个时间前 10 分钟刷新。
        return _TokenCache(
            token=str(token),
            expires_at=time.time() + expires_in,
        )

    async def recognize_pcm(self, pcm: bytes, *, cuid: str) -> str:
        """调用百度短语音识别标准版 RAW PCM 接口。

        ``pcm`` 必须是 16 kHz、16-bit、小端、单声道裸数据，不能包含 WAV 头。
        ``cuid`` 使用设备 SN，并截到百度要求的 60 字符以内。

        这是非流式接口：完整 PCM 放在一个 HTTP body 中，方法只返回最终最优
        识别文本，不返回 partial 结果。
        """

        if not pcm:
            raise BaiduSpeechError("百度 ASR 输入音频为空")

        # 最多两次：正常调用一次；仅鉴权失败时刷新 Token 后再调用一次。
        for attempt in range(2):
            token = await self._get_token("asr")
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        self.config.baidu_asr_url,
                        params={
                            # cuid 用于百度侧区分调用终端，不是鉴权密钥。
                            "cuid": cuid[:60],
                            # OAuth access_token 按百度 RAW 接口要求放在 URL 参数中。
                            "token": token,
                            # 1537 是普通话近场模型，可通过配置覆盖。
                            "dev_pid": self.config.baidu_asr_dev_pid,
                        },
                        headers={
                            # RAW 上传没有 WAV 头，必须在 Content-Type 中明确格式
                            # 和采样率；httpx 会自动补 Content-Length。
                            "Content-Type": (
                                f"audio/pcm;rate={self.config.baidu_asr_sample_rate}"
                            )
                        },
                        # 不做 base64，直接上传原始 PCM，避免额外 1/3 体积。
                        content=pcm,
                    )
            except httpx.HTTPError as exc:
                raise BaiduSpeechError(f"百度 ASR 网络请求失败：{exc}") from exc

            # 百度可能直接通过 HTTP 401/403 表示 Token/权限错误。
            if response.status_code in {401, 403}:
                self._invalidate_token("asr", token)
                if attempt == 0:
                    # 下一次循环会调用 _get_token，因缓存已清空而重新请求 OAuth。
                    continue
                raise BaiduSpeechError(
                    f"百度 ASR 鉴权失败：HTTP {response.status_code}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise BaiduSpeechError("百度 ASR 返回了无效 JSON") from exc

            error_number = int(payload.get("err_no") or 0)
            # 部分鉴权错误封装在 HTTP 200 的 err_no 中，处理方式与 401/403 相同。
            if error_number in AUTH_ERROR_CODES:
                self._invalidate_token("asr", token)
                if attempt == 0:
                    continue
                raise BaiduSpeechError(
                    f"百度 ASR 鉴权失败：{payload.get('err_msg') or error_number}"
                )
            if response.status_code != 200 or error_number != 0:
                # 音频质量、格式、配额等非鉴权错误刷新 Token 没有意义，直接返回。
                raise BaiduSpeechError(
                    "百度 ASR 识别失败："
                    f"{payload.get('err_msg') or response.status_code}"
                )

            results = payload.get("result") or []
            if not results:
                raise BaiduSpeechError("百度 ASR 没有识别到文字")
            # 标准版可能返回候选数组，当前业务使用第一个最优结果。
            return str(results[0]).strip()

        raise BaiduSpeechError("百度 ASR 鉴权重试失败")

    async def stream_synthesize_mp3(
        self,
        text: str,
        *,
        per: str | None = None,
    ) -> AsyncIterator[bytes]:
        """流式产出百度 TTS 返回的 16 kHz MP3 二进制分片。

        调用方应边迭代边发送给 ESP32，不应先 ``join`` 成完整 MP3，否则会失去
        流式首包优势。文本按官方限制截到 1000 字符；上层通常已经按短句调用。

        鉴权失败可以刷新 Token 重试，但一旦已经 yield 过音频，绝不能从头重试，
        否则硬件会听到重复的句子或收到两个 MP3 流的重复开头。
        """

        normalized_text = text.strip()
        if not normalized_text:
            raise BaiduSpeechError("百度 TTS 输入文本为空")

        # 标记是否已经把本句任何音频交给上层，用于决定失败后能否安全重试。
        yielded_audio = False
        for attempt in range(2):
            token = await self._get_token("tts")
            try:
                async for chunk in self._stream_tts_once(
                    normalized_text[:1000],
                    token,
                    per=per,
                ):
                    yielded_audio = True
                    yield chunk
                return
            except _BaiduAuthError:
                # 只清除这次失败使用的 Token；并发刷新出的新 Token 不会被误删。
                self._invalidate_token("tts", token)
                # 已经发过声音时重试会造成重复播放；第二次仍失败也必须停止。
                if yielded_audio or attempt > 0:
                    raise

        raise BaiduSpeechError("百度 TTS 鉴权重试失败")

    async def _stream_tts_once(
        self,
        text: str,
        token: str,
        *,
        per: str | None = None,
    ) -> AsyncIterator[bytes]:
        """使用指定 Token 完成一次百度 TTS WebSocket 会话。

        交互顺序严格对应官方协议：
        ``system.start -> system.started -> text -> system.finish ->``
        ``binary... -> system.finished``。
        """

        # 本实现选择 OAuth 方式，只在 URL 放 access_token；没有再同时发送
        # Authorization Header，避免混用百度文档中二选一的两种鉴权方式。
        query = urllib.parse.urlencode(
            {
                "access_token": token,
                "per": per or self.config.baidu_tts_per,
            }
        )
        url = f"{self.config.baidu_tts_ws_url}?{query}"
        received_audio = False

        try:
            # 每个短句独立建立 TTS 会话。open/close 超时限制握手和关闭等待；
            # ping_interval/ping_timeout 负责底层连接保活；max_size 限制单个消息。
            async with websockets.connect(
                url,
                open_timeout=15,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
                max_size=2 * 1024 * 1024,
            ) as websocket:
                # ==================== 1. 初始化合成参数 ====================
                # aue=3 表示 MP3；audio_ctrl 以 JSON 字符串形式要求降采样到 16k，
                # 与硬件 CI 播放器和本项目协议约定一致。
                await websocket.send(
                    json.dumps(
                        {
                            "type": "system.start",
                            "payload": {
                                "spd": self.config.baidu_tts_speed,
                                "pit": self.config.baidu_tts_pitch,
                                "vol": self.config.baidu_tts_volume,
                                "aue": 3,
                                "audio_ctrl": json.dumps(
                                    {
                                        "sampling_rate": (
                                            self.config.baidu_tts_sample_rate
                                        )
                                    },
                                    separators=(",", ":"),
                                ),
                            },
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )

                # 不用固定 sleep 猜百度是否准备好，而是等待明确的 system.started。
                started_frame = await asyncio.wait_for(websocket.recv(), timeout=20)
                self._validate_tts_control_frame(
                    started_frame,
                    expected_type="system.started",
                )

                # ==================== 2. 发送本次短句 ====================
                await websocket.send(
                    json.dumps(
                        {"type": "text", "payload": {"text": text}},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                # 告知百度不会再发送文本，要求立即合成缓冲区剩余内容。缺少这个
                # 控制帧可能造成尾部文字丢失或等待超时。
                await websocket.send('{"type":"system.finish"}')

                # ==================== 3. 接收音频和完成状态 ====================
                while True:
                    frame = await asyncio.wait_for(websocket.recv(), timeout=30)
                    if isinstance(frame, bytes):
                        # 二进制消息是原始 MP3 字节，直接 yield 给硬件路由。
                        if frame:
                            received_audio = True
                            yield frame
                        continue

                    # 文本消息是 system.finished 或 system.error 等控制帧。
                    message = self._parse_tts_control_frame(frame)
                    message_type = str(message.get("type") or "")
                    code = int(message.get("code") or 0)
                    if message_type == "system.error" or code != 0:
                        if self._is_auth_error(message):
                            raise _BaiduAuthError(
                                f"百度 TTS 鉴权失败：{message.get('message') or code}"
                            )
                        raise BaiduSpeechError(
                            f"百度 TTS 合成失败：{message.get('message') or code}"
                        )
                    if message_type == "system.finished":
                        # 收到 finished 表示所有音频二进制帧都已发送完毕。
                        break
        except _BaiduAuthError:
            # 保留专用异常类型，让外层决定是否刷新 Token 重试。
            raise
        except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as exc:
            # 握手阶段 401/403 同样归类为鉴权错误；其他连接/超时错误直接上报，
            # 因为重新申请 Token 通常无法解决网络问题。
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {401, 403}:
                raise _BaiduAuthError(
                    f"百度 TTS 握手鉴权失败：HTTP {status_code}"
                ) from exc
            raise BaiduSpeechError(f"百度 TTS WebSocket 失败：{exc}") from exc

        if not received_audio:
            raise BaiduSpeechError("百度 TTS 没有返回音频")

    @staticmethod
    def _parse_tts_control_frame(frame: object) -> dict[str, object]:
        """将百度 TTS 文本控制帧解析为 JSON 对象并做基础类型校验。"""

        if not isinstance(frame, str):
            raise BaiduSpeechError("百度 TTS 控制帧格式错误")
        try:
            payload = json.loads(frame)
        except json.JSONDecodeError as exc:
            raise BaiduSpeechError("百度 TTS 返回了无效控制帧") from exc
        if not isinstance(payload, dict):
            raise BaiduSpeechError("百度 TTS 控制帧不是 JSON 对象")
        return payload

    def _validate_tts_control_frame(
        self,
        frame: object,
        *,
        expected_type: str,
    ) -> None:
        """校验初始化响应类型和 code，并保留鉴权错误的可重试语义。"""

        message = self._parse_tts_control_frame(frame)
        code = int(message.get("code") or 0)
        if self._is_auth_error(message):
            raise _BaiduAuthError(
                f"百度 TTS 鉴权失败：{message.get('message') or code}"
            )
        if str(message.get("type") or "") != expected_type or code != 0:
            raise BaiduSpeechError(
                f"百度 TTS 初始化失败：{message.get('message') or message}"
            )

    @staticmethod
    def _is_auth_error(message: dict[str, object]) -> bool:
        """兼容按错误码或错误文本表达的百度鉴权失败。"""

        code = int(message.get("code") or 0)
        detail = str(message.get("message") or "").lower()
        return code in AUTH_ERROR_CODES or any(
            marker in detail for marker in ("token", "auth", "unauthorized", "鉴权")
        )


# 进程级单例让所有硬件连接共享 Token 缓存和刷新锁，避免每台设备都单独向
# 百度 OAuth 申请 Token。它不保存单轮对话状态，因此可以被并发连接安全复用。
baidu_speech_client = BaiduSpeechClient(settings)
