"""指定设备的冰淇淋 HTTP 动作；受理后独立查询，不依赖语音连接存活。"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
from collections.abc import AsyncIterator
from time import perf_counter

import httpx

from app.core.config import Settings, settings
from app.services.hardware_actions import ActionResult

logger = logging.getLogger(__name__)
_WORKFLOW = "make-icecream"
_ORDER = re.compile(
    r"(?:请|麻烦你?|劳驾)?(?:能不能|可以|能)?"
    r"(?:帮我下单|给我下单|帮我订|给我订|帮我点|给我点|帮我买|给我买|"
    r"帮我制作|给我制作|帮我做|给我做|帮我来|给我来|给我|我要|我想要|"
    r"我想买|我想吃|来上|来|下单|点|买|制作|做)"
    r"(?:一下)?(?P<count>[零一二两三四五六七八九十百\d]+)"
    r"(?:杯|个|份)(?:冰淇淋|冰激凌)(?:吧|呀|啊|好吗|可以吗|谢谢|谢谢你)?"
)
_STATUS = {
    "冰淇淋做好了吗",
    "冰淇淋好了没",
    "冰淇淋好了吗",
    "我的冰淇淋做好了吗",
    "我的冰淇淋好了吗",
    "我的冰淇淋好了没",
    "查一下冰淇淋状态",
    "查询冰淇淋状态",
}


def match_icecream_intent(text: str) -> str | None:
    # 整句匹配：不把“不要/如果/他说/多少钱”以及历史引用当成制作指令。
    compact = re.sub(r"[\s，。！？、,.!?]", "", text or "")
    if compact in _STATUS:
        return "status"
    match = _ORDER.fullmatch(compact)
    if match is None:
        return None
    return "order" if match["count"] in {"一", "1"} else "quantity"


class IcecreamProtocolError(RuntimeError):
    pass


class IcecreamService:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self._token: str | None = None
        self._lock = asyncio.Lock()
        self._state = "idle"
        self._baseline_ids: set[str] = set()
        self._instance_id: str | None = None
        self._monitor: asyncio.Task | None = None
        self._updates: asyncio.Queue[ActionResult] | None = None

    def _result(self, status: str, text: str) -> ActionResult:
        return ActionResult(
            "icecream",
            text,
            action_payload={
                "action": "icecream",
                "status": status,
            },
        )

    @property
    def _base_url(self) -> str:
        address = ipaddress.ip_address(self.config.icecream_ip.strip())
        host = f"[{address}]" if address.version == 6 else str(address)
        return f"http://{host}:{self.config.icecream_port}"

    async def _login(self, client: httpx.AsyncClient) -> None:
        started = perf_counter()
        response = await client.post(
            "/api/auth/login",
            json={
                "username": self.config.icecream_username,
                "password": self.config.icecream_password.get_secret_value(),
            },
            timeout=self.config.icecream_query_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("token")
        if (
            payload.get("success") is not True
            or not isinstance(token, str)
            or not token
        ):
            raise IcecreamProtocolError("登录响应缺少 token")
        self._token = token
        logger.info(
            "[ICECREAM][登录] 完成 耗时=%dms", int((perf_counter() - started) * 1000)
        )

    async def _request(
        self, client: httpx.AsyncClient, method: str, path: str
    ) -> httpx.Response:
        started = perf_counter()
        if not self._token:
            await self._login(client)
        kwargs = {"timeout": self.config.icecream_query_timeout_seconds}
        if method == "POST":
            kwargs.update(json={}, timeout=self.config.icecream_execute_timeout_seconds)
        response = await client.request(
            method,
            path,
            headers={
                "Authorization": f"Bearer {self._token}",
            },
            **kwargs,
        )
        # 只有明确 401 才重登重发。执行超时/断连不能确定是否受理，绝不自动重下单。
        if response.status_code == 401:
            self._token = None
            await self._login(client)
            response = await client.request(
                method,
                path,
                headers={
                    "Authorization": f"Bearer {self._token}",
                },
                **kwargs,
            )
        logger.info(
            "[ICECREAM][HTTP] %s %s status=%s 耗时=%dms",
            method,
            path,
            response.status_code,
            int((perf_counter() - started) * 1000),
        )
        return response

    async def _instances(self, client: httpx.AsyncClient) -> dict[str, dict]:
        response = await self._request(client, "GET", "/api/workflows-json/instances")
        response.raise_for_status()
        payload = response.json()
        items = payload.get("instances")
        if payload.get("success") is not True or not isinstance(items, dict):
            raise IcecreamProtocolError("状态响应不符合接口约定")
        if any(not isinstance(item, dict) for item in items.values()):
            raise IcecreamProtocolError("状态记录格式无效")
        if any(
            item.get("status") not in {"RUNNING", "COMPLETED", "FAILED"}
            or not isinstance(item.get("workflowId"), str)
            for item in items.values()
        ):
            raise IcecreamProtocolError("状态记录缺少有效工作流或状态")
        return items

    def _observe(self, items: dict[str, dict], trace: str) -> None:
        if self._state not in {"accepted", "running", "unknown"}:
            return
        if self._instance_id is None:
            candidates = [
                (key, item)
                for key, item in items.items()
                if key not in self._baseline_ids and item.get("workflowId") == _WORKFLOW
            ]
            if not candidates:
                return  # 不能把前一杯的 COMPLETED 误当作本次完成。
            self._instance_id = max(
                candidates, key=lambda entry: str(entry[1].get("startedAt") or "")
            )[0]
        previous = self._state
        item = items.get(self._instance_id, {})
        status = item.get("status")
        if status == "RUNNING":
            self._state = "running"
        elif status in {"COMPLETED", "FAILED"}:
            self._state = "completed" if status == "COMPLETED" else "failed"
            logger.info(
                "[ICECREAM][%s][制作结束] instance=%s status=%s",
                trace,
                self._instance_id,
                status,
            )
        if self._state != previous and self._updates is not None:
            messages = {
                "running": "冰淇淋正在制作，请稍等。",
                "completed": "你的冰淇淋已制作完成，请取走冰淇淋。",
                "failed": "这杯冰淇淋制作失败，请联系工作人员检查机器。",
            }
            if self._state in messages:
                self._updates.put_nowait(
                    self._result(self._state, messages[self._state])
                )

    async def resolve(
        self, text: str, device_sn: str, trace: str
    ) -> ActionResult | None:
        # 先校验本轮实际加载的设备 SN，其他设备不匹配、不登录、不访问机器。
        if (
            not self.config.icecream_device_sn
            or device_sn != self.config.icecream_device_sn
        ):
            return None
        intent = match_icecream_intent(text)
        if intent is None:
            return None
        if intent == "quantity":
            return self._result(
                "quantity", "目前一次只能制作一杯，请说帮我下单一杯冰淇淋。"
            )
        if (
            not self.config.icecream_enabled
            or not self.config.icecream_ip.strip()
            or not self.config.icecream_password
        ):
            return self._result(
                "not_configured", "冰淇淋下单功能还没配置好，请先联系工作人员。"
            )
        try:
            base_url = self._base_url
        except ValueError:
            return self._result(
                "not_configured", "冰淇淋机地址配置有误，请联系工作人员。"
            )
        if self._lock.locked():
            return self._result("busy", "正在处理冰淇淋请求，请稍等，不用重复下单。")
        started = perf_counter()
        async with self._lock:
            try:
                # 内网地址直接连接，不继承宿主机代理环境变量，不跟随重定向泄露 token。
                async with httpx.AsyncClient(
                    base_url=base_url, trust_env=False
                ) as client:
                    items = await self._instances(client)
                    self._observe(items, trace)
                    if intent == "status":
                        return self._status_result(items)
                    if self._state in {"accepted", "running", "unknown"}:
                        return self._result(
                            "busy",
                            "上一杯的制作结果还在确认中，请先等待，不用重复下单。",
                        )
                    if any(item.get("status") == "RUNNING" for item in items.values()):
                        return self._result(
                            "busy", "冰淇淋机正在制作，请等当前制作结束后再下单。"
                        )
                    self._baseline_ids = set(items)
                    self._instance_id = None
                    try:
                        response = await self._request(
                            client, "POST", f"/api/workflows-json/{_WORKFLOW}/execute"
                        )
                        payload = response.json()
                        if not isinstance(payload, dict):
                            raise TypeError("制作响应不是 JSON 对象")
                    except asyncio.CancelledError:
                        self._state = "unknown"
                        self._start_monitor(trace)
                        logger.warning(
                            "[ICECREAM][%s][语音取消] 继续确认制作结果，不重复执行",
                            trace,
                        )
                        raise
                    except (httpx.RequestError, ValueError, TypeError):
                        self._state = "unknown"
                        self._start_monitor(trace)
                        logger.warning(
                            "[ICECREAM][%s][受理待确认] 不自动重发制作请求", trace
                        )
                        return self._following_result(
                            "unknown",
                            "下单结果暂时无法确认，我会继续查询，请不要重复下单。",
                        )
                    if response.status_code == 400 and payload.get("success") is False:
                        reason = payload.get("failReason")
                        logger.warning(
                            "[ICECREAM][%s][未受理] reason=%s", trace, reason
                        )
                        if reason == "PARALLELISM_REJECT":
                            return self._result(
                                "busy", "冰淇淋机正在制作，请稍后再试。"
                            )
                        return self._result(
                            "rejected", "冰淇淋机没有受理这次制作，请联系工作人员。"
                        )
                    if response.status_code == 401:
                        return self._result(
                            "error", "冰淇淋机登录失败，请联系工作人员。"
                        )
                    if (
                        response.status_code != 200
                        or payload.get("success") is not True
                        or payload.get("workflowId") != _WORKFLOW
                    ):
                        self._state = "unknown"
                        self._start_monitor(trace)
                        return self._following_result(
                            "unknown", "下单结果暂时无法确认，请不要重复下单。"
                        )
                    self._state = "accepted"
                    self._start_monitor(trace)
                    return self._following_result(
                        "accepted",
                        "已为你下单一杯冰淇淋，机器已受理，正在确认制作状态。",
                    )
            except (
                httpx.HTTPError,
                IcecreamProtocolError,
                ValueError,
                TypeError,
                AttributeError,
            ) as exc:
                logger.warning(
                    "[ICECREAM][%s][接口失败] 类型=%s", trace, type(exc).__name__
                )
                return self._result(
                    "error", "暂时无法连接冰淇淋机，请联系工作人员检查网络或登录配置。"
                )
            finally:
                logger.info(
                    "[ICECREAM][%s][请求完成] intent=%s state=%s 耗时=%dms",
                    trace,
                    intent,
                    self._state,
                    int((perf_counter() - started) * 1000),
                )

    def _status_result(self, items: dict[str, dict]) -> ActionResult:
        if self._state in {"accepted", "running", "unknown"}:
            return self._result(self._state, "冰淇淋正在制作或等待状态确认，请稍等。")
        candidates = [
            item for item in items.values() if item.get("workflowId") == _WORKFLOW
        ]
        if not candidates:
            return self._result("idle", "暂时没有查到冰淇淋制作记录。")
        latest = max(candidates, key=lambda item: str(item.get("startedAt") or ""))
        status = latest.get("status")
        if status == "COMPLETED":
            return self._result("completed", "最新一杯冰淇淋已制作完成，请取走冰淇淋。")
        if status == "FAILED":
            return self._result(
                "failed", "最新一杯冰淇淋制作失败，请联系工作人员检查机器。"
            )
        return self._result("running", "冰淇淋正在制作，请稍等。")

    def _start_monitor(self, trace: str) -> None:
        if self._monitor is None or self._monitor.done():
            self._updates = asyncio.Queue()
            self._monitor = asyncio.create_task(self._monitor_status(trace))

    def _following_result(self, status: str, text: str) -> ActionResult:
        return ActionResult(
            "icecream",
            text,
            action_payload={
                "action": "icecream",
                "status": status,
            },
            status_updates=self._updates,
        )

    async def _monitor_status(self, trace: str) -> None:
        deadline = (
            asyncio.get_running_loop().time()
            + self.config.icecream_monitor_timeout_seconds
        )
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url, trust_env=False
            ) as client:
                while self._state in {"accepted", "running", "unknown"}:
                    if asyncio.get_running_loop().time() >= deadline:
                        self._state = "unknown"
                        logger.warning(
                            "[ICECREAM][%s][状态确认超时] 保留未确认状态，不自动重下单",
                            trace,
                        )
                        if self._updates is not None:
                            self._updates.put_nowait(
                                self._result(
                                    "unknown",
                                    "制作结果还没确认，请联系工作人员查看机器，不要重复下单。",
                                )
                            )
                        return
                    try:
                        async with self._lock:
                            self._observe(await self._instances(client), trace)
                    except (
                        httpx.HTTPError,
                        IcecreamProtocolError,
                        ValueError,
                        TypeError,
                        AttributeError,
                    ) as exc:
                        logger.warning(
                            "[ICECREAM][%s][查询失败] 类型=%s",
                            trace,
                            type(exc).__name__,
                        )
                    if self._state in {"accepted", "running", "unknown"}:
                        await asyncio.sleep(self.config.icecream_poll_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- 后台异常也必须结束语音等待，且不能重下单。
            self._state = "unknown"
            logger.error("[ICECREAM][%s][监控异常] 类型=%s", trace, type(exc).__name__)
            if self._updates is not None:
                self._updates.put_nowait(
                    self._result(
                        "unknown",
                        "制作状态暂时无法确认，请联系工作人员查看机器，不要重复下单。",
                    )
                )

    async def aclose(self) -> None:
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
            self._monitor = None
        self._token = None


async def iter_icecream_status_updates(
    action: ActionResult,
) -> AsyncIterator[ActionResult]:
    """只消费本次下单的队列，不把后一杯的状态播报到前一轮。"""
    queue = action.status_updates
    if queue is None:
        return
    while True:
        update = await queue.get()
        yield update
        if (update.action_payload or {}).get("status") in {
            "completed",
            "failed",
            "unknown",
        }:
            return


icecream_service = IcecreamService()
