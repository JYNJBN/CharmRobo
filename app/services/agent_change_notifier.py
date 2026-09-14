"""Agent 配置变更通知。

当前项目以单个 FastAPI 进程运行时，使用进程内订阅即可把手机端的 Agent
切换/编辑通知给正在运行的 Seeduplex 小程序会话。通知只携带设备和 Agent
ID，不携带人设、音色密钥或对话内容；会话收到通知后重新读取数据库配置。

后续硬件端改为 Seeduplex 长连接时，可以复用这里的通知入口：保持设备到
FastAPI 的连接，只重建 FastAPI 到 Seeduplex 的上游连接。多 worker/多实例
部署时，再把本模块的发布实现替换为 Redis Pub/Sub 即可。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

AgentChangeCallback = Callable[[dict[str, object]], Awaitable[None]]
# 同一 FastAPI 进程内，按 device_id 保存当前在线端到端会话的回调。
# 配置 API 提交成功后，通过这个表把变更推送给对应 WebSocket。
_listeners: dict[int, set[AgentChangeCallback]] = defaultdict(set)


async def register_agent_change_listener(
        device_id: int,
        callback: AgentChangeCallback,
) -> Callable[[], Awaitable[None]]:
    """注册某台设备的配置变更监听器，并返回异步注销函数。"""

    _listeners[int(device_id)].add(callback)
    logger.debug(
        "[Agent通知] 已注册监听 device_id=%s 当前监听数=%d",
        device_id,
        len(_listeners[int(device_id)]),
    )

    async def unregister() -> None:
        listeners = _listeners.get(int(device_id))
        if listeners is None:
            return
        listeners.discard(callback)
        if not listeners:
            _listeners.pop(int(device_id), None)
        logger.debug(
            "[Agent通知] 已注销监听 device_id=%s",
            device_id,
        )

    return unregister


async def notify_agent_config_changed(
        *,
        device_id: int,
        agent_id: int | None,
        reason: str,
) -> None:
    """通知当前设备的活跃语音会话重新加载 Agent 配置。

    这里只发送“配置变了”的轻量事件，不把 system_prompt、音色或历史文本
    放进事件。收到通知的会话会主动重连，重新从数据库读取完整配置。
    """

    event: dict[str, object] = {
        "type": "agent_config_changed",
        "device_id": int(device_id),
        "agent_id": int(agent_id) if agent_id is not None else None,
        "reason": reason,
    }
    listeners = tuple(_listeners.get(int(device_id), ()))
    logger.info(
        "[Agent通知] 发布配置变化 device_id=%s agent_id=%s 原因=%s "
        "监听数=%d",
        device_id,
        agent_id,
        reason,
        len(listeners),
    )

    for callback in listeners:
        try:
            await callback(event)
        except Exception:
            logger.exception(
                "[Agent通知] 回调执行失败 device_id=%s",
                device_id,
            )
