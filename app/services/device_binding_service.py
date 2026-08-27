import json
import logging
import secrets
from typing import Any

from redis.asyncio import Redis

BIND_TICKET_EXPIRE_SECONDS = 300
logger = logging.getLogger(__name__)


async def create_bind_ticket(
    redis: Redis,
    user_id: int,
    device_sn: str,
) -> str:
    """
    为当前用户和设备生成一次性绑定凭证。
    """
    # 生成无法猜测的随机数。
    ticket = secrets.token_urlsafe(32)

    # redis key
    redis_key = f"device:bind:{ticket}"
    # redis value
    ticket_data = {
        "user_id": user_id,
        "device_sn": device_sn,
    }
    await redis.set(
        redis_key,
        # redis 一般存字符串 字节 数字等类型 这里是将字典转成json字符串类型
        json.dumps(ticket_data),
        ex=BIND_TICKET_EXPIRE_SECONDS,
    )
    logger.info(
        "创建设备绑定码成功 user_id=%s device_sn=%s expires_in=%s",
        user_id,
        device_sn,
        BIND_TICKET_EXPIRE_SECONDS,
    )
    return ticket


async def get_bind_ticket(
    redis: Redis,
    ticket: str,
) -> dict[str, Any]:
    """
    读取绑定凭证，但暂时不删除。

    只有 MySQL 事务提交成功后才删除，避免数据库失败时
    绑定码已经丢失。
    """

    redis_key = f"device:bind:{ticket}"
    ticket_json = await redis.get(redis_key)

    if ticket_json is None:
        logger.warning("设备绑定码不存在或已过期")
        raise ValueError("绑定凭证不存在、已经使用或已经过期")

    return json.loads(ticket_json)


async def delete_bind_ticket(
    redis: Redis,
    ticket: str,
) -> None:
    """设备和用户关系提交成功后删除一次性绑定码。"""

    deleted_count = await redis.delete(f"device:bind:{ticket}")
    logger.info("删除设备绑定码完成 deleted_count=%s", deleted_count)
