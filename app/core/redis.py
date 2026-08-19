from typing import AsyncIterator

from redis.asyncio import Redis

from app.core.config import settings


def get_redis_password()->str | None:
    """
    SecretStr 不用直接当普通字符串使用
    要通过get_secret_value获取真实值
    """
    if settings.redis_password is None:
        return None
    return settings.redis_password.get_secret_value()
# Redis 客户端内部自带连接池。
#
# 创建这个对象不会立刻连接 Redis，
# 第一次执行 ping、get、set 等命令时才真正建立连接。
redis_client = Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    password=get_redis_password(),
    db=settings.redis_db,

    # Redis 返回的 bytes 自动转成 str。
    # 例如不设置时 get() 可能返回 b"hello"，
    # 设置后直接返回 "hello"。
    decode_responses=True,

    # 连接 Redis 最多等待 3 秒。
    socket_connect_timeout=3,

    # 一个 Redis 命令最多等待 3 秒。
    socket_timeout=3,

    # 定期检查连接是否仍然可用。
    health_check_interval=30,
)

async def get_redis() -> AsyncIterator[Redis]:
    """
    FastAPI Redis 依赖。

    这里不能在每个请求结束后关闭 redis_client，
    因为整个应用应该共用同一个 Redis 连接池。
    """
    yield redis_client

async def close_redis() -> None:
    """
    FastAPI 停止时关闭 Redis 连接池。
    """
    await redis_client.aclose()