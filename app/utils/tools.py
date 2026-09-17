from datetime import datetime, timezone


def local_now() -> datetime:
    """生成与数据库兼容的无时区 UTC 时间（不随容器时区变化）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)
