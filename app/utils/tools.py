from datetime import datetime, timezone
def local_now() -> datetime:
    """生成与 MySQL DATETIME 兼容的本地无时区时间。"""
    return datetime.now(timezone.utc).astimezone().replace(tzinfo=None)