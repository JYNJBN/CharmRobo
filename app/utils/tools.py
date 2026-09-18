from datetime import datetime, timezone


def local_now() -> datetime:
    """返回当前时刻（带 UTC 时区信息）。

    列类型已改为 timestamptz，可以接受带时区的值，因此不再需要
    replace(tzinfo=None) 去适配 naive 列 —— 那一步会把「这是绝对时刻」
    的语义丢掉，是此前时区分歧的根源之一。保留 tzinfo 后，读出来即可直接
    序列化成带 +00:00 的字符串，前端能正确转换到用户所在时区。

    函数名保留（原义为「本地时间」），以免改动 5 处调用点。
    """
    return datetime.now(timezone.utc)
