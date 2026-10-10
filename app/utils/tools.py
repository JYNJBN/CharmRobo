from datetime import datetime, timedelta, timezone

BEIJING_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def build_current_time_context(now: datetime | None = None) -> str:
    """每轮提供北京时间基准，独立于服务器 TZ 和模型训练/历史日期。"""
    current = (now if now is not None else local_now()).astimezone(BEIJING_TIMEZONE)
    weekday = "一二三四五六日"[current.weekday()]
    return (
        "本轮当前时间（服务器实时提供，回答当前日期时以此为准）：\n"
        f"当前北京时间：{current:%Y-%m-%d %H:%M:%S}，时区 Asia/Shanghai（UTC+08:00）。\n"
        f"今天是{current.year}年{current.month}月{current.day}日，星期{weekday}。\n"
        "用户询问今天几号、现在日期或星期几时，直接依据上述时间回答。"
        "历史对话、长期记忆、知识库及网页中的旧日期不代表今天；"
        "不要从模型训练时间猜测当前日期，也不要用历史资料覆盖本轮时间基准。"
    )


def local_now() -> datetime:
    """返回当前时刻（带 UTC 时区信息）。

    列类型已改为 timestamptz，可以接受带时区的值，因此不再需要
    replace(tzinfo=None) 去适配 naive 列 —— 那一步会把「这是绝对时刻」
    的语义丢掉，是此前时区分歧的根源之一。保留 tzinfo 后，读出来即可直接
    序列化成带 +00:00 的字符串，前端能正确转换到用户所在时区。

    函数名保留（原义为「本地时间」），以免改动 5 处调用点。
    """
    return datetime.now(timezone.utc)
