from datetime import datetime, timezone
from typing import Annotated, Generic, TypeVar

from pydantic import BaseModel, PlainSerializer

T = TypeVar("T")


def _to_utc_z(v: datetime) -> str:
    """把库里取出的时间渲染成带 Z 后缀的 ISO 8601 字符串。

    数据库列是 timestamp without time zone，取出来是 naive 值，但按约定它就是 UTC
    （写入侧 local_now() 与 func.now() 都写 UTC）。这里把被 replace(tzinfo=None)
    撕掉的时区标签贴回去，再输出 Z 后缀 —— 只贴标签、不换算数字。

    前端拿到带 Z 的串才能正确 new Date()，进而转换到用户所在时区。
    不带 Z 的串会被 ECMAScript 按「本地时区」解析，导致差 8 小时。
    """
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    else:
        v = v.astimezone(timezone.utc)
    return v.isoformat().replace("+00:00", "Z")


# 响应模型里所有对外的 datetime 字段都应该用它。
# 直接写 datetime 会序列化成不带时区标识的串，前端无从判断它是 UTC 还是本地时间。
UTCDateTime = Annotated[datetime, PlainSerializer(_to_utc_z, return_type=str)]


class ApiResponse(BaseModel, Generic[T]):
    """所有成功请求的统一返回格式。"""

    code: int = 0
    message: str = "success"
    data: T | None = None
