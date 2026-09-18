from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """所有成功请求的统一返回格式。"""

    code: int = 0
    message: str = "success"
    data: T | None = None
