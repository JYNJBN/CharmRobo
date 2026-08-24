from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CreateTicketRequest(BaseModel):
    device_sn: str = Field(
        min_length=4,
        max_length=64,
        description="设备通过 BLE 返回的唯一编号",
    )


class CreateTicketResponse(BaseModel):
    bind_ticket: str
    expires_in: int


class BootstrapDeviceRequest(BaseModel):
    """
    硬件第一次初始化请求。

    device_secret 由硬件生成并先保存到 Flash/NVS，
    后端只计算摘要并保存到 MySQL。
    """

    device_sn: str = Field(
        min_length=4,
        max_length=64,
        description="设备sn",
    )
    device_secret: str = Field(
        min_length=32,
        max_length=255,
        description="硬件生成并已持久化保存的永久设备密钥",
    )
    bind_ticket: str = Field(
        min_length=20,
        max_length=255,
        description="一次性绑定码",
    )
    product_key: str = Field(
        min_length=1,
        max_length=64,
        description="产品型号或产品系列标识",
    )
    firmware_version: str | None = Field(
        default=None,
        max_length=32,
        description="固件版本",
    )
    hardware_version: str | None = Field(
        default=None,
        max_length=32,
        description="硬件版本",
    )


class BootstrapDeviceResponse(BaseModel):
    """硬件初始化和用户绑定结果。"""

    device_id: int
    device_sn: str
    bound: bool


class UserDeviceResponse(BaseModel):
    """用户查询设备信息"""

    # device字段
    device_id: int
    device_sn: str
    product_key: str
    firmware_version: str | None
    hardware_version: str | None
    status: int
    last_online_time: datetime | None

    # user_device字段
    alias: str | None
    role: Literal["owner", "member"]
    bind_time: datetime
