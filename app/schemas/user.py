from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import UTCDateTime


class UserCreate(BaseModel):
    """创建用户时，前端允许提交的字段。"""

    openid: str
    nickname: str | None = None
    avatar: str | None = None
    phone: str | None = None


class UserUpdate(BaseModel):
    """修改用户时，所有字段都可选，只更新实际传入的字段。"""

    nickname: str | None = None
    avatar: str | None = None
    phone: str | None = None
    gender: int | None = Field(
        default=None,
        ge=0,
        le=2,
        description="性别：0 未知，1 男，2 女",
    )
    birthday: date | None = None


class UserResponse(BaseModel):
    """接口返回给前端的用户数据。"""

    id: int
    openid: str
    nickname: str | None
    avatar: str | None
    phone: str | None
    gender: int = 0
    birthday: date | None
    ip_location: str | None
    create_time: UTCDateTime
    update_time: UTCDateTime

    # 允许 Pydantic 直接从 SQLAlchemy User 对象读取属性。
    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    code: str = Field(
        ..., min_length=8, description="微信登录code必传为，为null其他登录方式"
    )


class PhoneLoginRequest(BaseModel):
    """微信手机号快捷登录（注册）。"""

    login_code: str = Field(
        ...,
        min_length=8,
        description="wx.login() 拿到的临时登录凭证 code",
    )
    phone_code: str = Field(
        ...,
        min_length=1,
        description="button open-type=getPhoneNumber 授权后返回的动态令牌 code",
    )


class LoginResponse(BaseModel):
    """登录接口返回值"""

    token_type: str = "bearer"
    access_token: str | None
    expires_in: int
    user: UserResponse

    # 打印方法
    def __repr__(self) -> dict[str, str | None | int]:
        return {
            "token_type": self.token_type,
            "access_token": self.access_token,
            "expires_in": self.expires_in,
            "user": self.user.nickname,
        }


class UpdateDeviceModelRequest(BaseModel):
    model_key: str = Field(
        min_length=1,
        max_length=128,
        description="火山方舟模型 ID 或 Endpoint ID",
    )


class UpdateDeviceModelResponse(BaseModel):
    device_id: int
    model_key: str
