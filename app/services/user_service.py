import logging

from sentry_sdk.integrations import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import create_access_token
from app.models.user import User
from app.schemas.user import LoginResponse, UserCreate, UserResponse, UserUpdate

logger = logging.getLogger(__name__)


async def get_user_by_id(
    db: AsyncSession,
    user_id: int,
) -> User | None:
    """根据 ID 查询一个未被逻辑删除的用户。"""

    result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.deleted == 0,
        )
    )
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    data: UserCreate,
) -> User:
    """创建用户，并提交事务。"""
    logger.debug("Pydantic 数据：%s", data)
    logger.debug("转换后的字典：%s", data.model_dump())
    # openid 是微信用户唯一标识，创建前先检查是否重复。
    result = await db.execute(select(User).where(User.openid == data.openid))
    if result.scalar_one_or_none() is not None:
        raise ValueError("openid 已经存在")
    logger.debug("data: %s", data)
    user = User(**data.model_dump())
    logger.debug("SQLAlchemy 对象：%s", user)
    logger.debug("user: %s", user)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def update_user(
    db: AsyncSession,
    user_id: int,
    data: UserUpdate,
) -> User | None:
    """修改用户，只更新请求中实际出现的字段。"""

    user = await get_user_by_id(db, user_id)
    if user is None:
        return None

    # exclude_unset=True 很重要：没有传的字段不会覆盖原来的值。
    update_data = data.model_dump(exclude_unset=True)
    for field_name, value in update_data.items():
        logger.debug("%s=%s", field_name, value)
        setattr(user, field_name, value)
    await db.commit()
    await db.refresh(user)
    return user


# 获取openid + session_key
async def get_wechat_session(code: str) -> dict[str, str]:
    url = "https://api.weixin.qq.com/sns/jscode2session"
    params = {
        "appid": settings.wechat_app_id,
        "secret": settings.wechat_app_secret.get_secret_value(),
        "js_code": code,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def login_service(db: AsyncSession, code: str) -> LoginResponse | None:
    """
    微信小程序登录流程：
    1. 前端 wx.login() 拿到 code
    2. 后端用 code 换 openid + session_key
    3. 根据 openid 查/建用户
    4. 生成 JWT 返回前端
    """
    session = await get_wechat_session(code)
    logger.debug("session: %s", session)
    openid = session.get("openid")
    # session_key = session.get('session_key')  # 安全取，key 不存在返回 None，不抛错
    logger.debug("openid: %s", openid)
    if not openid:
        error_message = session.get("errmsg", "微信登录失败")
        raise ValueError(error_message)
    result = await db.execute(
        select(User).where(User.openid == openid, User.deleted == 0)
    )
    user = result.scalar_one_or_none()
    # 数据库没有这个用户创建用户
    if user is None:
        user = User(openid=openid, nickname=f"用户{openid[-5:]}")
        db.add(user)
        await db.commit()
        await db.refresh(user)
    user_response = UserResponse.model_validate(user)
    logger.debug("user_response: %s", user_response)
    access_token, expires_in = create_access_token(user_response.id)

    return LoginResponse(
        access_token=access_token,
        expires_in=expires_in,
        token_type="bearer",
        user=user_response,
    )


async def get_wechat_access_token() -> str:
    """用 appid/secret 换取微信全局 access_token（有效期 7200 秒，生产建议缓存）。"""
    url = "https://api.weixin.qq.com/cgi-bin/token"
    params = {
        "appid": settings.wechat_app_id,
        "secret": settings.wechat_app_secret.get_secret_value(),
        "grant_type": "client_credential",
    }
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    access_token = data.get("access_token")
    if not access_token:
        raise ValueError(data.get("errmsg", "获取微信 access_token 失败"))
    return access_token


async def get_wechat_phone_number(phone_code: str) -> str:
    """用 getPhoneNumber 授权返回的动态令牌 code 换取手机号。"""
    access_token = await get_wechat_access_token()
    url = "https://api.weixin.qq.com/wxa/business/getuserphonenumber"
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.post(
            url,
            params={"access_token": access_token},
            json={"code": phone_code},
        )
        response.raise_for_status()
        data = response.json()
    if data.get("errcode") != 0:
        raise ValueError(data.get("errmsg", "获取手机号失败"))
    phone = (data.get("phone_info") or {}).get("purePhoneNumber")
    if not phone:
        raise ValueError("微信未返回手机号")
    return phone


async def phone_login_service(
    db: AsyncSession,
    login_code: str,
    phone_code: str,
) -> LoginResponse:
    """
    微信手机号快捷登录（注册）流程：
    1. login_code 换 openid（jscode2session）
    2. phone_code 换手机号（getuserphonenumber）
    3. 按 openid 查用户；没有则按 phone 查（老用户补绑 openid）；还没有则创建
    4. 生成 JWT 返回前端
    """
    # 1) 换 openid
    session = await get_wechat_session(login_code)
    openid = session.get("openid")
    if not openid:
        raise ValueError(session.get("errmsg", "微信登录失败"))

    # 2) 换手机号
    phone = await get_wechat_phone_number(phone_code)

    # 3) 查/建用户
    result = await db.execute(
        select(User).where(User.openid == openid, User.deleted == 0)
    )
    user = result.scalar_one_or_none()
    # 如果查询到用户存在并且手机号不一致的时候更新手机号
    if user is not None and user.phone != phone:
        user.phone = phone
        await db.commit()
        await db.refresh(user)

    if user is None:
        # 手机号已注册过的老用户：补绑 openid，下次可继续用
        result = await db.execute(
            select(User).where(User.phone == phone, User.deleted == 0)
        )
        user = result.scalar_one_or_none()
        if user is not None:
            user.openid = openid
            await db.commit()
            await db.refresh(user)

    if user is None:
        # 全新用户：用手机号注册
        user = User(
            openid=openid,
            phone=phone,
            nickname=f"用户{phone[-4:]}",
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    # 4) 发 JWT
    user_response = UserResponse.model_validate(user)
    access_token, expires_in = create_access_token(user_response.id)
    return LoginResponse(
        access_token=access_token,
        expires_in=expires_in,
        token_type="bearer",
        user=user_response,
    )
