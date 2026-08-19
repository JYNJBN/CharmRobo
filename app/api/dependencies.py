from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.redis import get_redis

#
# auto_error=False 表示没有 Token 时，由我们自己返回统一的错误格式。
bearer_scheme = HTTPBearer(auto_error=False)


# 数据库依赖的类型别名。
DbSession = Annotated[AsyncSession, Depends(get_db)]
# redis
RedisClient = Annotated[Redis, Depends(get_redis)]

async def get_current_user_id(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    db: DbSession,
) -> int:
    """
    获取当前登录用户。

    执行流程：
    1. 从 Authorization 请求头提取 Bearer Token
    2. 验证 JWT 签名和过期时间
    3. 从 JWT 的 sub 中读取用户 ID
    4. 查询数据库并返回 User 对象
    """

    # 没传 Authorization，或者格式不是 Bearer。
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="认证方式必须是 Bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # credentials.credentials 是不包含 Bearer 的 JWT 字符串。
    token = credentials.credentials

    try:
        payload = jwt.decode(
            jwt=token,
            key=settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )

        # 防止以后把 refresh token 当成 access token 使用。
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token 类型错误",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 创建 JWT 时，已经把 user_id 放进了 sub。
        subject = payload.get("sub")

        if subject is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token 缺少用户信息",
                headers={"WWW-Authenticate": "Bearer"},
            )

    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    except (jwt.InvalidTokenError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 无效",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    # 只获取用户id
    return int(payload["sub"])


# 以后接口只需要写： 用户id
# CurrentUserId: CurrentUserId
CurrentUserId = Annotated[int, Depends(get_current_user_id)]
