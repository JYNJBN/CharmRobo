from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import settings


def create_access_token(user_id: int) -> tuple[str, int]:
    """ "
    生成token
     返回：
    1. JWT 字符串
    2. 有效期，单位为秒
    """

    expire_minutes = settings.jwt_expire_minutes
    now = datetime.now(timezone.utc)
    # 过期时间
    expire_time = now + timedelta(minutes=expire_minutes)
    payload = {
        # sub 表示这个 Token 属于谁。
        # JWT 标准建议 sub 使用字符串。
        "sub": str(user_id),
        # Token 签发时间。
        "iat": now,
        # Token 过期时间。
        "exp": expire_time,
        # 区分 access token 和以后可能添加的 refresh token。
        "type": "access",
    }
    token = jwt.encode(
        payload,
        key=settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    return token, expire_minutes * 60
