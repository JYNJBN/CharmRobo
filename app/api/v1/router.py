from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user_id
from app.api.v1.auth import router as auth_router
from app.api.v1.upload import router as upload_router
from app.api.v1.users import router as users_router
from app.api.v1.device import (
    hardware_device_router,
    user_device_router,
)
# 所有 v1 接口的统一前缀。
api_router = APIRouter(prefix="/api/v1")
# 公开路由，相当于白名单。
# 登录时不需要 Token。
api_router.include_router(auth_router)


# 受保护路由。
# users_router / upload_router 中的所有接口都必须通过 JWT 验证。
api_router.include_router(
    users_router,
    dependencies=[Depends(get_current_user_id)],
)

# 头像上传接口同样需要用户 JWT。
# 路径最终为：POST /api/v1/upload/avatar
api_router.include_router(
    upload_router,
    dependencies=[Depends(get_current_user_id)],
)

# create_ticket_api 内部有 CurrentUserId，
# 所以自身会执行用户 JWT 校验。
api_router.include_router(user_device_router)

# 硬件没有用户 JWT，不能增加 get_current_user_id。
api_router.include_router(hardware_device_router)
