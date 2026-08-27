from fastapi import APIRouter, HTTPException, Request, status

from app.api.dependencies import CurrentUserId, DbSession
from app.schemas.common import ApiResponse
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.services.user_service import (
    create_user,
    get_user_by_id,
    update_user,
)

router = APIRouter(
    prefix="/users",
    tags=["用户"],
)


def _resolve_ip_location(client_host: str | None) -> str:
    """根据客户端 IP 简单解析属地（局域网/本地），后续可接入第三方 IP 库。"""
    if not client_host:
        return ""
    if client_host == "127.0.0.1" or client_host.startswith("::1"):
        return "本地"
    if client_host.startswith(("192.168.", "10.", "172.16.")):
        return "局域网"
    # 公网 IP 暂时返回空，后续接入 IP 解析服务
    return ""


@router.get("/info", response_model=ApiResponse[UserResponse])
async def get_user_info(
    current_user_id: CurrentUserId,
    db: DbSession,
    request: Request,
):
    """
    返回当前登录用户的信息
     current_user 不是前端传的参数，
    而是 get_current_user() 验证 JWT 后注入的。
    """
    user = await get_user_by_id(db, current_user_id)
    print(current_user_id,'current_user_id')
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    # IP 属地：数据库为空时从当前请求解析并回存
    if not user.ip_location:
        client_host = request.client.host if request.client else None
        user.ip_location = _resolve_ip_location(client_host)
        await db.commit()
        await db.refresh(user)
    return ApiResponse(data=user)


@router.patch("/info", response_model=ApiResponse[UserResponse])
async def update_my_info(
    current_user_id: CurrentUserId,
    data: UserUpdate,
    db: DbSession,
):
    """
    修改当前登录用户的资料。
    与 GET /info 一样，从 JWT 中拿当前用户 id，前端不需要传 user_id。
    只更新请求中实际传入的字段（nickname / avatar / phone / gender）。
    """
    user = await update_user(db, current_user_id, data)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    return ApiResponse(data=user)


@router.get(
    "/{user_id}",
    response_model=ApiResponse[UserResponse],
)
async def read_user(
    user_id: int,
    db: DbSession,
):
    """查询单个用户。最终地址：GET /api/v1/users/{user_id}"""

    user = await get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    return ApiResponse(data=user)


@router.post(
    "",
    response_model=ApiResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_user_api(
    data: UserCreate,
    db: DbSession,
):
    """创建用户。最终地址：POST /api/v1/users"""

    try:
        user = await create_user(db, data)
        return ApiResponse(data=user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.patch(
    "/{user_id}",
    response_model=ApiResponse[UserResponse],
)
async def update_user_api(
    user_id: int,
    data: UserUpdate,
    db: DbSession,
):
    """修改用户。最终地址：PATCH /api/v1/users/{user_id}"""

    user = await update_user(db, user_id, data)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    return ApiResponse(data=user)
