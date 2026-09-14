from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import (
    CurrentUserId,
    DbSession,
    RedisClient,
)
from app.schemas.common import ApiResponse
from app.schemas.device import (
    BootstrapDeviceRequest,
    BootstrapDeviceResponse,
    CreateTicketRequest,
    CreateTicketResponse,
    UserDeviceResponse,
)
from app.schemas.user import UpdateDeviceModelResponse, UpdateDeviceModelRequest
from app.services.device_binding_service import create_bind_ticket
from app.services.device_service import (
    bootstrap_device,
    get_devices_by_user_id,
    unbind_device_for_user,
    update_device_model_for_user,
)

# 小程序调用的接口，需要用户 JWT。
user_device_router = APIRouter(
    prefix="/device",
    tags=["用户设备"],
)

# 硬件调用的接口，不使用用户 JWT。
hardware_device_router = APIRouter(
    prefix="/device",
    tags=["硬件设备"],
)


@user_device_router.post(
    "/create/tickets",
    response_model=ApiResponse[CreateTicketResponse],
)
async def create_ticket_api(
        data: CreateTicketRequest,
        redis: RedisClient,
        current_user_id: CurrentUserId,
) -> ApiResponse[CreateTicketResponse]:
    """
    小程序申请一次性绑定码。
    """

    ticket = await create_bind_ticket(
        redis=redis,
        user_id=current_user_id,
        device_sn=data.device_sn,
    )

    return ApiResponse(
        data=CreateTicketResponse(
            bind_ticket=ticket,
            expires_in=300,
        )
    )


@hardware_device_router.post(
    "/bootstrap",
    response_model=ApiResponse[BootstrapDeviceResponse],
)
async def bootstrap_device_api(
        data: BootstrapDeviceRequest,
        db: DbSession,
        redis: RedisClient,
) -> ApiResponse[BootstrapDeviceResponse]:
    """
    硬件第一次初始化。

    这个接口不需要用户 JWT，
    用户身份来自 bind_ticket。
    """

    try:
        device = await bootstrap_device(
            db=db,
            redis=redis,
            data=data,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    return ApiResponse(
        data=BootstrapDeviceResponse(
            device_id=int(device.id),
            device_sn=device.device_sn,
            bound=True,
        )
    )


@user_device_router.get(
    "",
    response_model=ApiResponse[list[UserDeviceResponse]],
)
async def get_my_devices_api(
        db: DbSession,
        current_user_id: CurrentUserId,
) -> ApiResponse[list[UserDeviceResponse]]:
    """
    查询当前登录用户绑定的设备。

    用户 ID 从 JWT 获取，不允许前端自己传 user_id。
    """

    devices = await get_devices_by_user_id(
        db=db,
        user_id=current_user_id,
    )

    return ApiResponse(data=devices)


@user_device_router.delete(
    "/{device_id}",
    response_model=ApiResponse[None],
)
async def unbind_device_api(
        device_id: int,
        db: DbSession,
        current_user_id: CurrentUserId,
) -> ApiResponse[None]:
    """解除当前用户和设备的绑定，不删除硬件设备记录。"""

    try:
        await unbind_device_for_user(
            db=db,
            user_id=current_user_id,
            device_id=device_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return ApiResponse(message="设备已移除", data=None)


@user_device_router.patch(
    "/{device_id}/model",
    response_model=ApiResponse[UpdateDeviceModelResponse],
)
async def patch_device_model(
        db: DbSession,
        device_id: int,
        data: UpdateDeviceModelRequest,
        current_user_id: CurrentUserId,
) -> ApiResponse[UpdateDeviceModelResponse]:
    result = await update_device_model_for_user(
        db=db, user_id=current_user_id, device_id=device_id, model_key=data.model_key)
    return ApiResponse(data=result)
