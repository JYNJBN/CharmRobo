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
from app.services.device_binding_service import create_bind_ticket
from app.services.device_service import bootstrap_device, get_devices_by_user_id

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