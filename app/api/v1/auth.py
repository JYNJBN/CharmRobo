from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import DbSession
from app.schemas.common import ApiResponse
from app.schemas.user import LoginRequest, LoginResponse, PhoneLoginRequest
from app.services.user_service import login_service, phone_login_service

router = APIRouter(
    prefix="/auth",
    tags=["认证"],
)


@router.post(
    "/login",
    response_model=ApiResponse[LoginResponse],
)
async def login_api(
    data: LoginRequest,
    db: DbSession,
):
    """
    微信小程序登录。

    这是公开接口，因为用户此时还没有 Token。
    """

    try:
        result = await login_service(db, data.code)
        return ApiResponse(
            message="登录成功",
            data=result,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post(
    "/login/phone",
    response_model=ApiResponse[LoginResponse],
)
async def phone_login_api(
    data: PhoneLoginRequest,
    db: DbSession,
):
    """
    微信手机号快捷登录（注册）。

    前端 button open-type=getPhoneNumber 授权拿 phone_code，
    再配合 wx.login() 的 login_code，换取手机号完成注册/登录。
    公开接口。
    """

    try:
        result = await phone_login_service(db, data.login_code, data.phone_code)
        return ApiResponse(
            message="登录成功",
            data=result,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
