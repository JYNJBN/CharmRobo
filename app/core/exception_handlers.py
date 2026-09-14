import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.errors import BizError

logger = logging.getLogger(__name__)


async def http_exception_handler(
        request: Request,
        exc: HTTPException,
) -> JSONResponse:
    logger.debug("自定义 HTTPException 处理器已执行")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": str(exc.detail),
            "data": None,
        },
    )


async def biz_error_handler(request: Request, exc: BizError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.code,
        content={"code": exc.code, "message": exc.message, "data": None},
    )
