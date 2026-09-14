from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["健康检查"])


@router.get("")
async def health_check():
    return {"status": "ok"}
