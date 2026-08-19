from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exception_handlers import http_exception_handler
from app.core.redis import close_redis, redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期：

    yield 之前：应用启动时执行。
    yield 之后：应用停止时执行。
    """

    # 启动时检查 Redis 是否可以连接。
    await redis_client.ping()
    print("Redis 连接成功")

    try:
        yield
    finally:
        # FastAPI 停止时释放 Redis 连接池。
        await close_redis()
        print("Redis 连接已关闭")
# 确保上传目录存在（StaticFiles 要求目录必须已存在，否则启动报错）。
upload_dir = Path(settings.upload_dir)
upload_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Charming Device API",
    version="0.1.0",
    lifespan=lifespan,
)

print("当前代码已加载2")

app.include_router(health_router)
app.include_router(api_router)

# 上传文件的静态访问：/static/xxx → uploads/xxx
app.mount("/static", StaticFiles(directory=upload_dir), name="static")

app.add_exception_handler(
    HTTPException,
    http_exception_handler,
)
