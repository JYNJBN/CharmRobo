import logging
from contextlib import asynccontextmanager
from logging.config import dictConfig
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.api.v1.hardware_voice import router as hardware_voice_router
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.errors import BizError
from app.core.exception_handlers import biz_error_handler, http_exception_handler
from app.core.redis import close_redis, redis_client


def setup_logging(level: str) -> None:
    """
    把业务日志输出到 stdout，让 docker logs 能看见。

    不加这段的话，logging.getLogger(__name__) 会继承 root logger，
    而 root 默认是 WARNING 级别，所有 logger.info() 都会被静默丢弃。

    两个关键点：
    1. disable_existing_loggers 必须是 False。它默认是 True，
       会把 uvicorn 已经配好的 uvicorn.error / uvicorn.access 一起禁用，
       结果启动日志和访问日志反而没了。
    2. handler 只能指向 stdout 或 stderr。Docker 只捕获这两个流，
       写进文件的话 docker logs 看不到，只能进容器里翻。
    """
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)-8s %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "default",
                },
            },
            # SQLAlchemy/httpx 的底层调试输出默认是英文且非常密集；
            # 业务排查以各模块的中文 logger 为主。需要查 SQL 或 HTTP 细节时，
            # 再临时把这些 logger 调回 DEBUG/INFO。
            "loggers": {
                "sqlalchemy.engine": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "sqlalchemy.pool": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "httpx": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "httpcore": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "httpx2": {
                    "level": "WARNING",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
            "root": {"level": level, "handlers": ["console"]},
        }
    )


setup_logging(settings.log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期：

    yield 之前：应用启动时执行。
    yield 之后：应用停止时执行。
    """

    # 启动时检查 Redis 是否可以连接。
    await redis_client.ping()
    logger.info("Redis 连接成功")

    try:
        yield
    finally:
        # FastAPI 停止时释放 Redis 连接池。
        await close_redis()
        logger.info("Redis 连接已关闭")


# 确保上传目录存在（StaticFiles 要求目录必须已存在，否则启动报错）。
upload_dir = Path(settings.upload_dir)
upload_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Charming Device API",
    version="0.1.0",
    lifespan=lifespan,
)

logger.debug("当前代码已加载2")

app.include_router(health_router)
app.include_router(api_router)
# 硬件路由故意单独注册在应用根级别，使最终地址保持为 /v1/dialogue/ws；
# 原 api_router 的 /api/v1/voice/stream 继续服务小程序，两者不会互相覆盖。
app.include_router(hardware_voice_router)

# 上传文件的静态访问：/static/xxx → uploads/xxx
app.mount("/static", StaticFiles(directory=upload_dir), name="static")

app.add_exception_handler(
    HTTPException,
    http_exception_handler,
)
app.add_exception_handler(
    BizError,
    biz_error_handler
)  # ← 新增
