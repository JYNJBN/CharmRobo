from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    pass


DATABASE_URL = (
    f"mysql+aiomysql://"
    f"{settings.mysql_user}:{settings.mysql_password}"
    f"@{settings.mysql_host}:{settings.mysql_port}"
    f"/{settings.mysql_database}?charset=utf8mb4"
)
# 应用程序连接 MySQL 的总入口。
engine = create_async_engine(
    DATABASE_URL,
    echo=settings.debug,
)
# 它是一个 Session 工厂，用来创建数据库会话。
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 得到一个数据库会话。
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session