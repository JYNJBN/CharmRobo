from datetime import datetime, date

from sqlalchemy import BigInteger, Date, DateTime, Integer, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

"""用户表"""


class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    # 微信用户唯一标识
    openid: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )
    nickname: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    avatar: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    phone: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    # 性别：0 未知，1 男，2 女
    gender: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
    )

    birthday: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    ip_location: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    # 插入时自动生成
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    # 插入时生成，更新时自动修改
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # 逻辑删除：0 未删除，1 已删除
    deleted: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
    )
