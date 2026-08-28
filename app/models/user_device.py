from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class UserDevice(Base):
    __tablename__ = "user_device"
    __table_args__ = (
        UniqueConstraint("user_id", "device_id", name="uq_user_device"),
        {"comment": "用户与设备绑定关系表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="绑定关系主键 ID",
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        index=True,
        nullable=False,
        comment="用户 ID",
    )
    device_id: Mapped[int] = mapped_column(
        BigInteger,
        index=True,
        nullable=False,
        comment="设备 ID",
    )
    alias: Mapped[str | None] = mapped_column(
        String(100),
        comment="用户设置的设备名称",
    )
    role: Mapped[str] = mapped_column(
        String(20),
        default="owner",
        server_default=text("'owner'"),
        nullable=False,
        comment="用户对设备的角色：owner 或 member",
    )
    bind_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
        comment="绑定时间",
    )
    unbind_time: Mapped[datetime | None] = mapped_column(
        DateTime,
        comment="解绑时间",
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="更新时间",
    )
    deleted: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
        comment="逻辑删除：0 未删除，1 已删除",
    )