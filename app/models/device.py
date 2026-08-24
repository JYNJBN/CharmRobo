from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, SmallInteger, String, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

""""设备表"""
class Device(Base):
    __tablename__ = "device"
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="设备主键 ID",
    )
    device_sn: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
        comment="设备唯一编号",
    )
    product_key: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="产品型号或产品系列标识",
    )
    firmware_version: Mapped[str | None] = mapped_column(
        String(32),
        comment="固件版本",
    )
    hardware_version: Mapped[str | None] = mapped_column(
        String(32),
        comment="硬件版本",
    )

    # 数据库只保存摘要，不保存设备的原始密钥。
    device_secret_hash: Mapped[str | None] = mapped_column(
        String(255),
        comment="设备访问密钥摘要，不保存原始密钥",
    )
    secret_version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        server_default=text("1"),
        nullable=False,
        comment="设备密钥版本",
    )
    secret_updated_time: Mapped[datetime | None] = mapped_column(
        DateTime,
        comment="设备密钥最后更新时间",
    )
    status: Mapped[int] = mapped_column(
        SmallInteger,
        default=1,
        server_default=text("1"),
        nullable=False,
        comment="设备状态：1 正常，0 禁用",
    )
    last_online_time: Mapped[datetime | None] = mapped_column(
        DateTime,
        comment="最近一次上线时间",
    )

    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
        comment="创建时间",
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