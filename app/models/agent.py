from datetime import datetime

from sqlalchemy import BigInteger, String, text, Boolean, DateTime, func, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Agent(Base):
    __tablename__ = "agent"
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="Agent ID"
    )
    device_id: Mapped[int | None] = mapped_column(
        BigInteger,
        index=True,
        comment="所属设备 ID；系统模板（is_system=True）为 NULL",
    )
    name: Mapped[str] = mapped_column(
        String(32),
        default="小梦",
        nullable=False,
        server_default=text("'小梦'"),
        comment="agent_name"
    )
    system_prompt: Mapped[str | None] = mapped_column(
        String(500),
        comment="用户可编辑的人设，限 500 字"
    )
    model_key: Mapped[str] = mapped_column(
        String(32),
        default="doubao",
        nullable=False,
        server_default=text("'doubao'"),
        comment="默认 doubao，必须在 MODEL_REGISTRY 内"
    )
    voice: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="female_gentle",
        server_default=text("'female_gentle'"),
        comment="默认 female_gentle，存中立别名真实映射在后端"
    )
    is_system: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
        comment=("默认 False，系统模板为 True")
    )
    source_agent_id: Mapped[BigInteger | None] = mapped_column(
        BigInteger,
        comment="fork 自哪个模板"
    )
    create_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="创建时间"
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        onupdate=func.now(),
        comment="更新时间",
    )
    deleted: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
        comment="逻辑删除：0 未删除，1 已删除",
    )
