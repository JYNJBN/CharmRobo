from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Conversation(Base):
    """硬件设备与用户的一次连续对话会话。"""

    __tablename__ = "conversation"
    __table_args__ = {
        "comment": "AI 对话会话表",
    }

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="会话主键 ID",
    )
    user_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        index=True,
        comment="所属用户 ID；设备认证完成前可为空",
    )
    device_id: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
        index=True,
        comment="发起对话的设备 ID；设备认证完成前可为空",
    )

    title: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="会话标题；第一版可暂不生成",
    )
    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="已压缩的历史对话摘要；第一版暂不使用",
    )

    message_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text("0"),
        nullable=False,
        comment="会话消息总数",
    )

    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        comment="最后一条消息创建时间",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
        comment="会话创建时间",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="会话最后更新时间",
    )
