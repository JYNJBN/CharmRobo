import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Integer,
    String,
    Text,
    func,
    text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Conversation(Base):
    """硬件设备与用户的一次连续对话会话。"""

    __tablename__ = "conversation"
    __table_args__ = (
        # 联合唯一索引
        UniqueConstraint("agent_id", "session_id", name="uq_conversation_agent_session"),
        {"comment": "Ai对话表"}
    )
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="会话主键 ID",
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        default=lambda: str(uuid.uuid4()),  # 新行自动生成 UUID
        comment="对外会话 ID（UUID）；预留扩展口，老数据迁移时也填 UUID",
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
    agent_id: Mapped[int | None] = mapped_column(
        BigInteger,
        index=True,
        comment="agent id"
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
        DateTime(timezone=True),
        nullable=True,
        comment="最后一条消息创建时间",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="会话创建时间",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="会话最后更新时间",
    )
