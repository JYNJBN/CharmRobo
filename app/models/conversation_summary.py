from datetime import datetime

from sqlalchemy import BigInteger, DateTime, String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ConversationSummary(Base):
    """ "会话历史摘要"""

    __tablename__ = "conversation_summary"
    __table_args__ = {
        "comment": "AI 对话历史摘要表",
    }
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="摘要主键 ID",
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="所属会话 ID",
    )
    device_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="所属设备 ID",
    )
    agent_id: Mapped[int | None] = mapped_column(
        BigInteger,
        index=True,
        comment="agent id"
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False, comment="摘要正文")
    covered_start_message_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="摘要覆盖的起始消息 ID",
    )

    covered_end_message_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="摘要覆盖的结束消息 ID",
    )
    summary_level: Mapped[str] = mapped_column(
        String(20),
        default="segment",
        server_default=text("'segment'"),
        nullable=False,
        comment="摘要级别：segment 分段摘要，aggregate 合并摘要",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        server_default=text("'active'"),
        nullable=False,
        comment="摘要状态：active 有效，archived 已归档",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="创建时间",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="更新时间",
    )
