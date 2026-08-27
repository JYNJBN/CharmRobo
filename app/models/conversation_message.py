from datetime import datetime

from sqlalchemy import BigInteger, DateTime,  String, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ConversationMessage(Base):
    """会话中的单条用户或 AI 消息。"""

    __tablename__ = "conversation_message"
    __table_args__ = {
        "comment": "AI 对话消息表",
    }

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="消息主键 ID",
    )

    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="所属会话 ID",
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="消息角色：user 用户，assistant AI 助手",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="消息文本内容；语音场景中为 STT 识别结果或 AI 回复",
    )

    source: Mapped[str] = mapped_column(
        String(20),
        default="voice",
        server_default=text("'voice'"),
        nullable=False,
        comment="消息来源：voice 语音，text 文字，system 系统",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="消息创建时间",
    )