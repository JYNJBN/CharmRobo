from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AgentKnowledgeBase(Base):
    """设备智能体与知识库的授权关系。"""

    __tablename__ = "agent_knowledge_base"
    __table_args__ = (
        UniqueConstraint(
            "agent_id",
            "knowledge_base_id",
            name="uq_agent_kb_pair",
        ),
        CheckConstraint(
            "(deleted = 0 AND deleted_at IS NULL) "
            "OR (deleted = 1 AND deleted_at IS NOT NULL)",
            name="ck_agent_kb_deleted",
        ),
        Index(
            "ix_link_live_agent",
            "user_id",
            "agent_id",
            "knowledge_base_id",
            postgresql_where=text("deleted = 0"),
        ),
        Index(
            "ix_link_live_kb",
            "knowledge_base_id",
            "user_id",
            postgresql_where=text("deleted = 0"),
        ),
        {"comment": "智能体与知识库授权关系表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="建立关联时的设备 owner 与知识库 owner",
    )
    agent_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("agent.id"),
        nullable=False,
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("knowledge_base.id"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
