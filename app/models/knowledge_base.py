from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class KnowledgeBase(Base):
    """用户的知识库容器。"""

    __tablename__ = "knowledge_base"
    __table_args__ = (
        CheckConstraint(
            "(deleted = 0 AND deleted_at IS NULL) "
            "OR (deleted = 1 AND deleted_at IS NOT NULL)",
            name="ck_kb_deleted",
        ),
        Index(
            "ix_kb_owner_list",
            "user_id",
            text("updated_at DESC"),
            text("id DESC"),
            postgresql_where=text("deleted = 0"),
        ),
        {"comment": "用户知识库表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="知识库 ID",
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="创建者 ID；由 Service 校验用户归属",
    )
    name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="知识库展示名称",
    )
    description: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="知识库用途说明",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        server_default=text("'active'"),
        comment="业务状态；允许值和状态转换由 Service 校验",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最近更新时间",
    )
    deleted: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        comment="逻辑删除标记：0 有效，1 已删除",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="逻辑删除时间",
    )
