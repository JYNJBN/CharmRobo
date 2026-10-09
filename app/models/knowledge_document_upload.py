from datetime import datetime
from uuid import UUID as UUIDValue

from sqlalchemy import PrimaryKeyConstraint, UniqueConstraint, Index, text, ForeignKey, BigInteger, String, UUID, \
    CHAR, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class KnowledgeDocumentUpload(Base):
    """COS 文件的上传会话、对象位置和验证信息。"""

    __tablename__ = "knowledge_document_upload"
    __table_args__ = (
        PrimaryKeyConstraint(
            "document_id",
            name="pk_doc_upload",
        ),
        UniqueConstraint(
            "upload_id",
            name="uq_doc_upload_session",
        ),
        Index(
            "ix_doc_upload_deadline",
            "upload_deadline",
            postgresql_where=text(
                "upload_confirmed_at IS NULL"
            ),
        ),
        {"comment": "知识库文档 COS 上传记录表"},
    )
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "knowledge_document.id",
            name="fk_doc_upload_document",
        ),
        primary_key=True,
        autoincrement=False,
        nullable=False,
        comment="文档 ID；直接复用文档主键，一对一关联",
    )

    storage_bucket: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="COS 桶名，包含 APPID",
    )
    storage_region: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="COS 地域",
    )
    storage_key: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="服务端预分配的正式对象 key",
    )
    upload_id: Mapped[UUIDValue] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment="服务端生成的上传会话 UUID",
    )
    upload_key: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="客户端允许上传的 incoming 暂存 key",
    )
    upload_expected_sha256: Mapped[str] = mapped_column(
        CHAR(64),
        nullable=False,
        comment="客户端声明的 SHA-256，待 Worker 核实",
    )
    upload_deadline: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="上传会话截止时间",
    )

    upload_confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment="API 核实暂存对象并创建任务的时间",
    )

    upload_etag: Mapped[str | None] = mapped_column(
        String(256),
        nullable=True,
        default=None,
        comment="完成确认时从 COS HEAD 获取的 ETag",
    )

    source_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        comment="Worker 完成正式源文件验证的时间",
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
        comment="更新时间",
    )
    config_snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="申请上传时冻结的处理和存储配置，不保存密钥",
    )
