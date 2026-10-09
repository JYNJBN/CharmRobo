from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class KnowledgeDocument(Base):
    """用户提供的一份资料及其当前处理快照。"""

    __tablename__ = "knowledge_document"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "create_request_key",
            name="uq_doc_create_request",
        ),
        CheckConstraint(
            "ingest_generation > 0",
            name="ck_doc_generation",
        ),
        CheckConstraint(
            "embedding_dimension > 0",
            name="ck_doc_embedding_dimension",
        ),
        CheckConstraint(
            "file_size >= 0 AND chunk_count >= 0",
            name="ck_doc_size",
        ),
        CheckConstraint(
            "(deleted = 0 AND deleted_at IS NULL) "
            "OR (deleted = 1 AND deleted_at IS NOT NULL)",
            name="ck_doc_deleted",
        ),
        Index(
            "uq_doc_live_content",
            "knowledge_base_id",
            "content_hash",
            unique=True,
            postgresql_where=text(
                "deleted = 0 AND content_hash IS NOT NULL"
            ),
        ),
        Index(
            "ix_doc_list",
            "knowledge_base_id",
            text("created_at DESC"),
            text("id DESC"),
            postgresql_where=text("deleted = 0"),
        ),
        {"comment": "知识库文档表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="文档 ID",
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="所属知识库 ID；由 Service 校验归属",
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="所属用户 ID；由 Service 从认证上下文写入",
    )

    source_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="来源类型：manual、paste、file 或 qa_csv",
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="文档展示标题",
    )
    file_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="原始文件名；手动文字为空",
    )
    mime_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="确认后的 MIME 类型",
    )
    file_size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="手动文字 UTF-8 字节数或文件实际字节数",
    )
    storage_backend: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="原文存储位置：inline 或 cos",
    )

    raw_content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="manual / paste 的原始正文；COS 文件为空",
    )
    parsed_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="当前代次的规范化解析文本",
    )
    parsed_elements_json: Mapped[dict | list | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="当前代次的解析元素和来源定位",
    )
    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="服务端确认的原文 SHA-256",
    )

    create_request_key: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="创建文档的幂等请求键",
    )
    create_request_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="请求内容指纹，防止同一请求键对应不同内容",
    )

    parse_status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        comment="处理状态；允许值和转换由 Service / Worker 校验",
    )
    ingest_generation: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
        comment="当前处理代次",
    )
    parser_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="当前代次解析规则版本",
    )
    chunker_version: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="当前代次切块规则版本",
    )
    embedding_profile: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="向量配置身份",
    )
    embedding_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="当前代次使用的 Embedding 模型",
    )
    embedding_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="当前代次的向量维度",
    )
    vector_collection: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="当前代次目标 Milvus 集合",
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
        comment="当前代次已提交的完整切块数",
    )
    warnings_json: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
        comment="非致命解析警告",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="稳定的失败分类",
    )
    error_message: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
        comment="可展示的脱敏失败说明",
    )
    ready_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="当前代次发布为 ready 的时间",
    )
    purged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="原文和解析内容物理清理完成的时间",
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
