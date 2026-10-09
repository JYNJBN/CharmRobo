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


class KnowledgeChunk(Base):
    """一份知识文档中可检索的正文片段。"""

    __tablename__ = "knowledge_chunk"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "ingest_generation",
            "chunk_index",
            name="uq_chunk_generation",
        ),
        CheckConstraint(
            "chunk_index >= 0 AND ingest_generation > 0",
            name="ck_chunk_position",
        ),
        CheckConstraint(
            "(start_offset IS NULL AND end_offset IS NULL) OR "
            "(start_offset IS NOT NULL AND end_offset IS NOT NULL "
            "AND start_offset >= 0 AND end_offset > start_offset)",
            name="ck_chunk_offsets",
        ),
        CheckConstraint(
            "(token_count IS NULL OR token_count >= 0) "
            "AND (page_number IS NULL OR page_number > 0) "
            "AND (row_number IS NULL OR row_number > 0)",
            name="ck_chunk_numeric",
        ),
        CheckConstraint(
            "(deleted = 0 AND deleted_at IS NULL) "
            "OR (deleted = 1 AND deleted_at IS NOT NULL)",
            name="ck_chunk_deleted",
        ),
        Index(
            "ix_chunk_live_document",
            "document_id",
            "ingest_generation",
            "chunk_index",
            postgresql_where=text("deleted = 0"),
        ),
        {"comment": "知识库可检索切块表"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="切块 ID；同时作为 Milvus chunk_id",
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="所属用户 ID；由 Worker 写入前校验归属",
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="所属知识库 ID；用于查询和 Milvus 过滤",
    )
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment="来源文档 ID；高频表不设复合外键",
    )
    ingest_generation: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="本块所属处理代次",
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="同一文档、同一代次中的顺序，从 0 开始",
    )

    title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="所属章节标题",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="用于展示、引用和回答的正文",
    )
    embedding_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="实际发送给 Embedding 模型的文本快照",
    )
    embedding_text_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="embedding_text 的 SHA-256",
    )
    element_type: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        comment="解析元素类型，例如 paragraph、list、table 或 qa",
    )

    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="PDF 来源起始页码，从 1 开始",
    )
    row_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="CSV 来源记录序号，从 1 开始",
    )
    start_offset: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="当前代次 parsed_text 中的起始字符偏移",
    )
    end_offset: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="当前代次 parsed_text 中的结束字符偏移，不包含该字符",
    )
    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="可选；接入准确 tokenizer 前暂不保存",
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="标题路径、段落、跨页等来源元数据",
    )
    vector_sync_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        server_default=text("'pending'"),
        comment="向量同步状态；允许值和转换由 Worker 校验",
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
