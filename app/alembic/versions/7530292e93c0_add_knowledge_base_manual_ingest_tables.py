"""add knowledge base manual ingest tables

Revision ID: 7530292e93c0
Revises: b7f4e2a91c03
Create Date: 2026-09-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "7530292e93c0"
down_revision: str | Sequence[str] | None = "b7f4e2a91c03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tables needed for manual knowledge ingestion."""
    op.create_table(
        "knowledge_base",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
            comment="知识库 ID",
        ),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            nullable=False,
            comment="创建者 ID；由 Service 校验用户归属",
        ),
        sa.Column(
            "name",
            sa.String(length=64),
            nullable=False,
            comment="知识库展示名称",
        ),
        sa.Column(
            "description",
            sa.String(length=255),
            nullable=True,
            comment="知识库用途说明",
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'active'"),
            nullable=False,
            comment="业务状态；允许值和状态转换由 Service 校验",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="创建时间",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="最近更新时间",
        ),
        sa.Column(
            "deleted",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
            comment="逻辑删除标记：0 有效，1 已删除",
        ),
        sa.Column(
            "deleted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="逻辑删除时间",
        ),
        sa.CheckConstraint(
            "(deleted = 0 AND deleted_at IS NULL) "
            "OR (deleted = 1 AND deleted_at IS NOT NULL)",
            name="ck_kb_deleted",
        ),
        sa.PrimaryKeyConstraint("id"),
        comment="用户知识库表",
    )
    op.create_index(
        "ix_kb_owner_list",
        "knowledge_base",
        ["user_id", sa.text("updated_at DESC"), sa.text("id DESC")],
        unique=False,
        postgresql_where=sa.text("deleted = 0"),
    )

    op.create_table(
        "knowledge_document",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
            comment="文档 ID",
        ),
        sa.Column(
            "knowledge_base_id",
            sa.BigInteger(),
            nullable=False,
            comment="所属知识库 ID；由 Service 校验归属",
        ),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            nullable=False,
            comment="所属用户 ID；由 Service 从认证上下文写入",
        ),
        sa.Column(
            "source_type",
            sa.String(length=20),
            nullable=False,
            comment="来源类型：manual、paste、file 或 qa_csv",
        ),
        sa.Column(
            "title",
            sa.String(length=255),
            nullable=False,
            comment="文档展示标题",
        ),
        sa.Column(
            "file_name",
            sa.String(length=255),
            nullable=True,
            comment="原始文件名；手动文字为空",
        ),
        sa.Column(
            "mime_type",
            sa.String(length=100),
            nullable=True,
            comment="确认后的 MIME 类型",
        ),
        sa.Column(
            "file_size",
            sa.BigInteger(),
            nullable=False,
            comment="手动文字 UTF-8 字节数或文件实际字节数",
        ),
        sa.Column(
            "storage_backend",
            sa.String(length=20),
            nullable=False,
            comment="原文存储位置：inline 或 cos",
        ),
        sa.Column(
            "raw_content",
            sa.Text(),
            nullable=True,
            comment="manual / paste 的原始正文；COS 文件为空",
        ),
        sa.Column(
            "parsed_text",
            sa.Text(),
            nullable=True,
            comment="当前代次的规范化解析文本",
        ),
        sa.Column(
            "parsed_elements_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="当前代次的解析元素和来源定位",
        ),
        sa.Column(
            "content_hash",
            sa.String(length=64),
            nullable=True,
            comment="服务端确认的原文 SHA-256",
        ),
        sa.Column(
            "create_request_key",
            sa.String(length=64),
            nullable=False,
            comment="创建文档的幂等请求键",
        ),
        sa.Column(
            "create_request_hash",
            sa.String(length=64),
            nullable=False,
            comment="请求内容指纹，防止同一请求键对应不同内容",
        ),
        sa.Column(
            "parse_status",
            sa.String(length=24),
            nullable=False,
            comment="处理状态；允许值和转换由 Service / Worker 校验",
        ),
        sa.Column(
            "ingest_generation",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
            comment="当前处理代次",
        ),
        sa.Column(
            "parser_version",
            sa.String(length=32),
            nullable=False,
            comment="当前代次解析规则版本",
        ),
        sa.Column(
            "chunker_version",
            sa.String(length=32),
            nullable=False,
            comment="当前代次切块规则版本",
        ),
        sa.Column(
            "embedding_profile",
            sa.String(length=64),
            nullable=False,
            comment="向量配置身份",
        ),
        sa.Column(
            "embedding_model",
            sa.String(length=128),
            nullable=False,
            comment="当前代次使用的 Embedding 模型",
        ),
        sa.Column(
            "embedding_dimension",
            sa.Integer(),
            nullable=False,
            comment="当前代次的向量维度",
        ),
        sa.Column(
            "vector_collection",
            sa.String(length=128),
            nullable=False,
            comment="当前代次目标 Milvus 集合",
        ),
        sa.Column(
            "chunk_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
            comment="当前代次已提交的完整切块数",
        ),
        sa.Column(
            "warnings_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
            comment="非致命解析警告",
        ),
        sa.Column(
            "error_code",
            sa.String(length=64),
            nullable=True,
            comment="稳定的失败分类",
        ),
        sa.Column(
            "error_message",
            sa.String(length=1000),
            nullable=True,
            comment="可展示的脱敏失败说明",
        ),
        sa.Column(
            "ready_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="当前代次发布为 ready 的时间",
        ),
        sa.Column(
            "purged_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="原文和解析内容物理清理完成的时间",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="创建时间",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="最近更新时间",
        ),
        sa.Column(
            "deleted",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
            comment="逻辑删除标记：0 有效，1 已删除",
        ),
        sa.Column(
            "deleted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="逻辑删除时间",
        ),
        sa.CheckConstraint(
            "ingest_generation > 0",
            name="ck_doc_generation",
        ),
        sa.CheckConstraint(
            "embedding_dimension > 0",
            name="ck_doc_embedding_dimension",
        ),
        sa.CheckConstraint(
            "file_size >= 0 AND chunk_count >= 0",
            name="ck_doc_size",
        ),
        sa.CheckConstraint(
            "(deleted = 0 AND deleted_at IS NULL) "
            "OR (deleted = 1 AND deleted_at IS NOT NULL)",
            name="ck_doc_deleted",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "create_request_key",
            name="uq_doc_create_request",
        ),
        comment="知识库文档表",
    )
    op.create_index(
        "uq_doc_live_content",
        "knowledge_document",
        ["knowledge_base_id", "content_hash"],
        unique=True,
        postgresql_where=sa.text(
            "deleted = 0 AND content_hash IS NOT NULL"
        ),
    )
    op.create_index(
        "ix_doc_list",
        "knowledge_document",
        [
            "knowledge_base_id",
            sa.text("created_at DESC"),
            sa.text("id DESC"),
        ],
        unique=False,
        postgresql_where=sa.text("deleted = 0"),
    )

    op.create_table(
        "knowledge_chunk",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
            comment="切块 ID；同时作为 Milvus chunk_id",
        ),
        sa.Column(
            "user_id",
            sa.BigInteger(),
            nullable=False,
            comment="所属用户 ID；由 Worker 写入前校验归属",
        ),
        sa.Column(
            "knowledge_base_id",
            sa.BigInteger(),
            nullable=False,
            comment="所属知识库 ID；用于查询和 Milvus 过滤",
        ),
        sa.Column(
            "document_id",
            sa.BigInteger(),
            nullable=False,
            comment="来源文档 ID；高频表不设复合外键",
        ),
        sa.Column(
            "ingest_generation",
            sa.Integer(),
            nullable=False,
            comment="本块所属处理代次",
        ),
        sa.Column(
            "chunk_index",
            sa.Integer(),
            nullable=False,
            comment="同一文档、同一代次中的顺序，从 0 开始",
        ),
        sa.Column(
            "title",
            sa.String(length=255),
            nullable=True,
            comment="所属章节标题",
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
            comment="用于展示、引用和回答的正文",
        ),
        sa.Column(
            "embedding_text",
            sa.Text(),
            nullable=False,
            comment="实际发送给 Embedding 模型的文本快照",
        ),
        sa.Column(
            "embedding_text_hash",
            sa.String(length=64),
            nullable=False,
            comment="embedding_text 的 SHA-256",
        ),
        sa.Column(
            "element_type",
            sa.String(length=24),
            nullable=False,
            comment="解析元素类型，例如 paragraph、list、table 或 qa",
        ),
        sa.Column(
            "page_number",
            sa.Integer(),
            nullable=True,
            comment="PDF 来源起始页码，从 1 开始",
        ),
        sa.Column(
            "row_number",
            sa.Integer(),
            nullable=True,
            comment="CSV 来源记录序号，从 1 开始",
        ),
        sa.Column(
            "start_offset",
            sa.Integer(),
            nullable=True,
            comment="当前代次 parsed_text 中的起始字符偏移",
        ),
        sa.Column(
            "end_offset",
            sa.Integer(),
            nullable=True,
            comment="当前代次 parsed_text 中的结束字符偏移，不包含该字符",
        ),
        sa.Column(
            "token_count",
            sa.Integer(),
            nullable=False,
            comment="embedding_text 的 token 数",
        ),
        sa.Column(
            "metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
            comment="标题路径、段落、跨页等来源元数据",
        ),
        sa.Column(
            "vector_sync_status",
            sa.String(length=20),
            server_default=sa.text("'pending'"),
            nullable=False,
            comment="向量同步状态；允许值和转换由 Worker 校验",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="创建时间",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
            comment="最近更新时间",
        ),
        sa.Column(
            "deleted",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
            comment="逻辑删除标记：0 有效，1 已删除",
        ),
        sa.Column(
            "deleted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="逻辑删除时间",
        ),
        sa.CheckConstraint(
            "chunk_index >= 0 AND ingest_generation > 0",
            name="ck_chunk_position",
        ),
        sa.CheckConstraint(
            "(start_offset IS NULL AND end_offset IS NULL) OR "
            "(start_offset IS NOT NULL AND end_offset IS NOT NULL "
            "AND start_offset >= 0 AND end_offset > start_offset)",
            name="ck_chunk_offsets",
        ),
        sa.CheckConstraint(
            "token_count >= 0 "
            "AND (page_number IS NULL OR page_number > 0) "
            "AND (row_number IS NULL OR row_number > 0)",
            name="ck_chunk_numeric",
        ),
        sa.CheckConstraint(
            "(deleted = 0 AND deleted_at IS NULL) "
            "OR (deleted = 1 AND deleted_at IS NOT NULL)",
            name="ck_chunk_deleted",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "ingest_generation",
            "chunk_index",
            name="uq_chunk_generation",
        ),
        comment="知识库可检索切块表",
    )
    op.create_index(
        "ix_chunk_live_document",
        "knowledge_chunk",
        ["document_id", "ingest_generation", "chunk_index"],
        unique=False,
        postgresql_where=sa.text("deleted = 0"),
    )


def downgrade() -> None:
    """Drop the manual knowledge ingestion tables."""
    op.drop_index("ix_chunk_live_document", table_name="knowledge_chunk")
    op.drop_table("knowledge_chunk")

    op.drop_index("ix_doc_list", table_name="knowledge_document")
    op.drop_index("uq_doc_live_content", table_name="knowledge_document")
    op.drop_table("knowledge_document")

    op.drop_index("ix_kb_owner_list", table_name="knowledge_base")
    op.drop_table("knowledge_base")
