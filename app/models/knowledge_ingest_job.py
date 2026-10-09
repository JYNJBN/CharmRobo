from datetime import datetime
from uuid import UUID as UUIDValue

from sqlalchemy import UniqueConstraint, CheckConstraint, Index, text, BigInteger, String, CHAR, Integer, func, \
    DateTime, UUID
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class KnowledgeIngestJob(Base):
    __tablename__ = "knowledge_ingest_job"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_job_key"),
        CheckConstraint(
            "attempt_count >= 0 AND max_attempts > 0 "
            "AND attempt_count <= max_attempts",
            name="ck_job_attempts",
        ),
        CheckConstraint(
            "ingest_generation > 0",
            name="ck_job_generation",
        ),
        CheckConstraint(
            "delete_before_generation IS NULL "
            "OR delete_before_generation > 0",
            name="ck_job_delete_generation",
        ),
        Index(
            "uq_ingest_document_generation",
            "document_id",
            "ingest_generation",
            unique=True,
            postgresql_where=text("job_type = 'ingest'"),
        ),
        Index(
            "ix_job_queue",
            "available_at",
            "id",
            postgresql_where=text("status = 'queued'"),
        ),
        Index(
            "ix_job_expired_lease",
            "locked_until",
            postgresql_where=text("status = 'running'"),
        ),
        Index(
            "ix_job_document",
            "document_id",
            "ingest_generation",
            "job_type",
        ),
        Index(
            "ix_job_running_user",
            "user_id",
            postgresql_where=text("status = 'running'"),
        ),
        {"comment": "知识库后台处理任务表"},
    )
    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
        comment="任务 ID",
    )
    document_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="文档 ID",
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="知识库 ID",
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="用户 ID",
    )
    job_type: Mapped[str] = mapped_column(
        String(24), nullable=False,
        comment="ingest、delete_vectors 或 purge",
    )
    stage: Mapped[str] = mapped_column(
        String(24), nullable=False,
        comment="下一步处理阶段",
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False,
        default="queued",
        server_default=text("'queued'"),
        comment="queued、running、succeeded、failed 或 cancelled",
    )
    idempotency_key: Mapped[str] = mapped_column(
        CHAR(64), nullable=False,
        comment="服务端生成的任务幂等键",
    )
    ingest_generation: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="任务对应的文档处理代次",
    )
    delete_before_generation: Mapped[int | None] = mapped_column(
        Integer, nullable=True,
        comment="清理向量时，只删除小于此代次的向量",
    )
    config_snapshot: Mapped[dict] = mapped_column(
        JSONB, nullable=False,
        comment="创建任务时冻结的处理配置",
    )
    result_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
        comment="Worker 的执行检查点",
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False,
        default=0,
        server_default=text("0"),
        comment="已领取执行次数",
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False,
        default=3,
        server_default=text("3"),
        comment="允许执行的总次数",
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
        comment="最早可领取时间",
    )
    locked_by: Mapped[str | None] = mapped_column(
        String(128), nullable=True,
        comment="当前 Worker 标识",
    )
    lease_token: Mapped[UUIDValue | None] = mapped_column(
        UUID(as_uuid=True), nullable=True,
        comment="本次领取的随机凭证",
    )
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="本次领取时间",
    )
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="租约截止时间",
    )
    last_error_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
        comment="最近一次错误分类",
    )
    last_error_message: Mapped[str | None] = mapped_column(
        String(1000), nullable=True,
        comment="脱敏后的错误说明",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="首次开始时间",
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="最终结束时间",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最近更新时间",
    )
