from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, field_validator


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=255)


class KnowledgeBaseListItem(BaseModel):
    id: str
    name: str
    description: str | None
    status: str
    document_count: int
    selected_agent_count: int
    updated_at: datetime


class KnowledgeBaseList(BaseModel):
    items: list[KnowledgeBaseListItem]
    total: int
    total_document_count: int
    page: int
    page_size: int


class KnowledgeDocumentListItem(BaseModel):
    id: str
    title: str
    source_type: str
    file_name: str | None
    file_size: int
    parse_status: str
    chunk_count: int
    updated_at: datetime
    current_job_id: str | None = None
    current_job_status: str | None = None
    attempt_count: int | None = None
    max_attempts: int | None = None
    error_code: str | None = None
    error_message: str | None = None


class KnowledgeDocumentList(BaseModel):
    items: list[KnowledgeDocumentListItem]
    total: int
    page: int
    page_size: int


class ManualDocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=20_000)


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1_000)


class KnowledgeSearchHit(BaseModel):
    chunk_id: int
    document_id: int
    title: str | None
    content: str
    score: float


class KnowledgeBaseAgentOption(BaseModel):
    id: str
    name: str
    description: str | None = None
    device_id: str
    device_name: str
    active: bool
    selected: bool


class KnowledgeBaseAgentList(BaseModel):
    knowledge_base_id: str
    selected_count: int
    selected_agent_ids: list[str]
    agents: list[KnowledgeBaseAgentOption]


class KnowledgeBaseAgentUpdate(BaseModel):
    agent_ids: list[str]
    expected_agent_ids: list[str]


class RetryKnowledgeDocumentRequest(BaseModel):
    job_id: int = Field(
        gt=0,
        le=9_223_372_036_854_775_807,
    )
    expected_attempt_count: int = Field(ge=0)


class KnowledgeUploadInitRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    title: str = Field(min_length=1, max_length=255)
    file_name: str = Field(min_length=1, max_length=255)
    source_type: Literal["file"] = "file"
    file_size: int = Field(strict=True, gt=0)
    sha256: str = Field(
        pattern=r"^[0-9a-fA-F]{64}$",
    )

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.lower()


class KnowledgeUploadCompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: UUID
