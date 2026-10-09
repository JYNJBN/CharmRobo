from typing import Annotated

from fastapi import APIRouter, status, Header, Query

from app.api.dependencies import CurrentUserId, DbSession
from app.schemas.common import ApiResponse
from app.schemas.knowledge import (
    KnowledgeBaseAgentList,
    KnowledgeBaseAgentUpdate,
    KnowledgeBaseCreate,
    KnowledgeBaseList,
    KnowledgeDocumentList,
    KnowledgeSearchHit,
    KnowledgeSearchRequest,
    ManualDocumentCreate, RetryKnowledgeDocumentRequest, KnowledgeUploadInitRequest, KnowledgeUploadCompleteRequest,
)
from app.services.knowledge_retrieval_service import search_knowledge
from app.services.knowledge_service import (
    create_knowledge_base,
    create_manual_document,
    list_agents_for_knowledge_base,
    list_knowledge_base_documents,
    list_knowledge_bases,
    save_agents_for_knowledge_base, delete_knowledge_document_service, retry_knowledge_document_service,
    delete_knowledge_base_service, init_document_upload, complete_document_upload,
)

router = APIRouter(prefix="/knowledge-bases", tags=["知识库"])


@router.get("", response_model=ApiResponse[KnowledgeBaseList])
async def list_knowledge_bases_api(
        db: DbSession,
        user_id: CurrentUserId,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
) -> ApiResponse[KnowledgeBaseList]:
    result = await list_knowledge_bases(
        db=db,
        user_id=user_id,
        page=page,
        page_size=page_size,
    )
    return ApiResponse(data=result)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_knowledge_base_api(
        payload: KnowledgeBaseCreate,
        db: DbSession,
        user_id: CurrentUserId,
) -> ApiResponse[dict]:
    knowledge_base = await create_knowledge_base(
        db=db,
        user_id=user_id,
        payload=payload,
    )
    return ApiResponse(
        data={
            "id": str(knowledge_base.id),
            "name": knowledge_base.name,
            "description": knowledge_base.description,
            "status": knowledge_base.status,
        }
    )


@router.get(
    "/{knowledge_base_id}/documents",
    response_model=ApiResponse[KnowledgeDocumentList],
)
async def list_knowledge_base_documents_api(
        knowledge_base_id: int,
        db: DbSession,
        user_id: CurrentUserId,
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=100),
) -> ApiResponse[KnowledgeDocumentList]:
    result = await list_knowledge_base_documents(
        db=db,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        page=page,
        page_size=page_size,
    )
    return ApiResponse(data=result)


@router.post("/{knowledge_base_id}/documents/manual",
             status_code=status.HTTP_202_ACCEPTED,
             )
async def create_manual_document_api(
        knowledge_base_id: int,
        payload: ManualDocumentCreate,
        db: DbSession,
        user_id: CurrentUserId,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1, max_length=64),
        ],
) -> ApiResponse[dict]:
    document = await create_manual_document(
        db=db,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    return ApiResponse(
        data={
            "id": str(document.id),
            "knowledge_base_id": str(document.knowledge_base_id),
            "title": document.title,
            "parse_status": document.parse_status,
            "chunk_count": document.chunk_count,
        }
    )


@router.post(
    "/{knowledge_base_id}/search",
    response_model=ApiResponse[list[KnowledgeSearchHit]],
)
async def search_knowledge_api(
        knowledge_base_id: int,
        payload: KnowledgeSearchRequest,
        db: DbSession,
        user_id: CurrentUserId,
) -> ApiResponse[list[KnowledgeSearchHit]]:
    hits = await search_knowledge(
        db=db,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        query=payload.query,
    )
    return ApiResponse(data=hits)


@router.get(
    "/{knowledge_base_id}/agents",
    response_model=ApiResponse[KnowledgeBaseAgentList],
)
async def list_knowledge_base_agents_api(
        knowledge_base_id: int,
        db: DbSession,
        user_id: CurrentUserId,
) -> ApiResponse[KnowledgeBaseAgentList]:
    result = await list_agents_for_knowledge_base(
        db=db,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
    )
    return ApiResponse(data=result)


@router.put(
    "/{knowledge_base_id}/agents",
    response_model=ApiResponse[KnowledgeBaseAgentList],
)
async def save_knowledge_base_agents_api(
        knowledge_base_id: int,
        payload: KnowledgeBaseAgentUpdate,
        db: DbSession,
        user_id: CurrentUserId,
) -> ApiResponse[KnowledgeBaseAgentList]:
    result = await save_agents_for_knowledge_base(
        db=db,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        payload=payload,
    )
    return ApiResponse(data=result)


@router.delete(
    "/knowledge-documents/{document_id}",
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_knowledge_document_api(
        document_id: int,
        db: DbSession,
        user_id: CurrentUserId,
) -> ApiResponse[dict]:
    document, delete_job_ids = await delete_knowledge_document_service(
        db=db,
        user_id=user_id,
        document_id=document_id,
    )
    return ApiResponse(
        data={
            "document_id": str(document.id),
            "deleted": bool(document.deleted),
            "delete_job_ids": [str(job_id) for job_id in delete_job_ids],
        }
    )


@router.post(
    "/knowledge-documents/{document_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_knowledge_document_api(
        document_id: int,
        payload: RetryKnowledgeDocumentRequest,
        db: DbSession,
        user_id: CurrentUserId,
) -> ApiResponse[dict]:
    document, job = await retry_knowledge_document_service(
        db=db,
        user_id=user_id,
        document_id=document_id,
        payload=payload,
    )

    return ApiResponse(
        data={
            "document_id": str(document.id),
            "job_id": str(job.id),
            "parse_status": document.parse_status,
            "job_status": job.status,
            "stage": job.stage,
            "attempt_count": job.attempt_count,
            "max_attempts": job.max_attempts,
        }
    )


@router.delete(
    "/{knowledge_base_id}",
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_knowledge_base_api(
        knowledge_base_id: int,
        db: DbSession,
        user_id: CurrentUserId,
) -> ApiResponse[dict]:
    knowledge_base, delete_job_ids = await delete_knowledge_base_service(
        db=db,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
    )
    return ApiResponse(
        data={
            "knowledge_base_id": str(knowledge_base.id),
            "deleted": bool(knowledge_base.deleted),
            "delete_job_ids": [str(job_id) for job_id in delete_job_ids],
        }
    )


@router.post(
    "/{knowledge_base_id}/documents/upload-init",
    status_code=status.HTTP_201_CREATED,
)
async def init_document_upload_api(
        knowledge_base_id: int,
        payload: KnowledgeUploadInitRequest,
        db: DbSession,
        user_id: CurrentUserId,
        idempotency_key: Annotated[
            str,
            Header(
                alias="Idempotency-Key",
                min_length=1,
                max_length=64,
            ),
        ],
) -> ApiResponse[dict]:
    result = await init_document_upload(
        db,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    return ApiResponse(data=result)


@router.post(
    "/knowledge-documents/{document_id}/upload-complete",
    status_code=status.HTTP_202_ACCEPTED,
)
async def complete_document_upload_api(
        document_id: int,
        payload: KnowledgeUploadCompleteRequest,
        db: DbSession,
        user_id: CurrentUserId,
) -> ApiResponse[dict]:
    result = await complete_document_upload(
        db,
        user_id=user_id,
        document_id=document_id,
        upload_id=payload.upload_id,
    )
    return ApiResponse(data=result)
