import asyncio
import hashlib
import json
from copy import deepcopy
from datetime import timedelta, datetime
from pathlib import Path
from typing import cast
from uuid import UUID as UUIDValue
from uuid import uuid4

from qcloud_cos import CosServiceError, CosClientError
from sqlalchemy import and_, select, func, update, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import BizError
from app.integrations.knowledge_cos import build_cos_upload_form, get_knowledge_cos_client, head_knowledge_upload_object
from app.models import KnowledgeBase, KnowledgeDocument, Agent, Device, UserDevice, AgentKnowledgeBase, \
    KnowledgeIngestJob, KnowledgeChunk, KnowledgeDocumentUpload
from app.schemas.knowledge import (
    KnowledgeBaseAgentList,
    KnowledgeBaseAgentOption,
    KnowledgeBaseAgentUpdate,
    KnowledgeBaseCreate,
    KnowledgeBaseList,
    KnowledgeBaseListItem,
    KnowledgeDocumentList,
    ManualDocumentCreate, KnowledgeDocumentListItem, RetryKnowledgeDocumentRequest, KnowledgeUploadInitRequest,
)

RETRYABLE_JOB_ERROR_CODES = {
    "WORKER_STAGE_FAILED",
    "EMBEDDING_TIMEOUT",
    "MILVUS_UPSERT_FAILED",
    "WORKER_LEASE_EXPIRED",
}


async def create_knowledge_base(
        db: AsyncSession,
        user_id: int,
        payload: KnowledgeBaseCreate,
) -> KnowledgeBase:
    """创建知识库"""
    name = payload.name.strip()
    if not name:
        raise BizError(code=422, message="知识库名称不能为空")
    description = payload.description
    if description is not None:
        description = description.strip() or None

    knowledge_base = KnowledgeBase(
        user_id=user_id,
        name=name,
        description=description,
        status="active",
        deleted=0,
    )
    db.add(knowledge_base)
    await db.commit()
    await db.refresh(knowledge_base)
    return knowledge_base


async def list_knowledge_bases(
        db: AsyncSession,
        user_id: int,
        page: int = 1,
        page_size: int = 20,
) -> KnowledgeBaseList:
    """分页列出当前用户未删除的知识库及其文档、智能体数量。"""
    base_filter = (
        KnowledgeBase.user_id == user_id,
        KnowledgeBase.deleted == 0,
    )
    total = await db.scalar(
        select(func.count(KnowledgeBase.id)).where(*base_filter)
    ) or 0
    total_document_count = await db.scalar(
        select(func.count(KnowledgeDocument.id))
        .join(KnowledgeBase, KnowledgeBase.id == KnowledgeDocument.knowledge_base_id)
        .where(
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.deleted == 0,
            KnowledgeDocument.deleted == 0,
        )
    ) or 0

    document_counts = (
        select(
            KnowledgeDocument.knowledge_base_id.label("knowledge_base_id"),
            func.count(KnowledgeDocument.id).label("document_count"),
        )
        .where(KnowledgeDocument.deleted == 0)
        .group_by(KnowledgeDocument.knowledge_base_id)
        .subquery()
    )
    agent_counts = (
        select(
            AgentKnowledgeBase.knowledge_base_id.label("knowledge_base_id"),
            func.count(
                func.distinct(AgentKnowledgeBase.agent_id)
            ).label("selected_agent_count"),
        )
        .join(Agent, Agent.id == AgentKnowledgeBase.agent_id)
        .join(Device, Device.id == Agent.device_id)
        .join(UserDevice, UserDevice.device_id == Device.id)
        .where(
            AgentKnowledgeBase.user_id == user_id,
            AgentKnowledgeBase.deleted == 0,
            Agent.deleted == 0,
            Agent.is_system.is_(False),
            Agent.device_id.is_not(None),
            Device.deleted == 0,
            Device.status == 1,
            UserDevice.user_id == user_id,
            UserDevice.role == "owner",
            UserDevice.deleted == 0,
        )
        .group_by(AgentKnowledgeBase.knowledge_base_id)
        .subquery()
    )

    result = await db.execute(
        select(
            KnowledgeBase,
            func.coalesce(document_counts.c.document_count, 0),
            func.coalesce(agent_counts.c.selected_agent_count, 0),
        )
        .outerjoin(
            document_counts,
            document_counts.c.knowledge_base_id == KnowledgeBase.id,
        )
        .outerjoin(
            agent_counts,
            agent_counts.c.knowledge_base_id == KnowledgeBase.id,
        )
        .where(*base_filter)
        .order_by(KnowledgeBase.updated_at.desc(), KnowledgeBase.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    items = [
        KnowledgeBaseListItem(
            id=str(knowledge_base.id),
            name=knowledge_base.name,
            description=knowledge_base.description,
            status=knowledge_base.status,
            document_count=document_count,
            selected_agent_count=selected_agent_count,
            updated_at=knowledge_base.updated_at,
        )
        for knowledge_base, document_count, selected_agent_count in result.all()
    ]
    return KnowledgeBaseList(
        items=items,
        total=total,
        total_document_count=total_document_count,
        page=page,
        page_size=page_size,
    )


async def list_knowledge_base_documents(
        db: AsyncSession,
        user_id: int,
        knowledge_base_id: int,
        page: int = 1,
        page_size: int = 20,
) -> KnowledgeDocumentList:
    """分页列出用户指定知识库中的有效文档。"""
    knowledge_base_id_result = await db.scalar(
        select(KnowledgeBase.id).where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.deleted == 0,
        )
    )
    if knowledge_base_id_result is None:
        raise BizError("知识库不存在", code=404)

    document_filter = (
        KnowledgeDocument.knowledge_base_id == knowledge_base_id,
        KnowledgeDocument.user_id == user_id,
        KnowledgeDocument.deleted == 0,
    )
    total = await db.scalar(
        select(func.count(KnowledgeDocument.id)).where(*document_filter)
    ) or 0
    result = await db.execute(
        select(KnowledgeDocument, KnowledgeIngestJob)
        .outerjoin(
            KnowledgeIngestJob,
            and_(
                KnowledgeIngestJob.document_id == KnowledgeDocument.id,
                KnowledgeIngestJob.knowledge_base_id
                == KnowledgeDocument.knowledge_base_id,
                KnowledgeIngestJob.user_id == KnowledgeDocument.user_id,
                KnowledgeIngestJob.ingest_generation
                == KnowledgeDocument.ingest_generation,
                KnowledgeIngestJob.job_type == "ingest",
            ),
        )
        .where(*document_filter)
        .order_by(
            KnowledgeDocument.created_at.desc(),
            KnowledgeDocument.id.desc(),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = []
    for document, job in result.all():
        items.append(
            KnowledgeDocumentListItem(
                id=str(document.id),
                title=document.title,
                source_type=document.source_type,
                file_name=document.file_name,
                file_size=document.file_size,
                parse_status=document.parse_status,
                chunk_count=document.chunk_count,
                updated_at=document.updated_at,
                current_job_id=str(job.id) if job else None,
                current_job_status=job.status if job else None,
                attempt_count=job.attempt_count if job else None,
                max_attempts=job.max_attempts if job else None,
                error_code=document.error_code,
                error_message=document.error_message,
            )
        )
    return KnowledgeDocumentList(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


async def retry_knowledge_document_service(
        db: AsyncSession,
        user_id: int,
        document_id: int,
        payload: RetryKnowledgeDocumentRequest,
) -> tuple[KnowledgeDocument, KnowledgeIngestJob]:
    # 先确定文档所属知识库。
    knowledge_base_id = await db.scalar(
        select(KnowledgeDocument.knowledge_base_id).where(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.user_id == user_id,
        )
    )
    if knowledge_base_id is None:
        raise BizError("文档不存在", code=404)
    knowledge_base = await db.scalar(
        select(KnowledgeBase)
        .where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.deleted == 0,
        )
        .with_for_update()
    )
    if knowledge_base is None:
        raise BizError("知识库不存在", code=404)
    job = await db.scalar(
        select(KnowledgeIngestJob)
        .where(
            KnowledgeIngestJob.id == payload.job_id,
            KnowledgeIngestJob.document_id == document_id,
            KnowledgeIngestJob.knowledge_base_id == knowledge_base_id,
            KnowledgeIngestJob.user_id == user_id,
            KnowledgeIngestJob.job_type == "ingest",
        )
        .with_for_update()
    )
    if job is None:
        raise BizError("入库任务不存在", code=404)
    document = await db.scalar(
        select(KnowledgeDocument)
        .where(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.user_id == user_id,
            KnowledgeDocument.knowledge_base_id == knowledge_base_id,
        )
        .with_for_update()
    )
    if document is None:
        raise BizError("文档不存在", code=404)

    if document.deleted == 1 or document.purged_at is not None:
        raise BizError("文档已删除，不能重试", code=410)

    if job.ingest_generation != document.ingest_generation:
        raise BizError("任务属于旧代次，请刷新页面", code=409)
    # 重复点击时不再次增加重试预算。
    if job.status in {"queued", "running", "succeeded"}:
        await db.commit()
        return document, job
    if job.status != "failed":
        raise BizError("当前任务状态不能重试", code=409)
    if job.attempt_count != payload.expected_attempt_count:
        raise BizError("任务状态已变化，请刷新后重试", code=409)
    if job.last_error_code not in RETRYABLE_JOB_ERROR_CODES:
        raise BizError(
            "当前错误无法通过重试修复，请检查原文或处理配置",
            code=409,
        )

    # 当前 Embedding/Milvus 适配器读取 settings，
    # 因此配置改变后不能偷偷用新配置继续旧任务。
    current_snapshot = _manual_ingest_snapshot()
    if any(
            job.config_snapshot.get(key) != value
            for key, value in current_snapshot.items()
    ):
        raise BizError(
            "任务配置与当前配置不一致，请恢复原配置或重新添加文档",
            code=409,
        )
    stage_to_document_status = {
        "parse": "parsing",
        "chunk": "chunking",
        "embed": "embedding",
        "index": "indexing",
        "finalize": "indexing",
    }
    document_status = stage_to_document_status.get(job.stage)
    if document_status is None:
        raise BizError("当前处理阶段暂不支持重试", code=409)

    now = await db.scalar(select(func.now()))

    # 使用原任务，增加新一轮预算，保留历史执行次数。
    job.status = "queued"
    job.available_at = now
    job.max_attempts = job.attempt_count + 3
    job.finished_at = None
    job.locked_by = None
    job.lease_token = None
    job.locked_at = None
    job.locked_until = None

    document.parse_status = document_status
    document.error_code = None
    document.error_message = None
    document.ready_at = None
    knowledge_base.updated_at = now

    await db.commit()
    return document, job


async def _get_owned_active_knowledge_base(
        db: AsyncSession,
        user_id: int,
        knowledge_base_id: int,
) -> KnowledgeBase | None:
    """获取用户指定的知识库"""
    result = await db.execute(
        select(KnowledgeBase).where(
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.deleted == 0,
        )
    )
    knowledge_base = result.scalar_one_or_none()
    if knowledge_base is None:
        raise BizError(code=404, message="知识库不存在")

    if knowledge_base.status != "active":
        raise BizError(code=409, message="知识库当前不可用")
    return knowledge_base


def _manual_ingest_snapshot() -> dict:
    return {
        "parser_version": settings.knowledge_parser_version,
        "chunker_version": settings.knowledge_chunker_version,
        "embedding_profile": settings.knowledge_embedding_profile,
        "embedding_base_url": settings.embedding_base_url,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "metric_type": settings.embedding_metric_type,
        "vector_collection": settings.milvus_knowledge_collection,
        "chunk_target_chars": settings.knowledge_chunk_target_chars,
        "chunk_overlap_chars": settings.knowledge_chunk_overlap_chars,
    }


def _ingest_job_key(
        document_id: int,
        ingest_generation: int,
        snapshot: dict,
) -> str:
    def stable_json(value: object) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    snapshot_hash = hashlib.sha256(stable_json(snapshot)).hexdigest()
    identity = {
        "job_type": "ingest",
        "document_id": document_id,
        "ingest_generation": ingest_generation,
        "delete_before_generation": None,
        "target_resource": snapshot["vector_collection"],
        "config_snapshot_hash": snapshot_hash,
    }
    return hashlib.sha256(stable_json(identity)).hexdigest()


async def create_manual_document(
        db: AsyncSession,
        user_id: int,
        knowledge_base_id: int,
        payload: ManualDocumentCreate,
        idempotency_key: str,
) -> KnowledgeDocument:
    """创建手工输入的文档"""
    title = payload.title.strip()
    content = payload.content.strip()

    if not title:
        raise BizError(code=422, message="标题不能为空")
    if not content:
        raise BizError(code=422, message="正文不能为空")

    await _get_owned_active_knowledge_base(
        db=db,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id
    )
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
    # ensure_ascii 是让转化的json 保留中文原文 separators 是 让转换后的格式更紧凑
    request_data = json.dumps(
        [knowledge_base_id, title, content_hash],
        ensure_ascii=False,
        separators=(',', ':')
    )
    request_hash = hashlib.sha256(
        request_data.encode("utf-8")
    ).hexdigest()

    # 先查幂等键。相同键、相同请求就返回已有文档。
    existing_result = await db.execute(
        select(KnowledgeDocument).where(
            KnowledgeDocument.user_id == user_id,
            KnowledgeDocument.create_request_key == idempotency_key,
        )
    )
    existing = existing_result.scalar_one_or_none()

    if existing is not None:
        if existing.create_request_hash != request_hash:
            raise BizError(
                code=409,
                message="同一个 Idempotency-Key 不能用于不同内容",
            )
        return existing

    # 同一个知识库不允许有两份未删除的相同内容。
    duplicate_id = await db.scalar(
        select(KnowledgeDocument.id).where(
            KnowledgeDocument.knowledge_base_id == knowledge_base_id,
            KnowledgeDocument.content_hash == content_hash,
            KnowledgeDocument.deleted == 0,
        )
    )
    if duplicate_id is not None:
        raise BizError(
            code=409,
            message="此文档内容已存在"
        )
    # 文档
    snapshot = _manual_ingest_snapshot()
    document = KnowledgeDocument(
        knowledge_base_id=knowledge_base_id,
        user_id=user_id,
        source_type="manual",
        title=title,
        file_name=None,
        mime_type="text/plain",
        file_size=len(content.encode("utf-8")),
        storage_backend="inline",
        raw_content=content,
        parsed_text=None,
        parsed_elements_json=None,
        content_hash=content_hash,
        create_request_key=idempotency_key,
        create_request_hash=request_hash,
        parse_status="uploaded",
        ingest_generation=1,
        parser_version=settings.knowledge_parser_version,
        chunker_version=settings.knowledge_chunker_version,
        embedding_profile=settings.knowledge_embedding_profile,
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
        vector_collection=settings.milvus_knowledge_collection,
        chunk_count=0,
        warnings_json=[],
        deleted=0,
    )
    from sqlalchemy.exc import IntegrityError
    try:
        db.add(document)
        await db.flush()
        # 新建一个job 后面给woker操作
        job = KnowledgeIngestJob(
            document_id=document.id,
            knowledge_base_id=document.knowledge_base_id,
            user_id=document.user_id,
            job_type="ingest",
            stage="parse",
            status="queued",
            idempotency_key=_ingest_job_key(
                document.id,
                document.ingest_generation,
                snapshot,
            ),
            ingest_generation=document.ingest_generation,
            delete_before_generation=None,
            config_snapshot=snapshot,
            result_json={},
            attempt_count=0,
            max_attempts=3,
        )
        db.add(job)
        await db.commit()
        await db.refresh(document)
        return document
    except IntegrityError as exc:
        await db.rollback()

        # 并发情况下，同一个请求可能在前面的查询之后才被另一个请求提交。
        existing = await db.scalar(
            select(KnowledgeDocument).where(
                KnowledgeDocument.user_id == user_id,
                KnowledgeDocument.create_request_key == idempotency_key,
            )
        )
        if existing is not None:
            if existing.create_request_hash != request_hash:
                raise BizError(
                    code=409,
                    message="同一个 Idempotency-Key 不能用于不同内容",
                ) from exc
            return existing

        raise BizError(code=409, message="此文档内容已存在") from exc


def _parse_agent_ids(values: list[str]) -> set[int]:
    try:
        ids = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise BizError("agent_ids 必须是数字 ID", code=422) from exc

    if any(agent_id <= 0 for agent_id in ids):
        raise BizError("agent_ids 必须是正整数", code=422)

    if len(ids) != len(set(ids)):
        raise BizError("agent_ids 不能重复", code=422)

    return set(ids)


async def _get_owner_agents(
        db: AsyncSession,
        user_id: int,
) -> list[tuple[Agent, Device, UserDevice]]:
    result = await db.execute(
        select(Agent, Device, UserDevice)
        .join(Device, Device.id == Agent.device_id)
        .join(UserDevice, UserDevice.device_id == Device.id)
        .where(
            UserDevice.user_id == user_id,
            UserDevice.role == "owner",
            UserDevice.deleted == 0,
            Device.deleted == 0,
            Device.status == 1,
            Agent.deleted == 0,
            Agent.is_system.is_(False),
            Agent.device_id.is_not(None),
        )
        .order_by(Device.id, Agent.id)
    )
    return list(result.all())


async def list_agents_for_knowledge_base(
        db: AsyncSession,
        user_id: int,
        knowledge_base_id: int,
) -> KnowledgeBaseAgentList:
    knowledge_base = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.deleted == 0,
        )
    )
    if knowledge_base is None:
        raise BizError("知识库不存在", code=404)
    # 用户所以的agent
    owner_agents = await _get_owner_agents(db, user_id)
    # 用户所有的agent的id
    available_ids = {int(agent.id) for agent, _, _ in owner_agents}
    # 用户创建的知识库中选中的agent id
    selected_result = await db.scalars(
        select(AgentKnowledgeBase.agent_id).where(
            AgentKnowledgeBase.knowledge_base_id == knowledge_base_id,
            AgentKnowledgeBase.user_id == user_id,
            AgentKnowledgeBase.deleted == 0,
        )
    )
    # 当前选中的agent & 是集合交集 及要是选中的同时是要满足在可选里的
    selected_ids = set(selected_result.all()) & available_ids
    agents = [
        KnowledgeBaseAgentOption(
            id=str(agent.id),
            name=agent.name,
            description=(agent.system_prompt or "").strip()[:80] or None,
            device_id=str(device.id),
            device_name=binding.alias or device.device_sn,
            active=device.active_agent_id == agent.id,
            selected=agent.id in selected_ids,
        )
        for agent, device, binding in owner_agents
    ]

    return KnowledgeBaseAgentList(
        knowledge_base_id=str(knowledge_base.id),
        selected_count=len(selected_ids),
        selected_agent_ids=[
            str(agent_id) for agent_id in sorted(selected_ids)
        ],
        agents=agents,
    )


async def save_agents_for_knowledge_base(
        db: AsyncSession,
        user_id: int,
        knowledge_base_id: int,
        payload: KnowledgeBaseAgentUpdate,
) -> KnowledgeBaseAgentList:
    requested_ids = _parse_agent_ids(payload.agent_ids)
    expected_ids = _parse_agent_ids(payload.expected_agent_ids)

    try:
        # 锁定知识库行，串行处理对同一个库的关联保存。 行锁 其他查询在事务结束前只读 不能操作
        knowledge_base = await db.scalar(
            select(KnowledgeBase)
            .where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.deleted == 0,
            )
            .with_for_update()
        )
        if knowledge_base is None:
            raise BizError("知识库不存在", code=404)

        owner_agents = await _get_owner_agents(db, user_id)
        available_ids = {int(agent.id) for agent, _, _ in owner_agents}
        # 获取当前知识库和智能体的所以关系
        links_result = await db.scalars(
            select(AgentKnowledgeBase)
            .where(
                AgentKnowledgeBase.knowledge_base_id == knowledge_base_id,
            )
            .with_for_update()
        )
        links = list(links_result.all())
        # 用户还没发过来之前上一轮的选中状态agent ids
        current_ids = {
            int(link.agent_id)
            for link in links
            if link.deleted == 0
               and link.user_id == user_id
               and link.agent_id in available_ids
        }
        # expected_ids是用户传过来的当前页面的状态
        # 如果状态和数据库存储的状态不一致 则证明用户拿的是旧快照即在用户拿到快照后已经被其他人修改了
        if current_ids != expected_ids:
            raise BizError(
                "关联已被其他页面修改，请刷新后重试",
                code=409,
            )
        # 俩个集合 可以用减法 取差集
        invalid_ids = requested_ids - available_ids
        # 理论用户传入的都是在可选的智能体上 如果有差集则传的数组有误
        if invalid_ids:
            raise BizError(
                "只能关联当前用户 owner 设备上的有效智能体",
                code=403,
            )
        # 生成这样的对象 { 1: Link(user_id=1, agent_id=1, knowledge_id=10),}
        links_by_agent_id = {
            int(link.agent_id): link
            for link in links
        }

        # 取消未选中的有效关联。
        for link in links:
            if link.deleted == 0 and link.agent_id not in requested_ids:
                link.deleted = 1
                link.deleted_at = func.now()

        # 恢复旧行或新增选中的关联。 这里是遍历用户要给知识库关联的智能体然后从查出来的当前links里查找如果有则是恢复没有则是新增记录
        for agent_id in requested_ids:
            link = links_by_agent_id.get(agent_id)
            if link is None:
                db.add(
                    AgentKnowledgeBase(
                        user_id=user_id,
                        agent_id=agent_id,
                        knowledge_base_id=knowledge_base_id,
                        deleted=0,
                    )
                )
            else:
                link.user_id = user_id
                link.deleted = 0
                link.deleted_at = None

        await db.commit()

    except IntegrityError as exc:
        await db.rollback()
        raise BizError(
            "关联保存冲突，请刷新后重试",
            code=409,
        ) from exc
    except Exception:
        await db.rollback()
        raise
    # 修改后直接返回当前用户当前知识库关联的智能体 即修改后的
    return await list_agents_for_knowledge_base(
        db=db,
        user_id=user_id,
        knowledge_base_id=knowledge_base_id,
    )


def _delete_vectors_job_key(
        document_id: int,
        delete_before_generation: int,
        collection_name: str,
) -> str:
    identity = {
        "job_type": "delete_vectors",
        "document_id": document_id,
        "ingest_generation": delete_before_generation,
        "delete_before_generation": delete_before_generation,
        "target_resource": collection_name,
    }
    raw = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _soft_delete_document_in_tx(
        db: AsyncSession,
        *,
        document: KnowledgeDocument,
        related_jobs: list[KnowledgeIngestJob],
        now,
) -> list[int]:
    if document.deleted == 1:
        return []
    delete_before_generation = document.ingest_generation + 1
    document.deleted = 1
    document.deleted_at = now
    document.ingest_generation = delete_before_generation
    # 先撤销所有代次的切块。检索会立刻排除它们。
    await db.execute(
        update(KnowledgeChunk)
        .where(
            KnowledgeChunk.document_id == document.id,
            KnowledgeChunk.deleted == 0,
            KnowledgeChunk.ingest_generation
            < delete_before_generation,
        )
        .values(deleted=1, deleted_at=now)
    )
    # 取消尚未完成的入库任务，防止它们再发布文档。
    for job in related_jobs:
        if (
                job.job_type == "ingest"
                and job.status in {"queued", "running"}
                and job.ingest_generation < delete_before_generation
        ):
            job.status = "cancelled"
            job.finished_at = now
            job.locked_by = None
            job.lease_token = None
            job.locked_at = None
            job.locked_until = None
        # 查历史入库快照里的 Collection，避免旧代次曾写入其他 Collection。
    collections = {
        str(job.config_snapshot["vector_collection"])
        for job in related_jobs
        if (
                job.job_type == "ingest"
                and job.config_snapshot
                and job.config_snapshot.get("vector_collection")
        )
    }
    if document.vector_collection:
        collections.add(document.vector_collection)
    delete_jobs: list[KnowledgeIngestJob] = []
    for collection_name in sorted(collections):
        snapshot = {"vector_collection": collection_name}
        delete_job = KnowledgeIngestJob(
            document_id=document.id,
            knowledge_base_id=document.knowledge_base_id,
            user_id=document.user_id,
            job_type="delete_vectors",
            stage="delete_vectors",
            status="queued",
            idempotency_key=_delete_vectors_job_key(
                document.id,
                delete_before_generation,
                collection_name,
            ),
            ingest_generation=delete_before_generation,
            delete_before_generation=delete_before_generation,
            config_snapshot=snapshot,
            result_json={},
            attempt_count=0,
            max_attempts=3,
        )
        db.add(delete_job)
        delete_jobs.append(delete_job)

    await db.flush()
    return [job.id for job in delete_jobs]


async def delete_knowledge_document_service(
        db: AsyncSession,
        user_id: int,
        document_id: int,
) -> tuple[KnowledgeDocument, list[int]]:
    reference = await db.execute(
        select(KnowledgeDocument.knowledge_base_id).where(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.user_id == user_id,
        )
    )
    knowledge_base_id = reference.scalar_one_or_none()
    if knowledge_base_id is None:
        raise BizError("文档不存在", code=404)
        # 先锁知识库，防止同时创建文档或删除知识库。
    knowledge_base = await db.scalar(
        select(KnowledgeBase)
        .where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.deleted == 0,
        )
        .with_for_update()
    )
    if knowledge_base is None:
        raise BizError("知识库不存在", code=404)
    # 锁定文档相关任务，和 Worker 的“先锁任务、再写文档”顺序一致。
    jobs_result = await db.scalars(
        select(KnowledgeIngestJob)
        .where(KnowledgeIngestJob.document_id == document_id)
        .order_by(KnowledgeIngestJob.id.asc())
        .with_for_update()
    )
    related_jobs = list(jobs_result.all())

    document = await db.scalar(
        select(KnowledgeDocument)
        .where(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.user_id == user_id,
            KnowledgeDocument.knowledge_base_id == knowledge_base_id,
        )
        .with_for_update()
    )
    if document is None:
        raise BizError("文档不存在", code=404)

    now = await db.scalar(select(func.now()))
    delete_job_ids = await _soft_delete_document_in_tx(
        db,
        document=document,
        related_jobs=related_jobs,
        now=now,
    )

    knowledge_base.updated_at = now
    await db.commit()
    await db.refresh(document)
    return document, delete_job_ids


async def delete_knowledge_base_service(
        db: AsyncSession,
        user_id: int,
        knowledge_base_id: int,
) -> tuple[KnowledgeBase, list[int]]:
    """同一事务中撤销知识库、文档和关联，并创建向量清理任务。"""
    knowledge_base = await db.scalar(
        select(KnowledgeBase)
        .where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
        )
        .with_for_update()
    )
    if knowledge_base is None:
        raise BizError("知识库不存在", code=404)
    if knowledge_base.deleted == 1:
        await db.commit()
        return knowledge_base, []

    # 与单文档删除和 Worker 保持先任务、后文档的锁定顺序。
    jobs_result = await db.scalars(
        select(KnowledgeIngestJob)
        .where(
            KnowledgeIngestJob.knowledge_base_id == knowledge_base_id,
            KnowledgeIngestJob.user_id == user_id,
        )
        .order_by(KnowledgeIngestJob.id.asc())
        .with_for_update()
    )
    jobs_by_document: dict[int, list[KnowledgeIngestJob]] = {}
    for job in jobs_result.all():
        jobs_by_document.setdefault(job.document_id, []).append(job)

    documents_result = await db.scalars(
        select(KnowledgeDocument)
        .where(
            KnowledgeDocument.knowledge_base_id == knowledge_base_id,
            KnowledgeDocument.user_id == user_id,
            KnowledgeDocument.deleted == 0,
        )
        .order_by(KnowledgeDocument.id.asc())
        .with_for_update()
    )
    now = await db.scalar(select(func.clock_timestamp()))
    delete_job_ids: list[int] = []
    for document in documents_result.all():
        delete_job_ids.extend(
            await _soft_delete_document_in_tx(
                db,
                document=document,
                related_jobs=jobs_by_document.get(document.id, []),
                now=now,
            )
        )

    await db.execute(
        update(AgentKnowledgeBase)
        .where(
            AgentKnowledgeBase.knowledge_base_id == knowledge_base_id,
            AgentKnowledgeBase.deleted == 0,
        )
        .values(deleted=1, deleted_at=now)
    )
    knowledge_base.deleted = 1
    knowledge_base.deleted_at = now
    knowledge_base.updated_at = now

    await db.commit()
    return knowledge_base, delete_job_ids


async def _check_document_quota(
        db: AsyncSession,
        *,
        user_id: int,
        knowledge_base_id: int,
        additional_bytes: int,
        now,
) -> None:
    """校验知识库文档数量是否上限，以及容量是否上限"""
    result = await db.execute(

        select(
            func.count(KnowledgeDocument.id),
            # 这里的coalesce是用于返回非null的值 等于说如果没有文档的情况下这里取0
            func.coalesce(
                func.sum(KnowledgeDocument.file_size),
                0,
            ),
        ).outerjoin(
            KnowledgeDocumentUpload,
            KnowledgeDocumentUpload.document_id
            == KnowledgeDocument.id,
        ).where(
            KnowledgeDocument.user_id == user_id,
            KnowledgeDocument.knowledge_base_id
            == knowledge_base_id,
            KnowledgeDocument.deleted == 0,
            # 已过期且未确认的上传会话不再预占额度。
            or_(
                KnowledgeDocument.parse_status != "awaiting_upload",
                KnowledgeDocumentUpload.upload_confirmed_at.is_not(None),
                KnowledgeDocumentUpload.upload_deadline > now,
                KnowledgeDocumentUpload.document_id.is_(None),
            ),
        )
    )
    document_count, total_bytes = result.one()
    if document_count >= settings.knowledge_max_documents_per_base:
        raise BizError("知识库文档数量已达到上限", code=409)
    if (
            total_bytes + additional_bytes
            > settings.knowledge_max_base_content_bytes
    ):
        raise BizError("知识库容量不足", code=409)


async def _build_upload_init_response(
        db: AsyncSession,
        document: KnowledgeDocument,
        upload: KnowledgeDocumentUpload,
        now,
) -> dict:
    """创建上传知识库文件的返回对象"""
    result = {
        "document_id": str(document.id),
        "upload_id": str(upload.upload_id),
        "parse_status": document.parse_status,
        "job_id": None,
        "upload_deadline": upload.upload_deadline,
        "upload": None,
    }
    # 已确认的会话只返回当前状态，不再签发上传表单。
    if upload.upload_confirmed_at is not None:
        job_id = await db.scalar(
            select(KnowledgeIngestJob.id).where(
                KnowledgeIngestJob.document_id == document.id,
                KnowledgeIngestJob.ingest_generation
                == document.ingest_generation,
                KnowledgeIngestJob.job_type == "ingest",
            )
        )
        result["job_id"] = str(job_id) if job_id is not None else None
        return result
    if upload.upload_deadline <= now:
        raise BizError("上传会话已过期，请使用新的请求键重新申请", code=410)
    storage_class = upload.config_snapshot.get("cos_storage_class")
    if not storage_class:
        raise BizError("上传会话配置缺失，请重新申请", code=409)
    result["upload"] = build_cos_upload_form(
        bucket=upload.storage_bucket,
        region=upload.storage_region,
        upload_key=upload.upload_key,
        file_size=document.file_size,
        mime_type=document.mime_type,
        storage_class=storage_class,
        upload_deadline=upload.upload_deadline,
    )
    return result


async def init_document_upload(
        db: AsyncSession,
        *,
        user_id: int,
        knowledge_base_id: int,
        payload: KnowledgeUploadInitRequest,
        idempotency_key: str,
) -> dict:
    file_name = payload.file_name

    if any(char in file_name for char in ("/", "\\", "\x00")):
        raise BizError("file_name 必须是文件名，不能包含目录", code=422)

    if Path(file_name).suffix.lower() != ".docx":
        raise BizError("当前暂时只支持 DOCX 文件", code=422)

    if payload.file_size > settings.knowledge_max_file_bytes:
        raise BizError("单文件不能超过 10 MiB", code=413)

    # 只初始化客户端并检查配置，这里不会调用 COS 网络接口。
    get_knowledge_cos_client()

    request_data = {
        "request_type": "cos_upload",
        "knowledge_base_id": knowledge_base_id,
        **payload.model_dump(),
    }
    request_hash = hashlib.sha256(
        json.dumps(
            request_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    try:
        # 锁定知识库，把同一个库的额度检查和文档创建串行化。
        knowledge_base = await db.scalar(
            select(KnowledgeBase).where(
                KnowledgeBase.id == knowledge_base_id,
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.deleted == 0,
            )
            .with_for_update()
        )
        if knowledge_base is None:
            raise BizError("知识库不存在", code=404)

        if knowledge_base.status != "active":
            raise BizError("知识库当前不可用", code=409)
        now = await db.scalar(select(func.now()))

        existing = await db.scalar(
            select(KnowledgeDocument).where(
                KnowledgeDocument.user_id == user_id,
                KnowledgeDocument.create_request_key
                == idempotency_key,
            )
        )

        if existing is not None:
            if existing.create_request_hash != request_hash:
                raise BizError(
                    "同一个 Idempotency-Key 不能用于不同内容",
                    code=409,
                )

            if existing.deleted != 0:
                raise BizError(
                    "原文档已删除，请使用新的 Idempotency-Key",
                    code=409,
                )
            upload = await db.get(
                KnowledgeDocumentUpload,
                existing.id,
            )
            if upload is None:
                raise BizError("原上传记录不存在", code=409)

            result = await _build_upload_init_response(
                db, existing, upload, now
            )
            await db.commit()
            return result
        #     不存在的情况
        await _check_document_quota(
            db,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            additional_bytes=payload.file_size,
            now=now,
        )
        snapshot = _manual_ingest_snapshot()
        snapshot.update(
            {
                "parser_version": settings.knowledge_file_parser_version,
                "file_format": "docx",
                "cos_storage_class": settings.knowledge_cos_storage_class,
            }
        )
        document = KnowledgeDocument(
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            source_type="file",
            title=payload.title,
            file_name=file_name,
            mime_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            file_size=payload.file_size,
            storage_backend="cos",
            raw_content=None,
            parsed_text=None,
            parsed_elements_json=None,
            content_hash=None,
            create_request_key=idempotency_key,
            create_request_hash=request_hash,
            parse_status="awaiting_upload",
            ingest_generation=1,
            parser_version=snapshot["parser_version"],
            chunker_version=snapshot["chunker_version"],
            embedding_profile=snapshot["embedding_profile"],
            embedding_model=snapshot["embedding_model"],
            embedding_dimension=snapshot["embedding_dimension"],
            vector_collection=snapshot["vector_collection"],
            chunk_count=0,
            warnings_json=[],
            deleted=0,
        )
        db.add(document)
        await db.flush()
        upload_id = uuid4()
        incoming_prefix = settings.knowledge_cos_incoming_prefix.strip("/")
        source_prefix = settings.knowledge_cos_source_prefix.strip("/")

        if (
                not incoming_prefix
                or not source_prefix
                or incoming_prefix == source_prefix
        ):
            raise RuntimeError("COS 暂存和正式对象前缀配置不正确")
        upload = KnowledgeDocumentUpload(
            document_id=document.id,
            storage_bucket=settings.cos_bucket.strip(),
            storage_region=settings.cos_region.strip(),
            upload_id=upload_id,
            upload_key=(
                f"{incoming_prefix}/{user_id}/"
                f"{knowledge_base_id}/{document.id}/{upload_id}"
            ),
            storage_key=(
                f"{source_prefix}/{user_id}/{knowledge_base_id}/"
                f"{document.id}/g1/{upload_id}.docx"
            ),
            upload_expected_sha256=payload.sha256,
            upload_deadline=now + timedelta(
                seconds=settings.knowledge_upload_session_seconds
            ),
            config_snapshot=snapshot,
        )
        db.add(upload)
        knowledge_base.updated_at = now
        result = await _build_upload_init_response(
            db, document, upload, now
        )
        # 两张表同一事务提交；此时还不创建 ingest 任务。
        await db.commit()
        return result
    except IntegrityError as exc:
        await db.rollback()
        raise BizError(
            "上传申请发生并发冲突，请使用相同请求重试",
            code=409,
        ) from exc

    except Exception:
        await db.rollback()
        raise


def _check_cos_upload_record(
        document: KnowledgeDocument,
        upload: KnowledgeDocumentUpload | None,
        upload_id: UUIDValue,
) -> None:
    if (
            document.storage_backend != 'cos'
            or document.source_type not in {"file", "qa_csv"}
    ):
        raise BizError("该文档不是 COS 文件", code=409)

    if upload is None:
        raise BizError("上传记录不存在", code=404)

    if upload.upload_id != upload_id:
        raise BizError("上传会话不匹配", code=409)


def _upload_complete_response(
        document: KnowledgeDocument,
        upload: KnowledgeDocumentUpload,
        job: KnowledgeIngestJob,
) -> dict:
    return {
        "document_id": str(document.id),
        "job_id": str(job.id),
        "parse_status": document.parse_status,
        "source_verified_at": upload.source_verified_at,
        "ingest_generation": document.ingest_generation,
    }


async def complete_document_upload(
        db: AsyncSession,
        *,
        user_id: int,
        document_id: int,
        upload_id: UUIDValue,
) -> dict:
    try:
        # 第一段：读取当前用户拥有的文档和上传记录。
        document = await db.scalar(
            select(KnowledgeDocument)
            .join(
                KnowledgeBase,
                KnowledgeBase.id
                == KnowledgeDocument.knowledge_base_id,
            )
            .where(
                KnowledgeDocument.id == document_id,
                KnowledgeDocument.user_id == user_id,
                KnowledgeDocument.deleted == 0,
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.deleted == 0,
            )
        )

        if document is None:
            raise BizError("文档不存在", code=404)

        upload = await db.get(
            KnowledgeDocumentUpload,
            document_id,
        )
        # 显式判断，让 IDE 知道后面的 upload 一定不是 None。
        if upload is None:
            raise BizError("上传记录不存在", code=404)
        _check_cos_upload_record(document, upload, upload_id)
        # 明确告诉 IDE：这是可能为空的日期值。
        confirmed_at = cast(
            datetime | None,
            upload.upload_confirmed_at,
        )
        # 成功确认过的请求直接返回原任务。
        # 即使会话期限后来已经过去，也不能重复创建任务。
        if confirmed_at is not None:
            job = await db.scalar(
                select(KnowledgeIngestJob).where(
                    KnowledgeIngestJob.document_id == document_id,
                    KnowledgeIngestJob.ingest_generation
                    == document.ingest_generation,
                    KnowledgeIngestJob.job_type == "ingest",
                )
            )
            if job is None:
                raise BizError("上传确认记录与任务不一致", code=409)

            result = _upload_complete_response(document, upload, job)
            await db.commit()
            return result

        now = await db.scalar(select(func.clock_timestamp()))

        if upload.upload_deadline <= now:
            raise BizError("上传会话已过期，请重新申请", code=410)

        if document.parse_status != "awaiting_upload":
            raise BizError("当前文档状态不能确认上传", code=409)

        # 保存普通值，后面重新查询并检查是否发生变化。
        reference = {
            "knowledge_base_id": document.knowledge_base_id,
            "ingest_generation": document.ingest_generation,
            "bucket": upload.storage_bucket,
            "region": upload.storage_region,
            "key": upload.upload_key,
            "file_size": document.file_size,
        }

        # 结束读取事务，COS 网络请求不占着数据库事务。
        await db.commit()

        head_result = None
        head_error = None

        try:
            head_result = await asyncio.to_thread(
                head_knowledge_upload_object,
                bucket=reference["bucket"],
                region=reference["region"],
                key=reference["key"],
            )
        except CosServiceError as exc:
            if exc.get_status_code() == 404:
                head_error = BizError(
                    "COS 中尚未找到文件，请先完成上传",
                    code=409,
                )
            else:
                head_error = BizError(
                    "暂时无法访问 COS，请稍后重试",
                    code=503,
                )
        except CosClientError:
            head_error = BizError(
                "COS 网络请求失败，请稍后重试",
                code=503,
            )
        except ValueError:
            head_error = BizError(
                "COS 返回的文件元数据无效",
                code=502,
            )

        # 第二段：重新校验并锁定相关记录。
        # 与删除流程保持知识库 → 任务 → 文档的锁顺序。
        knowledge_base = await db.scalar(
            select(KnowledgeBase)
            .where(
                KnowledgeBase.id == reference["knowledge_base_id"],
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.deleted == 0,
            )
            .with_for_update()
        )
        if knowledge_base is None:
            raise BizError("知识库不存在", code=404)

        job = await db.scalar(
            select(KnowledgeIngestJob)
            .where(
                KnowledgeIngestJob.document_id == document_id,
                KnowledgeIngestJob.ingest_generation
                == reference["ingest_generation"],
                KnowledgeIngestJob.job_type == "ingest",
            )
            .with_for_update()
        )

        document = await db.scalar(
            select(KnowledgeDocument)
            .where(
                KnowledgeDocument.id == document_id,
                KnowledgeDocument.user_id == user_id,
                KnowledgeDocument.knowledge_base_id
                == reference["knowledge_base_id"],
                KnowledgeDocument.deleted == 0,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if document is None:
            raise BizError("文档不存在或已删除", code=404)

        upload = await db.scalar(
            select(KnowledgeDocumentUpload)
            .where(
                KnowledgeDocumentUpload.document_id == document_id
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        _check_cos_upload_record(document, upload, upload_id)

        if document.ingest_generation != reference["ingest_generation"]:
            raise BizError("文档处理代次已变化，请刷新后重试", code=409)

        # HEAD 等待期间，另一个请求可能已完成确认。
        if upload.upload_confirmed_at is not None:
            if job is None:
                raise BizError("上传确认记录与任务不一致", code=409)

            result = _upload_complete_response(document, upload, job)
            await db.commit()
            return result

        now = await db.scalar(select(func.clock_timestamp()))

        if upload.upload_deadline <= now:
            raise BizError("上传会话已过期，请重新申请", code=410)

        if knowledge_base.status != "active":
            raise BizError("知识库当前不可用", code=409)

        if document.parse_status != "awaiting_upload":
            raise BizError("当前文档状态不能确认上传", code=409)

        if (
                upload.storage_bucket != reference["bucket"]
                or upload.storage_region != reference["region"]
                or upload.upload_key != reference["key"]
                or document.file_size != reference["file_size"]
        ):
            raise BizError("上传信息已变化，请刷新后重试", code=409)

        if head_error is not None:
            raise head_error

        if head_result["size"] != document.file_size:
            raise BizError(
                "实际文件大小与申请时不一致，请重新上传",
                code=409,
            )

        if job is not None:
            raise BizError("任务已存在，但上传状态不一致", code=409)

        # 使用申请时保存的配置，不重新读取全局配置替换。
        snapshot = deepcopy(upload.config_snapshot)

        required_fields = {
            "parser_version",
            "chunker_version",
            "embedding_profile",
            "embedding_base_url",
            "embedding_model",
            "embedding_dimension",
            "metric_type",
            "vector_collection",
            "chunk_target_chars",
            "chunk_overlap_chars",
            "file_format",
            "cos_storage_class",
        }

        if not required_fields.issubset(snapshot):
            raise BizError("上传配置快照不完整，请重新申请", code=409)

        for field in (
                "parser_version",
                "chunker_version",
                "embedding_profile",
                "embedding_model",
                "embedding_dimension",
                "vector_collection",
        ):
            if snapshot[field] != getattr(document, field):
                raise BizError("文档与上传配置不一致", code=409)

        job = KnowledgeIngestJob(
            document_id=document.id,
            knowledge_base_id=document.knowledge_base_id,
            user_id=document.user_id,
            job_type="ingest",
            stage="verify",
            status="queued",
            idempotency_key=_ingest_job_key(
                document.id,
                document.ingest_generation,
                snapshot,
            ),
            ingest_generation=document.ingest_generation,
            delete_before_generation=None,
            config_snapshot=snapshot,
            result_json={},
            attempt_count=0,
            max_attempts=3,
            available_at=now,
        )
        db.add(job)

        upload.upload_confirmed_at = now
        upload.upload_etag = head_result["etag"]

        document.parse_status = "uploaded"
        document.error_code = None
        document.error_message = None
        knowledge_base.updated_at = now

        await db.flush()

        result = _upload_complete_response(document, upload, job)
        await db.commit()
        return result

    except IntegrityError as exc:
        await db.rollback()
        raise BizError(
            "上传确认发生并发冲突，请使用相同 upload_id 重试",
            code=409,
        ) from exc

    except Exception:
        await db.rollback()
        raise
