import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import BizError
from app.integrations.embedding import embed_text
from app.integrations.knowledge_milvus import search_knowledge_chunks, search_knowledge_chunks_for_bases
from app.models import KnowledgeBase, KnowledgeChunk, KnowledgeDocument, AgentKnowledgeBase, Agent, Device, UserDevice
from app.schemas.knowledge import KnowledgeSearchHit


async def search_knowledge(
        db: AsyncSession,
        user_id: int,
        knowledge_base_id: int,
        query: str,
) -> list[KnowledgeSearchHit]:
    query = query.strip()
    if not query:
        raise BizError(code=422, message="搜索内容不能为空")
    # 先验证当前用户有权使用这个知识库。
    knowledge_base = await db.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.deleted == 0,
            KnowledgeBase.status == "active",
        )
    )
    if knowledge_base is None:
        raise BizError(code=404, message="知识库不存在或不可用")
    query_vector = await embed_text(query)
    # 查询相关知识
    milvus_result = await asyncio.to_thread(
        search_knowledge_chunks,
        query_vector,
        user_id,
        knowledge_base_id,
        settings.knowledge_retrieval_top_k,
    )
    # 没有相关知识
    if not milvus_result or not milvus_result[0]:
        return []
    hits = []
    for hit in milvus_result[0]:
        entity = hit.get('entity') or {}
        chunk_id = hit.get('id', entity.get("chunk_id"))
        if chunk_id is not None:
            hits.append((int(chunk_id), float(hit.get("distance", 0.0))))
    if not hits:
        return []
    chunk_ids = [chunk_id for chunk_id, _ in hits]
    db_query = (select(KnowledgeChunk, KnowledgeDocument)
    .join(
        KnowledgeDocument,
        KnowledgeDocument.id == KnowledgeChunk.document_id,
    ).join(
        KnowledgeBase,
        KnowledgeBase.id == KnowledgeChunk.knowledge_base_id,
    ).where(
        # id是在向量数据库查出来的id中
        KnowledgeChunk.id.in_(chunk_ids),
        KnowledgeChunk.user_id == user_id,
        KnowledgeChunk.knowledge_base_id == knowledge_base_id,
        KnowledgeChunk.deleted == 0,
        KnowledgeChunk.vector_sync_status == "synced",
        KnowledgeDocument.user_id == user_id,
        KnowledgeDocument.knowledge_base_id == knowledge_base_id,
        KnowledgeDocument.deleted == 0,
        KnowledgeDocument.parse_status == "ready",
        KnowledgeDocument.ingest_generation
        == KnowledgeChunk.ingest_generation,
        KnowledgeBase.user_id == user_id,
        KnowledgeBase.deleted == 0,
        KnowledgeBase.status == "active",
    ))
    result = await db.execute(db_query)
    # 构建成 {101:(chunk,document)}这样的对象
    rows_by_id = {
        chunk.id: (chunk, document)
        for chunk, document in result.all()
    }
    # PostgreSQL 查询结果顺序不保证与 Milvus 一样，所以按 Milvus 命中顺序重排。
    response = []
    for chunk_id, score in hits:
        row = rows_by_id.get(chunk_id)
        if row is None:
            continue
        chunk, document = row
        response.append(
            KnowledgeSearchHit(
                chunk_id=chunk.id,
                document_id=document.id,
                title=document.title,
                content=chunk.content,
                score=score,
            )
        )
    return response


async def get_agent_knowledge_base_ids(
        db: AsyncSession,
        *,
        user_id: int,
        agent_id: int,
) -> list[int]:
    """返回当前用户的有效智能体可使用的知识库 ID。"""
    stmt = (
        select(KnowledgeBase.id)
        .join(
            AgentKnowledgeBase,
            AgentKnowledgeBase.knowledge_base_id == KnowledgeBase.id,
        ).join(
            Agent,
            Agent.id == AgentKnowledgeBase.agent_id,
        ).join(
            Device,
            Device.id == Agent.device_id,
        ).join(
            UserDevice,
            UserDevice.device_id == Device.id,
        ).where(
            # 关系属于当前用户，且仍有效
            AgentKnowledgeBase.user_id == user_id,
            AgentKnowledgeBase.agent_id == agent_id,
            AgentKnowledgeBase.deleted == 0,

            # 知识库属于当前用户，且启用、未删除
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.deleted == 0,
            KnowledgeBase.status == "active",

            # 智能体位于当前用户拥有的有效设备上
            Agent.deleted == 0,
            Agent.is_system.is_(False),
            Agent.device_id.is_not(None),
            Device.deleted == 0,
            Device.status == 1,
            UserDevice.user_id == user_id,
            UserDevice.role == "owner",
            UserDevice.deleted == 0,
        )
        .distinct()
        .order_by(KnowledgeBase.id)
    )
    result = await db.scalars(stmt)
    return list(result.all())


async def retrieve_agent_knowledge(
        db: AsyncSession,
        *,
        user_id: int,
        agent_id: int,
        query: str,
        limit: int | None = None,
        query_vector: list[float] | None = None,
) -> list[KnowledgeSearchHit]:
    query = query.strip()
    if not query:
        raise BizError(code=422, message="搜索内容不能为空")
    if limit is None:
        limit = settings.knowledge_retrieval_top_k
    if limit <= 0:
        raise BizError(code=422, message="limit 必须大于 0")

    # 只从数据库读取这个用户、这个智能体当前可用的关联库。
    knowledge_base_ids = await get_agent_knowledge_base_ids(
        db,
        user_id=user_id,
        agent_id=agent_id,
    )
    # 没有可用知识库
    if not knowledge_base_ids:
        return []

    # 同一个问题只向 Embedding 服务请求一次向量。
    if query_vector is None:
        query_vector = await embed_text(query)

    # 多取一些候选，供后面的 PostgreSQL 有效性检查过滤。 最少20条
    # 因为这里可能存在 已经被删除 的停用的等等情况
    candidate_limit = min(100, max(20, limit * 4))

    milvus_result = await asyncio.to_thread(
        search_knowledge_chunks_for_bases,
        query_vector,
        user_id,
        knowledge_base_ids,
        candidate_limit,
    )

    if not milvus_result or not milvus_result[0]:
        return []

    # Milvus 返回 chunk ID 和相似度；之后按 ID 顺序从 PostgreSQL 复核。
    ranked_hits: list[tuple[int, float]] = []
    for hit in milvus_result[0]:
        entity = hit.get("entity") or {}
        chunk_id = hit.get("id")
        if chunk_id is None:
            chunk_id = entity.get("chunk_id")
        if chunk_id is not None:
            ranked_hits.append((
                int(chunk_id),
                float(hit.get("distance") or 0.0),
            ))
    if not ranked_hits:
        return []

    chunk_ids = [chunk_id for chunk_id, _ in ranked_hits]

    db_query = (
        select(KnowledgeChunk, KnowledgeDocument).join(
            KnowledgeDocument,
            KnowledgeDocument.id == KnowledgeChunk.document_id,
        )
        .join(
            KnowledgeBase,
            KnowledgeBase.id == KnowledgeChunk.knowledge_base_id,
        )
        .join(
            AgentKnowledgeBase,
            AgentKnowledgeBase.knowledge_base_id == KnowledgeBase.id,
        )
        .join(Agent, Agent.id == AgentKnowledgeBase.agent_id)
        .join(Device, Device.id == Agent.device_id)
        .join(UserDevice, UserDevice.device_id == Device.id)
        .where(
            # 确认召回的切块仍有效，而且属于当前用户的关联库。
            KnowledgeChunk.id.in_(chunk_ids),
            KnowledgeChunk.user_id == user_id,
            KnowledgeChunk.knowledge_base_id.in_(knowledge_base_ids),
            KnowledgeChunk.deleted == 0,
            KnowledgeChunk.vector_sync_status == "synced",

            # 确认文档可用，且切块属于文档的当前处理代次和模型配置。
            KnowledgeDocument.user_id == user_id,
            KnowledgeDocument.deleted == 0,
            KnowledgeDocument.parse_status == "ready",
            KnowledgeDocument.ingest_generation
            == KnowledgeChunk.ingest_generation,
            KnowledgeDocument.embedding_profile
            == settings.knowledge_embedding_profile,
            KnowledgeDocument.vector_collection
            == settings.milvus_knowledge_collection,
            # 再确认库、关联、智能体和设备仍有效。
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.deleted == 0,
            KnowledgeBase.status == "active",
            AgentKnowledgeBase.user_id == user_id,
            AgentKnowledgeBase.agent_id == agent_id,
            AgentKnowledgeBase.deleted == 0,
            Agent.deleted == 0,
            Agent.is_system.is_(False),
            Device.deleted == 0,
            Device.status == 1,
            UserDevice.user_id == user_id,
            UserDevice.role == "owner",
            UserDevice.deleted == 0,
        )
    )
    result = await db.execute(db_query)
    rows_by_id = {
        chunk.id: (chunk, document)
        for chunk, document in result.all()
    }
    # Milvus 返回 chunk ID 和相似度；之后按 ID 顺序从 PostgreSQL 复核。
    # 数据库结果顺序不保证和 Milvus 一样，因此按 Milvus 的排名恢复顺序。
    response: list[KnowledgeSearchHit] = []
    for chunk_id, score in ranked_hits:
        row = rows_by_id.get(chunk_id)
        if row is None:
            continue

        chunk, document = row
        response.append(
            KnowledgeSearchHit(
                chunk_id=chunk.id,
                document_id=document.id,
                title=document.title,
                content=chunk.content,
                score=score,
            )
        )
        if len(response) >= limit:
            break

    return response
