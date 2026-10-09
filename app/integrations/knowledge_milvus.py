from pymilvus import MilvusClient

from app.core.config import settings

KNOWLEDGE_OUTPUT_FIELDS = [
    "chunk_id",
    "user_id",
    "knowledge_base_id",
    "document_id",
    "ingest_generation",
]


def get_knowledge_milvus_client() -> MilvusClient:
    return MilvusClient(uri=settings.milvus_uri)


"""批量插入"""


def upsert_knowledge_chunks(
        chunks: list,
        vectors: list[list[float]],
) -> dict:
    if len(chunks) != len(vectors):
        raise ValueError("chunk 数量与向量数量不一致")
    data = []
    client = get_knowledge_milvus_client()
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != settings.embedding_dimension:
            raise ValueError("Embedding 向量维度不匹配")
        data.append({
            "chunk_id": chunk.id,
            "embedding": vector,
            "user_id": chunk.user_id,
            "knowledge_base_id": chunk.knowledge_base_id,
            "document_id": chunk.document_id,
            "ingest_generation": chunk.ingest_generation,
        })

    return client.upsert(
        collection_name=settings.milvus_knowledge_collection,
        data=data,
    )


def search_knowledge_chunks(
        query_vector: list[float],
        user_id: int,
        knowledge_base_id: int,
        limit: int,
) -> list:
    if len(query_vector) != settings.embedding_dimension:
        raise ValueError(
            f"查询向量维度不匹配：实际 {len(query_vector)}，"
            f"配置 {settings.embedding_dimension}"
        )
    filter_expression = (
        f"user_id == {user_id} "
        f"and knowledge_base_id == {knowledge_base_id}"
    )
    client = get_knowledge_milvus_client()

    return client.search(
        collection_name=settings.milvus_knowledge_collection,
        data=[query_vector],
        anns_field="embedding",
        limit=limit,
        filter=filter_expression,
        output_fields=KNOWLEDGE_OUTPUT_FIELDS,
    )


def search_knowledge_chunks_for_bases(
        query_vector: list[float],
        user_id: int,
        knowledge_base_ids: list[int],
        limit: int,
) -> list:
    if len(query_vector) != settings.embedding_dimension:
        raise ValueError("查询向量维度不匹配")

    safe_user_id = int(user_id)
    safe_knowledge_base_ids = sorted({
        int(knowledge_base_id)
        for knowledge_base_id in knowledge_base_ids
    })

    if safe_user_id <= 0:
        raise ValueError("user_id 必须是正整数")
    if any(knowledge_base_id <= 0 for knowledge_base_id in safe_knowledge_base_ids):
        raise ValueError("knowledge_base_id 必须是正整数")
    if not safe_knowledge_base_ids or limit <= 0:
        return []

    ids_expression = ", ".join(
        str(knowledge_base_id)
        for knowledge_base_id in safe_knowledge_base_ids
    )
    filter_expression = (
        f"user_id == {safe_user_id} "
        f"and knowledge_base_id in [{ids_expression}]"
    )

    client = get_knowledge_milvus_client()
    return client.search(
        collection_name=settings.milvus_knowledge_collection,
        data=[query_vector],
        anns_field="embedding",
        limit=limit,
        filter=filter_expression,
        output_fields=KNOWLEDGE_OUTPUT_FIELDS,
    )


def delete_knowledge_vectors(
        *,
        collection_name: str,
        user_id: int,
        knowledge_base_id: int,
        document_id: int,
        delete_before_generation: int,
) -> dict:
    if min(
            user_id,
            knowledge_base_id,
            document_id,
            delete_before_generation,
    ) <= 0:
        raise ValueError("删除过滤条件里的 ID 和代次必须是正整数")
    filter_expression = (
        f"user_id == {user_id} "
        f"and knowledge_base_id == {knowledge_base_id} "
        f"and document_id == {document_id} "
        f"and ingest_generation < {delete_before_generation}"
    )

    client = get_knowledge_milvus_client()
    return client.delete(
        collection_name=collection_name,
        filter=filter_expression,
    )
