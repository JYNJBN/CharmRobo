from pymilvus import MilvusClient

from app.core.config import settings

OUTPUT_FIELDS = [
    "summary_id",
    "summary_text",
    "conversation_id",
    "device_id",
    "user_id",
    "status",
]
def get_milvus_client() -> MilvusClient:
    return MilvusClient(
        uri=settings.milvus_uri,
    )
def upsert_summary(
        summary_id: int,
        summary_text: str,
        embedding: list[float],
        conversation_id: int,
        device_id: int,
        user_id: int,
        status: str = "active",
)->dict:
    if len(embedding) != settings.embedding_dimension:
        raise ValueError(
            f"向量维度不匹配：实际 {len(embedding)}，"
            f"配置 {settings.embedding_dimension}"
        )
    client = get_milvus_client()
    return client.upsert(
        collection_name=settings.milvus_summary_collection,
        data=[
            {
            "summary_id": summary_id,
            "summary_text": summary_text,
            "embedding": embedding,
            "conversation_id": conversation_id,
            "device_id": device_id,
            "user_id": user_id,
            "status": status,
        }
        ]
    )
def search_summaries(
    query_vector: list[float],
    user_id: int,
    device_id: int,
    limit: int = 5,
) -> list:
    if len(query_vector) != settings.embedding_dimension:
        raise ValueError(
            f"查询向量维度不匹配：实际 {len(query_vector)}，"
            f"配置 {settings.embedding_dimension}"
        )
    client = get_milvus_client()
    filter_expression=(
        f"user_id == {user_id} "
        f"and device_id == {device_id} "
        f'and status == "active"'
    )
    return client.search(
        collection_name=settings.milvus_summary_collection,
        data=[query_vector],
        anns_field="embedding",
        limit=limit,
        filter=filter_expression,
        output_fields=OUTPUT_FIELDS,
    )
