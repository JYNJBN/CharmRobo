from pymilvus import MilvusClient, DataType

from app.core.config import settings


def main() -> None:
    client = MilvusClient(uri=settings.milvus_uri)
    name = settings.milvus_knowledge_collection
    if client.has_collection(collection_name=name):
        print(f"集合已存在：{name}")
        print(client.describe_collection(collection_name=name))
        return

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    schema.add_field(
        field_name="chunk_id",
        datatype=DataType.INT64,
        is_primary=True,
    )

    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=settings.embedding_dimension,
    )
    schema.add_field(field_name="user_id", datatype=DataType.INT64)
    schema.add_field(field_name="knowledge_base_id", datatype=DataType.INT64)
    schema.add_field(field_name="document_id", datatype=DataType.INT64)
    schema.add_field(field_name="ingest_generation", datatype=DataType.INT32)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )

    print(f"创建成功：{name}")
    print(client.describe_collection(collection_name=name))


if __name__ == "__main__":
    main()
