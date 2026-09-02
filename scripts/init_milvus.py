"""初始化 Milvus Collection。

这个脚本只负责创建 Collection 和向量索引，不负责写入业务数据。
Collection 已存在时不会删除或重建，避免误伤已有数据。
"""

from pymilvus import DataType, MilvusClient

from app.core.config import settings


def build_summary_schema():
    """构造对话摘要 Collection 的 Schema。"""

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    schema.add_field(
        field_name="summary_id",
        datatype=DataType.INT64,
        is_primary=True,
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=settings.embedding_dimension,
    )
    schema.add_field(
        field_name="summary_text",
        datatype=DataType.VARCHAR,
        max_length=4096,
    )
    schema.add_field(
        field_name="conversation_id",
        datatype=DataType.INT64,
    )
    schema.add_field(
        field_name="device_id",
        datatype=DataType.INT64,
    )
    schema.add_field(
        field_name="user_id",
        datatype=DataType.INT64,
    )
    schema.add_field(
        field_name="status",
        datatype=DataType.VARCHAR,
        max_length=20,
    )

    return schema


def build_index_params(client: MilvusClient):
    """构造向量索引参数。"""

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type=settings.embedding_metric_type,
    )
    return index_params


def main() -> None:
    collection_name = settings.milvus_summary_collection
    client = MilvusClient(uri=settings.milvus_uri)

    print(f"Milvus 地址：{settings.milvus_uri}")
    print(f"Collection：{collection_name}")
    print(f"向量维度：{settings.embedding_dimension}")

    if client.has_collection(collection_name=collection_name):
        print(f"Collection 已存在，跳过创建：{collection_name}")
        print(client.describe_collection(collection_name=collection_name))
        return

    client.create_collection(
        collection_name=collection_name,
        schema=build_summary_schema(),
        index_params=build_index_params(client),
    )

    print(f"Collection 创建成功：{collection_name}")
    print(client.describe_collection(collection_name=collection_name))
    print(client.get_load_state(collection_name=collection_name))


if __name__ == "__main__":
    main()
