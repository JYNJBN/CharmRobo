from pymilvus import DataType, MilvusClient

COLLECTION_NAME = "conversation_summary_v1"

VECTOR_DIM = 1024


def main() -> None:
    client = MilvusClient(url="http://127.0.0.1:19530")
    if client.has_collection(collection_name=COLLECTION_NAME):
        print(f"Collection 已存在：{COLLECTION_NAME}")
        return
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
        dim=VECTOR_DIM,
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
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    print(f"Collection 创建成功：{COLLECTION_NAME}")
    print(client.describe_collection(collection_name=COLLECTION_NAME))
    print(client.get_load_state(collection_name=COLLECTION_NAME))


if __name__ == "__main__":
    main()
