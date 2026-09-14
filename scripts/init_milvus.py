"""初始化或显式重建 Milvus Collection。

默认只负责创建 Collection 和向量索引，不负责写入业务数据；Collection
已存在时不会删除或重建，避免误伤已有数据。

当线上还是测试向量数据、需要按新字段重建时，必须同时传入
``--recreate --confirm-drop``。双开关防止把这个脚本误用于生产数据清库。
"""

import argparse

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
        field_name="agent_id",
        datatype=DataType.INT64,
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


def parse_args() -> argparse.Namespace:
    """解析危险操作的双重确认参数。"""

    parser = argparse.ArgumentParser(description="初始化 Milvus 对话摘要 Collection")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Collection 已存在时删除并按当前 Schema 重建",
    )
    parser.add_argument(
        "--confirm-drop",
        action="store_true",
        help="确认允许删除已有 Collection 及其中的全部向量数据",
    )
    args = parser.parse_args()
    if args.confirm_drop and not args.recreate:
        parser.error("--confirm-drop 必须与 --recreate 一起使用")
    if args.recreate and not args.confirm_drop:
        parser.error("重建 Collection 必须同时传入 --confirm-drop")
    return args


def main() -> None:
    args = parse_args()
    collection_name = settings.milvus_summary_collection
    client = MilvusClient(uri=settings.milvus_uri)

    print(f"Milvus 地址：{settings.milvus_uri}")
    print(f"Collection：{collection_name}")
    print(f"向量维度：{settings.embedding_dimension}")

    if client.has_collection(collection_name=collection_name):
        if args.recreate:
            # 仅在用户明确授权的测试库升级场景执行。此操作不可恢复，正式库
            # 必须改用“新 Collection + 数据迁移 + 切换名称”的迁移策略。
            print(f"正在删除已有 Collection：{collection_name}")
            client.drop_collection(collection_name=collection_name)
            print(f"已删除 Collection：{collection_name}")
        else:
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
