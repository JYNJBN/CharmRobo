from pymilvus import MilvusClient


def main() -> None:
    client = MilvusClient(
        uri="http://127.0.0.1:19530",
    )

    collections = client.list_collections()
    print("milvus链接成功")
    print("当前 Collection：", collections)


if __name__ == "__main__":
    main()
