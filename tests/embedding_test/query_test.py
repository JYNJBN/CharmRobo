from pymilvus import MilvusClient

COLLECTION_NAME = "conversation_summary_v1"


def main() -> None:
    client = MilvusClient(
        uri="http://127.0.0.1:19530",
    )

    result = client.query(
        collection_name=COLLECTION_NAME,
        # filter="summary_id==1",
        output_fields=[
            "summary_id",
            "summary_text",
            "conversation_id",
            "device_id",
            "agent_id",
            "user_id",
            "status",
        ],
        limit=10,
    )
    print("查询结果：")
    print(result)


if __name__ == "__main__":
    main()
