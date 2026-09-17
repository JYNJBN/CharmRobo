import httpx
from pymilvus import MilvusClient

SILICONFLOW_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "conversation_summary_v1"


def get_embedding(text: str) -> list[float]:
    # api_key = os.getenv("SILICONFLOW_API_KEY")
    api_key = "sk-zgzndnouqesrexdmdxdtopbltxbwjoifsyqnpzmyyoeyvjao"

    if not api_key:
        raise RuntimeError("没有找到 SILICONFLOW_API_KEY")

    response = httpx.post(
        SILICONFLOW_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "input": text,
            "encoding_format": "float",
        },
        timeout=30,
    )

    response.raise_for_status()

    result = response.json()
    return result["data"][0]["embedding"]


def main():
    client = MilvusClient(
        uri="http://127.0.0.1:19530",
    )

    query_text = "这个用户喜欢什么风格的音乐？"
    query_vector = get_embedding(query_text)
    result = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="embedding",
        limit=3,
        filter='user_id == 3001 and device_id == 2001 and status == "active"',
        output_fields=[
            "summary_id",
            "summary_text",
            "conversation_id",
            "device_id",
            "user_id",
            "status",
        ],
    )
    for hits in result:
        for hit in hits:
            print(hit)


if __name__ == "__main__":
    main()
    print("语义搜索结果：")
