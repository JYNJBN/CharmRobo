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


def main() -> None:
    summary_text = "用户喜欢晚上看视频"
    embedding = get_embedding(summary_text)
    client = MilvusClient(
        uri="http://127.0.0.1:19530",
    )

    result = client.upsert(
        collection_name=COLLECTION_NAME,
        data=[
            {
                "summary_id": 1,
                "embedding": embedding,
                "summary_text": summary_text,
                # 换成你实际的 ID
                "conversation_id": 1001,
                "device_id": 2001,
                "user_id": 3001,
                "status": "active",
            }
        ],
    )
    print("写入成功")
    print(result)


if __name__ == "__main__":
    main()
