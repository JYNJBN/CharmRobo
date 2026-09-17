import asyncio

from app.integrations.embedding import embed_text
from app.integrations.milvus import upsert_summary


async def main() -> None:
    summary_text = "用户喜欢晚上听轻音乐。"
    embedding = await embed_text(summary_text)
    # print(embedding)
    result = upsert_summary(
        summary_id=2,
        summary_text=summary_text,
        embedding=embedding,
        conversation_id=1001,
        device_id=2001,
        user_id=3001,
    )

    print("摘要写入成功")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
