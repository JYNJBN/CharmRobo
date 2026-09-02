import asyncio

from app.integrations.embedding import embed_text
from app.integrations.milvus import search_summaries


async def main() -> None:
    query_text = "用户喜欢什么音乐？"

    query_vector = await embed_text(query_text)

    results = search_summaries(
        query_vector=query_vector,
        user_id=3001,
        device_id=2001,
        limit=3,
    )

    print("语义搜索结果：results",results)
    memory_lines: list[str] = []

    for hits in results:
        for hit in hits:
            entity = hit.get("entity", {})
            summary_text = entity.get("summary_text")

            if summary_text:
                memory_lines.append(f"- {summary_text}")
            print(hit)

    print(memory_lines, "memory_lines")
    print( "\n".join(memory_lines))

if __name__ == "__main__":
    asyncio.run(main())