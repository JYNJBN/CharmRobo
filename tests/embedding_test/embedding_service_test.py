import asyncio

from app.integrations.embedding import embed_text


async def main() -> None:
    vector = await embed_text("用户喜欢晚上听轻音乐。")

    print("Embedding 服务调用成功")
    print("向量维度：", len(vector))
    print("前 5 个值：", vector[:5])


if __name__ == "__main__":
    asyncio.run(main())
