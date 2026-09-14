from openai import AsyncOpenAI

from app.core.config import settings


def get_embedding_client() -> AsyncOpenAI:
    if settings.siliconflow_api_key is None:
        raise RuntimeError("没有配置SILICONFLOW_API_KEY")

    return AsyncOpenAI(
        api_key=settings.siliconflow_api_key.get_secret_value(),
        # 清除右边多余/ 防止拼接出现//的情况
        base_url=settings.siliconflow_base_url.rstrip("/"),
        timeout=30,
    )


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Embedding文本不能是空字符串")
    client = get_embedding_client()

    response = await client.embeddings.create(
        model=settings.embedding_model, input=texts, encoding_format="float"
    )
    # print(response.data,'response')
    items = sorted(response.data, key=lambda item: item.index)

    vectors = [item.embedding for item in items]
    # print('vectors',vectors)
    for vector in vectors:
        if len(vector) != settings.embedding_dimension:
            raise RuntimeError(
                f"向量维度不匹配：实际 {len(vector)}，"
                f"配置 {settings.embedding_dimension}"
            )
    return vectors


async def embed_text(text: str) -> list[float]:
    vectors = await embed_texts([text])
    # print("vectors[0]",vectors[0])
    return vectors[0]
