import logging

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)
# 当前 Python 进程复用一个 Embedding 客户端。
# 每个 Uvicorn worker 都会各自拥有一个，这是正常的。
_embedding_client: AsyncOpenAI | None = None


def get_embedding_client() -> AsyncOpenAI:
    global _embedding_client
    api_key = settings.resolved_embedding_api_key
    if api_key is None:
        raise RuntimeError(
            "没有配置向量模型的 Key：请在 .env 设置 EMBEDDING_API_KEY，"
            "或复用百炼的 QWEN_API_KEY"
        )
    if _embedding_client is None:
        # 把模型名打进日志：换供应商之后「到底在用哪个模型」必须可查，
        # 而不是靠猜。
        logger.info(
            "[Embedding] 创建全局客户端 model=%s base_url=%s",
            settings.embedding_model,
            settings.embedding_base_url,
        )
        _embedding_client = AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            # 清除右边多余/ 防止拼接出现//的情况
            base_url=settings.embedding_base_url.rstrip("/"),
            timeout=30,
        )
    else:
        logger.debug("[Embedding] 复用全局客户端")
    return _embedding_client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Embedding文本不能是空字符串")
    client = get_embedding_client()

    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        encoding_format="float",
        # 显式指定维度，不吃服务端默认值：默认值一旦变化会直接撞上下面的维度
        # 校验并报错，比悄悄写入错误维度的向量（Milvus 照收，检索全乱）好排查。
        # 注意：并非所有 OpenAI 兼容端点都支持 dimensions 参数，换供应商时留意。
        dimensions=settings.embedding_dimension,
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


async def close_embedding_client() -> None:
    """应用关闭时释放 HTTP 连接池。"""

    global _embedding_client

    client = _embedding_client
    _embedding_client = None

    if client is not None:
        await client.close()
        logger.info("[Embedding] 全局客户端已关闭")
