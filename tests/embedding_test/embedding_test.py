import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

API_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"

# 密钥只从项目根目录的 .env 读；该文件已在 .gitignore 里，不会进仓库。
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def main() -> None:
    api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("没有找到 SILICONFLOW_API_KEY，请检查项目根目录的 .env")
    payload = {
        "model": MODEL,
        "input": "用户喜欢晚上听轻音乐。",
        "encoding_format": "float",
    }
    response = httpx.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    embedding = result["data"][0]["embedding"]
    print(result, embedding)

    print("Embedding 调用成功")
    print("模型：", result["model"])
    print("向量维度：", len(embedding))
    print("前 5 个向量值：", embedding[:5])


if __name__ == "__main__":
    main()
