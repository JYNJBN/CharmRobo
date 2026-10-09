from uuid import uuid4

from app.core.config import settings
from app.integrations.knowledge_cos import get_knowledge_cos_client


def main() -> None:
    client = get_knowledge_cos_client()
    bucket = settings.cos_bucket

    prefix = settings.knowledge_cos_incoming_prefix.rstrip("/")
    key = f"{prefix}/connection-check/{uuid4().hex}.txt"
    content = "知识库 COS 连接测试".encode("utf-8")

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=content,
        ContentType="text/plain; charset=utf-8",
    )
    print("上传成功")

    try:
        client.head_object(
            Bucket=bucket,
            Key=key,
        )
        print("HEAD 检查成功")

        response = client.get_object(
            Bucket=bucket,
            Key=key,
        )
        stream = response["Body"].get_raw_stream()

        try:
            downloaded = stream.read()
        finally:
            stream.close()

        assert downloaded == content, "下载内容与上传内容不一致"
        print("下载内容检查通过")

    finally:
        # 只删除这次随机生成的测试对象。
        client.delete_object(
            Bucket=bucket,
            Key=key,
        )
        print("测试对象已删除")


if __name__ == "__main__":
    main()
