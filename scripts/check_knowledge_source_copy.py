from uuid import uuid4

from app.core.config import settings
from app.integrations.knowledge_cos import (
    ensure_knowledge_source_object,
    get_knowledge_cos_client,
    head_knowledge_upload_object,
)


def main() -> None:
    client = get_knowledge_cos_client()
    bucket = settings.cos_bucket
    region = settings.cos_region

    test_id = uuid4().hex
    upload_key = f"incoming/source-copy-check/{test_id}.txt"
    storage_key = f"knowledge/source-copy-check/{test_id}.txt"

    try:
        client.put_object(
            Bucket=bucket,
            Key=upload_key,
            Body=b"knowledge source copy test",
            StorageClass=settings.knowledge_cos_storage_class,
        )

        metadata = head_knowledge_upload_object(
            bucket=bucket,
            region=region,
            key=upload_key,
        )

        arguments = {
            "bucket": bucket,
            "region": region,
            "upload_key": upload_key,
            "storage_key": storage_key,
            "upload_etag": metadata["etag"],
            "storage_class": settings.knowledge_cos_storage_class,
        }

        result = ensure_knowledge_source_object(**arguments)
        print("正式源对象已固定，字节数：", result["size"])

        # 相同参数再执行，应复用已有正式对象。
        result = ensure_knowledge_source_object(**arguments)
        print("重复执行成功，字节数：", result["size"])

    finally:
        # 只清理本次随机生成的两个测试对象。
        client.delete_object(
            Bucket=bucket,
            Key=upload_key,
        )
        client.delete_object(
            Bucket=bucket,
            Key=storage_key,
        )
        print("测试对象已清理")


if __name__ == "__main__":
    main()
