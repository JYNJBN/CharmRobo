from datetime import datetime, timedelta, timezone
from uuid import uuid4
from xml.etree import ElementTree

import httpx

from app.core.config import settings
from app.integrations.knowledge_cos import (
    build_cos_upload_form,
    get_knowledge_cos_client,
)


def main() -> None:
    client = get_knowledge_cos_client()

    content = "COS POST 表单上传测试".encode("utf-8")
    prefix = settings.knowledge_cos_incoming_prefix.rstrip("/")
    key = f"{prefix}/policy-check/{uuid4().hex}.txt"

    form = build_cos_upload_form(
        bucket=settings.cos_bucket,
        region=settings.cos_region,
        upload_key=key,
        file_size=len(content),
        mime_type="text/plain",
        storage_class=settings.knowledge_cos_storage_class,
        upload_deadline=(
                datetime.now(timezone.utc) + timedelta(minutes=30)
        ),
    )

    try:
        with httpx.Client(timeout=30) as http:
            response = http.post(
                form["url"],
                data=form["form_data"],
                files={
                    form["file_field"]: (
                        "check.txt",
                        content,
                        "text/plain",
                    )
                },
            )

        if response.status_code != 201:
            try:
                error_code = ElementTree.fromstring(
                    response.content
                ).findtext("Code")
            except ElementTree.ParseError:
                error_code = "UNKNOWN"

            raise RuntimeError(
                f"POST 上传失败："
                f"HTTP {response.status_code}，错误码 {error_code}"
            )

        print("COS POST 表单上传成功")

        client.head_object(
            Bucket=settings.cos_bucket,
            Key=key,
        )
        print("HEAD 检查成功")

    finally:
        client.delete_object(
            Bucket=settings.cos_bucket,
            Key=key,
        )
        print("测试对象已删除")


if __name__ == "__main__":
    main()
