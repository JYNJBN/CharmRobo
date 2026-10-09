import base64
import hashlib
import hmac
import json
import re
from contextlib import closing
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path

from qcloud_cos import CosS3Client, CosConfig, CosServiceError, CosClientError

from app.core.config import settings


@lru_cache(maxsize=8)
def get_knowledge_cos_client(
        region: str | None = None,
) -> CosS3Client:
    """每个进程复用一个 COS 客户端。"""
    actual_region = region or settings.cos_region
    secret_id = (
        settings.cos_secret_id.get_secret_value()
        if settings.cos_secret_id is not None
        else ""
    )
    secret_key = (
        settings.cos_secret_key.get_secret_value()
        if settings.cos_secret_key is not None
        else ""
    )
    required = {
        "COS_BUCKET": settings.cos_bucket,
        "COS_REGION": actual_region,
        "COS_SECRET_ID": secret_id,
        "COS_SECRET_KEY": secret_key,
    }
    missing = [
        name
        for name, value in required.items()
        if not value or not value.strip()
    ]
    if missing:
        raise RuntimeError(
            "缺少 COS 配置：" + ", ".join(missing)
        )
    token = (
        settings.cos_security_token.get_secret_value().strip()
        if settings.cos_security_token is not None
        else None
    )

    config = CosConfig(
        Region=actual_region.strip(),
        SecretId=secret_id.strip(),
        SecretKey=secret_key.strip(),
        Token=token or None,
        Scheme="https",
        Timeout=30,
    )

    return CosS3Client(config)


def build_cos_upload_form(
        *,
        bucket: str,
        region: str,
        upload_key: str,
        file_size: int,
        mime_type: str,
        storage_class: str,
        upload_deadline: datetime,
) -> dict:
    """生成只允许向指定暂存对象上传的 COS POST 表单。"""
    if not bucket or not region or not upload_key:
        raise ValueError("COS 桶、地域和上传 key 不能为空")
    if not 0 < file_size <= settings.knowledge_max_file_bytes:
        raise ValueError("文件大小超过允许范围")

    if storage_class not in {"STANDARD", "MAZ_STANDARD"}:
        raise ValueError("知识库只允许标准存储类型")

    if upload_deadline.tzinfo is None:
        raise ValueError("上传截止时间必须包含时区")

    if (
            settings.cos_secret_id is None
            or settings.cos_secret_key is None
    ):
        raise RuntimeError("缺少 COS_SECRET_ID 或 COS_SECRET_KEY")
    secret_id = settings.cos_secret_id.get_secret_value().strip()
    secret_key = settings.cos_secret_key.get_secret_value().strip()

    if not secret_id or not secret_key:
        raise RuntimeError("COS 密钥不能为空")

    now = datetime.now(timezone.utc)

    # 表单有效期不能超过上传会话截止时间。
    deadline = min(
        upload_deadline.astimezone(timezone.utc),
        now + timedelta(
            seconds=settings.knowledge_upload_form_ttl_seconds
        ),
    )
    start = int(now.timestamp())
    end = int(deadline.timestamp())
    if end <= start:
        raise ValueError("上传会话已过期")
    expires_at = datetime.fromtimestamp(end, timezone.utc)
    key_time = f"{start};{end}"
    # 小程序上传时需要携带的表单字段。
    form_data = {
        "key": upload_key,
        "acl": "private",
        "Content-Type": mime_type,
        "x-cos-storage-class": storage_class,
        "success_action_status": "201",
        "q-sign-algorithm": "sha1",
        "q-ak": secret_id,
        "q-key-time": key_time,
    }
    # COS 验证上传请求时必须满足的条件。
    conditions = [
        {"bucket": bucket},
        {"key": upload_key},
        {"acl": "private"},
        {"Content-Type": mime_type},
        {"x-cos-storage-class": storage_class},
        {"success_action_status": "201"},
        {"q-sign-algorithm": "sha1"},
        {"q-ak": secret_id},
        {"q-sign-time": key_time},
        ["content-length-range", file_size, file_size],
    ]
    # 使用临时凭证时，把安全令牌一起放入表单及策略。
    if settings.cos_security_token is not None:
        token = settings.cos_security_token.get_secret_value().strip()

        if token:
            form_data["x-cos-security-token"] = token
            conditions.append({"x-cos-security-token": token})

    policy = {
        "expiration": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "conditions": conditions,
    }
    policy_bytes = json.dumps(
        policy,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    # 第一步：SecretKey + 有效时间，生成签名密钥。
    sign_key = hmac.new(
        secret_key.encode("utf-8"),
        key_time.encode("ascii"),
        hashlib.sha1,
    ).hexdigest()
    # 第二步：计算整个上传策略的摘要。
    string_to_sign = hashlib.sha1(policy_bytes).hexdigest()
    # 第三步：对策略摘要签名。
    signature = hmac.new(
        sign_key.encode("ascii"),
        string_to_sign.encode("ascii"),
        hashlib.sha1,
    ).hexdigest()

    form_data["policy"] = base64.b64encode(
        policy_bytes
    ).decode("ascii")
    form_data["q-signature"] = signature

    return {
        "method": "POST",
        "url": f"https://{bucket}.cos.{region}.myqcloud.com",
        "file_field": "file",
        "expires_at": expires_at,
        "form_data": form_data,
    }


def head_knowledge_upload_object(
        *,
        bucket: str,
        region: str,
        key: str,
) -> dict:
    """读取暂存对象元数据，不下载整个文件。"""
    client = get_knowledge_cos_client(region=region)
    response = client.head_object(
        Bucket=bucket,
        Key=key,
    )
    headers = {
        str(name).lower(): value
        for name, value in response.items()
    }
    try:
        size = int(headers["content-length"])
        etag = str(headers["etag"]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("COS 没有返回有效的文件元数据") from exc

    if size < 0 or not etag:
        raise ValueError("COS 文件元数据不完整")

    return {
        "size": size,
        "etag": etag,
    }


class KnowledgeSourceError(RuntimeError):
    """源文件无法继续处理的业务错误。"""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _head_knowledge_object_if_exists(
        *,
        bucket: str,
        region: str,
        key: str,
) -> dict | None:
    """不存在返回 None；权限或网络错误继续抛出。"""
    try:
        return head_knowledge_upload_object(
            bucket=bucket,
            region=region,
            key=key,
        )
    except CosServiceError as exc:
        if exc.get_status_code() == 404:
            return None
        raise


def ensure_knowledge_source_object(
        *,
        bucket: str,
        region: str,
        upload_key: str,
        storage_key: str,
        upload_etag: str,
        storage_class: str,
) -> dict:
    """固定正式源对象；重试时复用已有对象，不覆盖。"""
    if not upload_etag:
        raise KnowledgeSourceError(
            "UPLOAD_NOT_CONFIRMED",
            "上传尚未确认，缺少暂存对象 ETag",
        )

    if (
            not upload_key
            or not storage_key
            or upload_key == storage_key
    ):
        raise KnowledgeSourceError(
            "INVALID_STORAGE_KEY",
            "暂存对象和正式对象路径配置不正确",
        )

    if storage_class not in {"STANDARD", "MAZ_STANDARD"}:
        raise KnowledgeSourceError(
            "INVALID_STORAGE_CLASS",
            "不支持的知识库存储类型",
        )
    client = get_knowledge_cos_client(region=region)
    # 当前文档方案不记录 versionId，因此要求桶未开启版本控制。
    versioning = client.get_bucket_versioning(
        Bucket=bucket,
    )
    if versioning.get("Status") not in (None, "", "Disabled"):
        raise KnowledgeSourceError(
            "UNSUPPORTED_BUCKET_VERSIONING",
            "当前入库方案要求存储桶未启用版本控制",
        )

        # 上一次可能已经复制成功，只是后续处理失败。
    existing = _head_knowledge_object_if_exists(
        bucket=bucket,
        region=region,
        key=storage_key,
    )
    if existing is not None:
        return existing

    try:
        response = client.copy_object(
            Bucket=bucket,
            Key=storage_key,
            CopySource={
                "Bucket": bucket,
                "Region": region,
                "Key": upload_key,
            },
            # 要求暂存文件的ETag与完成确认时一致，变化了就拒绝复制。
            # 等于说就是看cos文件的etag 是否和 数据库存储的一致
            CopySourceIfMatch=upload_etag,
            StorageClass=storage_class,
            ACL="private",
            # 禁止覆盖 如果目标对象已经存在，COS 应该拒绝再次写入同名对象。
            Metadata={
                "x-cos-forbid-overwrite": "true",
            },
        )
        # 不仅判断 HTTP 状态，还需要成功复制的结果字段。
        if not response.get("ETag"):
            raise RuntimeError("COS 复制响应缺少成功标识")
    except (CosServiceError, CosClientError) as exc:
        # 可能已经复制成功，但响应丢失；
        # 也可能另一个执行者已先创建正式对象。
        existing = _head_knowledge_object_if_exists(
            bucket=bucket,
            region=region,
            key=storage_key,
        )
        if existing is not None:
            return existing
        if (
                isinstance(exc, CosServiceError)
                and exc.get_status_code() == 412
        ):
            raise KnowledgeSourceError(
                "SOURCE_FILE_CHANGED",
                "暂存文件在上传确认后发生变化，请重新上传",
            ) from exc

        raise

    result = _head_knowledge_object_if_exists(
        bucket=bucket,
        region=region,
        key=storage_key,
    )
    if result is None:
        raise RuntimeError("COS 复制后尚未找到正式源文件")

    return result


def download_verified_knowledge_source(
        *,
        bucket: str,
        region: str,
        key: str,
        source_etag: str,
        expected_size: int,
        expected_sha256: str,
        destination: Path,
        max_file_size: int = 10 * 1024 * 1024,
) -> dict:
    """下载到专用临时目录，并校验实际大小和 SHA256。"""

    if not 0 < expected_size <= max_file_size:
        raise KnowledgeSourceError(
            "INVALID_FILE_SIZE",
            "声明的文件大小不符合限制",
        )

    if not source_etag:
        raise KnowledgeSourceError(
            "MISSING_SOURCE_ETAG",
            "缺少正式源文件 ETag",
        )
    expected_sha256 = expected_sha256.lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise KnowledgeSourceError(
            "INVALID_SHA256",
            "声明的 SHA256 格式不正确",
        )

    client = get_knowledge_cos_client(region=region)
    try:
        response = client.get_object(
            Bucket=bucket,
            Key=key,
            # 使用正式对象的 ETag。
            IfMatch=source_etag,
        )
    except CosServiceError as exc:
        if exc.get_status_code() == 404:
            raise KnowledgeSourceError(
                "SOURCE_FILE_NOT_FOUND",
                "正式源文件不存在",
            ) from exc

        if exc.get_status_code() == 412:
            raise KnowledgeSourceError(
                "SOURCE_FILE_CHANGED",
                "正式源文件发生变化",
            ) from exc

        # 网络、权限等其他错误继续向上抛出。
        raise
    digest = hashlib.sha256()
    actual_size = 0
    stream = response["Body"].get_raw_stream()
    # 无论成功还是异常，退出时都关闭下载流。
    with closing(stream):
        # xb：新建文件，如果文件已经存在就报错，避免覆盖。
        with destination.open("xb") as output:
            while True:
                # 每次最多读取 64 KB。
                block = stream.read(64 * 1024)

                if not block:
                    break

                actual_size += len(block)

                # 下载过程中就检查大小，超限立即停止。
                if actual_size > min(max_file_size, expected_size):
                    raise KnowledgeSourceError(
                        "FILE_SIZE_MISMATCH",
                        "实际文件大小超过声明大小或允许上限",
                    )

                digest.update(block)
                output.write(block)

    # 也要检查文件是否比声明的更小。
    if actual_size != expected_size:
        raise KnowledgeSourceError(
            "FILE_SIZE_MISMATCH",
            "实际文件大小与声明不一致",
        )

    actual_sha256 = digest.hexdigest()

    if not hmac.compare_digest(
            actual_sha256,
            expected_sha256,
    ):
        raise KnowledgeSourceError(
            "FILE_HASH_MISMATCH",
            "实际文件 SHA256 与声明不一致",
        )

    return {
        "size": actual_size,
        "sha256": actual_sha256,
        "etag": source_etag,
    }
