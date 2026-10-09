import asyncio
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile, BadZipFile

from sqlalchemy import select, func

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.integrations.embedding import embed_texts, logger
from app.integrations.knowledge_cos import KnowledgeSourceError, ensure_knowledge_source_object, \
    download_verified_knowledge_source
from app.integrations.knowledge_milvus import upsert_knowledge_chunks, delete_knowledge_vectors
from app.models import KnowledgeIngestJob, KnowledgeDocument, KnowledgeChunk, KnowledgeBase, KnowledgeDocumentUpload
from app.parsers.docling_parser import parse_docling_file, ParsedKnowledge, KnowledgeParseError
from app.services.knowledge_chunk_service import split_text, build_chunks, build_element_chunks, KnowledgeChunkError


async def claim_one_ingest_job(
        worker_id: str,
        lease_seconds: int = 120,
) -> dict | None:
    """抢任务"""
    if lease_seconds <= 0:
        raise ValueError("lease_seconds 必须大于 0")
    claim = None
    async with AsyncSessionLocal() as db:
        # 开启事务
        async with db.begin():
            # 获取数据库服务器时间
            now = await db.scalar(select(func.now()))
            # 排队中的并且可领取的任务
            job = await db.scalar(
                select(
                    KnowledgeIngestJob
                ).where(
                    KnowledgeIngestJob.status == "queued",
                    KnowledgeIngestJob.available_at <= now,
                    KnowledgeIngestJob.attempt_count
                    < KnowledgeIngestJob.max_attempts,
                ).order_by(
                    KnowledgeIngestJob.available_at.asc(),
                    KnowledgeIngestJob.id.asc(),
                ).limit(1)
                .with_for_update(skip_locked=True)
            )
            if job is not None:
                lease_token = uuid4()
                job.status = "running"
                job.locked_by = worker_id
                job.locked_at = now
                # timedelta(120)就是2分钟
                job.locked_until = now + timedelta(
                    seconds=lease_seconds
                )
                job.lease_token = lease_token
                job.attempt_count += 1
                # 第一次start的时间估这里如果有时间是取本身的时间
                job.started_at = job.started_at or now

                claim = {
                    "job_id": job.id,
                    "document_id": job.document_id,
                    "knowledge_base_id": job.knowledge_base_id,
                    "user_id": job.user_id,
                    "job_type": job.job_type,
                    "stage": job.stage,
                    "ingest_generation": job.ingest_generation,
                    "delete_before_generation": job.delete_before_generation,
                    "lease_token": lease_token,
                    "config_snapshot": dict(job.config_snapshot),
                }

    # 离开 db.begin() 时任务领取已经提交，再把 claim 交给 Worker。
    return claim


def _finish_file_job(
        job,
        *,
        status,
        now,
        code=None,
        message=None,
):
    """结束任务并释放租约。"""
    job.status = status
    job.finished_at = now
    job.locked_by = None
    job.lease_token = None
    job.locked_at = None
    job.locked_until = None
    job.last_error_code = code
    job.last_error_message = message


async def _lock_file_ingest_context(
        db,
        claim: dict,
        stage: str,
):
    # 锁顺序：知识库 → 任务 → 文档 → 上传记录。
    base = await db.scalar(
        select(KnowledgeBase)
        .where(
            KnowledgeBase.id == claim["knowledge_base_id"],
            KnowledgeBase.user_id == claim["user_id"],
        )
        .with_for_update()
    )

    job = await db.scalar(
        select(KnowledgeIngestJob)
        .where(
            KnowledgeIngestJob.id == claim["job_id"],
            KnowledgeIngestJob.document_id == claim["document_id"],
            KnowledgeIngestJob.knowledge_base_id
            == claim["knowledge_base_id"],
            KnowledgeIngestJob.user_id == claim["user_id"],
            KnowledgeIngestJob.ingest_generation
            == claim["ingest_generation"],
            KnowledgeIngestJob.job_type == "ingest",
            KnowledgeIngestJob.status == "running",
            KnowledgeIngestJob.stage == stage,
            KnowledgeIngestJob.lease_token == claim["lease_token"],
        )
        .with_for_update()
    )

    if job is None:
        return None

    document = await db.scalar(
        select(KnowledgeDocument)
        .where(
            KnowledgeDocument.id == job.document_id,
            KnowledgeDocument.user_id == job.user_id,
            KnowledgeDocument.knowledge_base_id
            == job.knowledge_base_id,
            KnowledgeDocument.ingest_generation
            == job.ingest_generation,
        )
        .with_for_update()
    )

    upload = await db.scalar(
        select(KnowledgeDocumentUpload)
        .where(
            KnowledgeDocumentUpload.document_id == job.document_id
        )
        .with_for_update()
    )

    # 等待行锁可能耗时，拿到锁后再读取当前数据库时间。
    now = await db.scalar(select(func.clock_timestamp()))

    if job.locked_until is None or job.locked_until <= now:
        return None

    if (
            base is None
            or base.deleted != 0
            or document is None
            or document.deleted != 0
            or document.purged_at is not None
    ):
        _finish_file_job(
            job,
            status="cancelled",
            now=now,
        )
        return None

    if (
            document.source_type != "file"
            or document.storage_backend != "cos"
            or upload is None
            or upload.upload_confirmed_at is None
            or not upload.upload_etag
    ):
        raise RuntimeError("文件任务与已确认上传记录不匹配")

    return job, document, upload, now


def _file_source_reference(job, document, upload) -> dict:
    """取出普通数据，结束事务后仍然可以安全使用。"""
    return {
        "bucket": upload.storage_bucket,
        "region": upload.storage_region,
        "upload_key": upload.upload_key,
        "storage_key": upload.storage_key,
        "upload_etag": upload.upload_etag,
        "expected_size": document.file_size,
        "expected_sha256": upload.upload_expected_sha256,
        "storage_class": job.config_snapshot["cos_storage_class"],
        "file_format": job.config_snapshot["file_format"],
    }


def _verify_cos_file(reference: dict) -> dict:
    if reference["file_format"] != "docx":
        raise KnowledgeSourceError(
            "UNSUPPORTED_FILE_TYPE",
            "当前仅开放 DOCX 文件",
        )

    # 固定正式源文件，重试时复用已有对象。
    source = ensure_knowledge_source_object(
        bucket=reference["bucket"],
        region=reference["region"],
        upload_key=reference["upload_key"],
        storage_key=reference["storage_key"],
        upload_etag=reference["upload_etag"],
        storage_class=reference["storage_class"],
    )

    with TemporaryDirectory(
            prefix="knowledge-verify-"
    ) as temp_dir:
        path = Path(temp_dir) / "source.docx"

        verified = download_verified_knowledge_source(
            bucket=reference["bucket"],
            region=reference["region"],
            key=reference["storage_key"],
            # 下载使用正式对象自己的 ETag。
            source_etag=source["etag"],
            expected_size=reference["expected_size"],
            expected_sha256=reference["expected_sha256"],
            destination=path,
        )

        # DOCX 是 ZIP 容器。
        # 此处只做基础容器检查，完整内容由下一阶段的 Docling 解析。
        try:
            with ZipFile(path) as archive:
                infos = archive.infolist()
                names = {info.filename for info in infos}

                required = {
                    "[Content_Types].xml",
                    "word/document.xml",
                }

                if not required <= names:
                    raise KnowledgeSourceError(
                        "INVALID_DOCX",
                        "文件不是有效的 DOCX 容器",
                    )

                # 初始保护上限：2000 个条目、展开大小 100 MiB。
                # 展开大小和上传文件的压缩大小是两个不同限制。
                if (
                        len(infos) > 2000
                        or sum(info.file_size for info in infos)
                        > 100 * 1024 * 1024
                ):
                    raise KnowledgeSourceError(
                        "DOCX_TOO_LARGE",
                        "DOCX 展开大小或内部文件数量超限",
                    )

                if any(info.flag_bits & 1 for info in infos):
                    raise KnowledgeSourceError(
                        "ENCRYPTED_DOCX",
                        "暂不支持加密文件",
                    )

        except BadZipFile as exc:
            raise KnowledgeSourceError(
                "INVALID_DOCX",
                "文件不是有效的 DOCX 容器",
            ) from exc

        return verified


async def process_verify_stage(claim: dict) -> bool:
    # 第一段：短事务读取并冻结本次处理的输入。
    async with AsyncSessionLocal() as db:
        async with db.begin():
            context = await _lock_file_ingest_context(
                db, claim, "verify"
            )
            if context is None:
                return False
            job, document, upload, _ = context
            reference = _file_source_reference(
                job, document, upload
            )
            document.parse_status = "verifying"
    # 此处已经退出数据库事务。
    verified = None
    source_error = None
    try:
        verified = await asyncio.to_thread(
            _verify_cos_file,
            reference,
        )
    except KnowledgeSourceError as exc:
        # 文件大小、哈希、格式等确定性错误，直接标记失败。
        source_error = exc

    # 网络等其他异常继续向上抛出，
    # 由现有 process_claimed_job → retry_or_fail_job 处理。

    # 第二段：重新检查后再提交结果。

    async with AsyncSessionLocal() as db:
        async with db.begin():
            context = await _lock_file_ingest_context(
                db, claim, "verify"
            )

            if context is None:
                return False

            job, document, upload, now = context
            # 下载期间，源文件配置也不能发生变化。
            if (
                    _file_source_reference(job, document, upload)
                    != reference
            ):
                raise RuntimeError(
                    "文件处理期间源文件配置发生变化"
                )
            if source_error is not None:
                document.parse_status = "failed"
                document.error_code = source_error.code
                document.error_message = source_error.message

                _finish_file_job(
                    job,
                    status="failed",
                    now=now,
                    code=source_error.code,
                    message=source_error.message,
                )
                return False
            # 你的数据库对同一知识库的有效 content_hash 有唯一约束。
            duplicate_id = await db.scalar(
                select(KnowledgeDocument.id)
                .where(
                    KnowledgeDocument.knowledge_base_id
                    == job.knowledge_base_id,
                    KnowledgeDocument.user_id == job.user_id,
                    KnowledgeDocument.deleted == 0,
                    KnowledgeDocument.id != document.id,
                    KnowledgeDocument.content_hash
                    == verified["sha256"],
                )
                .limit(1)
            )

            now = await db.scalar(
                select(func.clock_timestamp())
            )

            if (
                    job.locked_until is None
                    or job.locked_until <= now
            ):
                return False

            if duplicate_id is not None:
                document.parse_status = "failed"
                document.error_code = "DUPLICATE_CONTENT"
                document.error_message = (
                    "该知识库中已存在相同文件内容"
                )

                _finish_file_job(
                    job,
                    status="failed",
                    now=now,
                    code=document.error_code,
                    message=document.error_message,
                )
                return False

            # 以下修改在同一个事务中提交。
            upload.source_verified_at = now

            document.content_hash = verified["sha256"]
            document.file_size = verified["size"]
            document.parse_status = "parsing"
            document.error_code = None
            document.error_message = None

            job.last_error_code = None
            job.last_error_message = None
            job.result_json = {
                **(job.result_json or {}),
                "source_size": verified["size"],
                "source_sha256": verified["sha256"],
                "source_etag": verified["etag"],
            }

            # 下一个阶段是解析。
            job.stage = "parse"

        return True


def _file_parse_reference(job, document, upload) -> dict:
    """读取已验证的正式源文件信息。"""

    if (
            upload.source_verified_at is None
            or not document.content_hash
            or document.content_hash != upload.upload_expected_sha256
    ):
        raise RuntimeError("正式源文件尚未验证，不能开始解析")

    source_etag = (job.result_json or {}).get("source_etag")

    if not source_etag:
        raise RuntimeError("校验结果缺少正式源文件 ETag")

    if (
            document.parser_version
            != job.config_snapshot["parser_version"]
    ):
        raise RuntimeError("文档与任务的解析器版本不一致")

    reference = _file_source_reference(
        job, document, upload
    )

    reference.update(
        {
            "source_etag": source_etag,
            "content_hash": document.content_hash,
            "source_verified_at": upload.source_verified_at,
            "parser_version": job.config_snapshot["parser_version"],
            "title": document.title,
        }
    )

    return reference


def _parse_cos_file(reference: dict) -> ParsedKnowledge:
    # 沿用你当前的 docx-v1，表示本项目第一版 Word 解析规则。
    if (
            reference["file_format"] != "docx"
            or reference["parser_version"] != "docx-v1"
    ):
        raise KnowledgeParseError(
            "UNSUPPORTED_PARSER_VERSION",
            "当前任务的格式或解析器版本尚不支持",
        )

    with TemporaryDirectory(
            prefix="knowledge-parse-"
    ) as temp_dir:
        path = Path(temp_dir) / "source.docx"

        # 下载正式对象，重新核对实际内容。
        download_verified_knowledge_source(
            bucket=reference["bucket"],
            region=reference["region"],
            key=reference["storage_key"],
            source_etag=reference["source_etag"],
            expected_size=reference["expected_size"],
            expected_sha256=reference["content_hash"],
            destination=path,
        )

        # 下载校验成功后，交给 Docling。
        return parse_docling_file(
            path,
            title=reference["title"],
        )


async def process_file_parse_stage(claim: dict) -> bool:
    # 第一段：确认处理资格，读取文件信息。
    async with AsyncSessionLocal() as db:
        async with db.begin():
            context = await _lock_file_ingest_context(
                db, claim, "parse"
            )

            if context is None:
                return False

            job, document, upload, _ = context

            reference = _file_parse_reference(
                job, document, upload
            )
            document.parse_status = "parsing"

    # 数据库事务已经结束。
    parsed = None
    parse_error = None

    try:
        # 下载和解析在事务外执行，也不阻塞事件循环。
        parsed = await asyncio.to_thread(
            _parse_cos_file,
            reference,
        )
    except (
            KnowledgeParseError,
            KnowledgeSourceError,
    ) as exc:
        parse_error = exc

    # 其他异常继续向上抛出，由现有自动重试方法处理。

    # 第二段：重新确认租约、代次和删除状态，再保存结果。
    async with AsyncSessionLocal() as db:
        async with db.begin():
            context = await _lock_file_ingest_context(
                db, claim, "parse"
            )

            if context is None:
                return False

            job, document, upload, now = context

            if (
                    _file_parse_reference(job, document, upload)
                    != reference
            ):
                raise RuntimeError(
                    "解析期间源文件或解析配置发生变化"
                )

            if parse_error is not None:
                document.parse_status = "failed"
                document.error_code = parse_error.code
                document.error_message = str(parse_error)

                _finish_file_job(
                    job,
                    status="failed",
                    now=now,
                    code=parse_error.code,
                    message=str(parse_error),
                )
                return False

            # 正文、元素列表和阶段转换一起提交。
            document.parsed_text = parsed.parsed_text
            document.parsed_elements_json = (
                parsed.parsed_elements
            )
            document.warnings_json = parsed.warnings

            document.parse_status = "chunking"
            document.error_code = None
            document.error_message = None

            job.last_error_code = None
            job.last_error_message = None

            job.result_json = {
                **(job.result_json or {}),
                "parsed_chars": len(parsed.parsed_text),
                "parsed_element_count": len(
                    parsed.parsed_elements
                ),
            }

            job.stage = "chunk"

    return True


async def process_inline_parse_stage(claim: dict) -> bool:
    """将任务更新状态"""
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))
            # 只有持有当前有效租约的 Worker 才能继续。
            job = await db.scalar(
                select(KnowledgeIngestJob)
                .where(
                    KnowledgeIngestJob.id == claim["job_id"],
                    KnowledgeIngestJob.status == "running",
                    KnowledgeIngestJob.stage == "parse",
                    KnowledgeIngestJob.lease_token
                    == claim["lease_token"],
                    KnowledgeIngestJob.locked_until > now,
                )
                .with_for_update()
            )

            if job is None:
                # 租约过期、任务被接管，或任务已经推进；旧 Worker 停止。
                return False

            document = await db.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.id == job.document_id,
                    KnowledgeDocument.user_id == job.user_id,
                    KnowledgeDocument.knowledge_base_id
                    == job.knowledge_base_id,
                    KnowledgeDocument.ingest_generation
                    == job.ingest_generation,
                    KnowledgeDocument.deleted == 0,
                )
            )
            # 如果查不到文档
            if document is None:
                job.status = "cancelled"
                job.finished_at = now
                job.locked_by = None
                job.lease_token = None
                job.locked_at = None
                job.locked_until = None
                return False

            parsed_text = (document.raw_content or "").strip()
            if not parsed_text:
                document.parse_status = "failed"
                document.error_code = "EMPTY_RAW_CONTENT"
                document.error_message = "文档正文为空"

                job.status = "failed"
                job.last_error_code = "EMPTY_RAW_CONTENT"
                job.last_error_message = "文档原文为空"
                job.finished_at = now
                job.locked_by = None
                job.lease_token = None
                job.locked_at = None
                job.locked_until = None
                return False

            document.parsed_text = parsed_text
            document.parsed_elements_json = [
                {
                    "element_type": "paragraph",
                    "text": parsed_text,
                    "start_offset": 0,
                    "end_offset": len(parsed_text),
                }
            ]
            document.parse_status = "chunking"

            # 文档解析结果和任务阶段一起提交。
            job.stage = "chunk"

        return True


async def process_parse_stage(claim: dict) -> bool:
    # 文件上传任务的配置快照中包含 file_format。
    if "file_format" in claim["config_snapshot"]:
        return await process_file_parse_stage(claim)

    # 手动输入、粘贴文本沿用原来的处理方法。
    return await process_inline_parse_stage(claim)


async def process_chunk_stage(claim: dict) -> bool:
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))

            # 只允许持有有效租约、且仍处于 chunk 阶段的 Worker 继续。
            job = await db.scalar(
                select(KnowledgeIngestJob)
                .where(
                    KnowledgeIngestJob.id == claim["job_id"],
                    KnowledgeIngestJob.status == "running",
                    KnowledgeIngestJob.stage == "chunk",
                    KnowledgeIngestJob.lease_token
                    == claim["lease_token"],
                    KnowledgeIngestJob.locked_until > now,
                )
                .with_for_update()
            )

            if job is None:
                return False

            document = await db.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.id == job.document_id,
                    KnowledgeDocument.user_id == job.user_id,
                    KnowledgeDocument.knowledge_base_id
                    == job.knowledge_base_id,
                    KnowledgeDocument.ingest_generation
                    == job.ingest_generation,
                    KnowledgeDocument.deleted == 0,
                )
            )

            if document is None:
                job.status = "cancelled"
                job.finished_at = now
                job.locked_by = None
                job.lease_token = None
                job.locked_at = None
                job.locked_until = None
                return False

            if not document.parsed_text or not document.parsed_text.strip():
                document.parse_status = "failed"
                document.error_code = "EMPTY_PARSED_TEXT"
                document.error_message = "解析后的正文为空"

                job.status = "failed"
                job.last_error_code = "EMPTY_PARSED_TEXT"
                job.last_error_message = "解析后的正文为空"
                job.finished_at = now
                job.locked_by = None
                job.lease_token = None
                job.locked_at = None
                job.locked_until = None
                return False
            # 取快照里的状态
            snapshot = job.config_snapshot
            try:
                version = snapshot["chunker_version"]

                if version == "char-overlap-v1":
                    contents = split_text(
                        document.parsed_text,
                        target_chars=int(
                            snapshot["chunk_target_chars"]
                        ),
                        overlap_chars=int(
                            snapshot["chunk_overlap_chars"]
                        ),
                    )
                    chunks = build_chunks(document, contents)

                elif version == "element-overlap-v1":
                    chunks = build_element_chunks(
                        document,
                        target_chars=int(
                            snapshot["chunk_target_chars"]
                        ),
                        overlap_chars=int(
                            snapshot["chunk_overlap_chars"]
                        ),
                    )

                else:
                    raise KnowledgeChunkError(
                        "UNSUPPORTED_CHUNKER_VERSION",
                        "任务的切块器版本尚不支持",
                    )

                if not chunks:
                    raise KnowledgeChunkError(
                        "NO_CHUNKS",
                        "正文没有生成有效切块",
                    )

                # 同一代次如果已有块，检查后复用，避免生成新 ID。
                result = await db.scalars(
                    select(KnowledgeChunk)
                    .where(
                        KnowledgeChunk.user_id == document.user_id,
                        KnowledgeChunk.knowledge_base_id
                        == document.knowledge_base_id,
                        KnowledgeChunk.document_id == document.id,
                        KnowledgeChunk.ingest_generation
                        == document.ingest_generation,
                    )
                    .order_by(KnowledgeChunk.chunk_index)
                )
                stored = list(result.all())

                if stored:
                    fields = (
                        "chunk_index",
                        "title",
                        "content",
                        "embedding_text_hash",
                        "element_type",
                        "start_offset",
                        "end_offset",
                        "page_number",
                        "row_number",
                        "metadata_json",
                    )

                    if len(stored) != len(chunks) or any(
                            old.deleted != 0
                            or any(
                                getattr(old, key) != getattr(new, key)
                                for key in fields
                            )
                            for old, new in zip(stored, chunks)
                    ):
                        raise KnowledgeChunkError(
                            "CHUNK_SNAPSHOT_MISMATCH",
                            "当前代次已有切块与处理快照不一致，"
                            "需要新代次重建",
                        )

                    chunks = stored

                now = await db.scalar(
                    select(func.clock_timestamp())
                )

                if (
                        job.locked_until is None
                        or job.locked_until <= now
                ):
                    return False

                if not stored:
                    db.add_all(chunks)

            except KnowledgeChunkError as exc:
                now = await db.scalar(
                    select(func.clock_timestamp())
                )

                if (
                        job.locked_until is None
                        or job.locked_until <= now
                ):
                    return False

                document.parse_status = "failed"
                document.error_code = exc.code
                document.error_message = str(exc)

                _finish_file_job(
                    job,
                    status="failed",
                    now=now,
                    code=exc.code,
                    message=str(exc),
                )
                return False
            document.chunk_count = len(chunks)
            document.parse_status = "embedding"

            job.stage = "embed"
            job.result_json = {
                **(job.result_json or {}),
                "chunk_count": len(chunks),
            }

    return True


async def process_embed_index_stage(claim: dict) -> bool:
    # 第一段短事务：核对租约，读取文档和待同步切块。
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))

            job = await db.scalar(
                select(KnowledgeIngestJob)
                .where(
                    KnowledgeIngestJob.id == claim["job_id"],
                    KnowledgeIngestJob.status == "running",
                    KnowledgeIngestJob.stage.in_(["embed", "index"]),
                    KnowledgeIngestJob.lease_token
                    == claim["lease_token"],
                    KnowledgeIngestJob.locked_until > now,
                )
                .with_for_update()
            )
            if job is None:
                return False

            document = await db.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.id == job.document_id,
                    KnowledgeDocument.user_id == job.user_id,
                    KnowledgeDocument.knowledge_base_id
                    == job.knowledge_base_id,
                    KnowledgeDocument.ingest_generation
                    == job.ingest_generation,
                    KnowledgeDocument.deleted == 0,
                )
            )
            if document is None:
                job.status = "cancelled"
                job.finished_at = now
                job.locked_by = None
                job.lease_token = None
                job.locked_at = None
                job.locked_until = None
                return False

            snapshot = dict(job.config_snapshot)

            chunks_result = await db.scalars(
                select(KnowledgeChunk)
                .where(
                    KnowledgeChunk.document_id == job.document_id,
                    KnowledgeChunk.ingest_generation
                    == job.ingest_generation,
                    KnowledgeChunk.deleted == 0,
                    KnowledgeChunk.vector_sync_status == "pending",
                )
                .order_by(KnowledgeChunk.chunk_index.asc())
            )
            chunks = list(chunks_result.all())

            if not chunks:
                raise RuntimeError("没有待同步的知识切块")

    # Embedding 配置必须和建任务时保存的快照一致。
    # embed_texts 和 upsert_knowledge_chunks 当前都读取 settings。
    if (
            snapshot["embedding_model"] != settings.embedding_model
            or snapshot["embedding_base_url"] != settings.embedding_base_url
            or snapshot["embedding_dimension"] != settings.embedding_dimension
            or snapshot["embedding_profile"]
            != settings.knowledge_embedding_profile
            or snapshot["vector_collection"]
            != settings.milvus_knowledge_collection
    ):
        raise RuntimeError("当前模型配置与任务快照不一致")

    # 外部调用放在数据库事务之外，避免拿着行锁等待模型或 Milvus。
    vectors = await embed_texts(
        [chunk.embedding_text for chunk in chunks]
    )
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding 返回数量与切块数量不一致")
        # Embedding 完成、Milvus 写入前，先记录进入 index 阶段。
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))

            job = await db.scalar(
                select(KnowledgeIngestJob)
                .where(
                    KnowledgeIngestJob.id == claim["job_id"],
                    KnowledgeIngestJob.status == "running",
                    KnowledgeIngestJob.stage.in_(["embed", "index"]),
                    KnowledgeIngestJob.lease_token
                    == claim["lease_token"],
                    KnowledgeIngestJob.locked_until > now,
                )
                .with_for_update()
            )
            if job is None:
                return False

            document = await db.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.id == job.document_id,
                    KnowledgeDocument.user_id == job.user_id,
                    KnowledgeDocument.knowledge_base_id
                    == job.knowledge_base_id,
                    KnowledgeDocument.ingest_generation
                    == job.ingest_generation,
                    KnowledgeDocument.deleted == 0,
                )
            )
            if document is None:
                job.status = "cancelled"
                job.finished_at = now
                job.locked_by = None
                job.lease_token = None
                job.locked_at = None
                job.locked_until = None
                return False

            job.stage = "index"
            document.parse_status = "indexing"
    # Milvus SDK 是同步的，放进线程，避免阻塞 async Worker。
    # 写入使用稳定的 chunk.id；重试时会覆盖同一 ID。
    await asyncio.to_thread(
        upsert_knowledge_chunks,
        chunks,
        vectors,
    )

    # Milvus 成功后再用短事务更新 PostgreSQL。
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))

            job = await db.scalar(
                select(KnowledgeIngestJob)
                .where(
                    KnowledgeIngestJob.id == claim["job_id"],
                    KnowledgeIngestJob.status == "running",
                    KnowledgeIngestJob.stage == "index",
                    KnowledgeIngestJob.lease_token
                    == claim["lease_token"],
                    KnowledgeIngestJob.locked_until > now,
                )
                .with_for_update()
            )
            if job is None:
                # 租约已经失效，旧 Worker 不得回写任务状态。
                return False

            chunk_ids = [chunk.id for chunk in chunks]
            stored_chunks_result = await db.scalars(
                select(KnowledgeChunk).where(
                    KnowledgeChunk.id.in_(chunk_ids),
                    KnowledgeChunk.document_id == job.document_id,
                    KnowledgeChunk.ingest_generation
                    == job.ingest_generation,
                    KnowledgeChunk.deleted == 0,
                    KnowledgeChunk.vector_sync_status == "pending",
                )
            )
            stored_chunks = list(stored_chunks_result.all())

            if len(stored_chunks) != len(chunk_ids):
                raise RuntimeError("待同步切块数量发生变化，拒绝确认")

            for chunk in stored_chunks:
                chunk.vector_sync_status = "synced"

            job.stage = "finalize"

    return True


async def process_finalize_stage(claim: dict) -> bool:
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))
            job = await db.scalar(
                select(KnowledgeIngestJob)
                .where(
                    KnowledgeIngestJob.id == claim["job_id"],
                    KnowledgeIngestJob.status == "running",
                    KnowledgeIngestJob.stage == "finalize",
                    KnowledgeIngestJob.lease_token
                    == claim["lease_token"],
                    KnowledgeIngestJob.locked_until > now,
                )
                .with_for_update()
            )
            if job is None:
                return False
            document = await db.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.id == job.document_id,
                    KnowledgeDocument.user_id == job.user_id,
                    KnowledgeDocument.knowledge_base_id
                    == job.knowledge_base_id,
                    KnowledgeDocument.ingest_generation
                    == job.ingest_generation,
                    KnowledgeDocument.deleted == 0,
                )
            )
            if document is None:
                job.status = "cancelled"
                job.finished_at = now
                job.locked_by = None
                job.lease_token = None
                job.locked_at = None
                job.locked_until = None
                return False
            knowledge_base = await db.scalar(
                select(KnowledgeBase).where(
                    KnowledgeBase.id == job.knowledge_base_id,
                    KnowledgeBase.user_id == job.user_id,
                    KnowledgeBase.deleted == 0,
                )
            )
            if knowledge_base is None:
                job.status = "cancelled"
                job.finished_at = now
                job.locked_by = None
                job.lease_token = None
                job.locked_at = None
                job.locked_until = None
                return False
            total_chunks = await db.scalar(
                select(func.count(KnowledgeChunk.id)).where(
                    KnowledgeChunk.document_id == job.document_id,
                    KnowledgeChunk.ingest_generation
                    == job.ingest_generation,
                    KnowledgeChunk.deleted == 0,
                )
            ) or 0
            unsynced_chunks = await db.scalar(
                select(func.count(KnowledgeChunk.id)).where(
                    KnowledgeChunk.document_id == job.document_id,
                    KnowledgeChunk.ingest_generation
                    == job.ingest_generation,
                    KnowledgeChunk.deleted == 0,
                    KnowledgeChunk.vector_sync_status != "synced",
                )
            ) or 0

            if (
                    total_chunks == 0
                    or total_chunks != document.chunk_count
                    or unsynced_chunks != 0
            ):
                raise RuntimeError(
                    "不能发布文档：切块数量或向量同步状态不完整"
                )

            document.parse_status = "ready"
            document.ready_at = now
            document.error_code = None
            document.error_message = None

            job.status = "succeeded"
            job.finished_at = now
            job.locked_by = None
            job.lease_token = None
            job.locked_at = None
            job.locked_until = None

        return True


async def process_delete_vectors_job(claim: dict) -> bool:
    # 先短暂读取任务范围；不在数据库事务里等待 Milvus。
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))

            job = await db.scalar(
                select(KnowledgeIngestJob)
                .where(
                    KnowledgeIngestJob.id == claim["job_id"],
                    KnowledgeIngestJob.job_type == "delete_vectors",
                    KnowledgeIngestJob.stage == "delete_vectors",
                    KnowledgeIngestJob.status == "running",
                    KnowledgeIngestJob.lease_token
                    == claim["lease_token"],
                    KnowledgeIngestJob.locked_until > now,
                )
                .with_for_update()
            )
            if job is None:
                return False

            cutoff = job.delete_before_generation
            collection_name = (
                job.config_snapshot.get("vector_collection")
            )

            if cutoff is None or cutoff <= 0:
                raise RuntimeError(
                    "delete_vectors 任务缺少有效的删除代次上限"
                )
            if not collection_name:
                raise RuntimeError(
                    "delete_vectors 任务快照缺少 Milvus Collection"
                )

            delete_scope = {
                "collection_name": collection_name,
                "user_id": job.user_id,
                "knowledge_base_id": job.knowledge_base_id,
                "document_id": job.document_id,
                "delete_before_generation": cutoff,
            }

    # 用数据库任务中冻结的范围清理 Milvus。
    # 过滤条件只删除 ingest_generation < cutoff 的向量。
    await asyncio.to_thread(
        delete_knowledge_vectors,
        **delete_scope,
    )  # Milvus 成功后，再确认租约有效并提交清理结果。
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))

            job = await db.scalar(
                select(KnowledgeIngestJob)
                .where(
                    KnowledgeIngestJob.id == claim["job_id"],
                    KnowledgeIngestJob.job_type == "delete_vectors",
                    KnowledgeIngestJob.stage == "delete_vectors",
                    KnowledgeIngestJob.status == "running",
                    KnowledgeIngestJob.lease_token
                    == claim["lease_token"],
                    KnowledgeIngestJob.locked_until > now,
                )
                .with_for_update()
            )
            if job is None:
                # 租约已失效，旧 Worker 不能回写任务结果。
                return False
            chunks_result = await db.scalars(
                select(KnowledgeChunk)
                .where(
                    KnowledgeChunk.user_id == job.user_id,
                    KnowledgeChunk.knowledge_base_id
                    == job.knowledge_base_id,
                    KnowledgeChunk.document_id == job.document_id,
                    KnowledgeChunk.ingest_generation
                    < job.delete_before_generation,
                    KnowledgeChunk.deleted == 1,
                )
            )
            for chunk in chunks_result.all():
                chunk.vector_sync_status = "removed"

            job.status = "succeeded"
            job.finished_at = now
            job.locked_by = None
            job.lease_token = None
            job.locked_at = None
            job.locked_until = None

        return True


async def retry_or_fail_job(
        claim: dict,
        exc: Exception,
) -> None:
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))

            job = await db.scalar(
                select(KnowledgeIngestJob)
                .where(
                    KnowledgeIngestJob.id == claim["job_id"],
                    KnowledgeIngestJob.status == "running",
                    KnowledgeIngestJob.lease_token
                    == claim["lease_token"],
                    KnowledgeIngestJob.locked_until > now,
                )
                .with_for_update()
            )
            if job is None:
                # 租约已失效，旧 Worker 不再修改任务。
                return

            document = await db.scalar(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.id == job.document_id,
                    KnowledgeDocument.ingest_generation
                    == job.ingest_generation,
                    KnowledgeDocument.deleted == 0,
                )
            )

            job.last_error_code = "WORKER_STAGE_FAILED"
            # 详细异常留在服务日志里，数据库只存可给用户看的脱敏信息。
            job.last_error_message = "知识处理暂时失败，请稍后重试"

            if job.attempt_count < job.max_attempts:
                delay_seconds = {1: 5, 2: 30}.get(
                    job.attempt_count,
                    60,
                )
                job.status = "queued"
                job.available_at = now + timedelta(
                    seconds=delay_seconds
                )
                job.finished_at = None
            else:
                job.status = "failed"
                job.finished_at = now
                if document is not None and job.job_type == "ingest":
                    document.parse_status = "failed"
                    document.error_code = job.last_error_code
                    document.error_message = job.last_error_message

            job.locked_by = None
            job.lease_token = None
            job.locked_at = None
            job.locked_until = None


async def process_claimed_job(claim: dict) -> None:
    job_type = claim["job_type"]

    try:
        if job_type == "ingest":
            await process_ingest_job(claim)
        elif job_type == "delete_vectors":
            await process_delete_vectors_job(claim)
        else:
            raise RuntimeError(f"不支持的任务类型：{job_type}")

    except Exception as exc:
        logger.exception(
            "知识任务处理失败 job_id=%s job_type=%s",
            claim["job_id"],
            job_type,
        )
        await retry_or_fail_job(claim, exc)


async def process_ingest_job(claim: dict) -> None:
    """verify（文件）→ parse → chunk → embed/index → finalize。"""

    stage = claim["stage"]

    if stage == "verify":
        if not await process_verify_stage(claim):
            return

        stage = "parse"

    if stage == "parse":
        if not await process_parse_stage(claim):
            return

        stage = "chunk"

    if stage == "chunk":
        if not await process_chunk_stage(claim):
            return

        stage = "embed"

    if stage in {"embed", "index"}:
        if not await process_embed_index_stage(claim):
            return

        stage = "finalize"

    if stage == "finalize":
        await process_finalize_stage(claim)
        return

    raise RuntimeError(
        f"暂不支持的入库阶段：{stage}"
    )


async def recover_one_expired_job() -> bool:
    async with AsyncSessionLocal() as db:
        async with db.begin():
            now = await db.scalar(select(func.now()))
            job = await db.scalar(
                select(KnowledgeIngestJob).where(
                    KnowledgeIngestJob.status == "running",
                    # 超时任务
                    KnowledgeIngestJob.locked_until <= now,
                ).order_by(
                    KnowledgeIngestJob.locked_until.asc(),
                    KnowledgeIngestJob.id.asc()
                ).limit(1).with_for_update(
                    skip_locked=True
                )
            )
            if job is None:
                return False

            document = None
            # 只有入库任务需要同步修改文档状态。
            if job.job_type == "ingest":
                document = await db.scalar(
                    select(
                        KnowledgeDocument
                    ).where(
                        KnowledgeDocument.id == job.document_id,
                        KnowledgeDocument.user_id == job.user_id,
                        KnowledgeDocument.knowledge_base_id == job.knowledge_base_id,
                        KnowledgeDocument.ingest_generation == job.ingest_generation
                    ).with_for_update())
            # 已删除、已物理清理或属于旧代次的入库任务取消。 防止有人修改数据库的情况，正常删除不会走到这里
            invalid_ingest = (
                    job.job_type == "ingest"
                    and (
                            document is None
                            or document.deleted != 0
                            or document.purged_at is not None
                    )
            )
            if invalid_ingest:
                job.status = "cancelled"
                job.finished_at = now

            else:
                job.last_error_code = "WORKER_LEASE_EXPIRED"
                job.last_error_message = "任务租约已过期，请稍后重试"

                if job.attempt_count < job.max_attempts:
                    delay_seconds = {1: 5, 2: 30}.get(
                        job.attempt_count,
                        60
                    )
                    job.status = "queued"
                    job.available_at = now + timedelta(
                        seconds=delay_seconds
                    )
                    job.finished_at = None
                else:
                    job.status = "failed"
                    job.finished_at = now
                    if document is not None:
                        document.parse_status = "failed"
                        document.ready_at = None
                        document.error_code = job.last_error_code
                        document.error_message = job.last_error_message
                # 撤销旧 Worker 的执行资格。
            job.locked_by = None
            job.lease_token = None
            job.locked_at = None
            job.locked_until = None
        # 离开事务后，回收结果已经提交。
        logger.warning(
            "回收过期任务 job_id=%s status=%s attempt_count=%s/%s",
            job.id,
            job.status,
            job.attempt_count,
            job.max_attempts,
        )

        return True
