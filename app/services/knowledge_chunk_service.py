import hashlib
import re

from app.models import KnowledgeDocument, KnowledgeChunk


def split_text(
        text: str,
        target_chars: int,
        overlap_chars: int,
) -> list[str]:
    """切割文档成一段一段的chunk"""

    normalized = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        limit = min(start + target_chars, len(normalized))
        end = limit

        if limit < len(normalized):
            candidates = [
                normalized.rfind(mark, start, limit)
                for mark in "。！？；\n"
            ]
            boundary = max(candidates)

            if boundary > start + target_chars // 2:
                end = boundary + 1

        content = normalized[start:end].strip()

        if content:
            chunks.append(content)

        if end >= len(normalized):
            break

        start = max(end - overlap_chars, start + 1)

    return chunks


def build_chunks(
        document: KnowledgeDocument,
        contents: list[str]
) -> list[KnowledgeChunk]:
    """构建KnowledgeChunk"""
    chunks: list[KnowledgeChunk] = []
    for index, content in enumerate(contents):
        embedding_text = f"{document.title}\n\n{content}"
        embedding_text_hash = hashlib.sha256(
            embedding_text.encode("utf-8")
        ).hexdigest()
        chunks.append(
            KnowledgeChunk(
                user_id=document.user_id,
                knowledge_base_id=document.knowledge_base_id,
                document_id=document.id,
                ingest_generation=document.ingest_generation,
                chunk_index=index,
                title=document.title,
                content=content,
                embedding_text=embedding_text,
                embedding_text_hash=embedding_text_hash,
                element_type="paragraph",
                metadata_json={},
                vector_sync_status="pending",
                deleted=0,
            )
        )
    return chunks


class KnowledgeChunkError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _split_ranges(
        text: str,
        start: int,
        stop: int,
        target: int,
        overlap: int,
        marks: str = "。！？；\n",
):
    """在原始 parsed_text 上切分，返回准确的位置。"""
    while start < stop:
        limit = min(start + target, stop)
        end = limit

        if limit < stop:
            boundary = max(
                text.rfind(mark, start, limit)
                for mark in marks
            )
            if boundary - start > max(target // 2, overlap):
                end = boundary + 1

        raw = text[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())

        if right > left:
            yield start + left, start + right

        if end >= stop:
            break

        start = max(end - overlap, start + 1)


def _table_pieces(text: str, element: dict, target: int):
    """较小的表格保留完整；较大的表格按行拆分。"""
    start = element["start_offset"]
    end = element["end_offset"]
    table = text[start:end]

    if len(table) <= target:
        yield table, start, end, {}
        return

    lines = table.splitlines(keepends=True)

    if len(lines) < 3:
        raise KnowledgeChunkError(
            "TABLE_TOO_LARGE",
            "表格标题或单行内容过长，无法按行切分",
        )

    # Markdown 表格的第二行应当是列分隔符。
    cells = lines[1].strip().strip("|").split("|")

    if not all(
            re.fullmatch(r"\s*:?-{3,}:?\s*", cell)
            for cell in cells
    ):
        raise KnowledgeChunkError(
            "UNSUPPORTED_TABLE_LAYOUT",
            "表格格式暂不支持按行切分",
        )

    header = lines[0] + lines[1]
    header_end = len(header)
    batch_start = header_end
    cursor = header_end

    for line in lines[2:]:
        row_end = cursor + len(line)

        if len(header) + len(line.rstrip()) > target:
            raise KnowledgeChunkError(
                "TABLE_ROW_TOO_LARGE",
                "表格单行连同列标题超过切块上限",
            )

        if (
                len(header) + row_end - batch_start > target
                and cursor > batch_start
        ):
            yield (
                header + table[batch_start:cursor].rstrip(),
                start + batch_start,
                start + cursor,
                {
                    "table_header_range": [
                        start,
                        start + header_end,
                    ]
                },
            )
            batch_start = cursor

        cursor = row_end

    if cursor > batch_start:
        yield (
            header + table[batch_start:cursor].rstrip(),
            start + batch_start,
            start + cursor,
            {
                "table_header_range": [
                    start,
                    start + header_end,
                ]
            },
        )


def build_element_chunks(
        document: KnowledgeDocument,
        *,
        target_chars: int,
        overlap_chars: int,
) -> list[KnowledgeChunk]:
    """根据有序解析元素生成切块。"""
    if (
            target_chars <= 0
            or not 0 <= overlap_chars < target_chars
    ):
        raise KnowledgeChunkError(
            "INVALID_CHUNK_CONFIG",
            "切块长度或重叠长度配置不正确",
        )

    text = document.parsed_text or ""
    elements = document.parsed_elements_json

    if not isinstance(elements, list) or not elements:
        raise KnowledgeChunkError(
            "MISSING_PARSED_ELEMENTS",
            "缺少有序解析元素列表",
        )

    # 元素必须和保存的正文一致，不能使用已经变动的位置。
    last_end = 0

    for element in elements:
        start = element["start_offset"]
        end = element["end_offset"]

        if (
                not 0 <= start < end <= len(text)
                or start < last_end
                or text[start:end] != element["text"]
        ):
            raise KnowledgeChunkError(
                "INVALID_ELEMENT_OFFSETS",
                "解析元素与正文位置不一致",
            )

        last_end = end

    chunks: list[KnowledgeChunk] = []
    pending: list[int] = []
    pending_key = None

    def append_chunk(
            indices, content, start, end, kind, extra
    ):
        first = elements[indices[0]]
        path = list(first.get("heading_path") or [])

        prefix = document.title
        if path:
            prefix += "\n" + " > ".join(path)

        if len(prefix) > target_chars:
            raise KnowledgeChunkError(
                "CHUNK_CONTEXT_TOO_LARGE",
                "文档或章节标题超过当前切块预算",
            )

        embedding_text = prefix + "\n\n" + content

        metadata = {
            "heading_path": path,
            "source_elements": [
                {
                    "element_index": index,
                    "element_type": elements[index]["element_type"],
                    "metadata": (
                            elements[index].get("metadata") or {}
                    ),
                }
                for index in indices
            ],
            **extra,
        }

        chunks.append(
            KnowledgeChunk(
                user_id=document.user_id,
                knowledge_base_id=document.knowledge_base_id,
                document_id=document.id,
                ingest_generation=document.ingest_generation,
                chunk_index=len(chunks),
                title=str(
                    first.get("title") or document.title
                )[:255],
                content=content,
                embedding_text=embedding_text,
                embedding_text_hash=hashlib.sha256(
                    embedding_text.encode("utf-8")
                ).hexdigest(),
                element_type=kind,
                page_number=first.get("page_number"),
                row_number=first.get("row_number"),
                start_offset=start,
                end_offset=end,
                metadata_json=metadata,
                vector_sync_status="pending",
                deleted=0,
            )
        )

    def flush():
        """将当前章节收集的段落组合后切分。"""
        if not pending:
            return

        first = elements[pending[0]]
        last = elements[pending[-1]]

        kind = (
            "paragraph"
            if first["element_type"] == "heading"
            else first["element_type"]
        )

        for start, end in _split_ranges(
                text,
                first["start_offset"],
                last["end_offset"],
                target_chars,
                overlap_chars,
        ):
            indices = [
                index
                for index in pending
                if elements[index]["start_offset"] < end
                   and elements[index]["end_offset"] > start
            ]

            if indices:
                append_chunk(
                    indices,
                    text[start:end],
                    start,
                    end,
                    kind,
                    {},
                )

        pending.clear()

    for index, element in enumerate(elements):
        kind = element["element_type"]

        key = (
            tuple(element.get("heading_path") or []),
            element.get("page_number"),
            element.get("row_number"),
            "paragraph" if kind == "heading" else kind,
        )

        if kind in {"heading", "paragraph", "list_item"}:
            # 新标题或来源位置变化时，先结束前一组。
            if kind == "heading" or key != pending_key:
                flush()

            pending.append(index)
            pending_key = key
            continue

        flush()
        pending_key = None

        if kind == "table":
            for content, start, end, extra in _table_pieces(
                    text, element, target_chars
            ):
                append_chunk(
                    [index], content, start, end, kind, extra
                )

        elif kind in {"code", "formula"}:
            for start, end in _split_ranges(
                    text,
                    element["start_offset"],
                    element["end_offset"],
                    target_chars,
                    overlap_chars,
                    marks=(
                            "\n"
                            if kind == "code"
                            else "。！？；\n"
                    ),
            ):
                append_chunk(
                    [index],
                    text[start:end],
                    start,
                    end,
                    kind,
                    {},
                )

        else:
            raise KnowledgeChunkError(
                "UNSUPPORTED_ELEMENT_TYPE",
                f"暂不支持元素类型：{kind}",
            )

    flush()

    if not chunks:
        raise KnowledgeChunkError(
            "NO_CHUNKS",
            "正文没有生成有效切块",
        )

    return chunks
