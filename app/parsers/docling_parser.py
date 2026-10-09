from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter
from docling_core.types.doc import (
    DoclingDocument,
    SectionHeaderItem,
    TableItem,
    TextItem,
)


class KnowledgeParseError(RuntimeError):
    """文档解析的业务错误。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class ParsedKnowledge:
    """知识库统一解析结果。"""

    parsed_text: str
    parsed_elements: list[dict]
    warnings: list[str]


def normalize_document(
        document: DoclingDocument,
        *,
        title: str,
        max_chars: int = 2_000_000,
) -> ParsedKnowledge:
    """将 Docling 文档转换为知识库使用的文本和元素列表。"""

    parts: list[str] = []
    elements: list[dict] = []

    # 保存当前章节层级，例如：
    # [(1, "酒店介绍"), (2, "入住说明")]
    headings: list[tuple[int, str]] = []

    # 当前已生成文本的长度。
    cursor = 0

    for item, _ in document.iterate_items():
        if isinstance(item, TableItem):
            # 表格保留为 Markdown 表格，避免直接拼接导致列关系丢失。
            text = item.export_to_markdown(doc=document)
            element_type = "table"

        elif isinstance(item, TextItem):
            text = item.text

            element_type = {
                "title": "heading",
                "section_header": "heading",
                "list_item": "list_item",
                "code": "code",
                "formula": "formula",
            }.get(item.label.value, "paragraph")

        else:
            # 图片等非文本元素暂不进行识别。
            continue

        # 统一换行符，去掉元素首尾空白。
        text = (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

        if not text:
            continue

        if item.label.value == "title":
            # 文档总标题。
            headings = [(0, text)]

        elif isinstance(item, SectionHeaderItem):
            # 遇到同级或更高级标题，移除之前不再适用的章节。
            while headings and headings[-1][0] >= item.level:
                headings.pop()

            headings.append((item.level, text))

        # 元素之间统一插入两个换行符。
        start = cursor + (2 if parts else 0)
        end = start + len(text)

        if end > max_chars:
            raise KnowledgeParseError(
                "PARSED_TEXT_TOO_LARGE",
                "解析文本超过长度上限",
            )

        heading_path = [value for _, value in headings]

        elements.append(
            {
                "text": text,
                "title": (
                    heading_path[-1]
                    if heading_path
                    else title
                ),
                "heading_path": heading_path,
                "element_type": element_type,
                # DOCX 无法可靠确定分页，暂不填写页码。
                "page_number": None,
                "row_number": None,
                "start_offset": start,
                "end_offset": end,
                "metadata": {
                    "element_index": len(elements),
                    "source_ref": item.self_ref,
                    "source_label": item.label.value,
                    "ocr_used": False,
                },
            }
        )

        parts.append(text)
        cursor = end

    if not parts:
        raise KnowledgeParseError(
            "NO_EXTRACTABLE_TEXT",
            "文档没有可提取的有效文本",
        )

    warnings = []

    if document.pictures:
        warnings.append("包含图片，当前未识别图片中的文字")

    return ParsedKnowledge(
        parsed_text="\n\n".join(parts),
        parsed_elements=elements,
        warnings=warnings,
    )


def parse_docling_file(
        path: Path,
        *,
        title: str,
        max_file_size: int = 10 * 1024 * 1024,
) -> ParsedKnowledge:
    """解析已经下载到本地的文件，当前先开放 DOCX。"""

    if path.suffix.lower() != ".docx":
        raise KnowledgeParseError(
            "UNSUPPORTED_FILE_TYPE",
            "当前仅开放 DOCX 解析",
        )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.DOCX],
    )

    result = converter.convert(
        path,
        raises_on_error=False,
        max_file_size=max_file_size,
    )

    # 部分成功也不直接发布，避免资料遗漏。
    if result.status != ConversionStatus.SUCCESS:
        raise KnowledgeParseError(
            "DOCUMENT_PARSE_FAILED",
            f"解析未完整成功：{result.status.value}",
        )

    return normalize_document(
        result.document,
        title=title,
    )
