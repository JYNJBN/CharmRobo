import json
import sys
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            '用法：python scripts/check_docling.py "文件的完整路径.docx"'
        )

    source = Path(sys.argv[1]).resolve()

    if not source.is_file():
        raise SystemExit(f"文件不存在：{source}")

    if source.suffix.lower() != ".docx":
        raise SystemExit("这一步先测试 .docx 文件")

    # 当前只开放 DOCX，其他格式后续逐个配置和验证。
    converter = DocumentConverter(
        allowed_formats=[InputFormat.DOCX],
    )

    result = converter.convert(
        source,
        raises_on_error=False,
        max_file_size=10 * 1024 * 1024,
    )

    print(f"解析状态：{result.status.value}")

    # 部分成功也先拒绝，避免遗漏内容后仍然发布到知识库。
    if result.status != ConversionStatus.SUCCESS:
        for error in result.errors:
            print(f"解析错误：{error}")
        raise SystemExit("解析未完整成功，请检查文件和错误信息")

    document = result.document
    markdown = document.export_to_markdown()

    if not markdown.strip():
        raise SystemExit("文件没有解析出可用文本")

    output_dir = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "docling-preview"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / f"{source.stem}.md"
    json_path = output_dir / f"{source.stem}.json"

    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            document.export_to_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"文本长度：{len(markdown)}")
    print(f"文本预览：\n{markdown[:1000]}")
    print(f"\nMarkdown 文件：{markdown_path}")
    print(f"结构化 JSON：{json_path}")


if __name__ == "__main__":
    main()
