import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.parsers.docling_parser import parse_docling_file


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("请传入 DOCX 文件路径")

    source = Path(sys.argv[1]).resolve()

    if not source.is_file():
        raise SystemExit(f"文件不存在：{source}")

    result = parse_docling_file(
        source,
        title=source.stem,
    )

    # 检查每个元素的偏移量能否准确取回原文本。
    for element in result.parsed_elements:
        start = element["start_offset"]
        end = element["end_offset"]

        assert result.parsed_text[start:end] == element["text"], (
            f"元素偏移量不正确：{element['metadata']['element_index']}"
        )

    output_dir = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "docling-preview"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{source.stem}.normalized.json"
    output_path.write_text(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"解析成功，正文长度：{len(result.parsed_text)}")
    print(f"元素数量：{len(result.parsed_elements)}")
    print(f"警告：{result.warnings}")
    print(f"统一解析结果：{output_path}")


if __name__ == "__main__":
    main()
