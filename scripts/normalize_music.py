"""把本地音乐转换成 CI 播放器兼容的 16k 单声道 CBR MP3。"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg

SUPPORTED_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def normalize_track(
    source: Path,
    target: Path,
    *,
    bitrate: str,
    lead_silence_ms: int,
    overwrite: bool,
) -> None:
    """执行单首歌曲归一化。

    输出参数与目标硬件播放器保持一致：16kHz、单声道、64kbps CBR、无 ID3，
    并默认在开头补 300ms 静音，让 CI 精简 MP3 播放器更容易识别首帧。
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y" if overwrite else "-n"]
    if lead_silence_ms > 0:
        silence_seconds = max(1, lead_silence_ms) / 1000.0
        command.extend(
            [
                "-f",
                "lavfi",
                "-t",
                f"{silence_seconds:.3f}",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-i",
                str(source),
                "-filter_complex",
                "[0:a][1:a]concat=n=2:v=0:a=1[a]",
                "-map",
                "[a]",
            ]
        )
    else:
        command.extend(["-i", str(source), "-map", "0:a:0"])

    command.extend(
        [
            "-vn",
            "-map_metadata",
            "-1",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-write_xing",
            "0",
            "-id3v2_version",
            "0",
            "-write_id3v1",
            "0",
            str(target),
        ]
    )
    subprocess.run(command, check=True)
    patch_ci_simple_player_header(target)


def patch_ci_simple_player_header(path: Path) -> None:
    """清零 CI 精简播放器可能误读的 MP3 首帧长度字段。"""

    with path.open("r+b") as file:
        head = file.read(10)
        if len(head) < 10 or head[0] != 0xFF or (head[1] & 0xE0) != 0xE0:
            return
        if head[6:10] == b"\x00\x00\x00\x00":
            return
        file.seek(6)
        file.write(b"\x00\x00\x00\x00")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize local music files for CI MP3 playback"
    )
    parser.add_argument("--src", required=True, help="原始音乐目录")
    parser.add_argument(
        "--out", required=True, help="输出目录，服务器 MUSIC_DIR 指向这里"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bitrate", default="64k")
    parser.add_argument("--lead-silence-ms", type=int, default=300)
    args = parser.parse_args()

    source_dir = Path(args.src)
    output_dir = Path(args.out)
    for source in sorted(source_dir.iterdir(), key=lambda item: item.name.lower()):
        if not source.is_file() or source.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        target = output_dir / f"{source.stem}.mp3"
        if target.exists() and not args.overwrite:
            print(f"skip exists: {target.name}")
            continue
        print(f"convert: {source.name} -> {target.name}")
        normalize_track(
            source,
            target,
            bitrate=args.bitrate,
            lead_silence_ms=args.lead_silence_ms,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
