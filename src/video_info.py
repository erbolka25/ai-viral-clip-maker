"""Video inspection using FFprobe.

Extracts duration, resolution, FPS, codecs and file size and saves them to
``data/video_info.json``.
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from src.deps import require_binary


class VideoProbeError(RuntimeError):
    """Raised when FFprobe fails or returns unusable data."""


def _parse_fps(rate: str | None) -> float | None:
    """Convert FFprobe's rational frame rate (e.g. '30000/1001') to float."""
    if not rate or rate in ("0/0", "0"):
        return None
    try:
        return round(float(Fraction(rate)), 3)
    except (ValueError, ZeroDivisionError):
        return None


def _run_ffprobe(video_path: Path) -> dict[str, Any]:
    """Run ffprobe and return its parsed JSON output."""
    ffprobe = require_binary("ffprobe")
    cmd = [
        str(ffprobe),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False,
        )
    except OSError as exc:
        raise VideoProbeError(f"Could not start ffprobe: {exc}") from exc

    if result.returncode != 0:
        raise VideoProbeError(
            f"ffprobe failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"ffprobe returned invalid JSON: {exc}") from exc


def inspect_video(video_path: Path) -> dict[str, Any]:
    """Inspect a video file and return a summary dictionary."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    probe = _run_ffprobe(video_path)
    streams: list[dict[str, Any]] = probe.get("streams", [])
    fmt: dict[str, Any] = probe.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise VideoProbeError("No video stream found in the file.")

    # Duration: prefer container duration, fall back to the video stream.
    duration_raw = fmt.get("duration") or video.get("duration")
    duration = round(float(duration_raw), 3) if duration_raw else None

    width = video.get("width")
    height = video.get("height")
    size_bytes = video_path.stat().st_size

    return {
        "file": video_path.name,
        "path": str(video_path),
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}" if width and height else None,
        "orientation": (
            "vertical" if height and width and height > width
            else "horizontal" if width and height and width > height
            else "square"
        ),
        "fps": _parse_fps(video.get("avg_frame_rate")) or _parse_fps(video.get("r_frame_rate")),
        "video_codec": video.get("codec_name"),
        "video_bitrate": int(video["bit_rate"]) if video.get("bit_rate") else None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "audio_sample_rate": int(audio["sample_rate"]) if audio and audio.get("sample_rate") else None,
        "audio_channels": audio.get("channels") if audio else None,
        "has_audio": audio is not None,
        "container": fmt.get("format_name"),
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_bytes / (1024 * 1024), 2),
    }


def save_video_info(info: dict[str, Any], output_path: Path) -> Path:
    """Write the video info dictionary as pretty JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


if __name__ == "__main__":  # Quick manual test: python -m src.video_info
    from src import config

    print(json.dumps(inspect_video(config.INPUT_VIDEO), indent=2))
