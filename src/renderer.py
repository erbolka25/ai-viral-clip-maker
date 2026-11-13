"""Render one moment of the source video as a vertical 9:16 clip.

Pipeline (one FFmpeg call, the source video is only read):

    source 16:9 --center crop 9:16--> scale to 1080x1920 x zoom(t) --center crop 1080x1920-->
    eq (contrast 1.05, saturation 1.1) --> [ASS subtitles / hook overlay] --> H.264

* Smooth zoom 1.00 -> 1.08 over the clip: the scale size is re-evaluated every
  frame and then center-cropped back to 1080x1920 (zoompan rounds its crop
  window to whole pixels and visibly shakes; this does not).
* The source frame rate is kept (no fps filter, no speed change).
* Audio: EBU R128 loudness normalisation to -14 LUFS (TikTok/Shorts/Reels level).
* Output: H.264 High / yuv420p + AAC 48 kHz, faststart MP4.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import config
from src.deps import require_binary

OUT_W, OUT_H = 1080, 1920
ZOOM_END = 1.08
CONTRAST, SATURATION = 1.05, 1.1
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"

# The source has burned-in (Russian) subtitles, bottom-anchored, 1-3 lines.
# This band (fractions of the source height) is blurred and darkened so the
# old text is unreadable under our English subtitles. None = clean source.
SOURCE_SUBTITLE_BAND: tuple[float, float] | None = (0.70, 0.945)
BAND_BLUR_SIGMA = 28
BAND_DARKEN = -0.18  # eq brightness offset inside the band
BAND_FEATHER = 18    # source pixels of soft edge at the band's top/bottom
FONTS_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"


class RenderError(RuntimeError):
    """FFmpeg failed to render a clip."""


@dataclass
class SourceInfo:
    width: int
    height: int
    duration: float


def source_info() -> SourceInfo:
    """Read source dimensions from data/video_info.json (or probe the video)."""
    if config.VIDEO_INFO_JSON.is_file():
        info = json.loads(config.VIDEO_INFO_JSON.read_text(encoding="utf-8"))
    else:
        from src.video_info import inspect_video
        info = inspect_video(config.INPUT_VIDEO)
    return SourceInfo(int(info["width"]), int(info["height"]), float(info["duration_seconds"]))


def _even(expr: str) -> str:
    """Wrap an FFmpeg expression so it yields an even integer (x264 needs it)."""
    return f"trunc(({expr})/2)*2"


def build_video_filter(src: SourceInfo, duration: float, ass_name: str | None) -> str:
    """FFmpeg -filter_complex graph ([0:v] -> [v]) for crop, subtitle-band
    blur, zoom, color and (optionally) burned-in ASS subtitles."""
    # 1) Largest 9:16 window centered in the source (720p -> 405x720).
    crop_w = min(src.width, round(src.height * OUT_W / OUT_H))
    crop_h = min(src.height, round(src.width * OUT_H / OUT_W))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    graph = [f"[0:v]crop={crop_w}:{crop_h}:(in_w-{crop_w})/2:(in_h-{crop_h})/2[c]"]

    # 2) Hide the source's burned-in subtitles (blur + darken that band).
    if SOURCE_SUBTITLE_BAND:
        y0 = int(crop_h * SOURCE_SUBTITLE_BAND[0]) // 2 * 2
        y1 = int(crop_h * SOURCE_SUBTITLE_BAND[1]) // 2 * 2
        graph += [
            "[c]split[base][band]",
            f"[band]crop={crop_w}:{y1 - y0}:0:{y0},"
            f"gblur=sigma={BAND_BLUR_SIGMA},eq=brightness={BAND_DARKEN},"
            # Alpha ramps 0->1 over BAND_FEATHER px at both edges (soft blend).
            f"format=yuva420p,geq=lum='p(X,Y)':cb='p(X,Y)':cr='p(X,Y)':"
            f"a='255*min(1,min(Y,H-1-Y)/{BAND_FEATHER})'[blurred]",
            f"[base][blurred]overlay=0:{y0}[c2]",
        ]
        current = "[c2]"
    else:
        current = "[c]"

    # 3) Zoom factor grows linearly from 1.0 to ZOOM_END over the clip.
    zoom = f"(1+{ZOOM_END - 1:.4f}*min(t/{duration:.3f},1))"
    chain = [
        f"scale=w='{_even(f'{OUT_W}*{zoom}')}':h='{_even(f'{OUT_H}*{zoom}')}'"
        ":eval=frame:flags=bicubic",
        f"crop={OUT_W}:{OUT_H}:(in_w-{OUT_W})/2:(in_h-{OUT_H})/2",
        "setsar=1",
        f"eq=contrast={CONTRAST}:saturation={SATURATION}",
    ]
    if ass_name:
        # Relative file name: FFmpeg runs with cwd = the .ass folder, which
        # avoids Windows drive-letter escaping problems in filter arguments.
        fonts = FONTS_DIR.as_posix().replace(":", "\\:")
        chain.append(f"ass={ass_name}:fontsdir='{fonts}'")
    chain.append("format=yuv420p")
    graph.append(current + ",".join(chain) + "[v]")
    return ";".join(graph)


def render_clip(
    start: float,
    end: float,
    out_path: Path,
    ass_path: Path | None = None,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Render source[start:end] to ``out_path`` (overwrites it).

    ``ass_path`` (optional) is burned in; its timestamps must be relative to
    the clip start (0 = first frame of the clip).
    """
    if not config.INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Source video not found: {config.INPUT_VIDEO}")
    src = source_info()
    if not (0 <= start < end <= src.duration + 0.05):
        raise RenderError(f"Clip {start:.2f}-{end:.2f}s is outside the video (0-{src.duration:.2f}s).")
    duration = end - start

    ffmpeg = require_binary("ffmpeg")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name(out_path.stem + ".part.mp4")

    cwd = ass_path.parent if ass_path else None
    vf = build_video_filter(src, duration, ass_path.name if ass_path else None)
    cmd = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(config.INPUT_VIDEO.resolve()),
        "-filter_complex", vf,
        "-map", "[v]", "-map", "0:a:0?",
        "-af", f"{LOUDNORM},aresample=48000",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        str(tmp_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", cwd=cwd, check=False)
    if result.returncode != 0 or not tmp_out.is_file():
        tmp_out.unlink(missing_ok=True)
        raise RenderError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr.strip()[-2000:]}")
    tmp_out.replace(out_path)
    return out_path


def extract_frames(video: Path, times: list[float], out_dir: Path, prefix: str) -> list[Path]:
    """Save single frames (PNG) at the given clip-relative times."""
    ffmpeg = require_binary("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(times, 1):
        png = out_dir / f"{prefix}_{i}_{t:.1f}s.png"
        subprocess.run(
            [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}",
             "-i", str(video), "-frames:v", "1", str(png)],
            check=True, capture_output=True,
        )
        paths.append(png)
    return paths


def probe(video: Path) -> dict[str, Any]:
    """Return width/height/fps/duration/codecs of a rendered clip."""
    from src.video_info import inspect_video
    return inspect_video(video)


if __name__ == "__main__":  # Test: render the top-scoring moment without subtitles
    import time

    data = json.loads(config.HIGHLIGHTS_JSON.read_text(encoding="utf-8"))
    best = max(data["moments"], key=lambda m: m["score"])
    out = config.OUTPUT_DIR / "test_clip.mp4"
    t0 = time.perf_counter()
    render_clip(best["start"], best["end"], out)
    print(f"Rendered #{best['id']} {best['title']!r} -> {out} in {time.perf_counter() - t0:.1f}s")
    info = probe(out)
    print({k: info[k] for k in ("resolution", "fps", "duration_seconds", "video_codec", "audio_codec")})
    d = info["duration_seconds"]
    for p in extract_frames(out, [0.5, d / 2, d - 0.5], config.TEST_DIR / "frames", "3a"):
        print("frame:", p)
