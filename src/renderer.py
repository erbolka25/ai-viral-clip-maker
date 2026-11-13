"""Render one clip of the source video as a vertical 9:16 MP4 from an EditPlan.

Pipeline (one FFmpeg call; the source video is only read):

    source --[jump cuts: select/aselect on frame-aligned ranges]-->
    center crop 9:16 --> blur the source's burned-in subtitle band -->
    scale to 1080x1920 x zoom(t) --> center crop 1080x1920 -->
    eq (contrast 1.05, saturation 1.1) --> ASS (subtitles, hook, progress bar) --> H.264

* Cuts: video keeps the frames inside each kept range and audio keeps exactly
  the same time ranges; ranges are whole frames, so A/V cannot drift.
* Zoom: the scale size is re-evaluated every frame from plan.zoom_expr
  (punch-ins / slow zoom), then center-cropped back to 1080x1920.
* The source frame rate is kept (no fps filter, no speed change).
* Audio: loudness normalised to -14 LUFS (TikTok/Shorts/Reels level).
* Output: H.264 High / yuv420p + AAC 48 kHz, faststart MP4.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import config
from src.deps import require_binary
from src.editing import EditPlan

OUT_W, OUT_H = 1080, 1920
CONTRAST, SATURATION = 1.05, 1.1
LOUDNORM = "loudnorm=I=-14:TP=-1.5:LRA=11"

# The source has burned-in (Russian) subtitles, bottom-anchored, 1-3 lines.
# This band (fractions of the source height) is blurred and darkened so the
# old text is unreadable under our English subtitles. None = clean source.
SOURCE_SUBTITLE_BAND: tuple[float, float] | None = (0.70, 0.945)
BAND_BLUR_SIGMA = 28
BAND_DARKEN = -0.18  # eq brightness offset inside the band
BAND_FEATHER = 18    # source pixels of soft edge at the band's top/bottom


class RenderError(RuntimeError):
    """FFmpeg failed to render a clip."""


@dataclass
class SourceInfo:
    width: int
    height: int
    duration: float
    fps: float


def source_info() -> SourceInfo:
    """Read source properties from data/video_info.json (or probe the video)."""
    if config.VIDEO_INFO_JSON.is_file():
        info = json.loads(config.VIDEO_INFO_JSON.read_text(encoding="utf-8"))
    else:
        from src.video_info import inspect_video
        info = inspect_video(config.INPUT_VIDEO)
    return SourceInfo(int(info["width"]), int(info["height"]),
                      float(info["duration_seconds"]), float(info["fps"]))


def _even(expr: str) -> str:
    """Wrap an FFmpeg expression so it yields an even integer (x264 needs it)."""
    return f"trunc(({expr})/2)*2"


def _select_expr(plan: EditPlan, shift: float) -> str:
    """'between(t,a,b)+...' over the kept ranges, moved by ``shift`` seconds
    (the pre-roll, minus half a frame for video so each range gets exactly
    round((b-a)*fps) frames)."""
    return "+".join(
        f"between(t,{a + shift:.4f},{b + shift:.4f})" for a, b in plan.segments
    )


def _preroll(plan: EditPlan, fps: float) -> float:
    """Seek this much before the clip: accurate seeking drops a frame lying
    exactly on the seek point, which would put video 1 frame ahead of audio."""
    return min(plan.ss, 2.0 / fps)


def build_filter_graph(src: SourceInfo, plan: EditPlan, ass_name: str | None) -> str:
    """FFmpeg -filter_complex graph: [0:v] -> [v], [0:a] -> [a]."""
    half = 0.5 / src.fps
    pre = _preroll(plan, src.fps)

    # 0) Jump cuts (also used for a single range, to drop the pre-roll).
    v_head = f"[0:v]select='{_select_expr(plan, pre - half)}',setpts=N/FRAME_RATE/TB,"
    a_head = f"[0:a]aselect='{_select_expr(plan, pre)}',asetpts=N/SR/TB,"

    # 1) Largest 9:16 window centered in the source (720p -> 405x720).
    crop_w = min(src.width, round(src.height * OUT_W / OUT_H))
    crop_h = min(src.height, round(src.width * OUT_H / OUT_W))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    graph = [f"{v_head}crop={crop_w}:{crop_h}:(in_w-{crop_w})/2:(in_h-{crop_h})/2[c]"]

    # 2) Hide the source's burned-in subtitles (blur + darken that band).
    current = "[c]"
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
            "[base][blurred]overlay=0:" + str(y0) + "[c2]",
        ]
        current = "[c2]"

    # 3) Zoom (per frame), crop back to 1080x1920, color, subtitles.
    zoom = f"({plan.zoom_expr})"
    chain = [
        f"scale=w='{_even(f'{OUT_W}*{zoom}')}':h='{_even(f'{OUT_H}*{zoom}')}'"
        ":eval=frame:flags=bicubic",
        f"crop={OUT_W}:{OUT_H}:(in_w-{OUT_W})/2:(in_h-{OUT_H})/2",
        "setsar=1",
        f"eq=contrast={CONTRAST}:saturation={SATURATION}",
    ]
    if ass_name:
        # Relative name: FFmpeg runs with cwd = the .ass folder (avoids Windows
        # drive-letter escaping). Fonts are found by name via DirectWrite.
        chain.append(f"ass={ass_name}")
    chain.append("format=yuv420p")
    graph.append(current + ",".join(chain) + "[v]")

    # Audio: same ranges, then loudness normalisation.
    graph.append(f"{a_head}{LOUDNORM},aresample=48000[a]")
    return ";".join(graph)


def render_plan(
    plan: EditPlan,
    out_path: Path,
    ass_path: Path | None = None,
    crf: int = 20,
    preset: str = "medium",
) -> Path:
    """Render ``plan`` to ``out_path`` (overwrites it). ``ass_path`` times
    must be on the clip timeline (0 = first frame of the clip)."""
    if not config.INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Source video not found: {config.INPUT_VIDEO}")
    src = source_info()
    end = plan.ss + plan.segments[-1][1]
    if not (0 <= plan.ss < end <= src.duration + 0.05):
        raise RenderError(f"Clip {plan.ss:.2f}-{end:.2f}s is outside the video (0-{src.duration:.2f}s).")

    ffmpeg = require_binary("ffmpeg")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_name(out_path.stem + ".part.mp4")

    graph = build_filter_graph(src, plan, ass_path.name if ass_path else None)
    cmd = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{plan.ss - _preroll(plan, src.fps):.4f}",
        "-t", f"{plan.segments[-1][1] + _preroll(plan, src.fps) + 0.1:.4f}",
        "-i", str(config.INPUT_VIDEO.resolve()),
        "-filter_complex", graph,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-t", f"{plan.duration:.3f}",
        "-movflags", "+faststart",
        str(tmp_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", cwd=ass_path.parent if ass_path else None,
                            check=False)
    if result.returncode != 0 or not tmp_out.is_file():
        tmp_out.unlink(missing_ok=True)
        raise RenderError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr.strip()[-2000:]}")
    tmp_out.replace(out_path)
    return out_path


def extract_frames(video: Path, times: list[float], out_dir: Path, prefix: str) -> list[Path]:
    """Save single frames (PNG) at the given clip times."""
    ffmpeg = require_binary("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(times, 1):
        png = out_dir / f"{prefix}_{i}_{t:.2f}s.png"
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


def stream_durations(video: Path) -> dict[str, float]:
    """Per-stream durations (seconds) - used to check audio/video sync."""
    ffprobe = require_binary("ffprobe")
    out = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_entries", "stream=codec_type,duration,nb_frames",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    result: dict[str, float] = {}
    for s in json.loads(out)["streams"]:
        result[s["codec_type"]] = float(s.get("duration", 0))
        if s["codec_type"] == "video" and s.get("nb_frames"):
            result["video_frames"] = float(s["nb_frames"])
    return result
