"""Retention edit plan for one clip.

An :class:`EditPlan` says
* which parts of the source to keep (jump cuts + clean ending),
* how to zoom over time (alternating punch-ins + punches on loud peaks),
* how to map a source timestamp to the clip's timeline (for subtitles).

All cut points are snapped to the video frame grid, so audio and video are
cut at exactly the same instants and stay in sync.

Every effect is switched on/off in src/config.py (EDIT_* settings).
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src import config


@dataclass
class EditPlan:
    ss: float                                   # source time of the clip's first frame
    segments: list[tuple[float, float]]         # kept ranges, relative to ss, frame-aligned
    fps: float
    zoom_expr: str = "1"
    zoom_keys: list[tuple[float, float, float]] = field(default_factory=list)  # (t, level, transition)
    peaks: list[float] = field(default_factory=list)                           # clip times

    @property
    def duration(self) -> float:
        return sum(b - a for a, b in self.segments)

    @property
    def source_span(self) -> float:
        return self.segments[-1][1] - self.segments[0][0]

    @property
    def cut_times(self) -> list[float]:
        """Clip times where a jump cut happens."""
        out, acc = [], 0.0
        for a, b in self.segments[:-1]:
            acc += b - a
            out.append(acc)
        return out

    def to_clip_time(self, t_source: float) -> float:
        """Map an absolute source time to clip time (times inside a removed
        part snap to the next kept frame)."""
        rel = t_source - self.ss
        acc = 0.0
        for a, b in self.segments:
            if rel < a:
                return acc
            if rel <= b:
                return acc + rel - a
            acc += b - a
        return acc


# ---------------------------------------------------------------------------
# 1 + 6. Jump cuts and clean ending
# ---------------------------------------------------------------------------

def keep_ranges(moment: dict[str, Any], video_duration: float) -> list[tuple[float, float]]:
    """Absolute source ranges to keep for a moment."""
    start, end = float(moment["start"]), float(moment["end"])
    words = sorted(
        (float(w["start"]), float(w["end"])) for w in moment.get("words", [])
        if str(w.get("word", "")).strip()
    )
    words = [(a, b) for a, b in words if b > start and a < end]
    if not words:
        return [(start, end)]

    pad, gap_min = config.JUMP_CUT_PAD, config.JUMP_CUT_MIN_SILENCE
    if config.EDIT_CLEAN_ENDING:
        end = min(video_duration, words[-1][1] + config.END_PAD)
    if not config.EDIT_JUMP_CUTS:
        return [(start, end)]

    # Speech gaps longer than gap_min; their interior keeps `pad` around words.
    gaps: list[tuple[float, float]] = []
    if words[0][0] - start > gap_min:                      # leading silence
        gaps.append((start, words[0][0] - pad))
    spoken_until = words[0][1]
    for a, b in words[1:]:
        if a - spoken_until > gap_min:
            gaps.append((spoken_until + pad, a - pad))
        spoken_until = max(spoken_until, b)

    # Decide what to remove inside each gap.
    min_cut = gap_min - 2 * pad          # same minimum as a plain 0.5 s silence
    cuts: list[tuple[float, float]] = []
    for g0, g1 in gaps:
        if g1 <= g0:
            continue
        if config.JUMP_CUT_SMART:
            # Only quiet stretches go; loud parts are gameplay action.
            cuts += [(r0, r1) for r0, r1 in quiet_runs(g0, g1) if r1 - r0 >= min_cut]
        else:
            cuts.append((g0, g1))

    keep: list[tuple[float, float]] = []
    cur = start
    for c0, c1 in sorted(cuts):
        if c0 > cur:
            keep.append((cur, c0))
        cur = max(cur, c1)
    if end > cur:
        keep.append((cur, end))
    return keep


def quiet_runs(a: float, b: float) -> list[tuple[float, float]]:
    """Stretches inside [a, b] where every JUMP_CUT_WINDOW window of the audio
    is quieter than JUMP_CUT_QUIET_DB (RMS, dBFS)."""
    if not config.AUDIO_WAV.is_file():
        return [(a, b)]                   # no audio analysis possible: plain cut
    win = config.JUMP_CUT_WINDOW
    samples, rate = _read_audio(a, b)
    n = int(len(samples) / (rate * win))
    if n == 0:
        return []
    w = int(rate * win)
    rms = np.sqrt(np.mean(samples[: n * w].reshape(n, w) ** 2, axis=1))
    quiet = list(20 * np.log10(rms + 1e-9) < config.JUMP_CUT_QUIET_DB)
    # Short loud bursts (footsteps, clicks) are not action: treat them as quiet
    # when they sit between quiet windows, so they do not leave tiny slivers.
    max_burst = int(round(config.JUMP_CUT_IGNORE_BURST / win))
    i = 0
    while i < n:
        if not quiet[i]:
            j = i
            while j < n and not quiet[j]:
                j += 1
            if 0 < i and j < n and j - i <= max_burst:
                quiet[i:j] = [True] * (j - i)
            i = j
        else:
            i += 1
    runs: list[tuple[float, float]] = []
    run_start: int | None = None
    for i, q in enumerate(quiet + [False]):
        if q and run_start is None:
            run_start = i
        elif not q and run_start is not None:
            runs.append((a + run_start * win, min(b, a + i * win)))
            run_start = None
    return runs


def frame_align(ranges: list[tuple[float, float]], fps: float) -> tuple[float, list[tuple[float, float]]]:
    """Make ranges relative to the first one and snap them to whole frames."""
    ss = ranges[0][0]
    out: list[tuple[float, float]] = []
    for a, b in ranges:
        fa, fb = round((a - ss) * fps), round((b - ss) * fps)
        if out and fa <= round(out[-1][1] * fps):          # touching -> merge
            out[-1] = (out[-1][0], fb / fps)
        elif fb - fa >= 2:                                   # at least 2 frames
            out.append((fa / fps, fb / fps))
    return ss, out


# ---------------------------------------------------------------------------
# 2b. Loud audio peaks
# ---------------------------------------------------------------------------

def _read_audio(start: float, end: float) -> tuple[np.ndarray, int]:
    with wave.open(str(config.AUDIO_WAV), "rb") as wf:
        rate = wf.getframerate()
        first = max(0, int(start * rate))
        wf.setpos(min(first, wf.getnframes()))
        raw = wf.readframes(max(0, int((end - start) * rate)))
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rate


def find_peaks(plan: EditPlan) -> list[float]:
    """Clip times of the loudest moments (RMS over 50 ms windows)."""
    if not config.AUDIO_WAV.is_file():
        return []
    times, levels = [], []
    acc = 0.0
    for a, b in plan.segments:
        samples, rate = _read_audio(plan.ss + a, plan.ss + b)
        win = int(rate * 0.05)
        for i in range(0, len(samples) - win, win):
            rms = float(np.sqrt(np.mean(samples[i:i + win] ** 2)))
            times.append(acc + i / rate)
            levels.append(rms)
        acc += b - a
    if not levels:
        return []
    threshold = float(np.percentile(levels, 90))
    order = sorted(range(len(levels)), key=lambda i: -levels[i])
    peaks: list[float] = []
    for i in order:
        t = times[i]
        if levels[i] < threshold or len(peaks) >= config.PEAK_PUNCH_MAX:
            break
        if t < 0.5 or t > plan.duration - 0.6:
            continue
        if all(abs(t - p) >= config.PEAK_PUNCH_MIN_GAP for p in peaks):
            peaks.append(round(t, 3))
    return sorted(peaks)


# ---------------------------------------------------------------------------
# 2a. Zoom curve
# ---------------------------------------------------------------------------

def zoom_keyframes(plan: EditPlan) -> list[tuple[float, float, float]]:
    """(time, new level, transition seconds). Level changes every interval;
    one that falls within 1 s of a jump cut moves onto the cut and is instant."""
    levels = config.PUNCH_ZOOM_LEVELS
    keys: list[tuple[float, float, float]] = []
    cuts = plan.cut_times
    t, k, last = config.PUNCH_ZOOM_INTERVAL, 1, 0.0
    instant = 0.5 / plan.fps
    while t < plan.duration - 1.0:
        near = [c for c in cuts if abs(c - t) <= 1.0 and c - last >= 1.5]
        if near:
            at, trans = min(near, key=lambda c: abs(c - t)), instant
        else:
            at, trans = t, config.PUNCH_ZOOM_TRANSITION
        keys.append((round(at, 3), levels[k % len(levels)], trans))
        last, k = at, k + 1
        t = at + config.PUNCH_ZOOM_INTERVAL
    return keys


def _smooth(x: str) -> str:
    """smoothstep(x) = x*x*(3-2x) as an FFmpeg expression."""
    return f"({x})*({x})*(3-2*({x}))"


def zoom_expression(plan: EditPlan) -> str:
    """FFmpeg expression of t (clip time) giving the zoom factor."""
    if config.EDIT_PUNCH_ZOOMS:
        levels = config.PUNCH_ZOOM_LEVELS
        expr = f"{levels[0]}"
        prev = levels[0]
        for t0, level, trans in plan.zoom_keys:
            x = f"clip((t-{t0:.3f})/{trans:.4f},0,1)"
            expr += f"+({level - prev:.4f})*{_smooth(x)}"
            prev = level
    else:
        expr = f"1+{config.SLOW_ZOOM_END - 1:.4f}*min(t/{plan.duration:.3f},1)"
    if config.EDIT_PEAK_PUNCH:
        for p in plan.peaks:
            expr += (f"+{config.PEAK_PUNCH_AMOUNT}*sin(PI*clip((t-{p - 0.05:.3f})"
                     f"/{config.PEAK_PUNCH_DURATION},0,1))")
    return expr


# ---------------------------------------------------------------------------

def build_plan(moment: dict[str, Any], fps: float, video_duration: float) -> EditPlan:
    ss, segments = frame_align(keep_ranges(moment, video_duration), fps)
    plan = EditPlan(ss=ss, segments=segments, fps=fps)
    if config.EDIT_PUNCH_ZOOMS:
        plan.zoom_keys = zoom_keyframes(plan)
    if config.EDIT_PEAK_PUNCH:
        plan.peaks = find_peaks(plan)
    plan.zoom_expr = zoom_expression(plan)
    return plan


def clip_words(moment: dict[str, Any], plan: EditPlan) -> list[dict[str, Any]]:
    """The moment's words with clip-timeline timestamps."""
    out = []
    for w in moment.get("words", []):
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        a, b = plan.to_clip_time(float(w["start"])), plan.to_clip_time(float(w["end"]))
        if b <= 0 or a >= plan.duration:
            continue
        out.append({"start": a, "end": max(b, a + 0.05), "word": text})
    return out


def describe(plan: EditPlan) -> str:
    return (f"{len(plan.segments)} segment(s), {len(plan.cut_times)} cut(s), "
            f"{plan.source_span:.1f}s -> {plan.duration:.1f}s, "
            f"{len(plan.zoom_keys)} zoom change(s), peaks at {plan.peaks}")


__all__ = ["EditPlan", "build_plan", "clip_words", "describe"]
