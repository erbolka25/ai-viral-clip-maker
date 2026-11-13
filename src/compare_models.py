"""Compare Whisper models / glossary on short excerpts of data/audio.wav.

Everything is written to data/test/ only; the main transcript is read
(as the baseline) but never modified.

    python -m src.compare_models                     # default excerpts A and B
    python -m src.compare_models --models small medium
    python -m src.compare_models --excerpt C=600:720 # custom excerpt (seconds)

Output per excerpt:
    data/test/excerpt_<X>.wav                 the cut audio
    data/test/<X>_<model>_glossary.json/.txt  transcript (ORIGINAL-video timestamps)
    data/test/comparison_<X>.md               side-by-side table
    data/test/timings.json                    processing time per run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import config
from src.transcriber import load_glossary, save_transcript, transcribe

DEFAULT_EXCERPTS = ["A=0:180", "B=1530:1620"]  # 00:00-03:00, 25:30-27:00


@dataclass
class Excerpt:
    name: str
    start: float
    end: float


def parse_excerpt(spec: str) -> Excerpt:
    """Parse 'NAME=START:END' where START and END are seconds (e.g. B=1530:1620)."""
    name, _, rng = spec.partition("=")
    start_s, _, end_s = rng.partition(":")
    try:
        ex = Excerpt(name, float(start_s), float(end_s))
    except ValueError:
        ex = None
    if not name or ex is None:
        raise argparse.ArgumentTypeError(f"Bad excerpt '{spec}', expected NAME=START:END in seconds")
    if ex.end <= ex.start:
        raise argparse.ArgumentTypeError(f"Excerpt '{spec}': end must be after start")
    return ex


def cut_wav(src: Path, dst: Path, start: float, end: float) -> float:
    """Copy [start, end) seconds of a PCM WAV to ``dst``. Returns real duration."""
    with wave.open(str(src), "rb") as r:
        rate = r.getframerate()
        first = int(start * rate)
        last = min(int(end * rate), r.getnframes())
        if first >= last:
            raise ValueError(f"Excerpt {start}-{end}s is outside the audio.")
        r.setpos(first)
        frames = r.readframes(last - first)
        params = r.getparams()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)
    return (last - first) / rate


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def segments_in(segments: list[dict[str, Any]], ex: Excerpt) -> list[dict[str, Any]]:
    """Baseline segments that overlap the excerpt."""
    return [s for s in segments if s["end"] > ex.start and s["start"] < ex.end]


def build_table(columns: dict[str, list[dict[str, Any]]]) -> str:
    """Side-by-side markdown table, rows aligned on the first column's segments.

    Different runs split speech into different segments, so each segment of
    the other columns is placed in the row of the baseline segment it starts in.
    """
    names = list(columns)
    base = columns[names[0]]
    if not base:  # no baseline rows: use the union of starts
        base = sorted((s for c in columns.values() for s in c), key=lambda s: s["start"])
    starts = [s["start"] for s in base]

    def row_of(t: float) -> int:
        idx = 0
        for i, st in enumerate(starts):
            if t + 0.5 >= st:  # small tolerance for boundary jitter
                idx = i
        return idx

    rows: list[dict[str, list[str]]] = [{n: [] for n in names} for _ in starts]
    for n in names:
        for s in columns[n]:
            rows[row_of(s["start"])][n].append(s["text"].replace("|", "\\|"))

    out = ["| Time | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for st, row in zip(starts, rows):
        cells = [" ".join(row[n]) or "—" for n in names]
        out.append(f"| {mmss(st)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def count_terms(segments: list[dict[str, Any]], terms: list[str]) -> dict[str, int]:
    """Count case-insensitive whole-word occurrences of each term."""
    text = " ".join(s["text"] for s in segments)
    return {t: len(re.findall(rf"\b{re.escape(t)}\b", text, re.I)) for t in terms}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.compare_models", description=__doc__.splitlines()[0])
    p.add_argument("--excerpt", action="append", type=parse_excerpt,
                   help="NAME=START:END in seconds (repeatable; default A=0:180, B=1530:1620)")
    p.add_argument("--models", nargs="+", default=["small", "medium"])
    gl = p.add_mutually_exclusive_group()
    gl.add_argument("--glossary", default=str(config.GLOSSARY_TXT))
    gl.add_argument("--no-glossary", action="store_true", help="run without hotwords")
    args = p.parse_args(argv)

    excerpts: list[Excerpt] = args.excerpt or [parse_excerpt(e) for e in DEFAULT_EXCERPTS]
    glossary = [] if args.no_glossary else load_glossary(Path(args.glossary))
    # Suffix keeps glossary / no-glossary results in separate files.
    suffix = "_glossary" if glossary else "_noglossary"
    out_dir = config.TEST_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not config.AUDIO_WAV.is_file():
        print(f"Missing {config.AUDIO_WAV}; run python -m src.main first.")
        return 1
    baseline = (
        json.loads(config.TRANSCRIPT_JSON.read_text(encoding="utf-8"))["segments"]
        if config.TRANSCRIPT_JSON.is_file() else []
    )

    timings_path = out_dir / "timings.json"
    timings: list[dict[str, Any]] = (
        json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.is_file() else []
    )
    print(f"Glossary: {', '.join(glossary) or '(none)'}")

    for ex in excerpts:
        wav = out_dir / f"excerpt_{ex.name}.wav"
        dur = cut_wav(config.AUDIO_WAV, wav, ex.start, ex.end)
        print(f"\nExcerpt {ex.name}: {mmss(ex.start)}-{mmss(ex.end)} ({dur:.0f}s audio)")

        columns: dict[str, list[dict[str, Any]]] = {
            "existing small": segments_in(baseline, ex)
        }
        for model in args.models:
            label = f"{model}+glossary" if glossary else f"{model} (no glossary)"
            print(f"  {label:<16}", end="", flush=True)
            t0 = time.perf_counter()
            t = transcribe(wav, model_name=model, glossary=glossary, time_offset=ex.start)
            secs = time.perf_counter() - t0
            stem = out_dir / f"{ex.name}_{model}{suffix}"
            save_transcript(t, stem.with_suffix(".json"), stem.with_suffix(".txt"))
            print(f"{secs:6.1f}s  ({dur / secs:.1f}x real time, {len(t.segments)} segments)")
            timings = [r for r in timings if not (r["excerpt"] == ex.name and r["run"] == label)]
            timings.append({"excerpt": ex.name, "run": label, "model": model,
                            "audio_seconds": round(dur, 1), "processing_seconds": round(secs, 1),
                            "device": t.device, "compute_type": t.compute_type})
            columns[label] = t.to_dict()["segments"]

        # Always count the known name variants, plus any glossary terms.
        terms = list(dict.fromkeys(glossary + ["Boobie", "Raymond", "Booby", "Bobby"]))
        md = [f"# Excerpt {ex.name}: {mmss(ex.start)}–{mmss(ex.end)}", "",
              f"Glossary: {', '.join(glossary) or '(none)'}", "",
              "| Run | " + " | ".join(terms) + " |", "|---" * (len(terms) + 1) + "|"]
        for name, segs in columns.items():
            counts = count_terms(segs, terms)
            md.append(f"| {name} | " + " | ".join(str(counts[t]) for t in terms) + " |")
        md += ["", build_table(columns), ""]
        (out_dir / f"comparison_{ex.name}{'' if glossary else suffix}.md").write_text("\n".join(md), encoding="utf-8")

    timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print(f"\nResults in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
