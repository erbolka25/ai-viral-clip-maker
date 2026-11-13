"""Render vertical clips from data/highlights.json.

    python -m src.render                 # top 10 moments -> output/gta6_clip_01.mp4 ... _10.mp4
    python -m src.render --top 5         # top 5 instead
    python -m src.render --clip 3        # re-render clip 3 after editing highlights.json
    python -m src.render --test          # top moment -> data/test/test_clip.mp4 (+ frames)
    python -m src.render -y              # batch: overwrite existing clips without asking

Clip numbers are fixed when the batch is rendered (output/clips_manifest.json
maps clip number -> moment id), so editing scores later does not renumber
clips. --clip N always uses the CURRENT text/times in highlights.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from src import config
from src.renderer import RenderError, extract_frames, probe, render_clip
from src.subtitles import write_ass

SUBS_DIR = config.DATA_DIR / "subs"
MANIFEST = config.OUTPUT_DIR / "clips_manifest.json"
SUMMARY = config.OUTPUT_DIR / "clips_summary.md"
CLIP_NAME = "gta6_clip_{:02d}.mp4"


def load_moments() -> list[dict[str, Any]]:
    if not config.HIGHLIGHTS_JSON.is_file():
        raise FileNotFoundError(f"{config.HIGHLIGHTS_JSON} not found - run: python -m src.highlights")
    data = json.loads(config.HIGHLIGHTS_JSON.read_text(encoding="utf-8"))
    return data["moments"]


def render_moment(moment: dict[str, Any], out_path: Path, hook: bool = True) -> Path:
    """Write the moment's .ass file and render the finished clip."""
    ass = write_ass(moment, SUBS_DIR / f"{out_path.stem}.ass", hook=hook)
    return render_clip(float(moment["start"]), float(moment["end"]), out_path, ass_path=ass)


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def write_summary(manifest: list[dict[str, Any]], moments: list[dict[str, Any]]) -> Path:
    """output/clips_summary.md from the manifest + current highlights.json."""
    by_id = {m["id"]: m for m in moments}
    lines = [
        "# GTA 6 clips", "",
        "| # | File | Hook title | Title | Category | Score | Start - End | Length |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in manifest:
        m = by_id.get(entry["moment_id"])
        if not m:
            lines.append(f"| {entry['clip']} | {entry['file']} | (moment {entry['moment_id']} "
                         "missing from highlights.json) | | | | | |")
            continue
        lines.append(
            f"| {entry['clip']} | {entry['file']} | {m['hook_title']} | {m['title']} | "
            f"{m['category']} | {m['score']} | {mmss(m['start'])} - {mmss(m['end'])} | "
            f"{m['end'] - m['start']:.1f}s |"
        )
    lines += ["", "Re-render one clip after editing data/highlights.json: "
              "`python -m src.render --clip N`", ""]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    return SUMMARY


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:  # no interactive input available
        return False


def render_batch(top: int, overwrite: bool) -> int:
    moments = load_moments()
    chosen = sorted(moments, key=lambda m: (-m["score"], m["start"]))[:top]
    existing = [config.OUTPUT_DIR / CLIP_NAME.format(i) for i in range(1, len(chosen) + 1)]
    existing = [p for p in existing if p.exists()]
    if existing and not overwrite and not confirm(
            f"{len(existing)} clip file(s) already exist in output/. Overwrite them?"):
        print("Nothing rendered. Use -y to overwrite, or --clip N for a single clip.")
        return 1

    manifest: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for n, m in enumerate(chosen, 1):
        out = config.OUTPUT_DIR / CLIP_NAME.format(n)
        t0 = time.perf_counter()
        print(f"[{n}/{len(chosen)}] moment #{m['id']} score {m['score']} "
              f"{m['hook_title']!r} ({m['end'] - m['start']:.0f}s)...", end=" ", flush=True)
        render_moment(m, out)
        print(f"{time.perf_counter() - t0:.0f}s")
        manifest.append({"clip": n, "file": out.name, "moment_id": m["id"]})

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest, moments)
    print(f"\n{len(chosen)} clips in {config.OUTPUT_DIR} ({time.perf_counter() - t_all:.0f}s)")
    print(f"Summary: {SUMMARY}")
    return 0


def render_single(clip: int) -> int:
    if not MANIFEST.is_file():
        print("No clips_manifest.json yet - run the batch first: python -m src.render")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = next((e for e in manifest if e["clip"] == clip), None)
    if entry is None:
        print(f"Clip {clip} not found; valid: {', '.join(str(e['clip']) for e in manifest)}")
        return 1
    moments = load_moments()
    moment = next((m for m in moments if m["id"] == entry["moment_id"]), None)
    if moment is None:
        print(f"Moment id {entry['moment_id']} (clip {clip}) is no longer in highlights.json.")
        return 1
    out = config.OUTPUT_DIR / entry["file"]
    t0 = time.perf_counter()
    render_moment(moment, out)
    write_summary(manifest, moments)
    print(f"Clip {clip} (moment #{moment['id']}, {moment['hook_title']!r}) -> {out} "
          f"({time.perf_counter() - t0:.0f}s)")
    return 0


def render_test(hook: bool) -> int:
    best = max(load_moments(), key=lambda m: m["score"])
    out = config.TEST_DIR / "test_clip.mp4"
    t0 = time.perf_counter()
    render_moment(best, out, hook=hook)
    print(f"#{best['id']} {best['title']!r} -> {out} ({time.perf_counter() - t0:.1f}s)")
    d = probe(out)["duration_seconds"]
    for f in extract_frames(out, [1.0, d / 2, d - 1.5], config.TEST_DIR / "frames", "test"):
        print("frame:", f)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.render", description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:8]))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--clip", type=int, metavar="N", help="re-render clip N (gta6_clip_NN.mp4)")
    mode.add_argument("--test", action="store_true", help="render the top moment to data/test/")
    p.add_argument("--top", type=int, default=10, help="number of clips in a batch (default 10)")
    p.add_argument("-y", "--yes", action="store_true", help="batch: overwrite existing clips")
    p.add_argument("--no-hook", action="store_true", help="--test only: leave out the hook title")
    args = p.parse_args(argv)

    config.ensure_dirs()
    try:
        if args.test:
            return render_test(hook=not args.no_hook)
        if args.clip is not None:
            return render_single(args.clip)
        return render_batch(args.top, args.yes)
    except (FileNotFoundError, RenderError, KeyError, ValueError) as exc:
        print(f"\nError: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
