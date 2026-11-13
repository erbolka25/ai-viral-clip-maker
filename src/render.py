"""Render vertical clips from data/highlights.json.

    python -m src.render                 # top 10 moments -> output/<version>/gta6_clip_01..10.mp4
    python -m src.render --top 5         # top 5 instead
    python -m src.render --clip 3        # re-render clip 3 after editing highlights.json
    python -m src.render --test          # top moment -> data/test/test_clip.mp4 (+ frames)
    python -m src.render --new           # render new moments (score >= 7) as clip 11, 12, ...
    python -m src.render -y              # batch: overwrite existing clips without asking

<version> is config.RENDER_VERSION (e.g. v2); older versions stay in their own
folders. Clip numbers are fixed when a batch is rendered (clips_manifest.json
maps clip number -> moment id), so editing scores later does not renumber
clips. --clip N always uses the CURRENT text/times in highlights.json.
Editing effects are switched on/off in src/config.py (EDIT_* settings).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from src import config
from src.editing import EditPlan, build_plan, describe
from src.renderer import RenderError, extract_frames, render_plan, source_info, stream_durations
from src.subtitles import write_ass

OUT_DIR = config.OUTPUT_DIR / config.RENDER_VERSION
SUBS_DIR = config.DATA_DIR / "subs" / config.RENDER_VERSION
MANIFEST = OUT_DIR / "clips_manifest.json"
SUMMARY = OUT_DIR / "clips_summary.md"
CLIP_NAME = "gta6_clip_{:02d}.mp4"


def load_moments() -> list[dict[str, Any]]:
    if not config.HIGHLIGHTS_JSON.is_file():
        raise FileNotFoundError(f"{config.HIGHLIGHTS_JSON} not found - run: python -m src.highlights")
    data = json.loads(config.HIGHLIGHTS_JSON.read_text(encoding="utf-8"))
    return data["moments"]


def render_moment(moment: dict[str, Any], out_path: Path, hook: bool = True) -> EditPlan:
    """Plan the edit, write the .ass overlay and render the finished clip."""
    src = source_info()
    plan = build_plan(moment, src.fps, src.duration)
    ass = write_ass(moment, plan, SUBS_DIR / f"{out_path.stem}.ass", hook=hook)
    render_plan(plan, out_path, ass_path=ass)
    return plan


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def write_summary(manifest: list[dict[str, Any]], moments: list[dict[str, Any]]) -> Path:
    """clips_summary.md from the manifest + current highlights.json."""
    by_id = {m["id"]: m for m in moments}
    lines = [
        f"# GTA 6 clips ({config.RENDER_VERSION})", "",
        "| # | File | Hook title | Title | Category | Score | Source start - end | Source length | Clip length |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    # Sorted by score (best first); ties keep clip order.
    ordered = sorted(manifest, key=lambda e: (-by_id.get(e["moment_id"], {}).get("score", 0), e["clip"]))
    for e in ordered:
        m = by_id.get(e["moment_id"])
        exists = (OUT_DIR / e["file"]).is_file()
        name = e["file"] if exists else f"{e['file']} (not rendered)"
        if not m:
            lines.append(f"| {e['clip']} | {name} | (moment {e['moment_id']} missing) | | | | | | |")
            continue
        clip_len = f"{e['clip_seconds']:.1f}s" if exists and e.get("clip_seconds") else "-"
        lines.append(
            f"| {e['clip']} | {name} | {m['hook_title']} | {m['title']} | {m['category']} | "
            f"{m['score']} | {mmss(m['start'])} - {mmss(m['end'])} | "
            f"{m['end'] - m['start']:.1f}s | {clip_len} |"
        )
    lines += ["", "Clip length is after jump cuts and the clean ending.", "",
              "Re-render one clip after editing data/highlights.json: "
              "`python -m src.render --clip N`", ""]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    return SUMMARY


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:  # no interactive input available
        return False


def _render_entry(entry: dict[str, Any], moment: dict[str, Any]) -> None:
    out = OUT_DIR / entry["file"]
    t0 = time.perf_counter()
    plan = render_moment(moment, out)
    entry["source_seconds"] = round(float(moment["end"]) - float(moment["start"]), 2)
    entry["clip_seconds"] = round(plan.duration, 2)
    entry["render_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"clip {entry['clip']:>2}: moment #{moment['id']} {moment['hook_title']!r}  "
          f"{entry['source_seconds']:.1f}s -> {entry['clip_seconds']:.1f}s  "
          f"({entry['render_seconds']:.0f}s)  [{describe(plan)}]")


def render_batch(top: int, overwrite: bool) -> int:
    moments = load_moments()
    chosen = sorted(moments, key=lambda m: (-m["score"], m["start"]))[:top]
    existing = [p for i in range(1, len(chosen) + 1) if (p := OUT_DIR / CLIP_NAME.format(i)).exists()]
    if existing and not overwrite and not confirm(
            f"{len(existing)} clip file(s) already exist in {OUT_DIR}. Overwrite them?"):
        print("Nothing rendered. Use -y to overwrite, or --clip N for a single clip.")
        return 1

    manifest = [{"clip": n, "file": CLIP_NAME.format(n), "moment_id": m["id"]}
                for n, m in enumerate(chosen, 1)]
    t_all = time.perf_counter()
    for entry, m in zip(manifest, chosen):
        _render_entry(entry, m)
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest, moments)
    print(f"\n{len(chosen)} clips in {OUT_DIR} ({time.perf_counter() - t_all:.0f}s)")
    print(f"Summary: {SUMMARY}")
    return 0


def _load_manifest() -> list[dict[str, Any]] | None:
    """This version's manifest, else the newest other version's (same clip numbers)."""
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    others = sorted(config.OUTPUT_DIR.glob("*/clips_manifest.json"), reverse=True)
    if others:
        print(f"Using clip numbers from {others[0].parent.name}/clips_manifest.json")
        return [{k: e[k] for k in ("clip", "file", "moment_id")}
                for e in json.loads(others[0].read_text(encoding="utf-8"))]
    return None


def render_single(clip: int) -> int:
    manifest = _load_manifest()
    if manifest is None:
        print("No clips_manifest.json yet - run the batch first: python -m src.render")
        return 1
    entry = next((e for e in manifest if e["clip"] == clip), None)
    if entry is None:
        print(f"Clip {clip} not found; valid: {', '.join(str(e['clip']) for e in manifest)}")
        return 1
    moments = load_moments()
    moment = next((m for m in moments if m["id"] == entry["moment_id"]), None)
    if moment is None:
        print(f"Moment id {entry['moment_id']} (clip {clip}) is no longer in highlights.json.")
        return 1
    _render_entry(entry, moment)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest, moments)
    print(f"-> {OUT_DIR / entry['file']}")
    return 0


def render_new(min_score: int) -> int:
    """Render moments that are not clips yet (score >= min_score) as the next
    clip numbers, best score first. Existing clips are not touched."""
    manifest = _load_manifest()
    if manifest is None:
        print("No clips_manifest.json yet - run the batch first: python -m src.render")
        return 1
    moments = load_moments()
    done = {e["moment_id"] for e in manifest}
    todo = sorted((m for m in moments if m["id"] not in done and m["score"] >= min_score),
                  key=lambda m: (-m["score"], m["start"]))
    if not todo:
        print(f"No new moments with score >= {min_score}.")
        return 0
    n = max(e["clip"] for e in manifest)
    for m in todo:
        n += 1
        entry = {"clip": n, "file": CLIP_NAME.format(n), "moment_id": m["id"]}
        manifest.append(entry)
        _render_entry(entry, m)
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(manifest, moments)
    print(f"Summary: {SUMMARY}")
    return 0


def render_test(hook: bool) -> int:
    best = max(load_moments(), key=lambda m: m["score"])
    out = config.TEST_DIR / "test_clip.mp4"
    t0 = time.perf_counter()
    plan = render_moment(best, out, hook=hook)
    print(f"#{best['id']} {best['title']!r} -> {out} ({time.perf_counter() - t0:.1f}s) [{describe(plan)}]")
    print("streams:", stream_durations(out))
    d = plan.duration
    for f in extract_frames(out, [1.0, d / 2, d - 1.0], config.TEST_DIR / "frames", "test"):
        print("frame:", f)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.render", description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:8]))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--clip", type=int, metavar="N", help="re-render clip N (gta6_clip_NN.mp4)")
    mode.add_argument("--test", action="store_true", help="render the top moment to data/test/")
    mode.add_argument("--new", action="store_true",
                      help=f"render moments that are not clips yet (score >= {config.MIN_NEW_SCORE}) "
                           "as the next clip numbers")
    p.add_argument("--top", type=int, default=10, help="number of clips in a batch (default 10)")
    p.add_argument("-y", "--yes", action="store_true", help="batch: overwrite existing clips")
    p.add_argument("--no-hook", action="store_true", help="--test only: leave out the hook title")
    args = p.parse_args(argv)

    config.ensure_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if args.test:
            return render_test(hook=not args.no_hook)
        if args.clip is not None:
            return render_single(args.clip)
        if args.new:
            return render_new(config.MIN_NEW_SCORE)
        return render_batch(args.top, args.yes)
    except (FileNotFoundError, RenderError, KeyError, ValueError) as exc:
        print(f"\nError: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130



if __name__ == "__main__":
    sys.exit(main())
