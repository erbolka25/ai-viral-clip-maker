"""Render clips from data/highlights.json.

    python -m src.render --test              # top-scoring moment -> output/test_clip.mp4
    python -m src.render --test --no-hook    # same, without the hook title
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


def load_moments() -> list[dict[str, Any]]:
    if not config.HIGHLIGHTS_JSON.is_file():
        raise FileNotFoundError(f"{config.HIGHLIGHTS_JSON} not found - run: python -m src.highlights")
    data = json.loads(config.HIGHLIGHTS_JSON.read_text(encoding="utf-8"))
    return data["moments"]


def render_moment(moment: dict[str, Any], out_path: Path, hook: bool = True) -> Path:
    """Write the moment's .ass file and render the finished clip."""
    ass = write_ass(moment, SUBS_DIR / f"{out_path.stem}.ass", hook=hook)
    return render_clip(float(moment["start"]), float(moment["end"]), out_path, ass_path=ass)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.render", description=__doc__.splitlines()[0])
    p.add_argument("--test", action="store_true", help="render the top moment to output/test_clip.mp4")
    p.add_argument("--no-hook", action="store_true", help="leave out the hook title")
    p.add_argument("--frames", action="store_true", help="also save 3 PNG frames to data/test/frames")
    args = p.parse_args(argv)

    try:
        moments = load_moments()
        if args.test:
            best = max(moments, key=lambda m: m["score"])
            out = config.OUTPUT_DIR / "test_clip.mp4"
            t0 = time.perf_counter()
            render_moment(best, out, hook=not args.no_hook)
            print(f"#{best['id']} {best['title']!r} -> {out} ({time.perf_counter() - t0:.1f}s)")
            if args.frames:
                d = probe(out)["duration_seconds"]
                for f in extract_frames(out, [1.0, d / 2, d - 1.5], config.TEST_DIR / "frames", "test"):
                    print("frame:", f)
            return 0
        p.print_help()
        return 1
    except (FileNotFoundError, RenderError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
