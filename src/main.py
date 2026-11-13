"""Phase 1 pipeline: inspect video -> extract audio -> transcribe.

Run from the project root:

    python -m src.main             # asks before overwriting existing outputs
    python -m src.main --yes       # overwrite without asking
    python -m src.main --skip-existing   # reuse existing outputs
    python -m src.main --model small     # use a different Whisper model
    python -m src.main --no-glossary     # disable glossary hotwords
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from src import config
from src.deps import DependencyError, check_dependencies

# rich is a hard dependency; give a plain message if it is missing.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    from rich.prompt import Confirm
    from rich.table import Table
except ImportError:  # pragma: no cover
    print("Missing package 'rich'. Run:  pip install -r requirements.txt")
    sys.exit(1)

console = Console()


class Overwrite:
    """Policy for handling outputs that already exist."""

    ASK = "ask"
    YES = "yes"
    SKIP = "skip"


def should_write(paths: list[Path], policy: str) -> bool:
    """Decide whether a step should (re)create its output files.

    Returns False when outputs exist and the user chose to keep them.
    """
    existing = [p for p in paths if p.exists()]
    if not existing:
        return True
    if policy == Overwrite.YES:
        return True
    if policy == Overwrite.SKIP:
        return False
    names = ", ".join(p.name for p in existing)
    if not sys.stdin.isatty():
        console.print(f"[yellow]{names} exists; non-interactive run, keeping it.[/]")
        return False
    try:
        return Confirm.ask(f"[yellow]{names} already exists. Overwrite?[/]", default=False)
    except EOFError:
        # Some shells (e.g. Git Bash pipes) report a TTY but provide no input.
        console.print("[yellow]No input available; keeping existing file.[/]")
        return False


def step_dependencies() -> None:
    console.rule("[bold]1. Dependencies")
    report = check_dependencies()
    for name, where in report.found.items():
        console.print(f"  [green]OK[/]  {name}: [dim]{where}[/]")
    for name, hint in report.missing.items():
        console.print(f"  [red]MISSING[/]  {name}  ->  {hint}")
    if not report.ok:
        raise DependencyError("Install the missing dependencies above and try again.")


def step_input() -> None:
    console.rule("[bold]2. Input video")
    config.ensure_dirs()
    video = config.INPUT_VIDEO
    if not video.is_file():
        raise FileNotFoundError(
            "Input video not found.\n"
            f"  Expected: {video}\n"
            f"  Place your source video in the '{config.INPUT_DIR.name}' folder "
            f"and name it '{video.name}'."
        )
    if video.stat().st_size == 0:
        raise ValueError(f"Input video is empty (0 bytes): {video}")
    console.print(f"  [green]OK[/]  {video}")


def step_inspect(policy: str) -> dict[str, Any]:
    from src.video_info import inspect_video, save_video_info

    console.rule("[bold]3. Video inspection")
    out = config.VIDEO_INFO_JSON
    if not should_write([out], policy):
        console.print(f"  [cyan]Reusing[/] {out}")
        return json.loads(out.read_text(encoding="utf-8"))
    info = inspect_video(config.INPUT_VIDEO)
    save_video_info(info, out)
    console.print(f"  [green]Saved[/] {out}")
    return info


def step_audio(policy: str) -> Path:
    from src.audio_extractor import extract_audio

    console.rule("[bold]4. Audio extraction")
    out = config.AUDIO_WAV
    if not should_write([out], policy):
        console.print(f"  [cyan]Reusing[/] {out}")
        return out
    with console.status("Extracting 16 kHz mono WAV with FFmpeg..."):
        t0 = time.perf_counter()
        extract_audio(config.INPUT_VIDEO, out)
    console.print(f"  [green]Saved[/] {out} ({time.perf_counter() - t0:.1f}s)")
    return out


def resolve_glossary(arg: str | None, disabled: bool) -> list[str]:
    """Return glossary terms from --glossary, or glossary.txt if it exists."""
    from src.transcriber import load_glossary

    if disabled:
        return []
    if arg:  # explicit path: must exist
        return load_glossary(Path(arg))
    if config.GLOSSARY_TXT.is_file():
        return load_glossary(config.GLOSSARY_TXT)
    return []


def step_transcribe(
    policy: str, model: str, language: str | None, glossary: list[str]
) -> dict[str, Any]:
    from src.transcriber import detect_device, save_transcript, transcribe

    console.rule("[bold]5. Speech recognition")
    outs = [config.TRANSCRIPT_JSON, config.TRANSCRIPT_TXT]
    if not should_write(outs, policy):
        console.print(f"  [cyan]Reusing[/] {config.TRANSCRIPT_JSON}")
        return json.loads(config.TRANSCRIPT_JSON.read_text(encoding="utf-8"))

    device, compute = detect_device()
    console.print(
        f"  Model [bold]{model}[/] on [bold]{device}[/] ({compute}). "
        "First run downloads the model (small ~0.5 GB, medium ~1.5 GB)."
    )
    console.print(f"  Glossary: {', '.join(glossary) if glossary else '(none)'}")
    t0 = time.perf_counter()
    with Progress(
        TextColumn("  Transcribing"),
        BarColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TextColumn("[dim]{task.fields[pos]}[/]"),
        TimeElapsedColumn(),
        console=console,
    ) as bar:
        task = bar.add_task("transcribe", total=100, pos="loading model...")

        def on_progress(done: float, total: float) -> None:
            pct = 100 * done / total if total else 100
            bar.update(task, completed=pct, pos=f"{done:.0f}s / {total:.0f}s audio")

        transcript = transcribe(
            config.AUDIO_WAV, model_name=model, language=language,
            progress=on_progress, glossary=glossary,
        )
    save_transcript(transcript, config.TRANSCRIPT_JSON, config.TRANSCRIPT_TXT)
    console.print(
        f"  [green]Saved[/] {config.TRANSCRIPT_JSON.name} and {config.TRANSCRIPT_TXT.name} "
        f"({time.perf_counter() - t0:.0f}s)"
    )
    return transcript.to_dict()


def print_summary(info: dict[str, Any], transcript: dict[str, Any]) -> None:
    console.rule("[bold green]Summary")
    table = Table(show_header=False, box=None, padding=(0, 2))
    dur = info.get("duration_seconds") or 0
    rows = [
        ("Duration", f"{dur:.1f}s ({int(dur // 60)}m {dur % 60:.0f}s)"),
        ("Resolution", f"{info.get('resolution')} ({info.get('orientation')})"),
        ("FPS", str(info.get("fps"))),
        ("Video codec", str(info.get("video_codec"))),
        ("Audio codec", str(info.get("audio_codec"))),
        ("File size", f"{info.get('file_size_mb')} MB"),
        ("Language", f"{transcript['language']} ({transcript['language_probability']:.0%})"),
        ("Segments", str(len(transcript["segments"]))),
        ("Speech", f"{sum(s['end'] - s['start'] for s in transcript['segments']):.0f}s"),
    ]
    for k, v in rows:
        table.add_row(f"[bold]{k}[/]", v)
    console.print(table)

    files = "\n".join(
        f"  {p.relative_to(config.PROJECT_ROOT)}"
        for p in (config.VIDEO_INFO_JSON, config.AUDIO_WAV,
                  config.TRANSCRIPT_JSON, config.TRANSCRIPT_TXT)
        if p.exists()
    )
    console.print(Panel(files, title="Generated files", expand=False))

    preview = transcript["segments"][:5]
    if preview:
        console.print("[bold]Transcript preview:[/]")
        for s in preview:
            console.print(f"  [dim]{s['start']:7.2f}s[/] {s['text']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m src.main", description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("-y", "--yes", action="store_true", help="overwrite existing outputs")
    g.add_argument("--skip-existing", action="store_true", help="reuse existing outputs")
    p.add_argument("--model", default=config.WHISPER_MODEL,
                   help="Whisper model: tiny/base/small/medium/large-v3 "
                        f"(default: {config.WHISPER_MODEL})")
    p.add_argument("--language", default=None,
                   help="force source language code (default: auto-detect)")
    gl = p.add_mutually_exclusive_group()
    gl.add_argument("--glossary", metavar="PATH", default=None,
                    help="terms file, one per line (default: glossary.txt if it exists)")
    gl.add_argument("--no-glossary", action="store_true", help="disable the glossary")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    policy = Overwrite.YES if args.yes else Overwrite.SKIP if args.skip_existing else Overwrite.ASK

    console.print(Panel.fit("[bold magenta]GTA 6 Viral Clip Maker[/] - Phase 1: Video processing"))
    try:
        step_dependencies()
        step_input()
        info = step_inspect(policy)
        step_audio(policy)
        glossary = resolve_glossary(args.glossary, args.no_glossary)
        transcript = step_transcribe(policy, args.model, args.language, glossary)
        print_summary(info, transcript)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/]")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level: show a clean message
        console.print(f"\n[bold red]Error:[/] {exc}")
        return 1
    console.print("[bold green]Phase 1 complete.[/]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
