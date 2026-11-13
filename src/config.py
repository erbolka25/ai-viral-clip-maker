"""Central project paths and settings.

All paths are resolved relative to the project root (the folder that contains
``src/``), so the app works no matter which directory it is launched from.
"""

from __future__ import annotations

from pathlib import Path

# Project root = parent of the src/ directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

INPUT_DIR: Path = PROJECT_ROOT / "input"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
DATA_DIR: Path = PROJECT_ROOT / "data"

# Phase 1 inputs / outputs
INPUT_VIDEO: Path = INPUT_DIR / "gta6.mp4"
VIDEO_INFO_JSON: Path = DATA_DIR / "video_info.json"
AUDIO_WAV: Path = DATA_DIR / "audio.wav"
TRANSCRIPT_JSON: Path = DATA_DIR / "transcript.json"
TRANSCRIPT_TXT: Path = DATA_DIR / "transcript.txt"

# Names / slang Whisper should spell correctly (one term per line).
GLOSSARY_TXT: Path = PROJECT_ROOT / "glossary.txt"

# Scratch area for model comparisons / experiments.
TEST_DIR: Path = DATA_DIR / "test"

# Audio extraction settings (what Whisper expects natively)
AUDIO_SAMPLE_RATE: int = 16_000
AUDIO_CHANNELS: int = 1

# Whisper settings
# "medium" spells names/slang far better than "small" (see data/test/
# comparison_*.md) at roughly 3-5x the CPU time.
WHISPER_MODEL: str = "medium"
WHISPER_CPU_COMPUTE_TYPE: str = "int8"
WHISPER_GPU_COMPUTE_TYPE: str = "float16"


def ensure_dirs() -> None:
    """Create the working directories if they do not exist yet."""
    for directory in (INPUT_DIR, OUTPUT_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
