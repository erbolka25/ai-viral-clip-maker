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

# Audio extraction settings (what Whisper expects natively)
AUDIO_SAMPLE_RATE: int = 16_000
AUDIO_CHANNELS: int = 1

# Whisper settings
WHISPER_MODEL: str = "small"
WHISPER_CPU_COMPUTE_TYPE: str = "int8"
WHISPER_GPU_COMPUTE_TYPE: str = "float16"


def ensure_dirs() -> None:
    """Create the working directories if they do not exist yet."""
    for directory in (INPUT_DIR, OUTPUT_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
