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

# Phase 2: LLM-selected viral moments (user-editable).
HIGHLIGHTS_JSON: Path = DATA_DIR / "highlights.json"

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


# --- LLM (Phase 2) ---------------------------------------------------------
# Secrets live in .env (never committed); see .env.example.
ENV_FILE: Path = PROJECT_ROOT / ".env"

# LLM_PROVIDER in .env selects one of these; "groq" when unset.
LLM_DEFAULT_PROVIDER: str = "groq"

# Model used for each provider.
#  groq: gpt-oss-120b - Groq's strongest production model: 131K context,
#        strict JSON-schema output, reasoning model. (Llama 3.x models on Groq
#        were shut down for free/dev tiers on 2026-08-16.)
#  anthropic: Claude Opus 5.5 - not used yet.
LLM_MODELS: dict[str, str] = {
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-opus-5-5",
}

# Environment variable holding each provider's API key.
LLM_API_KEY_VARS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Default output-token cap per request. Kept modest on Groq because the free
# tier allows only ~8K tokens per minute (input + output combined).
LLM_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "groq": 4096,
    "anthropic": 16000,
}

# gpt-oss models think before answering; "low" | "medium" | "high".
GROQ_REASONING_EFFORT: str = "medium"


# --- Highlight boundaries -------------------------------------------------------
# Moments start at the beginning of a sentence and end at the end of one
# (python -m src.highlights --fix-boundaries, also applied to new highlights).
# A sentence ends at . ? ! , at the end of a transcript segment, or before a
# pause longer than SENTENCE_GAP. Boundaries move at most SENTENCE_MAX_SHIFT s
# if possible; the 20-60 s target (before jump cuts) is kept.
SENTENCE_GAP: float = 1.5
SENTENCE_MAX_SHIFT: float = 6.0
MOMENT_MIN_SECONDS: float = 20.0
MOMENT_MAX_SECONDS: float = 60.0


# --- Rendering / retention editing (Phase 3) ---------------------------------
# Clips are written to output/<RENDER_VERSION>/ so older versions are kept.
RENDER_VERSION: str = "v3"

# 1. Jump cuts: remove silences longer than JUMP_CUT_MIN_SILENCE between words,
#    keeping JUMP_CUT_PAD seconds around the words. Subtitles follow the cuts.
EDIT_JUMP_CUTS: bool = True
JUMP_CUT_MIN_SILENCE: float = 0.5
JUMP_CUT_PAD: float = 0.15
# Smart cuts: inside a speech gap only stretches whose audio is QUIET (every
# 50 ms window below JUMP_CUT_QUIET_DB dBFS) are removed; loud gaps are
# gameplay action (driving, shooting, music) and are kept.
# For reference, this video's audio: median -37 dBFS, action/speech -26..-32.
JUMP_CUT_SMART: bool = True
JUMP_CUT_QUIET_DB: float = -42.0
JUMP_CUT_WINDOW: float = 0.05
# Loud bursts shorter than this (footsteps, clicks) do not count as action.
JUMP_CUT_IGNORE_BURST: float = 0.3

# Audio crossfade at every cut (seconds, 0 = hard cut). Total audio length is
# unchanged, so audio and video stay in sync.
AUDIO_CROSSFADE: float = 0.04

# 2a. Punch-in zooms: framing alternates between these zoom levels every
#     PUNCH_ZOOM_INTERVAL seconds with a quick smooth transition. A level change
#     that falls near a jump cut happens instantly ON the cut (hides the cut).
#     When off, a slow linear zoom 1.0 -> SLOW_ZOOM_END is used instead.
EDIT_PUNCH_ZOOMS: bool = True
PUNCH_ZOOM_LEVELS: tuple[float, ...] = (1.0, 1.12, 1.05)
PUNCH_ZOOM_INTERVAL: float = 3.5
PUNCH_ZOOM_TRANSITION: float = 0.25
SLOW_ZOOM_END: float = 1.08

# 2b. Short extra punch-in on the loudest audio peaks.
EDIT_PEAK_PUNCH: bool = True
PEAK_PUNCH_AMOUNT: float = 0.08     # added to the current zoom level
PEAK_PUNCH_DURATION: float = 0.3
PEAK_PUNCH_MAX: int = 3             # per clip
PEAK_PUNCH_MIN_GAP: float = 4.0     # seconds between peak punches

# 3. Animated subtitles: words pop in as they are spoken, UPPERCASE, current
#    word yellow, 1-2 key words per line coloured (names green, strong words red).
EDIT_ANIMATED_SUBS: bool = True
SUB_UPPERCASE: bool = True
SUB_KEYWORD_COLORS: bool = True
# Stable lines: the whole line is laid out from the start (unspoken words
# invisible) so shown words never move; the pop is vertical only.
SUB_STABLE_LAYOUT: bool = True

# 4. Hook title pops in, stays HOOK_DURATION seconds, fades out.
EDIT_HOOK_ANIMATION: bool = True
HOOK_DURATION: float = 2.5

# 5. Thin progress bar that fills over the clip ("top" or "bottom").
EDIT_PROGRESS_BAR: bool = True
PROGRESS_BAR_POSITION: str = "top"
PROGRESS_BAR_HEIGHT: int = 12

# 6. Clean ending: cut the clip END_PAD seconds after the last spoken word.
EDIT_CLEAN_ENDING: bool = True
END_PAD: float = 0.3


def ensure_dirs() -> None:
    """Create the working directories if they do not exist yet."""
    for directory in (INPUT_DIR, OUTPUT_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
