"""Shared fixtures and synthetic-data helpers.

Nothing here needs a real video, FFmpeg, Whisper models or API keys.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src import config

SAMPLE_RATE = 16_000


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every data file at an empty temp folder so tests never read the
    real data/ (audio.wav, glossary.txt, video_info.json) of a local checkout."""
    monkeypatch.setattr(config, "AUDIO_WAV", tmp_path / "audio.wav")
    monkeypatch.setattr(config, "GLOSSARY_TXT", tmp_path / "glossary.txt")
    monkeypatch.setattr(config, "VIDEO_INFO_JSON", tmp_path / "video_info.json")
    return tmp_path


def w(start: float, end: float, text: str) -> dict[str, Any]:
    """A Whisper-style word: text with a leading space."""
    return {"start": start, "end": end, "word": f" {text}"}


def make_transcript(
    n_sentences: int,
    words_per_sentence: int = 8,
    t0: float = 0.0,
    step: float = 0.5,
    word_len: float = 0.4,
    duration: float | None = None,
) -> dict[str, Any]:
    """Continuous speech: one segment per sentence, a word every ``step`` s.

    Each sentence is 'Hello word word ... end.' (capitalised start, period
    at the end), so sentence boundaries are unambiguous.
    """
    segments = []
    t = t0
    for _ in range(n_sentences):
        words = []
        for k in range(words_per_sentence):
            text = "Hello" if k == 0 else "end." if k == words_per_sentence - 1 else "word"
            words.append(w(round(t, 2), round(t + word_len, 2), text))
            t += step
        segments.append({
            "start": words[0]["start"], "end": words[-1]["end"],
            "text": "".join(x["word"] for x in words), "words": words,
        })
    end = segments[-1]["end"] if segments else t0
    return {"duration": duration if duration is not None else end + 5.0, "segments": segments}


def write_wav(path: Path, duration: float, loud: list[tuple[float, float]]) -> Path:
    """16 kHz mono 16-bit WAV: silence, with a 440 Hz tone (about -13 dBFS)
    inside each ``loud`` (start, end) range."""
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    signal = np.zeros(n, dtype=np.float32)
    for a, b in loud:
        i, j = int(a * SAMPLE_RATE), int(b * SAMPLE_RATE)
        signal[i:j] = 0.3 * np.sin(2 * np.pi * 440 * t[i:j])
    pcm = (signal * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return path
