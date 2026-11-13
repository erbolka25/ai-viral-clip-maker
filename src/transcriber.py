"""Speech recognition with Faster-Whisper.

* Uses CUDA when available, otherwise CPU with int8 quantisation.
* Detects the spoken language automatically.
* Keeps the ORIGINAL language (task="transcribe", never "translate");
  English translation happens in a later phase.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from src import config

if TYPE_CHECKING:
    import numpy as np

# Windows without Developer Mode can't create symlinks; HF Hub still works
# (it copies files) but prints a noisy warning on every model load.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Called with (seconds_processed, total_seconds) as segments are produced.
ProgressCallback = Callable[[float, float], None]


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    language_probability: float
    duration: float
    model: str
    device: str
    compute_type: str
    segments: list[Segment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_device() -> tuple[str, str]:
    """Return (device, compute_type): CUDA/float16 if usable, else CPU/int8."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", config.WHISPER_GPU_COMPUTE_TYPE
    except Exception:  # noqa: BLE001 - any CUDA probing failure -> CPU
        pass
    return "cpu", config.WHISPER_CPU_COMPUTE_TYPE


def load_wav(audio_path: Path) -> "np.ndarray":
    """Load a 16 kHz mono 16-bit PCM WAV as float32 samples in [-1, 1].

    We decode the WAV ourselves instead of letting faster-whisper use PyAV:
    it is faster for plain PCM and avoids PyAV version incompatibilities
    (PyAV 19 broke faster-whisper 1.2.1's decode_audio).
    """
    import wave

    import numpy as np

    with wave.open(str(audio_path), "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            raise ValueError(f"{audio_path.name} must be 16-bit mono PCM WAV.")
        if wf.getframerate() != config.AUDIO_SAMPLE_RATE:
            raise ValueError(
                f"{audio_path.name} is {wf.getframerate()} Hz; expected "
                f"{config.AUDIO_SAMPLE_RATE} Hz. Re-run audio extraction."
            )
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def transcribe(
    audio_path: Path,
    model_name: str = config.WHISPER_MODEL,
    language: str | None = None,
    progress: ProgressCallback | None = None,
) -> Transcript:
    """Transcribe ``audio_path`` and return a :class:`Transcript`.

    ``language=None`` enables automatic language detection.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Imported lazily so the rest of the app works (and can report a clear
    # error) even if faster-whisper is not installed.
    from faster_whisper import WhisperModel

    device, compute_type = detect_device()
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception:
        if device == "cpu":
            raise
        # GPU initialisation can fail (missing cuDNN etc.) - fall back to CPU.
        device, compute_type = "cpu", config.WHISPER_CPU_COMPUTE_TYPE
        model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments_iter, info = model.transcribe(
        load_wav(audio_path),
        task="transcribe",          # keep original language
        language=language,          # None -> auto-detect
        beam_size=5,
        word_timestamps=True,       # useful later for word-by-word subtitles
        vad_filter=True,            # skip music / silence (common in GTA videos)
        vad_parameters={"min_silence_duration_ms": 500},
    )

    total = float(info.duration)
    segments: list[Segment] = []
    # The iterator is lazy: transcription actually happens in this loop.
    for seg in segments_iter:
        segments.append(
            Segment(
                id=len(segments),
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
                avg_logprob=round(seg.avg_logprob, 4),
                no_speech_prob=round(seg.no_speech_prob, 4),
                words=[
                    Word(round(w.start, 3), round(w.end, 3), w.word, round(w.probability, 4))
                    for w in (seg.words or [])
                ],
            )
        )
        if progress:
            progress(min(seg.end, total), total)
    if progress:
        progress(total, total)

    return Transcript(
        language=info.language,
        language_probability=round(info.language_probability, 4),
        duration=round(total, 3),
        model=model_name,
        device=device,
        compute_type=compute_type,
        segments=segments,
    )


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def save_transcript(transcript: Transcript, json_path: Path, txt_path: Path) -> None:
    """Write the transcript as structured JSON and human-readable text."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# Language: {transcript.language} "
        f"(confidence {transcript.language_probability:.0%})",
        f"# Model: {transcript.model} on {transcript.device} ({transcript.compute_type})",
        f"# Segments: {len(transcript.segments)}",
        "",
    ]
    lines += [
        f"[{_fmt_ts(s.start)} --> {_fmt_ts(s.end)}] {s.text}" for s in transcript.segments
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # Quick manual test: python -m src.transcriber
    t = transcribe(
        config.AUDIO_WAV,
        progress=lambda done, total: print(f"\r{done:7.1f}/{total:.1f}s", end=""),
    )
    print(f"\nLanguage: {t.language}, segments: {len(t.segments)}")
