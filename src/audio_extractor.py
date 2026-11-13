"""Audio extraction using FFmpeg.

Produces a 16 kHz, mono, 16-bit PCM WAV - the format Whisper works with
natively. The source video is only read, never modified.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src import config
from src.deps import require_binary


class AudioExtractionError(RuntimeError):
    """Raised when FFmpeg fails to extract audio."""


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = config.AUDIO_SAMPLE_RATE,
    channels: int = config.AUDIO_CHANNELS,
) -> Path:
    """Extract the audio track of ``video_path`` to a WAV file.

    The caller is responsible for deciding whether an existing output may be
    overwritten; FFmpeg is invoked with ``-y`` once that decision is made.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if output_path.resolve() == video_path.resolve():
        raise AudioExtractionError("Output path must differ from the source video.")

    ffmpeg = require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file first so a failed/interrupted run never leaves a
    # truncated audio.wav behind.
    tmp_path = output_path.with_suffix(".tmp.wav")
    cmd = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel", "error",
        "-y",                      # overwrite the temp file only
        "-i", str(video_path),
        "-vn",                     # drop video
        "-map", "0:a:0",           # first audio track
        "-ac", str(channels),      # mono
        "-ar", str(sample_rate),   # 16 kHz
        "-c:a", "pcm_s16le",       # 16-bit PCM WAV
        str(tmp_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", check=False,
        )
    except OSError as exc:
        raise AudioExtractionError(f"Could not start ffmpeg: {exc}") from exc

    if result.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        stderr = result.stderr.strip()
        if "matches no streams" in stderr:
            raise AudioExtractionError("The video has no audio track to extract.")
        raise AudioExtractionError(
            f"ffmpeg failed (exit {result.returncode}):\n{stderr or '(no error output)'}"
        )

    tmp_path.replace(output_path)
    return output_path


if __name__ == "__main__":  # Quick manual test: python -m src.audio_extractor
    out = extract_audio(config.INPUT_VIDEO, config.AUDIO_WAV)
    print(f"Audio written to {out} ({out.stat().st_size / 1_048_576:.1f} MB)")
