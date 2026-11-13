# GTA 6 Viral Clip Maker

Turns a long GTA 6 video (used with the creator's permission) into short, fast-paced,
vertical 9:16 clips (20–60 s) with English subtitles for YouTube Shorts, TikTok and Reels.

**Status: Phase 1 — video processing** (inspect → extract audio → transcribe).

## Project structure

```
gta6-clip-maker/
  input/            gta6.mp4  (source video — never modified)
  data/             video_info.json, audio.wav, transcript.json, transcript.txt
  output/           final clips (later phases)
  src/
    config.py           paths & settings
    deps.py             FFmpeg / package discovery
    video_info.py       FFprobe inspection
    audio_extractor.py  16 kHz mono WAV extraction
    transcriber.py      Faster-Whisper transcription (original language)
    main.py             CLI pipeline
  requirements.txt
```

## Setup (Windows)

1. **FFmpeg** (ffmpeg + ffprobe):
   ```powershell
   winget install Gyan.FFmpeg
   ```
   Open a **new** terminal afterwards. If FFmpeg still isn't found, set `FFMPEG_DIR` to the
   folder containing `ffmpeg.exe`. The app also checks the winget install folder automatically.

2. **Python environment** (Python 3.10–3.13):
   ```powershell
   py -3.13 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Place the source video at `input\gta6.mp4`.

## Run

```powershell
.venv\Scripts\Activate.ps1
python -m src.main
```

Options:

| Flag | Effect |
|---|---|
| *(none)* | Asks before overwriting each existing output |
| `-y`, `--yes` | Overwrite existing outputs without asking |
| `--skip-existing` | Reuse existing outputs, run only missing steps |
| `--model medium` | Use a different Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `--language es` | Force the source language instead of auto-detecting it |

Individual steps can also be run on their own: `python -m src.video_info`,
`python -m src.audio_extractor`, `python -m src.transcriber`.

## Outputs

- `data/video_info.json` — duration, resolution, FPS, codecs, file size
- `data/audio.wav` — 16 kHz, mono, 16-bit PCM
- `data/transcript.json` — detected language, segments with start/end, text and word-level timestamps
- `data/transcript.txt` — human-readable `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`

The transcript keeps the **original spoken language**; English translation happens in a later phase.

## Notes

- Uses CUDA (float16) when available, otherwise CPU with int8. On CPU the `small` model
  processes about 4× real time (27 min of audio ≈ 7 min).
- The first run downloads the Whisper model (~480 MB for `small`) to `%USERPROFILE%\.cache\huggingface`.
- Voice-activity detection skips music and silence, so gaps in the transcript are expected.
