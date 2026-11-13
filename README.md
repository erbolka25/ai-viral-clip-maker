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
    compare_models.py   model / glossary comparison on short excerpts
    main.py             CLI pipeline
  glossary.txt          names Whisper should spell correctly
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
| `--model small` | Whisper model: `tiny`, `base`, `small`, `medium` (default), `large-v3` |
| `--glossary PATH` | Use a different glossary file (default: `glossary.txt` if it exists) |
| `--no-glossary` | Transcribe without the glossary |
| `--language es` | Force the source language instead of auto-detecting it |

Individual steps can also be run on their own: `python -m src.video_info`,
`python -m src.audio_extractor`, `python -m src.transcriber`.

## Glossary

`glossary.txt` lists names and slang Whisper should spell correctly, one term per line
(`#` starts a comment). The terms are passed to faster-whisper as **hotwords**, which are
placed in front of *every* 30-second decoding window, so they stay in effect for the
whole video (an `initial_prompt` would only influence the first window).

## Comparing models

`src.compare_models` cuts short excerpts from `data/audio.wav` and transcribes them with
one or more models, writing **only** to `data/test/`. The main transcript is read as the
baseline but never modified. Timestamps are relative to the original video.

```powershell
python -m src.compare_models                                  # excerpts A=0:180, B=1530:1620; small + medium
python -m src.compare_models --models medium large-v3         # other models
python -m src.compare_models --excerpt C=600:720 --models medium   # custom range, in seconds
python -m src.compare_models --excerpt A=0:180 --no-glossary  # without glossary
```

Results:
- `data/test/<excerpt>_<model>_glossary.txt` / `.json`, or `_noglossary` for runs without the glossary
- `data/test/comparison_<excerpt>.md` (`comparison_<excerpt>_noglossary.md`) — side-by-side table with name counts
- `data/test/timings.json` — processing time per run

## Outputs

- `data/video_info.json` — duration, resolution, FPS, codecs, file size
- `data/audio.wav` — 16 kHz, mono, 16-bit PCM
- `data/transcript.json` — detected language, segments with start/end, text and word-level timestamps
- `data/transcript.txt` — human-readable `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`

The transcript keeps the **original spoken language**; English translation happens in a later phase.

## Notes

- Uses CUDA (float16) when available, otherwise CPU with int8. On CPU, `small` runs at
  about 4–7× real time (27-minute video ≈ 5 min); `medium` is about 3–5× slower than that.
- The first run downloads the Whisper model (`small` ~0.5 GB, `medium` ~1.5 GB) to
  `%USERPROFILE%\.cache\huggingface`.
- Voice-activity detection skips music and silence, so gaps in the transcript are expected.
