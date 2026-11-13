# GTA 6 Viral Clip Maker

Turns a long GTA 6 video (used with the creator's permission) into short, fast-paced,
vertical 9:16 clips (20–60 s) with English subtitles for YouTube Shorts, TikTok and Reels.

**Pipeline:** inspect → extract audio → transcribe → LLM picks highlights → render vertical
clips with word-by-word subtitles and a hook title.

## Quick start (full pipeline)

```powershell
.venv\Scripts\Activate.ps1
python -m src.main              # 1. video info, audio, transcript (~30 min on CPU with medium)
python -m src.highlights        # 2. LLM picks 10-15 moments -> data\highlights.json
python -m src.render            # 3. top 10 clips -> output\gta6_clip_01.mp4 ... _10.mp4
```

### Fix a clip and re-render it

1. Open `output\clips_summary.md` to find the clip number.
2. Edit `data\highlights.json`: fix words in the moment's `"words"` list (the subtitle
   text), or change `hook_title`, `start` or `end`.
3. `python -m src.render --clip 3`

## Project structure

```
gta6-clip-maker/
  input/            gta6.mp4  (source video — never modified)
  data/             video_info.json, audio.wav, transcript.json, transcript.txt
  output/           gta6_clip_NN.mp4, clips_summary.md, clips_manifest.json
  src/
    config.py           paths & settings
    deps.py             FFmpeg / package discovery
    video_info.py       FFprobe inspection
    audio_extractor.py  16 kHz mono WAV extraction
    transcriber.py      Faster-Whisper transcription (original language)
    compare_models.py   model / glossary comparison on short excerpts
    main.py             Phase 1 CLI pipeline
    llm.py              provider-neutral LLM call (Groq / Anthropic)
    highlights.py       LLM highlight detection -> data/highlights.json
    subtitles.py        .ass subtitles (word highlight, hook title)
    renderer.py         FFmpeg 9:16 render of one clip
    render.py           batch / single-clip render CLI
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

## LLM provider (Phase 2)

Copy `.env.example` to `.env` and fill in your key (`.env` is git-ignored):

```
LLM_PROVIDER=groq          # or: anthropic
GROQ_API_KEY=...
```

Model names per provider are set in `src/config.py` (`LLM_MODELS`); Groq defaults to
`openai/gpt-oss-120b`. All code calls `src.llm.complete(prompt, json_mode=...)` and never
depends on the provider. Check the setup with:

```powershell
python -m src.llm      # asks the model for {"ok": true}; prints model, time, tokens, rate limits
```

The Groq free tier allows about 8K tokens per minute for `gpt-oss-120b`, input and output combined.

## Highlights (Phase 2)

`python -m src.highlights` sends a compact transcript to the LLM and saves 10-15 moments
to `data/highlights.json`. For each moment: `id, start, end, title, category, score,
reason, hook_title` and `words`, the exact words with timestamps copied from
`data/transcript.json` in code. The subtitles are built **only** from `words`, so
editing them fixes typos in the clip.

- Lines are re-split from word timestamps, because some Whisper segments include long music gaps.
- Code snaps moments to word boundaries, forces 20-60 s, drops mostly-silent
  moments and removes overlaps (> 5 s, the lower score loses).
- If the transcript is too big for one request, it is split into overlapping chunks.
- The command refuses to overwrite an existing `highlights.json` (your edits) unless you pass `-y`.
- **More moments:** `python -m src.highlights --more` asks the LLM for up to 10 NEW moments
  outside all existing ones. It applies the same validation, keeps only moments with no overlap
  and score ≥ 7 (`--min-score`), appends them and never changes existing moments. A backup is
  made first. Then run `python -m src.render --new`.
- **Sentence boundaries:** moments start at a sentence start and end at a sentence end,
  staying 20-60 s long before jump cuts. New runs do this automatically. For an existing file, run
  `python -m src.highlights --fix-boundaries`. It keeps your edited words, adds transcript
  words for any extended part, and saves a backup `data/highlights.backup-<time>.json` first.
  A sentence ends at `. ! ?`, or at a segment end or pause that comes after a word able to end a
  sentence (not "to", "the", "and"...), when the next word is capitalised.

## Rendering (Phase 3)

| Command | Effect |
|---|---|
| `python -m src.render` | Top 10 by score → `output/v3/gta6_clip_01..10.mp4` + `clips_summary.md` (asks before overwriting) |
| `python -m src.render -y` | Same, overwrite without asking |
| `python -m src.render --top 5` | Only the top 5 |
| `python -m src.render --clip N` | Re-render clip N from the current `highlights.json` |
| `python -m src.render --new` | Render moments that are not clips yet (score ≥ `MIN_NEW_SCORE`, 7) as the next clip numbers |
| `python -m src.render --test` | Top moment → `data/test/test_clip.mp4` + 3 PNG frames |

Clips go to `output/<RENDER_VERSION>/` (now `v3`; earlier versions stay in `output/v1/` and `output/v2/`).
Clip numbers are fixed at batch time (`clips_manifest.json`), so changing scores
later does not renumber clips.

Look: center crop to 1080x1920, slow zoom 1.00 → 1.08, source frame rate (23.976),
contrast 1.05 / saturation 1.1, loudness normalised to -14 LUFS, H.264 + AAC.
Subtitles: Arial Black, 2-4 words per line, current word in yellow, at ~67% height.
Names from `glossary.txt` are capitalised (boobie → Boobie). The hook title shows in a dark
bar at the top for the first 3 seconds.

### Retention editing (v2, v3)

Every effect can be switched on and off in `src/config.py`:

| Setting | Effect |
|---|---|
| `EDIT_JUMP_CUTS` | Removes silences longer than 0.5 s between words, keeping 0.15 s around words. Subtitles follow the cuts. |
| `JUMP_CUT_SMART` | v3: only removes the parts of a gap whose audio is quiet (below `JUMP_CUT_QUIET_DB`, default -42 dBFS). Loud gaps such as driving, shooting or music are kept. Bursts shorter than `JUMP_CUT_IGNORE_BURST` (0.3 s), like footsteps, don't count as action. |
| `JUMP_CUT_MIN_PIECE` | Never leaves a kept piece shorter than 1.2 s. A cut that would create one is skipped. |
| `AUDIO_CROSSFADE` | v3: 40 ms audio crossfade at every cut (0 = hard cut). Length and sync are unchanged. |
| `SUB_STABLE_LAYOUT` | v3: the full subtitle line is laid out from the start, unspoken words are invisible, and the pop is vertical only, so shown words never move. |
| `EDIT_PUNCH_ZOOMS` | Framing alternates 1.00 → 1.12 → 1.05 every 3.5 s. A change near a jump cut happens instantly on the cut. Off = slow 1.00 → 1.08 zoom. |
| `EDIT_PEAK_PUNCH` | Short (0.3 s) extra punch-in on the 3 loudest moments. |
| `EDIT_ANIMATED_SUBS` | Words pop in as they are spoken, UPPERCASE, current word yellow; names green and strong words red (max 2 per line). |
| `EDIT_HOOK_ANIMATION` | The hook title pops in, stays 2.5 s, then fades out. It always stays on one line: a long hook gets a smaller font, measured with the system font engine. |
| `EDIT_PROGRESS_BAR` | Thin yellow bar at the top that fills over the clip. |
| `EDIT_CLEAN_ENDING` | The clip ends 0.3 s after the last word. |

Cut points are snapped to whole video frames, so audio and video are cut at the same
instants. Audio and video durations match exactly in every v2 clip. No music, sound
effects or speed changes are added.

**Source subtitles:** the source video has burned-in Russian subtitles at the bottom. That
band is blurred and darkened in every clip (`SOURCE_SUBTITLE_BAND` in `src/renderer.py`;
set it to `None` for a clean source).

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
