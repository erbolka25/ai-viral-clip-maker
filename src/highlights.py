"""Highlight detection: transcript -> data/highlights.json.

1. A compact transcript (whole-second timestamps, no word data) is sent to the
   LLM, which proposes 10-15 viral moments.
2. If the request would not fit Groq's per-minute token budget, the transcript
   is split into overlapping chunks (waiting between requests as needed) and
   the results are merged and de-duplicated.
3. CODE (not the LLM) then snaps every moment to word boundaries, enforces
   20-60 s durations and removes big overlaps, and attaches the exact words
   with timestamps from data/transcript.json as a "words" field.

The "words" field is the single source of subtitle text: fix transcript typos
by editing highlights.json, then re-render.

    python -m src.highlights        # refuses to overwrite an edited file
    python -m src.highlights -y     # overwrite data/highlights.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

from src import config
from src.llm import LLMError, LLMResult, complete_with_meta

CATEGORIES = ("funny", "surprising", "exciting", "interesting_facts", "strong_reactions")
MIN_LEN, MAX_LEN = 20.0, 60.0
TARGET_MIN, TARGET_MAX = 10, 15
MAX_OVERLAP = 5.0  # seconds two moments may share before the weaker one is dropped
PAD_BEFORE, PAD_AFTER = 0.3, 0.5  # breathing room around the first/last word
LINE_SPLIT_GAP = 1.5      # new transcript line after this much silence
MIN_WORDS_PER_SEC = 0.5   # below this a moment is mostly silence -> rejected
MAX_SILENT_GAP = 15.0     # longest silence allowed inside a moment

# Token budget per request on the Groq free tier (input + reasoning + output).
TOKENS_PER_MINUTE = 8000
OUTPUT_RESERVE = 3500  # reasoning + JSON answer
CHUNK_OVERLAP_SECONDS = 60

SYSTEM_PROMPT = """You are a senior short-form video editor. You pick moments from \
long GTA 6 videos that will go viral as vertical clips on YouTube Shorts, TikTok and \
Instagram Reels for English-speaking viewers. You value: a strong opening line, \
a clear payoff (joke, twist, action peak or big reaction), and moments that make \
sense without the rest of the video."""

USER_PROMPT = """Below is a transcript of a GTA 6 video. Each line is \
"[start-end] text" with times in whole seconds from the start of the video. \
Silence and music are not transcribed, so there are gaps.

Find {n_min}-{n_max} of the best moments for viral short clips.

Rules for each moment:
- 20 to 60 seconds long (end - start). Ideal: 25-45 s.
- Start at the beginning of a transcript line (the setup), end at the end of a \
line, just after the payoff. Never cut a sentence in half.
- Moments must not overlap each other.
- Spread them across the whole video; prefer the strongest ones.
- category: one of {categories}.
- score: integer 1-10 (10 = will definitely go viral).
- title: short descriptive English title.
- reason: one sentence on why it works as a short.
- hook_title: catchy English text shown in the first 3 seconds, MAX 5 WORDS, \
no hashtags, no emojis.
{extra}
Return ONLY a JSON object in exactly this shape:
{{"moments": [{{"id": 1, "start": 0, "end": 30, "title": "...", "category": "funny", \
"score": 8, "reason": "...", "hook_title": "..."}}]}}

TRANSCRIPT:
{transcript}"""


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def load_transcript(path: Path = config.TRANSCRIPT_JSON) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run: python -m src.main")
    return json.loads(path.read_text(encoding="utf-8"))


def speech_units(segments: list[dict[str, Any]], max_gap: float = LINE_SPLIT_GAP) -> list[dict[str, Any]]:
    """Re-split Whisper segments into lines of continuous speech.

    Whisper segment times can include long music/silence (e.g. one segment
    spans 917-975 s but holds 7 words), which misleads the LLM into thinking
    there is a minute of dialogue. Word timestamps are accurate, so a new
    line starts wherever words are more than ``max_gap`` seconds apart.
    """
    units: list[dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            if seg["text"].strip():
                units.append({"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()})
            continue
        cur: dict[str, Any] | None = None
        for w in words:
            piece = w["word"]
            if cur and w["start"] - cur["end"] <= max_gap:
                cur["text"] += piece
                cur["end"] = w["end"]
            else:
                if cur and cur["text"].strip():
                    units.append(cur)
                cur = {"start": w["start"], "end": w["end"], "text": piece}
        if cur and cur["text"].strip():
            units.append(cur)
    for u in units:
        u["text"] = re.sub(r"\s+", " ", u["text"]).strip()
    return units


def compact_lines(units: list[dict[str, Any]]) -> list[str]:
    """One line per unit: '[start-end] text' with whole seconds."""
    return [
        f"[{int(u['start'])}-{int(math.ceil(u['end']))}] {u['text']}"
        for u in units if u["text"]
    ]


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (~3.5 characters per token)."""
    return int(len(text) / 3.5) + 1


def make_chunks(segments: list[dict[str, Any]], max_tokens: int) -> list[list[dict[str, Any]]]:
    """Split segments into chunks under ``max_tokens``, overlapping by
    CHUNK_OVERLAP_SECONDS so moments on a boundary are not lost."""
    chunks: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(segments):
        chunk: list[dict[str, Any]] = []
        used = 0
        j = i
        while j < len(segments):
            cost = estimate_tokens(compact_lines([segments[j]])[0] if segments[j]["text"].strip() else "")
            if chunk and used + cost > max_tokens:
                break
            chunk.append(segments[j])
            used += cost
            j += 1
        chunks.append(chunk)
        if j >= len(segments):
            break
        # Next chunk starts CHUNK_OVERLAP_SECONDS before this chunk's end.
        overlap_from = chunk[-1]["end"] - CHUNK_OVERLAP_SECONDS
        k = j
        while k - 1 > i and segments[k - 1]["start"] >= overlap_from:
            k -= 1
        i = max(k, i + 1)
    return chunks


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _parse_duration(value: str | None) -> float:
    """Parse Groq reset headers like '7.66s', '1m26.4s', '120ms'."""
    if not value:
        return 0.0
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", value):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


class RateLimiter:
    """Waits before a request when the last response says the token budget
    is too low for it."""

    def __init__(self) -> None:
        self.remaining: int | None = None
        self.reset_seconds = 0.0

    def update(self, result: LLMResult) -> None:
        rl = result.rate_limits
        try:
            self.remaining = int(rl.get("x-ratelimit-remaining-tokens", ""))
        except ValueError:
            self.remaining = None
        self.reset_seconds = _parse_duration(rl.get("x-ratelimit-reset-tokens"))

    def wait_for(self, needed_tokens: int) -> None:
        if self.remaining is not None and self.remaining < needed_tokens and self.reset_seconds:
            wait = self.reset_seconds + 1
            print(f"  waiting {wait:.0f}s for the token budget to refill...")
            time.sleep(wait)


def ask_llm(lines: list[str], n_min: int, n_max: int, extra: str,
            limiter: RateLimiter) -> tuple[list[dict[str, Any]], LLMResult]:
    prompt = USER_PROMPT.format(
        n_min=n_min, n_max=n_max, categories=", ".join(CATEGORIES),
        extra=extra, transcript="\n".join(lines),
    )
    needed = estimate_tokens(SYSTEM_PROMPT + prompt) + OUTPUT_RESERVE
    limiter.wait_for(needed)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = complete_with_meta(prompt, system=SYSTEM_PROMPT, json_mode=True,
                                        max_tokens=OUTPUT_RESERVE + 2000)
            limiter.update(result)
            data = json.loads(result.text)
            moments = data.get("moments") if isinstance(data, dict) else None
            if not isinstance(moments, list):
                raise ValueError("response has no 'moments' list")
            return moments, result
        except (LLMError, ValueError) as exc:  # JSONDecodeError is a ValueError
            last_error = exc
            if "too large" in str(exc).lower() or "413" in str(exc):
                raise  # caller will chunk
            wait = 15 * (attempt + 1)
            print(f"  attempt {attempt + 1} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise LLMError(f"LLM failed 3 times: {last_error}")


# ---------------------------------------------------------------------------
# Validation / post-processing (pure code)
# ---------------------------------------------------------------------------

def _words_in(all_words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    """Words whose midpoint lies inside [start, end], as clean editable words.

    Whisper emits sub-word pieces: a piece WITHOUT a leading space continues
    the previous word (e.g. " y" + "'all" -> "y'all"), so those are merged.
    """
    merged: list[dict[str, Any]] = []
    for w in all_words:
        if not start <= (w["start"] + w["end"]) / 2 <= end:
            continue
        text = w["word"]
        if merged and text and not text[0].isspace():
            merged[-1]["word"] += text.strip()
            merged[-1]["end"] = round(w["end"], 2)
        elif text.strip():
            merged.append({"start": round(w["start"], 2), "end": round(w["end"], 2),
                           "word": text.strip()})
    return merged


def snap_and_fix(m: dict[str, Any], all_words: list[dict[str, Any]],
                 duration: float) -> tuple[float, float] | None:
    """Snap a moment to word boundaries and force 20-60 s. None if unusable."""
    try:
        start, end = float(m["start"]), float(m["end"])
    except (KeyError, TypeError, ValueError):
        return None
    start, end = max(0.0, start), min(duration, end)
    if end <= start:
        return None

    # Snap to the first/last spoken word inside the LLM's range (+ small padding).
    inside = [w for w in all_words if w["end"] > start - 0.5 and w["start"] < end + 0.5]
    if not inside:
        return None
    start = max(0.0, inside[0]["start"] - PAD_BEFORE)
    end = min(duration, inside[-1]["end"] + PAD_AFTER)

    if end - start > MAX_LEN:
        # Trim to the last word that ends within 60 s.
        limit = start + MAX_LEN - PAD_AFTER
        kept = [w for w in inside if w["end"] <= limit]
        if not kept:
            return None
        end = kept[-1]["end"] + PAD_AFTER
    if end - start < MIN_LEN:
        # Pad both sides evenly (game audio keeps playing), clamped to the video.
        missing = MIN_LEN - (end - start)
        start = max(0.0, start - missing / 2)
        end = min(duration, start + MIN_LEN)
        start = max(0.0, end - MIN_LEN)
    if not (MIN_LEN - 0.01 <= end - start <= MAX_LEN + 0.01):
        return None
    return round(start, 2), round(end, 2)


def clean_text(value: Any, max_words: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip('"')
    if max_words:
        text = " ".join(text.split()[:max_words])
    return text


# ---------------------------------------------------------------------------
# Sentence-aware boundaries
# ---------------------------------------------------------------------------

# Words that (almost) never end an English sentence.
NON_FINAL_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "because", "if", "when", "while", "to",
    "of", "for", "with", "at", "in", "on", "from", "by", "as", "than", "that", "which",
    "who", "my", "your", "his", "her", "our", "their", "its", "this", "these", "those",
    "i", "i'm", "you're", "we're", "they're", "he's", "she's", "it's", "gotta", "gonna",
    "wanna", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have",
    "has", "had", "can", "could", "will", "would", "should", "just", "about", "into",
    "like", "what's", "there's", "let's", "y'all", "ain't", "don't", "can't", "won't",
    "some", "any", "every", "very", "really", "not",
}


def sentence_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """All transcript words (sub-word pieces merged) with sentence flags.

    A word ends a sentence if it ends with . ! ? , is the last word of a
    Whisper segment, or is followed by a pause > config.SENTENCE_GAP.
    """
    words: list[dict[str, Any]] = []
    for seg in transcript["segments"]:
        ws = _words_in(seg.get("words") or [], -1e9, 1e9)
        if ws:
            ws[-1]["seg_end"] = True
        words += ws
    words.sort(key=lambda w: w["start"])
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else None
        punctuated = bool(re.search(r"[.!?][\"')\]]*$", w["word"]))
        # Whisper often ends a segment mid-sentence ("know when to | get off"),
        # so a segment end / pause only counts after a word that can end one.
        # Whisper capitalises a new sentence, so the next word must start upper-case.
        can_end = re.sub(r"[^\w']", "", w["word"]).lower() not in NON_FINAL_WORDS
        next_upper = nxt is None or words[i + 1]["word"][:1].isupper()
        boundary = bool(w.get("seg_end")) or nxt is None or nxt - w["end"] > config.SENTENCE_GAP
        w["sent_end"] = punctuated or (boundary and can_end and next_upper) or nxt is None
    for i, w in enumerate(words):
        w["sent_start"] = i == 0 or words[i - 1]["sent_end"]
    return words


def snap_to_sentences(start: float, end: float, words: list[dict[str, Any]],
                      duration: float) -> tuple[float, float]:
    """Move start to a sentence start and end to a sentence end, keeping the
    moment 20-60 s if possible and moving each boundary by at most
    SENTENCE_MAX_SHIFT s when there is a choice."""
    idx = [i for i, w in enumerate(words) if start <= (w["start"] + w["end"]) / 2 <= end]
    if not idx:
        return start, end
    f, l = idx[0], idx[-1]
    lo, hi, shift = config.MOMENT_MIN_SECONDS, config.MOMENT_MAX_SECONDS, config.SENTENCE_MAX_SHIFT

    def span(a: int, b: int) -> float:
        return (words[b]["end"] + PAD_AFTER) - (words[a]["start"] - PAD_BEFORE)

    def prev_start(i: int) -> int | None:
        return next((k for k in range(i, -1, -1) if words[k]["sent_start"]), None)

    def next_end(i: int) -> int | None:
        return next((k for k in range(i, len(words)) if words[k]["sent_end"]), None)

    # Start: back to the sentence start (preferred) or forward to the next one.
    if not words[f]["sent_start"]:
        back = prev_start(f)
        fwd = next((k for k in range(f + 1, l + 1) if words[k]["sent_start"]), None)
        if back is not None and words[f]["start"] - words[back]["start"] <= shift and span(back, l) <= hi:
            f = back
        elif fwd is not None and span(fwd, l) >= lo:
            f = fwd
        elif back is not None and span(back, l) <= hi:
            f = back
    # End: forward to the sentence end (preferred) or back to the previous one.
    if not words[l]["sent_end"]:
        fwd = next_end(l)
        back = next((k for k in range(l - 1, f - 1, -1) if words[k]["sent_end"]), None)
        if fwd is not None and words[fwd]["end"] - words[l]["end"] <= shift and span(f, fwd) <= hi:
            l = fwd
        elif back is not None and span(f, back) >= lo:
            l = back
        elif fwd is not None and span(f, fwd) <= hi:
            l = fwd
    # Keep 20-60 s: add whole sentences (after, then before) or drop from the end.
    while span(f, l) < lo:
        nxt = next_end(l + 1) if l + 1 < len(words) else None
        if nxt is not None and span(f, nxt) <= hi:
            l = nxt
            continue
        prv = prev_start(f - 1) if f > 0 else None
        if prv is not None and span(prv, l) <= hi:
            f = prv
            continue
        break
    while span(f, l) > hi:
        back = next((k for k in range(l - 1, f - 1, -1) if words[k]["sent_end"]), None)
        if back is None:
            break
        l = back

    # Pad around the boundary words, but never into the neighbouring words
    # (and never past the boundary word itself when words touch).
    new_start = max(0.0, words[f]["start"] - PAD_BEFORE)
    if f > 0:
        new_start = max(new_start, min(words[f - 1]["end"] + 0.05, words[f]["start"]))
    new_end = min(duration, words[l]["end"] + PAD_AFTER)
    if l + 1 < len(words):
        new_end = min(new_end, max(words[l + 1]["start"] - 0.05, words[l]["end"]))
    return round(new_start, 2), round(new_end, 2)


def rebuild_words(moment: dict[str, Any], new_start: float, new_end: float,
                  words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Words for the new range: the moment's own (possibly hand-edited) words
    where the ranges overlap, transcript words for the newly added parts."""
    old_s, old_e = float(moment["start"]), float(moment["end"])
    mid = lambda w: (float(w["start"]) + float(w["end"])) / 2  # noqa: E731
    kept = [w for w in moment.get("words", []) if new_start <= mid(w) <= new_end]
    added = [{"start": w["start"], "end": w["end"], "word": w["word"]}
             for w in words if new_start <= mid(w) <= new_end and not old_s <= mid(w) <= old_e]
    return sorted(kept + added, key=lambda w: float(w["start"]))


def fix_boundaries(data: dict[str, Any], transcript: dict[str, Any]) -> list[str]:
    """Apply sentence-aware boundaries to every moment in place; return a report."""
    words = sentence_words(transcript)
    duration = float(transcript["duration"])
    report = []
    for m in data["moments"]:
        s, e = snap_to_sentences(float(m["start"]), float(m["end"]), words, duration)
        if (s, e) == (float(m["start"]), float(m["end"])):
            report.append(f"#{m['id']:>2}: unchanged")
            continue
        old = (float(m["start"]), float(m["end"]))
        m["words"] = rebuild_words(m, s, e, words)
        m["start"], m["end"], m["duration"] = s, e, round(e - s, 2)
        last = " ".join(w["word"] for w in m["words"][-4:])
        report.append(f"#{m['id']:>2}: {old[0]:.1f}-{old[1]:.1f} ({old[1] - old[0]:.1f}s) -> "
                      f"{s:.1f}-{e:.1f} ({e - s:.1f}s)  ends: ...{last}")
    return report


def postprocess(raw_moments: list[dict[str, Any]], transcript: dict[str, Any]) -> list[dict[str, Any]]:
    duration = float(transcript["duration"])
    all_words = [w for s in transcript["segments"] for w in s.get("words", [])]
    sentences = sentence_words(transcript)

    candidates: list[dict[str, Any]] = []
    for m in raw_moments:
        span = snap_and_fix(m, all_words, duration)
        if span is None:
            print(f"  dropped invalid moment: {m.get('title', m)!r}")
            continue
        span = snap_to_sentences(span[0], span[1], sentences, duration)
        # Reject moments that are mostly silence/music: the LLM only "saw"
        # the dialogue, so its description would not match the clip.
        spoken = [w for w in all_words if span[0] <= w["start"] and w["end"] <= span[1]]
        gaps = [b["start"] - a["end"] for a, b in zip(spoken, spoken[1:])]
        density = len(spoken) / (span[1] - span[0])
        if density < MIN_WORDS_PER_SEC or (gaps and max(gaps) > MAX_SILENT_GAP):
            print(f"  dropped mostly-silent moment: {m.get('title')!r} "
                  f"({density:.2f} words/s, longest gap {max(gaps, default=0):.0f}s)")
            continue
        category = str(m.get("category", "")).strip().lower().replace(" ", "_")
        if category not in CATEGORIES:
            category = "exciting"
        try:
            score = int(round(float(m.get("score", 5))))
        except (TypeError, ValueError):
            score = 5
        candidates.append({
            "start": span[0], "end": span[1],
            "title": clean_text(m.get("title")),
            "category": category,
            "score": max(1, min(10, score)),
            "reason": clean_text(m.get("reason")),
            "hook_title": clean_text(m.get("hook_title"), max_words=5),
        })

    # De-duplicate / remove big overlaps: keep the higher score.
    kept: list[dict[str, Any]] = []
    for c in sorted(candidates, key=lambda c: (-c["score"], c["start"])):
        clash = any(min(c["end"], k["end"]) - max(c["start"], k["start"]) > MAX_OVERLAP for k in kept)
        if clash:
            print(f"  dropped overlapping moment: {c['title']!r}")
            continue
        kept.append(c)

    kept.sort(key=lambda c: c["start"])
    out = []
    for i, c in enumerate(kept, 1):
        words = _words_in(all_words, c["start"], c["end"])
        out.append({
            "id": i,
            "start": c["start"], "end": c["end"],
            "duration": round(c["end"] - c["start"], 2),
            "title": c["title"], "category": c["category"], "score": c["score"],
            "reason": c["reason"], "hook_title": c["hook_title"],
            "words": words,
        })
    return out


def validate(moments: list[dict[str, Any]], duration: float) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems = []
    for m in moments:
        d = m["end"] - m["start"]
        if not (0 <= m["start"] < m["end"] <= duration):
            problems.append(f"#{m['id']}: times outside video")
        if not (MIN_LEN - 0.01 <= d <= MAX_LEN + 0.01):
            problems.append(f"#{m['id']}: duration {d:.1f}s not in 20-60s")
        if m["category"] not in CATEGORIES:
            problems.append(f"#{m['id']}: bad category {m['category']}")
        if len(m["hook_title"].split()) > 5 or not m["hook_title"]:
            problems.append(f"#{m['id']}: hook_title must be 1-5 words")
        if not m["words"]:
            problems.append(f"#{m['id']}: no words")
    ordered = sorted(moments, key=lambda m: m["start"])
    for a, b in zip(ordered, ordered[1:]):
        if a["end"] - b["start"] > MAX_OVERLAP:
            problems.append(f"#{a['id']} and #{b['id']} overlap by {a['end'] - b['start']:.1f}s")
    return problems


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def detect(transcript: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments = speech_units([s for s in transcript["segments"] if s["text"].strip()])
    lines = compact_lines(segments)
    budget = TOKENS_PER_MINUTE - OUTPUT_RESERVE - estimate_tokens(SYSTEM_PROMPT + USER_PROMPT)
    limiter = RateLimiter()
    stats: dict[str, Any] = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

    def run(seg_chunks: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        total = sum(len(c) for c in seg_chunks)
        for n, chunk in enumerate(seg_chunks, 1):
            share = len(chunk) / total
            n_min = max(3, round(TARGET_MIN * share)) if len(seg_chunks) > 1 else TARGET_MIN
            n_max = max(n_min + 1, round(TARGET_MAX * share) + 1) if len(seg_chunks) > 1 else TARGET_MAX
            print(f"  request {n}/{len(seg_chunks)}: {int(chunk[0]['start'])}-"
                  f"{int(chunk[-1]['end'])}s, asking for {n_min}-{n_max} moments")
            moments, res = ask_llm(compact_lines(chunk), n_min, n_max, "", limiter)
            stats["requests"] += 1
            stats["input_tokens"] += res.input_tokens or 0
            stats["output_tokens"] += res.output_tokens or 0
            stats["seconds"] += res.elapsed_seconds
            found += moments
        return found

    estimate = estimate_tokens("\n".join(lines))
    print(f"Transcript: {len(lines)} lines, ~{estimate} tokens (budget per request ~{budget})")
    try:
        if estimate <= budget:
            raw = run([segments])
        else:
            raise LLMError("too large for one request")
    except LLMError as exc:
        if "too large" not in str(exc).lower() and "413" not in str(exc):
            raise
        chunks = make_chunks(segments, max(1000, budget))
        print(f"Splitting into {len(chunks)} overlapping chunks")
        raw = run(chunks)

    moments = postprocess(raw, transcript)

    # Too few after validation? Ask once more for moments outside the used ranges.
    if len(moments) < TARGET_MIN:
        used = ", ".join(f"{int(m['start'])}-{int(m['end'])}" for m in moments)
        extra = (f"\nThese ranges are already taken, do NOT overlap them: {used}. "
                 f"Find {TARGET_MIN - len(moments) + 2} additional moments.\n")
        print(f"Only {len(moments)} valid moments; asking for more")
        more, res = ask_llm(lines, TARGET_MIN - len(moments), TARGET_MAX - len(moments) + 2,
                            extra, limiter)
        stats["requests"] += 1
        stats["input_tokens"] += res.input_tokens or 0
        stats["output_tokens"] += res.output_tokens or 0
        stats["seconds"] += res.elapsed_seconds
        moments = postprocess(raw + more, transcript)

    stats["raw_moments"] = raw
    return moments, stats


def _overlap(a: dict[str, Any], b: tuple[float, float]) -> float:
    return min(float(a["end"]), b[1]) - max(float(a["start"]), b[0])


def find_more(data: dict[str, Any], transcript: dict[str, Any], max_new: int,
              min_score: int) -> dict[str, Any]:
    """Ask the LLM for NEW moments outside all existing ones.

    Existing moments are never changed. New ones go through the same
    validation as a normal run (20-60 s, sentence boundaries, words attached
    in code); only those that overlap nothing and score >= min_score are kept.
    """
    taken = sorted((float(m["start"]), float(m["end"])) for m in data["moments"])
    used = ", ".join(f"{int(a)}-{int(math.ceil(b))}" for a, b in taken)
    extra = (
        f"\nThese time ranges are ALREADY USED: {used}.\n"
        "New moments must NOT overlap any of them, not even partly.\n"
        "Quality over quantity: only return moments you would honestly score 7 or "
        "higher. Returning fewer moments, even zero, is fine.\n"
    )
    units = speech_units([s for s in transcript["segments"] if s["text"].strip()])
    raw, res = ask_llm(compact_lines(units), 1, max_new, extra, RateLimiter())
    candidates = postprocess(raw, transcript)

    kept: list[dict[str, Any]] = []
    rejected: list[str] = []
    for m in sorted(candidates, key=lambda m: (-m["score"], m["start"])):
        blocked = taken + [(k["start"], k["end"]) for k in kept]
        worst = max((_overlap(m, r) for r in blocked), default=0.0)
        if worst > 0:
            rejected.append(f"[{m['score']}] {m['hook_title']!r}: overlaps an existing moment by {worst:.1f}s")
        elif m["score"] < min_score:
            rejected.append(f"[{m['score']}] {m['hook_title']!r}: score below {min_score}")
        else:
            kept.append(m)

    next_id = max((int(m["id"]) for m in data["moments"]), default=0) + 1
    for m in sorted(kept, key=lambda m: m["start"]):
        m["id"] = next_id
        next_id += 1
    return {"raw": raw, "validated": candidates, "kept": kept, "rejected": rejected,
            "tokens": (res.input_tokens, res.output_tokens), "seconds": res.elapsed_seconds}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.highlights", description=__doc__.splitlines()[0])
    p.add_argument("-y", "--yes", action="store_true",
                   help="overwrite data/highlights.json (it may contain your manual edits)")
    p.add_argument("--fix-boundaries", action="store_true",
                   help="move existing moments to sentence boundaries (keeps your word edits; "
                        "backs up highlights.json first)")
    p.add_argument("--more", action="store_true",
                   help="add NEW moments that do not overlap existing ones (existing moments "
                        "are not changed; backs up highlights.json first)")
    p.add_argument("--max-new", type=int, default=10, help="--more: ask for up to N moments")
    p.add_argument("--min-score", type=int, default=config.MIN_NEW_SCORE,
                   help=f"--more: keep only scores >= this (default {config.MIN_NEW_SCORE})")
    args = p.parse_args(argv)

    out = config.HIGHLIGHTS_JSON
    if args.more:
        if not out.is_file():
            print(f"{out} not found - run: python -m src.highlights")
            return 1
        data = json.loads(out.read_text(encoding="utf-8"))
        backup = out.with_name(f"highlights.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
        backup.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        transcript = load_transcript()
        try:
            result = find_more(data, transcript, args.max_new, args.min_score)
        except LLMError as exc:
            print(f"Error: {exc}")
            return 1
        (config.DATA_DIR / "highlights_raw_more.json").write_text(
            json.dumps(result["raw"], indent=2, ensure_ascii=False), encoding="utf-8")
        data["moments"] += result["kept"]
        problems = validate(data["moments"], float(transcript["duration"]))
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"LLM proposed {len(result['raw'])}, {len(result['validated'])} passed validation, "
              f"{len(result['kept'])} kept (score >= {args.min_score}, no overlap). "
              f"Tokens in/out {result['tokens'][0]}/{result['tokens'][1]}, {result['seconds']:.1f}s")
        for m in result["kept"]:
            print(f"  + #{m['id']} [{m['score']}] {m['start']:.1f}-{m['end']:.1f} ({m['duration']:.1f}s) "
                  f"{m['category']}: {m['hook_title']!r}")
        for line in result["rejected"]:
            print(f"  - {line}")
        print(f"Saved {out} (backup: {backup.name})")
        if problems:
            print("Validation problems:\n  " + "\n  ".join(problems))
            return 2
        return 0

    if args.fix_boundaries:
        if not out.is_file():
            print(f"{out} not found - run: python -m src.highlights")
            return 1
        data = json.loads(out.read_text(encoding="utf-8"))
        backup = out.with_name(f"highlights.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
        backup.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        transcript = load_transcript()
        for line in fix_boundaries(data, transcript):
            print(line)
        problems = validate(data["moments"], float(transcript["duration"]))
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {out} (backup: {backup.name})")
        if problems:
            print("Validation problems:\n  " + "\n  ".join(problems))
            return 2
        print("Validation: OK")
        return 0

    if out.exists() and not args.yes:
        print(f"{out} already exists and may contain your manual edits.\n"
              "Re-run with -y to overwrite it.")
        return 1

    try:
        transcript = load_transcript()
        moments, stats = detect(transcript)
    except (LLMError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1

    problems = validate(moments, float(transcript["duration"]))
    payload = {
        "source": config.INPUT_VIDEO.name,
        "video_duration": transcript["duration"],
        "model": stats.get("model") or config.LLM_MODELS.get("groq"),
        "note": "Edit 'words' to fix subtitle typos, then re-render with: python -m src.render --clip N",
        "moments": moments,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.DATA_DIR / "highlights_raw.json").write_text(
        json.dumps(stats.pop("raw_moments"), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(moments)} moments saved to {out}")
    print(f"LLM: {stats['requests']} request(s), {stats['input_tokens']} in / "
          f"{stats['output_tokens']} out tokens, {stats['seconds']:.1f}s")
    for m in sorted(moments, key=lambda m: -m["score"]):
        print(f"  #{m['id']:>2} [{m['score']}] {m['start']:7.1f}-{m['end']:7.1f} "
              f"({m['duration']:4.1f}s) {m['category']:<17} {m['hook_title']!r}")
    if problems:
        print("Validation problems:\n  " + "\n  ".join(problems))
        return 2
    print("Validation: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
