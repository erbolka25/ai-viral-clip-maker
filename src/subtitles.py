"""Build .ass subtitle files for a clip from highlights.json.

* Text comes ONLY from the moment's "words" list (your manual edits win).
* 2-4 words per line, broken at punctuation and pauses.
* Word-by-word highlight: one event per word; the spoken word is yellow,
  the rest of the line white.
* Large bold font with a thick outline at ~67.5% of the screen height.
* Name capitalisation fixed with glossary.txt (boobie -> Boobie).
* Optional hook title in a box at the top for the first seconds.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src import config

PLAY_W, PLAY_H = 1080, 1920
SUB_Y = round(PLAY_H * 0.675)        # line centre: 67.5% of the height
SUB_FONT = "Arial Black"
SUB_SIZE = 100
HOOK_FONT = "Arial Black"
HOOK_SIZE = 88
HOOK_Y = 210                          # box centre from the top
HOOK_SECONDS = 3.0

MAX_WORDS, MIN_WORDS = 4, 2
MAX_CHARS = 18          # keep a line on one row at this font size
PAUSE_BREAK = 0.6       # start a new line after a pause this long (s)
HOLD_AFTER = 0.6        # keep the last line on screen this long after speech

# ASS colours are &HAABBGGRR
WHITE = "&H00FFFFFF"
YELLOW = "&H0000E5FF"
BLACK = "&H00000000"
SHADOW = "&H80000000"
HOOK_BOX = "&H00101010"   # near-black opaque box behind the hook text


def load_glossary_terms() -> list[str]:
    if not config.GLOSSARY_TXT.is_file():
        return []
    from src.transcriber import load_glossary
    return load_glossary(config.GLOSSARY_TXT)


def fix_capitalisation(word: str, terms: list[str]) -> str:
    """'boobie,' -> 'Boobie,'  and  "boobie's" -> "Boobie's" for glossary terms."""
    m = re.match(r"^(\W*)(.*?)(('s)?)(\W*)$", word, re.IGNORECASE)
    if not m:
        return word
    lead, core, poss, _, trail = m.groups()
    for term in terms:
        if core.lower() == term.lower():
            return f"{lead}{term}{poss}{trail}"
    return word


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    """Neutralise ASS override characters in user text."""
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def group_words(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split words into display lines of 2-4 words."""
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for i, w in enumerate(words):
        if cur:
            gap = w["start"] - cur[-1]["end"]
            chars = sum(len(x["word"]) + 1 for x in cur) + len(w["word"])
            ends_sentence = bool(re.search(r"[.!?,;:]$", cur[-1]["word"]))
            if (len(cur) >= MAX_WORDS or chars > MAX_CHARS
                    or (len(cur) >= MIN_WORDS and (ends_sentence or gap > PAUSE_BREAK))
                    or gap > 2 * PAUSE_BREAK):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        # Avoid a lonely last word: merge into the previous line if it fits.
        if len(cur) == 1 and groups and len(groups[-1]) < MAX_WORDS \
                and cur[0]["start"] - groups[-1][-1]["end"] <= PAUSE_BREAK:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return groups


def build_ass(moment: dict[str, Any], hook: bool = True) -> str:
    """Return the full .ass document for one moment (clip-relative times)."""
    clip_start, clip_end = float(moment["start"]), float(moment["end"])
    clip_len = clip_end - clip_start
    terms = load_glossary_terms()

    words = [
        {"start": float(w["start"]) - clip_start, "end": float(w["end"]) - clip_start,
         "word": fix_capitalisation(_escape(str(w["word"]).strip()), terms)}
        for w in moment.get("words", []) if str(w.get("word", "")).strip()
    ]
    words = [w for w in words if w["end"] > 0 and w["start"] < clip_len]

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {PLAY_W}
PlayResY: {PLAY_H}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{SUB_FONT},{SUB_SIZE},{WHITE},{WHITE},{BLACK},{SHADOW},-1,0,0,0,100,100,0,0,1,8,4,5,50,50,0,1
Style: Hook,{HOOK_FONT},{HOOK_SIZE},{WHITE},{WHITE},{HOOK_BOX},{HOOK_BOX},-1,0,0,0,100,100,0,0,3,22,0,8,70,70,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []

    groups = group_words(words)
    for gi, group in enumerate(groups):
        next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else clip_len
        for wi, w in enumerate(group):
            start = w["start"] if wi else max(0.0, w["start"])
            if wi + 1 < len(group):
                end = group[wi + 1]["start"]
            else:  # last word of the line: hold until the next line (or briefly)
                end = min(next_start, w["end"] + HOLD_AFTER)
            end = min(max(end, w["end"]), clip_len)
            if end - start < 0.05:
                continue
            parts = [
                f"{{\\c{YELLOW}}}{x['word']}{{\\c{WHITE}}}" if j == wi else x["word"]
                for j, x in enumerate(group)
            ]
            events.append(
                f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Sub,,0,0,0,,"
                f"{{\\an5\\pos({PLAY_W // 2},{SUB_Y})}}{' '.join(parts)}"
            )

    hook_text = str(moment.get("hook_title", "")).strip()
    if hook and hook_text:
        text = re.sub(r"\s+", " ", _escape(hook_text)).upper()  # hook is ALL CAPS by design
        events.append(
            f"Dialogue: 2,{_ass_time(0)},{_ass_time(min(HOOK_SECONDS, clip_len))},Hook,,0,0,0,,"
            f"{{\\an8\\pos({PLAY_W // 2},{HOOK_Y})\\fad(120,200)}}{text}"
        )

    return header + "\n".join(events) + "\n"


def write_ass(moment: dict[str, Any], path: Path, hook: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_ass(moment, hook=hook), encoding="utf-8-sig")
    return path
