"""Build .ass overlays for a clip: subtitles, hook title, progress bar.

* Text comes ONLY from the moment's "words" list in highlights.json.
* Times are mapped through the clip's EditPlan (jump cuts move words earlier).
* 2-4 words per line, broken at punctuation and pauses.
* Animated mode (config.EDIT_ANIMATED_SUBS): words appear as they are spoken
  and pop (quick scale up and back); the current word is yellow; text is
  UPPERCASE; 1-2 key words per line are coloured (names green, strong words red).
  The line shows the words spoken so far, centered, and grows word by word.
* Name capitalisation is fixed with glossary.txt (boobie -> Boobie).
* Hook title box at the top (pops in, stays HOOK_DURATION s, fades out).
* Progress bar that fills over the clip duration.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src import config
from src.editing import EditPlan, clip_words

PLAY_W, PLAY_H = 1080, 1920
SUB_Y = round(PLAY_H * 0.675)        # line centre: 67.5% of the height
SUB_FONT = "Arial Black"
SUB_SIZE = 100
HOOK_FONT = "Arial Black"
HOOK_SIZE = 88
HOOK_Y = 210                          # box top from the top edge
HOOK_SECONDS_STATIC = 3.0             # when the hook animation is off

MAX_WORDS, MIN_WORDS = 4, 2
MAX_CHARS = 18          # keep a line on one row at this font size
PAUSE_BREAK = 0.6       # start a new line after a pause this long (s)
HOLD_AFTER = 0.6        # keep the last line on screen this long after speech

# ASS colours are &HAABBGGRR
WHITE = "&H00FFFFFF"
YELLOW = "&H0000E5FF"
GREEN = "&H0014FF39"
RED = "&H003C3CFF"
BLACK = "&H00000000"
SHADOW = "&H80000000"
HOOK_BOX = "&H00101010"   # near-black opaque box behind the hook text

# Words that get the "strong word" colour (compared in upper case, no punctuation).
STRONG_WORDS = {
    "FUCK", "FUCKING", "FUCKED", "SHIT", "DAMN", "HELL", "DEAD", "DIED", "DIE", "DEATH",
    "KILL", "KILLED", "GUN", "GUNS", "MONEY", "CASH", "COP", "COPS", "POLICE", "CRAZY",
    "INSANE", "HOLY", "JUMP", "HEIST", "BANK", "ROBBERY", "ROB", "STEAL", "STOLE",
    "STOLEN", "CRASH", "CRACKED", "BLOOD", "SHOOT", "SHOT", "RUN", "NEVER", "WOW",
    "GOD", "PARTY", "DRUNK", "BUZZED", "PICKUP", "DROP", "GLORY",
}

# Frequent words Whisper capitalises at (unpunctuated) sentence starts; never names.
COMMON_WORDS = {
    "A", "ALL", "ALRIGHT", "AN", "AND", "ANY", "ARE", "AS", "AT", "BE", "BECAUSE", "BUT",
    "BRO", "BY", "CAN", "COME", "DID", "DO", "DON'T", "EVEN", "FOR", "FROM", "GET", "GO",
    "GOOD", "GOT", "HAVE", "HE", "HE'S", "HER", "HERE", "HEY", "HIM", "HIS", "HOLD", "HOW",
    "IF", "IN", "IS", "IT", "IT'S", "JUST", "KNOW", "LET'S", "LIKE", "LOOK", "MAN", "ME",
    "MY", "NAH", "NO", "NOT", "NOW", "OF", "OH", "ON", "ONE", "OR", "OUR", "OUT", "PLEASE",
    "RIGHT", "SEE", "SHE", "SO", "SOME", "THAT", "THAT'S", "THE", "THEN", "THERE", "THEY",
    "THIS", "TO", "UH", "UM", "UP", "WAIT", "WANT", "WAS", "WE", "WE'RE", "WELL", "WHAT",
    "WHAT'S", "WHEN", "WHERE", "WHO", "WHY", "WITH", "WOULD", "YEAH", "YEP", "YES", "YO",
    "YOU", "YOU'RE", "YOUR", "Y'ALL",
}

# Pop animation for the word being spoken (ms).
POP = r"\fscx72\fscy72\t(0,70,\fscx118\fscy118)\t(70,150,\fscx100\fscy100)"


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
    for w in words:
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


def _core(word: str) -> str:
    return re.sub(r"[^\w']", "", word).upper().removesuffix("'S")


def mark_keywords(words: list[dict[str, Any]], terms: list[str]) -> None:
    """Set w['kind'] = 'name' | 'strong' | None.

    Names: glossary terms, plus capitalised words that do not start a sentence,
    are not common words and do not look like verbs/adverbs (-ed/-ing/-ly) -
    Whisper capitalises sentence starts even without a period before them."""
    glossary = {t.upper() for t in terms}
    sentence_start = True
    for w in words:
        core = _core(w["word"])
        raw = re.sub(r"^\W+", "", w["word"])
        name_like = (raw[:1].isupper() and not sentence_start and len(core) >= 3
                     and core not in COMMON_WORDS and not core.startswith("I'")
                     and not re.search(r"(ED|ING|LY)$", core))
        if core in glossary or name_like:
            w["kind"] = "name"
        elif core in STRONG_WORDS:
            w["kind"] = "strong"
        else:
            w["kind"] = None
        sentence_start = bool(re.search(r"[.!?]$", w["word"]))


def _limit_keywords(group: list[dict[str, Any]], max_per_line: int = 2) -> list[str | None]:
    """Colour for each word of a line (at most ``max_per_line`` coloured)."""
    colours: list[str | None] = [None] * len(group)
    ranked = sorted(
        (i for i, w in enumerate(group) if w.get("kind")),
        key=lambda i: (group[i]["kind"] != "name", -len(group[i]["word"])),
    )
    for i in ranked[:max_per_line]:
        colours[i] = GREEN if group[i]["kind"] == "name" else RED
    return colours


def _header() -> str:
    return f"""[Script Info]
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
Style: Bar,Arial,10,{WHITE},{WHITE},{BLACK},{BLACK},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _subtitle_events(words: list[dict[str, Any]], clip_len: float) -> list[str]:
    animated = config.EDIT_ANIMATED_SUBS
    events: list[str] = []
    groups = group_words(words)
    for gi, group in enumerate(groups):
        next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else clip_len
        colours = _limit_keywords(group) if (animated and config.SUB_KEYWORD_COLORS) else [None] * len(group)
        for wi, w in enumerate(group):
            start = w["start"]
            if wi + 1 < len(group):
                end = group[wi + 1]["start"]
            else:  # last word of the line: hold until the next line (or briefly)
                end = min(next_start, w["end"] + HOLD_AFTER)
            end = min(max(end, w["end"]), clip_len)
            if end - start < 0.04:
                continue
            parts = []
            for j, x in enumerate(group):
                if not animated:
                    parts.append(f"{{\\c{YELLOW}}}{x['word']}{{\\c{WHITE}}}" if j == wi else x["word"])
                elif j < wi:     # already spoken: white or keyword colour
                    parts.append(f"{{\\c{colours[j] or WHITE}}}{x['word']}")
                elif j == wi:    # being spoken: yellow + pop, then reset
                    parts.append(f"{{\\c{YELLOW}{POP}}}{x['word']}{{\\r}}")
                # Words not spoken yet are left out: the line stays centered
                # and grows word by word.
            events.append(
                f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Sub,,0,0,0,,"
                f"{{\\an5\\pos({PLAY_W // 2},{SUB_Y})}}{' '.join(parts)}"
            )
    return events


def _hook_event(hook_title: str, clip_len: float) -> str | None:
    text = re.sub(r"\s+", " ", _escape(hook_title)).strip().upper()  # ALL CAPS by design
    if not text:
        return None
    if config.EDIT_HOOK_ANIMATION:
        dur = min(config.HOOK_DURATION, clip_len)
        anim = r"\fscx40\fscy40\t(0,170,\fscx112\fscy112)\t(170,280,\fscx100\fscy100)\fad(0,350)"
    else:
        dur = min(HOOK_SECONDS_STATIC, clip_len)
        anim = r"\fad(120,200)"
    return (f"Dialogue: 2,{_ass_time(0)},{_ass_time(dur)},Hook,,0,0,0,,"
            f"{{\\an8\\pos({PLAY_W // 2},{HOOK_Y}){anim}}}{text}")


def _progress_bar_events(clip_len: float) -> list[str]:
    h = config.PROGRESS_BAR_HEIGHT
    y = 0 if config.PROGRESS_BAR_POSITION == "top" else PLAY_H - h
    rect = f"m 0 0 l {PLAY_W} 0 {PLAY_W} {h} 0 {h}"
    end = _ass_time(clip_len + 0.5)
    ms = int(clip_len * 1000)
    return [
        # Dim track.
        f"Dialogue: 3,{_ass_time(0)},{end},Bar,,0,0,0,,"
        f"{{\\an7\\pos(0,{y})\\1c&H000000&\\1a&H80&\\p1}}{rect}{{\\p0}}",
        # Fill grows from the left edge: x-scale 0% -> 100% over the clip.
        f"Dialogue: 4,{_ass_time(0)},{end},Bar,,0,0,0,,"
        f"{{\\an7\\pos(0,{y})\\1c{YELLOW}\\fscx0\\t(0,{ms},\\fscx100)\\p1}}{rect}{{\\p0}}",
    ]


def build_ass(moment: dict[str, Any], plan: EditPlan, hook: bool = True) -> str:
    """Full .ass document for one clip (clip-timeline times)."""
    clip_len = plan.duration
    terms = load_glossary_terms()
    words = clip_words(moment, plan)
    for w in words:
        w["word"] = fix_capitalisation(_escape(w["word"]), terms)
    if config.EDIT_ANIMATED_SUBS:
        mark_keywords(words, terms)
        if config.SUB_UPPERCASE:
            for w in words:
                w["word"] = w["word"].upper()

    events = _subtitle_events(words, clip_len)
    if hook:
        ev = _hook_event(str(moment.get("hook_title", "")), clip_len)
        if ev:
            events.append(ev)
    if config.EDIT_PROGRESS_BAR:
        events += _progress_bar_events(clip_len)
    return _header() + "\n".join(events) + "\n"


def write_ass(moment: dict[str, Any], plan: EditPlan, path: Path, hook: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_ass(moment, plan, hook=hook), encoding="utf-8-sig")
    return path
