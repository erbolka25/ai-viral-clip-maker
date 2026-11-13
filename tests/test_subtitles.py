"""Subtitle generation (src/subtitles.py): capitalisation, hooks and .ass events."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from src import config, subtitles as subs
from src.editing import EditPlan
from src.transcriber import load_glossary


def fake_width(text: str, size: int, font: str) -> int:
    """Deterministic text width, independent of the OS font engine."""
    return int(len(text) * size * 0.75)


@pytest.fixture(autouse=True)
def deterministic_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subs, "text_width", fake_width)


@pytest.fixture
def glossary(isolated_paths: Path) -> Path:
    config.GLOSSARY_TXT.write_text("# names\nBoobie\nRaymond  # boss\n\nBoobie\n", encoding="utf-8")
    return config.GLOSSARY_TXT


# ---------------------------------------------------------------------------
# Glossary + capitalisation
# ---------------------------------------------------------------------------

def test_load_glossary_skips_comments_blanks_and_duplicates(glossary: Path) -> None:
    assert load_glossary(glossary) == ["Boobie", "Raymond"]


@pytest.mark.parametrize("word, expected", [
    ("boobie", "Boobie"),
    ("boobie,", "Boobie,"),
    ("boobie's", "Boobie's"),
    ("(raymond!)", "(Raymond!)"),
    ("BOOBIE", "Boobie"),
    ("boobies", "boobies"),        # different word: unchanged
    ("hello", "hello"),
])
def test_fix_capitalisation(word: str, expected: str) -> None:
    assert subs.fix_capitalisation(word, ["Boobie", "Raymond"]) == expected


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("t, expected", [
    (0, "0:00:00.00"), (1.234, "0:00:01.23"), (61.5, "0:01:01.50"), (3725.999, "1:02:06.00"), (-1, "0:00:00.00"),
])
def test_ass_time_format(t: float, expected: str) -> None:
    assert subs._ass_time(t) == expected


def test_escape_neutralises_ass_override_characters() -> None:
    assert subs._escape("{\\b1}hi\nthere") == "(/b1)hi there"


def test_group_words_makes_lines_of_two_to_four_words() -> None:
    words = [{"start": i * 0.3, "end": i * 0.3 + 0.25, "word": "go"} for i in range(11)]
    groups = subs.group_words(words)
    assert sum(len(g) for g in groups) == 11
    assert all(subs.MIN_WORDS <= len(g) <= subs.MAX_WORDS for g in groups)


def test_group_words_breaks_after_punctuation_and_pauses() -> None:
    words = [
        {"start": 0.0, "end": 0.3, "word": "WAIT"}, {"start": 0.3, "end": 0.6, "word": "UP!"},
        {"start": 0.6, "end": 0.9, "word": "COME"}, {"start": 0.9, "end": 1.2, "word": "ON"},
        {"start": 3.0, "end": 3.3, "word": "LATER"}, {"start": 3.3, "end": 3.6, "word": "ON"},
    ]
    assert [[x["word"] for x in g] for g in subs.group_words(words)] == [
        ["WAIT", "UP!"], ["COME", "ON"], ["LATER", "ON"]]


# ---------------------------------------------------------------------------
# Hook title
# ---------------------------------------------------------------------------

def test_short_hook_keeps_full_font_size() -> None:
    assert subs.hook_font_size("DEADLY JUMP") == subs.HOOK_SIZE


def test_long_hook_shrinks_to_fit_on_one_line() -> None:
    size = subs.hook_font_size("BIGGEST HEIST EVER")
    assert subs.HOOK_MIN_SIZE <= size < subs.HOOK_SIZE
    assert fake_width("BIGGEST HEIST EVER", size, subs.HOOK_FONT) <= subs.HOOK_MAX_WIDTH


def test_very_long_hook_stops_at_minimum_size() -> None:
    assert subs.hook_font_size("UNBELIEVABLE INSANE HEIST ESCAPE") == subs.HOOK_MIN_SIZE


def test_hook_event_is_uppercase_single_line_and_timed() -> None:
    ev = subs._hook_event("  watch   {this} ", clip_len=30.0)
    assert ev is not None
    assert ev.startswith(f"Dialogue: 2,0:00:00.00,{subs._ass_time(config.HOOK_DURATION)},Hook,")
    assert ev.endswith("WATCH (THIS)")
    assert "\\q2" in ev and f"\\fs{subs.HOOK_SIZE}" in ev
    assert subs._hook_event("   ", 30.0) is None


# ---------------------------------------------------------------------------
# Full .ass document
# ---------------------------------------------------------------------------

def sub_events(ass: str) -> list[tuple[str, str, str]]:
    """(start, end, text) of every subtitle-line event."""
    out = []
    for line in ass.splitlines():
        if line.startswith("Dialogue: 1,"):
            parts = line.split(",", 9)
            out.append((parts[1], parts[2], parts[9]))
    return out


def visible_words(text: str) -> list[str]:
    """Words of an event that are drawn (not alpha FF = invisible)."""
    out = []
    for chunk in text.split(" "):
        if "\\alpha&HFF&" in chunk:
            continue
        out.append(re.sub(r"\{[^}]*\}", "", chunk))
    return [x for x in out if x]


MOMENT: dict[str, Any] = {
    "hook_title": "Wait for it",
    "words": [
        {"start": 10.5, "end": 10.8, "word": "hey"},
        {"start": 10.8, "end": 11.1, "word": "boobie,"},
        {"start": 11.1, "end": 11.4, "word": "look"},
        {"start": 11.4, "end": 11.7, "word": "out!"},
        {"start": 13.2, "end": 13.5, "word": "{run}"},     # after the jump cut
        {"start": 13.5, "end": 13.9, "word": "now"},
    ],
}
# Source 10-11.5 s and 13-15 s are kept: 1.5 s removed in the middle.
PLAN = EditPlan(ss=10.0, segments=[(0.0, 1.5), (3.0, 5.0)], fps=25.0)


def test_build_ass_stable_layout_and_timings(glossary: Path) -> None:
    ass = subs.build_ass(MOMENT, PLAN)
    events = sub_events(ass)

    assert len(events) == len(MOMENT["words"])                # one event per spoken word
    starts = [e[0] for e in events]
    assert starts == ["0:00:00.50", "0:00:00.80", "0:00:01.10", "0:00:01.40",
                      "0:00:01.70", "0:00:02.00"]             # 13.2 s source -> 1.7 s clip (after the cut)
    # Within a line each event ends exactly when the next word starts.
    assert events[0][1] == events[1][0]

    # Stable layout: the whole line is in every event, unspoken words invisible,
    # and every event uses the same anchor so nothing moves.
    first = events[0][2]
    assert "HEY" in first and "BOOBIE," in first
    assert "\\alpha&HFF&}BOOBIE," in first
    assert visible_words(first) == ["HEY"]
    assert visible_words(events[1][2]) == ["HEY", "BOOBIE,"]
    anchors = {re.match(r"\{(\\an\d\\pos\(\d+,\d+\))", e[2]).group(1) for e in events}
    assert len(anchors) == 1 and anchors.pop().startswith("\\an2")

    assert "(RUN)" in ass and "{run}" not in ass               # user text escaped
    assert "Dialogue: 2," in ass and "WAIT FOR IT" in ass      # hook
    assert ass.count("Dialogue: 3,") == 1 and ass.count("Dialogue: 4,") == 1   # progress bar
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass


def test_build_ass_applies_glossary_when_not_uppercased(glossary: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SUB_UPPERCASE", False)
    ass = subs.build_ass(MOMENT, PLAN, hook=False)
    assert "Boobie," in ass and "boobie," not in ass
    assert "Dialogue: 2," not in ass                           # hook disabled


def test_keywords_names_green_strong_words_red_max_two_per_line() -> None:
    words = [{"word": x} for x in ["Hey", "boobie,", "that", "was", "Insane"]]
    subs.mark_keywords(words, ["Boobie"])
    kinds = [x["kind"] for x in words]
    assert kinds[1] == "name"                 # glossary term
    assert kinds[0] is None                   # capitalised sentence start is not a name

    line = [{"word": "A", "kind": "strong"}, {"word": "Raymond", "kind": "name"},
            {"word": "BIG", "kind": "strong"}, {"word": "x", "kind": None}]
    colours = subs._limit_keywords(line)
    assert colours[1] == subs.GREEN                       # names first
    assert colours.count(None) == 2                        # at most 2 coloured
    assert colours[2] == subs.RED                          # longer strong word wins


def test_build_ass_non_stable_layout_omits_unspoken_words(glossary: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SUB_STABLE_LAYOUT", False)
    first = sub_events(subs.build_ass(MOMENT, PLAN))[0][2]
    assert "BOOBIE" not in first and "\\an5" in first
