"""Moment boundaries and validation (src/highlights.py) - no LLM calls."""

from __future__ import annotations

from typing import Any

import pytest

from src import highlights as hl
from tests.conftest import make_transcript, w


def words_in_range(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [x for x in words if start <= (x["start"] + x["end"]) / 2 <= end]


# ---------------------------------------------------------------------------
# Sentence detection
# ---------------------------------------------------------------------------

def test_sentence_words_marks_punctuation_and_pauses() -> None:
    transcript = {"duration": 30, "segments": [
        # "going to" | "Get off": a segment end after "to" is NOT a sentence end.
        {"start": 0, "end": 1.5, "text": " We are going to", "words": [
            w(0.0, 0.3, "We"), w(0.3, 0.6, "are"), w(0.6, 1.0, "going"), w(1.0, 1.5, "to")]},
        {"start": 1.6, "end": 2.4, "text": " Get off", "words": [
            w(1.6, 2.0, "Get"), w(2.0, 2.4, "off")]},
        # "off" + long pause + capitalised "Then": a sentence end.
        {"start": 5.0, "end": 6.0, "text": " Then run!", "words": [
            w(5.0, 5.5, "Then"), w(5.5, 6.0, "run!")]},
        # "run!" ends with punctuation; " y" + "'all" is merged into one word.
        {"start": 6.5, "end": 7.5, "text": " Come on y'all", "words": [
            w(6.5, 6.8, "Come"), w(6.8, 7.0, "on"), w(7.0, 7.2, "y"),
            {"start": 7.2, "end": 7.5, "word": "'all"}]},
    ]}
    words = hl.sentence_words(transcript)
    by_text = {x["word"]: x for x in words}

    assert by_text["to"]["sent_end"] is False          # non-final word at segment end
    assert by_text["Get"]["sent_start"] is False
    assert by_text["off"]["sent_end"] is True           # pause > SENTENCE_GAP, next word capitalised
    assert by_text["Then"]["sent_start"] is True
    assert by_text["run!"]["sent_end"] is True          # punctuation
    assert "y'all" in by_text                           # sub-word pieces merged
    assert words[-1]["sent_end"] is True                # last word always ends a sentence


# ---------------------------------------------------------------------------
# Sentence boundary snapping
# ---------------------------------------------------------------------------

def test_snap_to_sentences_moves_both_ends_to_sentence_boundaries() -> None:
    # 20 sentences of 8 words, 4 s each: sentence k spans [4k, 4k + 3.9].
    transcript = make_transcript(20)
    words = hl.sentence_words(transcript)

    start, end = hl.snap_to_sentences(6.2, 31.3, words, transcript["duration"])

    inside = words_in_range(words, start, end)
    assert inside[0]["word"] == "Hello" and inside[0]["start"] == 4.0   # back to sentence 2's start
    assert inside[-1]["word"] == "end." and inside[-1]["end"] == 31.9    # forward to sentence 8's end
    assert hl.MIN_LEN <= end - start <= hl.MAX_LEN
    # Padding never reaches into the neighbouring words.
    assert start > 3.9 and end < 32.0


def test_snap_to_sentences_extends_short_moment_to_minimum_length() -> None:
    transcript = make_transcript(20)
    words = hl.sentence_words(transcript)

    start, end = hl.snap_to_sentences(40.0, 47.9, words, transcript["duration"])

    assert end - start >= hl.MIN_LEN - 0.5    # whole sentences added (pads may be clipped)
    inside = words_in_range(words, start, end)
    assert inside[0]["sent_start"] and inside[-1]["sent_end"]


def test_snap_to_sentences_without_words_returns_input() -> None:
    assert hl.snap_to_sentences(5.0, 30.0, [], 100.0) == (5.0, 30.0)


# ---------------------------------------------------------------------------
# Word snapping and 20-60 s enforcement
# ---------------------------------------------------------------------------

@pytest.fixture
def all_words() -> list[dict[str, Any]]:
    transcript = make_transcript(40)            # 160 s of continuous speech
    return [x for s in transcript["segments"] for x in s["words"]]


def test_snap_and_fix_pads_short_moment_to_20_seconds(all_words: list[dict[str, Any]]) -> None:
    span = hl.snap_and_fix({"start": 10, "end": 15}, all_words, 200.0)
    assert span is not None
    assert span[1] - span[0] == pytest.approx(hl.MIN_LEN, abs=0.01)


def test_snap_and_fix_trims_long_moment_to_60_seconds(all_words: list[dict[str, Any]]) -> None:
    span = hl.snap_and_fix({"start": 10, "end": 110}, all_words, 200.0)
    assert span is not None
    assert span[1] - span[0] <= hl.MAX_LEN + 0.01


@pytest.mark.parametrize("moment", [
    {"start": 30, "end": 20},              # end before start
    {"start": "x", "end": 20},             # not a number
    {"title": "no times"},                 # missing keys
    {"start": 500, "end": 540},            # no words there
])
def test_snap_and_fix_rejects_unusable_moments(all_words: list[dict[str, Any]], moment: dict[str, Any]) -> None:
    assert hl.snap_and_fix(moment, all_words, 600.0) is None


# ---------------------------------------------------------------------------
# Full post-processing: overlaps, silent moments, field cleanup
# ---------------------------------------------------------------------------

@pytest.fixture
def transcript_with_silence() -> dict[str, Any]:
    """0-120 s: continuous speech. 150-190 s: only three words (music)."""
    t = make_transcript(30, duration=300.0)
    t["segments"].append({"start": 150, "end": 190.5, "text": " Hey ... What? ... Go!", "words": [
        w(150.0, 150.5, "Hey"), w(170.0, 170.5, "What?"), w(190.0, 190.5, "Go!")]})
    return t


def test_postprocess_drops_overlaps_and_silent_moments(transcript_with_silence: dict[str, Any]) -> None:
    raw = [
        {"start": 10, "end": 40, "title": "Best", "category": "funny", "score": 9,
         "reason": "r", "hook_title": "Watch this"},
        {"start": 12, "end": 42, "title": "Duplicate", "category": "funny", "score": 6,
         "reason": "r", "hook_title": "Same thing"},
        {"start": 150, "end": 190, "title": "Silent", "category": "exciting", "score": 8,
         "reason": "r", "hook_title": "Music only"},
        {"start": 60, "end": 90, "title": "  Second   one ", "category": "Funny Stuff", "score": "11",
         "reason": "r", "hook_title": "\"This hook is far too long\""},
    ]
    out = hl.postprocess(raw, transcript_with_silence)

    assert [m["title"] for m in out] == ["Best", "Second one"]      # duplicate + silent dropped
    assert [m["id"] for m in out] == [1, 2]                         # renumbered in time order
    second = out[1]
    assert second["category"] == "exciting"                         # unknown category -> default
    assert second["score"] == 10                                    # clamped to 1-10
    assert second["hook_title"] == "This hook is far"               # max HOOK_MAX_WORDS words
    for m in out:
        assert hl.MIN_LEN - 0.5 <= m["duration"] <= hl.MAX_LEN + 0.5
        assert m["words"] and all(x["word"] in {"Hello", "word", "end."} for x in m["words"])
        assert m["start"] <= m["words"][0]["start"] and m["words"][-1]["end"] <= m["end"]
    assert hl.validate(out, transcript_with_silence["duration"]) == []


def _moment(id_: int, start: float, end: float, **kw: Any) -> dict[str, Any]:
    m = {"id": id_, "start": start, "end": end, "category": "funny",
         "hook_title": "Wait for it", "words": [{"start": start, "end": start + 1, "word": "Hi"}]}
    m.update(kw)
    return m


def test_validate_accepts_good_moments() -> None:
    assert hl.validate([_moment(1, 0, 30), _moment(2, 33, 60)], 100.0) == []


@pytest.mark.parametrize("moments, expected", [
    ([_moment(1, 0, 15)], "not in 20-60s"),
    ([_moment(1, 0, 75)], "not in 20-60s"),
    ([_moment(1, 90, 120)], "outside video"),
    ([_moment(1, 0, 30, category="boring")], "bad category"),
    ([_moment(1, 0, 30, hook_title="one two three four five")], "hook_title"),
    ([_moment(1, 0, 30, hook_title="")], "hook_title"),
    ([_moment(1, 0, 30, words=[])], "no words"),
    ([_moment(1, 0, 30), _moment(2, 20, 50)], "overlap"),
])
def test_validate_reports_problems(moments: list[dict[str, Any]], expected: str) -> None:
    problems = hl.validate(moments, 100.0)
    assert any(expected in p for p in problems), problems


def test_small_overlap_is_allowed() -> None:
    # Up to MAX_OVERLAP (5 s) of shared time is fine.
    assert hl.validate([_moment(1, 0, 30), _moment(2, 26, 50)], 100.0) == []


# ---------------------------------------------------------------------------
# Hook title word limit
# ---------------------------------------------------------------------------

def test_clean_text_limits_hook_words_and_whitespace() -> None:
    assert hl.HOOK_MAX_WORDS == 4
    assert hl.clean_text('  "Wait   for it\n right now"  ', max_words=hl.HOOK_MAX_WORDS) == "Wait for it right"
    assert hl.clean_text(None) == ""
    assert hl.clean_text("Short hook", max_words=4) == "Short hook"
