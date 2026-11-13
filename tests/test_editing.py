"""Jump-cut planning (src/editing.py) with synthetic audio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src import config, editing
from tests.conftest import write_wav


def moment(words: list[tuple[float, float]], start: float = 0.0, end: float | None = None) -> dict[str, Any]:
    return {
        "start": start,
        "end": end if end is not None else words[-1][1] + 1.0,
        "words": [{"start": a, "end": b, "word": "word"} for a, b in words],
    }


def covered(ranges: list[tuple[float, float]], t: float) -> bool:
    return any(a <= t <= b for a, b in ranges)


# Speech 0-2 s, quiet gap 2-6 s, speech 6-8 s, LOUD gap 8-12 s (action), speech 12-14 s.
SPEECH = [(0.2, 1.0), (1.0, 2.0), (6.0, 7.0), (7.0, 8.0), (12.0, 13.0), (13.0, 14.0)]


@pytest.fixture
def action_audio(isolated_paths: Path) -> Path:
    return write_wav(config.AUDIO_WAV, 30.0, loud=[(0.0, 2.0), (6.0, 14.0)])


def test_plain_jump_cuts_remove_every_long_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "JUMP_CUT_SMART", False)
    keep = editing.keep_ranges(moment(SPEECH), 100.0)
    pad = config.JUMP_CUT_PAD
    assert keep == [
        (0.0, 2.0 + pad),
        pytest.approx((6.0 - pad, 8.0 + pad)),
        pytest.approx((12.0 - pad, 14.0 + config.END_PAD)),   # clean ending
    ]


def test_smart_cuts_remove_quiet_gaps_but_keep_loud_action(action_audio: Path) -> None:
    assert config.JUMP_CUT_SMART
    keep = editing.keep_ranges(moment(SPEECH), 100.0)

    assert not covered(keep, 4.0)           # quiet gap removed
    for t in (9.0, 10.0, 11.0):             # loud gap (driving/shooting) kept
        assert covered(keep, t)
    for a, b in SPEECH:                     # every word stays
        assert covered(keep, a) and covered(keep, b)
    assert len(keep) == 2


def test_without_audio_file_smart_cuts_fall_back_to_plain_cuts() -> None:
    assert not config.AUDIO_WAV.is_file()
    keep = editing.keep_ranges(moment(SPEECH), 100.0)
    assert not covered(keep, 10.0)


def test_short_loud_burst_does_not_split_a_quiet_run(isolated_paths: Path) -> None:
    write_wav(config.AUDIO_WAV, 30.0, loud=[(22.0, 22.2)])     # 0.2 s footstep
    runs = editing.quiet_runs(20.0, 24.0)
    assert len(runs) == 1
    assert runs[0][0] == pytest.approx(20.0) and runs[0][1] == pytest.approx(24.0)


def test_long_loud_part_splits_quiet_runs(isolated_paths: Path) -> None:
    write_wav(config.AUDIO_WAV, 30.0, loud=[(22.0, 23.0)])     # 1 s of action
    runs = editing.quiet_runs(20.0, 25.0)
    assert len(runs) == 2
    assert runs[0][1] == pytest.approx(22.0, abs=0.06)
    assert runs[1][0] == pytest.approx(23.0, abs=0.06)


def test_cut_that_would_leave_a_tiny_piece_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "JUMP_CUT_SMART", False)
    # A lone 0.5 s word between two 1 s pauses: cutting both pauses would leave
    # a 0.8 s piece (< JUMP_CUT_MIN_PIECE), so the second cut is skipped.
    words = [(0.0, 2.0), (3.0, 3.5), (4.5, 8.0)]
    keep = editing.keep_ranges(moment(words), 100.0)

    assert len(keep) == 2
    assert all(b - a >= config.JUMP_CUT_MIN_PIECE for a, b in keep)
    assert covered(keep, 4.0)                # the skipped pause stays in the clip


def test_cut_before_a_tiny_last_piece_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "JUMP_CUT_SMART", False)
    monkeypatch.setattr(config, "EDIT_CLEAN_ENDING", True)
    keep = editing.keep_ranges(moment([(0.0, 3.0), (4.5, 4.7)]), 100.0)
    assert keep == [(0.0, pytest.approx(5.0))]   # no cut: last piece would be 0.65 s


def test_jump_cuts_disabled_keeps_one_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EDIT_JUMP_CUTS", False)
    assert editing.keep_ranges(moment(SPEECH), 100.0) == [(0.0, pytest.approx(14.3))]


def test_frame_align_snaps_to_frames_and_merges_touching_ranges() -> None:
    ss, segs = editing.frame_align([(10.01, 12.0), (12.01, 13.0), (15.0, 15.02)], fps=25)
    assert ss == 10.01
    assert segs == [(0.0, pytest.approx(3.0, abs=1 / 25))]  # touching merged, 1-frame piece dropped
    for a, b in segs:
        assert round(a * 25) == pytest.approx(a * 25) and round(b * 25) == pytest.approx(b * 25)


def test_edit_plan_maps_source_time_across_cuts() -> None:
    plan = editing.EditPlan(ss=10.0, segments=[(0.0, 2.0), (4.0, 6.0)], fps=25)
    assert plan.duration == 4.0
    assert plan.cut_times == [2.0]
    assert plan.to_clip_time(11.0) == 1.0
    assert plan.to_clip_time(13.0) == 2.0      # inside the removed part -> next kept frame
    assert plan.to_clip_time(14.5) == 2.5


def test_clip_words_drops_words_outside_the_clip() -> None:
    plan = editing.EditPlan(ss=10.0, segments=[(0.0, 2.0), (4.0, 6.0)], fps=25)
    m = {"words": [{"start": 10.5, "end": 10.8, "word": "a"},
                   {"start": 14.2, "end": 14.6, "word": "b"},
                   {"start": 30.0, "end": 30.5, "word": "late"},
                   {"start": 11.0, "end": 11.2, "word": "  "}]}
    out = editing.clip_words(m, plan)
    assert [(x["word"], x["start"]) for x in out] == [("a", 0.5), ("b", pytest.approx(2.2))]
