"""--input PATH and source-video resolution (src/config.py, src/main.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config, main


def test_input_flag_is_optional() -> None:
    assert main.parse_args([]).input is None
    assert main.parse_args(["--input", "clip.mp4"]).input == "clip.mp4"


def test_source_video_defaults_to_input_video() -> None:
    assert not config.VIDEO_INFO_JSON.is_file()
    assert config.source_video() == config.INPUT_VIDEO


def test_source_video_uses_path_recorded_by_main(isolated_paths: Path) -> None:
    video = isolated_paths / "my video.mp4"
    video.write_bytes(b"x")
    config.VIDEO_INFO_JSON.write_text(json.dumps({"path": str(video)}), encoding="utf-8")
    assert config.source_video() == video


def test_source_video_resolves_relative_paths_against_project_root(
        isolated_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PROJECT_ROOT", isolated_paths)
    (isolated_paths / "videos").mkdir()
    (isolated_paths / "videos" / "a.mp4").write_bytes(b"x")
    config.VIDEO_INFO_JSON.write_text(json.dumps({"path": "videos/a.mp4"}), encoding="utf-8")
    assert config.source_video() == isolated_paths / "videos" / "a.mp4"


@pytest.mark.parametrize("content", ['{"path": "missing.mp4"}', "not json", "{}"])
def test_source_video_falls_back_when_record_is_unusable(content: str) -> None:
    config.VIDEO_INFO_JSON.write_text(content, encoding="utf-8")
    assert config.source_video() == config.INPUT_VIDEO


def test_switching_videos_requires_overwrite(isolated_paths: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, new = isolated_paths / "old.mp4", isolated_paths / "new.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    config.VIDEO_INFO_JSON.write_text(json.dumps({"path": str(old)}), encoding="utf-8")
    monkeypatch.setattr(config, "INPUT_VIDEO", new)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)

    with pytest.raises(ValueError, match="another video"):
        main.step_input(main.Overwrite.SKIP)
    main.step_input(main.Overwrite.YES)               # -y: allowed
    monkeypatch.setattr(config, "INPUT_VIDEO", old)
    main.step_input(main.Overwrite.SKIP)              # same video: allowed
