"""Dependency discovery: FFmpeg / FFprobe binaries and Python packages.

On Windows, a freshly installed FFmpeg (e.g. via winget) is often not visible
to an already-open terminal because PATH is only read at shell start-up. To be
robust we look in several places, in this order:

1. ``FFMPEG_DIR`` environment variable (folder containing ffmpeg.exe)
2. The system PATH
3. Common winget / Chocolatey / manual install locations
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


class DependencyError(RuntimeError):
    """Raised when a required external tool or package is missing."""


def _candidate_dirs() -> list[Path]:
    """Return extra folders where FFmpeg is commonly installed on Windows."""
    dirs: list[Path] = []

    env_dir = os.environ.get("FFMPEG_DIR")
    if env_dir:
        dirs.append(Path(env_dir))

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        winget_root = Path(local_appdata) / "Microsoft" / "WinGet"
        dirs.append(winget_root / "Links")
        packages = winget_root / "Packages"
        if packages.is_dir():
            # e.g. Packages/Gyan.FFmpeg_.../ffmpeg-9.0.2-full_build/bin
            dirs.extend(sorted(packages.glob("*FFmpeg*/*/bin"), reverse=True))

    dirs.extend(
        [
            Path("C:/ProgramData/chocolatey/bin"),
            Path("C:/ffmpeg/bin"),
            Path("C:/Program Files/ffmpeg/bin"),
        ]
    )
    return dirs


@lru_cache(maxsize=None)
def find_binary(name: str) -> Path | None:
    """Locate an executable such as ``ffmpeg`` or ``ffprobe``.

    Returns the absolute path, or ``None`` if it cannot be found.
    """
    exe_name = f"{name}.exe" if os.name == "nt" else name

    # FFMPEG_DIR takes priority over PATH so users can pin a specific build.
    env_dir = os.environ.get("FFMPEG_DIR")
    if env_dir and (Path(env_dir) / exe_name).is_file():
        return (Path(env_dir) / exe_name).resolve()

    on_path = shutil.which(name)
    if on_path:
        return Path(on_path).resolve()

    for directory in _candidate_dirs():
        candidate = directory / exe_name
        if candidate.is_file():
            return candidate.resolve()
    return None


def require_binary(name: str) -> Path:
    """Like :func:`find_binary` but raises a helpful error if missing."""
    path = find_binary(name)
    if path is None:
        raise DependencyError(
            f"'{name}' was not found.\n"
            "  Install FFmpeg, e.g.:  winget install Gyan.FFmpeg\n"
            "  then open a NEW terminal, or set FFMPEG_DIR to the folder "
            f"containing {name}.exe."
        )
    return path


def has_package(module_name: str) -> bool:
    """Return True if a Python module can be imported."""
    return importlib.util.find_spec(module_name) is not None


@dataclass
class DependencyReport:
    """Result of checking all Phase 1 dependencies."""

    found: dict[str, str] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing


def check_dependencies() -> DependencyReport:
    """Check every external tool and Python package Phase 1 needs."""
    report = DependencyReport()

    for binary in ("ffmpeg", "ffprobe"):
        path = find_binary(binary)
        if path:
            report.found[binary] = str(path)
        else:
            report.missing[binary] = "winget install Gyan.FFmpeg  (then open a new terminal)"

    packages = {
        "faster_whisper": "pip install -r requirements.txt",
        "rich": "pip install -r requirements.txt",
    }
    for module, hint in packages.items():
        if has_package(module):
            report.found[module] = "installed"
        else:
            report.missing[module] = hint

    return report
