"""Environment and identity capture for release reports."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from poker_tracker.safety.redaction import redact_text

_SECRET_KEY = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|apikey|authorization|credential"
    r"|_pw$|_key$|access[_-]?key|private[_-]?key|session[_-]?id)",
    re.IGNORECASE,
)


def _git_identity(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                args,
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(dirty) if dirty is not None else None,
        "dirty_paths": dirty.splitlines() if dirty else [],
    }


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if _SECRET_KEY.search(str(key)):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = _redact_value(value)
    return redacted


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        # Key-name matching alone misses a credential stored under a name nobody
        # anticipated, so the value is also scrubbed by shape. Environment
        # literals are excluded here: this IS the environment, and substituting
        # every configured value against itself would blank the whole snapshot.
        return redact_text(value, include_environment=False)
    return value


# Dependencies whose version materially changes a reconstruction verdict.
RELEASE_CRITICAL_PACKAGES: tuple[str, ...] = (
    "streamlit",
    "numpy",
    "opencv-python-headless",
    "opencv-python",
    "torch",
    "ultralytics",
    "av",
    "pydantic",
    "pillow",
)


def _dependency_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in RELEASE_CRITICAL_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def ffmpeg_version() -> str | None:
    """First line of ``ffmpeg -version``, or None when it is not installed."""
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    first = completed.stdout.strip().splitlines()
    return first[0] if first else None


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return None


def collect_environment(repo_root: Path) -> dict[str, Any]:
    env_subset = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("POKER")
        or key in {"TEXAS_SOLVER_PATH", "POKERTRAINER_REQUIRE_AUTH", "APP_PASSWORD"}
    }
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "ffmpeg": ffmpeg_version(),
        "dependencies": _dependency_versions(),
        "git": _git_identity(repo_root),
        "env": redact_mapping(env_subset),
    }
