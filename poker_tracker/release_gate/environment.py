"""Environment and identity capture for release reports."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_SECRET_KEY = re.compile(
    r"(password|secret|token|api[_-]?key|authorization|credential)",
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
    if isinstance(value, str) and _SECRET_KEY.search(value):
        # Value itself looks credential-bearing (e.g. embedded password=).
        return "<redacted>"
    if isinstance(value, str) and ("://" in value and "@" in value.split("://", 1)[-1]):
        # user:pass@host DSNs
        return "<redacted>"
    return value


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
        "git": _git_identity(repo_root),
        "env": redact_mapping(env_subset),
    }
