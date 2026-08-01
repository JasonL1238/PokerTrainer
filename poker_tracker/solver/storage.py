from __future__ import annotations

import hashlib
import os
import shutil
from functools import lru_cache
from pathlib import Path

from poker_tracker.persistence.models import SolverRun

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("POKER_DATA_DIR", PROJECT_ROOT / "data"))
SOLVER_RUNS_DIR = DATA_DIR / "solver_runs"
_RETAINED_ARTIFACTS = (
    ("command file", "command_path"),
    ("result JSON", "result_path"),
    ("solver log", "log_path"),
)


def solver_run_directory(run_id: int) -> Path:
    SOLVER_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = SOLVER_RUNS_DIR / f"run_{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_solver_run_artifacts(run: SolverRun) -> bool:
    paths = [run.command_path, run.result_path, run.log_path]
    directories = {Path(path).resolve().parent for path in paths if path}
    root = SOLVER_RUNS_DIR.resolve()
    removed = False
    for directory in directories:
        if directory == root or root not in directory.parents:
            continue
        if directory.exists():
            shutil.rmtree(directory)
            removed = True
    return removed


def missing_run_artifacts(run: SolverRun) -> list[str]:
    """Name the retained artifacts a completed run points at but no longer has.

    A completed row goes on presenting its frequencies as evidence after its run
    directory is gone -- deleted with the hand, pruned by an operator, or simply
    never mounted into a fresh container. The result is not wrong, but it is no
    longer reproducible or auditable, and a reader who is not told that will
    assume it is both. Callers use this on completed runs only; a queued or
    failed run legitimately has no artifacts yet.
    """

    missing: list[str] = []
    for label, attribute in _RETAINED_ARTIFACTS:
        path = str(getattr(run, attribute, "") or "")
        if not path:
            missing.append(f"{label} (no path was ever recorded)")
            continue
        try:
            present = Path(path).is_file()
        except OSError:
            present = False
        if not present:
            missing.append(f"{label} ({path})")
    return missing


def resolved_backend_identity(binary: Path) -> str:
    """Identify the configured solver by its content rather than by the pin.

    ``PINNED_CONSOLE_COMMIT`` is a constant this build asserts and stamps onto
    every run; nothing ever compares it against the file ``TEXAS_SOLVER_PATH``
    points at, so a result produced by some other build is retained under the
    pinned commit and reads as verified provenance. The console exposes no
    version flag worth trusting -- executing an unidentified binary to ask it
    what it is has its own cost -- so the honest identity available without
    running it is the digest of the file that ran. Returns an empty string when
    the file cannot be read, because an unknown identity has to stay visibly
    unknown rather than fall back to the claim.
    """

    try:
        stat = binary.stat()
    except OSError:
        return ""
    try:
        return _binary_digest(str(binary), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return ""


def backend_identity_assumption(binary: Path | None, pinned_version: str) -> str:
    """State, for retention on the run, which binary produced the result."""

    identity = resolved_backend_identity(binary) if binary is not None else ""
    if not identity:
        return (
            "The solver binary could not be identified, so the backend version "
            f"retained for this run is the configured pin {pinned_version} rather "
            "than a verified build."
        )
    return (
        f"Solver binary identity {identity} ({binary}). The pinned commit "
        f"{pinned_version} is a configuration claim and was not verified against "
        "this binary."
    )


@lru_cache(maxsize=8)
def _binary_digest(path: str, size: int, mtime_ns: int) -> str:
    # Keyed on size and mtime as well as the path so that repointing
    # TEXAS_SOLVER_PATH, or rebuilding in place, is not served a stale digest.
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
