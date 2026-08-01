"""What the study history pointed at when a snapshot was taken.

A SQLite snapshot restores rows, not the study history. The recordings, extracted
frames, reconstruction timelines and solver outputs those rows reference live on
disk as files, and they are deliberately never copied into a backup -- PLAN keeps
videos out of SQL and out of anything routinely copied around. Restoring a
three-month-old snapshot therefore produces a database full of paths whose
current state nobody knows, and no way to enumerate what would have to be put
back.

The inventory is the missing half. For every artifact the snapshot's own rows
reference -- derived from ``PokerDatabase.ARTIFACT_PATH_COLUMNS`` so a column
added later is covered without editing this module -- plus the reconstruction
timelines that are named by convention rather than by a column, it records the
stored path, whether the file existed at snapshot time, its size, its mtime and,
within a bounded budget, its SHA-256. It records identity; it never copies bytes.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from poker_tracker.persistence.backup import DATA_DIR, resolve_artifact_path

INVENTORY_VERSION = 1
INVENTORY_SUFFIX = ".inventory.json"

# Hashing is bounded twice over. Per file, because a hand from a 12 GB recording
# must not make a snapshot take minutes; in total, because a study database can
# reference tens of thousands of extracted frames and re-reading all of them on
# every backup is a runtime cost this project counts. An unhashed file still
# records its size and mtime, which catches truncation and replacement.
HASH_FILE_LIMIT_BYTES = 4 * 1024 * 1024
HASH_BUDGET_BYTES = 64 * 1024 * 1024

# The timeline a CV job wrote has no column anywhere; it is found by convention,
# and a missing one hard-blocks every later validated-hand import for that job.
# Inventorying it is the only way a restored machine can discover that.
_COMPLETED_CV_JOBS = (
    "SELECT id FROM processing_jobs "
    "WHERE job_type = 'cv_reconstruction' AND status = 'completed'"
)
TIMELINE_SOURCE = "processing_jobs.timeline"
# The subdirectory of the data root that ensure_data_directories creates for
# timelines. Kept beside the naming convention it completes so the inventory and
# the health audit look in the same place from any data root, including a
# restored one that is not the one the environment points at.
TIMELINE_DIR_NAME = "cv_timelines"


def timeline_dir_for(data_dir: Path) -> Path:
    return Path(data_dir) / TIMELINE_DIR_NAME


def timeline_paths(connection: sqlite3.Connection, timeline_dir: Path) -> list[str]:
    """Where each completed CV job's timeline should be, by the naming convention.

    One definition, used by both the snapshot inventory and the health audit, so
    the two cannot look for a job's timeline in different places.
    """
    rows = connection.execute(_COMPLETED_CV_JOBS).fetchall()
    return [str(Path(timeline_dir) / f"job_{row[0]}_timeline.json") for row in rows]


def inventory_path(snapshot: Path) -> Path:
    """The inventory beside a snapshot.

    The suffix is appended rather than substituted so the name still ends in the
    snapshot's own name: the rotation patterns anchor on ``.sqlite3$`` and must
    not see an inventory as a snapshot, and the audit must be able to find the
    inventory from the snapshot without a second naming rule.
    """
    return Path(f"{snapshot}{INVENTORY_SUFFIX}")


def write_inventory(
    snapshot: Path,
    *,
    database_path: Path,
    data_dir: Path | None = None,
    timeline_dir: Path | None = None,
) -> Path:
    """Write the inventory for ``snapshot``; never raise into the backup caller.

    ``db.py`` aborts a migration when taking a snapshot raises, and losing the
    rollback point because its inventory could not be built would trade the
    important artifact for the useful one. A failure is written INTO the
    inventory instead, where the health audit reports it, rather than swallowed.
    """
    payload = build_inventory(
        snapshot,
        database_path=database_path,
        data_dir=data_dir,
        timeline_dir=timeline_dir,
    )
    destination = inventory_path(snapshot)
    try:
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    except OSError:
        # A backup directory that cannot take the inventory has already taken the
        # snapshot; the audit reports the absent inventory in its own message.
        pass
    return destination


def build_inventory(
    snapshot: Path,
    *,
    database_path: Path,
    data_dir: Path | None = None,
    timeline_dir: Path | None = None,
) -> dict[str, Any]:
    """Describe every artifact the snapshot's rows reference, without copying one.

    Read from the SNAPSHOT rather than the live database: the snapshot is the
    consistent copy, so its reference set is exactly the set that the restored
    database will point at, even if the live file changes mid-backup.
    """
    data_root = DATA_DIR if data_dir is None else Path(data_dir)
    timelines = timeline_dir_for(data_root) if timeline_dir is None else Path(timeline_dir)
    entries: list[dict[str, Any]] = []
    unreadable: list[str] = []
    error: str | None = None
    budget = _HashBudget(HASH_BUDGET_BYTES)
    try:
        with closing(_open_read_only(snapshot)) as connection:
            for source, stored in _referenced_paths(connection, unreadable):
                entries.append(
                    _describe(
                        source,
                        stored,
                        database_path=Path(database_path),
                        data_dir=data_root,
                        budget=budget,
                    )
                )
            for stored in _inventory_timeline_paths(connection, timelines, unreadable):
                entries.append(
                    _describe(
                        TIMELINE_SOURCE,
                        stored,
                        database_path=Path(database_path),
                        data_dir=data_root,
                        budget=budget,
                    )
                )
    except (OSError, sqlite3.Error) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "inventory_version": INVENTORY_VERSION,
        "snapshot": snapshot.name,
        "database_path": str(database_path),
        "data_dir": str(data_root),
        "error": error,
        # "could not be read" and "holds no references" must never look the same:
        # a caller acting on the first would conclude a restore needs nothing.
        "unreadable_sources": sorted(unreadable),
        "hash_file_limit_bytes": HASH_FILE_LIMIT_BYTES,
        "hash_budget_bytes": HASH_BUDGET_BYTES,
        "hash_budget_exhausted": budget.exhausted,
        "artifacts": sorted(entries, key=lambda item: (item["source"], item["path"])),
    }


def load_inventory(snapshot: Path) -> dict[str, Any] | None:
    """The inventory beside a snapshot, or None when there is none to read."""
    path = inventory_path(snapshot)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def inventory_findings(
    inventory: dict[str, Any],
    *,
    database_path: Path,
    data_dir: Path,
) -> list[str]:
    """What has happened since the snapshot to the files it needs.

    Only artifacts that EXISTED at snapshot time are reported as gone: a path
    that was already dangling then is a live-database problem the artifact check
    already reports, and repeating it here would bury the new information.
    Resolution is redone from the stored path so a restore onto a machine with a
    different data root still finds its files.
    """
    findings: list[str] = []
    if inventory.get("error"):
        findings.append(f"artifact inventory failed: {inventory['error']}")
    unreadable = list(inventory.get("unreadable_sources") or [])
    if unreadable:
        # One line, not one per source. A snapshot at an older schema genuinely
        # has no solver_runs or regression_cases table, and nine separate
        # warnings about that would bury the artifacts that really went missing.
        findings.append(
            f"artifact inventory could not read {len(unreadable)} source(s): "
            + ", ".join(unreadable)
        )
    missing: list[str] = []
    changed: list[str] = []
    for entry in inventory.get("artifacts") or []:
        if not isinstance(entry, dict) or not entry.get("present"):
            continue
        stored = str(entry.get("path") or "")
        resolved = resolve_artifact_path(stored, Path(database_path), Path(data_dir))
        source = str(entry.get("source") or "artifact")
        try:
            if not resolved.is_file():
                missing.append(f"{source}: {stored}")
                continue
            recorded_size = entry.get("bytes")
            actual_size = resolved.stat().st_size
        except OSError as exc:
            missing.append(f"{source}: {stored} ({exc})")
            continue
        if isinstance(recorded_size, int) and actual_size != recorded_size:
            changed.append(
                f"{source}: {stored} was {recorded_size} bytes, is now {actual_size}"
            )
    findings.extend(f"artifact missing since the snapshot -- {item}" for item in missing)
    findings.extend(f"artifact changed since the snapshot -- {item}" for item in changed)
    return findings


class _HashBudget:
    """How many more bytes this inventory may read to hash."""

    def __init__(self, total: int) -> None:
        self.remaining = total
        self.exhausted = False

    def allowance(self) -> int:
        if self.remaining <= 0:
            self.exhausted = True
            return 0
        return min(HASH_FILE_LIMIT_BYTES, self.remaining)

    def spend(self, size: int) -> None:
        self.remaining -= size
        if self.remaining <= 0:
            self.exhausted = True


def _describe(
    source: str,
    stored: str,
    *,
    database_path: Path,
    data_dir: Path,
    budget: _HashBudget,
) -> dict[str, Any]:
    from poker_tracker.services.issue_bundle import identify_artifact

    resolved = resolve_artifact_path(stored, database_path, data_dir)
    allowance = budget.allowance()
    identity = identify_artifact(resolved, hash_limit_bytes=allowance)
    entry: dict[str, Any] = {
        "source": source,
        "path": stored,
        "resolved_path": str(resolved),
        "present": False,
        "bytes": None,
        "sha256": None,
        "mtime": None,
    }
    if identity is None:
        return entry
    entry["present"] = identity.present
    entry["bytes"] = identity.bytes
    entry["sha256"] = identity.sha256
    if identity.sha256 is not None and identity.bytes:
        budget.spend(identity.bytes)
    if identity.present:
        try:
            entry["mtime"] = resolved.stat().st_mtime
        except OSError:
            entry["mtime"] = None
    return entry


def _referenced_paths(
    connection: sqlite3.Connection, unreadable: list[str]
) -> list[tuple[str, str]]:
    """Every stored path, labelled by the column it came from.

    The queries come from ``ARTIFACT_PATH_COLUMNS``, the list retention already
    consults before deleting anything, so the inventory and retention can never
    disagree about what counts as referenced. A pre-migration snapshot legitimately
    lacks some of those tables; each source is read independently so an old shape
    yields the sources it has plus a named record of the ones it does not.
    """
    from poker_tracker.persistence.db import PokerDatabase

    found: list[tuple[str, str]] = []
    for label, sql in PokerDatabase.ARTIFACT_PATH_COLUMNS:
        try:
            rows = connection.execute(sql).fetchall()
        except sqlite3.Error:
            unreadable.append(label)
            continue
        for row in rows:
            raw = row[0]
            if isinstance(raw, str) and raw.strip():
                found.append((label, raw.strip()))
    return found


def _inventory_timeline_paths(
    connection: sqlite3.Connection,
    timeline_dir: Path,
    unreadable: list[str],
) -> list[str]:
    try:
        return timeline_paths(connection, timeline_dir)
    except sqlite3.Error:
        unreadable.append(TIMELINE_SOURCE)
        return []


def _open_read_only(snapshot: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{snapshot.resolve().as_uri()}?mode=ro", uri=True, timeout=5
    )
    connection.execute("PRAGMA query_only = ON")
    return connection
