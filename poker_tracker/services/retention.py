"""Explicit retention behavior for generated artifacts, with a dry-run audit.

Phase 4 requires two things that are easy to get backwards. Retention has to be
*defined* — an operator must be able to say how long frames, timelines, logs and
exports live — and nothing may be deleted without the operator seeing an audit
first.

The safety rule this module is built around: **a file the database still points
at is never deletable, at any age.** Retention windows only ever apply to files
nothing references. That inverts the usual "delete things older than N days",
which would happily remove the source frame behind a saved issue because the
issue is six months old. Age is the second question here, never the first.

Source videos are deliberately never expired by age. They are the irreplaceable
input; every derived artifact can be rebuilt from them and they cannot be
rebuilt from anything. Only an unreferenced *orphan* video is ever offered, and
only through an explicit opt-in.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poker_tracker.persistence.db import PokerDatabase

# Category -> environment variable holding its retention window in days.
RETENTION_ENV_VARS: dict[str, str] = {
    "frames": "POKER_RETAIN_FRAMES_DAYS",
    "cv_timelines": "POKER_RETAIN_TIMELINES_DAYS",
    "job_logs": "POKER_RETAIN_JOB_LOGS_DAYS",
    "exports": "POKER_RETAIN_EXPORTS_DAYS",
    "roi_previews": "POKER_RETAIN_ROI_PREVIEWS_DAYS",
    "videos": "POKER_RETAIN_ORPHAN_VIDEOS_DAYS",
}

# Defaults in days. Generous, because the cost of keeping a stale frame is disk
# and the cost of deleting a needed one is unrecoverable evidence.
DEFAULT_RETENTION_DAYS: dict[str, int] = {
    "frames": 30,
    "cv_timelines": 90,
    "job_logs": 30,
    "exports": 90,
    "roi_previews": 30,
    # Orphan videos are only ever removed on explicit opt-in; the window still
    # applies so a video orphaned seconds ago by an in-flight edit is safe.
    "videos": 365,
}

# Backups rotate under poker_tracker.persistence.backup and are deliberately
# absent here: two components expiring the same directory on different rules is
# how a verified restore point disappears.
NEVER_MANAGED = frozenset({"backups", "data"})


@dataclass(frozen=True)
class RetentionPolicy:
    days: dict[str, int]
    include_orphan_videos: bool = False

    @classmethod
    def from_env(cls, *, include_orphan_videos: bool = False) -> RetentionPolicy:
        days = dict(DEFAULT_RETENTION_DAYS)
        for category, variable in RETENTION_ENV_VARS.items():
            raw = os.environ.get(variable, "").strip()
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{variable} must be an integer number of days, got {raw!r}"
                ) from exc
            if value < 0:
                raise ValueError(f"{variable} must not be negative, got {value}")
            days[category] = value
        return cls(days=days, include_orphan_videos=include_orphan_videos)

    def window_days(self, category: str) -> int:
        return self.days.get(category, DEFAULT_RETENTION_DAYS.get(category, 0))


@dataclass(frozen=True)
class AuditedFile:
    path: Path
    category: str
    size_bytes: int
    age_days: float
    referenced: bool
    deletable: bool
    reason: str


@dataclass
class StorageAudit:
    """What retention *would* remove, and why it would keep everything else."""

    files: list[AuditedFile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unreadable_references: list[str] = field(default_factory=list)

    @property
    def deletable(self) -> list[AuditedFile]:
        return [f for f in self.files if f.deletable]

    @property
    def reclaimable_bytes(self) -> int:
        return sum(f.size_bytes for f in self.deletable)

    def by_category(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for entry in self.files:
            bucket = summary.setdefault(
                entry.category,
                {
                    "files": 0,
                    "bytes": 0,
                    "referenced": 0,
                    "deletable": 0,
                    "deletable_bytes": 0,
                },
            )
            bucket["files"] += 1
            bucket["bytes"] += entry.size_bytes
            if entry.referenced:
                bucket["referenced"] += 1
            if entry.deletable:
                bucket["deletable"] += 1
                bucket["deletable_bytes"] += entry.size_bytes
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": self.by_category(),
            "reclaimable_bytes": self.reclaimable_bytes,
            "deletable_files": [str(f.path) for f in self.deletable],
            "errors": list(self.errors),
            "unreadable_references": list(self.unreadable_references),
        }


def referenced_paths(db: PokerDatabase) -> tuple[set[Path], list[str]]:
    """Every filesystem path the database still points at, resolved.

    Resolution matters: the database stores absolute paths, but a walk may reach
    the same file through a differently-spelled parent (a symlinked data root, a
    relative configuration). Comparing unresolved strings would call a
    referenced file an orphan.
    """
    raw_paths, unreadable = db.referenced_artifact_paths()
    found: set[Path] = set()
    for raw in raw_paths:
        try:
            found.add(Path(raw).resolve())
        except OSError:
            continue
    return found, unreadable


def _managed_directories(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        name: path
        for name, path in paths.items()
        if name not in NEVER_MANAGED and name in RETENTION_ENV_VARS
    }


def audit_storage(
    db: PokerDatabase,
    paths: dict[str, Path],
    policy: RetentionPolicy | None = None,
    *,
    now: float | None = None,
) -> StorageAudit:
    """Classify every managed artifact as referenced, too new, or deletable.

    This never deletes. It is the thing an operator reads before deciding.
    """
    rules = policy or RetentionPolicy.from_env()
    current = now if now is not None else time.time()
    audit = StorageAudit()
    references, unreadable = referenced_paths(db)
    audit.unreadable_references = unreadable

    for category, directory in _managed_directories(paths).items():
        if not directory.exists():
            continue
        window = rules.window_days(category)
        for root, _dirs, names in os.walk(directory, followlinks=False):
            for name in names:
                candidate = Path(root) / name
                try:
                    info = candidate.lstat()
                except OSError as exc:
                    audit.errors.append(f"{candidate}: {exc}")
                    continue
                try:
                    resolved = candidate.resolve()
                except OSError:
                    resolved = candidate
                referenced = resolved in references
                age_days = max(0.0, (current - info.st_mtime) / 86400.0)
                deletable, reason = _classify(
                    category=category,
                    referenced=referenced,
                    age_days=age_days,
                    window=window,
                    rules=rules,
                )
                audit.files.append(
                    AuditedFile(
                        path=candidate,
                        category=category,
                        size_bytes=info.st_size,
                        age_days=round(age_days, 2),
                        referenced=referenced,
                        deletable=deletable,
                        reason=reason,
                    )
                )

    if unreadable:
        # A reference source we could not read may well have named some of the
        # files above. Refuse to offer anything for deletion in that case.
        for index, entry in enumerate(audit.files):
            if entry.deletable:
                audit.files[index] = AuditedFile(
                    path=entry.path,
                    category=entry.category,
                    size_bytes=entry.size_bytes,
                    age_days=entry.age_days,
                    referenced=entry.referenced,
                    deletable=False,
                    reason=(
                        "held back: could not read "
                        f"{', '.join(unreadable)} to prove it is unreferenced"
                    ),
                )
    return audit


def _classify(
    *,
    category: str,
    referenced: bool,
    age_days: float,
    window: int,
    rules: RetentionPolicy,
) -> tuple[bool, str]:
    if referenced:
        return False, "referenced by the database"
    if category == "videos" and not rules.include_orphan_videos:
        return False, "source recording; orphan removal not requested"
    if age_days < window:
        return False, f"unreferenced but only {age_days:.1f}d old (window {window}d)"
    return True, f"unreferenced and {age_days:.1f}d old (window {window}d)"


@dataclass(frozen=True)
class RetentionOutcome:
    removed: list[Path]
    reclaimed_bytes: int
    failures: list[str]


def apply_retention(audit: StorageAudit, *, confirm: bool = False) -> RetentionOutcome:
    """Delete exactly what ``audit`` marked deletable, and nothing else.

    ``confirm`` must be passed explicitly. The default is a no-op so a caller
    that forgets it removes nothing rather than everything, and so the audit
    stays usable as a pure report.
    """
    if not confirm:
        return RetentionOutcome(removed=[], reclaimed_bytes=0, failures=[])
    removed: list[Path] = []
    failures: list[str] = []
    reclaimed = 0
    for entry in audit.deletable:
        try:
            entry.path.unlink()
        except OSError as exc:
            failures.append(f"{entry.path}: {exc}")
            continue
        removed.append(entry.path)
        reclaimed += entry.size_bytes
    return RetentionOutcome(
        removed=removed, reclaimed_bytes=reclaimed, failures=failures
    )
