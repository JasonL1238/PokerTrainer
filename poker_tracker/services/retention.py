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

Two consequences of that rule shape the code below more than anything else.

First, "the database still points at this file" is a question about file
identity, not about spelling. macOS ships a case-insensitive filesystem, so
``Session.MOV`` on disk and ``session.mov`` in SQLite are one file that two
strings describe; comparing the strings reports an orphan and deletes live
evidence. Every comparison in this module therefore goes through
:func:`path_identity_keys`, which prefers ``(st_dev, st_ino)`` — the only answer
that is true regardless of spelling — and falls back to a normalized, case-
folded textual key only when the file cannot be stat'd. The fallback deliberately
over-matches: mistaking two files for one keeps an orphan, mistaking one file for
two deletes something live.

Second, an audit is a proposal and never an authorization. A CV job that
completes between the audit and the sweep makes a row point at a file the audit
already classified as an orphan, so :func:`apply_retention` re-confirms every
path against the database through the same :class:`ReferenceCheck` the audit
used, immediately before the unlink.

Source videos are deliberately never expired by age. They are the irreplaceable
input; every derived artifact can be rebuilt from them and they cannot be
rebuilt from anything. Only an unreferenced *orphan* video is ever offered, and
only through an explicit opt-in.

One limit is worth stating plainly, because the obvious reading of the output is
wrong: the age shown is the FILE's mtime, not how long it has been unreferenced.
Nothing records when a row stopped pointing at a file, so a recording orphaned
one second ago by a session delete still reports whatever age it was written
with. The reason text for videos says so rather than implying a dormancy the
data cannot demonstrate.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Sequence
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
    # Orphan videos are only ever removed on explicit opt-in. The window is
    # measured against file mtime, which is NOT when the file became
    # unreferenced -- see the module docstring.
    "videos": 365,
}

# Backups rotate under poker_tracker.persistence.backup and are deliberately
# absent here: two components expiring the same directory on different rules is
# how a verified restore point disappears.
NEVER_MANAGED = frozenset({"backups", "data"})

# The data root is in NEVER_MANAGED because it is not a category of its own, not
# because its subtree is off limits -- every managed directory lives inside it.
# The other never-managed names are subtrees nothing here may walk.
DATA_ROOT_KEY = "data"


# --- Path identity ----------------------------------------------------------
#
# One helper, used by every comparison in this module. Sprinkling normcase at
# call sites is how half the comparisons end up correct and the other half
# delete things.


def _resolved_text(path: Path | str) -> str:
    """The most normalized spelling of ``path`` this process can produce.

    Absolute, with ``.``/``..`` collapsed and symlinks followed as far as they
    exist. It does not fix letter case: ``realpath`` on macOS returns whatever
    case the caller supplied even though the filesystem ignores it, which is
    precisely why the textual key below case-folds on top of this.
    """
    try:
        return os.path.realpath(os.fspath(path))
    except OSError:
        return os.path.abspath(os.path.normpath(os.fspath(path)))


def _textual_key(path: Path | str) -> str:
    """A spelling-insensitive key for a path that may not exist.

    Case-folded unconditionally, including on case-sensitive filesystems where
    two spellings really can be two files. That over-matches by design: the
    worst outcome is keeping an orphan, and the alternative under-matches on the
    filesystem this product actually runs on.
    """
    return os.path.normcase(_resolved_text(path)).casefold()


def path_identity_keys(path: Path | str) -> frozenset[tuple[Any, ...]]:
    """Every key under which ``path`` names one particular file.

    Two paths refer to the same file when their key sets intersect. Where the
    file exists the ``(st_dev, st_ino)`` key settles it outright — it survives
    case folding, symlinks, hard links and non-normalized spellings alike. The
    textual key is kept as well so that a path whose file has since vanished can
    still be recognized by name.
    """
    keys: set[tuple[Any, ...]] = {("name", _textual_key(path))}
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        pass
    else:
        keys.add(("file", info.st_dev, info.st_ino))
    return frozenset(keys)


def _is_within(child: Path | str, parent: Path | str) -> bool:
    """Whether ``child`` is ``parent`` or sits underneath it, spelling aside."""
    child_key = _textual_key(child)
    parent_key = _textual_key(parent)
    return child_key == parent_key or child_key.startswith(
        os.path.join(parent_key, "")
    )


# --- Policy -----------------------------------------------------------------


def _validate_window(category: str, value: int, *, purge_immediately: bool) -> int:
    """Reject a window that would purge on the next sweep without saying so.

    A zero reaches this from an operator typo or from a shell that expanded an
    unset variable to something ``int()`` happily coerced. "Retain for zero days"
    then removes every managed artifact, which is too destructive a behavior to
    be one keystroke away from a correct configuration. Purging now is a real
    operator need, so it has its own explicit request instead.
    """
    if purge_immediately:
        return value
    if value <= 0:
        variable = RETENTION_ENV_VARS.get(category, "the retention window")
        raise ValueError(
            f"{variable} must be a positive number of days, got {value}. "
            f"A window of {value} would delete every unreferenced artifact in "
            f"{category} on the next --apply. To purge unreferenced artifacts "
            "deliberately, pass --purge-now, which says so out loud."
        )
    return value


@dataclass(frozen=True)
class RetentionPolicy:
    days: dict[str, int]
    include_orphan_videos: bool = False
    # Windows are ignored and every unreferenced managed artifact is eligible.
    # Reachable only by asking for it by name; see _validate_window.
    purge_immediately: bool = False

    def __post_init__(self) -> None:
        # Validating here rather than only in from_env keeps a directly
        # constructed policy from carrying a window from_env would have refused.
        object.__setattr__(self, "days", dict(self.days))
        for category, value in sorted(self.days.items()):
            _validate_window(
                category, value, purge_immediately=self.purge_immediately
            )

    @classmethod
    def from_env(
        cls,
        *,
        include_orphan_videos: bool = False,
        purge_immediately: bool = False,
    ) -> RetentionPolicy:
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
            days[category] = value
        return cls(
            days=days,
            include_orphan_videos=include_orphan_videos,
            purge_immediately=purge_immediately,
        )

    def window_days(self, category: str) -> int:
        """The window in days, revalidated on every read.

        ``days`` is a plain dict, so a caller can still write a zero into it
        after construction. Checking at the point of use means no path through
        this object can hand out a window that silently purges.
        """
        if self.purge_immediately:
            return 0
        value = self.days.get(category, DEFAULT_RETENTION_DAYS.get(category, 0))
        return _validate_window(category, value, purge_immediately=False)


# --- The one reference check ------------------------------------------------


@dataclass(frozen=True)
class AuditedFile:
    path: Path
    category: str
    size_bytes: int
    age_days: float
    referenced: bool
    deletable: bool
    reason: str


class ReferenceCheck:
    """The single answer to "does the database still point at this file?".

    The audit and the deletion loop both ask this object, because they must ask
    the same question the same way and because the audit's answer expires. A CV
    reconstruction that finishes while an operator reads the plan makes a row
    point at a file the plan calls an orphan; deleting on the strength of the
    plan destroys a live frame. So the snapshot is rebuilt whenever the database
    has changed underneath it, and :meth:`confirm_unreferenced` — the only thing
    that may authorize an unlink — checks that immediately before each one.

    Detecting the change cheaply matters: rebuilding unconditionally would scan
    every artifact-path column once per deleted file. ``total_changes()`` covers
    writes on this connection and ``PRAGMA data_version`` covers writes on any
    other, so between them a stale snapshot cannot go unnoticed. If either is
    unavailable the token is ``None``, which forces a rebuild every time: slower,
    never wrong.
    """

    def __init__(
        self, db: PokerDatabase, *, anchors: Sequence[Path | str] = ()
    ) -> None:
        self._db = db
        self._anchors = [Path(anchor) for anchor in anchors]
        self._keys: set[tuple[Any, ...]] = set()
        self._unreadable: list[str] = []
        self._token: tuple[int, int] | None = None
        self.refresh()

    def _reference_keys(self, raw: str) -> frozenset[tuple[Any, ...]]:
        """Identity keys for one stored reference, however it was spelled.

        A relative reference has no single meaning: resolving it against the
        working directory is what the process happens to be doing, not what the
        row meant. data_health already reads them against the database directory
        and the data root, so this recognizes all of those spellings at once
        rather than deleting a file because the sweep ran from elsewhere.
        """
        stored = Path(raw).expanduser()
        keys = path_identity_keys(stored)
        if not stored.is_absolute():
            for anchor in self._anchors:
                keys |= path_identity_keys(anchor / stored)
        return keys

    @property
    def unreadable(self) -> list[str]:
        """Reference sources that could not be read, ever, during this check.

        Sticky on purpose. A source that failed once may have named the file in
        front of us, and a later successful read cannot prove it did not.
        """
        return list(self._unreadable)

    def refresh(self) -> None:
        raw_paths, unreadable = self._db.referenced_artifact_paths()
        keys: set[tuple[Any, ...]] = set()
        for raw in raw_paths:
            keys |= self._reference_keys(raw)
        self._keys = keys
        for label in unreadable:
            if label not in self._unreadable:
                self._unreadable.append(label)
        self._token = self._change_token()

    def _change_token(self) -> tuple[int, int] | None:
        try:
            changes = self._db._execute("SELECT total_changes()").fetchone()[0]
            data_version = self._db._execute("PRAGMA data_version").fetchone()[0]
        except (sqlite3.Error, TypeError, IndexError):
            return None
        return int(changes), int(data_version)

    def _refresh_if_changed(self) -> None:
        token = self._change_token()
        if token is None or token != self._token:
            self.refresh()

    def is_referenced(self, path: Path | str) -> bool:
        """Whether the snapshot points at ``path``, by identity rather than name."""
        return bool(path_identity_keys(path) & self._keys)

    def confirm_unreferenced(self, path: Path | str) -> str | None:
        """Re-read the database and say why ``path`` must be kept, or ``None``.

        ``None`` is the only thing in this module that authorizes an unlink.
        """
        self._refresh_if_changed()
        if self._unreadable:
            return (
                "could not read "
                f"{', '.join(self._unreadable)} to prove it is unreferenced"
            )
        if self.is_referenced(path):
            return "the database now references it"
        return None


@dataclass
class StorageAudit:
    """What retention *would* remove, and why it would keep everything else."""

    files: list[AuditedFile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unreadable_references: list[str] = field(default_factory=list)
    # The live check this audit was computed from. apply_retention re-asks it
    # before every unlink; an audit that arrives without one cannot authorize a
    # deletion, which is why the default refuses rather than assuming a database.
    reference_check: ReferenceCheck | None = field(
        default=None, repr=False, compare=False
    )

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


def _managed_directories(paths: dict[str, Path]) -> dict[str, Path]:
    """The directories retention may walk, with never-managed subtrees removed.

    The name filter is the first line; the containment check behind it is the
    one that holds when a data root is laid out unusually, because a category
    directory that happens to resolve inside the backups directory would
    otherwise let retention expire a restore point through the side door.
    """
    forbidden = [
        path
        for name, path in paths.items()
        if name in NEVER_MANAGED and name != DATA_ROOT_KEY
    ]
    managed: dict[str, Path] = {}
    for name, path in paths.items():
        if name in NEVER_MANAGED or name not in RETENTION_ENV_VARS:
            continue
        if any(_is_within(path, blocked) for blocked in forbidden):
            continue
        managed[name] = path
    return managed


def _reference_anchors(db: PokerDatabase, paths: dict[str, Path]) -> list[Path]:
    """Directories a relative reference could have been written against."""
    anchors = [Path(db.db_path).expanduser().parent]
    root = paths.get(DATA_ROOT_KEY)
    if root is not None:
        anchors.append(Path(root))
    return anchors


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
    check = ReferenceCheck(db, anchors=_reference_anchors(db, paths))
    audit = StorageAudit(reference_check=check)

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
                referenced = check.is_referenced(candidate)
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

    audit.unreadable_references = check.unreadable
    if audit.unreadable_references:
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
                        f"{', '.join(audit.unreadable_references)} to prove it "
                        "is unreferenced"
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
        return False, f"unreferenced but file is only {age_days:.1f}d old (window {window}d)"
    # The age is the FILE's, not how long it has been unreferenced -- nothing
    # records when a row stopped pointing at it. Saying "unused for N days"
    # would be a claim this cannot support: a recording orphaned one second ago
    # by a session delete still carries whatever mtime it was written with.
    if category == "videos":
        return True, (
            f"no database row references it; file is {age_days:.1f}d old. "
            "This does NOT mean it has been unused that long -- it may have "
            "been orphaned moments ago."
        )
    if rules.purge_immediately:
        return True, f"unreferenced, file is {age_days:.1f}d old (purge requested)"
    return True, f"unreferenced, file is {age_days:.1f}d old (window {window}d)"


@dataclass(frozen=True)
class RetentionOutcome:
    removed: list[Path]
    reclaimed_bytes: int
    failures: list[str]
    # Files the audit offered that the re-check refused to delete. Distinct from
    # failures: nothing went wrong, the plan simply went stale.
    skipped: list[str] = field(default_factory=list)


def apply_retention(audit: StorageAudit, *, confirm: bool = False) -> RetentionOutcome:
    """Delete what ``audit`` marked deletable, re-proving each one first.

    ``confirm`` must be passed explicitly. The default is a no-op so a caller
    that forgets it removes nothing rather than everything, and so the audit
    stays usable as a pure report.

    The audit is a proposal. Every unlink is authorized by a fresh reference
    check against the database, taken through the same :class:`ReferenceCheck`
    that produced the proposal, so a file that became referenced while the
    operator was reading the plan is skipped rather than destroyed.
    """
    if not confirm:
        return RetentionOutcome(removed=[], reclaimed_bytes=0, failures=[])
    check = audit.reference_check
    if check is None:
        return RetentionOutcome(
            removed=[],
            reclaimed_bytes=0,
            failures=[
                "refusing to delete: this audit carries no live reference check, "
                "so no deletion can be re-proved against the database"
            ],
        )
    removed: list[Path] = []
    failures: list[str] = []
    skipped: list[str] = []
    reclaimed = 0
    for entry in audit.deletable:
        refusal = check.confirm_unreferenced(entry.path)
        if refusal is not None:
            skipped.append(f"{entry.path}: {refusal}")
            continue
        try:
            entry.path.unlink()
        except OSError as exc:
            failures.append(f"{entry.path}: {exc}")
            continue
        removed.append(entry.path)
        reclaimed += entry.size_bytes
    return RetentionOutcome(
        removed=removed,
        reclaimed_bytes=reclaimed,
        failures=failures,
        skipped=skipped,
    )
