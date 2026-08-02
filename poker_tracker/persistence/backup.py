"""SQLite-consistent snapshots retained before migrations and CV imports.

Stdlib-only on purpose: ``db.py`` takes a consistent snapshot before migrating a
real file database, and ``poker_tracker.ui.run_cv_job`` (which imports
persistence) re-exports this helper, so a cycle would exist if the function
stayed in the UI package.

Retention follows from what a snapshot is FOR, not from when it arrived. A
snapshot taken immediately before an operation the product cannot undo -- a
schema migration, a batch of validated-hand imports -- is evidence: the state it
holds is the state that existed before that operation, and nothing else can
reproduce it. A routine periodic copy is fungible; the newest few are as good as
any. Each purpose therefore gets its own retention pool, so a burst of one kind
of snapshot can never evict the rollback point of another kind.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("POKER_DATA_DIR", PROJECT_ROOT / "data"))
BACKUPS_DIR = DATA_DIR / "backups"
# The exact stamp `backup_database` writes: %Y%m%dT%H%M%S%fZ. Rotation matches on
# this, not on a glob alone, because the product must never delete a rollback
# point it did not write. `poker_tracker_manual_keepme.sqlite3` -- an operator's
# own snapshot, in the operator's own backup directory -- matches
# `poker_tracker_*.sqlite3` and was being evicted alongside the product's.
_STAMP = r"\d{8}T\d{6}\d{6}Z"
# A scope narrows an evidence snapshot to the one operation it precedes, so
# "have we already snapshotted for this job" is answerable from the directory
# listing alone, with no marker file and no schema column.
_SCOPE = r"[a-z0-9]+"


@dataclass(frozen=True)
class SnapshotClass:
    """One retention pool: a purpose, the names it writes, and how many survive."""

    purpose: str
    prefix: str
    keep: int
    name: re.Pattern[str]
    scoped: bool = False

    @property
    def glob(self) -> str:
        return f"{self.prefix}*.sqlite3"


BACKUP_KEEP_COUNT = 5
# Evidence snapshots are exempt from the routine rotation, not from retention.
# A migration that fails is retried on the next start and snapshots the whole
# database again each time, so with no cap at all a failing upgrade filled the
# operator's backup mount with full copies.
PINNED_KEEP_COUNT = 5
PREIMPORT_KEEP_COUNT = 5
PREDELETE_KEEP_COUNT = 5

ROUTINE = SnapshotClass(
    purpose="routine",
    prefix="poker_tracker_",
    keep=BACKUP_KEEP_COUNT,
    name=re.compile(rf"^poker_tracker_{_STAMP}\.sqlite3$"),
)
PREMIGRATION = SnapshotClass(
    purpose="premigration",
    prefix="poker_tracker-premigration-",
    keep=PINNED_KEEP_COUNT,
    name=re.compile(rf"^poker_tracker-premigration-{_STAMP}\.sqlite3$"),
)
PREIMPORT = SnapshotClass(
    purpose="preimport",
    prefix="poker_tracker-preimport-",
    keep=PREIMPORT_KEEP_COUNT,
    name=re.compile(rf"^poker_tracker-preimport-{_SCOPE}-{_STAMP}\.sqlite3$"),
    scoped=True,
)
PREDELETE = SnapshotClass(
    purpose="predelete",
    prefix="poker_tracker-predelete-",
    keep=PREDELETE_KEEP_COUNT,
    name=re.compile(rf"^poker_tracker-predelete-{_SCOPE}-{_STAMP}\.sqlite3$"),
    scoped=True,
)
SNAPSHOT_CLASSES: tuple[SnapshotClass, ...] = (
    ROUTINE,
    PREMIGRATION,
    PREIMPORT,
    PREDELETE,
)
# Everything that is not a routine periodic copy is a rollback point for an
# operation that already happened; the audit treats the two groups differently.
EVIDENCE_CLASSES: tuple[SnapshotClass, ...] = tuple(
    item for item in SNAPSHOT_CLASSES if item is not ROUTINE
)
_BY_PURPOSE = {item.purpose: item for item in SNAPSHOT_CLASSES}

# Retained spellings: db.py calls backup_database(pinned=True) and both the audit
# and the adversarial suite import these names.
PINNED_PREFIX = PREMIGRATION.prefix
PINNED_GLOB = PREMIGRATION.glob

# The database the product runs against. Must agree with db.DEFAULT_DB_PATH,
# which cannot be imported here -- db.py imports this module. A test pins the
# two together so the duplication cannot drift.
LIVE_DB_PATH = Path(
    os.environ.get(
        "POKER_DB_PATH", str(Path(__file__).resolve().parent.parent.parent / "poker_tracker.db")
    )
)


def backups_dir_for(db_path: Path, *, live_db_path: Path | None = None) -> Path:
    """Where a snapshot of ``db_path`` belongs.

    A backup is a rollback point for one particular database, so it has to live
    with that database rather than in whichever directory the constant happens
    to name. Opening a temporary fixture, a restored copy, or a backup being
    audited used to write a pre-migration snapshot into the operator's real
    ``data/backups`` and evict a genuine rollback point of the live database --
    a snapshot of a throwaway file competing for the slots that protect the
    study history.

    The live database keeps BACKUPS_DIR, which is where operators, the runbooks
    and the restore drill all look for it.
    """
    live = LIVE_DB_PATH if live_db_path is None else live_db_path
    try:
        same = db_path.resolve() == live.resolve()
    except OSError:
        same = str(db_path) == str(live)
    return BACKUPS_DIR if same else db_path.parent / "backups"


def backup_database(
    db_path: Path,
    backup_dir: Path | None = None,
    *,
    pinned: bool = False,
    purpose: str | None = None,
    scope: str | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Create a SQLite-consistent, self-contained snapshot and retain its pool.

    ``backup_dir`` defaults to ``backups_dir_for(db_path)``, resolved at call time
    so a test or an operator can redirect snapshots without rotating the real
    retained set, and so a snapshot of a database that is not the live one never
    competes for the live one's slots.

    ``purpose`` selects the retention pool. ``routine`` snapshots share the
    five-slot rotation; ``premigration`` and ``preimport`` snapshots are evidence
    of a state that no longer exists anywhere else -- a pre-migration snapshot is
    the only artifact that can undo an irreversible migration, and v13 forces
    ``review_status = 'needs_correction'`` on every reconstructed hand -- so each
    keeps its own slots. ``pinned=True`` is the pre-migration spelling ``db.py``
    uses.

    ``predelete`` is the same argument applied to the product's own destructive
    controls. Deleting a session removes its hands, actions, reviews, corrections
    and settlement rows, and nothing else in the product can bring any of it
    back; the snapshot taken immediately before is the only artifact that can, so
    it keeps its own slots rather than competing with a routine copy.

    ``scope`` names the single operation an evidence snapshot precedes (a CV job,
    say). It becomes part of the filename, which is what lets a caller ask
    ``find_snapshots`` whether that operation has already been snapshotted
    instead of taking one copy of the database per imported row. A ``predelete``
    scope names the row being removed instead, and is deliberately NOT used to
    skip a second snapshot: two deletions are two different states to roll back
    to.

    An artifact inventory is written beside the snapshot: the rows are only half
    the study history, and the recordings, frames, timelines and solver outputs
    they point at are deliberately not copied. ``data_dir`` is the root those
    artifacts are recorded against; it defaults to DATA_DIR, and a caller that
    already resolved its own data root should pass it rather than rely on the
    environment agreeing.

    The snapshot is left in ``journal_mode=delete``. ``Connection.backup()`` copies
    page 1 of the live database, so it would otherwise inherit ``wal`` -- and
    opening a WAL database read-only requires creating a ``-shm`` sidecar, which
    requires write permission on the containing directory. That would make the
    read-only backup audit and the restore drill fail on a read-only or archival
    mount, on backups that are actually intact.
    """
    if str(db_path) == ":memory:":
        raise ValueError("Cannot back up an in-memory database.")
    snapshot_class = _resolve_class(purpose, pinned=pinned)
    backup_dir = backups_dir_for(db_path) if backup_dir is None else backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / _snapshot_name(snapshot_class, scope)
    source = sqlite3.connect(str(db_path))
    target = sqlite3.connect(str(destination))
    try:
        source.backup(target)
        # Collapse any inherited WAL into the main file, then leave the snapshot in
        # a journal mode that needs no sidecar to be read.
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target.execute("PRAGMA journal_mode=DELETE")
        target.commit()
    finally:
        # Closed explicitly: `with sqlite3.connect(...)` commits but never closes,
        # which used to leave -shm/-wal sidecars behind that rotation never removed.
        target.close()
        source.close()
    _remove_sidecars(destination)
    _write_inventory(destination, Path(db_path), data_dir)
    _rotate(backup_dir, snapshot_class)
    return destination


def find_snapshots(
    backup_dir: Path, *, purpose: str, scope: str | None = None
) -> list[Path]:
    """Snapshots of one purpose (optionally one scope), newest first.

    Answering "has this operation already been snapshotted?" from the retained
    set is what keeps a batch import to one copy of the database: the marker and
    the rollback point are the same file, so they cannot disagree.
    """
    snapshot_class = _class_for_purpose(purpose)
    if scope is not None and not snapshot_class.scoped:
        raise ValueError(f"{purpose} snapshots carry no scope.")
    try:
        candidates = list(backup_dir.glob(snapshot_class.glob))
    except OSError:
        return []
    prefix = (
        snapshot_class.prefix if scope is None else f"{snapshot_class.prefix}{scope}-"
    )
    matched = [
        path
        for path in candidates
        if snapshot_class.name.match(path.name) and path.name.startswith(prefix)
    ]
    return sorted(matched, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def classify_snapshot(name: str) -> SnapshotClass | None:
    """The pool a filename belongs to, or None when this product did not write it."""
    for snapshot_class in SNAPSHOT_CLASSES:
        if snapshot_class.name.match(name):
            return snapshot_class
    return None


def resolve_artifact_path(
    stored_path: str, database_path: Path, data_dir: Path
) -> Path:
    """Turn a recorded artifact path into the file it means on this machine.

    One definition, shared by the snapshot inventory and the health audit: a
    stored path that resolved one way for the inventory and another way for the
    audit would make a present file look missing, or worse.
    """
    path = Path(stored_path).expanduser()
    if path.is_absolute():
        return path
    candidates = (database_path.parent / path, data_dir / path)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return candidates[0]


def _resolve_class(purpose: str | None, *, pinned: bool) -> SnapshotClass:
    if purpose is None:
        return PREMIGRATION if pinned else ROUTINE
    snapshot_class = _class_for_purpose(purpose)
    if pinned and snapshot_class is ROUTINE:
        raise ValueError("A routine snapshot cannot also be pinned.")
    return snapshot_class


def _class_for_purpose(purpose: str) -> SnapshotClass:
    try:
        return _BY_PURPOSE[purpose]
    except KeyError:
        raise ValueError(
            f"Unknown snapshot purpose {purpose!r}; "
            f"expected one of {sorted(_BY_PURPOSE)}."
        ) from None


def _snapshot_name(snapshot_class: SnapshotClass, scope: str | None) -> str:
    """Build the name, then prove it against the pattern rotation matches on.

    Checking the constructed name against the class's own regex is what stops a
    snapshot from being written under a name its rotation will never see, which
    is the one mistake here that produces unbounded growth in silence.
    """
    if scope is not None and not re.fullmatch(_SCOPE, scope):
        raise ValueError(
            f"Snapshot scope {scope!r} must match {_SCOPE} so the filename stays parseable."
        )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    middle = "" if scope is None else f"{scope}-"
    name = f"{snapshot_class.prefix}{middle}{stamp}.sqlite3"
    if not snapshot_class.name.match(name):
        expectation = "require a scope" if snapshot_class.scoped else "take no scope"
        raise ValueError(
            f"{snapshot_class.purpose} snapshots {expectation}: {name!r}"
        )
    return name


def _write_inventory(
    snapshot: Path, database_path: Path, data_dir: Path | None
) -> None:
    """Record what the snapshot's rows point at, never raising into the caller.

    Imported here rather than at module scope: the inventory needs the database
    package's artifact-column list, and ``db.py`` imports this module to snapshot
    before a migration. A failed inventory is written as an inventory carrying
    the failure -- the audit reports that -- because the snapshot itself is the
    thing that must not be lost, and ``db.py`` aborts a migration when this call
    raises.
    """
    from poker_tracker.persistence import backup_inventory

    backup_inventory.write_inventory(
        snapshot, database_path=database_path, data_dir=data_dir
    )


def _rotate(backup_dir: Path, snapshot_class: SnapshotClass) -> None:
    """Retain the newest ``keep`` snapshots this product wrote for one purpose.

    The class's name pattern is the real filter; its glob only narrows the
    listing. A glob alone cannot tell one of our snapshots from an operator's own
    file that happens to share the shape, and deleting one of those would break
    the promise that nothing here removes a rollback point it did not write.
    """
    snapshots = sorted(
        (
            path
            for path in backup_dir.glob(snapshot_class.glob)
            if snapshot_class.name.match(path.name)
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for old in snapshots[snapshot_class.keep :]:
        old.unlink(missing_ok=True)
        _remove_sidecars(old)
        _remove_inventory(old)


def _remove_sidecars(database: Path) -> None:
    """Delete the -shm/-wal companions of a snapshot that no longer needs them."""
    for suffix in ("-shm", "-wal"):
        Path(f"{database}{suffix}").unlink(missing_ok=True)


def _remove_inventory(database: Path) -> None:
    """An inventory outliving its snapshot describes a restore point that is gone."""
    from poker_tracker.persistence.backup_inventory import inventory_path

    inventory_path(database).unlink(missing_ok=True)
