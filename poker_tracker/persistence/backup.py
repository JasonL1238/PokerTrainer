"""SQLite-consistent snapshots retained before migrations and CV imports.

Stdlib-only on purpose: ``db.py`` takes a consistent snapshot before migrating a
real file database, and ``poker_tracker.ui.run_cv_job`` (which imports
persistence) re-exports this helper, so a cycle would exist if the function
stayed in the UI package.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("POKER_DATA_DIR", PROJECT_ROOT / "data"))
BACKUPS_DIR = DATA_DIR / "backups"
BACKUP_KEEP_COUNT = 5
# Pinned snapshots are never rotated away. The prefix deliberately does not match
# the rotation glob below.
PINNED_PREFIX = "poker_tracker-premigration-"
# Pinned snapshots are exempt from the rotation above, not from retention: they
# get their own slots so a per-import snapshot can never delete a rollback point,
# while a repeatedly failing migration cannot fill the backup mount either.
PINNED_KEEP_COUNT = 5
PINNED_GLOB = f"{PINNED_PREFIX}*.sqlite3"
# The exact stamp `backup_database` writes: %Y%m%dT%H%M%S%fZ. Rotation matches on
# this, not on the glob alone, because the product must never delete a rollback
# point it did not write. `poker_tracker_manual_keepme.sqlite3` -- an operator's
# own snapshot, in the operator's own backup directory -- matches
# `poker_tracker_*.sqlite3` and was being evicted alongside the product's.
_STAMP = r"\d{8}T\d{6}\d{6}Z"
_ROTATING_NAME = re.compile(rf"^poker_tracker_{_STAMP}\.sqlite3$")
_PINNED_NAME = re.compile(rf"^{re.escape(PINNED_PREFIX)}{_STAMP}\.sqlite3$")


def backup_database(
    db_path: Path, backup_dir: Path | None = None, *, pinned: bool = False
) -> Path:
    """Create a SQLite-consistent, self-contained snapshot and retain the newest five.

    ``backup_dir`` defaults to BACKUPS_DIR, resolved at call time so a test or an
    operator can redirect snapshots without rotating the real retained set.

    ``pinned=True`` writes the snapshot under a name the rotation never matches.
    A pre-migration snapshot is the only artifact that can undo an irreversible
    migration -- v13 forces ``review_status = 'needs_correction'`` on every
    reconstructed hand -- so it must not compete for retention slots with routine
    per-import snapshots, five of which would otherwise delete it.

    The snapshot is left in ``journal_mode=delete``. ``Connection.backup()`` copies
    page 1 of the live database, so it would otherwise inherit ``wal`` -- and
    opening a WAL database read-only requires creating a ``-shm`` sidecar, which
    requires write permission on the containing directory. That would make the
    read-only backup audit and the restore drill fail on a read-only or archival
    mount, on backups that are actually intact.
    """
    # The rotating filename pattern is a contract with
    # maintenance.data_health.BACKUP_GLOB; a non-matching name makes the snapshot
    # invisible to the backup audit. A pinned pre-migration snapshot is
    # deliberately outside it: it is stamped at the OLD schema version, which the
    # audit is entitled to treat as a failure on a rotating snapshot.
    if str(db_path) == ":memory:":
        raise ValueError("Cannot back up an in-memory database.")
    backup_dir = BACKUPS_DIR if backup_dir is None else backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = PINNED_PREFIX if pinned else "poker_tracker_"
    destination = backup_dir / f"{prefix}{stamp}.sqlite3"
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
    if pinned:  # noqa: SIM108 - the two branches differ in more than the argument
        # Pinned snapshots keep their own slots. Exempting them from the five-slot
        # rotation is what stops a routine per-import snapshot from deleting the
        # only rollback point; it must not also mean "unbounded". A migration that
        # fails is retried on the next start and snapshots the whole database
        # again each time, so with no cap at all a failing upgrade filled the
        # operator's backup mount with full copies.
        _rotate(backup_dir, PINNED_GLOB, PINNED_KEEP_COUNT, _PINNED_NAME)
    else:
        _rotate(
            backup_dir, "poker_tracker_*.sqlite3", BACKUP_KEEP_COUNT, _ROTATING_NAME
        )
    return destination


def _rotate(
    backup_dir: Path, pattern: str, keep: int, written_by_us: re.Pattern[str]
) -> None:
    """Retain the newest ``keep`` snapshots this product wrote.

    ``written_by_us`` is the real filter; ``pattern`` only narrows the listing.
    A glob alone cannot tell one of our snapshots from an operator's own file
    that happens to share the shape, and deleting one of those would break the
    promise that nothing here removes a rollback point it did not write.
    """
    snapshots = sorted(
        (path for path in backup_dir.glob(pattern) if written_by_us.match(path.name)),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for old in snapshots[keep:]:
        old.unlink(missing_ok=True)
        _remove_sidecars(old)


def _remove_sidecars(database: Path) -> None:
    """Delete the -shm/-wal companions of a snapshot that no longer needs them."""
    for suffix in ("-shm", "-wal"):
        Path(f"{database}{suffix}").unlink(missing_ok=True)
