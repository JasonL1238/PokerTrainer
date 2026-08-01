"""Read-only health checks for PokerTrainer's database and stored artifacts."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from poker_tracker.persistence.backup import EVIDENCE_CLASSES, ROUTINE, resolve_artifact_path
from poker_tracker.persistence.backup import PINNED_GLOB as _PINNED_GLOB
from poker_tracker.persistence.backup_inventory import (
    inventory_findings,
    inventory_path,
    load_inventory,
    timeline_dir_for,
    timeline_paths,
)
from poker_tracker.persistence.db import (
    SCHEMA_VERSION,
    PokerDatabase,
    inspect_schema_integrity,
)

CheckStatus = Literal["pass", "warning", "fail"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = Path(
    os.environ.get("POKER_DB_PATH", PROJECT_ROOT / "poker_tracker.db")
)
DEFAULT_DATA_DIR = Path(os.environ.get("POKER_DATA_DIR", PROJECT_ROOT / "data"))
# Taken from the retention classes rather than re-spelled here: a literal that
# drifted from the one backup_database writes is exactly what hid the
# pre-migration snapshot from this audit in the first place. Every evidence class
# is scanned, so a purpose added there is audited here without an edit.
BACKUP_GLOB = ROUTINE.glob
PINNED_GLOB = _PINNED_GLOB
DETAIL_LIMIT = 20


def _artifact_reference_columns() -> tuple[tuple[str, str], ...]:
    """Derive the audited path columns from the list retention already trusts.

    The audit used to carry its own list of three columns while
    ``ARTIFACT_PATH_COLUMNS`` named nine, so solver outputs, per-action frame
    provenance and regression fixtures could all dangle under a report that said
    every recorded artifact was present. Deriving removes the possibility rather
    than the instance: a column added there is audited here without an edit.

    The import-time check is on the pairing, which derivation cannot guarantee on
    its own -- a label that disagreed with its own SQL would silently audit a
    different set of rows than retention protects.
    """
    references: list[tuple[str, str]] = []
    for label, sql in PokerDatabase.ARTIFACT_PATH_COLUMNS:
        table, _, column = label.partition(".")
        if not table or not column or sql.strip() != f"SELECT {column} FROM {table}":
            raise RuntimeError(
                "ARTIFACT_PATH_COLUMNS entry cannot be audited: "
                f"{label!r} does not describe {sql!r}"
            )
        references.append((table, column))
    return tuple(references)


_ARTIFACT_REFERENCES: tuple[tuple[str, str], ...] = _artifact_reference_columns()

# Stable minimum contract shared by all supported PokerTrainer databases. Newer
# version-specific tables may be additive, but these tables and columns are
# required for the core completed-session workflow to be usable.
_MINIMUM_SCHEMA: dict[str, set[str]] = {
    "schema_metadata": {"key", "value"},
    "sessions": {"id", "name", "date_played"},
    "hands": {"id", "session_id", "hand_number", "review_status", "source_type"},
    "hand_players": {"id", "hand_id", "player_key"},
    "actions": {"id", "hand_id", "street", "action_index", "action_type"},
    "hand_reviews": {"id", "hand_id"},
    "coaching_reviews": {"id", "review_type"},
    "videos": {"id", "stored_path", "file_size_bytes"},
    "processing_jobs": {"id", "status", "video_id"},
    "extracted_frames": {"id", "video_id", "job_id", "image_path"},
}

# Columns a database is only required to have once it claims the schema version
# that introduced them. Keeping them out of _MINIMUM_SCHEMA lets a retained
# pre-v13 backup stay healthy; keying them on the stored version means a database
# stamped 13 without them is reported instead of silently passing.
_VERSIONED_SCHEMA: tuple[tuple[int, str, frozenset[str]], ...] = (
    (13, "hands", frozenset({"completion_status", "completion_evidence"})),
    (14, "videos", frozenset({"content_sha256"})),
)


@dataclass(frozen=True)
class CheckResult:
    """One independently actionable health-check result."""

    name: str
    status: CheckStatus
    message: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class HealthReport:
    """Complete read-only health report for one PokerTrainer data store."""

    database_path: str
    data_dir: str
    backup_dir: str
    checked_at: str
    checks: tuple[CheckResult, ...]

    @property
    def healthy(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(check.status == "warning" for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "database_path": self.database_path,
            "data_dir": self.data_dir,
            "backup_dir": self.backup_dir,
            "checked_at": self.checked_at,
            "healthy": self.healthy,
            "has_warnings": self.has_warnings,
            "checks": [asdict(check) for check in self.checks],
        }


def audit_data_health(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    backup_dir: str | Path | None = None,
    expected_schema_version: int | None = None,
    restore_backups: bool = True,
) -> HealthReport:
    """Audit the live SQLite file, recorded artifacts, and retained backups.

    The live database and backups are opened in SQLite read-only/query-only
    mode. SQLite may still maintain its WAL shared-memory sidecars while reading.
    A restore drill copies each backup into an isolated temporary database; it
    never overwrites the live database or any backup file.

    ``restore_backups`` defaults to True. An opt-in verification that nothing in
    the product ever asked for meant a snapshot truncated by a full disk stayed
    undetected until the day it was needed; the drill costs one temporary copy
    per snapshot, which is the cheapest thing in this report worth having.
    """

    database = _absolute_path(database_path)
    data = _absolute_path(data_dir)
    backups = (
        _absolute_path(backup_dir)
        if backup_dir is not None
        else data / "backups"
    )

    checks = list(
        _audit_database(
            database,
            data,
            expected_schema_version=expected_schema_version,
        )
    )
    checks.append(
        _audit_backups(
            backups,
            live_database=database,
            data_dir=data,
            expected_schema_version=expected_schema_version,
            restore_backups=restore_backups,
        )
    )
    return HealthReport(
        database_path=str(database),
        data_dir=str(data),
        backup_dir=str(backups),
        checked_at=datetime.now(UTC).isoformat(),
        checks=tuple(checks),
    )


def _absolute_path(value: str | Path) -> Path:
    path = Path(value)
    try:
        path = path.expanduser()
    except RuntimeError:
        # Preserve an unresolvable "~user" literally so the audit returns a
        # structured missing/inaccessible result instead of crashing.
        pass
    return path.absolute()


def _audit_database(
    database_path: Path,
    data_dir: Path,
    *,
    expected_schema_version: int | None,
) -> tuple[CheckResult, ...]:
    try:
        database_stat = database_path.stat()
    except FileNotFoundError:
        return (
            CheckResult(
                "database_file",
                "fail",
                "Database file does not exist.",
                (str(database_path),),
            ),
        )
    except OSError as exc:
        return (
            CheckResult(
                "database_file",
                "fail",
                "Database file cannot be inspected.",
                (f"{type(exc).__name__}: {exc}",),
            ),
        )
    if not stat.S_ISREG(database_stat.st_mode):
        return (
            CheckResult(
                "database_file",
                "fail",
                "Database path is not a regular file.",
                (str(database_path),),
            ),
        )

    results = [
        CheckResult(
            "database_file",
            "pass",
            f"Database file is present ({database_stat.st_size} bytes).",
        )
    ]
    try:
        with closing(_connect_read_only(database_path)) as connection:
            results.append(_quick_check(connection))
            results.append(_foreign_key_check(connection))
            results.append(_schema_check(connection, expected_schema_version))
            results.append(_schema_contract_check(connection))
            results.append(_attestation_corroboration_check(connection))
            results.append(_artifact_check(connection, database_path, data_dir))
            results.append(_timeline_check(connection, data_dir))
    except (OSError, sqlite3.Error) as exc:
        results.append(
            CheckResult(
                "database_open",
                "fail",
                "Database could not be audited in read-only mode.",
                (f"{type(exc).__name__}: {exc}",),
            )
        )
    return tuple(results)


def _is_wal_mode(database_path: Path) -> bool:
    """Read the file header's write-version byte; 2 means WAL. Opens nothing."""
    try:
        with database_path.open("rb") as handle:
            header = handle.read(20)
    except OSError:
        return False
    return len(header) >= 19 and header[18] == 2


def _open_snapshot_read_only(
    snapshot_path: Path, staging: Path
) -> sqlite3.Connection:
    """Audit a retained snapshot without writing anything beside it.

    SQLite cannot read a WAL-mode database at all without creating a ``-shm``
    sidecar -- ``mode=ro`` and ``PRAGMA query_only`` do not change that, because
    the shared-memory index is how WAL readers find committed frames. So the
    "read-only" backup audit was writing ``-shm``/``-wal`` pairs into the
    operator's backup directory and leaving them there, becoming an ongoing
    producer of exactly the orphaned sidecars data health reports on; and on a
    read-only or archival mount -- which PLAN requires backups to be able to live
    on -- the same open raised ``attempt to write a readonly database`` and an
    intact backup was reported as FAILED.

    ``backup_database`` writes new snapshots in ``journal_mode=DELETE`` for this
    reason, but that only covers snapshots this build writes: every snapshot an
    older build left behind is still WAL. Those are staged into the caller's
    temporary directory first, with any sidecars that legitimately belong to
    them, so the copy carries the sidecars and the original is only ever read.
    ``immutable=1`` would avoid the copy and is wrong here: it tells SQLite to
    ignore the ``-wal`` file, which would silently audit a stale prefix of a
    snapshot that still had committed frames outstanding.
    """
    if not _is_wal_mode(snapshot_path):
        return _connect_read_only(snapshot_path)
    staged = staging / f"staged-{snapshot_path.name}"
    shutil.copyfile(snapshot_path, staged)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{snapshot_path}{suffix}")
        if sidecar.exists():
            shutil.copyfile(sidecar, Path(f"{staged}{suffix}"))
    return _connect_read_only(staged)


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{database_path.resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=5,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _quick_check(connection: sqlite3.Connection) -> CheckResult:
    rows = [str(row[0]) for row in connection.execute("PRAGMA quick_check")]
    if rows == ["ok"]:
        return CheckResult("sqlite_quick_check", "pass", "SQLite quick check passed.")
    return CheckResult(
        "sqlite_quick_check",
        "fail",
        "SQLite reported structural corruption.",
        _limited_details(rows),
    )


def _foreign_key_check(connection: sqlite3.Connection) -> CheckResult:
    count = 0
    details: list[str] = []
    for row in connection.execute("PRAGMA foreign_key_check"):
        count += 1
        if len(details) < DETAIL_LIMIT:
            details.append(
                f"{row[0]} row {row[1]} references missing {row[2]} row "
                f"(constraint {row[3]})."
            )
    if count == 0:
        return CheckResult(
            "foreign_key_check",
            "pass",
            "No broken database relationships were found.",
        )
    return CheckResult(
        "foreign_key_check",
        "fail",
        f"Found {count} broken database relationship(s).",
        _details_with_hidden_count(details, count),
    )


def _attestation_corroboration_check(connection: sqlite3.Connection) -> CheckResult:
    """Every settlement attestation should have the correction row that made it.

    A settlement-assumption attestation is the one half of that mechanism that is
    NOT re-derived on read: the dependence is re-measured from the chips every
    time, but the operator's answer to it is a string in a column, and
    ``study_readiness.unattested_assumption_dependence`` trusts it. The codes are
    deterministic -- a blake2s over the declared and neutral policy text, the
    gross pot, the declared dead money and the formatted movement -- so a
    hand-edited database can compute one without ever having seen the product,
    and that hand reads back study-ready with an empty blocker tuple.

    Storing a human assertion means the assertion can be forged; that is true of
    every attestation in this product and is not fixable by re-derivation. What
    IS checkable is corroboration: ``db.acknowledge_accounting_assumption`` writes
    a ``hand_corrections`` row naming the code in the same transaction, so an
    attestation with no such row was not written by this product. A warning, not
    a failure -- a hand imported from an older build, or one whose correction
    history was pruned, is unproven rather than proven false.
    """
    for table in ("hands", "hand_corrections"):
        if not _table_exists(connection, table):
            return CheckResult(
                "settlement_attestations",
                "warning",
                "Settlement attestations could not be audited.",
                (f"missing table: {table}",),
            )
    details: list[str] = []
    for row in connection.execute(
        "SELECT id, completion_evidence FROM hands "
        "WHERE completion_evidence LIKE '%confirmed_assumption_codes%'"
    ):
        try:
            evidence = json.loads(row["completion_evidence"] or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(evidence, dict):
            continue
        codes = evidence.get("confirmed_assumption_codes")
        if not isinstance(codes, list):
            continue
        recorded = "\n".join(
            str(item[0] or "")
            for item in connection.execute(
                "SELECT notes FROM hand_corrections WHERE hand_id = ?", (row["id"],)
            )
        )
        for code in codes:
            if isinstance(code, str) and code and code not in recorded:
                details.append(f"hand {row['id']}: {code}")
    if not details:
        return CheckResult(
            "settlement_attestations",
            "pass",
            "Every stored settlement attestation has a matching correction record.",
        )
    return CheckResult(
        "settlement_attestations",
        "warning",
        f"{len(details)} settlement attestation(s) have no correction record.",
        _limited_details(details),
    )


def _schema_check(
    connection: sqlite3.Connection,
    expected_schema_version: int | None,
) -> CheckResult:
    if not _table_exists(connection, "schema_metadata"):
        return CheckResult(
            "schema_version",
            "fail",
            "PokerTrainer schema metadata is missing.",
        )
    row = connection.execute(
        "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return CheckResult(
            "schema_version",
            "fail",
            "PokerTrainer schema version is missing.",
        )
    try:
        version = int(row[0])
    except (TypeError, ValueError):
        return CheckResult(
            "schema_version",
            "fail",
            "PokerTrainer schema version is not an integer.",
            (repr(row[0]),),
        )
    if version <= 0:
        return CheckResult(
            "schema_version",
            "fail",
            "PokerTrainer schema version must be positive.",
            (str(version),),
        )
    if expected_schema_version is not None and version > expected_schema_version:
        return CheckResult(
            "schema_version",
            "fail",
            "Database schema is newer than this PokerTrainer build supports.",
            (f"database={version}", f"supported={expected_schema_version}"),
        )
    if expected_schema_version is not None and version < expected_schema_version:
        return CheckResult(
            "schema_version",
            "warning",
            "Database schema is older than this build and needs migration.",
            (f"database={version}", f"current={expected_schema_version}"),
        )
    return CheckResult(
        "schema_version",
        "pass",
        f"PokerTrainer schema version {version} is readable.",
    )


def _stored_schema_version(connection: sqlite3.Connection) -> int | None:
    """The version the database claims, or None when it is missing or unreadable."""
    if not _table_exists(connection, "schema_metadata"):
        return None
    row = connection.execute(
        "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return None
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return None


def _schema_contract_check(connection: sqlite3.Connection) -> CheckResult:
    problems: list[str] = []
    required: dict[str, set[str]] = {
        table: set(columns) for table, columns in _MINIMUM_SCHEMA.items()
    }
    stored_version = _stored_schema_version(connection)
    if stored_version is not None:
        for introduced_in, table, columns in _VERSIONED_SCHEMA:
            if stored_version >= introduced_in:
                required.setdefault(table, set()).update(columns)
    for table, required_columns in required.items():
        if not _table_exists(connection, table):
            problems.append(f"missing table: {table}")
            continue
        columns = _table_columns(connection, table)
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            problems.append(
                f"{table}: missing column(s) {', '.join(missing_columns)}"
            )
    if problems:
        return CheckResult(
            "schema_contract",
            "fail",
            f"Core schema is incomplete in {len(problems)} place(s).",
            _limited_details(problems),
        )
    # The contract above names tables and columns only, and its version list
    # stops wherever someone last remembered to extend it, so a database missing
    # idx_actions_hand_street_order -- the unique index migration 8 exists to
    # create -- reported "present" while duplicate action order was being
    # written. inspect_schema_integrity derives the requirement from a database
    # this build creates, so it cannot fall behind the schema.
    #
    # Reported as a warning rather than a failure because opening the file is
    # what repairs it: init_db recreates every missing index and restores a
    # column its DDL can default, and refuses outright over anything else. The
    # audit reads without opening, so what it adds here is naming the damage
    # before the operator meets it, not blocking on it.
    #
    # Only meaningful for a file already at the current version: an older one
    # legitimately lacks later additions, and a newer one is not ours to judge.
    derived: list[str] = []
    if stored_version == SCHEMA_VERSION:
        report = inspect_schema_integrity(connection, check_foreign_keys=False)
        derived.extend(f"missing table: {name}" for name in report.missing_tables)
        derived.extend(f"missing column: {name}" for name in report.missing_columns)
        derived.extend(f"missing index: {name}" for name in report.missing_indexes)
    if derived:
        return CheckResult(
            "schema_contract",
            "warning",
            f"Core tables and columns are present, but this database is missing "
            f"{len(derived)} structure(s) the current schema declares; opening it "
            f"restores what can be restored and refuses over the rest.",
            _limited_details(derived),
        )
    return CheckResult(
        "schema_contract",
        "pass",
        "Core PokerTrainer tables and columns are present.",
    )


def _artifact_check(
    connection: sqlite3.Connection,
    database_path: Path,
    data_dir: Path,
) -> CheckResult:
    problem_details: list[str] = []
    missing_count = 0
    size_mismatch_count = 0
    reference_count = 0
    absent_sources: list[str] = []

    for table, path_column in _ARTIFACT_REFERENCES:
        required_columns = {path_column}
        if table == "videos":
            required_columns.add("file_size_bytes")
        if not _table_has_columns(connection, table, required_columns):
            # An older shape genuinely has no such column. Saying so is the
            # difference between "nine references all present" and "three of nine
            # looked at", which is the whole failure this check used to have.
            absent_sources.append(f"{table}.{path_column}")
            continue
        extra = ", file_size_bytes" if table == "videos" else ""
        # rowid, not a named id column: every one of these tables declares
        # INTEGER PRIMARY KEY, and rowid keeps the query derivable from the
        # column list alone rather than from a second table-to-key mapping.
        query = f"SELECT rowid AS _rowid, {path_column}{extra} FROM {table}"
        for row in connection.execute(query):
            raw = row[path_column]
            # Several of these columns default to the empty string, meaning "no
            # file recorded" rather than "a file that should be there";
            # referenced_artifact_paths skips them for the same reason.
            if not isinstance(raw, str) or not raw.strip():
                continue
            reference_count += 1
            stored_path = raw.strip()
            resolved = resolve_artifact_path(stored_path, database_path, data_dir)
            label = f"{table} row {row['_rowid']}"
            try:
                is_file = resolved.is_file()
            except OSError as exc:
                missing_count += 1
                _append_limited(
                    problem_details,
                    f"{label}: cannot inspect {stored_path}: {exc}",
                )
                continue
            if not is_file:
                missing_count += 1
                _append_limited(problem_details, f"{label}: missing {stored_path}")
                continue
            if table == "videos":
                try:
                    expected_size = int(row["file_size_bytes"])
                    actual_size = resolved.stat().st_size
                except (OSError, TypeError, ValueError) as exc:
                    size_mismatch_count += 1
                    _append_limited(
                        problem_details,
                        f"{label}: cannot verify size: {exc}",
                    )
                    continue
                if actual_size != expected_size:
                    size_mismatch_count += 1
                    _append_limited(
                        problem_details,
                        f"{label}: expected {expected_size} bytes, found {actual_size}"
                    )

    problem_count = missing_count + size_mismatch_count
    if problem_count:
        return CheckResult(
            "artifact_files",
            "fail",
            (
                f"Found {missing_count} missing artifact(s) and "
                f"{size_mismatch_count} video size mismatch(es)."
            ),
            _details_with_hidden_count(problem_details, problem_count),
        )
    scope = ""
    if absent_sources:
        scope = f" ({len(absent_sources)} source column(s) absent from this schema)"
    return CheckResult(
        "artifact_files",
        "pass",
        f"All {reference_count} recorded artifact reference(s) are present{scope}.",
        tuple(f"not in this schema: {source}" for source in absent_sources),
    )


def _timeline_check(connection: sqlite3.Connection, data_dir: Path) -> CheckResult:
    """Reconstruction timelines are referenced by convention, not by a column.

    Nothing in the database points at ``cv_timelines/job_<id>_timeline.json``, so
    the artifact check cannot see it, yet a completed job whose timeline is gone
    can no longer have any of its remaining hands imported. A warning rather than
    a failure: once every hand from a job has been imported the timeline is
    genuinely disposable, and retention is entitled to remove it.
    """
    if not _table_has_columns(
        connection, "processing_jobs", {"id", "job_type", "status"}
    ):
        return CheckResult(
            "timeline_files",
            "warning",
            "Reconstruction timelines could not be audited.",
            ("processing_jobs is missing the columns that name a CV job",),
        )
    details: list[str] = []
    missing_count = 0
    checked = 0
    for stored in timeline_paths(connection, timeline_dir_for(data_dir)):
        checked += 1
        try:
            present = Path(stored).is_file()
        except OSError as exc:
            present = False
            stored = f"{stored} ({exc})"
        if not present:
            missing_count += 1
            _append_limited(details, stored)
    if missing_count:
        return CheckResult(
            "timeline_files",
            "warning",
            f"{missing_count} completed reconstruction timeline(s) are missing.",
            _details_with_hidden_count(details, missing_count),
        )
    return CheckResult(
        "timeline_files",
        "pass",
        f"All {checked} completed reconstruction timeline(s) are present.",
    )


@dataclass(frozen=True)
class _StoreCounts:
    """How much study history a database actually holds."""

    sessions: int
    hands: int
    completed_hands: int

    def __str__(self) -> str:
        return (
            f"{self.sessions} session(s), {self.hands} hand(s), "
            f"{self.completed_hands} completed"
        )


@dataclass(frozen=True)
class _BackupOutcome:
    """One snapshot's verification result, and what it was found to contain."""

    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: str


def _audit_backups(
    backup_dir: Path,
    *,
    live_database: Path,
    data_dir: Path,
    expected_schema_version: int | None,
    restore_backups: bool,
) -> CheckResult:
    try:
        backup_dir_stat = backup_dir.stat()
    except FileNotFoundError:
        return CheckResult(
            "backups",
            "warning",
            "Backup directory does not exist yet.",
            (str(backup_dir),),
        )
    except OSError as exc:
        return CheckResult(
            "backups",
            "fail",
            "Backup directory cannot be inspected.",
            (f"{type(exc).__name__}: {exc}",),
        )
    if not stat.S_ISDIR(backup_dir_stat.st_mode):
        return CheckResult(
            "backups",
            "fail",
            "Backup path is not a directory.",
            (str(backup_dir),),
        )

    try:
        with os.scandir(backup_dir) as entries:
            names = sorted(Path(entry.path) for entry in entries)
    except OSError as exc:
        return CheckResult(
            "backups",
            "fail",
            "Backup directory cannot be read.",
            (f"{type(exc).__name__}: {exc}",),
        )
    rotating = [path for path in names if fnmatch.fnmatchcase(path.name, BACKUP_GLOB)]
    # Evidence snapshots -- pre-migration and pre-import -- sit outside BACKUP_GLOB
    # so the routine rotation cannot delete them. That also made them invisible
    # here: right after a migration the report said no backups existed about a
    # directory holding exactly the one snapshot that can undo it, and the restore
    # drill never opened it, so a truncated rollback point stayed undetectable
    # until it was needed. They are stamped at the state that preceded the
    # operation, which is why they are verified and restore-drilled without an
    # expected-version comparison.
    pinned = [
        path
        for path in names
        if any(
            fnmatch.fnmatchcase(path.name, snapshot_class.glob)
            for snapshot_class in EVIDENCE_CLASSES
        )
    ]
    backups = [*rotating, *pinned]
    if not backups:
        return CheckResult(
            "backups",
            "warning",
            "No retained PokerTrainer backups were found.",
            (str(backup_dir),),
        )

    live_counts = _live_counts(live_database)
    failures: list[str] = []
    warnings: list[str] = []
    summaries: list[str] = []
    for backup, expected in (
        *((path, expected_schema_version) for path in rotating),
        *((path, None) for path in pinned),
    ):
        outcome = _backup_issues(
            backup,
            live_database=live_database,
            data_dir=data_dir,
            expected_schema_version=expected,
            restore=restore_backups,
            live_counts=live_counts,
        )
        failures.extend(outcome.failures)
        warnings.extend(outcome.warnings)
        summaries.append(f"{backup.name}: {outcome.summary}")
    # A snapshot from an older build carries no inventory, so nothing can say
    # which recordings, frames, timelines and solver outputs it needs. Counting
    # them here states that limitation instead of leaving "passed a restore
    # drill" to imply an answer nobody has.
    inventoried = sum(1 for path in backups if inventory_path(path).is_file())
    summaries.append(
        f"artifact inventory: {inventoried} of {len(backups)} snapshot(s) carry one"
    )
    if live_counts is not None:
        # The counts to compare the snapshots against, so the operator is not
        # asked to look them up by hand as the runbook used to require.
        summaries.insert(0, f"live database: {live_counts}")

    action = "passed a restore drill" if restore_backups else "were read"
    if failures:
        return CheckResult(
            "backups",
            "fail",
            f"{len(failures)} backup verification problem(s) were found.",
            _limited_details([*failures, *summaries]),
        )
    if warnings:
        return CheckResult(
            "backups",
            "warning",
            (
                f"All {len(backups)} retained backup(s) {action} "
                f"with {len(warnings)} warning(s)."
            ),
            _limited_details([*warnings, *summaries]),
        )
    return CheckResult(
        "backups",
        "pass",
        f"All {len(backups)} retained backup(s) {action}.",
        _limited_details(summaries),
    )


def verify_snapshot(
    snapshot_path: str | Path,
    *,
    live_database: str | Path | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    expected_schema_version: int | None = None,
) -> CheckResult:
    """Restore one snapshot into an isolated temporary root and read it back.

    The caller is whatever just wrote the snapshot. A rollback point is worth
    exactly what it can be shown to restore, and the moment it was written is the
    only moment at which a caller can still refuse to proceed; a verification
    that waits for an operator to remember a flag verifies nothing for months.

    Never touches the live database or the snapshot: the restore target is a
    file inside a temporary directory that is removed on the way out.
    """
    snapshot = _absolute_path(snapshot_path)
    live = _absolute_path(live_database) if live_database is not None else None
    outcome = _backup_issues(
        snapshot,
        live_database=live,
        data_dir=_absolute_path(data_dir),
        expected_schema_version=expected_schema_version,
        restore=True,
        live_counts=None if live is None else _live_counts(live),
    )
    if outcome.failures:
        return CheckResult(
            "backup_verification",
            "fail",
            f"{snapshot.name} did not survive an isolated restore.",
            _limited_details([*outcome.failures, outcome.summary]),
        )
    if outcome.warnings:
        return CheckResult(
            "backup_verification",
            "warning",
            f"{snapshot.name} restored with {len(outcome.warnings)} warning(s).",
            _limited_details([*outcome.warnings, outcome.summary]),
        )
    return CheckResult(
        "backup_verification",
        "pass",
        f"{snapshot.name} passed an isolated restore ({outcome.summary}).",
    )


def _backup_issues(
    backup_path: Path,
    *,
    live_database: Path | None,
    data_dir: Path,
    expected_schema_version: int | None,
    restore: bool,
    live_counts: _StoreCounts | None,
) -> _BackupOutcome:
    if backup_path.is_symlink():
        return _BackupOutcome(
            (f"{backup_path.name}: symlinks are not independent backups",),
            (),
            "not verified",
        )
    try:
        backup_stat = backup_path.stat()
        if not stat.S_ISREG(backup_stat.st_mode):
            return _BackupOutcome(
                (f"{backup_path.name}: not a regular file",), (), "not verified"
            )
        same_as_live = False
        if live_database is not None:
            try:
                same_as_live = os.path.samefile(backup_path, live_database)
            except FileNotFoundError:
                same_as_live = False
        if same_as_live or backup_stat.st_nlink > 1:
            return _BackupOutcome(
                (f"{backup_path.name}: hard-linked files are not independent backups",),
                (),
                "not verified",
            )
        warnings = [
            f"{backup_path.name}: {finding}"
            for finding in _inventory_warnings(
                # A relative artifact path is stored relative to the live
                # database; with no live database named, the snapshot's own
                # location is the closest thing to that root.
                backup_path,
                database_path=live_database or backup_path,
                data_dir=data_dir,
            )
        ]
        with tempfile.TemporaryDirectory(prefix="pokertrainer-audit-") as temp_dir:
            staging = Path(temp_dir)
            with closing(_open_snapshot_read_only(backup_path, staging)) as source:
                failures, connection_warnings = _connection_issues(
                    source,
                    expected_schema_version=expected_schema_version,
                )
                warnings.extend(
                    f"{backup_path.name}: {warning}" for warning in connection_warnings
                )
                if failures or not restore:
                    return _BackupOutcome(
                        tuple(
                            f"{backup_path.name}: {failure}" for failure in failures
                        ),
                        tuple(warnings),
                        "not restored" if failures else "read without restoring",
                    )
                restored_path = staging / "restored.sqlite3"
                with closing(sqlite3.connect(restored_path)) as restored:
                    source.backup(restored)
                with closing(_connect_read_only(restored_path)) as restored:
                    restore_failures, _ = _connection_issues(
                        restored,
                        expected_schema_version=None,
                    )
                    content = _restored_content_issues(
                        restored, live_counts=live_counts
                    )
            return _BackupOutcome(
                tuple(
                    f"{backup_path.name} restore: {failure}"
                    for failure in [*restore_failures, *content.failures]
                ),
                tuple(
                    [
                        *warnings,
                        *(
                            f"{backup_path.name} restore: {warning}"
                            for warning in content.warnings
                        ),
                    ]
                ),
                content.summary,
            )
    except (OSError, sqlite3.Error) as exc:
        return _BackupOutcome(
            (f"{backup_path.name}: {type(exc).__name__}: {exc}",), (), "not verified"
        )


def _inventory_warnings(
    backup_path: Path, *, database_path: Path, data_dir: Path
) -> list[str]:
    """What the snapshot's artifact inventory says about the files it needs now.

    A snapshot written by an older build has no inventory. That is reported in
    the check's message as a count rather than as a warning per file: the absence
    is a limitation of the snapshot, not a defect in the data, and drowning the
    real findings would defeat the point.
    """
    inventory = load_inventory(backup_path)
    if inventory is None:
        return []
    return inventory_findings(
        inventory, database_path=database_path, data_dir=data_dir
    )


def _restored_content_issues(
    connection: sqlite3.Connection, *, live_counts: _StoreCounts | None
) -> _BackupOutcome:
    """Verify the restored copy holds usable study history, not merely valid pages.

    quick_check and foreign_key_check prove the file is a well-formed database.
    They pass just as happily on an empty one, so a snapshot taken from a
    truncated file was reported as having "passed a restore drill". What a
    recovery point has to be able to do is hand back the history: countable
    sessions and hands, issue evidence that still parses, and at least one hand
    that reads end to end -- its players, its actions and its settlement.
    """
    failures: list[str] = []
    warnings: list[str] = []
    counts = _counts_from(connection)
    if counts is None:
        return _BackupOutcome(
            ("sessions and hands could not be counted",), (), "unreadable"
        )
    if counts.hands == 0 and live_counts is not None and live_counts.hands > 0:
        warnings.append(
            f"holds no hands while the live database holds {live_counts.hands}; "
            "this snapshot cannot restore any study history"
        )
    issue_failures, issue_warnings = _issue_evidence_issues(connection)
    failures.extend(issue_failures)
    warnings.extend(issue_warnings)
    attestations = _attestation_corroboration_check(connection)
    if attestations.status == "warning":
        warnings.append(f"settlement_attestations: {attestations.message}")
    hand_failures, hand_warnings = _hand_readback_issues(connection)
    failures.extend(hand_failures)
    warnings.extend(hand_warnings)
    return _BackupOutcome(tuple(failures), tuple(warnings), str(counts))


def _issue_evidence_issues(
    connection: sqlite3.Connection,
) -> tuple[list[str], list[str]]:
    """Frozen issue evidence is the only record of what a flagged hand looked like.

    It is written as JSON by ``create_hand_issue`` and never re-derived, so an
    unparseable snapshot is a corrupt one and an empty snapshot is an issue that
    can no longer be reproduced at all.
    """
    if not _table_has_columns(connection, "hand_issues", {"id", "evidence_snapshot"}):
        return [], []
    failures: list[str] = []
    warnings: list[str] = []
    for row in connection.execute("SELECT id, evidence_snapshot FROM hand_issues"):
        try:
            evidence = json.loads(row["evidence_snapshot"] or "")
        except (TypeError, ValueError):
            _append_limited(
                failures, f"issue {row['id']}: evidence_snapshot is not readable JSON"
            )
            continue
        if not isinstance(evidence, dict):
            _append_limited(
                failures, f"issue {row['id']}: evidence_snapshot is not an object"
            )
        elif not evidence:
            _append_limited(warnings, f"issue {row['id']}: evidence_snapshot is empty")
    return failures, warnings


def _hand_readback_issues(
    connection: sqlite3.Connection,
) -> tuple[list[str], list[str]]:
    """Read one hand end to end, preferring a completed one.

    The point is not the row count: it is that the tables a study session
    actually joins still join, and that the JSON columns the readers parse still
    parse, on the RESTORED copy rather than on the original.
    """
    if not _table_exists(connection, "hands"):
        return [], []
    columns = _table_columns(connection, "hands")
    order = (
        "ORDER BY CASE WHEN completion_status = 'complete' THEN 0 ELSE 1 END, id"
        if "completion_status" in columns
        else "ORDER BY id"
    )
    try:
        hand = connection.execute(f"SELECT * FROM hands {order} LIMIT 1").fetchone()
    except sqlite3.Error as exc:
        return [f"no hand could be read back: {exc}"], []
    if hand is None:
        return [], []
    hand_id = hand["id"]
    failures: list[str] = []
    warnings: list[str] = []
    if "completion_evidence" in columns:
        try:
            json.loads(hand["completion_evidence"] or "{}")
        except (TypeError, ValueError):
            failures.append(
                f"hand {hand_id}: completion_evidence is not readable JSON"
            )
    related = {
        "hand_players": "SELECT COUNT(*) FROM hand_players WHERE hand_id = ?",
        "actions": "SELECT COUNT(*) FROM actions WHERE hand_id = ?",
        "hand_settlements": "SELECT COUNT(*) FROM hand_settlements WHERE hand_id = ?",
    }
    present: dict[str, int] = {}
    for table, sql in related.items():
        if not _table_exists(connection, table):
            continue
        try:
            present[table] = int(connection.execute(sql, (hand_id,)).fetchone()[0])
        except (sqlite3.Error, TypeError, ValueError) as exc:
            failures.append(f"hand {hand_id}: {table} could not be read back: {exc}")
    for table in ("hand_players", "actions"):
        if present.get(table) == 0:
            warnings.append(
                f"hand {hand_id} restored with no {table} row; "
                "it would read back as an empty hand"
            )
    return failures, warnings


def _live_counts(database_path: Path) -> _StoreCounts | None:
    """The live counts a snapshot is compared against, or None when unreadable."""
    try:
        with closing(_connect_read_only(database_path)) as connection:
            return _counts_from(connection)
    except (OSError, sqlite3.Error):
        return None


def _counts_from(connection: sqlite3.Connection) -> _StoreCounts | None:
    try:
        sessions = int(connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        hands = int(connection.execute("SELECT COUNT(*) FROM hands").fetchone()[0])
        completed = 0
        if "completion_status" in _table_columns(connection, "hands"):
            completed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM hands WHERE completion_status = 'complete'"
                ).fetchone()[0]
            )
    except (sqlite3.Error, TypeError, ValueError):
        return None
    return _StoreCounts(sessions=sessions, hands=hands, completed_hands=completed)


def _connection_issues(
    connection: sqlite3.Connection,
    *,
    expected_schema_version: int | None,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    quick_rows = [str(row[0]) for row in connection.execute("PRAGMA quick_check")]
    if quick_rows != ["ok"]:
        failures.extend(f"quick_check: {item}" for item in quick_rows)
    foreign_count = 0
    for row in connection.execute("PRAGMA foreign_key_check"):
        foreign_count += 1
        if len(failures) < DETAIL_LIMIT:
            failures.append(
                f"foreign_key_check: {row[0]} row {row[1]} references {row[2]}"
            )
    if foreign_count > DETAIL_LIMIT:
        failures.append(
            f"foreign_key_check: ... and {foreign_count - DETAIL_LIMIT} more"
        )

    schema = _schema_check(connection, expected_schema_version)
    if schema.status == "fail":
        failures.append(f"schema_version: {schema.message}")
        failures.extend(f"schema_version: {detail}" for detail in schema.details)
    elif schema.status == "warning":
        warnings.append(f"schema_version: {schema.message}")
        warnings.extend(f"schema_version: {detail}" for detail in schema.details)

    contract = _schema_contract_check(connection)
    if contract.status == "fail":
        failures.append(f"schema_contract: {contract.message}")
        failures.extend(f"schema_contract: {detail}" for detail in contract.details)
    return failures, warnings


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_has_columns(
    connection: sqlite3.Connection,
    table: str,
    required_columns: set[str],
) -> bool:
    if not _table_exists(connection, table):
        return False
    return required_columns <= _table_columns(connection, table)


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _append_limited(details: list[str], value: str) -> None:
    if len(details) < DETAIL_LIMIT:
        details.append(value)


def _details_with_hidden_count(
    visible_details: Sequence[str],
    total_count: int,
) -> tuple[str, ...]:
    details = list(visible_details)
    hidden = total_count - len(details)
    if hidden > 0:
        details.append(f"... and {hidden} more")
    return tuple(details)


def _limited_details(details: Sequence[str]) -> tuple[str, ...]:
    values = list(details)
    if len(values) <= DETAIL_LIMIT:
        return tuple(values)
    hidden = len(values) - DETAIL_LIMIT
    return (*values[:DETAIL_LIMIT], f"... and {hidden} more")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit PokerTrainer's local database, artifacts, and backups."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument(
        "--restore-backups",
        action="store_true",
        default=True,
        help=(
            "Restore each retained backup into an isolated temporary directory "
            "and verify it (default)."
        ),
    )
    parser.add_argument(
        "--no-restore-backups",
        dest="restore_backups",
        action="store_false",
        help="Only open each retained backup; skip the isolated restore drill.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = audit_data_health(
        args.db,
        data_dir=args.data_dir,
        backup_dir=args.backup_dir,
        expected_schema_version=SCHEMA_VERSION,
        restore_backups=args.restore_backups,
    )
    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_report(report)
    return 0 if report.healthy else 1


def _print_report(report: HealthReport) -> None:
    state = "HEALTHY" if report.healthy else "UNHEALTHY"
    print(f"PokerTrainer data health: {state}")
    print(f"Database: {report.database_path}")
    print(f"Data directory: {report.data_dir}")
    print(f"Backup directory: {report.backup_dir}")
    for check in report.checks:
        print(f"[{check.status.upper()}] {check.name}: {check.message}")
        for detail in check.details:
            print(f"  - {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
