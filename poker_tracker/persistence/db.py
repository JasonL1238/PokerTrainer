from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from ast import literal_eval
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

from pydantic import ValidationError

from poker_tracker.math.accounting import (
    LedgerError,
    RakePolicy,
    blind_structure,
    build_ledger_from_records,
)
from poker_tracker.math.cards import CardParseError, parse_visible_cards
from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence.backup import backup_database
from poker_tracker.persistence.completion import (
    DERIVED_EVIDENCE_KEYS,
    OPERATOR_MANUAL_COMPLETION_KEY,
    UNREADABLE_CARDS_KEY,
    UNREADABLE_HAND_COLUMNS_KEY,
    confirm_assumption,
    derive_completion_status,
    dump_completion_evidence,
    has_operator_manual_completion,
    is_assumption_dependence_code,
    parse_completion_evidence,
    requires_assumption_attestation,
    set_declared_settlement_code,
    strip_derived_evidence_markers,
    strip_operator_attestation,
)
from poker_tracker.persistence.models import (
    RELEASE_BLOCKING_ISSUE_TYPES,
    Action,
    CoachingResponse,
    CompletionStatus,
    ExtractedFrame,
    Hand,
    HandCorrection,
    HandIssue,
    HandIssueType,
    HandPlayer,
    HandReview,
    HandSettlement,
    PersistedModel,
    ProcessingJob,
    ReconstructionFrameReview,
    ReviewStatus,
    ROIProfile,
    ROIRegion,
    Session,
    SettlementEntry,
    SolverRangeProfile,
    SolverRun,
    SourceType,
    StudyInclusion,
    VideoRecord,
)
from poker_tracker.persistence.validation import CardValidationError, normalize_cards
from poker_tracker.safety.redaction import redact_text
from poker_tracker.ui.roi import validate_roi_bounds

# Anchored to the project root so launching from another directory does not
# silently create a second, empty database.
DEFAULT_DB_PATH = os.environ.get(
    "POKER_DB_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "poker_tracker.db"),
)
SCHEMA_VERSION = 20
_PROCESSING_JOB_PID_UNSET = object()
# A migration on a real database can outlast SQLite's 5s default, and a second
# opener must wait for it rather than failing startup with "database is locked".
BUSY_TIMEOUT_MS = int(os.environ.get("POKER_DB_BUSY_TIMEOUT_MS", "30000"))
# Recorded in a reconstructed hand's completion evidence when its source facts are
# corrected. An acknowledgeable warning, not a rejection: the operator changed a
# fact and may attest to the result, which is exactly what the completion
# blocker's clearing action tells them to do.
SOURCE_CORRECTION_CODE = "source_facts_corrected"
# Recorded when a debugging issue demotes a hand. Same contract: the demotion has
# to live in the evidence, or the column disagrees with its own evidence forever
# and the Source warnings panel -- which only renders when a code is present --
# never gives the operator the action the completion blocker names.
DEBUGGING_FLAG_CODE = "flagged_for_debugging"
# Recorded when a reconstructed hand's settlement declares chips that no observed
# action accounts for. Dead money is a legitimate modelling input -- antes, dead
# blinds, a straddle from a seat that left -- and the ledger models it faithfully.
# It is also the one free parameter that can always be tuned until the recorded
# pot matches the derived one, so on a hand the PIPELINE reconstructed it is an
# operator assertion the evidence cannot corroborate, and the reconciled verdict
# must not rest on it silently.
DECLARED_DEAD_MONEY_CODE = "declared_unobserved_chips"
# Recorded when a reconstructed hand's settlement declares a rake policy that
# actually takes chips. The mirror image of the code above: dead money CREATES
# chips the observed action line never saw, and a rake DESTROYS them. Both are
# legitimate modelling inputs the ledger models faithfully, and both are free
# parameters the operator (or an import payload) can tune until the recorded
# figures match the derived ones -- the rake by moving the DERIVED side of the
# cross-check, which no amount of comparing exactly can detect. A reconstructed
# hand's hero result is an observation read off the source; the rake taken from
# its pot is not observed at all, so the reconciled verdict must not rest on it
# silently.
#
# Both codes above are raised when the declaration actually MOVES CHIPS on this
# hand, measured by `_declared_chips_taken`, never from a list of fields that
# look suspicious. They are the writer-side audit trail and they are NOT
# acknowledgeable: they live in `completion_evidence.declared_settlement_codes`,
# the operator's own channel, because what the operator declared and what the
# pipeline could not prove are different claims and filing the first with the
# second demoted the reconstruction's completion status on the strength of a rake
# somebody typed in. The readiness gate is `ACCOUNTING_ASSUMPTION_DEPENDENT`,
# derived per read, which asks the stricter question of whether the hand's
# reconciliation SURVIVES those chips being removed and which no writer can
# bypass; the answer to it is an attestation to the measured movement.
DECLARED_RAKE_CODE = "declared_unobserved_rake"
# How a settlement row this build cannot validate is reported to the reader. It
# is a read-time annotation on a degraded row (see `_degraded_hand_settlement`),
# never a persisted fact: `upsert_hand_settlement` strips it, so a row rewritten
# through the editor cannot inherit the note that its predecessor was unreadable.
UNREADABLE_SETTLEMENT_PREFIX = "Stored settlement columns could not be read:"
# One definition of "a solver run a background worker still owns", shared by the
# active-run query and by delete_solver_run's refusal. They used to disagree about
# 'cancelling'.
_LIVE_SOLVER_STATUSES: tuple[str, ...] = ("queued", "running", "cancelling")

# The ``hands`` columns an IMPORT owns rather than the payload, and therefore the
# only columns ``restore_unreadable_columns`` will not write back. The completion
# trio is re-derived by ``_apply_completion_import_defaults`` and floored by
# ``_enforce_review_status_floor``; ``completion_evidence`` is the channel the
# unreadable-column marker itself travels in, so restoring it would persist the
# derivation the marker exists to keep derived; ``id``, ``session_id`` and
# ``created_at`` are the importing database's own identity and row-creation record,
# and a timestamp column is never at a restorable fallback anyway (the reader's
# degradation stamps a real time), so a marker naming one is reported by
# UNREADABLE_HAND_COLUMNS in the exporting database and re-derived here.
_EVIDENCE_OWNED_COLUMNS = frozenset(
    {
        "id",
        "session_id",
        "created_at",
        "review_status",
        "completion_status",
        "source_type",
        "completion_evidence",
        # Operator Study-queue preference. Default is the non-empty enum
        # ``auto``, never a blank restorable fallback, and restore must not
        # overwrite an intentional inclusion choice from a marker payload.
        "study_inclusion",
    }
)

# The stored forms of the model defaults every restorable ``hands`` column falls
# back to when the reader has to give it up. A restore only overwrites one of
# these, which is the round-8 guard ("a marker may not replace a readable card
# column") stated once for every column instead of once for the two card ones.
# ``test_every_restorable_hand_column_degrades_into_a_restorable_fallback`` walks
# ``Hand.model_fields`` and fails if a column is added whose default is not in
# here, so the restore cannot silently start skipping a column.
_RESTORABLE_FALLBACKS: tuple[str, ...] = ("", "[]")

# One rule, referenced from every site that used to break it.
#
# _MODEL_SPACE_CLASSIFICATION: no SQL predicate may classify a row that a
# ``_*_from_row`` reader reclassifies.
#
# Several readers in this module deliberately answer a different question from
# the raw column, always in the conservative direction: ``_hand_issue_from_row``
# forces ``status='open'`` on a row it cannot fully read, ``_review_from_row``
# and ``_coaching_response_from_row`` force ``is_stale=True``,
# ``_solver_run_from_row`` degrades ``completed`` to ``stale``,
# ``_hand_player_from_row`` decides heroism with ``bool(...)``, and
# ``_degraded_hand`` forces ``review_status='needs_correction'``. Readiness,
# the blockers, and every list view read the MODEL.
#
# A ``WHERE is_stale = 1`` / ``WHERE status = 'open'`` / ``WHERE is_hero = 1``
# predicate answers in the COLUMN's space instead, and the two spaces disagree on
# exactly the rows the readers exist for. The consequences were symmetric and
# both bad: a blocker whose named clearing action matched nothing and reported
# success anyway (``discard_stale_coaching`` on ``is_stale = 2``,
# ``resolve_hand_issue`` on ``status='in_progress'``), and a store-level floor
# blind to the very row it is the floor for (``update_hand_status``,
# ``_validate_single_hero``, ``fetch_cached_solver_run``).
#
# The fix is not a cleverer SQL predicate — SQLite's storage classes cannot
# express Python truthiness, so any translation is one more enumerated list to
# fall behind. Every site below selects its candidate rows by IDENTITY
# (``hand_id``, ``input_hash``, ``id``) and then classifies them through the
# same reader every consumer uses, so the writer clears exactly what the reader
# calls stale and the floor sees exactly what the blocker sees.
# ``test_no_sql_predicate_classifies_a_row_the_reader_reclassifies`` fails on a
# new raw-column classification predicate, so the family cannot grow back.


def _readable_schema_version(connection: sqlite3.Connection) -> int | None:
    """The stored schema version, 0 when unstamped, or None when unreadable.

    Never raises: it runs before the version check, on a file this build may be
    about to refuse, and before any pragma writes to it.
    """
    try:
        row = connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
    except sqlite3.DatabaseError:
        return 0  # fresh database: schema_metadata does not exist yet
    if row is None:
        return 0
    try:
        return int(str(row["value"]).strip())
    except (TypeError, ValueError):
        return None


# The artefact of the one migration that is not safe to replay. Migrations 6-12
# are additive DDL and idempotent row repair, so re-running them against a
# database that already carries them changes nothing. _migrate_to_v13 is
# different: it REWRITES completion_status and review_status for every
# reconstructed hand, so replaying it against a live database discards every
# operator confirmation. Its columns are therefore the discriminator that tells
# "a brand-new or pre-versioning database, which legitimately has no stamp" apart
# from "a migrated database whose stamp row was lost", which must be refused.
_DESTRUCTIVE_MIGRATION_VERSION = 13
_DESTRUCTIVE_MIGRATION_FEATURE = ("hands", "completion_status")


def _physical_schema_floor(connection: sqlite3.Connection) -> int:
    """The lowest schema version the file's own tables and columns could be.

    Never raises: it runs before the version check, on a file this build may be
    about to refuse, and before any pragma writes to it.
    """
    table, column = _DESTRUCTIVE_MIGRATION_FEATURE
    try:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.DatabaseError:
        return 0
    if rows and any(str(row[1]) == column for row in rows):
        return _DESTRUCTIVE_MIGRATION_VERSION
    return 0


def _physical_schema_ceiling(connection: sqlite3.Connection) -> int | None:
    """The highest version the file's own schema could be, when that is knowable.

    The mirror of ``_physical_schema_floor``. ``None`` means "no evidence": the
    ``hands`` table does not exist yet (a fresh or pre-versioning file), or it
    already carries the v13 artefact, in which case the floor governs.

    Never raises: it runs before the version check, on a file this build may be
    about to refuse, and before any pragma writes to it.
    """
    table, column = _DESTRUCTIVE_MIGRATION_FEATURE
    try:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.DatabaseError:
        return None
    if not rows:
        return None
    if any(str(row[1]) == column for row in rows):
        return None
    return _DESTRUCTIVE_MIGRATION_VERSION - 1


def _assert_schema_matches_stamp(version: int | None, ceiling: int | None) -> None:
    """Refuse a database whose version stamp is ahead of the schema it carries.

    The discriminator used to be one-sided: a schema ahead of its stamp was
    refused, a stamp ahead of its schema was not. A v13 stamp over a pre-v13
    ``hands`` table therefore opened, ``init_db`` saw ``stored_version ==
    SCHEMA_VERSION`` and skipped both the migration chain and the pre-migration
    snapshot, ``CREATE TABLE IF NOT EXISTS`` left the old table alone, and the
    database then failed every write with a bare ``sqlite3.OperationalError`` --
    with no backup taken, on a file the operator was never warned about. Reads
    fail safe (the hand degrades to ``uncertain`` and blocks), which is what kept
    this from being worse.

    Refused rather than repaired: this build cannot know whether the missing
    column means an interrupted migration, a partial hand-rebuild, or a restore
    that mixed two files, and re-running v13 against it would rewrite every
    reconstructed hand's review_status.
    """
    if version is None or ceiling is None or version <= ceiling:
        return
    raise RuntimeError(
        f"Database schema version stamp says {version} but the database is "
        f"physically at version {ceiling} or lower: it is missing structures "
        f"version {version} requires. Restore the database from a backup before "
        "opening it — this build cannot tell an interrupted migration from a "
        "partially rebuilt file, and writing to it would fail hand by hand. If "
        "this file holds no sessions of your own — an interrupted first start, "
        "or a scratch copy — delete it and start again; a new database is "
        "created automatically."
    )


def _assert_stamp_matches_schema(version: int | None, floor: int) -> None:
    """Refuse a database whose schema is ahead of the version stamp it carries.

    ``_readable_schema_version`` reports 0 for three different states: a genuinely
    fresh file, a file whose ``schema_metadata`` row was deleted, and a file whose
    ``schema_metadata`` table was dropped. Only the first may migrate. The other
    two used to replay the whole chain against a live database, silently knocking
    every reconstructed hand back to uncertain/needs_correction and destroying
    every ``reviewed`` confirmation -- while a stamp of ``'abc'`` or ``-1`` was
    already refused outright. A pre-versioning database is still migrated, because
    the discriminator is what the schema physically contains, not whether a stamp
    happens to be present.
    """
    if version is None or version >= floor:
        return
    raise RuntimeError(
        f"Database schema version stamp says {version} but the database already "
        f"contains schema version {floor} structures; its version stamp is "
        "missing or out of date. Restore the database from a backup before "
        "opening it — re-running the migrations against it would discard "
        "recorded review confirmations. If this file holds no sessions of your "
        "own — an interrupted first start, or a scratch copy — delete it and "
        "start again; a new database is created automatically."
    )


def _assert_supported_schema_version(version: int | None) -> None:
    """Refuse anything this build cannot safely migrate, with one clear message.

    A negative stamp is refused rather than clamped: ``range(max(v, 5) + 1, ...)``
    would replay the whole chain, and v13 re-run against a live database resets
    every operator confirmation.
    """
    if version is None:
        raise RuntimeError(
            "Database schema version stamp is not a readable version number. "
            "Restore the database from a backup before opening it."
        )
    if version < 0:
        raise RuntimeError(
            f"Database schema version {version} is not a valid version. "
            "Restore the database from a backup before opening it."
        )
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {version} is newer than this app "
            f"understands ({SCHEMA_VERSION}). Update the app before opening it."
        )


@dataclass(frozen=True)
class SchemaIntegrityReport:
    """Which structures a database lacks compared with the schema this build creates.

    Reported rather than raised, because the callers want different things:
    ``init_db`` restores what it can and refuses over the rest, while an audit
    wants to describe every gap at once so the operator sees the whole damage
    rather than the first item of it.

    The reference is the schema a fresh database of THIS build ends up with, so
    the report is only meaningful for a file that has been migrated to
    ``SCHEMA_VERSION``. A retained snapshot written by an older build
    legitimately lacks later additions; migrate a copy of it first, which is what
    an isolated restore does anyway.
    """

    missing_tables: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    missing_indexes: tuple[str, ...] = ()
    foreign_key_violations: tuple[str, ...] = ()

    @property
    def is_intact(self) -> bool:
        return not (
            self.missing_tables
            or self.missing_columns
            or self.missing_indexes
            or self.foreign_key_violations
        )

    def describe(self) -> str:
        """Name what is missing, so the message is actionable without a schema dump."""
        parts: list[str] = []
        for label, items in (
            ("missing table(s)", self.missing_tables),
            ("missing column(s)", self.missing_columns),
            ("missing index(es)", self.missing_indexes),
            ("foreign-key violation(s)", self.foreign_key_violations),
        ):
            if not items:
                continue
            shown = ", ".join(items[:_INTEGRITY_DETAIL_LIMIT])
            if len(items) > _INTEGRITY_DETAIL_LIMIT:
                shown += f", and {len(items) - _INTEGRITY_DETAIL_LIMIT} more"
            parts.append(f"{label}: {shown}")
        return "; ".join(parts) if parts else "nothing missing"


_INTEGRITY_DETAIL_LIMIT = 10


def _read_physical_schema(
    connection: sqlite3.Connection,
) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
    """The file's own tables with their columns, and its indexes with their tables.

    SQLite's implicit ``sqlite_autoindex_*`` entries are left out: they follow
    from a UNIQUE constraint in the table DDL, so comparing them would report a
    difference that the table comparison has already reported.
    """
    tables: dict[str, frozenset[str]] = {}
    for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        name = str(row[0])
        tables[name] = frozenset(
            str(column[1])
            for column in connection.execute(f"PRAGMA table_info({name})").fetchall()
        )
    indexes = {
        str(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT name, tbl_name FROM sqlite_master "
            "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    return tables, indexes


def _read_column_declarations(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, str | None]]:
    """Each column's ``ALTER TABLE ADD COLUMN`` spec, or None when it has none.

    None means SQLite cannot add the column back: it is NOT NULL without a
    default, or part of the primary key. A table missing one of those cannot be
    repaired, only restored.
    """
    declarations: dict[str, dict[str, str | None]] = {}
    for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        table = str(row[0])
        columns: dict[str, str | None] = {}
        for column in connection.execute(f"PRAGMA table_info({table})").fetchall():
            name, kind, notnull, default, primary_key = (
                str(column[1]),
                str(column[2]),
                bool(column[3]),
                column[4],
                bool(column[5]),
            )
            if primary_key or (notnull and default is None):
                columns[name] = None
                continue
            spec = kind
            if notnull:
                spec += " NOT NULL"
            if default is not None:
                spec += f" DEFAULT {default}"
            columns[name] = spec
        declarations[table] = columns
    return declarations


@dataclass(frozen=True)
class _ReferenceSchema:
    tables: dict[str, frozenset[str]]
    indexes: dict[str, str]
    column_declarations: dict[str, dict[str, str | None]]


_REFERENCE_SCHEMA: _ReferenceSchema | None = None


def _reference_schema() -> _ReferenceSchema:
    """The schema a fresh database of this build ends up with, built by creating one.

    Derived rather than hand-listed on purpose: a contract written out as a
    literal falls behind the DDL the moment somebody adds a table, and the audit
    then reports a database that is missing it as healthy. Creating the schema
    the product creates cannot fall behind it.

    Cached for the process; it costs one in-memory database.
    """
    global _REFERENCE_SCHEMA
    if _REFERENCE_SCHEMA is None:
        reference = PokerDatabase(":memory:")
        # The reference cannot verify itself against the reference it is building.
        reference._verify_schema_on_init = False
        try:
            reference.init_db()
            tables, indexes = _read_physical_schema(reference._connection)
            _REFERENCE_SCHEMA = _ReferenceSchema(
                tables=tables,
                indexes=indexes,
                column_declarations=_read_column_declarations(reference._connection),
            )
        finally:
            reference.close()
    return _REFERENCE_SCHEMA


def _foreign_key_violations(
    connection: sqlite3.Connection, *, limit: int = 100
) -> tuple[str, ...]:
    """Rows whose parent no longer exists, named by table and rowid.

    Bounded: a database with a broken relationship usually has many, and the
    report is read by a person.
    """
    try:
        rows = connection.execute("PRAGMA foreign_key_check").fetchmany(limit)
    except sqlite3.DatabaseError:
        return ()
    return tuple(
        f"{row[0]} row {row[1]} references a missing {row[2]} row" for row in rows
    )


def inspect_schema_integrity(
    connection: sqlite3.Connection, *, check_foreign_keys: bool = True
) -> SchemaIntegrityReport:
    """Compare a database against the schema this build requires.

    Works on any connection, including a read-only one opened over a retained
    snapshot, so a restore drill can ask the same question of a recovered file
    that ``init_db`` asks of the live one.
    """
    reference = _reference_schema()
    expected_tables, expected_indexes = reference.tables, reference.indexes
    actual_tables, actual_indexes = _read_physical_schema(connection)
    missing_tables = tuple(sorted(set(expected_tables) - set(actual_tables)))
    missing_columns = tuple(
        sorted(
            f"{table}.{column}"
            for table, columns in expected_tables.items()
            if table in actual_tables
            for column in columns - actual_tables[table]
        )
    )
    # An index on a table that is itself missing is not reported twice.
    missing_indexes = tuple(
        sorted(
            name
            for name, table in expected_indexes.items()
            if name not in actual_indexes and table not in missing_tables
        )
    )
    return SchemaIntegrityReport(
        missing_tables=missing_tables,
        missing_columns=missing_columns,
        missing_indexes=missing_indexes,
        foreign_key_violations=(
            _foreign_key_violations(connection) if check_foreign_keys else ()
        ),
    )


def _finalize_correction_notes(
    notes: str, *, rejection_codes: tuple[str, ...]
) -> str:
    """Audit text for operator finalize; blank notes stay informative when rejecting."""

    cleaned = notes.strip()
    if cleaned:
        return cleaned
    if rejection_codes:
        return (
            "Operator finalized incomplete hand after filling blanks; "
            "overrode rejection_codes: " + ", ".join(rejection_codes)
        )
    return "Operator finalized incomplete hand after filling blanks."


class PokerDatabase:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        *,
        busy_timeout_ms: int = BUSY_TIMEOUT_MS,
    ) -> None:
        self.db_path = str(db_path)
        self._busy_timeout_ms = max(int(busy_timeout_ms), 0)
        # Cleared only by _reference_schema, which opens one of these to learn
        # what a complete schema looks like and so cannot be checked against it.
        self._verify_schema_on_init = True
        # One connection is shared across Streamlit's script-run threads, so every
        # statement goes through _execute() under a re-entrant lock, and grouped
        # writes use transaction() for atomicity.
        self._lock = threading.RLock()
        self._txn_depth = 0
        # Why this file was refused, or None when it was accepted. Read by every
        # commit, so a caller that never reaches init_db still cannot write to a
        # database this build does not understand.
        self._refusal: str | None = None
        # Columns init_db had to add back because the file arrived without them.
        # Empty for every healthy database; non-empty means the file was damaged
        # and an audit should say so.
        self.restored_columns: tuple[str, ...] = ()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._execute("PRAGMA foreign_keys = ON")
        # A migration can take longer than SQLite's default 5s busy timeout on a
        # real database; a second opener must wait for it rather than dying with
        # "database is locked" at startup.
        self._execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        # Never write to a database this build refuses. journal_mode=WAL rewrites
        # the file header, so a refused open used to convert a restored
        # journal_mode=delete snapshot in place -- and to raise
        # "attempt to write a readonly database" on an archival mount, before the
        # operator ever saw the version message. init_db() raises the clear error.
        try:
            opened_version = _readable_schema_version(self._connection)
            _assert_supported_schema_version(opened_version)
            _assert_stamp_matches_schema(
                opened_version, _physical_schema_floor(self._connection)
            )
            _assert_schema_matches_stamp(
                opened_version, _physical_schema_ceiling(self._connection)
            )
        except RuntimeError as exc:
            # Every production caller reaches init_db immediately and gets this
            # message there. Recording it also closes the other door: the object
            # this constructor hands back has a live connection on a file whose
            # tables may all be present -- a database from a NEWER build has every
            # table this one knows -- so a caller that skipped init_db could write
            # to it, and the write would look completely ordinary.
            self._refusal = str(exc)
            return
        self._enter_wal_mode()
        self._execute("PRAGMA synchronous = NORMAL")

    def _enter_wal_mode(self) -> None:
        """Switch the file to WAL, waiting out a concurrent writer.

        SQLite does not invoke the busy handler for a journal-mode change, so the
        `PRAGMA busy_timeout` set two lines above does not cover this statement:
        it returns SQLITE_BUSY immediately while any other connection holds a
        lock, and the raw `sqlite3.OperationalError: database is locked` escaped
        the constructor, leaking the half-built object and its open connection.

        That only bites while the file is still in rollback-journal mode -- a
        brand-new install's first start, and any database restored from one of
        this product's own pinned snapshots, which `backup.backup_database`
        deliberately writes in `journal_mode=DELETE`. Both are exactly the moments
        the app, a CV reconstruction job and a solver job may start together, and
        `PRAGMA journal_mode = WAL` on a file that is ALREADY in WAL is a no-op
        that needs no exclusive lock, so retrying converges on the first opener.

        Waits up to the connection's own busy timeout, then reports what is
        actually wrong instead of "database is locked". A read-only or full mount
        is named too, for the same reason `_backup_before_migration` names the
        backups directory: SQLite's own wording for it points at the database.
        """
        deadline = time.monotonic() + self._busy_timeout_ms / 1000
        delay = 0.005
        while True:
            try:
                self._execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "readonly" in message or "read-only" in message:
                    self._connection.close()
                    raise RuntimeError(
                        f"The database at {self.db_path} is on a read-only "
                        "filesystem, so it could not be opened for writing. The "
                        "database itself is intact and unchanged. Mount its "
                        "directory writable — a container needs the whole "
                        "directory, not just the file, because SQLite writes "
                        "journal files beside it."
                    ) from exc
                if "locked" not in message and "busy" not in message:
                    raise
                if time.monotonic() >= deadline:
                    self._connection.close()
                    raise RuntimeError(
                        "Another process is still writing to this database, so it "
                        f"could not be opened within {self._busy_timeout_ms}ms. "
                        "This usually means a migration, a CV reconstruction job "
                        "or a solver job is still running; wait for it to finish "
                        "and start again."
                    ) from exc
                time.sleep(min(delay, max(deadline - time.monotonic(), 0)))
                delay = min(delay * 2, 0.25)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._connection.execute(sql, params)

    def _commit(self) -> None:
        # Inside transaction() the outermost exit owns the commit/rollback.
        with self._lock:
            self._refuse_if_unsupported()
            if self._txn_depth == 0:
                self._connection.commit()

    def _refuse_if_unsupported(self) -> None:
        """Fail every commit on a database the constructor refused.

        The commit is the chokepoint: sqlite3 keeps DML and DDL alike inside the
        implicit transaction, so a statement that is never committed is never
        durable. Reads stay available on purpose -- ``schema_version()`` and the
        backup path both have to work on a file this build will not write.
        """
        if self._refusal is None:
            return
        # Discard whatever the caller already executed. It is not durable without
        # the commit, but leaving it pending would let a later read on this same
        # connection see a write this build refused to make.
        self._connection.rollback()
        raise RuntimeError(self._refusal)

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[PokerDatabase]:
        """Group multiple writes into a single atomic commit.

        Re-entrant: nested transaction() blocks (and the per-method commits of
        the CRUD helpers called inside) defer to the outermost block, which
        commits on success and rolls back everything on the first exception.
        ``immediate=True`` acquires SQLite's write reservation before any reads,
        allowing check-then-insert policies to serialize across connections.
        """
        with self._lock:
            self._refuse_if_unsupported()
            if self._txn_depth == 0 and immediate:
                self._connection.execute("BEGIN IMMEDIATE")
            self._txn_depth += 1
            try:
                yield self
            except BaseException:
                self._txn_depth -= 1
                if self._txn_depth == 0:
                    self._connection.rollback()
                raise
            else:
                self._txn_depth -= 1
                if self._txn_depth == 0:
                    self._connection.commit()

    def init_db(self) -> None:
        """Create the schema and apply any pending versioned migrations."""
        with self._lock:
            opened_version = _readable_schema_version(self._connection)
            _assert_supported_schema_version(opened_version)
            # Decided before _create_base_tables, which would otherwise make every
            # brand-new file look like an existing database worth snapshotting --
            # and would make every fresh file's physical schema read as version 13.
            has_existing_tables = self._has_user_tables()
            # Measured on the file as it arrived. _create_base_tables writes the
            # CURRENT schema, so a legacy database that has some tables but no
            # `hands` yet would otherwise be handed a v13 `hands` and then refused
            # for carrying it.
            schema_floor = _physical_schema_floor(self._connection)
            # ONE transaction covering the table DDL, the migration chain and the
            # version stamp, so a failed or interrupted start leaves no partial
            # schema and no version stamp: SQLite DDL is transactional under an
            # explicit BEGIN.
            #
            # _create_base_tables used to run before this block, committed on its
            # own, on the argument that CREATE ... IF NOT EXISTS leaves only
            # idempotent artifacts. The artifacts are idempotent for the CREATE
            # statements and NOT for the discriminator _assert_stamp_matches_schema
            # keys on: on a brand-new file that commit writes the CURRENT schema,
            # including hands.completion_status, so an interruption anywhere after
            # it (power loss, an OOM kill, a container restart, Ctrl-C on the first
            # launch) left a file whose physical floor read 13 with no stamp. Every
            # later start then refused it forever and told the operator to restore
            # a backup that was never taken -- _backup_before_migration returns
            # early for a fresh file, correctly, because there is nothing yet to
            # preserve. Inside the transaction the same interruption rolls the file
            # back to empty and the next start simply creates it.
            with self.transaction(immediate=True):
                # Under the write reservation, and before schema_version() is read
                # from it: on a fresh file schema_metadata does not exist yet, and
                # _readable_schema_version reports 0 either way.
                self._create_base_tables()
                # Re-read the version under SQLite's write reservation. Two
                # processes -- the app, a CV job, a solver job, the exporter --
                # can each read the pre-migration version before either commits,
                # and re-running the v13 chain resets every operator confirmation
                # written in between. The loser finds the new stamp and stops.
                stored_version = self.schema_version()
                _assert_supported_schema_version(stored_version)
                # The stamp is the one re-read under the write reservation, never
                # the stale pre-reservation read: a concurrent opener that saw a
                # pre-migration version while the winner was committing must defer
                # to the new stamp, not be refused for a mismatch that no longer
                # exists.
                _assert_stamp_matches_schema(stored_version, schema_floor)
                # Measured AFTER _create_base_tables and under the reservation,
                # unlike the floor. `CREATE TABLE IF NOT EXISTS hands` adds no
                # column to an existing table, so a pre-v13 `hands` still reads
                # as pre-v13 here -- while a fresh file, and a file a concurrent
                # opener has just migrated, both read as current. Measuring it on
                # arrival instead would refuse the loser of a legitimate race,
                # whose pre-reservation read predates the winner's commit.
                _assert_schema_matches_stamp(
                    stored_version, _physical_schema_ceiling(self._connection)
                )
                if stored_version < SCHEMA_VERSION:
                    self._backup_before_migration(
                        has_existing_tables=has_existing_tables
                    )
                    if stored_version < 5:
                        # Pre-versioning databases: idempotent column backfill.
                        self._apply_legacy_backfill()
                    for version in range(max(stored_version, 5) + 1, SCHEMA_VERSION + 1):
                        migration = _MIGRATIONS.get(version)
                        if migration is None:
                            raise RuntimeError(
                                f"No migration registered for schema version {version}."
                            )
                        migration(self)
                    self._execute(
                        """
                        INSERT INTO schema_metadata (key, value)
                        VALUES ('schema_version', ?)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value
                        """,
                        (str(SCHEMA_VERSION),),
                    )
            # Indexes come last because some of them cover columns that only exist
            # after the legacy backfill or a migration has run. Creating
            # idx_roi_profiles_active alongside the tables bricked every open of a
            # pre-v5 database that already had a roi_profiles table: the index ran
            # before _apply_legacy_backfill added is_active, executescript() had
            # already committed the statements before it, and the pre-migration
            # snapshot inside the transaction was never reached.
            self._create_base_indexes()
            if self._verify_schema_on_init:
                self._assert_schema_is_intact()

    def _assert_schema_is_intact(self) -> None:
        """Refuse a database still missing structures once the whole chain has run.

        ``CREATE TABLE IF NOT EXISTS``, the legacy backfill and the versioned
        migrations between them repair everything this build knows how to
        repair, so anything still absent here is damage no migration covers: a
        hand-edited file, a half-applied ALTER, a table rebuilt by a repair
        script, a database assembled from two restores. Those all used to open
        cleanly and then fail one write at a time with a bare
        ``sqlite3.OperationalError`` naming a single column, which reads as "this
        feature is broken" rather than "this file is not usable" -- and a missing
        UNIQUE index is worse than that, because nothing fails at all: the rows
        it forbids simply start being written.

        A missing column is restored rather than refused when the current DDL
        declares a default for it, because this schema is forward-only and
        additive: the value that restores is exactly the value the migration that
        introduced the column would have written, and the alternative is a
        database that cannot be opened at all. The one column whose absence
        genuinely cannot be repaired that way -- ``hands.completion_status``, on a
        file stamped past 13 -- never reaches here: ``_assert_schema_matches_stamp``
        refuses it in the constructor, because re-running that migration is what
        discards operator confirmations. What was restored is recorded on
        ``restored_columns`` so an audit can report the repair rather than
        discovering it later.

        Foreign keys are deliberately not scanned here. ``PRAGMA
        foreign_key_check`` walks every row of every child table, which is
        startup latency proportional to the size of the database; the audit
        surfaces, which run on demand, ask ``inspect_schema_integrity`` for it.
        """
        report = inspect_schema_integrity(self._connection, check_foreign_keys=False)
        if report.is_intact:
            return
        self._restore_missing_columns(report.missing_columns)
        report = inspect_schema_integrity(self._connection, check_foreign_keys=False)
        if report.is_intact:
            return
        raise RuntimeError(
            f"The database at {self.db_path} is missing structures this app "
            f"requires ({report.describe()}). The migration chain ran and could "
            "not supply them, so this build will not use the file. Restore it "
            f"from a backup in {backup_module.BACKUPS_DIR} before opening it "
            "again."
        )

    def _restore_missing_columns(self, missing: tuple[str, ...]) -> None:
        """Add back every missing column the current DDL can declare a default for.

        The same operation ``_apply_legacy_backfill`` performs for the columns
        that predate versioning, driven by the reference schema instead of by a
        list, so it covers a column added tomorrow. A column SQLite cannot add --
        NOT NULL with no default, or part of the primary key -- is left for the
        caller to refuse over.
        """
        declarations = _reference_schema().column_declarations
        for reference in missing:
            table, _, column = reference.partition(".")
            spec = declarations.get(table, {}).get(column)
            if spec is None:
                continue
            self._ensure_column(table, column, spec)
            self.restored_columns = (*self.restored_columns, reference)
        self._commit()

    def schema_integrity(self) -> SchemaIntegrityReport:
        """What this database lacks compared with the schema this build requires."""
        with self._lock:
            return inspect_schema_integrity(self._connection)

    def _has_user_tables(self) -> bool:
        row = self._execute(
            "SELECT COUNT(*) AS n FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()
        return bool(row and row["n"])

    def _backup_before_migration(self, *, has_existing_tables: bool) -> None:
        """Snapshot a real, non-empty database before applying pending migrations.

        Fails closed: if the snapshot cannot be written, init_db propagates and no
        migration runs. ``has_existing_tables`` is required because a pre-versioning
        database also reports schema_version() == 0, and gating on the version alone
        would skip the backup for exactly the databases that most need one.

        Runs under the migration's write reservation, so exactly one process
        snapshots and the snapshot is of the pre-migration state. It may contain
        empty tables ``_create_base_tables`` just added; no existing row is
        changed, so it remains a complete rollback point.
        """
        if not has_existing_tables:
            return  # brand-new file: nothing to preserve
        if self.db_path == ":memory:" or not Path(self.db_path).exists():
            return
        try:
            backup_database(Path(self.db_path), pinned=True)
        except (OSError, sqlite3.Error) as exc:
            # sqlite3 reports an unwritable backups directory as "unable to open
            # database file", whose plain reading is that the operator's
            # poker_tracker.db is broken -- the opposite of the truth. On the
            # container path a read-only or full data mount produces exactly this
            # at startup, so the message has to name the directory it could not
            # write and say that nothing was migrated.
            raise RuntimeError(
                "Could not write the pre-migration backup to "
                f"{backup_module.backups_dir_for(Path(self.db_path))}; the migration "
                "was not applied and the database "
                "is unchanged. Make that directory writable with free space, then "
                f"start again. Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

    def _create_base_tables(self) -> None:
        """Table DDL only. Indexes are created by _create_base_indexes, after the
        legacy backfill and the migration chain have supplied every column.

        Executed through _execute_script, never executescript(): the latter
        implicitly COMMITs, which is exactly what used to strand a brand-new file
        at schema-13 structures with no version stamp when the first start was
        interrupted.

        MIGRATION IMPACT (no version change)

        ``hand_settlements`` and ``settlement_entries`` were added to this DDL
        without a new schema version, because every database that legitimately
        reaches this build already has them: migration 7 creates them, they are
        declared here exactly as it declares them, and CREATE TABLE IF NOT EXISTS
        makes both statements a no-op for such a file. No row is read, written or
        deleted, and the version stamp does not move. What changes is the file
        that DOES NOT have them -- one whose stamp is 7 or later, so the chain
        never replays step 7 -- which previously stayed permanently without a
        settlements table and failed every accounting read against it. They were
        the last two tables of the current schema that existed only inside their
        migration; hand_corrections, hand_issues, solver_runs and
        regression_cases were already declared in both places.
        """
        self._execute_script(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date_played TEXT NOT NULL,
                platform TEXT NOT NULL DEFAULT '',
                stakes TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS hands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                hand_number INTEGER NOT NULL,
                game_type TEXT NOT NULL DEFAULT '',
                blinds_antes TEXT NOT NULL DEFAULT '',
                table_size INTEGER,
                effective_stack REAL,
                hero_position TEXT NOT NULL DEFAULT '',
                hero_cards TEXT NOT NULL DEFAULT '',
                board_cards TEXT NOT NULL DEFAULT '',
                pot_size REAL,
                result TEXT NOT NULL DEFAULT '',
                hero_bb_won REAL,
                review_status TEXT NOT NULL DEFAULT 'unreviewed',
                confidence_score REAL,
                source_type TEXT NOT NULL DEFAULT 'manual',
                tags TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                completion_status TEXT NOT NULL DEFAULT 'not_applicable',
                completion_evidence TEXT NOT NULL DEFAULT '{}',
                study_inclusion TEXT NOT NULL DEFAULT 'auto',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hand_players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                player_key TEXT NOT NULL,
                seat_index INTEGER,
                player_name TEXT NOT NULL,
                position TEXT NOT NULL DEFAULT '',
                starting_stack REAL,
                is_hero INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                player_key TEXT,
                street TEXT NOT NULL,
                action_index INTEGER NOT NULL,
                player_name TEXT NOT NULL,
                position TEXT NOT NULL DEFAULT '',
                action_type TEXT NOT NULL,
                amount REAL,
                amount_semantics TEXT NOT NULL DEFAULT 'incremental',
                forced_bet_type TEXT,
                is_live_post INTEGER,
                pot_before REAL,
                stack_before REAL,
                notes TEXT NOT NULL DEFAULT '',
                source_image TEXT,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hand_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                hand_summary TEXT NOT NULL,
                theory_coach TEXT NOT NULL,
                exploit_coach TEXT NOT NULL,
                ev_math_notes TEXT NOT NULL DEFAULT '',
                study_lesson TEXT NOT NULL,
                next_review_question TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                is_stale INTEGER NOT NULL DEFAULT 0,
                stale_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS coaching_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                raw_prompt TEXT NOT NULL,
                raw_response TEXT NOT NULL,
                review_type TEXT NOT NULL,
                safety_mode TEXT NOT NULL DEFAULT 'post_session_only',
                hand_id INTEGER,
                session_id INTEGER,
                parsed_sections TEXT NOT NULL DEFAULT '{}',
                is_stale INTEGER NOT NULL DEFAULT 0,
                stale_reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            -- Also created by migration 7, like hand_corrections, hand_issues,
            -- solver_runs and regression_cases below. These two were the only
            -- tables of the current schema that lived ONLY in their migration,
            -- so a database stamped 7 or later that lost them never got them
            -- back: the chain was already past the step that creates them.
            CREATE TABLE IF NOT EXISTS hand_settlements (
                hand_id INTEGER PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'unsettled',
                small_blind REAL,
                big_blind REAL,
                straddles TEXT NOT NULL DEFAULT '[]',
                ante_mode TEXT,
                dead_money REAL NOT NULL DEFAULT 0,
                rake_rate REAL NOT NULL DEFAULT 0,
                rake_cap REAL,
                rake_rounding_unit REAL NOT NULL DEFAULT 0.01,
                no_flop_no_drop INTEGER NOT NULL DEFAULT 0,
                gross_pot REAL,
                rake_amount REAL,
                net_pot REAL,
                is_balanced INTEGER NOT NULL DEFAULT 0,
                warnings TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS settlement_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                entry_type TEXT NOT NULL,
                pot_index INTEGER,
                player_key TEXT,
                player_name TEXT NOT NULL,
                amount REAL,
                entry_order INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hand_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                correction_type TEXT NOT NULL,
                before_state TEXT NOT NULL DEFAULT '{}',
                after_state TEXT NOT NULL DEFAULT '{}',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hand_issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                issue_types TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL,
                evidence_snapshot TEXT NOT NULL DEFAULT '{}',
                resolution_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS regression_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue_id INTEGER NOT NULL,
                correction_id INTEGER,
                kind TEXT NOT NULL,
                fixture_path TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'proposed',
                failing_before INTEGER NOT NULL DEFAULT 0,
                passing_after INTEGER NOT NULL DEFAULT 0,
                fixing_commit TEXT NOT NULL DEFAULT '',
                report_path TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (issue_id) REFERENCES hand_issues(id) ON DELETE CASCADE,
                FOREIGN KEY (correction_id) REFERENCES hand_corrections(id) ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_regression_cases_issue
                ON regression_cases(issue_id, status, id);

            CREATE TABLE IF NOT EXISTS solver_range_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                notation TEXT NOT NULL,
                table_size INTEGER,
                position TEXT NOT NULL DEFAULT '',
                scenario TEXT NOT NULL DEFAULT '',
                pot_type TEXT NOT NULL DEFAULT '',
                stack_bb REAL,
                description TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS solver_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hand_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                backend_name TEXT NOT NULL DEFAULT 'texassolver',
                backend_version TEXT NOT NULL DEFAULT '',
                run_parameters TEXT NOT NULL DEFAULT '{}',
                input_hash TEXT NOT NULL,
                spot TEXT NOT NULL DEFAULT '{}',
                range_ip TEXT NOT NULL DEFAULT '{}',
                range_oop TEXT NOT NULL DEFAULT '{}',
                assumptions TEXT NOT NULL DEFAULT '[]',
                evidence TEXT NOT NULL DEFAULT '{}',
                command_path TEXT NOT NULL DEFAULT '',
                result_path TEXT NOT NULL DEFAULT '',
                log_path TEXT NOT NULL DEFAULT '',
                exploitability_pct REAL,
                runtime_seconds REAL,
                error_message TEXT NOT NULL DEFAULT '',
                pid INTEGER,
                heartbeat_at TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                original_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL DEFAULT '',
                duration_seconds REAL,
                fps REAL,
                width INTEGER,
                height INTEGER,
                frame_count INTEGER,
                uploaded_at TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS processing_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                video_id INTEGER NOT NULL,
                progress_percent REAL NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                pid INTEGER,
                heartbeat_at TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS extracted_frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                timestamp_seconds REAL NOT NULL,
                frame_index INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                FOREIGN KEY (job_id) REFERENCES processing_jobs(id) ON DELETE CASCADE,
                UNIQUE(video_id, frame_index, image_path)
            );

            CREATE TABLE IF NOT EXISTS reconstruction_frame_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                hand_number INTEGER NOT NULL,
                source_image TEXT NOT NULL,
                timestamp_seconds REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'unreviewed',
                issue_types TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES processing_jobs(id) ON DELETE CASCADE,
                UNIQUE(job_id, hand_number, source_image)
            );

            CREATE TABLE IF NOT EXISTS roi_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                platform TEXT NOT NULL DEFAULT 'ClubWPT Gold',
                table_layout TEXT NOT NULL DEFAULT '',
                video_width INTEGER,
                video_height INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS roi_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                roi_key TEXT NOT NULL,
                roi_type TEXT NOT NULL DEFAULT 'unknown',
                label TEXT NOT NULL DEFAULT '',
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                seat_index INTEGER,
                card_index INTEGER,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES roi_profiles(id) ON DELETE CASCADE,
                UNIQUE(profile_id, roi_key)
            );
            """
        )

    def _create_base_indexes(self) -> None:
        """Every index the current schema requires, created idempotently on each open.

        The uniqueness indexes at the end are also created by migrations 7 and 8,
        and they are repeated here because a migration only ever runs once: an
        index lost after its migration -- a hand-edited file, a table rebuilt by a
        repair script, a restore that mixed two files -- would never come back,
        and unlike a missing column nothing would fail. The duplicate action
        order or the second seat that index forbids would simply start being
        written, silently. Recreating them costs nothing on a database that
        already has them.
        """
        try:
            self._create_index_script()
        except sqlite3.IntegrityError as exc:
            # Only reachable when the rows already violate the rule, which means
            # they were written while the index was absent. SQLite reports it as
            # a bare constraint failure, which reads as an app bug rather than as
            # a database this build cannot safely use.
            raise RuntimeError(
                f"The database at {self.db_path} holds rows that break a "
                "uniqueness rule this app requires, so the index that enforces "
                f"it could not be created ({exc}). The rows were written while "
                "the index was missing. Restore the database from a backup in "
                f"{backup_module.BACKUPS_DIR} before opening it again."
            ) from exc

    def _create_index_script(self) -> None:
        self._connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_hands_session_id ON hands(session_id);
            CREATE INDEX IF NOT EXISTS idx_hand_players_hand_id ON hand_players(hand_id);
            CREATE INDEX IF NOT EXISTS idx_actions_hand_id ON actions(hand_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_hand_id ON hand_reviews(hand_id);
            CREATE INDEX IF NOT EXISTS idx_coaching_reviews_hand_id ON coaching_reviews(hand_id);
            CREATE INDEX IF NOT EXISTS idx_coaching_reviews_session_id ON coaching_reviews(session_id);
            CREATE INDEX IF NOT EXISTS idx_hand_corrections_hand_id
                ON hand_corrections(hand_id, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_hand_issues_status
                ON hand_issues(status, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_hand_issues_hand
                ON hand_issues(hand_id, status, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_solver_runs_hand
                ON solver_runs(hand_id, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_solver_runs_cache
                ON solver_runs(input_hash, status);
            CREATE INDEX IF NOT EXISTS idx_solver_runs_status
                ON solver_runs(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_videos_session_id ON videos(session_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_video_id ON processing_jobs(video_id);
            CREATE INDEX IF NOT EXISTS idx_frames_video_id ON extracted_frames(video_id);
            CREATE INDEX IF NOT EXISTS idx_reconstruction_reviews_job_hand
                ON reconstruction_frame_reviews(job_id, hand_number);
            CREATE INDEX IF NOT EXISTS idx_roi_profiles_active ON roi_profiles(is_active);
            CREATE INDEX IF NOT EXISTS idx_roi_regions_profile_id ON roi_regions(profile_id);
            CREATE INDEX IF NOT EXISTS idx_actions_player_key
                ON actions(hand_id, player_key);
            CREATE INDEX IF NOT EXISTS idx_settlement_entries_hand
                ON settlement_entries(hand_id, entry_type, pot_index, entry_order);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_hand_players_hand_key
                ON hand_players(hand_id, player_key);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_hand_players_hand_seat
                ON hand_players(hand_id, seat_index)
                WHERE seat_index IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_hand_street_order
                ON actions(hand_id, street, action_index);
            """
        )

    def _apply_legacy_backfill(self) -> None:
        """Backfill columns added before schema versioning existed (idempotent)."""
        self._ensure_column("hands", "game_type", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hands", "blinds_antes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hands", "table_size", "INTEGER")
        self._ensure_column("hands", "effective_stack", "REAL")
        self._ensure_column("hands", "review_status", "TEXT NOT NULL DEFAULT 'unreviewed'")
        self._ensure_column("hands", "confidence_score", "REAL")
        self._ensure_column("hands", "source_type", "TEXT NOT NULL DEFAULT 'manual'")
        self._ensure_column("hands", "tags", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("hand_reviews", "ev_math_notes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hand_reviews", "next_review_question", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("hand_reviews", "notes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(
            "coaching_reviews", "safety_mode", "TEXT NOT NULL DEFAULT 'post_session_only'"
        )
        self._ensure_column("coaching_reviews", "parsed_sections", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column("videos", "frame_count", "INTEGER")
        self._ensure_column("roi_profiles", "table_layout", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("roi_profiles", "is_active", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("roi_regions", "notes", "TEXT NOT NULL DEFAULT ''")

    def _execute_script(self, script: str) -> None:
        """Run a multi-statement DDL script inside the caller's transaction.

        sqlite3.Connection.executescript() implicitly COMMITs any open
        transaction, which would silently defeat migration rollback, so the
        script is split and executed one statement at a time.
        """
        statement = ""
        for line in script.splitlines(keepends=True):
            statement += line
            if sqlite3.complete_statement(statement):
                if statement.strip():
                    self._execute(statement)
                statement = ""
        if statement.strip():
            self._execute(statement)

    def _ensure_column(self, table_name: str, column_name: str, column_spec: str) -> None:
        columns = {
            row["name"] for row in self._execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            self._execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_spec}")

    def create_session(self, session: Session) -> Session:
        payload = session.model_dump()
        cursor = self._execute(
            """
            INSERT INTO sessions (name, date_played, platform, stakes, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["name"],
                _serialize_date(payload["date_played"]),
                payload["platform"],
                payload["stakes"],
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        return session.model_copy(update={"id": cursor.lastrowid})

    def update_session(self, session: Session) -> Session:
        """Update mutable session fields. Schema unchanged — date_played already exists."""

        if session.id is None:
            raise ValueError("Cannot update a session without an id.")
        if self.fetch_session(session.id) is None:
            raise ValueError("Session not found.")
        payload = session.model_dump()
        name = str(payload["name"]).strip()
        if not name:
            raise ValueError("Session name cannot be empty.")
        self._execute(
            """
            UPDATE sessions
            SET name = ?, date_played = ?, platform = ?, stakes = ?, notes = ?
            WHERE id = ?
            """,
            (
                name,
                _serialize_date(payload["date_played"]),
                payload["platform"],
                payload["stakes"],
                payload["notes"],
                payload["id"],
            ),
        )
        self._commit()
        updated = self.fetch_session(session.id)
        if updated is None:
            raise RuntimeError("Updated session could not be reloaded.")
        return updated

    def create_hand(self, hand: Hand) -> Hand:
        _refuse_display_copy(hand, "store")
        payload = hand.model_dump()
        cursor = self._execute(
            """
            INSERT INTO hands (
                session_id, hand_number, game_type, blinds_antes, table_size,
                effective_stack, hero_position, hero_cards, board_cards, pot_size,
                result, hero_bb_won, review_status, confidence_score, source_type,
                tags, notes, created_at, completion_status, completion_evidence,
                study_inclusion
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["session_id"],
                payload["hand_number"],
                payload["game_type"],
                payload["blinds_antes"],
                payload["table_size"],
                payload["effective_stack"],
                payload["hero_position"],
                payload["hero_cards"],
                payload["board_cards"],
                payload["pot_size"],
                payload["result"],
                payload["hero_bb_won"],
                payload["review_status"],
                payload["confidence_score"],
                payload["source_type"],
                _serialize_json(payload["tags"]),
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
                payload["completion_status"],
                # A caller that round-trips a fetched hand -- import_session, a CV
                # re-write, a test helper -- must not persist the reader's own
                # unreadable-card annotation as if the pipeline had produced it.
                # An operator finalize attestation is earned by acting on an
                # existing hand, so a hand being CREATED cannot carry one:
                # finalize_incomplete_hand is its only writer.
                _serialize_json(
                    strip_operator_attestation(
                        strip_derived_evidence_markers(payload["completion_evidence"])
                    )
                ),
                # Study inclusion is an operator preference set after the hand
                # exists; create always starts at auto (update_study_inclusion).
                "auto",
            ),
        )
        self._commit()
        return hand.model_copy(update={"id": cursor.lastrowid, "study_inclusion": "auto"})

    def update_hand_status(self, hand_id: int, review_status: str) -> None:
        """Set the review status, refusing to promote a hand the store knows is unproven.

        This is the unbypassable floor, not the full readiness rule: db.py can only
        see single-table facts. Accounting, coaching, solver, and per-render user
        confirmation are enforced at the UI choke point, because hand_accounting
        imports db.py and the layering must not invert.

        The open-issue half asks ``fetch_hand_issues``, not a SQL
        ``status = 'open'`` subquery. See ``_MODEL_SPACE_CLASSIFICATION``.
        """
        if review_status not in get_args(ReviewStatus):
            raise ValueError(f"Unknown review status: {review_status!r}")
        if review_status == "reviewed":
            row = self._execute(
                """
                SELECT
                    h.completion_status AS completion_status,
                    h.completion_evidence AS completion_evidence,
                    h.source_type AS source_type
                FROM hands AS h
                WHERE h.id = ?
                """,
                (hand_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Hand not found.")
            has_open_issue = any(
                issue.status == "open"
                for issue in self.fetch_hand_issues(hand_id=hand_id)
            )
            if row["completion_status"] not in {"complete", "not_applicable"}:
                raise ValueError(
                    f"Hand {hand_id} is {row['completion_status']}; a partial or uncertain "
                    "hand cannot be marked reviewed."
                )
            evidence = parse_completion_evidence(
                _parse_json_object(row["completion_evidence"], {})
            )
            # The mirror image of the reconstructed-with-not_applicable pair
            # below, and the same verdict import_session and _hand_from_row
            # reach: a 'manual' claim carrying pipeline-stamped evidence is a
            # reconstructed hand wearing the exemption's strings. Checked on the
            # raw row deliberately -- this writer takes an id, so it must not
            # trust a fetch that happened before a hand-edited UPDATE.
            if row["source_type"] == "manual" and evidence.claims_reconstruction:
                raise ValueError(
                    f"Hand {hand_id} declares source_type 'manual' but carries "
                    "reconstruction completion evidence and cannot be marked "
                    "reviewed."
                )
            # The column alone proves nothing: create_hand writes whatever it is
            # given. A 'complete' claim must still be re-derived from the stored
            # evidence, which is the only thing that can promote a hand.
            derived = derive_completion_status(
                evidence,
                source_type=row["source_type"],
            )
            if row["completion_status"] == "complete" and derived != "complete":
                raise ValueError(
                    f"Hand {hand_id} declares completion status 'complete' but its "
                    f"stored evidence derives {derived!r}; it cannot be marked reviewed."
                )
            # 'not_applicable' exempts a hand from every completion blocker, so it is
            # only legitimate for a manual hand. import_session rejects the same pair
            # (import_export._apply_completion_import_defaults); refusing it here too
            # closes the equivalent hole for rows written through create_hand.
            if (
                row["source_type"] != "manual"
                and row["completion_status"] == "not_applicable"
            ):
                raise ValueError(
                    f"Hand {hand_id} declares source {row['source_type']!r} with "
                    "completion status 'not_applicable' and cannot be marked reviewed."
                )
            if has_open_issue:
                raise ValueError(
                    f"Hand {hand_id} has an open debugging issue and cannot be marked reviewed."
                )
        self._execute(
            "UPDATE hands SET review_status = ? WHERE id = ?",
            (review_status, hand_id),
        )
        self._commit()

    def restore_unreadable_card_columns(
        self, hand_id: int, recorded: dict[str, object]
    ) -> None:
        """The card-column entry point, kept: it is one case of the general rule."""
        self.restore_unreadable_columns(hand_id, recorded)

    def restore_unreadable_columns(
        self, hand_id: int, recorded: dict[str, object]
    ) -> None:
        """Write back ANY unreadable hand column verbatim, so a round trip is faithful.

        ``_hand_from_row`` degrades a column it cannot read to a conservative
        fallback and records what it held: card columns under
        ``UNREADABLE_CARDS_KEY``, everything else under
        ``UNREADABLE_HAND_COLUMNS_KEY``. The exporter therefore emits the fallback,
        and the importer strips the marker (it is a derivation about the current
        row, and persisting it made the blocker permanent). Between them the
        blocker's PRODUCER disappeared and the text that proved the corruption was
        gone from both databases: an ordinary export/import round trip became an
        undocumented third clearing action that repairs by discarding -- including
        for the two columns UNREADABLE_HAND_COLUMNS says "cannot be repaired in the
        product".

        The card half was repaired in round 5 for the two card columns by name.
        UNREADABLE_HAND_COLUMNS arrived later with no equivalent, which is the
        enumerated-list decay this method now avoids: the columns to restore are
        read off the MARKER, intersected with the ``hands`` table's own PRAGMA
        column list, so a column added to the schema later and a marker added to
        ``DERIVED_EVIDENCE_KEYS`` later are both covered.

        ``_EVIDENCE_OWNED_COLUMNS`` is excluded, and it is the only list here: those
        are the columns the IMPORT owns rather than the payload
        (``_apply_completion_import_defaults`` re-derives the completion status and
        stamps the imported marker, ``_enforce_review_status_floor`` sets the review
        status), and ``completion_evidence`` is the channel the marker itself travels
        in -- restoring it would persist the derivation this whole mechanism exists
        to keep derived.

        Values bypass the model deliberately -- ``Hand`` refuses them, which is why
        they were degraded -- so this can only ever ADD a blocker. The recorded text
        is a ``repr``, which is what makes a stored ``'42.0'`` distinguishable from
        a stored ``42.0`` in the blocker's detail; ``literal_eval`` is the exact
        inverse for every primitive SQLite can hold, and a value it cannot invert is
        skipped rather than guessed at.

        The round-8 guard is kept and generalised the same way: a column is only
        written when it currently holds the fallback the degradation leaves behind
        (``_RESTORABLE_FALLBACKS``), so a marker cannot replace a hand's real hero
        and board cards -- the source facts every card gate derives from -- with a
        payload's marker text.
        """
        if not recorded:
            return
        columns = {
            str(row["name"])
            for row in self._execute("PRAGMA table_info(hands)").fetchall()
        } - _EVIDENCE_OWNED_COLUMNS
        for column, raw in recorded.items():
            if str(column) not in columns:
                continue
            value = _recorded_column_value(raw)
            if value is None:
                continue
            self._execute(
                f"UPDATE hands SET {column} = ? "  # noqa: S608
                f"WHERE id = ? AND ({column} IS NULL OR {column} IN (?, ?))",
                (value, hand_id, *_RESTORABLE_FALLBACKS),
            )
        self._commit()

    def update_hand_completion(
        self,
        hand_id: int,
        *,
        completion_evidence: dict[str, object],
        notes: str = "",
    ) -> Hand:
        """Persist reconstruction evidence and re-derive the completion status.

        Acknowledging a source warning is not a source-fact change, so this does
        not restale coaching, solver, or settlement evidence. An evidence write
        that WEAKENS the hand is a different matter: the evidence is precisely
        what a promotion was granted on, so the hand returns to needs_correction.
        """
        stored = self.fetch_hand(hand_id)
        if stored is None:
            raise ValueError("Hand not found.")
        # Stripped on BOTH sides: the fetched hand's evidence carries the
        # reader's own derived annotations (unreadable card/column markers),
        # which describe the current row and must never be persisted as if the
        # pipeline had produced them.
        previous = parse_completion_evidence(
            strip_derived_evidence_markers(stored.completion_evidence)
        )
        submitted = parse_completion_evidence(
            strip_derived_evidence_markers(completion_evidence)
        )
        # This door records acknowledgements and pipeline demotions; it records
        # no observations. The base of the write is therefore the STORED
        # evidence, whole, and the caller's blob moves exactly three things --
        # the code channels, and each only by ADDITION:
        #
        # * a warning or a rejection may be added (which can only ever demote)
        #   and may never be removed: a removal is a promotion of a hand whose
        #   facts nobody corrected, and only a new reconstruction may make one;
        # * an acknowledgement may be added for a warning code actually present,
        #   which is the promotion this writer exists for.
        #
        # Everything else in the blob is ignored. It used to be the other way
        # round -- the caller's blob was the base and three channels were pinned
        # from the stored row -- and that enumerated pin decayed exactly the way
        # enumerated pins do: the blob still rewrote the pipeline's OBSERVATIONS
        # (`boundary_confidence`, `terminal_event`, the partial flags), so one
        # call manufactured the evidence a promotion is granted on for a hand
        # the pipeline never finished observing; and it rewrote the open
        # ``extra`` mapping, so dropping the ``imported_from_payload`` stamp
        # walked an imported hand into the manual exemption. A blob may never
        # state something only the pipeline can observe, and inverting the merge
        # closes that for every evidence field that exists or is added later.
        #
        # Settlement attestations (`confirmed_assumption_codes`) are likewise
        # never writable through this door: the only writer that may change them
        # is `acknowledge_accounting_assumption`, whose control states the chip
        # movement being attested to.
        merged_warnings = _preserving_codes(
            previous.warning_codes, submitted.warning_codes
        )
        evidence = parse_completion_evidence(
            {
                **dump_completion_evidence(previous),
                "warning_codes": merged_warnings,
                "rejection_codes": _preserving_codes(
                    previous.rejection_codes, submitted.rejection_codes
                ),
                "acknowledged_codes": _preserving_codes(
                    previous.acknowledged_codes,
                    tuple(
                        code
                        for code in submitted.acknowledged_codes
                        if code in merged_warnings
                    ),
                ),
            }
        )
        status = derive_completion_status(evidence, source_type=stored.source_type)
        if status == "not_applicable" and stored.completion_status != "not_applicable":
            # An evidence write may record evidence; it may never move a hand
            # INTO the manual exemption. `derive_completion_status` returns
            # `not_applicable` for ANY manual row regardless of evidence, so
            # re-deriving on a hand-edited ('manual', 'complete') pair -- a pair
            # create_hand accepts -- turned one press of the generic Acknowledge
            # into the exemption: `requires_assumption_attestation` flipped to
            # False and COMPLETION_NOT_COMPLETE and
            # ACCOUNTING_ASSUMPTION_DEPENDENT vanished together.
            # `acknowledge_accounting_assumption` closes the same hazard by not
            # recomputing the column at all; this writer must recompute (it is
            # the promotion path for acknowledged warnings), so it pins the
            # exemption boundary instead.
            status = stored.completion_status
        if stored.completion_status == "partial" and not has_operator_manual_completion(
            evidence
        ):
            # Sticky, exactly as in _record_source_correction_in_evidence: no
            # evidence write restores missing footage. A hand whose column was set
            # to `partial` by a source this evidence does not repeat -- the v13
            # migration, or an import that honoured a payload's stronger claim over
            # a weaker re-derivation -- would otherwise be laundered up to
            # `complete` by one acknowledgement. Operator finalize is the sole
            # exception: see finalize_incomplete_hand.
            status = "partial"
        with self.transaction():
            self._execute(
                """
                UPDATE hands
                SET completion_status = ?, completion_evidence = ?
                WHERE id = ?
                """,
                (status, _serialize_json(dump_completion_evidence(evidence)), hand_id),
            )
            if status not in {"complete", "not_applicable"}:
                # `reviewed` never outlives the evidence it was granted on. This
                # writer re-derived the column and left review_status alone, so a
                # hand could sit at completion_status 'uncertain' -- with a
                # pipeline REJECTION in its own evidence -- while still labelled
                # 'reviewed' and still counted in the landing hero's
                # "N% marked reviewed". update_hand_status refuses to create that
                # pair; this closes the writer that created it directly.
                self._demote_reviewed_hand(hand_id)
            # Flat string maps only: correction states round-trip through
            # _parse_json_dict, which would str() a nested structure into a repr.
            self._record_hand_correction(
                HandCorrection(
                    hand_id=hand_id,
                    correction_type="hand_facts",
                    before_state={
                        "completion_status": stored.completion_status,
                        "acknowledged_codes": ", ".join(previous.acknowledged_codes),
                    },
                    after_state={
                        "completion_status": status,
                        "acknowledged_codes": ", ".join(evidence.acknowledged_codes),
                    },
                    notes=notes.strip(),
                )
            )
        refreshed = self.fetch_hand(hand_id)
        if refreshed is None:
            raise RuntimeError("Updated hand could not be reloaded.")
        return refreshed

    def update_study_inclusion(self, hand_id: int, study_inclusion: str) -> Hand:
        """Set whether this hand should be studied, skipped, or follow readiness."""
        if study_inclusion not in get_args(StudyInclusion):
            raise ValueError(f"Unknown study inclusion: {study_inclusion!r}")
        stored = self.fetch_hand(hand_id)
        if stored is None:
            raise ValueError("Hand not found.")
        if stored.study_inclusion == study_inclusion:
            return stored
        with self.transaction():
            self._execute(
                "UPDATE hands SET study_inclusion = ? WHERE id = ?",
                (study_inclusion, hand_id),
            )
            self._record_hand_correction(
                HandCorrection(
                    hand_id=hand_id,
                    correction_type="hand_facts",
                    before_state={"study_inclusion": stored.study_inclusion},
                    after_state={"study_inclusion": study_inclusion},
                    notes="Operator updated study inclusion preference.",
                )
            )
        refreshed = self.fetch_hand(hand_id)
        if refreshed is None:
            raise RuntimeError("Updated hand could not be reloaded.")
        return refreshed

    def finalize_incomplete_hand(
        self,
        hand_id: int,
        *,
        terminal_event: str,
        notes: str = "",
    ) -> Hand:
        """Operator attestation that an incomplete CV draft is now complete.

        This is the only writer allowed to clear sticky partial truncation and to
        override pipeline rejection codes by reconstructing gaps by hand (for
        example a recording that joined late on preflop). The operator must
        already have filled hero cards and acknowledged remaining warnings.
        Pipeline observation fields (partial flags, rejection codes, boundary
        confidence, evidence_version, terminal_event) are preserved; the operator
        claim lives under ``operator_manual_completion`` and
        ``operator_terminal_event``.
        """
        if terminal_event not in {"showdown", "fold_win", "hero_fold"}:
            raise ValueError(
                f"terminal_event must be showdown, fold_win, or hero_fold; "
                f"got {terminal_event!r}"
            )
        stored = self.fetch_hand(hand_id)
        if stored is None:
            raise ValueError("Hand not found.")
        if stored.source_type == "manual":
            raise ValueError("Manual hands are operator-owned; nothing to finalize.")
        if stored.completion_status not in {"partial", "uncertain"}:
            raise ValueError(
                "Only partial or uncertain reconstructed drafts can be finalized."
            )
        if not (stored.hero_cards or "").strip():
            raise ValueError(
                "Fill in hero cards before finalizing this incomplete hand."
            )
        board_tokens = [token for token in (stored.board_cards or "").split() if token]
        if terminal_event == "showdown":
            if len(board_tokens) != 5:
                raise ValueError(
                    "Showdown finalize requires five board cards; fill them in first."
                )
        previous = parse_completion_evidence(
            strip_derived_evidence_markers(stored.completion_evidence)
        )
        # Soft-blanked drafts can clear a contradictory board while the pipeline
        # still records an observed terminal. When the pipeline observed an
        # outcome, the operator must attest that same outcome (not fold past a
        # showdown, etc.).
        if previous.terminal_event in {"showdown", "fold_win", "hero_fold"}:
            if terminal_event != previous.terminal_event:
                raise ValueError(
                    f"Pipeline observed terminal_event={previous.terminal_event!r}; "
                    f"finalize as {previous.terminal_event} (or correct the hand "
                    f"facts / re-run reconstruction) instead of {terminal_event}."
                )
        if not previous.is_known:
            raise ValueError(
                "Cannot finalize a hand with no readable reconstruction evidence. "
                "Run CV reconstruction again, or enter the hand manually."
            )
        if has_operator_manual_completion(previous):
            raise ValueError("This hand has already been finalized by the operator.")
        # Rejection codes stay in evidence for audit, but finalize may override
        # them: a late-joined recording often rejects on coverage/OCR gaps even
        # when the operator reconstructed the whole action line by observation.
        # Finalize notes are optional; blank notes keep the default audit text.
        # Preserve every pipeline observation. Attestation is additive only.
        payload = dump_completion_evidence(previous)
        payload[OPERATOR_MANUAL_COMPLETION_KEY] = True
        payload["operator_terminal_event"] = terminal_event
        evidence = parse_completion_evidence(payload)
        source_type = "corrected_cv"
        status = derive_completion_status(evidence, source_type=source_type)
        if status != "complete":
            unresolved = ", ".join(evidence.unresolved_warning_codes) or "missing evidence"
            raise ValueError(
                "Finalize would not promote this hand to complete "
                f"(status would be {status}; unresolved: {unresolved}). "
                "Acknowledge source warnings and fill required facts first."
            )
        with self.transaction():
            self._execute(
                """
                UPDATE hands
                SET completion_status = ?, completion_evidence = ?, source_type = ?
                WHERE id = ?
                """,
                (
                    status,
                    _serialize_json(dump_completion_evidence(evidence)),
                    source_type,
                    hand_id,
                ),
            )
            # Stale retained analysis without the source_facts_corrected demotion:
            # finalize itself is the attestation, and recording another acknowledgeable
            # warning would immediately undo the promotion this writer exists for.
            self._stale_retained_analysis(hand_id)
            self._invalidate_hand_settlement(hand_id)
            self._execute(
                "UPDATE hands SET review_status = 'needs_correction' WHERE id = ?",
                (hand_id,),
            )
            self._record_hand_correction(
                HandCorrection(
                    hand_id=hand_id,
                    correction_type="hand_facts",
                    before_state={
                        "completion_status": stored.completion_status,
                        "partial_start": str(previous.partial_start),
                        "partial_end": str(previous.partial_end),
                        "terminal_event": previous.terminal_event,
                    },
                    after_state={
                        "completion_status": status,
                        "operator_terminal_event": terminal_event,
                        OPERATOR_MANUAL_COMPLETION_KEY: "true",
                    },
                    notes=_finalize_correction_notes(
                        notes, rejection_codes=previous.rejection_codes
                    ),
                )
            )
        refreshed = self.fetch_hand(hand_id)
        if refreshed is None:
            raise RuntimeError("Updated hand could not be reloaded.")
        return refreshed

    def update_hand_facts(self, hand: Hand, *, correction_notes: str = "") -> Hand:
        """Persist corrected source facts and retain an auditable before/after event."""

        _refuse_display_copy(hand, "correct")
        hand = Hand.model_validate(hand.model_dump())
        if hand.id is None:
            raise ValueError("Cannot update a hand without an id.")
        stored = self.fetch_hand(hand.id)
        if stored is None:
            raise ValueError("Hand not found.")
        if hand.session_id != stored.session_id or hand.hand_number != stored.hand_number:
            raise ValueError("Hand corrections cannot move or renumber the hand.")

        fields = (
            "game_type",
            "blinds_antes",
            "table_size",
            "effective_stack",
            "hero_position",
            "hero_cards",
            "board_cards",
            "pot_size",
            "result",
            "hero_bb_won",
            "tags",
            "notes",
        )
        source_type = (
            "corrected_cv"
            if stored.source_type in {"cv_import", "corrected_cv"}
            else stored.source_type
        )
        corrected = hand.model_copy(
            update={
                "source_type": source_type,
                "review_status": "needs_correction",
            }
        )
        payload = corrected.model_dump()
        # Both sides of the no-op test live in the COLUMN's space, never in the
        # model's. `stored` is what `_hand_from_row` produced, and that reader
        # normalizes and DEGRADES: an unreadable card column is blanked and
        # recorded under UNREADABLE_CARDS_KEY, and a readable one is rewritten by
        # `normalize_cards`. Comparing the submitted hand against that projection
        # made the writer skip precisely the corrections that matter. A hand whose
        # board column held 'Qd 7s' -- two cards, which no board can hold -- reads
        # back as '', so an operator correcting a preflop hand to "no board"
        # submitted '' against a stored '', the UPDATE never fired, no correction
        # was recorded, the raw column kept its corrupt text, and
        # INVALID_HERO_OR_BOARD_CARDS was permanent while the product reported
        # "Corrected facts saved". Nothing else in the product writes a card
        # column, so no reachable action cleared it.
        #
        # Comparing what would be WRITTEN against what the ROW HOLDS has no such
        # blind spot, for any column and any future reader-side normalization:
        # the writer skips only when the UPDATE provably changes no byte.
        written = {
            field: (
                _serialize_json(payload[field]) if field == "tags" else payload[field]
            )
            for field in fields
        }
        row = self._execute(
            f"SELECT {', '.join(fields)} FROM hands WHERE id = ?", (hand.id,)
        ).fetchone()
        if row is None:
            raise ValueError("Hand not found.")
        # The audit records the text that was actually replaced, which is the
        # point of the record: on a degraded column the model view and the
        # submitted value are equal and only the row shows what changed.
        before_state = {field: row[field] for field in fields}
        after_state = written
        if before_state == after_state:
            return stored

        with self.transaction():
            # Built from `fields` and bound from `written`, so the values the
            # no-op test compared are literally the values this statement writes.
            # Spelling them twice is what let the two drift in the first place.
            assignments = ", ".join(f"{field} = ?" for field in fields)
            cursor = self._execute(
                f"""
                UPDATE hands
                SET {assignments},
                    review_status = 'needs_correction', source_type = ?
                WHERE id = ?
                """,
                (
                    *(written[field] for field in fields),
                    payload["source_type"],
                    payload["id"],
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Hand not found.")
            self._record_hand_correction(
                HandCorrection(
                    hand_id=hand.id,
                    correction_type="hand_facts",
                    before_state=before_state,
                    after_state=after_state,
                    notes=correction_notes.strip(),
                )
            )
            self._invalidate_hand_derivatives(hand.id, force_review_status=True)

        refreshed = self.fetch_hand(hand.id)
        if refreshed is None:
            raise RuntimeError("Corrected hand could not be reloaded.")
        return refreshed

    def create_hand_correction(self, correction: HandCorrection) -> HandCorrection:
        with self.transaction():
            return self._record_hand_correction(correction)

    def _record_hand_correction(self, correction: HandCorrection) -> HandCorrection:
        if self.fetch_hand(correction.hand_id) is None:
            raise ValueError("Hand not found.")
        payload = correction.model_dump()
        cursor = self._execute(
            """
            INSERT INTO hand_corrections (
                hand_id, correction_type, before_state, after_state, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["correction_type"],
                _serialize_json(payload["before_state"]),
                _serialize_json(payload["after_state"]),
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        return correction.model_copy(update={"id": cursor.lastrowid})

    def fetch_hand_corrections(self, hand_id: int) -> list[HandCorrection]:
        rows = self._execute(
            """
            SELECT * FROM hand_corrections
            WHERE hand_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (hand_id,),
        ).fetchall()
        return [_hand_correction_from_row(row) for row in rows]

    def create_hand_issue(
        self, issue: HandIssue, *, apply_workflow: bool = True
    ) -> HandIssue:
        """Save a debug-later report and freeze the evidence visible at flag time."""

        hand = self.fetch_hand(issue.hand_id)
        if hand is None:
            raise ValueError("Hand not found.")
        snapshot = issue.evidence_snapshot or {
            "hand": hand.model_dump(mode="json"),
            "session": (
                None
                if (session := self.fetch_session(hand.session_id)) is None
                else session.model_dump(mode="json")
            ),
            "players": [
                player.model_dump(mode="json")
                for player in self.fetch_players_by_hand(issue.hand_id)
            ],
            "actions": [
                action.model_dump(mode="json")
                for action in self.fetch_actions_by_hand(issue.hand_id)
            ],
            "corrections": [
                correction.model_dump(mode="json")
                for correction in self.fetch_hand_corrections(issue.hand_id)
            ],
        }
        saved_issue = issue.model_copy(
            update={
                "description": issue.description.strip(),
                "evidence_snapshot": snapshot,
            }
        )
        payload = saved_issue.model_dump()
        with self.transaction():
            cursor = self._execute(
                """
                INSERT INTO hand_issues (
                    hand_id, status, issue_types, description, evidence_snapshot,
                    resolution_notes, created_at, updated_at, resolved_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["hand_id"],
                    payload["status"],
                    _serialize_json(payload["issue_types"]),
                    payload["description"],
                    _serialize_json(payload["evidence_snapshot"]),
                    payload["resolution_notes"],
                    _serialize_datetime(payload["created_at"]),
                    _serialize_datetime(payload["updated_at"]),
                    _serialize_optional_datetime(payload["resolved_at"]),
                ),
            )
            if apply_workflow and saved_issue.status == "open":
                self._flag_hand_for_debugging(saved_issue.hand_id)
        return saved_issue.model_copy(update={"id": cursor.lastrowid})

    def fetch_hand_issues(
        self,
        *,
        hand_id: int | None = None,
        status: str | None = None,
    ) -> list[HandIssue]:
        """Hand issues, filtered in the MODEL's space.

        The ``status`` filter is applied to ``_hand_issue_from_row``'s verdict,
        not to the column: a row this build cannot fully read is forced to
        ``open``, and ``fetch_hand_issues(status="open")`` has to return it or the
        queue that lists open issues disagrees with the blocker that counts them.
        See ``_MODEL_SPACE_CLASSIFICATION``.
        """
        clauses: list[str] = []
        params: list[object] = []
        if hand_id is not None:
            clauses.append("hand_id = ?")
            params.append(hand_id)
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        rows = self._execute(
            f"""
            SELECT * FROM hand_issues
            {where}
            ORDER BY created_at DESC, id DESC
            """,
            tuple(params),
        ).fetchall()
        issues = [_hand_issue_from_row(row) for row in rows]
        if status is None:
            return issues
        return [issue for issue in issues if issue.status == status]

    def fetch_hand_issue(self, issue_id: int) -> HandIssue | None:
        """One issue, read through the same model-space rule as the queue."""
        row = self._execute(
            "SELECT * FROM hand_issues WHERE id = ?", (issue_id,)
        ).fetchone()
        return None if row is None else _hand_issue_from_row(row)

    def _regression_blocker(self, issue_row: sqlite3.Row) -> str | None:
        """Why this issue may not be closed yet, or None when it may.

        A regression must have been observed BOTH failing for the original
        defect and passing after the fix. One without the other proves nothing:
        a test that only ever passed may not exercise the defect at all.

        Which issues the gate covers is read from ``issue_types``, so a row whose
        categories could not be read is gated too. The salvage falls back to
        ``other`` -- the one category outside the set -- and it cannot show that
        the operator did not file this under ``pot_or_result``. A degraded row
        may only ever ADD a requirement, never clear one, which is the same rule
        that forces its status to ``open``.
        """
        issue = _hand_issue_from_row(issue_row)
        categories_unreadable = "issue_types" in issue.unreadable_columns
        if not categories_unreadable and not RELEASE_BLOCKING_ISSUE_TYPES.intersection(
            issue.issue_types
        ):
            return None
        rows = self._execute(
            """
            SELECT failing_before, passing_after FROM regression_cases
            WHERE issue_id = ?
            """,
            (issue_row["id"],),
        ).fetchall()
        if not rows:
            if categories_unreadable:
                return (
                    "This issue's stored categories could not be read, so it is "
                    "treated as release-blocking and closing it needs a permanent "
                    "regression. Promote it to a regression case first."
                )
            return (
                "This issue is release-blocking, so closing it needs a permanent "
                "regression. Promote it to a regression case first."
            )
        if any(row["failing_before"] and row["passing_after"] for row in rows):
            return None
        return (
            "The linked regression is not proven yet: it must be observed failing "
            "for the original defect and passing after the fix."
        )

    def resolve_hand_issue(
        self, issue_id: int, *, resolution_notes: str
    ) -> HandIssue:
        """Resolve the issue OPEN_DEBUGGING_ISSUE names, in the MODEL's space.

        The open-ness test goes through ``_hand_issue_from_row``, which forces
        ``open`` on a row it cannot fully read. A ``WHERE status = 'open'`` clause
        answered in the column's space instead, so a stored ``'in_progress'``
        raised the blocker, drew this form, and then refused the submission with
        "Open hand issue not found." — the blocker's own clearing action rejecting
        the row it was drawn for. See ``_MODEL_SPACE_CLASSIFICATION``.
        """
        notes = resolution_notes.strip()
        if not notes:
            raise ValueError("Resolution notes are required.")
        now = datetime.now(UTC)
        with self.transaction():
            row = self._execute(
                "SELECT * FROM hand_issues WHERE id = ?", (issue_id,)
            ).fetchone()
            if row is None or _hand_issue_from_row(row).status != "open":
                raise ValueError("Open hand issue not found.")
            # The regression gate is enforced by the writer, not by whichever UI
            # happens to call it. Putting it in a service would leave this
            # method as an ungated side door onto the same table.
            blocker = self._regression_blocker(row)
            if blocker is not None:
                raise ValueError(blocker)
            cursor = self._execute(
                """
                UPDATE hand_issues
                SET status = 'resolved', resolution_notes = ?,
                    updated_at = ?, resolved_at = ?
                WHERE id = ?
                """,
                (
                    notes,
                    _serialize_datetime(now),
                    _serialize_datetime(now),
                    issue_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Open hand issue not found.")
        row = self._execute(
            "SELECT * FROM hand_issues WHERE id = ?", (issue_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("Resolved hand issue could not be reloaded.")
        return _hand_issue_from_row(row)

    def _flag_hand_for_debugging(self, hand_id: int) -> None:
        reason = "Hand was flagged for future debugging."
        # This path writes its own UPDATE instead of calling
        # _invalidate_hand_derivatives, so it repeats the completion demotion --
        # and, like every other demotion, records it in the evidence. Demoting the
        # column alone stranded the hand permanently: the stored evidence still
        # derived 'complete', so COMPLETION_NOT_COMPLETE fired forever while the
        # Source warnings panel it points at was never drawn.
        self._execute(
            """
            UPDATE hands
            SET review_status = 'needs_correction',
                completion_status = CASE
                    WHEN completion_status = 'complete' THEN 'uncertain'
                    ELSE completion_status
                END
            WHERE id = ?
            """,
            (hand_id,),
        )
        self._record_source_correction_in_evidence(hand_id, code=DEBUGGING_FLAG_CODE)
        self._execute(
            """
            UPDATE hand_reviews
            SET is_stale = 1, stale_reason = ?
            WHERE hand_id = ?
            """,
            (reason, hand_id),
        )
        self._execute(
            """
            UPDATE coaching_reviews
            SET is_stale = 1, stale_reason = ?
            WHERE hand_id = ? AND review_type = 'hand'
            """,
            (reason, hand_id),
        )
        self._execute(
            """
            UPDATE coaching_reviews
            SET is_stale = 1,
                stale_reason = 'A hand in this session was flagged for future debugging.'
            WHERE review_type = 'session'
              AND session_id = (SELECT session_id FROM hands WHERE id = ?)
            """,
            (hand_id,),
        )
        self._execute(
            """
            UPDATE solver_runs
            SET status = CASE
                    WHEN status IN ('queued', 'running') THEN 'cancelling'
                    ELSE 'stale'
                END,
                error_message = 'Hand was flagged for future debugging.'
            WHERE hand_id = ? AND status IN ('queued', 'running', 'completed')
            """,
            (hand_id,),
        )

    def move_session_coaching_reviews(self, from_session_id: int, to_session_id: int) -> int:
        """Re-parent session-level coaching so a session delete cannot cascade it away.

        ``move_hand_to_session`` only re-parents ``review_type='hand'`` rows, which
        left session-level reviews owned by a session the caller was about to
        delete. They are marked stale because they were written about a different
        set of hands.
        """
        if from_session_id == to_session_id:
            return 0
        cursor = self._execute(
            """
            UPDATE coaching_reviews
            SET session_id = ?,
                is_stale = 1,
                stale_reason = 'Hands moved into this session; rerun coaching.'
            WHERE review_type = 'session' AND session_id = ?
            """,
            (to_session_id, from_session_id),
        )
        self._commit()
        return cursor.rowcount

    def move_hand_to_session(self, hand_id: int, session_id: int) -> Hand:
        """Move a hand between sessions and resolve hand-number collisions safely."""

        hand = self.fetch_hand(hand_id)
        if hand is None:
            raise ValueError("Hand not found.")
        if self.fetch_session(session_id) is None:
            raise ValueError("Target session not found.")
        if hand.session_id == session_id:
            return hand

        with self.transaction():
            existing_numbers = {
                item.hand_number for item in self.fetch_hands_by_session(session_id)
            }
            hand_number = hand.hand_number
            if hand_number in existing_numbers:
                hand_number = max(existing_numbers, default=0) + 1
            self._execute(
                "UPDATE hands SET session_id = ?, hand_number = ? WHERE id = ?",
                (session_id, hand_number, hand_id),
            )
            self._execute(
                """
                UPDATE coaching_reviews
                SET session_id = ?
                WHERE review_type = 'hand' AND hand_id = ?
                """,
                (session_id, hand_id),
            )
            self._execute(
                """
                UPDATE coaching_reviews
                SET is_stale = 1,
                    stale_reason = 'A hand moved into or out of this session; rerun coaching.'
                WHERE review_type = 'session' AND session_id IN (?, ?)
                """,
                (hand.session_id, session_id),
            )

        moved = self.fetch_hand(hand_id)
        if moved is None:
            raise RuntimeError("Moved hand could not be reloaded.")
        return moved

    def update_hand_accounting_evidence(
        self,
        hand_id: int,
        *,
        pot_size: float | None,
        hero_bb_won: float | None,
    ) -> None:
        """Rewrite the recorded pot and hero result -- the hand's OBSERVED summary.

        The authoritative gate lives in
        ``services.settlement_sync.sync_recorded_figures_from_ledger``, which is the
        only caller: it refuses unless ``accounting_is_established`` holds, and this
        module cannot ask that question without inverting the layering
        (``hand_accounting`` imports ``db``).

        What db.py CAN measure is ``_declared_chips_taken`` -- does this hand's
        stored settlement declaration actually move chips? -- and it refuses on that
        alone when the hand has attested to nothing, which is the defence in depth
        the round-10 lesson asks for: "fixing the call site fixes one call site".
        Any figure derived under a chip-moving declaration that nobody has confirmed
        is refused here whatever the caller believes, so a second writer added later
        cannot re-open the hole. Once an attestation exists, the narrower question
        this layer can ask is answered and the service-layer gate is the one that
        decides.
        """
        stored = self.fetch_hand(hand_id)
        if stored is None:
            raise ValueError("Hand not found.")
        settlement = self.fetch_hand_settlement(hand_id)
        declared = (
            {}
            if settlement is None
            else self._declared_chips_taken(settlement)
        )
        if any(abs(float(amount)) > 0 for amount in declared.values()):
            evidence = parse_completion_evidence(stored.completion_evidence)
            # The manual-hand exemption applies here for the same reason it applies
            # to the blocker: on a hand this operator entered in this database a
            # declared ante or room rake is their own observation, and no
            # attestation control is drawn for it anywhere in the product.
            owes_attestation = requires_assumption_attestation(
                source_type=stored.source_type,
                completion_status=stored.completion_status,
                evidence=evidence,
            )
            if owes_attestation and not evidence.confirmed_assumption_codes:
                moved = ", ".join(
                    f"{name} {float(amount):g}"
                    for name, amount in sorted(declared.items())
                    if abs(float(amount)) > 0
                )
                raise ValueError(
                    "Refusing to record a derived pot or hero result on a hand "
                    f"whose settlement declaration moves chips ({moved}) and which "
                    "has attested to none of them; confirm the assumption in Study "
                    "→ Summary → Accounting reconciliation first."
                )
        before_state = {
            "pot_size": stored.pot_size,
            "hero_bb_won": stored.hero_bb_won,
        }
        after_state = {"pot_size": pot_size, "hero_bb_won": hero_bb_won}
        if before_state == after_state:
            return
        with self.transaction():
            cursor = self._execute(
                """
                UPDATE hands
                SET pot_size = ?, hero_bb_won = ?,
                    review_status = 'needs_correction',
                    source_type = CASE
                        WHEN source_type IN ('cv_import', 'corrected_cv') THEN 'corrected_cv'
                        ELSE source_type
                    END
                WHERE id = ?
                """,
                (pot_size, hero_bb_won, hand_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Hand not found.")
            self._record_hand_correction(
                HandCorrection(
                    hand_id=hand_id,
                    correction_type="hand_facts",
                    before_state=before_state,
                    after_state=after_state,
                    notes="Updated accounting evidence.",
                )
            )
            self._invalidate_hand_derivatives(hand_id, force_review_status=True)

    def delete_hand(self, hand_id: int) -> None:
        """Delete one hand, and stop presenting session coaching that described it.

        Every hand-scoped row cascades, but a `review_type='session'` coaching
        review does not: it summarises hands it no longer has. `create_hand_player`
        stales it when a hand GAINS a seat and `move_hand_to_session` stales it
        when a hand leaves the session, so removing one entirely must too --
        otherwise the session review keeps rendering as CURRENT while the hands it
        reports on are gone.
        """
        with self.transaction():
            row = self._execute(
                "SELECT session_id FROM hands WHERE id = ?", (hand_id,)
            ).fetchone()
            if row is None:
                return
            self._execute("DELETE FROM hands WHERE id = ?", (hand_id,))
            self._execute(
                """
                UPDATE coaching_reviews
                SET is_stale = 1,
                    stale_reason = 'A hand was removed from this session; rerun coaching.'
                WHERE review_type = 'session' AND session_id = ?
                """,
                (row["session_id"],),
            )

    def create_hand_player(self, player: HandPlayer) -> HandPlayer:
        payload = player.model_dump()
        self._validate_single_hero(payload["hand_id"], payload["is_hero"], exclude_player_id=None)
        cursor = self._execute(
            """
            INSERT INTO hand_players (
                hand_id, player_key, seat_index, player_name, position,
                starting_stack, is_hero, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["player_key"],
                payload["seat_index"],
                payload["player_name"],
                payload["position"],
                payload["starting_stack"],
                int(payload["is_hero"]),
                payload["notes"],
            ),
        )
        self._invalidate_hand_derivatives(player.hand_id)
        self._demote_reviewed_hand(player.hand_id)
        self._commit()
        return player.model_copy(update={"id": cursor.lastrowid})

    def update_hand_player(
        self, player: HandPlayer, *, correction_notes: str = ""
    ) -> HandPlayer:
        if player.id is None:
            raise ValueError("Cannot update a player without an id.")
        payload = player.model_dump()
        with self.transaction():
            stored_row = self._execute(
                "SELECT * FROM hand_players WHERE id = ?", (payload["id"],)
            ).fetchone()
            if stored_row is None:
                raise ValueError("Cannot update a player that no longer exists.")
            stored = _hand_player_from_row(stored_row)
            if stored.hand_id != payload["hand_id"] or stored.player_key != payload["player_key"]:
                raise ValueError("Saved player identity no longer matches this hand.")
            self._validate_single_hero(
                payload["hand_id"],
                payload["is_hero"],
                exclude_player_id=payload["id"],
            )
            cursor = self._execute(
                """
                UPDATE hand_players
                SET seat_index = ?, player_name = ?, position = ?,
                    starting_stack = ?, is_hero = ?, notes = ?
                WHERE id = ? AND hand_id = ? AND player_key = ?
                """,
                (
                    payload["seat_index"],
                    payload["player_name"],
                    payload["position"],
                    payload["starting_stack"],
                    int(payload["is_hero"]),
                    payload["notes"],
                    payload["id"],
                    payload["hand_id"],
                    payload["player_key"],
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Saved player identity no longer matches this hand.")
            self._execute(
                """
                UPDATE actions
                SET player_name = ?, position = ?
                WHERE hand_id = ? AND player_key = ?
                """,
                (
                    payload["player_name"],
                    payload["position"],
                    payload["hand_id"],
                    payload["player_key"],
                ),
            )
            self._execute(
                """
                UPDATE settlement_entries
                SET player_name = ?
                WHERE hand_id = ? AND player_key = ?
                """,
                (
                    payload["player_name"],
                    payload["hand_id"],
                    payload["player_key"],
                ),
            )
            updated = player.model_copy()
            if stored != updated:
                self._record_hand_correction(
                    HandCorrection(
                        hand_id=player.hand_id,
                        correction_type="player_update",
                        before_state=stored.model_dump(mode="json"),
                        after_state=updated.model_dump(mode="json"),
                        notes=correction_notes.strip(),
                    )
                )
                self._invalidate_hand_derivatives(
                    payload["hand_id"], force_review_status=True
                )
        return updated

    def _validate_single_hero(
        self,
        hand_id: int,
        is_hero: bool,
        *,
        exclude_player_id: int | None,
    ) -> None:
        """Refuse a second hero, deciding heroism the way the reader decides it.

        ``_hand_player_from_row`` answers ``bool(is_hero)``, so a stored ``2``
        reads as the hero while a ``WHERE is_hero = 1`` guard did not see it and
        accepted a second one. See ``_MODEL_SPACE_CLASSIFICATION``.
        """
        if not is_hero:
            return
        rows = self._execute(
            """
            SELECT * FROM hand_players
            WHERE hand_id = ? AND (? IS NULL OR id != ?)
            """,
            (hand_id, exclude_player_id, exclude_player_id),
        ).fetchall()
        if any(_hand_player_from_row(row).is_hero for row in rows):
            raise ValueError("A hand can have only one Hero player.")

    def create_action(self, action: Action) -> Action:
        payload = action.model_dump()
        with self.transaction():
            self._resolve_action_player(payload)
            action_index = payload["action_index"] or self.next_action_index(
                payload["hand_id"], payload["street"]
            )
            self._assert_action_index_available(payload["hand_id"], payload["street"], action_index)
            return self._insert_action(action, payload, action_index)

    def create_corrected_action(
        self, action: Action, *, correction_notes: str = ""
    ) -> Action:
        """Add an action during review and preserve it as correction evidence."""

        with self.transaction():
            saved = self.create_action(action)
            self._record_hand_correction(
                HandCorrection(
                    hand_id=saved.hand_id,
                    correction_type="action_create",
                    before_state={},
                    after_state=saved.model_dump(mode="json"),
                    notes=correction_notes.strip(),
                )
            )
            self._invalidate_hand_derivatives(
                saved.hand_id, force_review_status=True
            )
        return saved

    def _insert_action(self, action: Action, payload: dict, action_index: int) -> Action:
        cursor = self._execute(
            """
            INSERT INTO actions (
                hand_id, player_key, street, action_index, player_name, position,
                action_type, amount, amount_semantics, forced_bet_type,
                is_live_post, pot_before, stack_before, notes, source_image
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["player_key"],
                payload["street"],
                action_index,
                payload["player_name"],
                payload["position"],
                payload["action_type"],
                payload["amount"],
                payload["amount_semantics"],
                payload["forced_bet_type"],
                (None if payload["is_live_post"] is None else int(payload["is_live_post"])),
                payload["pot_before"],
                payload["stack_before"],
                payload["notes"],
                payload["source_image"],
            ),
        )
        self._invalidate_hand_derivatives(payload["hand_id"])
        self._demote_reviewed_hand(payload["hand_id"])
        self._commit()
        return action.model_copy(
            update={
                "id": cursor.lastrowid,
                "action_index": action_index,
                "player_key": payload["player_key"],
            }
        )

    def next_action_index(self, hand_id: int, street: str) -> int:
        row = self._execute(
            """
            SELECT COALESCE(MAX(action_index), 0) + 1 AS next_index
            FROM actions
            WHERE hand_id = ? AND street = ?
            """,
            (hand_id, street),
        ).fetchone()
        return int(row["next_index"])

    def set_action_source_image(self, action_id: int, source_image: str) -> None:
        """Record which frame produced an action, without touching anything else.

        Provenance repair for rows imported before schema 16, so it writes no
        correction record and does not demote a reviewed hand — nothing about
        the hand's facts changes.
        """

        self._execute(
            "UPDATE actions SET source_image = ? "
            "WHERE id = ? AND (source_image IS NULL OR TRIM(source_image) = '')",
            (source_image, action_id),
        )
        self._commit()

    def update_action(
        self, action: Action, *, correction_notes: str = ""
    ) -> Action:
        if action.id is None:
            raise ValueError("Cannot update an action without an id.")
        payload = action.model_dump()
        with self.transaction():
            stored_row = self._execute(
                "SELECT * FROM actions WHERE id = ?", (payload["id"],)
            ).fetchone()
            if stored_row is None:
                raise ValueError("Cannot update an action that no longer exists.")
            stored = _action_from_row(stored_row)
            stored_hand_id = stored.hand_id
            if stored_hand_id != payload["hand_id"]:
                raise ValueError("Action does not belong to the requested hand.")
            self._resolve_action_player(payload)
            action_index = payload["action_index"] or self.next_action_index(
                stored_hand_id, payload["street"]
            )
            self._assert_action_index_available(
                stored_hand_id,
                payload["street"],
                action_index,
                exclude_action_id=payload["id"],
            )
            payload["action_index"] = action_index
            self._update_action_row(payload, stored_hand_id)
            updated = action.model_copy(
                update={
                    "action_index": action_index,
                    "player_key": payload["player_key"],
                    # The UPDATE never writes source_image, and callers rebuild
                    # the row without it, so comparing it would make every CV
                    # row look changed — un-approving the hand and recording a
                    # correction that claims provenance was cleared.
                    "source_image": stored.source_image,
                }
            )
            if stored != updated:
                self._record_hand_correction(
                    HandCorrection(
                        hand_id=stored_hand_id,
                        correction_type="action_update",
                        before_state=stored.model_dump(mode="json"),
                        after_state=updated.model_dump(mode="json"),
                        notes=correction_notes.strip(),
                    )
                )
                self._invalidate_hand_derivatives(
                    stored_hand_id, force_review_status=True
                )
        return updated

    def _update_action_row(self, payload: dict, stored_hand_id: int) -> None:
        cursor = self._execute(
            """
            -- source_image is deliberately not updated: it records which
            -- frame produced this line, and must survive the operator
            -- correcting the street, order, actor, type, or amount.
            UPDATE actions
            SET player_key = ?, street = ?, action_index = ?, player_name = ?, position = ?,
                action_type = ?, amount = ?, amount_semantics = ?,
                forced_bet_type = ?, is_live_post = ?,
                pot_before = ?, stack_before = ?, notes = ?
            WHERE id = ? AND hand_id = ?
            """,
            (
                payload["player_key"],
                payload["street"],
                payload["action_index"],
                payload["player_name"],
                payload["position"],
                payload["action_type"],
                payload["amount"],
                payload["amount_semantics"],
                payload["forced_bet_type"],
                (None if payload["is_live_post"] is None else int(payload["is_live_post"])),
                payload["pot_before"],
                payload["stack_before"],
                payload["notes"],
                payload["id"],
                stored_hand_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("Saved action identity no longer matches this hand.")

    def _assert_action_index_available(
        self,
        hand_id: int,
        street: str,
        action_index: int,
        *,
        exclude_action_id: int | None = None,
    ) -> None:
        row = self._execute(
            """
            SELECT id
            FROM actions
            WHERE hand_id = ? AND street = ? AND action_index = ?
              AND (? IS NULL OR id != ?)
            LIMIT 1
            """,
            (
                hand_id,
                street,
                action_index,
                exclude_action_id,
                exclude_action_id,
            ),
        ).fetchone()
        if row is not None:
            raise ValueError("Action index must be unique within each hand and street.")

    def delete_action(self, action_id: int, *, correction_notes: str = "") -> None:
        with self.transaction():
            row = self._execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
            self._execute("DELETE FROM actions WHERE id = ?", (action_id,))
            if row is not None:
                stored = _action_from_row(row)
                self._record_hand_correction(
                    HandCorrection(
                        hand_id=stored.hand_id,
                        correction_type="action_delete",
                        before_state=stored.model_dump(mode="json"),
                        after_state={},
                        notes=correction_notes.strip(),
                    )
                )
                self._invalidate_hand_derivatives(
                    stored.hand_id, force_review_status=True
                )

    def _invalidate_hand_settlement(self, hand_id: int) -> None:
        self._execute(
            """
            UPDATE hand_settlements
            SET status = 'needs_correction',
                is_balanced = 0,
                warnings = ?,
                updated_at = ?
            WHERE hand_id = ?
            """,
            (
                _serialize_json(["Players or actions changed; reconcile again."]),
                _serialize_datetime(datetime.now(UTC)),
                hand_id,
            ),
        )

    def _demote_reviewed_hand(self, hand_id: int) -> None:
        """Return a promoted hand to needs_correction after an evidence change.

        Deliberately narrower than _invalidate_hand_derivatives: adding a seat, an
        action, or a settlement award does not change the reconstruction's
        boundaries, so completion evidence, coaching, and solver runs are left
        alone. What it does change is the basis the promotion was granted on, and
        'reviewed' must never outlive that basis -- the Study accounting editor
        could otherwise award the pot to the wrong player on a hand that stayed
        reviewed everywhere the UI reads review_status.
        """
        self._execute(
            """
            UPDATE hands
            SET review_status = 'needs_correction'
            WHERE id = ? AND review_status = 'reviewed'
            """,
            (hand_id,),
        )

    def _invalidate_hand_derivatives(
        self, hand_id: int, *, force_review_status: bool = False
    ) -> None:
        """Flag derived reviews and reconciliation state after source edits."""
        has_saved_hand_review = self._execute(
            """
            SELECT (
                EXISTS(SELECT 1 FROM hand_reviews WHERE hand_id = ?)
                OR EXISTS(
                    SELECT 1 FROM coaching_reviews
                    WHERE hand_id = ? AND review_type = 'hand'
                )
            ) AS has_review
            """,
            (hand_id, hand_id),
        ).fetchone()
        self._invalidate_hand_settlement(hand_id)
        self._stale_retained_analysis(hand_id)
        if force_review_status or (
            has_saved_hand_review is not None and bool(has_saved_hand_review["has_review"])
        ):
            self._execute(
                "UPDATE hands SET review_status = 'needs_correction' WHERE id = ?",
                (hand_id,),
            )
            self._record_source_correction_in_evidence(hand_id)

    def _stale_retained_analysis(self, hand_id: int) -> None:
        """Stop presenting retained coaching and solver output as current.

        Split out of _invalidate_hand_derivatives so the settlement writers can
        reuse it without also invalidating the settlement row they are authoring
        and without rewriting completion evidence: re-assigning a pot award does
        not change the reconstruction's boundaries, but it does change the
        winners, the ledger and the hero result that the coaching prompt and the
        solver input were built from. Demoting review_status alone was cosmetic --
        one click restored it, with wrong-winner coaching still labelled CURRENT.
        """
        self._execute(
            """
            UPDATE hand_reviews
            SET is_stale = 1,
                stale_reason = 'Hand evidence changed; rerun coaching.'
            WHERE hand_id = ?
            """,
            (hand_id,),
        )
        self._execute(
            """
            UPDATE coaching_reviews
            SET is_stale = 1,
                stale_reason = 'Hand evidence changed; rerun coaching.'
            WHERE hand_id = ? AND review_type = 'hand'
            """,
            (hand_id,),
        )
        self._execute(
            """
            UPDATE solver_runs
            SET status = CASE
                    WHEN status IN ('queued', 'running') THEN 'cancelling'
                    ELSE 'stale'
                END,
                error_message = 'Hand evidence changed; rerun solver analysis.'
            WHERE hand_id = ? AND status IN ('queued', 'running', 'completed')
            """,
            (hand_id,),
        )
        self._execute(
            """
            UPDATE coaching_reviews
            SET is_stale = 1,
                stale_reason = 'A hand in this session changed; rerun coaching.'
            WHERE review_type = 'session'
              AND session_id = (SELECT session_id FROM hands WHERE id = ?)
            """,
            (hand_id,),
        )

    def _declared_chips_taken(self, settlement: HandSettlement) -> dict[str, float]:
        """How many chips each declared settlement input actually moves, in chips.

        Measured, never enumerated. ``rake`` is the amount the stored policy
        genuinely removes from this hand's pot, so a rate with a zero cap, a
        no-flop-no-drop policy on a hand that saw no flop, and a rounding unit
        coarser than the whole rake all report 0 and are therefore not
        disclosures at all -- while any combination of those fields that DOES
        take chips reports the amount, without anyone having had to think of that
        combination in advance.

        ``dead_money`` is the declared amount itself: it is added straight into
        pot 0, so every non-zero value moves the derived gross pot and the split
        granularity by construction.

        Winners are deliberately not fetched. Neither figure depends on who was
        awarded which pot -- the rake is computed from the gross pot and the
        flop-seen fact alone -- which is what makes this safe to call from
        ``upsert_hand_settlement``, whose caller may not have written this
        hand's award rows yet.

        A ledger that refuses to build tells us nothing, so it fails closed onto
        the declaration itself: an unbuildable hand with a declared rate is
        disclosed rather than silently cleared.
        """
        hand = self.fetch_hand(settlement.hand_id)
        if hand is None:
            return {"rake": 0.0, "dead_money": float(settlement.dead_money)}
        actions = self.fetch_actions_by_hand(settlement.hand_id)
        try:
            ledger = build_ledger_from_records(
                self.fetch_players_by_hand(settlement.hand_id),
                actions,
                dead_money=settlement.dead_money,
                blinds=blind_structure(
                    settlement.small_blind,
                    settlement.big_blind,
                    settlement.straddles,
                ),
                ante_mode=settlement.ante_mode,
                rake=RakePolicy(
                    rate=settlement.rake_rate,
                    cap=settlement.rake_cap,
                    rounding_unit=settlement.rake_rounding_unit,
                    no_flop_no_drop=settlement.no_flop_no_drop,
                ),
                flop_seen=bool(hand.board_cards)
                or any(
                    action.street in {"flop", "turn", "river", "showdown"}
                    for action in actions
                ),
            )
        except LedgerError:
            return {
                "rake": float(settlement.rake_rate),
                "dead_money": float(settlement.dead_money),
            }
        return {"rake": float(ledger.rake), "dead_money": float(settlement.dead_money)}

    def acknowledge_accounting_assumption(
        self, hand_id: int, code: str, *, notes: str = ""
    ) -> bool:
        """Attest to one measured settlement-assumption dependence, by its quantity.

        Returns True when this hand now carries the attestation and False when
        the write was refused -- because the hand is exempt from attesting at
        all, or because the code names no dependence this hand currently
        measures -- so a caller can never report success for a write this
        refused. It used to return ``None`` either way, and the Study page
        flashed "Confirmed the declared rake_policy for this hand" over a write
        it had silently discarded.

        ``code`` is produced by ``hand_accounting._derive_assumption_dependence``
        and carries both the declaration and the chip movement the operator is
        attesting to, so this writer needs no policy comparison of its own: a
        re-measured quantity, or the same quantity under a different declaration,
        is a different string and is simply not covered by what was recorded
        here. The code is verified against the CURRENT measurement below, so a
        shape-valid fabrication is refused by this writer itself, not only by
        the ``attest_assumption`` door.

        The attestation is recorded ONLY in ``confirmed_assumption_codes``. It
        used to be written into ``warning_codes`` and ``acknowledged_codes`` for
        auditability, and that made it a pipeline warning: after an ordinary
        export/import round trip -- which resets acknowledged codes and keeps
        warning codes -- the attestation arrived as an unacknowledged source
        warning, demoted the completion status, and was offered to the generic
        one-click Acknowledge that clears pipeline warnings, which cleared this
        blocker without the operator ever seeing a chip figure. The audit trail is
        the ``hand_corrections`` row written below, which is where every other
        attestation in this product is retained.

        At most one attestation survives per declared input: a fresh measurement
        REPLACES the previous measurement of the same input instead of leaving
        two contradictory chip figures in the evidence, and so that a stale one
        cannot sit there waiting to re-clear the blocker if the hand is later
        corrected back.

        The scope is ``requires_assumption_attestation``, the same predicate the
        blocker is emitted under: a hand this operator entered here is exempt, and
        an imported or reconstructed hand is not. Scoping the writer on
        ``source_type == 'manual'`` alone meant a manual row whose completion
        status was anything but ``not_applicable`` emitted a blocker whose only
        stated clearing action this writer refused to perform.
        """
        if not is_assumption_dependence_code(code):
            raise ValueError(
                "Only a measured settlement-assumption dependence code can be "
                f"acknowledged here; got {code!r}."
            )
        row = self._execute(
            "SELECT source_type, completion_status, completion_evidence "
            "FROM hands WHERE id = ?",
            (hand_id,),
        ).fetchone()
        if row is None:
            raise ValueError("Hand not found.")
        evidence = parse_completion_evidence(
            _parse_json_object(row["completion_evidence"], {})
        )
        if not requires_assumption_attestation(
            source_type=row["source_type"],
            completion_status=row["completion_status"],
            evidence=evidence,
        ):
            return False
        # The code must name a dependence this hand CURRENTLY measures. The
        # shape test above validates only the prefix, so this writer used to
        # accept any well-formed string a direct caller supplied -- and, because
        # at most one attestation survives per declared input, a forged
        # `rake_policy:...` code EVICTED the operator's genuine rake attestation
        # on the way past while filing a hand_corrections row for an attestation
        # nobody made. Measuring here (rather than only in the
        # `attest_assumption` door, which every product caller uses) closes the
        # writer itself: a code naming no measured dependence is refused, nothing
        # is written, and nothing is evicted. The import is deferred because
        # `hand_accounting` imports this module at module level; at call time it
        # is already loaded, and this method is never called during import.
        from poker_tracker.services.hand_accounting import reconcile_persisted_hand

        try:
            measured = {
                item.code
                for item in reconcile_persisted_hand(self, hand_id).assumption_dependence
            }
        except LedgerError:
            # The measurement itself cannot be taken (no players, an impossible
            # action line). Fail closed: a code that cannot be verified against
            # a current measurement is refused, never recorded.
            return False
        if code not in measured:
            return False
        confirmed = confirm_assumption(evidence, code)
        if confirmed is None:
            return True
        updated = confirmed
        # completion_status is deliberately NOT recomputed here. This writer
        # records an accounting attestation, which `derive_completion_status`
        # does not read and must not be moved by: re-deriving it would let this
        # button PROMOTE a hand. It nearly did -- on a `manual` row whose stored
        # completion status was `complete` (a pair `create_hand` accepts), one
        # press re-derived `not_applicable`, which is the manual exemption, and
        # three unrelated blockers vanished with it.
        with self.transaction():
            self._execute(
                "UPDATE hands SET completion_evidence = ? WHERE id = ?",
                (
                    _serialize_json(dump_completion_evidence(updated)),
                    hand_id,
                ),
            )
            self._record_hand_correction(
                HandCorrection(
                    hand_id=hand_id,
                    correction_type="hand_facts",
                    before_state={
                        "confirmed_assumption_codes": ", ".join(
                            evidence.confirmed_assumption_codes
                        )
                    },
                    after_state={
                        "confirmed_assumption_codes": ", ".join(
                            updated.confirmed_assumption_codes
                        )
                    },
                    notes=notes
                    or (
                        "Attested to a settlement assumption the reconciliation "
                        f"depends on: {code}."
                    ),
                )
            )
        return True

    def _record_declared_chip_adjustment(
        self, hand_id: int, *, code: str, declared: bool
    ) -> None:
        """Keep one declared-chips disclosure in step with the stored settlement.

        Two settlement inputs move chips that the observed action line never
        accounts for, in opposite directions. Dead money CREATES them -- antes,
        dead blinds, a straddle from a seat that left. A rake policy DESTROYS
        them. Both are legitimate modelling inputs the ledger models faithfully,
        and both are free parameters that can be tuned until the recorded figures
        match the derived ones. Dead money makes the pot cross-check
        self-satisfying; the rake goes further and moves the DERIVED side of the
        hero-result and payout cross-checks, so comparing those exactly detects
        nothing. The reconciled verdict may rest on declared chips; it may not
        rest on them silently.

        The record goes in ``completion_evidence.declared_settlement_codes``,
        which is the OPERATOR's channel. It used to go into ``warning_codes``,
        which is the PIPELINE's: ``derive_completion_status`` demotes on an
        unresolved entry there, so declaring a rake on a hand whose reconstruction
        evidence was complete and clean turned that hand ``uncertain`` and
        produced two blockers telling the operator "The pipeline could not prove
        this hand was fully reconstructed" and "The pipeline flagged 1 unresolved
        source warning(s)" -- about a figure the pipeline neither claimed nor
        observed -- and directing them to Correct hand facts, a form with no rake
        field in it, to fix a value that exists only in the Accounting
        reconciliation panel. Round 10 gave the ATTESTATION its own channel for
        the same reason and left this half sharing.

        Nothing rests on this record. The gate is ``ACCOUNTING_ASSUMPTION_DEPENDENT``,
        derived per read in `services.hand_accounting` from the chips themselves,
        which no writer -- including one that never calls this method -- can
        bypass, and which re-measures a policy left alone while the pot it is
        applied to grows. This is the audit trail beside it.

        Manual hands are untouched: a manual hand has no pipeline claim to
        contradict -- every figure on it, the rake and the hero result alike, is
        the same operator's own entry.

        ``declared`` is derived from `_declared_chips_taken`, so the disclosure
        reports whether the declaration MOVES chips rather than whether a field
        was filled in. That removes the last field list from the writer path: a
        rate whose cap is zero, a no-flop-no-drop policy on a hand with no flop,
        and a rounding unit coarser than the whole rake all take nothing and are
        silent, and any combination that does take chips is disclosed without
        anyone having enumerated it.
        """
        row = self._execute(
            "SELECT source_type, completion_evidence FROM hands WHERE id = ?",
            (hand_id,),
        ).fetchone()
        if row is None or row["source_type"] == "manual":
            return
        evidence = parse_completion_evidence(
            _parse_json_object(row["completion_evidence"], {})
        )
        updated = set_declared_settlement_code(evidence, code, declared=declared)
        if updated is None:
            return
        # completion_status is deliberately untouched. It is the reconstruction's
        # own classification, and an operator's declaration is not evidence about
        # the reconstruction.
        self._execute(
            "UPDATE hands SET completion_evidence = ? WHERE id = ?",
            (_serialize_json(dump_completion_evidence(updated)), hand_id),
        )

    def _record_source_correction_in_evidence(
        self, hand_id: int, *, code: str = SOURCE_CORRECTION_CODE
    ) -> None:
        """Write the demotion into the evidence, not just the status column.

        Demoting the column alone left a hand whose own stored evidence still
        derived 'complete': COMPLETION_NOT_COMPLETE then fired forever with no
        reachable clearing action, because the Source warnings panel only renders
        when the evidence carries a code -- and replaying that unchanged evidence
        through update_hand_completion silently restored 'complete'.

        Recording ``code`` as an acknowledgeable warning keeps the column and the
        evidence in agreement, gives the operator the exact action the blocker
        text promises, and cannot resurrect a truncated recording:
        partial_start/partial_end are untouched, so a partial hand stays partial.
        Only ``code`` itself is rewritten, so a hand that was both corrected and
        flagged for debugging keeps both warnings and must clear both.
        """
        row = self._execute(
            "SELECT source_type, completion_status, completion_evidence "
            "FROM hands WHERE id = ?",
            (hand_id,),
        ).fetchone()
        if row is None or row["source_type"] == "manual":
            return
        evidence = parse_completion_evidence(
            _parse_json_object(row["completion_evidence"], {})
        )
        payload = dump_completion_evidence(evidence)
        warnings = [item for item in evidence.warning_codes if item != code]
        payload["warning_codes"] = [*warnings, code]
        payload["acknowledged_codes"] = [
            item for item in evidence.acknowledged_codes if item != code
        ]
        # Keep operator finalize attestation across fill-blanks edits. The new
        # SOURCE_CORRECTION_CODE warning demotes to uncertain until acknowledged;
        # sticky partial still cannot clear without the attestation remaining.
        updated = parse_completion_evidence(payload)
        status = derive_completion_status(updated, source_type=row["source_type"])
        if row["completion_status"] == "partial" and not has_operator_manual_completion(
            updated
        ):
            # Sticky: no correction restores missing footage, and a hand whose
            # evidence has become unreadable must not lose the stronger claim.
            # Operator finalize (finalize_incomplete_hand) is the sole exception.
            status = "partial"
        self._execute(
            """
            UPDATE hands
            SET completion_status = ?, completion_evidence = ?
            WHERE id = ?
            """,
            (status, _serialize_json(dump_completion_evidence(updated)), hand_id),
        )

    def _resolve_action_player(self, payload: dict) -> None:
        player_key = payload["player_key"]
        if player_key is not None:
            row = self._execute(
                """
                SELECT 1 FROM hand_players
                WHERE hand_id = ? AND player_key = ?
                """,
                (payload["hand_id"], player_key),
            ).fetchone()
            if row is None:
                raise ValueError("Action player key does not belong to this hand.")
            return
        rows = self._execute(
            """
            SELECT player_key, position
            FROM hand_players
            WHERE hand_id = ? AND player_name = ?
            ORDER BY id
            """,
            (payload["hand_id"], payload["player_name"]),
        ).fetchall()
        exact = [row for row in rows if row["position"] == payload["position"]]
        candidates = exact if len(exact) == 1 else rows
        if len(candidates) == 1:
            payload["player_key"] = candidates[0]["player_key"]
        elif rows and payload["amount_semantics"] != "unknown":
            raise ValueError("A monetary action must resolve to exactly one player in the hand.")

    def create_hand_review(self, review: HandReview) -> HandReview:
        payload = review.model_dump()
        cursor = self._execute(
            """
            INSERT INTO hand_reviews (
                hand_id, hand_summary, theory_coach, exploit_coach, ev_math_notes,
                study_lesson, next_review_question, notes, is_stale,
                stale_reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["hand_summary"],
                payload["theory_coach"],
                payload["exploit_coach"],
                payload["ev_math_notes"],
                payload["study_lesson"],
                payload["next_review_question"],
                payload["notes"],
                int(payload["is_stale"]),
                payload["stale_reason"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        return review.model_copy(update={"id": cursor.lastrowid})

    def fetch_sessions(self) -> list[Session]:
        rows = self._execute("SELECT * FROM sessions ORDER BY date_played DESC, id DESC").fetchall()
        return [_session_from_row(row) for row in rows]

    def fetch_hands_by_session(self, session_id: int) -> list[Hand]:
        rows = self._execute(
            "SELECT * FROM hands WHERE session_id = ? ORDER BY hand_number, id",
            (session_id,),
        ).fetchall()
        return [_hand_from_row(row) for row in rows]

    def fetch_all_hands(self) -> list[Hand]:
        """Return all hands for portfolio-level browsing and insights."""
        rows = self._execute("SELECT * FROM hands ORDER BY created_at DESC, id DESC").fetchall()
        return [_hand_from_row(row) for row in rows]

    def fetch_session(self, session_id: int) -> Session | None:
        row = self._execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return None if row is None else _session_from_row(row)

    def fetch_hand(self, hand_id: int) -> Hand | None:
        row = self._execute("SELECT * FROM hands WHERE id = ?", (hand_id,)).fetchone()
        return None if row is None else _hand_from_row(row)

    def fetch_actions_by_hand(self, hand_id: int) -> list[Action]:
        rows = self._execute(
            """
            SELECT * FROM actions
            WHERE hand_id = ?
            ORDER BY
                CASE street
                    WHEN 'preflop' THEN 1
                    WHEN 'flop' THEN 2
                    WHEN 'turn' THEN 3
                    WHEN 'river' THEN 4
                    WHEN 'showdown' THEN 5
                    ELSE 5
                END,
                action_index,
                id
            """,
            (hand_id,),
        ).fetchall()
        return [_action_from_row(row) for row in rows]

    def fetch_players_by_hand(self, hand_id: int) -> list[HandPlayer]:
        rows = self._execute(
            """
            SELECT * FROM hand_players
            WHERE hand_id = ?
            ORDER BY is_hero DESC, seat_index, position, id
            """,
            (hand_id,),
        ).fetchall()
        return [_hand_player_from_row(row) for row in rows]

    def upsert_hand_settlement(self, settlement: HandSettlement) -> HandSettlement:
        # Re-validated at the WRITE, not trusted because the argument is typed.
        # ``model_copy(update=...)`` -- how every editor in the app builds the row
        # it saves -- does not run model validators, so a ``HandSettlement``
        # arriving here can hold a shape its own class refuses. A transposed
        # "5/10" entered as small 10 / big 5 reached the disk that way, and the
        # reader then salvaged half of it into a smaller, valid, wrong structure.
        # Validating here makes the class's rules true of every row on disk
        # regardless of how the caller assembled it, which is the only place that
        # can be guaranteed once.
        HandSettlement.model_validate(settlement.model_dump())
        payload = settlement.model_dump()
        # Read before the write, so the invalidation below can ask whether this
        # save changed anything rather than assuming a save is a change.
        before = _declared_settlement_inputs(
            self.fetch_hand_settlement(settlement.hand_id)
        )
        # `UNREADABLE_SETTLEMENT_PREFIX` describes the row a reader could not
        # validate, so it is a derivation and no writer may persist it -- exactly
        # as `strip_derived_evidence_markers` treats the unreadable-card marker.
        # Round-tripping a degraded settlement through this writer (which
        # `persist_reconciliation` does, from `result.issues`) would otherwise
        # stamp the note permanently onto a row that is now perfectly readable.
        payload["warnings"] = [
            note
            for note in payload["warnings"]
            if not str(note).startswith(UNREADABLE_SETTLEMENT_PREFIX)
        ]
        self._execute(
            """
            INSERT INTO hand_settlements (
                hand_id, status, small_blind, big_blind, straddles, ante_mode,
                dead_money, rake_rate, rake_cap,
                rake_rounding_unit, no_flop_no_drop, gross_pot, rake_amount,
                net_pot, is_balanced, warnings, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hand_id) DO UPDATE SET
                status = excluded.status,
                small_blind = excluded.small_blind,
                big_blind = excluded.big_blind,
                straddles = excluded.straddles,
                ante_mode = excluded.ante_mode,
                dead_money = excluded.dead_money,
                rake_rate = excluded.rake_rate,
                rake_cap = excluded.rake_cap,
                rake_rounding_unit = excluded.rake_rounding_unit,
                no_flop_no_drop = excluded.no_flop_no_drop,
                gross_pot = excluded.gross_pot,
                rake_amount = excluded.rake_amount,
                net_pot = excluded.net_pot,
                is_balanced = excluded.is_balanced,
                warnings = excluded.warnings,
                updated_at = excluded.updated_at
            """,
            (
                payload["hand_id"],
                payload["status"],
                payload["small_blind"],
                payload["big_blind"],
                _serialize_json(payload["straddles"]),
                payload["ante_mode"],
                payload["dead_money"],
                payload["rake_rate"],
                payload["rake_cap"],
                payload["rake_rounding_unit"],
                int(payload["no_flop_no_drop"]),
                payload["gross_pot"],
                payload["rake_amount"],
                payload["net_pot"],
                int(payload["is_balanced"]),
                _serialize_json(payload["warnings"]),
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
            ),
        )
        # Both disclosures below are raised by MEASUREMENT, not by a list of
        # suspicious fields. `_declared_chips_taken` derives the hand's ledger ONCE,
        # under the stored policy, and reports the chips that policy removes plus
        # the dead money it declares; the code is written exactly when one of those
        # numbers is non-zero, and the attestation is bound to the number rather
        # than to the policy tuple. It is deliberately NOT the dependence rule's
        # dual reconciliation -- there is no neutral pass here and no comparison,
        # because this method is called from `upsert_hand_settlement`, before the
        # hand's award rows may exist, so its input set is strictly smaller
        # (`_declared_chips_taken`: "Winners are deliberately not fetched"). This
        # comment used to credit it with the neutral pass, contradicting both the
        # code below it and PLAN.md; the measurement it actually takes is pinned by
        # `test_the_writer_side_audit_takes_a_single_pass_measurement`.
        #
        # Enumerating fields is what failed for eight rounds. `rake_rate > 0`
        # over-disclosed (a zero cap, or no-flop-no-drop on a hand with no board,
        # takes nothing and was still announced -- training the operator to click
        # Acknowledge past disclosures that mean nothing) and under-disclosed at
        # every combination not yet demonstrated. `_rake_policy(previous) !=
        # _rake_policy(settlement)` compared the policy and never the chips, so
        # an attestation earned when the policy took 0.01 chips carried over
        # unchanged once a corrected action line made the same policy take 80.01.
        #
        # This gate is the writer-side audit trail, recorded in the operator's own
        # evidence channel. It is deliberately not the thing that decides study
        # readiness: that is derived per read from `reconcile_persisted_hand`,
        # which no writer -- including one that never calls this method -- can
        # bypass, and which re-measures a policy left alone while a corrected
        # action line grows the pot it is applied to.
        taken = self._declared_chips_taken(settlement)
        for code, key in (
            (DECLARED_DEAD_MONEY_CODE, "dead_money"),
            (DECLARED_RAKE_CODE, "rake"),
        ):
            self._record_declared_chip_adjustment(
                settlement.hand_id, code=code, declared=taken[key] != 0
            )
        saved = self.fetch_hand_settlement(settlement.hand_id)
        if saved is None:
            raise RuntimeError("Settlement upsert did not persist a row.")
        # Only a save that moved a DECLARED input invalidates what was derived
        # from it. `persist_reconciliation` re-saves this row on every call, and
        # the settlement editor nulls the derived summaries before re-deriving
        # them, so an unconditional invalidation meant that reconciling a hand
        # nothing had happened to staled its coaching, staled its saved hand
        # review, flipped its completed solver run to `stale`, cancelled a solve
        # still in flight, and demoted the hand out of `reviewed`.
        #
        # After a correction it was an ordering trap with no signal: clearing the
        # blockers needs coaching re-run AND the accounting reconciled, doing them
        # in that order silently discarded the coaching just paid for, the other
        # order worked, and nothing said so. See `_declared_settlement_inputs` for
        # why the derived columns and the cross-check's verdict are not evidence
        # changes.
        if before != _declared_settlement_inputs(saved):
            self._stale_retained_analysis(settlement.hand_id)
            self._demote_reviewed_hand(settlement.hand_id)
        self._commit()
        return saved

    def fetch_hand_settlement(self, hand_id: int) -> HandSettlement | None:
        row = self._execute(
            "SELECT * FROM hand_settlements WHERE hand_id = ?", (hand_id,)
        ).fetchone()
        return None if row is None else _hand_settlement_from_row(row)

    def fetch_reconciled_settlement_hand_ids(self) -> set[int]:
        """Every hand id whose settlement READS as reconciled, in one query.

        Exists so a list view can skip reconciling hands that provably cannot
        produce a derived hero result: ``reconcile_persisted_hand`` marks a hand
        authoritative only when its settlement row exists and reads
        ``reconciled``, and only an authoritative reconciliation is ever
        substituted into a displayed result. The skip is therefore exact rather
        than a heuristic -- a hand excluded here would have reconciled to
        ``is_authoritative=False`` and kept its stored result unchanged.

        Selection is by nothing at all and classification is
        ``_hand_settlement_from_row(row).status`` -- never ``WHERE status =
        'reconciled'`` -- because that reader forces ``status`` off ``reconciled``
        on a row this build cannot validate. A raw column predicate would answer
        the question in the column's space while the reconciliation answers it in
        the model's, which is the drift ``_MODEL_SPACE_CLASSIFICATION`` exists to
        prevent.
        """
        rows = self._execute("SELECT * FROM hand_settlements").fetchall()
        return {
            int(row["hand_id"])
            for row in rows
            if row["hand_id"] is not None
            and _hand_settlement_from_row(row).status == "reconciled"
        }

    def create_settlement_entry(self, entry: SettlementEntry) -> SettlementEntry:
        """A public writer: an award added here re-declares the winner.

        It changes who the pot was pushed to just as much as a write through
        ``replace_settlement_entries`` does -- adding a second award row for a pot
        turns a single winner into a chop and moves every derived payout -- so it
        owes the same audit trail. It used to demote and stale without recording
        a ``settlement_award_update`` correction or writing
        ``source_facts_corrected`` into the completion evidence, so the identical
        semantic change cost an acknowledgement through one writer and nothing
        through its sibling.
        """
        with self.transaction():
            before = _declared_award_state(self.fetch_settlement_entries(entry.hand_id))
            saved = self._insert_settlement_entry(entry)
            after = _declared_award_state(self.fetch_settlement_entries(entry.hand_id))
            self._record_redeclared_awards(entry.hand_id, before=before, after=after)
        return saved

    def _insert_settlement_entry(
        self, entry: SettlementEntry, *, stale_retained_analysis: bool = True
    ) -> SettlementEntry:
        payload = entry.model_dump()
        self._resolve_settlement_entry_player(payload)
        cursor = self._execute(
            """
            INSERT INTO settlement_entries (
                hand_id, entry_type, pot_index, player_key, player_name,
                amount, entry_order
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["entry_type"],
                payload["pot_index"],
                payload["player_key"],
                payload["player_name"],
                payload["amount"],
                payload["entry_order"],
            ),
        )
        # An award row changes the winner, so `reviewed` cannot outlive it and
        # the coaching and solver output derived from the old award is stale.
        # `create_settlement_entry` adds a row to whatever was there, which is
        # always a change; `replace_settlement_entries` rebuilds the whole set and
        # asks its own before/after question, so it opts out here and invalidates
        # once, only when the rebuilt set differs.
        if stale_retained_analysis:
            self._stale_retained_analysis(payload["hand_id"])
            self._demote_reviewed_hand(payload["hand_id"])
        self._commit()
        return entry.model_copy(
            update={"id": cursor.lastrowid, "player_key": payload["player_key"]}
        )

    def replace_settlement_entries(
        self, hand_id: int, entries: list[SettlementEntry]
    ) -> list[SettlementEntry]:
        if any(entry.hand_id != hand_id for entry in entries):
            raise ValueError("All settlement entries must belong to the requested hand.")
        with self.transaction():
            stored_before = self.fetch_settlement_entries(hand_id)
            before = _declared_award_state(stored_before)
            self._execute("DELETE FROM settlement_entries WHERE hand_id = ?", (hand_id,))
            # The private insert: this method took its own before/after snapshot
            # above, and the public writer's per-row snapshot would compare each
            # half-rebuilt row set against the last and record a correction per
            # entry for an unchanged declaration.
            saved = [
                self._insert_settlement_entry(entry, stale_retained_analysis=False)
                for entry in entries
            ]
            after = _declared_award_state(saved)
            # Read back rather than compared against `entries`, because the writer
            # resolves each row's player identity: the two sides have to come from
            # the same reader or an unchanged declaration compares unequal.
            #
            # Wider than the award snapshot above, which exists to describe a
            # re-declared WINNER for the audit trail. Refund rows are derived
            # rather than declared, so they record no correction, but they move
            # the net results the coaching and the solver input were built from
            # and so they still invalidate. Also fires when `entries` is empty and
            # there were awards, which clears every declared winner.
            if _settlement_entry_state(stored_before) != _settlement_entry_state(
                self.fetch_settlement_entries(hand_id)
            ):
                self._stale_retained_analysis(hand_id)
                self._demote_reviewed_hand(hand_id)
            self._record_redeclared_awards(hand_id, before=before, after=after)
        return saved

    def _record_redeclared_awards(
        self,
        hand_id: int,
        *,
        before: dict[str, str],
        after: dict[str, str],
    ) -> None:
        """Disclose a re-declared pot winner the way every other source fact is.

        The declared winner is the most consequential observed fact on a hand: it
        is the sole input the derived payouts, and therefore the hero-result
        cross-check, are computed from. Flipping it in the Accounting
        reconciliation panel used to clear ACCOUNTING_NOT_AUTHORITATIVE while
        leaving no HandCorrection, no completion-evidence disclosure and
        completion_status still 'complete' -- so the recorded hero result was
        cross-checked against a freely editable, unaudited, undisclosed
        declaration. Correcting a single board card left a permanent auditable
        record; re-assigning who won the pot left none.

        Only AWARD rows are compared, and only once the hand already had some.
        Refund rows are derived, not declared: ``persist_reconciliation`` writes
        the ledger's own refunds back through this method on the first reconcile,
        and treating that as an operator correction would demote every hand it
        touched. An empty ``before`` is the hand's first declaration -- an import,
        or the CV exporter -- not a re-declaration of anything.
        """
        if not before or before == after:
            return
        self._record_hand_correction(
            HandCorrection(
                hand_id=hand_id,
                correction_type="settlement_award_update",
                before_state=dict(before),
                after_state=dict(after),
                notes="Declared pot awards were re-declared in the settlement editor.",
            )
        )
        self._record_source_correction_in_evidence(hand_id)

    def fetch_settlement_entries(self, hand_id: int) -> list[SettlementEntry]:
        """Read one hand's declared awards and refunds.

        ``pot_index`` is a durable ORDINAL into a structure this product derives
        rather than stores, and the derivation is not frozen: commit 3c3144e
        changed which commitments cut a pot level, which moves both the count and
        the numbering of the layers of any hand containing a forced post. Rows
        written under an earlier layering therefore survive intact while the
        thing they point at moves underneath them, and the same is true of any
        payload imported from a store built by a different revision.

        MIGRATION IMPACT (none; no schema version change)

        No column, index or value is altered here, and no migration renumbers
        these rows. Renumbering would have to decide which derived layer an old
        ordinal MEANT, and the layering that produced it is not recoverable from
        the row -- only the ordinal survives -- so every rule for it is a guess,
        and a guessed award written back into a durable column is exactly the
        silently-accepted wrong result this product treats as a release blocker.
        A store that never meets the mismatch is unaffected, and an older build
        reads these rows exactly as it wrote them.

        The mismatch is instead detected and reported where the ordinal is USED:
        ``services.hand_accounting._ledger_under_declaration`` rebuilds the hand
        with the unusable awards withdrawn and reports a correction naming the
        stale claim and the layer count, so the hand becomes ``needs_correction``
        rather than an unhandled ``LedgerError``, and the operator re-declares
        the winners through the settlement editor.
        """
        rows = self._execute(
            """
            SELECT * FROM settlement_entries
            WHERE hand_id = ?
            ORDER BY
                CASE entry_type WHEN 'award' THEN 1 ELSE 2 END,
                pot_index,
                entry_order,
                id
            """,
            (hand_id,),
        ).fetchall()
        return [_settlement_entry_from_row(row) for row in rows]

    def _resolve_settlement_entry_player(self, payload: dict) -> None:
        if payload["player_key"] is not None:
            row = self._execute(
                """
                SELECT player_name FROM hand_players
                WHERE hand_id = ? AND player_key = ?
                """,
                (payload["hand_id"], payload["player_key"]),
            ).fetchone()
            if row is None:
                raise ValueError("Settlement player key does not belong to this hand.")
            return
        rows = self._execute(
            """
            SELECT player_key
            FROM hand_players
            WHERE hand_id = ? AND player_name = ?
            """,
            (payload["hand_id"], payload["player_name"]),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("A settlement entry must resolve to exactly one player in the hand.")
        payload["player_key"] = rows[0]["player_key"]

    def fetch_reviews_by_hand(self, hand_id: int) -> list[HandReview]:
        rows = self._execute(
            "SELECT * FROM hand_reviews WHERE hand_id = ? ORDER BY created_at DESC, id DESC",
            (hand_id,),
        ).fetchall()
        return [_review_from_row(row) for row in rows]

    def fetch_stale_review_hand_ids(self) -> set[int]:
        """Every hand id carrying at least one stale retained review, in two queries.

        Both retained tables, because ``_coaching_blockers`` considers both and a
        list that reported only one would show a clean row beside a blocker the
        operator cannot see the cause of.

        Selection is by nothing at all and classification is
        ``reader(row).is_stale`` -- never ``WHERE is_stale = 1`` -- so a stored
        ``2`` or ``'yes'``, which the readers degrade to stale, is counted here
        exactly as the blocker counts it. See _MODEL_SPACE_CLASSIFICATION.

        Exists so a surface listing many hands can show staleness without paying
        two queries per row; ``fetch_reviews_by_hand`` and
        ``fetch_coaching_reviews_by_hand`` remain the right calls for one hand.
        """
        stale: set[int] = set()
        for table, reader in (
            ("coaching_reviews", _coaching_response_from_row),
            ("hand_reviews", _review_from_row),
        ):
            rows = self._execute(
                f"SELECT * FROM {table}"  # noqa: S608
            ).fetchall()
            for row in rows:
                hand_id = row["hand_id"]
                if hand_id is not None and reader(row).is_stale:
                    stale.add(int(hand_id))
        return stale

    def fetch_retained_reviews_by_hand(
        self,
    ) -> dict[int, list[CoachingResponse | HandReview]]:
        """Every retained review in both tables, grouped by hand, in two queries.

        The sibling of ``fetch_stale_review_hand_ids`` for a caller that needs the
        review CONTENT rather than only the staleness flag -- the Insights theme
        index reads each review's study lesson and has to know which reviews a
        correction invalidated, so a set of hand ids is not enough.

        Two queries whatever the corpus size, because the alternative on a list
        surface is two per row. Hands with no retained review are simply absent
        from the mapping; ``build_hand_evidence`` treats a missing key as "no
        retained coaching", which is what it means.
        """
        grouped: dict[int, list[CoachingResponse | HandReview]] = {}
        for table, reader in (
            ("coaching_reviews", _coaching_response_from_row),
            ("hand_reviews", _review_from_row),
        ):
            rows = self._execute(
                f"SELECT * FROM {table} ORDER BY created_at DESC, id DESC"  # noqa: S608
            ).fetchall()
            for row in rows:
                hand_id = row["hand_id"]
                if hand_id is None:
                    continue
                grouped.setdefault(int(hand_id), []).append(reader(row))
        return grouped

    def create_coaching_response(self, response: CoachingResponse) -> CoachingResponse:
        payload = response.model_dump()
        cursor = self._execute(
            """
            INSERT INTO coaching_reviews (
                provider_name, model_name, raw_prompt, raw_response, review_type,
                safety_mode, hand_id, session_id, parsed_sections, is_stale,
                stale_reason, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["provider_name"],
                payload["model_name"],
                payload["raw_prompt"],
                payload["raw_response"],
                payload["review_type"],
                payload["safety_mode"],
                payload["hand_id"],
                payload["session_id"],
                _serialize_json(payload["parsed_sections"]),
                int(payload["is_stale"]),
                payload["stale_reason"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        return response.model_copy(update={"id": cursor.lastrowid})

    def fetch_coaching_reviews_by_hand(self, hand_id: int) -> list[CoachingResponse]:
        rows = self._execute(
            """
            SELECT * FROM coaching_reviews
            WHERE hand_id = ? AND review_type = 'hand'
            ORDER BY created_at DESC, id DESC
            """,
            (hand_id,),
        ).fetchall()
        return [_coaching_response_from_row(row) for row in rows]

    def discard_stale_coaching(self, hand_id: int) -> int:
        """Discard every stale retained coaching review for one hand.

        STALE_COACHING_EVIDENCE named exactly one clearing action -- "Re-run
        coaching in Study -> Coach" -- and the only writer that could satisfy it
        was ``create_coaching_response``, which needs a configured LLM provider.
        The Coach button is disabled when there is none, so an operator who is
        offline, whose key has rotated, or who has just imported a colleague's
        session had a permanently unstudyable hand and a blocker naming an action
        they could not take. ``import_session`` stales every imported coaching row
        by construction, so that is the ordinary case, not a corner.

        This is the twin of ``delete_solver_run``, which exists for the same
        reason. It covers both retained tables, because ``_coaching_blockers``
        considers both and clearing one would leave the blocker standing. Current
        reviews are never touched: a review that still describes the hand as it is
        now is not stale evidence presented as current, and deleting it would
        throw away the only coaching the hand has.

        Staleness is decided by the same readers the blocker reads
        (``bool(is_stale)``), never by ``WHERE is_stale = 1``: a stored ``2``,
        ``-1`` or ``'yes'`` reads stale, raised the blocker, drew this control, and
        then matched no row — so the product flashed "Discarded 0 stale coaching
        review(s)." as a SUCCESS and re-rendered the identical blocker, with no
        other control able to clear it. See ``_MODEL_SPACE_CLASSIFICATION``.
        """
        deleted = 0
        for table, reader in (
            ("coaching_reviews", _coaching_response_from_row),
            ("hand_reviews", _review_from_row),
        ):
            rows = self._execute(
                f"SELECT * FROM {table} WHERE hand_id = ?", (hand_id,)  # noqa: S608
            ).fetchall()
            stale_ids = [
                row["id"] for row in rows if reader(row).is_stale and row["id"] is not None
            ]
            if not stale_ids:
                continue
            placeholders = ", ".join("?" for _ in stale_ids)
            cursor = self._execute(
                f"DELETE FROM {table} WHERE id IN ({placeholders})",  # noqa: S608
                tuple(stale_ids),
            )
            deleted += cursor.rowcount or 0
        self._commit()
        return deleted

    def fetch_coaching_reviews_by_session(self, session_id: int) -> list[CoachingResponse]:
        rows = self._execute(
            """
            SELECT * FROM coaching_reviews
            WHERE session_id = ? AND review_type = 'session'
            ORDER BY created_at DESC, id DESC
            """,
            (session_id,),
        ).fetchall()
        return [_coaching_response_from_row(row) for row in rows]

    def create_solver_range_profile(
        self, profile: SolverRangeProfile
    ) -> SolverRangeProfile:
        payload = profile.model_dump()
        cursor = self._execute(
            """
            INSERT INTO solver_range_profiles (
                name, notation, table_size, position, scenario, pot_type,
                stack_bb, description, source, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["name"].strip(),
                payload["notation"].strip(),
                payload["table_size"],
                payload["position"].strip(),
                payload["scenario"].strip(),
                payload["pot_type"].strip(),
                payload["stack_bb"],
                payload["description"].strip(),
                payload["source"],
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
            ),
        )
        self._commit()
        return profile.model_copy(update={"id": cursor.lastrowid})

    def update_solver_range_profile(
        self, profile: SolverRangeProfile
    ) -> SolverRangeProfile:
        if profile.id is None:
            raise ValueError("Cannot update a solver range profile without an id.")
        payload = profile.model_dump()
        cursor = self._execute(
            """
            UPDATE solver_range_profiles
            SET name = ?, notation = ?, table_size = ?, position = ?,
                scenario = ?, pot_type = ?, stack_bb = ?, description = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                payload["name"].strip(),
                payload["notation"].strip(),
                payload["table_size"],
                payload["position"].strip(),
                payload["scenario"].strip(),
                payload["pot_type"].strip(),
                payload["stack_bb"],
                payload["description"].strip(),
                _serialize_datetime(payload["updated_at"]),
                payload["id"],
            ),
        )
        self._commit()
        if cursor.rowcount != 1:
            raise ValueError("Solver range profile not found.")
        return profile

    def fetch_solver_range_profiles(self) -> list[SolverRangeProfile]:
        rows = self._execute(
            "SELECT * FROM solver_range_profiles ORDER BY name COLLATE NOCASE, id"
        ).fetchall()
        return [_solver_range_profile_from_row(row) for row in rows]

    def delete_solver_range_profile(self, profile_id: int) -> None:
        self._execute("DELETE FROM solver_range_profiles WHERE id = ?", (profile_id,))
        self._commit()

    def create_solver_run(self, run: SolverRun) -> SolverRun:
        payload = run.model_dump()
        cursor = self._execute(
            """
            INSERT INTO solver_runs (
                hand_id, status, backend_name, backend_version, run_parameters,
                input_hash,
                spot, range_ip, range_oop, assumptions, evidence,
                command_path, result_path, log_path, exploitability_pct,
                runtime_seconds, error_message, pid, heartbeat_at, created_at,
                started_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["status"],
                payload["backend_name"],
                payload["backend_version"],
                _serialize_json(payload["run_parameters"]),
                payload["input_hash"],
                _serialize_json(payload["spot"]),
                _serialize_json(payload["range_ip"]),
                _serialize_json(payload["range_oop"]),
                _serialize_json(payload["assumptions"]),
                _serialize_json(payload["evidence"]),
                payload["command_path"],
                payload["result_path"],
                payload["log_path"],
                payload["exploitability_pct"],
                payload["runtime_seconds"],
                _scrubbed_job_text(payload["error_message"]),
                payload["pid"],
                _serialize_optional_datetime(payload["heartbeat_at"]),
                _serialize_datetime(payload["created_at"]),
                _serialize_optional_datetime(payload["started_at"]),
                _serialize_optional_datetime(payload["completed_at"]),
            ),
        )
        self._commit()
        return run.model_copy(update={"id": cursor.lastrowid})

    def update_solver_run(
        self,
        run_id: int,
        *,
        expected_statuses: tuple[str, ...] | None = None,
        **changes: object,
    ) -> SolverRun:
        allowed = {
            "status",
            "backend_version",
            "evidence",
            "exploitability_pct",
            "runtime_seconds",
            "error_message",
            "pid",
            "command_path",
            "result_path",
            "log_path",
            "heartbeat_at",
            "started_at",
            "completed_at",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported solver-run fields: {', '.join(sorted(unknown))}")
        if not changes:
            saved = self.fetch_solver_run(run_id)
            if saved is None:
                raise ValueError("Solver run not found.")
            return saved
        assignments: list[str] = []
        values: list[object] = []
        for key, value in changes.items():
            assignments.append(f"{key} = ?")
            if key == "evidence":
                values.append(_serialize_json(value))
            elif key in {"heartbeat_at", "started_at", "completed_at"}:
                values.append(_serialize_optional_datetime(value))  # type: ignore[arg-type]
            elif key == "error_message":
                values.append(_scrubbed_job_text(value))  # type: ignore[arg-type]
            else:
                values.append(value)
        values.append(run_id)
        status_clause = ""
        if expected_statuses:
            placeholders = ", ".join("?" for _ in expected_statuses)
            status_clause = f" AND status IN ({placeholders})"
            values.extend(expected_statuses)
        cursor = self._execute(
            f"UPDATE solver_runs SET {', '.join(assignments)} "
            f"WHERE id = ?{status_clause}",
            tuple(values),
        )
        self._commit()
        if cursor.rowcount != 1:
            if expected_statuses:
                saved = self.fetch_solver_run(run_id)
                if saved is not None:
                    return saved
            raise ValueError("Solver run not found.")
        saved = self.fetch_solver_run(run_id)
        if saved is None:
            raise RuntimeError("Updated solver run could not be reloaded.")
        return saved

    def fetch_solver_run(self, run_id: int) -> SolverRun | None:
        row = self._execute("SELECT * FROM solver_runs WHERE id = ?", (run_id,)).fetchone()
        return None if row is None else _solver_run_from_row(row)

    def fetch_solver_runs_by_hand(self, hand_id: int) -> list[SolverRun]:
        rows = self._execute(
            """
            SELECT * FROM solver_runs
            WHERE hand_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (hand_id,),
        ).fetchall()
        return [_solver_run_from_row(row) for row in rows]

    def delete_solver_run(self, run_id: int) -> None:
        """Discard one retained solver run.

        STALE_SOLVER_EVIDENCE names this as a clearing action, and re-running the
        solve is not always available: a hand that stopped being solver-eligible
        after a correction had no way at all to clear a stale run. Refuses a live
        run so a background worker's row cannot vanish underneath it.
        """
        row = self._execute(
            "SELECT status FROM solver_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Solver run not found.")
        # The same set fetch_active_solver_runs calls active. 'cancelling' was
        # missing, so a run the store itself reported as live could be deleted out
        # from under the worker that was still winding it down.
        if row["status"] in _LIVE_SOLVER_STATUSES:
            raise ValueError("Cancel the solver run before deleting it.")
        self._execute("DELETE FROM solver_runs WHERE id = ?", (run_id,))
        self._commit()

    def fetch_cached_solver_run(self, input_hash: str) -> SolverRun | None:
        """The newest run this build reads as ``completed`` for these inputs.

        ``_solver_run_from_row`` degrades a run to ``stale`` when any of its
        columns is unreadable, precisely so a result nobody can inspect is not
        presented as study evidence. A ``WHERE status = 'completed'`` predicate
        answered in the column's space instead and handed such a run back as a
        cache hit. See ``_MODEL_SPACE_CLASSIFICATION``.
        """
        rows = self._execute(
            """
            SELECT * FROM solver_runs
            WHERE input_hash = ?
            ORDER BY completed_at DESC, id DESC
            """,
            (input_hash,),
        ).fetchall()
        for row in rows:
            run = _solver_run_from_row(row)
            if run.status == "completed":
                return run
        return None

    def fetch_active_solver_runs(self) -> list[SolverRun]:
        placeholders = ", ".join("?" for _ in _LIVE_SOLVER_STATUSES)
        rows = self._execute(
            f"""
            SELECT * FROM solver_runs
            WHERE status IN ({placeholders})
            ORDER BY created_at, id
            """,
            _LIVE_SOLVER_STATUSES,
        ).fetchall()
        return [_solver_run_from_row(row) for row in rows]

    def create_video(self, video: VideoRecord) -> VideoRecord:
        payload = video.model_dump()
        cursor = self._execute(
            """
            INSERT INTO videos (
                session_id, original_filename, stored_path, file_size_bytes,
                content_sha256, duration_seconds, fps, width, height, frame_count,
                uploaded_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["session_id"],
                payload["original_filename"],
                payload["stored_path"],
                payload["file_size_bytes"],
                payload.get("content_sha256") or "",
                payload["duration_seconds"],
                payload["fps"],
                payload["width"],
                payload["height"],
                payload["frame_count"],
                _serialize_datetime(payload["uploaded_at"]),
                payload["notes"],
            ),
        )
        self._commit()
        return video.model_copy(update={"id": cursor.lastrowid})

    def update_video_metadata(
        self,
        video_id: int,
        *,
        duration_seconds: float | None = None,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        frame_count: int | None = None,
    ) -> None:
        self._execute(
            """
            UPDATE videos
            SET duration_seconds = ?, fps = ?, width = ?, height = ?, frame_count = ?
            WHERE id = ?
            """,
            (duration_seconds, fps, width, height, frame_count, video_id),
        )
        self._commit()

    def update_video_session(self, video_id: int, session_id: int | None) -> VideoRecord:
        """Attach a stored video to a session, move it, or leave it unassigned."""

        if self.fetch_video(video_id) is None:
            raise ValueError("Video not found.")
        if session_id is not None and self.fetch_session(session_id) is None:
            raise ValueError("Target session not found.")
        self._execute(
            "UPDATE videos SET session_id = ? WHERE id = ?",
            (session_id, video_id),
        )
        self._commit()
        updated = self.fetch_video(video_id)
        if updated is None:
            raise RuntimeError("Updated video could not be reloaded.")
        return updated

    def fetch_video(self, video_id: int) -> VideoRecord | None:
        row = self._execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return None if row is None else _video_from_row(row)

    def fetch_videos(self, session_id: int | None = None) -> list[VideoRecord]:
        if session_id is None:
            rows = self._execute(
                "SELECT * FROM videos ORDER BY uploaded_at DESC, id DESC"
            ).fetchall()
        else:
            rows = self._execute(
                "SELECT * FROM videos WHERE session_id = ? ORDER BY uploaded_at DESC, id DESC",
                (session_id,),
            ).fetchall()
        return [_video_from_row(row) for row in rows]

    # Every column that stores a path to a file on disk. Retention consults this
    # before deleting anything, so a column added here without being listed
    # leaves its files looking unreferenced. Keep it exhaustive.
    ARTIFACT_PATH_COLUMNS: tuple[tuple[str, str], ...] = (
        ("videos.stored_path", "SELECT stored_path FROM videos"),
        ("extracted_frames.image_path", "SELECT image_path FROM extracted_frames"),
        (
            "reconstruction_frame_reviews.source_image",
            "SELECT source_image FROM reconstruction_frame_reviews",
        ),
        ("actions.source_image", "SELECT source_image FROM actions"),
        ("solver_runs.result_path", "SELECT result_path FROM solver_runs"),
        ("solver_runs.log_path", "SELECT log_path FROM solver_runs"),
        ("solver_runs.command_path", "SELECT command_path FROM solver_runs"),
        # A regression fixture is frequently a frame or a recording under a
        # managed directory, so omitting these let retention delete the very
        # evidence that proves a closed issue stays closed.
        ("regression_cases.fixture_path", "SELECT fixture_path FROM regression_cases"),
        ("regression_cases.report_path", "SELECT report_path FROM regression_cases"),
    )

    def referenced_artifact_paths(self) -> tuple[set[str], list[str]]:
        """Paths the database still points at, plus sources it could not read.

        The unreadable list matters as much as the paths: "this table holds no
        references" and "this table could not be queried" must never produce the
        same answer, because a caller acting on the first would delete files the
        second was about to protect.
        """
        found: set[str] = set()
        unreadable: list[str] = []
        for label, sql in self.ARTIFACT_PATH_COLUMNS:
            try:
                rows = self._execute(sql).fetchall()
            except sqlite3.Error:
                unreadable.append(label)
                continue
            for row in rows:
                raw = row[0]
                if isinstance(raw, str) and raw.strip():
                    found.add(raw.strip())
        return found, unreadable

    def create_processing_job(self, job: ProcessingJob) -> ProcessingJob:
        payload = job.model_dump()
        cursor = self._execute(
            """
            INSERT INTO processing_jobs (
                job_type, status, video_id, progress_percent, message, error_message,
                pid, heartbeat_at, created_at, started_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["job_type"],
                payload["status"],
                payload["video_id"],
                payload["progress_percent"],
                _scrubbed_job_text(payload["message"]),
                _scrubbed_job_text(payload["error_message"]),
                payload["pid"],
                _serialize_optional_datetime(payload["heartbeat_at"]),
                _serialize_datetime(payload["created_at"]),
                _serialize_optional_datetime(payload["started_at"]),
                _serialize_optional_datetime(payload["completed_at"]),
            ),
        )
        self._commit()
        return job.model_copy(update={"id": cursor.lastrowid})

    def update_processing_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        progress_percent: float | None = None,
        message: str | None = None,
        error_message: str | None = None,
        pid: object = _PROCESSING_JOB_PID_UNSET,
        clear_pid: bool = False,
        heartbeat_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        expected_statuses: tuple[str, ...] | None = None,
    ) -> ProcessingJob:
        current = self.fetch_processing_job(job_id)
        if current is None:
            raise ValueError(f"Processing job not found: {job_id}")
        if clear_pid:
            next_pid: int | None = None
        elif pid is _PROCESSING_JOB_PID_UNSET:
            next_pid = current.pid
        else:
            next_pid = pid  # type: ignore[assignment]
        status_clause = ""
        params: list[object] = [
            status or current.status,
            current.progress_percent if progress_percent is None else progress_percent,
            _scrubbed_job_text(current.message if message is None else message),
            _scrubbed_job_text(
                current.error_message if error_message is None else error_message
            ),
            next_pid,
            _serialize_optional_datetime(
                heartbeat_at if heartbeat_at is not None else current.heartbeat_at
            ),
            _serialize_optional_datetime(
                started_at if started_at is not None else current.started_at
            ),
            _serialize_optional_datetime(
                completed_at if completed_at is not None else current.completed_at
            ),
            job_id,
        ]
        if expected_statuses:
            placeholders = ", ".join("?" for _ in expected_statuses)
            status_clause = f" AND status IN ({placeholders})"
            params.extend(expected_statuses)
        cursor = self._execute(
            f"""
            UPDATE processing_jobs
            SET status = ?, progress_percent = ?, message = ?, error_message = ?,
                pid = ?, heartbeat_at = ?, started_at = ?, completed_at = ?
            WHERE id = ?{status_clause}
            """,
            tuple(params),
        )
        self._commit()
        if cursor.rowcount != 1 and expected_statuses:
            saved = self.fetch_processing_job(job_id)
            if saved is not None:
                return saved
            raise ValueError(f"Processing job not found: {job_id}")
        saved = self.fetch_processing_job(job_id)
        if saved is None:
            raise RuntimeError("Updated processing job could not be reloaded.")
        return saved

    def fetch_processing_job(self, job_id: int) -> ProcessingJob | None:
        row = self._execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,)).fetchone()
        return None if row is None else _processing_job_from_row(row)

    def fetch_jobs_by_video(self, video_id: int) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs WHERE video_id = ? ORDER BY created_at DESC, id DESC",
            (video_id,),
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def fetch_recent_jobs(self, limit: int = 20) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def fetch_all_jobs(self) -> list[ProcessingJob]:
        """Every processing job, newest first, for a surface that states job counts.

        Unbounded on purpose: ``fetch_recent_jobs`` answers "what happened
        lately", and a count derived from a truncated window is not a count. The
        table holds one row per reconstruction or extraction run on a local-first
        store, so the full read is small; a list view should still take the
        bounded call.
        """
        rows = self._execute(
            "SELECT * FROM processing_jobs ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def fetch_running_jobs(self) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs WHERE status IN "
            "('queued', 'running', 'cancelling') ORDER BY created_at, id"
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def fetch_active_jobs(self) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs WHERE status IN "
            "('queued', 'running', 'cancelling') "
            "ORDER BY created_at, id"
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def create_extracted_frame(self, frame: ExtractedFrame) -> ExtractedFrame:
        payload = frame.model_dump()
        cursor = self._execute(
            """
            INSERT OR IGNORE INTO extracted_frames (
                video_id, job_id, timestamp_seconds, frame_index, image_path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload["video_id"],
                payload["job_id"],
                payload["timestamp_seconds"],
                payload["frame_index"],
                payload["image_path"],
                _serialize_datetime(payload["created_at"]),
            ),
        )
        self._commit()
        frame_id = (
            cursor.lastrowid
            or self._execute(
                """
            SELECT id FROM extracted_frames
            WHERE video_id = ? AND frame_index = ? AND image_path = ?
            """,
                (payload["video_id"], payload["frame_index"], payload["image_path"]),
            ).fetchone()["id"]
        )
        return frame.model_copy(update={"id": frame_id})

    def fetch_frames_by_video(self, video_id: int) -> list[ExtractedFrame]:
        rows = self._execute(
            """
            SELECT * FROM extracted_frames
            WHERE video_id = ?
            ORDER BY timestamp_seconds, frame_index, id
            """,
            (video_id,),
        ).fetchall()
        return [_extracted_frame_from_row(row) for row in rows]

    def fetch_extracted_frame(self, frame_id: int) -> ExtractedFrame | None:
        row = self._execute("SELECT * FROM extracted_frames WHERE id = ?", (frame_id,)).fetchone()
        return None if row is None else _extracted_frame_from_row(row)

    def delete_frame_records_by_video(self, video_id: int) -> None:
        self._execute("DELETE FROM extracted_frames WHERE video_id = ?", (video_id,))
        self._commit()

    def upsert_reconstruction_frame_review(
        self, review: ReconstructionFrameReview
    ) -> ReconstructionFrameReview:
        payload = review.model_dump()
        self._execute(
            """
            INSERT INTO reconstruction_frame_reviews (
                job_id, hand_number, source_image, timestamp_seconds, status,
                issue_types, notes, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, hand_number, source_image) DO UPDATE SET
                timestamp_seconds = excluded.timestamp_seconds,
                status = excluded.status,
                issue_types = excluded.issue_types,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (
                payload["job_id"],
                payload["hand_number"],
                payload["source_image"],
                payload["timestamp_seconds"],
                payload["status"],
                _serialize_json(payload["issue_types"]),
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
            ),
        )
        self._commit()
        saved = self._execute(
            """
            SELECT * FROM reconstruction_frame_reviews
            WHERE job_id = ? AND hand_number = ? AND source_image = ?
            """,
            (review.job_id, review.hand_number, review.source_image),
        ).fetchone()
        if saved is None:
            raise RuntimeError("The frame review could not be reloaded.")
        return _reconstruction_frame_review_from_row(saved)

    def fetch_reconstruction_frame_reviews(
        self, job_id: int, hand_number: int | None = None
    ) -> list[ReconstructionFrameReview]:
        if hand_number is None:
            rows = self._execute(
                """
                SELECT * FROM reconstruction_frame_reviews
                WHERE job_id = ?
                ORDER BY hand_number, timestamp_seconds, id
                """,
                (job_id,),
            ).fetchall()
        else:
            rows = self._execute(
                """
                SELECT * FROM reconstruction_frame_reviews
                WHERE job_id = ? AND hand_number = ?
                ORDER BY timestamp_seconds, id
                """,
                (job_id, hand_number),
            ).fetchall()
        return [_reconstruction_frame_review_from_row(row) for row in rows]

    def create_roi_profile(self, profile: ROIProfile) -> ROIProfile:
        payload = profile.model_dump()
        cursor = self._execute(
            """
            INSERT INTO roi_profiles (
                name, description, platform, table_layout, video_width, video_height,
                created_at, updated_at, is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["name"],
                payload["description"],
                payload["platform"],
                payload["table_layout"],
                payload["video_width"],
                payload["video_height"],
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
                int(payload["is_active"]),
            ),
        )
        self._commit()
        saved = profile.model_copy(update={"id": cursor.lastrowid})
        if saved.is_active and saved.id is not None:
            self.mark_roi_profile_active(saved.id)
            saved = saved.model_copy(update={"is_active": True})
        return saved

    def update_roi_profile(self, profile: ROIProfile) -> ROIProfile:
        if profile.id is None:
            raise ValueError("Cannot update an ROI profile without an id.")
        payload = profile.model_dump()
        self._execute(
            """
            UPDATE roi_profiles
            SET name = ?, description = ?, platform = ?, table_layout = ?,
                video_width = ?, video_height = ?, updated_at = ?, is_active = ?
            WHERE id = ?
            """,
            (
                payload["name"],
                payload["description"],
                payload["platform"],
                payload["table_layout"],
                payload["video_width"],
                payload["video_height"],
                _serialize_datetime(payload["updated_at"]),
                int(payload["is_active"]),
                payload["id"],
            ),
        )
        self._commit()
        if profile.is_active:
            self.mark_roi_profile_active(profile.id)
        return profile

    def fetch_roi_profile(self, profile_id: int) -> ROIProfile | None:
        row = self._execute("SELECT * FROM roi_profiles WHERE id = ?", (profile_id,)).fetchone()
        return None if row is None else _roi_profile_from_row(row)

    def fetch_roi_profiles(self) -> list[ROIProfile]:
        rows = self._execute(
            "SELECT * FROM roi_profiles ORDER BY is_active DESC, updated_at DESC, id DESC"
        ).fetchall()
        return [_roi_profile_from_row(row) for row in rows]

    def mark_roi_profile_active(self, profile_id: int) -> None:
        if self.fetch_roi_profile(profile_id) is None:
            raise ValueError(f"ROI profile not found: {profile_id}")
        self._execute("UPDATE roi_profiles SET is_active = 0")
        self._execute(
            "UPDATE roi_profiles SET is_active = 1, updated_at = ? WHERE id = ?",
            (_serialize_datetime(datetime.now().astimezone()), profile_id),
        )
        self._commit()

    def delete_roi_profile(self, profile_id: int) -> None:
        self._execute("DELETE FROM roi_profiles WHERE id = ?", (profile_id,))
        self._commit()

    def create_roi_region(self, region: ROIRegion) -> ROIRegion:
        self._validate_roi_region_for_profile(region)
        payload = region.model_dump()
        cursor = self._execute(
            """
            INSERT INTO roi_regions (
                profile_id, roi_key, roi_type, label, x, y, width, height,
                seat_index, card_index, notes, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["profile_id"],
                payload["roi_key"],
                payload["roi_type"],
                payload["label"],
                payload["x"],
                payload["y"],
                payload["width"],
                payload["height"],
                payload["seat_index"],
                payload["card_index"],
                payload["notes"],
                _serialize_datetime(payload["created_at"]),
                _serialize_datetime(payload["updated_at"]),
            ),
        )
        self._commit()
        return region.model_copy(update={"id": cursor.lastrowid})

    def update_roi_region(self, region: ROIRegion) -> ROIRegion:
        if region.id is None:
            raise ValueError("Cannot update an ROI region without an id.")
        self._validate_roi_region_for_profile(region)
        payload = region.model_dump()
        self._execute(
            """
            UPDATE roi_regions
            SET roi_key = ?, roi_type = ?, label = ?, x = ?, y = ?, width = ?,
                height = ?, seat_index = ?, card_index = ?, notes = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                payload["roi_key"],
                payload["roi_type"],
                payload["label"],
                payload["x"],
                payload["y"],
                payload["width"],
                payload["height"],
                payload["seat_index"],
                payload["card_index"],
                payload["notes"],
                _serialize_datetime(payload["updated_at"]),
                payload["id"],
            ),
        )
        self._commit()
        return region

    def delete_roi_region(self, region_id: int) -> None:
        self._execute("DELETE FROM roi_regions WHERE id = ?", (region_id,))
        self._commit()

    def fetch_roi_regions_by_profile(self, profile_id: int) -> list[ROIRegion]:
        rows = self._execute(
            """
            SELECT * FROM roi_regions
            WHERE profile_id = ?
            ORDER BY roi_type, seat_index, card_index, roi_key, id
            """,
            (profile_id,),
        ).fetchall()
        return [_roi_region_from_row(row) for row in rows]

    def _validate_roi_region_for_profile(self, region: ROIRegion) -> None:
        profile = self.fetch_roi_profile(region.profile_id)
        if profile is None:
            raise ValueError(f"ROI profile not found: {region.profile_id}")
        validate_roi_bounds(
            region,
            image_width=profile.video_width,
            image_height=profile.video_height,
        )

    def schema_version(self) -> int:
        """The stored schema version, or 0 for a fresh database.

        An unreadable stamp raises the same clear operator message the
        newer-database path raises, instead of a bare ``int()`` ValueError.
        """
        with self._lock:
            version = _readable_schema_version(self._connection)
        if version is None:
            _assert_supported_schema_version(version)
        return version or 0

    def delete_session(self, session_id: int) -> None:
        """Delete a session; hands, actions, reviews cascade. Videos are kept (unlinked)."""
        self._execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._commit()

    def delete_video(self, video_id: int) -> None:
        """Delete a video row; jobs and extracted frames cascade. Files are the caller's job."""
        self._execute("DELETE FROM videos WHERE id = ?", (video_id,))
        self._commit()


def _migrate_to_v6(db: PokerDatabase) -> None:
    db._ensure_column("processing_jobs", "pid", "INTEGER")
    db._ensure_column("processing_jobs", "heartbeat_at", "TEXT")


def _migrate_to_v7(db: PokerDatabase) -> None:
    db._ensure_column("hand_players", "player_key", "TEXT")
    db._ensure_column("hand_players", "seat_index", "INTEGER")
    db._ensure_column("actions", "player_key", "TEXT")
    db._ensure_column("actions", "amount_semantics", "TEXT NOT NULL DEFAULT 'unknown'")
    db._ensure_column("actions", "forced_bet_type", "TEXT")
    db._ensure_column("actions", "is_live_post", "INTEGER")
    db._execute(
        """
        UPDATE hand_players
        SET player_key = 'hand:' || hand_id || ':player:' || id
        WHERE player_key IS NULL OR TRIM(player_key) = ''
        """
    )
    db._execute(
        """
        UPDATE actions
        SET player_key = (
            SELECT hp.player_key
            FROM hand_players AS hp
            WHERE hp.hand_id = actions.hand_id
              AND hp.player_name = actions.player_name
              AND hp.position = actions.position
              AND (
                  SELECT COUNT(*)
                  FROM hand_players AS exact_hp
                  WHERE exact_hp.hand_id = actions.hand_id
                    AND exact_hp.player_name = actions.player_name
                    AND exact_hp.position = actions.position
              ) = 1
            ORDER BY hp.id
            LIMIT 1
        )
        WHERE player_key IS NULL
        """
    )
    db._execute(
        """
        UPDATE actions
        SET player_key = (
            SELECT hp.player_key
            FROM hand_players AS hp
            WHERE hp.hand_id = actions.hand_id
              AND hp.player_name = actions.player_name
              AND (
                  SELECT COUNT(*)
                  FROM hand_players AS name_hp
                  WHERE name_hp.hand_id = actions.hand_id
                    AND name_hp.player_name = actions.player_name
              ) = 1
            ORDER BY hp.id
            LIMIT 1
        )
        WHERE player_key IS NULL
        """
    )
    db._execute_script(
        """
        CREATE TABLE IF NOT EXISTS hand_settlements (
            hand_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'unsettled',
            dead_money REAL NOT NULL DEFAULT 0,
            rake_rate REAL NOT NULL DEFAULT 0,
            rake_cap REAL,
            rake_rounding_unit REAL NOT NULL DEFAULT 0.01,
            no_flop_no_drop INTEGER NOT NULL DEFAULT 0,
            gross_pot REAL,
            rake_amount REAL,
            net_pot REAL,
            is_balanced INTEGER NOT NULL DEFAULT 0,
            warnings TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS settlement_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            entry_type TEXT NOT NULL,
            pot_index INTEGER,
            player_key TEXT,
            player_name TEXT NOT NULL,
            amount REAL,
            entry_order INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_hand_players_hand_key
            ON hand_players(hand_id, player_key);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_hand_players_hand_seat
            ON hand_players(hand_id, seat_index)
            WHERE seat_index IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_actions_player_key
            ON actions(hand_id, player_key);
        CREATE INDEX IF NOT EXISTS idx_settlement_entries_hand
            ON settlement_entries(hand_id, entry_type, pot_index, entry_order);
        """
    )


def _migrate_to_v8(db: PokerDatabase) -> None:
    """Make persisted action order deterministic before enforcing uniqueness."""
    db._execute(
        """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY hand_id, street
                    ORDER BY action_index, id
                ) AS normalized_index
            FROM actions
        )
        UPDATE actions
        SET action_index = (
            SELECT normalized_index
            FROM ranked
            WHERE ranked.id = actions.id
        )
        """
    )
    db._execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_hand_street_order
        ON actions(hand_id, street, action_index)
        """
    )


def _migrate_to_v9(db: PokerDatabase) -> None:
    db._execute_script(
        """
        CREATE TABLE IF NOT EXISTS reconstruction_frame_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            hand_number INTEGER NOT NULL,
            source_image TEXT NOT NULL,
            timestamp_seconds REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'unreviewed',
            issue_types TEXT NOT NULL DEFAULT '[]',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES processing_jobs(id) ON DELETE CASCADE,
            UNIQUE(job_id, hand_number, source_image)
        );
        CREATE INDEX IF NOT EXISTS idx_reconstruction_reviews_job_hand
            ON reconstruction_frame_reviews(job_id, hand_number);
        """
    )


def _migrate_to_v10(db: PokerDatabase) -> None:
    """Retain correction evidence and make superseded coaching explicit."""

    db._ensure_column("hand_reviews", "is_stale", "INTEGER NOT NULL DEFAULT 0")
    db._ensure_column("hand_reviews", "stale_reason", "TEXT NOT NULL DEFAULT ''")
    db._ensure_column("coaching_reviews", "is_stale", "INTEGER NOT NULL DEFAULT 0")
    db._ensure_column("coaching_reviews", "stale_reason", "TEXT NOT NULL DEFAULT ''")
    db._execute_script(
        """
        CREATE TABLE IF NOT EXISTS hand_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            correction_type TEXT NOT NULL,
            before_state TEXT NOT NULL DEFAULT '{}',
            after_state TEXT NOT NULL DEFAULT '{}',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_hand_corrections_hand_id
            ON hand_corrections(hand_id, created_at, id);
        """
    )


def _migrate_to_v11(db: PokerDatabase) -> None:
    """Add auditable external-solver runs and reusable user range profiles."""

    db._execute_script(
        """
        CREATE TABLE IF NOT EXISTS solver_range_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            notation TEXT NOT NULL,
            table_size INTEGER,
            position TEXT NOT NULL DEFAULT '',
            scenario TEXT NOT NULL DEFAULT '',
            pot_type TEXT NOT NULL DEFAULT '',
            stack_bb REAL,
            description TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS solver_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            backend_name TEXT NOT NULL DEFAULT 'texassolver',
            backend_version TEXT NOT NULL DEFAULT '',
            input_hash TEXT NOT NULL,
            spot TEXT NOT NULL DEFAULT '{}',
            range_ip TEXT NOT NULL DEFAULT '{}',
            range_oop TEXT NOT NULL DEFAULT '{}',
            assumptions TEXT NOT NULL DEFAULT '[]',
            evidence TEXT NOT NULL DEFAULT '{}',
            command_path TEXT NOT NULL DEFAULT '',
            result_path TEXT NOT NULL DEFAULT '',
            log_path TEXT NOT NULL DEFAULT '',
            exploitability_pct REAL,
            runtime_seconds REAL,
            error_message TEXT NOT NULL DEFAULT '',
            pid INTEGER,
            heartbeat_at TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_solver_runs_hand
            ON solver_runs(hand_id, created_at, id);
        CREATE INDEX IF NOT EXISTS idx_solver_runs_cache
            ON solver_runs(input_hash, status);
        CREATE INDEX IF NOT EXISTS idx_solver_runs_status
            ON solver_runs(status, created_at);
        """
    )


def _migrate_to_v12(db: PokerDatabase) -> None:
    """Add a persistent queue for hands that need future debugging."""

    db._execute_script(
        """
        CREATE TABLE IF NOT EXISTS hand_issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hand_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            issue_types TEXT NOT NULL DEFAULT '[]',
            description TEXT NOT NULL,
            evidence_snapshot TEXT NOT NULL DEFAULT '{}',
            resolution_notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT,
            FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_hand_issues_status
            ON hand_issues(status, created_at, id);
        CREATE INDEX IF NOT EXISTS idx_hand_issues_hand
            ON hand_issues(hand_id, status, created_at, id);
        """
    )


def _migrate_to_v13(db: PokerDatabase) -> None:
    """Record explicit hand completion and demote every unproven reconstructed hand.

    MIGRATION IMPACT (schema 12 -> 13)

    Added:
      - hands.completion_status TEXT NOT NULL DEFAULT 'not_applicable'
      - hands.completion_evidence TEXT NOT NULL DEFAULT '{}'

    Rewritten (hands only, two columns):
      - completion_status is set for every existing row: 'not_applicable' where
        source_type = 'manual', 'uncertain' otherwise.
      - review_status is forced to 'needs_correction' for every row where
        source_type <> 'manual'. This is a visible product change: a CV-derived
        hand previously marked reviewed returns to needs_correction and must be
        re-confirmed. Manual hands keep their review_status untouched.

    A row is treated as a manual hand only when source_type says so exactly;
    every other value - including cv_import, corrected_cv, and any unrecognized
    value a hand-edited database may hold - is unproven and blocks.

    completion_evidence is left at '{}' for every migrated row. That parses to
    unknown evidence, which blocks study readiness; no evidence is fabricated for
    historical hands.

    Not touched: hand_corrections, hand_issues, hand_reviews, coaching_reviews,
    hand_settlements, settlement_entries, solver_runs, solver_range_profiles,
    videos, extracted_frames, processing_jobs, reconstruction_frame_reviews,
    roi_profiles, roi_regions. No row in any of these tables is deleted or
    rewritten, and no file on disk is read, moved, or deleted.

    Both UPDATEs are idempotent in the strict sense: they key on source_type, which
    this migration never writes, so applying them twice in a row gives the same
    result as applying them once. They are NOT a no-op on a database that has
    already been migrated AND then used: re-running would reset every non-manual
    hand to uncertain/needs_correction, discarding operator confirmations. No
    production path re-runs a completed migration (the chain and the version stamp
    share one transaction); a repair path that replays migrations must not run this
    one against a live database.
    """

    db._ensure_column("hands", "completion_status", "TEXT NOT NULL DEFAULT 'not_applicable'")
    db._ensure_column("hands", "completion_evidence", "TEXT NOT NULL DEFAULT '{}'")
    # ``IS`` / ``IS NOT``, never ``=`` / ``<>``: a legacy build that added
    # source_type without NOT NULL leaves NULL rows, for which both two-valued
    # comparisons evaluate to NULL. Neither UPDATE fired, so those rows kept
    # review_status = 'reviewed' and fell through to the column default
    # 'not_applicable' -- the manual exemption, on a row that is not provably
    # manual. Under three-valued-safe operators a NULL row is classified as
    # unproven, exactly as the docstring above promises.
    db._execute(
        """
        UPDATE hands
        SET completion_status = 'not_applicable'
        WHERE source_type IS 'manual'
        """
    )
    db._execute(
        """
        UPDATE hands
        SET completion_status = 'uncertain',
            review_status = 'needs_correction'
        WHERE source_type IS NOT 'manual'
        """
    )


def _migrate_to_v14(db: PokerDatabase) -> None:
    """Persist content hashes for stored completed-session videos.

    MIGRATION IMPACT (schema 13 -> 14)

    Added:
      - videos.content_sha256 TEXT NOT NULL DEFAULT ''

    Existing video rows keep an empty hash. Size checks still apply; hash
    enforcement activates only when a non-empty hash is present (new uploads
    after this version). No video files are read, rewritten, or deleted. No
    other tables are touched.
    """

    db._ensure_column("videos", "content_sha256", "TEXT NOT NULL DEFAULT ''")


def _migrate_to_v15(db: PokerDatabase) -> None:
    """Persist operator study-inclusion preference per hand.

    MIGRATION IMPACT (schema 14 -> 15)

    Added:
      - hands.study_inclusion TEXT NOT NULL DEFAULT 'auto'

    Existing hands become ``auto`` (follow derived study readiness). Values are
    ``auto`` | ``study`` | ``skip``. No hand facts, completion evidence, or
    review status change. Export/import treat a missing field as ``auto``.
    """

    db._ensure_column("hands", "study_inclusion", "TEXT NOT NULL DEFAULT 'auto'")


def _migrate_to_v16(db: PokerDatabase) -> None:
    """Remember which source frame produced each reconstructed action.

    MIGRATION IMPACT (schema 15 -> 16)

    Added:
      - actions.source_image TEXT NULL

    Purely additive and unbackfilled. Existing rows read NULL and fall back to
    the slot lookup used today, so nothing about an existing hand changes.

    Why it is needed: action indexes are per-street, so correcting a row's
    street or order moves it onto a different slot, and the validation UI —
    which had no other way to know where a row came from — either lost every
    frame-derived warning for it or, after a delete, attached another line's
    frame and stack figure. Storing the frame at import makes provenance
    survive any later edit.

    Written by the CV import path, and backfilled for pre-16 rows when Import
    validation opens the hand they belong to. Manual hands leave it NULL.
    """

    db._ensure_column("actions", "source_image", "TEXT")


def _migrate_to_v17(db: PokerDatabase) -> None:
    """Link a closed issue to the regression that proves it stays closed.

    MIGRATION IMPACT (schema 16 -> 17)

    Added:
      - regression_cases table (issue_id, correction_id, kind, fixture_path,
        status, failing_before, passing_after, fixing_commit, report_path,
        notes, created_at, updated_at)

    Purely additive. No existing table or column changes, and no existing row is
    read differently. A database migrated from 16 has zero regression cases, so
    every previously resolved issue stays resolved exactly as it was.

    Why it is needed: PLAN.md requires that every release-blocking closed issue
    have a passing permanent regression, and that an issue not be resolved until
    the regression failed before the fix and passed after it. Nothing recorded
    that relationship, so "resolved" meant only that somebody typed a note. The
    fail-before/pass-after evidence lives here alongside the fixing commit and
    the report that demonstrated it.
    """

    db._execute(
        """
        CREATE TABLE IF NOT EXISTS regression_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_id INTEGER NOT NULL,
            correction_id INTEGER,
            kind TEXT NOT NULL,
            fixture_path TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'proposed',
            failing_before INTEGER NOT NULL DEFAULT 0,
            passing_after INTEGER NOT NULL DEFAULT 0,
            fixing_commit TEXT NOT NULL DEFAULT '',
            report_path TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (issue_id) REFERENCES hand_issues(id) ON DELETE CASCADE,
            FOREIGN KEY (correction_id) REFERENCES hand_corrections(id) ON DELETE SET NULL
        )
        """
    )
    db._execute(
        """
        CREATE INDEX IF NOT EXISTS idx_regression_cases_issue
            ON regression_cases(issue_id, status, id)
        """
    )
    db._commit()


def _migrate_to_v18(db: PokerDatabase) -> None:
    """Keep what a solver run solved on the run, not only in its run directory.

    MIGRATION IMPACT (schema 17 -> 18)

    Added:
      - solver_runs.run_parameters TEXT NOT NULL DEFAULT '{}'

    Purely additive and deliberately unbackfilled. Existing solver runs read
    ``{}``, which ``SolverRunParameters`` reports as not retained, and the UI
    then says the abstraction behind those frequencies is unknown. Backfilling
    them with today's tree would be the failure this column exists to stop: it
    would assert that an old result was produced under settings nobody recorded.
    No existing row is deleted, rewritten, or read differently, and no file on
    disk is touched. The column is write-once at INSERT and is absent from
    ``update_solver_run``'s allowed fields, so the settings a retained result
    was produced under cannot be edited after the fact.

    Why it is needed: the betting abstraction, accuracy target and iteration
    cap existed only as text inside the run directory's ``input.txt``. The
    product deletes that directory when a hand or session is deleted, operators
    prune it, and a container without a persistent mount never has it -- while
    the row goes on presenting its frequencies as evidence. A frequency vector
    with no record of the tree it came from is a claim that cannot be checked,
    which is exactly what must not survive silently.
    """

    db._ensure_column("solver_runs", "run_parameters", "TEXT NOT NULL DEFAULT '{}'")


def _migrate_to_v19(db: PokerDatabase) -> None:
    """Record the room's blind structure, because the action line cannot show it.

    MIGRATION IMPACT (schema 18 -> 19)

    Added:
      - hand_settlements.small_blind REAL NULL
      - hand_settlements.big_blind REAL NULL
      - hand_settlements.straddles TEXT NOT NULL DEFAULT '[]'

    Forward-only and purely additive. No existing column changes type, no
    existing row is rewritten, deleted, or re-keyed, and no file on disk is
    touched. A hand with no ``hand_settlements`` row at all is not given one.

    DELIBERATELY UNBACKFILLED, and this is the whole point of the column.
    Backfilling a big blind -- from ``hands.blinds_antes``, from the largest
    observed blind post, from the session stakes -- would assert a fact about a
    room nobody recorded, and asserting exactly that fact from the largest
    observed post is the defect this migration exists to end. A small blind of 5
    is equally consistent with 5/10 and 5/5.

    WHAT HAPPENS TO EXISTING HANDS, precisely, so that "they silently become
    wrong" is not one of the answers:

      * A hand whose forced posts were all made in full -- effectively every
        ordinary hand -- reads ``big_blind IS NULL``, declares no structure, and
        derives byte-identically to schema 18. The reducer's preflop floor is
        combined with the observed street maximum by ``max``, and a floor of
        zero is the identity. Nothing about its pot, rake, payouts, hero result,
        settlement status, or study readiness moves.

      * A hand whose recording IDENTIFIES a live forced post that left its
        poster all-in -- by an action type of ``post_blind``, or by a
        ``forced_bet_type`` naming a live structural bet on a row booked under
        another type -- gains one legality issue naming that seat, so
        ``is_legal`` goes False, the
        reconciliation stops being authoritative, and study readiness blocks on
        ACCOUNTING_NOT_AUTHORITATIVE. That is a deliberate, visible demotion of
        hands that were previously reconciled around an amount-to-call the
        product had inferred from a short post -- the reported case reconciled a
        14-chip pot whose truth was 24, silently and with an empty blocker
        tuple. The clearing action is an ordinary settlement edit: declare the
        blind structure in Edit settlement and save. It is not a data repair and
        it destroys nothing; the previous figures are still derived and still
        displayed while the hand is blocked.

      * A hand whose recording does NOT identify the short post as a forced one
        -- a reconstruction that books a blind which took its poster's last chip
        as a plain ``all-in`` with no forced-bet type -- is NOT demoted, because
        nothing distinguishes it from an ordinary short shove. Its amount to
        call still comes from the observed maximum. Declaring the structure
        fixes such a hand, but the product does not ask for it.

      * ``hands.blinds_antes`` is untouched and is still free display text. It
        is not parsed, and nothing reads a chip size out of it.

    DOWNGRADE: an older build does not read a v19 database at all.
    ``_assert_supported_schema_version`` refuses any stamp above the build's own
    ``SCHEMA_VERSION`` with "Update the app before opening it", so the file is
    not opened, not read, and not written. That is safer than the partial read
    it replaces, but it means a downgrade is an app rollback, not a file
    operation: keep the pre-migration backup ``init_db`` took. ``_ensure_column``
    is idempotent, so re-running the chain forward is a no-op.
    """

    db._ensure_column("hand_settlements", "small_blind", "REAL")
    db._ensure_column("hand_settlements", "big_blind", "REAL")
    db._ensure_column("hand_settlements", "straddles", "TEXT NOT NULL DEFAULT '[]'")


def _migrate_to_v20(db: PokerDatabase) -> None:
    """Record HOW this hand's antes were taken, because it changes the pots.

    MIGRATION IMPACT (schema 19 -> 20)

    Added:
      - hand_settlements.ante_mode TEXT NULL
        (one of 'NONE', 'PER_PLAYER', 'SINGLE_PAYER_TABLE_ANTE'; NULL means
        not declared)

    Forward-only and purely additive. No existing column changes type, no
    existing row is rewritten, deleted, or re-keyed, and no file on disk is
    touched. A hand with no ``hand_settlements`` row at all is not given one.

    DELIBERATELY UNBACKFILLED, AND THIS IS THE ENTIRE POINT OF THE COLUMN.
    Backfilling ``NONE`` would be a lie on any hand that contains an ante.
    Backfilling ``PER_PLAYER`` -- today's arithmetic -- would assert that every
    stored hand took its antes individually, and a big-blind ante is the
    commonest ante structure in modern tournaments. Backfilling
    ``SINGLE_PAYER_TABLE_ANTE`` when exactly one seat anted, which is the
    tempting rule, is INFERENCE FROM THE SHAPE OF THE POSTS, and the operator
    ruled against exactly that: one seat anting is equally consistent with a
    big-blind ante and with a late-entry seat posting its own. The two give
    different pots on the same recording -- blinds 1/2, a 2-chip big-blind ante
    and an all-in 1-chip small blind is a 5-chip main pot one way and a 4-chip
    main pot the other -- so a guess here is a wrong payout published as
    authoritative, which is the failure class this module has already shipped
    five times.

    WHAT HAPPENS TO EXISTING HANDS, precisely.

      * A HAND WITH NO ANTE ROWS AND NO DECLARED ``dead_money`` -- the
        overwhelming majority, and every ordinary cash-game hand -- is
        COMPLETELY UNTOUCHED. It reads ``ante_mode IS NULL``, the reducer
        resolves that to ``NONE`` without a refusal because ``NONE`` is not a
        guess for a hand that has no antes, and every figure, verdict, status
        and readiness result is byte-identical to schema 19. There is no new
        blocker, no new warning, and no re-save needed.

      * A HAND WITH ``dead_money > 0`` IS NOT IN THAT SET, WHETHER OR NOT IT
        CONTAINS AN ANTE, AND THAT IS WHY THIS MIGRATION WRITES ROWS AT ALL.
        The same release carries ruling 5: operator-typed external dead money,
        which schema 19 dropped WHOLE into the lowest layer, is now capped
        against the collecting seat's own total commitment exactly like a
        recorded dead post. On any stored hand where the declared amount exceeds
        the smallest total commitment among the seats contesting the main pot,
        THE STORED HERO RESULT MOVES on the day the build is upgraded -- the
        gross pot, the pot count and every eligible set stay identical, so the
        distribution changes underneath a stored award row and every existing
        cross-check (recorded gross, recorded net, ``is_balanced``,
        ``_validate_winners``) still passes.

        Ruling 5 is the operator's and the new arithmetic is the right one; the
        re-derivation is what an upgrade is FOR and nothing here second-guesses
        it. What is not acceptable is the SECOND-ORDER effect: coaching and
        solver output retained beside those hands was written against a hero
        result this build no longer produces, and ``is_stale`` is a stored flag
        that only the explicit correction paths set, so a change in the
        DERIVATION RULE stales nothing. The wrong number then survives in the
        retained text, labelled current, on a hand whose accounting reconciles
        cleanly. So this migration calls what ``_stale_retained_analysis``
        calls, set-wise, over that population: hand reviews, hand-level and
        session-level coaching, and queued/running/completed solver runs stop
        being presented as current. Study readiness then blocks on
        STALE_COACHING_EVIDENCE, which is the visible rejection.

        WHAT IT DELIBERATELY DOES NOT DO IS TOUCH ``review_status``.
        ``_stale_retained_analysis`` exists as a separate method from
        ``_invalidate_hand_derivatives`` precisely because those two are
        different acts, and only the second discards an operator's own
        confirmation. A hand carrying no retained analysis has nothing that was
        derived under the old rule -- it simply reads correctly the next time it
        is opened -- so demoting it would destroy a confirmation to announce a
        change that left no artifact behind. That is also what
        ``_migrate_to_v13``'s own fixture warns about: it seeds a reviewed,
        complete hand specifically as a state a migration must not casually
        knock back.

        THE PREDICATE IS ``dead_money > 0``, NOT "the amount exceeds the floor",
        because the floor is the smallest total commitment among the seats
        contesting the main pot and SQL cannot compute it -- it needs the whole
        action line run through the reducer, which a schema migration does not
        have. So it is DELIBERATELY OVER-STRICT: it also stales analysis beside
        hands whose declared amount sat under the floor and whose figures did
        not move. Staling is the right place to spend that imprecision --
        ``is_stale`` already means "may have been derived from something that
        changed", and the settlement writers set it on every save without
        checking whether the figures moved either. A rerun the operator did not
        need is cheap; a stale hero result labelled current is the failure class
        this module has shipped five times.

        THE CLEARING ACTION is to rerun coaching, or to dismiss the staleness
        after checking the hand. Nothing is destroyed: the settlement row, the
        awards, the actions, the review status and the coaching text itself are
        all untouched, and only the freshness flags move.

        WHY THIS LIVES IN THE v20 STEP rather than a step of its own. Ruling 5
        changes no schema, so it has no version of its own to hang on; every
        database that reaches the new build's behaviour passes through exactly
        this migration, so this is the one place that runs once per upgraded
        file. A database already stamped 20 has already been through it.

      * A HAND CONTAINING ANY ANTE POST gains one legality issue naming the
        anteing seats and the missing declaration. ``is_legal`` goes False, the
        reconciliation stops being authoritative, ``persist_reconciliation``
        writes ``needs_correction``, and study readiness blocks on
        ACCOUNTING_NOT_AUTHORITATIVE. THESE ARE HANDS THAT PREVIOUSLY
        RECONCILED, and demoting them is the ruling, not a side effect: the
        product was laying them out under one of two readings without recording
        which, and the operator has ruled that an undeclared ante mode is
        ambiguous rather than defaulted.

        THE CLEARING ACTION is an ordinary settlement edit and nothing else:
        open Edit settlement, choose the ante mode this hand was dealt under,
        and save. It is not a data repair, it destroys nothing, and the previous
        pot figures are still derived and still displayed while the hand is
        blocked -- the layers shown alongside the refusal are the capped
        (PER_PLAYER) reading, which is both the strict direction and exactly
        what this product derived before the column existed, so nothing moves
        underneath the operator on the day it blocks.

        Declaring ``SINGLE_PAYER_TABLE_ANTE`` on such a hand CAN move its chips,
        by design: that is the reading in which the consolidated ante is table
        money. Because ``ante_mode`` is in ``_declared_settlement_inputs``,
        saving a mode that differs from what was stored stales any retained
        coaching or solver output and demotes a reviewed hand, exactly as
        editing the blind structure or the rake policy does.

      * Antes are identified the same way the rest of this module identifies
        forced posts: an action typed ``ante``, or any action carrying a
        ``forced_bet_type`` of ``ante`` or ``big_blind_ante``. A row spelled
        ``ante`` but typed ``dead_blind`` is a dead blind and does not trigger
        the refusal, because the mode is an ANTE mode.

      * ``hands.blinds_antes`` is untouched and is still free display text. It
        is not parsed, and no ante mode is read out of it -- for the same reason
        no chip size is.

    HOW MANY HANDS. Query both populations rather than guessing:
    ``SELECT COUNT(DISTINCT hand_id) FROM actions WHERE action_type = 'ante' OR
    forced_bet_type IN ('ante','big_blind_ante')`` asks for a declaration, and
    ``SELECT COUNT(*) FROM hand_settlements WHERE dead_money > 0`` is re-derived
    under ruling 5 and demoted here. A hand in neither set is untouched.

    DOWNGRADE: an older build does not read a v20 database at all.
    ``_assert_supported_schema_version`` refuses any stamp above the build's own
    ``SCHEMA_VERSION`` with "Update the app before opening it", so the file is
    not opened, not read, and not written. Keep the pre-migration backup
    ``init_db`` took. ``_ensure_column`` is idempotent, so re-running the chain
    forward is a no-op.

    RE-RUNNING THE STALING is idempotent in the strict sense -- it keys on
    ``dead_money``, which this migration never writes, and it only ever sets
    flags that are already set -- so replaying it re-stales analysis the
    operator may have rerun since, and nothing else. That is a weaker hazard
    than ``_migrate_to_v13``'s replay, which discards confirmations outright,
    and it is weaker on purpose: this step touches no ``review_status``, no
    ``completion_status``, and no analysis TEXT. No production path replays a
    completed migration (the chain and the version stamp share one
    transaction) in any case.
    """

    db._ensure_column("hand_settlements", "ante_mode", "TEXT")

    # RULING 5's stored population. See MIGRATION IMPACT above: these hands are
    # re-derived by the same release, so the analysis retained beside them was
    # written against a figure this build no longer produces.
    _reason = (
        "External dead money is now capped against each seat's own total "
        "commitment; this hand's pot layers were re-derived. Re-check the "
        "result and rerun coaching."
    )
    _affected = "SELECT hand_id FROM hand_settlements WHERE dead_money > 0"
    db._execute(
        f"""
        UPDATE hand_reviews
        SET is_stale = 1, stale_reason = ?
        WHERE hand_id IN ({_affected})
        """,
        (_reason,),
    )
    db._execute(
        f"""
        UPDATE coaching_reviews
        SET is_stale = 1, stale_reason = ?
        WHERE review_type = 'hand' AND hand_id IN ({_affected})
        """,
        (_reason,),
    )
    db._execute(
        f"""
        UPDATE coaching_reviews
        SET is_stale = 1,
            stale_reason = 'A hand in this session was re-derived under the '
                           || 'amended dead-money rule; rerun coaching.'
        WHERE review_type = 'session'
          AND session_id IN (
              SELECT session_id FROM hands WHERE id IN ({_affected})
          )
        """
    )
    db._execute(
        f"""
        UPDATE solver_runs
        SET status = CASE
                WHEN status IN ('queued', 'running') THEN 'cancelling'
                ELSE 'stale'
            END,
            error_message = ?
        WHERE status IN ('queued', 'running', 'completed')
          AND hand_id IN ({_affected})
        """,
        (_reason,),
    )


# Versioned migrations run in order and refuse databases written by newer apps.
_MIGRATIONS: dict[int, Callable[[PokerDatabase], None]] = {
    6: _migrate_to_v6,
    7: _migrate_to_v7,
    8: _migrate_to_v8,
    9: _migrate_to_v9,
    10: _migrate_to_v10,
    11: _migrate_to_v11,
    12: _migrate_to_v12,
    13: _migrate_to_v13,
    14: _migrate_to_v14,
    15: _migrate_to_v15,
    16: _migrate_to_v16,
    17: _migrate_to_v17,
    18: _migrate_to_v18,
    19: _migrate_to_v19,
    20: _migrate_to_v20,
}


def _serialize_date(value: date) -> str:
    return value.isoformat()


def _serialize_datetime(value: datetime) -> str:
    return value.isoformat()


def _serialize_optional_datetime(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _serialize_json(value: Any) -> str:
    return json.dumps(_json_representable(value), allow_nan=False)


def _json_representable(value: Any) -> Any:
    """Replace non-finite floats with None so every stored blob is strict JSON.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens by default. Those are
    not RFC 8259 JSON: Python's own reader accepts them, so a NaN written into
    ``hands.completion_evidence`` survived a store/fetch/export/import round trip
    while no standards-compliant consumer could read the export at all. ``None``
    is the honest replacement -- an unreadable measurement, which every parser in
    this package already degrades to "unknown" and which blocks study readiness.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_representable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_representable(item) for item in value]
    return value


def _scrubbed_job_text(value: str | None) -> str | None:
    """Scrub credentials out of a job's free-text column on the way in.

    Job status and failure text is written by whichever worker happened to fail,
    and an exception raised inside a client library carries the request it was
    making -- key included. Scrubbing where the column is written rather than in
    each worker is what makes the guarantee survive the next writer: there is no
    path to these columns that can skip it. Redaction is idempotent, so a caller
    that already scrubbed and bounded its own message loses nothing here.
    """
    if not value:
        return value
    return redact_text(value)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _parse_optional_datetime(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


# What a degraded reader shows where a required text column could not be read.
_UNREADABLE_LABEL = "(unreadable)"

_ModelT = TypeVar("_ModelT", bound=PersistedModel)


def _coerced_int(value: object, default: int) -> int:
    """An int for a fallback identity, or ``default`` when the column has none."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _recorded_column_value(raw: object) -> str | float | int | None:
    """Invert what a degradation marker recorded, or None when it cannot be inverted.

    ``_degrade_unreadable_cards`` records the raw string; ``_degraded_hand``
    records ``repr(value)`` so the blocker's detail can tell a stored ``'42.0'``
    from a stored ``42.0``. ``literal_eval`` is the exact inverse of ``repr`` for
    every primitive SQLite can hold, and the plain-string case falls through to
    itself, so one function reads both markers.

    A value that is neither -- a repr of something this build has no literal for,
    a nested structure -- is skipped. Skipping loses the round trip's fidelity for
    that column; guessing would write a value nobody recorded.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int | float):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        value = literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, TypeError):
        return raw
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        return value
    if value is None:
        return None
    return raw


def _salvaged_row(
    model_cls: type[_ModelT],
    data: dict[str, Any],
    fallbacks: dict[str, Any],
) -> tuple[_ModelT, tuple[str, ...]]:
    """Read a row this build cannot validate, column by column, without raising.

    The generic half of the round-10 settlement repair
    (``_degraded_hand_settlement``), applied through one helper rather than
    re-implemented per table: every ``_*_from_row`` reader that fails whole-row
    validation lands here, pydantic itself names the offending columns, each one
    is dropped to the model default (or to the caller's ``fallbacks`` for a
    required identity column), and everything that still validates is kept. The
    MODEL's field set and pydantic's own error report drive the loop, so a
    column added to any model later is covered without a hand-kept list — the
    hand-kept lists are exactly what decayed: ``_hand_from_row`` guarded four
    columns while ``table_size = 99`` in the fifth raised a ValidationError out
    of ``fetch_hands_by_session`` and took the entire application down on load.

    The salvage degrades to the most conservative reading. The dropped columns
    are named on the returned model's ``unreadable_columns`` (a read-time,
    dump-excluded marker), which the accounting cross-check treats as an issue
    and the hand reader surfaces as a study-readiness blocker, so a degraded
    row can only ever ADD blockers, never clear one. Nothing on disk is
    touched: correcting and re-saving the record is what rewrites the row.
    """
    candidate = {
        key: value for key, value in data.items() if key in model_cls.model_fields
    }
    unreadable: list[str] = []
    instance: _ModelT | None = None
    for _ in range(len(model_cls.model_fields) + 2):
        try:
            instance = model_cls(**{**fallbacks, **candidate})
        except (ValidationError, TypeError, ValueError) as exc:
            named: set[str] = set()
            if isinstance(exc, ValidationError):
                named = {
                    str(error["loc"][0])
                    for error in exc.errors()
                    if error["loc"] and str(error["loc"][0]) in candidate
                }
            if not named:
                # A failure pydantic cannot attribute to one remaining column
                # (a cross-field validator, or damage beyond attribution).
                # Nothing further can be salvaged: keep the identity fallbacks
                # alone and mark everything else.
                unreadable.extend(sorted(set(candidate) - set(fallbacks)))
                candidate = {}
                continue
            unreadable.extend(sorted(named))
            for name in named:
                del candidate[name]
        else:
            break
    if instance is None:
        # Even the fallback skeleton kept failing, which is a programming error
        # in the fallbacks, not row damage; constructing it bare surfaces that
        # error instead of hiding it.
        instance = model_cls(**fallbacks)
    marked = tuple(dict.fromkeys(unreadable))
    return instance.model_copy(update={"unreadable_columns": marked}), marked


def _session_from_row(row: sqlite3.Row) -> Session:
    data = _row_dict(row)
    try:
        data["date_played"] = _parse_date(data["date_played"])
        data["created_at"] = _parse_datetime(data["created_at"])
        return Session(**data)
    except (ValidationError, TypeError, ValueError):
        session, _ = _salvaged_row(
            Session, _row_dict(row), {"name": _UNREADABLE_LABEL}
        )
        return session


def _hand_from_row(row: sqlite3.Row) -> Hand:
    data = _row_dict(row)
    unreadable_blobs = _unreadable_blob_columns(Hand, data)
    data["tags"] = _parse_json_list(data.get("tags", "[]"))
    # _parse_json_object, never _parse_json_dict: the latter str()s every value and
    # would flatten nested evidence into a Python repr.
    evidence = _parse_json_object(data.get("completion_evidence", "{}"), {})
    data["completion_evidence"] = evidence if isinstance(evidence, dict) else {}
    if unreadable_blobs:
        # The same read-time channel a degraded SCALAR column is recorded in, so
        # the demotion in ``_demote_degraded_hand``, the UNREADABLE_HAND_COLUMNS
        # blocker and the export/import restore are all inherited unchanged --
        # there is no second rule to keep in step. Written here rather than left
        # to ``_degraded_hand``, because a damaged blob does not fail validation
        # and so never reaches the salvage path at all.
        data["completion_evidence"] = {
            **data["completion_evidence"],
            UNREADABLE_HAND_COLUMNS_KEY: dict(unreadable_blobs),
        }
    # Symmetry with the evidence blob above. The hands table deliberately carries
    # no CHECK constraint, so a hand-edited row can hold 'COMPLETE', a trailing
    # space, or an integer. Passing those straight to the pydantic Literal made
    # one bad row raise a ValidationError out of the whole session's hand list.
    # An unreadable value degrades to the most conservative classification, which
    # can only ever add study-readiness blockers.
    if data.get("source_type") not in get_args(SourceType):
        # Unprovable provenance is not provably manual, so it is reconstructed.
        data["source_type"] = "cv_import"
    if (
        data["source_type"] == "manual"
        and parse_completion_evidence(data["completion_evidence"]).claims_reconstruction
    ):
        # A row claiming to be somebody's own entry while carrying evidence a
        # pipeline stamped. import_session refuses the same row verbatim
        # ("Imported hand declares source_type 'manual' but carries
        # reconstruction completion evidence") on the ground that it is claiming
        # the exemption for a hand the pipeline built; the reader reaches the
        # same verdict. Pre-repair it read the pair back verbatim, so ONE
        # hand-edited UPDATE walked a blocked CV hand out of the dependence rule
        # and every other reconstructed-hand blocker while its reconstruction
        # evidence -- and its measured, un-attested assumption dependence --
        # stayed attached and simply stopped being acted on.
        data["source_type"] = "cv_import"
    if data.get("completion_status") not in get_args(CompletionStatus):
        data["completion_status"] = (
            "not_applicable" if data["source_type"] == "manual" else "uncertain"
        )
    if data["source_type"] != "manual" and data["completion_status"] == "not_applicable":
        # The pair import_session rejects and update_hand_status refuses. Reading
        # it back verbatim let a row written through create_hand be reconstructed
        # for study_readiness and exempt for the Study page's own predicate.
        data["completion_status"] = "uncertain"
    if data.get("study_inclusion") not in get_args(StudyInclusion):
        # Conservative, like every degradation above it: 'auto' would silently
        # re-admit a hand the operator excluded, and this reader's contract is
        # that degradation can only ever ADD study-readiness blockers. 'skip'
        # emits STUDY_EXCLUDED_BY_OPERATOR, whose clearing action is the Save
        # control the operator already has.
        data["study_inclusion"] = "skip"
    _degrade_unreadable_cards(data)
    try:
        hand = Hand(**{**data, "created_at": _parse_datetime(data["created_at"])})
    except (ValidationError, TypeError, ValueError):
        hand = _degraded_hand(data)
    return _demote_degraded_hand(_with_unreadable_blobs(hand, unreadable_blobs))


def _demote_degraded_hand(hand: Hand) -> Hand:
    """One place applies the degradation contract to ``review_status``.

    The contract is stated in ``_degraded_hand``: "a hand whose stored facts
    cannot be read was not reviewed in the state it is being shown in, and this
    reader's contract is that degradation can only ever add study-readiness
    blockers". It was applied by ``_degraded_hand`` and by nothing else, so the
    two degradations on the same row reached opposite verdicts: a hand whose
    ``confidence_score`` could not be read reported ``needs_correction``, while a
    hand whose BOARD could not be read -- the columns that ARE the study material,
    and the ones ``_degrade_unreadable_cards`` blanks -- read ``review_status``
    back verbatim. A ``reviewed`` hand hand-edited to a two-card board therefore
    counted as reviewed in ``analytics.compute_session_stats``, in the Insights
    "Unresolved" KPI and in every list row, while the Study page refused it with
    INVALID_HERO_OR_BOARD_CARDS. ``restore_unreadable_card_columns`` -- which
    writes such a column back deliberately, so an import derives the blocker for
    itself -- had the same gap from the writer side.

    Keying the demotion on "does this hand carry ANY read-time degradation
    marker?" rather than on which degradation produced it means the two are one
    rule, and a third marker added to ``DERIVED_EVIDENCE_KEYS`` inherits it
    without anyone remembering this function exists.
    """
    if not any(key in hand.completion_evidence for key in DERIVED_EVIDENCE_KEYS):
        return hand
    if hand.review_status == "needs_correction":
        return hand
    return hand.model_copy(update={"review_status": "needs_correction"})


def _degraded_hand(data: dict[str, Any]) -> Hand:
    """Read a hands row this build cannot validate, without raising into a fetch.

    Same degradation family as ``_degraded_hand_settlement`` and the card
    columns, via ``_salvaged_row``. The offending text is not silently dropped:
    every column that had to be given up is recorded, with what it held, under
    ``UNREADABLE_HAND_COLUMNS_KEY`` in the hand's completion evidence — the same
    read-time, writer-stripped channel ``UNREADABLE_CARDS_KEY`` uses — so study
    readiness reports the exact stored value under UNREADABLE_HAND_COLUMNS
    rather than silently presenting a fallback as the record. ``review_status``
    is forced to ``needs_correction``: a hand whose stored facts cannot be read
    was not reviewed in the state it is being shown in, and this reader's
    contract is that degradation can only ever add study-readiness blockers.
    """
    hand, unreadable = _salvaged_row(
        Hand,
        data,
        {
            "session_id": _coerced_int(data.get("session_id"), 0),
            "hand_number": max(_coerced_int(data.get("hand_number"), 1), 1),
        },
    )
    if not unreadable:
        return hand
    # Merged, never replaced: ``_unreadable_blob_columns`` has already recorded
    # every JSON blob column it had to give up under this same key, and a plain
    # assignment silently dropped that record whenever a scalar column on the
    # same row also failed -- taking the blob's blocker with it.
    already = hand.completion_evidence.get(UNREADABLE_HAND_COLUMNS_KEY)
    recorded: dict[str, object] = dict(already) if isinstance(already, dict) else {}
    recorded.update({name: repr(data.get(name)) for name in unreadable})
    return hand.model_copy(
        update={
            "review_status": "needs_correction",
            "completion_evidence": {
                **hand.completion_evidence,
                UNREADABLE_HAND_COLUMNS_KEY: recorded,
            },
        }
    )


def _blob_columns(model_cls: type[PersistedModel]) -> tuple[tuple[str, type], ...]:
    """The columns of a table stored as a JSON blob, read off the MODEL.

    A field whose annotation is a ``list`` or ``dict`` container is written
    through ``_serialize_json`` and read back through ``_parse_json_object``;
    every other field is a scalar column that pydantic itself validates. Deriving
    the set from ``model_fields`` rather than listing the blob columns of each
    table is the same rule ``_salvaged_row`` follows: a blob column added to any
    model later is covered without anyone remembering this function exists.
    """
    columns: list[tuple[str, type]] = []
    for name, spec in model_cls.model_fields.items():
        if spec.exclude:
            continue
        origin = get_origin(spec.annotation)
        if origin in (list, dict):
            columns.append((name, origin))
    return tuple(columns)


_BLOB_COLUMNS_BY_MODEL: dict[type[PersistedModel], tuple[tuple[str, type], ...]] = {}


# How much of an unreadable blob's stored text a marker keeps. Generous enough
# that every realistic value is recorded whole -- which is what makes
# ``restore_unreadable_columns``'s ``literal_eval`` round trip exact -- and
# bounded, because a blob column is the one place a row can hold megabytes: the
# marker is rendered into a study blocker's detail and travels through export and
# import, so an unbounded copy of the damaged text would be carried into both.
_RECORDED_BLOB_TEXT_LIMIT = 4096


def _recorded_blob_text(raw: object) -> str:
    """What a marker stores for an unreadable blob: its ``repr``, bounded."""
    recorded = repr(raw)
    if len(recorded) <= _RECORDED_BLOB_TEXT_LIMIT:
        return recorded
    return (
        f"{recorded[:_RECORDED_BLOB_TEXT_LIMIT]}... "
        f"(truncated; {len(recorded)} characters stored)"
    )


def _unreadable_blob_columns(
    model_cls: type[PersistedModel], data: dict[str, Any]
) -> dict[str, str]:
    """Name every JSON blob column of this row that could not be read back.

    Detection only: what each reader STORES for a damaged blob is unchanged, so
    no value semantics move. What changes is that the loss stops being silent.

    Every scalar column this build cannot read is named on ``unreadable_columns``
    by ``_salvaged_row``, which is what forces ``review_status='needs_correction'``
    on a hand, ``status='open'`` on a debugging issue, ``is_stale=True`` on
    coaching, and what the accounting cross-check refuses to build a ledger over.
    The BLOB columns were the hole in that rule: ``_parse_json_list`` and
    ``_parse_json_object`` degrade a damaged blob to an empty list/dict, pydantic
    accepts an empty container as a perfectly valid value, and the row comes back
    with ``unreadable_columns == ()`` -- indistinguishable from a row that
    legitimately holds nothing there.

    Two instances show why this had to be closed as a class rather than at a call
    site. ``hands.completion_evidence`` is the CHANNEL the degradation markers
    travel in, so its own damage destroyed the only record that could have
    reported it: the hand kept its Reviewed / Complete / confidence-High badges
    and its place in the Overview "Confirmed result ... from N reviewed hands"
    KPI, for a status ``update_hand_status`` re-derives from that same evidence
    and refuses to issue. ``solver_runs.spot`` is the same silence in another
    table: a run whose spot, ranges or frequencies could not be read came back
    ``completed`` -- the one status study evidence is granted on -- holding empty
    dicts, so a solve was offered as evidence for a spot nobody could read.

    Not every blob was exposed. ``hand_issues.issue_types`` declares
    ``min_length=1``, so its empty degraded value fails validation, reaches
    ``_salvaged_row`` and was already named; that is exactly the difference this
    function removes -- whether a blob column happens to have a constraint that
    rejects its own empty value should not decide whether its loss is reported.

    An empty or absent column is untouched: a hand with no tags, an issue with no
    evidence snapshot and a settlement with no warnings are ordinary states. Only
    text that is PRESENT and cannot be read back as its own container is a
    degradation.
    """
    blob_columns = _BLOB_COLUMNS_BY_MODEL.get(model_cls)
    if blob_columns is None:
        blob_columns = _blob_columns(model_cls)
        _BLOB_COLUMNS_BY_MODEL[model_cls] = blob_columns
    unreadable: dict[str, str] = {}
    for column, container in blob_columns:
        raw = data.get(column)
        if raw is None or raw == "" or isinstance(raw, container):
            continue
        if isinstance(_parse_json_object(raw, None), container):
            continue
        unreadable[column] = _recorded_blob_text(raw)
    return unreadable


def _with_unreadable_blobs(model: _ModelT, unreadable: dict[str, str]) -> _ModelT:
    """Name a degraded blob on the model, exactly where a degraded scalar is named."""
    if not unreadable:
        return model
    return model.model_copy(
        update={
            "unreadable_columns": tuple(
                dict.fromkeys((*model.unreadable_columns, *unreadable))
            )
        }
    )


def _degrade_unreadable_cards(data: dict[str, Any]) -> None:
    """Make a hand-edited card column block instead of hiding the whole session.

    ``Hand`` refuses a board of two cards, a hero card repeated on the board, or
    any unparseable token, so one row written outside the model raised a
    ValidationError out of ``fetch_hands_by_session`` and every other hand in the
    session disappeared with it. That also made two of the three
    INVALID_HERO_OR_BOARD_CARDS branches unreachable: the blocker documented as
    "defense in depth for rows written outside the model" could never see one.

    The offending text is not silently dropped. It is recorded under
    ``UNREADABLE_CARDS_KEY`` in the hand's completion evidence -- an open mapping
    whose unknown keys survive a round trip -- so ``_card_blockers`` can report the
    exact stored value, and so the record travels through export and import. The
    key carries no ``evidence_version``, so it never makes a manual hand look like
    it is carrying reconstruction evidence.
    """
    unreadable: dict[str, str] = {}
    for column, counts in (("hero_cards", {0, 2}), ("board_cards", {0, 3, 4, 5})):
        raw = data.get(column) or ""
        try:
            data[column] = normalize_cards(str(raw), expected_counts=counts)
        except (CardValidationError, ValueError):
            unreadable[column] = str(raw)
            data[column] = ""
    if not unreadable and data.get("hero_cards") and data.get("board_cards"):
        try:
            parse_visible_cards(data["hero_cards"], data["board_cards"])
        except CardParseError as exc:
            # A card visible in two places at once. The board is the field that
            # loses its claim, because the hero's own cards are the anchor the
            # rest of the record is attributed to.
            unreadable["board_cards"] = f"{data['board_cards']} ({exc})"
            data["board_cards"] = ""
    if not unreadable:
        return
    evidence = data.get("completion_evidence")
    data["completion_evidence"] = {
        **(evidence if isinstance(evidence, dict) else {}),
        UNREADABLE_CARDS_KEY: unreadable,
    }


def _preserving_codes(
    stored: tuple[str, ...], submitted: tuple[str, ...]
) -> list[str]:
    """Union preserving stored order: a pipeline code may be added, never removed.

    Used by ``update_hand_completion`` to pin ``warning_codes`` and
    ``rejection_codes`` against a caller-supplied blob. An addition can only
    ever demote the hand; a removal is a promotion of a hand whose facts nobody
    corrected, and only a new reconstruction may make one.
    """
    return [*stored, *[code for code in submitted if code not in stored]]


def _hand_player_from_row(row: sqlite3.Row) -> HandPlayer:
    data = _row_dict(row)
    data["is_hero"] = bool(data["is_hero"])
    try:
        return HandPlayer(**data)
    except (ValidationError, TypeError, ValueError):
        # The salvaged row's `unreadable_columns` marker is an accounting issue
        # (`hand_accounting._unreadable_row_issues`), so a degraded player can
        # never support an authoritative — or study-ready — verdict. That is
        # what keeps this conservative: a stack this build cannot read must not
        # weaken the overcommit check silently.
        player, _ = _salvaged_row(
            HandPlayer,
            data,
            {
                "hand_id": _coerced_int(data.get("hand_id"), 0),
                "player_name": _UNREADABLE_LABEL,
            },
        )
        return player


def _action_from_row(row: sqlite3.Row) -> Action:
    data = _row_dict(row)
    if data.get("is_live_post") is not None:
        data["is_live_post"] = bool(data["is_live_post"])
    try:
        return Action(**data)
    except (ValidationError, TypeError, ValueError):
        # The fallback street/action_type are placeholders, not observations;
        # the `unreadable_columns` marker makes the whole hand's accounting
        # non-authoritative (see `hand_accounting._unreadable_row_issues`), so
        # nothing derived from a degraded action is ever presented as proven.
        action, _ = _salvaged_row(
            Action,
            data,
            {
                "hand_id": _coerced_int(data.get("hand_id"), 0),
                "street": "preflop",
                "action_type": "check",
                "player_name": _UNREADABLE_LABEL,
            },
        )
        return action


def _refuse_display_copy(hand: Hand, verb: str) -> None:
    """A hand whose hero result was substituted for display may never be written.

    ``Hand.derived_result_substituted`` is set only by the read-time substitution
    that shows the derived ledger result in place of ``hands.hero_bb_won`` on an
    authoritative hand. Persisting such an object turns a derivation into an
    observation, and the accounting cross-check compares that column EXACTLY
    precisely because it is meant to be independent evidence of what the hero won.
    """
    if hand.derived_result_substituted:
        raise ValueError(
            f"Refusing to {verb} a hand whose Hero result was substituted with the "
            "derived ledger result for display; re-read the stored hand first."
        )


def _hand_settlement_from_row(row: sqlite3.Row) -> HandSettlement:
    data = _row_dict(row)
    unreadable_blobs = _unreadable_blob_columns(HandSettlement, data)
    try:
        data["no_flop_no_drop"] = bool(data["no_flop_no_drop"])
        data["is_balanced"] = bool(data["is_balanced"])
        data["warnings"] = _parse_json_list(data.get("warnings", "[]"))
        # Deliberately NOT ``_parse_json_list``, which degrades an unreadable
        # column to an empty list. That contract is right for ``warnings``, where
        # losing a note only ever removes noise, and wrong here: dropping a
        # straddle LOWERS the structural forced bet the amount to call is floored
        # at, which is a declaration quietly getting weaker. Raising sends the row
        # through ``_degraded_hand_settlement``, which names the column, forces
        # the status off ``reconciled`` and blocks the hand instead.
        raw_straddles = data.get("straddles", "[]")
        if isinstance(raw_straddles, str):
            parsed_straddles = json.loads(raw_straddles or "[]")
            if not isinstance(parsed_straddles, list):
                raise ValueError("straddles must hold a JSON list of chip amounts.")
            data["straddles"] = parsed_straddles
        elif raw_straddles is None:
            data["straddles"] = []
        data["created_at"] = _parse_datetime(data["created_at"])
        data["updated_at"] = _parse_datetime(data["updated_at"])
        settlement = HandSettlement(**data)
    except (ValidationError, ValueError, TypeError):
        # Every column this row carries, including the timestamps: the whole
        # conversion is inside the guard so no single unreadable cell can raise
        # out of a fetch. See `_degraded_hand_settlement`.
        settlement = _degraded_hand_settlement(data)
    return _with_unreadable_blobs(settlement, unreadable_blobs)


def _degraded_hand_settlement(data: dict[str, Any]) -> HandSettlement:
    """Read a settlement row this build cannot validate, without raising into a fetch.

    ``hand_settlements`` carries no CHECK constraint -- the same threat model the
    dependence rule exists for -- so a hand-edited or forged row can hold a
    negative rake rate, a zero rounding unit, or a NaN. Feeding those straight to
    the validating model raised a ValidationError out of
    ``fetch_hand_settlement`` into ``reconcile_persisted_hand`` and on into the
    Study and Insights pages, which catch ``LedgerError`` only: one unreadable
    row rendered an exception instead of the hand, and took every other hand in
    the session's list down with it.

    Its neighbours already degrade rather than raise -- ``_parse_json_list`` for
    the warnings column two lines above, ``_hand_from_row`` for ``source_type``,
    ``completion_status`` and the card columns -- and they degrade to the most
    conservative reading, which can only ever ADD blockers. So does this:

    * every column that cannot be read falls back to the model default, so no
      unreadable declaration is ever used to derive a number;
    * ``status`` is forced off ``reconciled`` and ``is_balanced`` off True, so
      the hand cannot be authoritative and cannot be study-ready;
    * the reason is named in ``warnings``, which the reconciliation surfaces as
      an issue, so the operator is told which column to fix rather than shown a
      hand that quietly lost its policy.

    Nothing on disk is touched: this is a read-time degradation, and saving the
    settlement through the editor is what rewrites the row.
    """
    safe: dict[str, Any] = {
        "hand_id": data.get("hand_id"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }
    unreadable: list[str] = []
    # The three blind-structure columns are probed TOGETHER, as one declaration,
    # and stand or fall together.
    #
    # Probing them one at a time cannot work, in either order. Probed
    # small-blind-first, a perfectly readable small blind is refused against a
    # ``safe`` mapping that does not yet carry its big blind, and an intact
    # declaration is dropped from a row degraded for an unrelated reason. Probed
    # big-blind-first -- which is what this did -- a TRANSPOSED "5/10" stored as
    # small 10 / big 5 keeps the 5 and drops the 10, converting a declaration
    # nobody could have meant into a smaller one that is perfectly valid. That is
    # strictly worse than dropping it: a floor of 5 covers a big blind that went
    # all-in for 4, so the refusal the declaration was made to answer never
    # fires, and the hand reconciles around a 14-chip pot whose truth is 24.
    #
    # A row whose three columns cannot be read together is a row that did not
    # declare a structure. Dropping all three leaves the floor at zero, which
    # blocks the hand and names the columns -- the conservative direction this
    # whole function exists to take.
    structure_columns = [
        name for name in ("small_blind", "big_blind", "straddles") if name in data
    ]
    if structure_columns:
        declared = {name: data[name] for name in structure_columns}
        try:
            HandSettlement(**{**safe, **declared})
        except ValidationError:
            unreadable.extend(structure_columns)
        else:
            safe.update(declared)
    for name in (
        "status",
        # Probed on its own, and DROPPED to NULL when it cannot be read. That is
        # the conservative direction here and it is not obvious, so it is stated:
        # an unreadable mode becomes an UNDECLARED mode, which on a hand carrying
        # antes is a refusal the operator must clear, whereas keeping a
        # half-readable value would let a mode nobody typed decide whether a
        # consolidated ante is capped.
        "ante_mode",
        "dead_money",
        "rake_rate",
        "rake_cap",
        "rake_rounding_unit",
        "no_flop_no_drop",
        "gross_pot",
        "rake_amount",
        "net_pot",
        "is_balanced",
        "warnings",
    ):
        if name not in data:
            continue
        try:
            HandSettlement(**{**safe, name: data[name]})
        except ValidationError:
            unreadable.append(name)
        else:
            safe[name] = data[name]
    try:
        settlement = HandSettlement(**safe)
    except ValidationError:
        # Even the identity columns are unreadable. There is nothing left to
        # preserve, so the row is reported as present, unreadable, and unusable.
        settlement = HandSettlement(hand_id=int(data.get("hand_id") or 0))
        unreadable = sorted({*unreadable, "hand_id", "created_at", "updated_at"})
    return settlement.model_copy(
        update={
            "status": "unsettled" if settlement.status == "reconciled" else settlement.status,
            "is_balanced": False,
            "warnings": [
                *settlement.warnings,
                f"{UNREADABLE_SETTLEMENT_PREFIX} {', '.join(unreadable) or 'unknown'}.",
            ],
        }
    )


def _settlement_entry_from_row(row: sqlite3.Row) -> SettlementEntry:
    data = _row_dict(row)
    try:
        return SettlementEntry(**data)
    except (ValidationError, TypeError, ValueError):
        # entry_type falls back to "refund" because the fallback must satisfy
        # the model's own pot-index rule with everything else defaulted; the
        # `unreadable_columns` marker blocks the hand's accounting either way.
        entry, _ = _salvaged_row(
            SettlementEntry,
            data,
            {
                "hand_id": _coerced_int(data.get("hand_id"), 0),
                "entry_type": "refund",
                "player_name": _UNREADABLE_LABEL,
            },
        )
        return entry


def _declared_award_state(entries: list[SettlementEntry]) -> dict[str, str]:
    """The declared winner of each pot, as a comparable, auditable snapshot.

    One flat ``{"pot 0": "Hero (20.0)"}`` mapping, because ``hand_corrections``
    stores before/after states as string-valued JSON objects and a nested list
    would be persisted as a Python repr. Row ids are deliberately excluded:
    re-saving the same declaration through the editor renumbers them, and that is
    not a correction of anything.

    The declared ORDER is not excluded, and used to be: the claims were sorted
    alphabetically, so the snapshot could not see the 'Odd-chip order' column at
    all. That column is not cosmetic. ``reconcile_persisted_hand`` sorts the award
    rows by ``(pot_index, entry_order)`` to build ``odd_chip_order``, and
    ``_split_pot`` hands the odd chip to the first name in it, so on a chopped pot
    that cannot divide evenly, swapping two rows moves chips between seats and
    flips the derived hero result -- and with the order outside the snapshot,
    ``before == after``, no ``settlement_award_update`` correction was recorded,
    no ``source_facts_corrected`` was written into the evidence, and
    ``ACCOUNTING_NOT_AUTHORITATIVE`` cleared with the audit trail recording
    nothing. Re-saving an untouched editor writes the same orders back, so an
    idempotent save still compares equal.
    """
    by_pot: dict[str, list[tuple[int, str]]] = {}
    for entry in entries:
        if entry.entry_type != "award":
            continue
        label = entry.player_key or entry.player_name or "?"
        amount = "unspecified" if entry.amount is None else f"{entry.amount:g}"
        by_pot.setdefault(f"pot {entry.pot_index if entry.pot_index is not None else 0}", []).append(
            (entry.entry_order, f"{label} ({amount})")
        )
    return {
        pot: "; ".join(claim for _, claim in sorted(claims))
        for pot, claims in sorted(by_pot.items())
    }


def _declared_settlement_inputs(
    settlement: HandSettlement | None,
) -> dict[str, object] | None:
    """Everything one settlement row DECLARES, and nothing it derives.

    The declared half is the blind structure, the ante mode, the rake policy and
    the dead money: figures nothing observed, which move the net pot and the hero
    result the coaching prompt and the solver input were built from. Changing any
    of them invalidates those.

    The ANTE MODE is in for a stronger reason than the blind structure: it moves
    CHIPS. The same recording laid out under ``SINGLE_PAYER_TABLE_ANTE`` and
    under ``PER_PLAYER`` can give different pots, different eligible sets and a
    different hero result -- blinds 1/2 with a 2-chip big-blind ante against an
    all-in small blind is main 5 one way and main 4 the other. Retained analysis
    produced under one reading is not evidence about the other.

    The blind structure is in even though it moves no chip figure on its own.
    It decides whether the preflop wagering can be judged at all, so it decides
    whether the hand is authoritative -- and an authoritative hand is exactly
    the one whose DERIVED hero result is substituted into every list view, the
    coaching prompt and the solver spot. A hand that became study-ready because
    a structure was declared, and stops being study-ready when it is withdrawn,
    is a hand whose retained analysis rested on it.

    The rest of the row is derived and is deliberately out. ``gross_pot``,
    ``rake_amount`` and ``net_pot`` are functions of the players, the actions,
    the declared awards and the policy above, and every writer of those already
    invalidates on its own -- so a derived figure cannot move without the
    invalidation having happened at its source. ``status``, ``is_balanced`` and
    ``warnings`` are the cross-check's VERDICT on those same figures; a hand
    re-blessed after a correction moves from ``needs_correction`` back to
    ``reconciled`` without a chip moving, and treating that as an evidence change
    would stale the coaching the operator had just re-run in order to get there.
    ``updated_at`` moves on every write by construction.

    ``None`` means the hand had no settlement row at all, which is not equal to
    any declaration and therefore always counts as a change.
    """
    if settlement is None:
        return None
    return settlement.model_dump(
        include={
            "small_blind",
            "big_blind",
            "straddles",
            "ante_mode",
            "dead_money",
            "rake_rate",
            "rake_cap",
            "rake_rounding_unit",
            "no_flop_no_drop",
        }
    )


def _settlement_entry_state(entries: list[SettlementEntry]) -> tuple[tuple, ...]:
    """The declared awards and refunds as an order-insensitive comparable set.

    Row ids are out for the same reason ``_declared_award_state`` leaves them
    out: rebuilding an unchanged declaration renumbers them. ``entry_order`` is
    in, because it decides who takes the odd chip on a chopped pot.
    """
    return tuple(
        sorted(
            (
                entry.entry_type,
                -1 if entry.pot_index is None else int(entry.pot_index),
                entry.player_key or "",
                entry.player_name,
                "" if entry.amount is None else f"{entry.amount:.6f}",
                entry.entry_order,
            )
            for entry in entries
        )
    )


def _review_from_row(row: sqlite3.Row) -> HandReview:
    data = _row_dict(row)
    data["is_stale"] = bool(data.get("is_stale", 0))
    try:
        return HandReview(**{**data, "created_at": _parse_datetime(data["created_at"])})
    except (ValidationError, TypeError, ValueError):
        review, unreadable = _salvaged_row(
            HandReview,
            data,
            {
                "hand_id": _coerced_int(data.get("hand_id"), 0),
                "hand_summary": _UNREADABLE_LABEL,
                "theory_coach": _UNREADABLE_LABEL,
                "exploit_coach": _UNREADABLE_LABEL,
                "study_lesson": _UNREADABLE_LABEL,
            },
        )
        # A retained review this build cannot fully read is not current
        # evidence; presenting it as current would clear the staleness blocker
        # on the strength of a row nobody can inspect. Stale adds blockers only.
        return review.model_copy(
            update={
                "is_stale": True,
                "stale_reason": (
                    "Stored review row could not be fully read: "
                    f"{', '.join(unreadable)}."
                ),
            }
        )


def _hand_correction_from_row(row: sqlite3.Row) -> HandCorrection:
    data = _row_dict(row)
    unreadable_blobs = _unreadable_blob_columns(HandCorrection, data)
    data["before_state"] = _parse_json_dict(data.get("before_state", "{}"))
    data["after_state"] = _parse_json_dict(data.get("after_state", "{}"))
    try:
        correction = HandCorrection(
            **{**data, "created_at": _parse_datetime(data["created_at"])}
        )
    except (ValidationError, TypeError, ValueError):
        correction, _ = _salvaged_row(
            HandCorrection,
            data,
            {
                "hand_id": _coerced_int(data.get("hand_id"), 0),
                "correction_type": "hand_facts",
            },
        )
    return _with_unreadable_blobs(correction, unreadable_blobs)


def _recognised_issue_types(stored: object) -> list[str]:
    """The categories a damaged ``issue_types`` column still names, in order.

    Salvaging the whole column to ``["other"]`` discarded the readable half of a
    partly-damaged list, and ``other`` is the one category deliberately outside
    ``RELEASE_BLOCKING_ISSUE_TYPES``. ``["cards", "solver_output"]`` -- what a
    database written by a build with one more category looks like to this one --
    therefore read back as an ordinary ``other`` issue that
    ``resolve_hand_issue`` closed with no regression at all, and exported as one
    too, so the downgrade outlived the damaged row.
    """
    if not isinstance(stored, list):
        return []
    return list(
        dict.fromkeys(value for value in stored if value in get_args(HandIssueType))
    )


def _hand_issue_from_row(row: sqlite3.Row) -> HandIssue:
    data = _row_dict(row)
    # Read before the parses below turn a damaged column into an empty container.
    # `issue_types` was already covered, but only because `min_length=1` rejects
    # its own empty degraded value and pushes the row into `_salvaged_row`;
    # `evidence_snapshot` has no such constraint, so the immutable snapshot an
    # issue is filed on used to disappear with nothing recorded.
    unreadable_blobs = _unreadable_blob_columns(HandIssue, data)
    data["issue_types"] = _parse_json_list(data.get("issue_types", "[]"))
    data["evidence_snapshot"] = _parse_json_object(
        data.get("evidence_snapshot", "{}"), {}
    )
    try:
        return _with_unreadable_blobs(
            HandIssue(
                **{
                    **data,
                    "created_at": _parse_datetime(data["created_at"]),
                    "updated_at": _parse_datetime(data["updated_at"]),
                    "resolved_at": _parse_optional_datetime(data.get("resolved_at")),
                }
            ),
            unreadable_blobs,
        )
    except (ValidationError, TypeError, ValueError):
        issue, unreadable = _salvaged_row(
            HandIssue,
            data,
            {
                "hand_id": _coerced_int(data.get("hand_id"), 0),
                "description": _UNREADABLE_LABEL,
                "issue_types": _recognised_issue_types(data.get("issue_types"))
                or ["other"],
            },
        )
        issue = _with_unreadable_blobs(issue, unreadable_blobs)
        if not unreadable and not unreadable_blobs:
            return issue
        # A resolution recorded in a row this build cannot fully read is not a
        # resolution anyone can verify. Open blocks study readiness
        # (OPEN_DEBUGGING_ISSUE); resolved clears it, so resolved is the one
        # reading a degraded row may not claim.
        return issue.model_copy(update={"status": "open"})


def _coaching_response_from_row(row: sqlite3.Row) -> CoachingResponse:
    data = _row_dict(row)
    unreadable_blobs = _unreadable_blob_columns(CoachingResponse, data)
    data["is_stale"] = bool(data.get("is_stale", 0))
    data["parsed_sections"] = _parse_json_dict(data.get("parsed_sections", "{}"))
    try:
        return _with_unreadable_blobs(
            CoachingResponse(
                **{**data, "created_at": _parse_datetime(data["created_at"])}
            ),
            unreadable_blobs,
        )
    except (ValidationError, TypeError, ValueError):
        response, unreadable = _salvaged_row(
            CoachingResponse,
            data,
            {
                "provider_name": _UNREADABLE_LABEL,
                "model_name": _UNREADABLE_LABEL,
                "raw_prompt": "",
                "raw_response": "",
                "review_type": "hand",
            },
        )
        # Same argument as _review_from_row: degraded coaching is never current.
        return _with_unreadable_blobs(
            response.model_copy(
                update={
                    "is_stale": True,
                    "stale_reason": (
                        "Stored coaching row could not be fully read: "
                        f"{', '.join(unreadable)}."
                    ),
                }
            ),
            unreadable_blobs,
        )


def _solver_range_profile_from_row(row: sqlite3.Row) -> SolverRangeProfile:
    data = _row_dict(row)
    try:
        return SolverRangeProfile(
            **{
                **data,
                "created_at": _parse_datetime(data["created_at"]),
                "updated_at": _parse_datetime(data["updated_at"]),
            }
        )
    except (ValidationError, TypeError, ValueError):
        profile, _ = _salvaged_row(
            SolverRangeProfile,
            data,
            {"name": _UNREADABLE_LABEL, "notation": _UNREADABLE_LABEL},
        )
        return profile


def _solver_run_from_row(row: sqlite3.Row) -> SolverRun:
    data = _row_dict(row)
    unreadable_blobs = _unreadable_blob_columns(SolverRun, data)
    # run_parameters is read back explicitly, including the empty object a
    # pre-18 row carries, so the model default never backfills an old run with
    # settings it was not solved under.
    for key in ("spot", "range_ip", "range_oop", "evidence", "run_parameters"):
        data[key] = _parse_json_object(data.get(key, "{}"), {})
    data["assumptions"] = _parse_json_object(data.get("assumptions", "[]"), [])
    try:
        run = SolverRun(
            **{
                **data,
                "created_at": _parse_datetime(data["created_at"]),
                "started_at": _parse_optional_datetime(data.get("started_at")),
                "completed_at": _parse_optional_datetime(data.get("completed_at")),
                "heartbeat_at": _parse_optional_datetime(data.get("heartbeat_at")),
            }
        )
        unreadable: tuple[str, ...] = ()
    except (ValidationError, TypeError, ValueError):
        run, unreadable = _salvaged_row(
            SolverRun,
            data,
            {
                "hand_id": _coerced_int(data.get("hand_id"), 0),
                "input_hash": _UNREADABLE_LABEL,
            },
        )
    run = _with_unreadable_blobs(run, unreadable_blobs)
    # `completed` is the status study evidence is granted on, and an unreadable
    # status is not evidence of anything; both degrade to `stale`, which blocks
    # (STALE_SOLVER_EVIDENCE) until re-run or deleted. A run whose SPOT, ranges
    # or frequencies could not be read is the same case: `completed` would grant
    # study evidence over a blob nobody can read. Queued/running are left alone
    # so job control still sees them.
    degraded = bool(unreadable) or bool(unreadable_blobs)
    if "status" in unreadable or (degraded and run.status == "completed"):
        run = run.model_copy(update={"status": "stale"})
    return run


def _video_from_row(row: sqlite3.Row) -> VideoRecord:
    data = _row_dict(row)
    try:
        return VideoRecord(
            **{**data, "uploaded_at": _parse_datetime(data["uploaded_at"])}
        )
    except (ValidationError, TypeError, ValueError):
        video, _ = _salvaged_row(
            VideoRecord,
            data,
            {
                "original_filename": _UNREADABLE_LABEL,
                "stored_path": "",
                "file_size_bytes": 0,
            },
        )
        return video


def _processing_job_from_row(row: sqlite3.Row) -> ProcessingJob:
    data = _row_dict(row)
    try:
        return ProcessingJob(
            **{
                **data,
                "created_at": _parse_datetime(data["created_at"]),
                "started_at": _parse_optional_datetime(data["started_at"]),
                "completed_at": _parse_optional_datetime(data["completed_at"]),
                "heartbeat_at": _parse_optional_datetime(data.get("heartbeat_at")),
            }
        )
    except (ValidationError, TypeError, ValueError):
        job, unreadable = _salvaged_row(
            ProcessingJob,
            data,
            {
                "job_type": "cv_reconstruction",
                "video_id": _coerced_int(data.get("video_id"), 0),
            },
        )
        if "status" in unreadable:
            # An unreadable status must not read as `queued` (the salvage
            # default), which the job runner would treat as work to start.
            job = job.model_copy(update={"status": "failed"})
        return job


def _extracted_frame_from_row(row: sqlite3.Row) -> ExtractedFrame:
    data = _row_dict(row)
    try:
        return ExtractedFrame(
            **{**data, "created_at": _parse_datetime(data["created_at"])}
        )
    except (ValidationError, TypeError, ValueError):
        frame, _ = _salvaged_row(
            ExtractedFrame,
            data,
            {
                "video_id": _coerced_int(data.get("video_id"), 0),
                "job_id": _coerced_int(data.get("job_id"), 0),
                "timestamp_seconds": 0.0,
                "frame_index": 0,
                "image_path": "",
            },
        )
        return frame


def _reconstruction_frame_review_from_row(
    row: sqlite3.Row,
) -> ReconstructionFrameReview:
    data = _row_dict(row)
    unreadable_blobs = _unreadable_blob_columns(ReconstructionFrameReview, data)
    data["issue_types"] = _parse_json_list(data.get("issue_types", "[]"))
    try:
        return _with_unreadable_blobs(
            ReconstructionFrameReview(
                **{
                    **data,
                    "created_at": _parse_datetime(data["created_at"]),
                    "updated_at": _parse_datetime(data["updated_at"]),
                }
            ),
            unreadable_blobs,
        )
    except (ValidationError, TypeError, ValueError):
        review, unreadable = _salvaged_row(
            ReconstructionFrameReview,
            data,
            {
                "job_id": _coerced_int(data.get("job_id"), 0),
                "hand_number": max(_coerced_int(data.get("hand_number"), 1), 1),
                "source_image": _UNREADABLE_LABEL,
                "timestamp_seconds": 0.0,
            },
        )
        if "status" in unreadable:
            # The salvage default is already `unreviewed`, but state it: an
            # unreadable human verdict is no verdict.
            review = review.model_copy(update={"status": "unreviewed"})
        return _with_unreadable_blobs(review, unreadable_blobs)


def _roi_profile_from_row(row: sqlite3.Row) -> ROIProfile:
    data = _row_dict(row)
    data["is_active"] = bool(data["is_active"])
    try:
        return ROIProfile(
            **{
                **data,
                "created_at": _parse_datetime(data["created_at"]),
                "updated_at": _parse_datetime(data["updated_at"]),
            }
        )
    except (ValidationError, TypeError, ValueError):
        profile, _ = _salvaged_row(ROIProfile, data, {"name": _UNREADABLE_LABEL})
        return profile


def _roi_region_from_row(row: sqlite3.Row) -> ROIRegion:
    data = _row_dict(row)
    try:
        return ROIRegion(
            **{
                **data,
                "created_at": _parse_datetime(data["created_at"]),
                "updated_at": _parse_datetime(data["updated_at"]),
            }
        )
    except (ValidationError, TypeError, ValueError):
        region, _ = _salvaged_row(
            ROIRegion,
            data,
            {
                "profile_id": _coerced_int(data.get("profile_id"), 0),
                "roi_key": _UNREADABLE_LABEL,
                "x": 0,
                "y": 0,
                "width": 1,
                "height": 1,
            },
        )
        return region


# ``_rake_policy`` -- the four-field tuple this module used to compare two
# settlements by -- is gone. It was the last field list in the disclosure path,
# and it failed twice in the way a field list always fails: once by omitting
# ``rake_rounding_unit`` (rounds 8), and once by comparing the policy rather than
# the chips, so an unchanged policy over a corrected action line inherited an
# attestation earned against a completely different quantity. Both writers now
# compare ``_declared_chips_taken``, which is a measurement of this hand and has
# no field list to keep in step.


def _parse_json_list(value: str | None) -> list[str]:
    """Degrade an unreadable stored list to an empty one; never raise into a fetch.

    Same contract as ``_parse_json_object``, and for the same reason: ``tags`` and
    ``hand_settlements.warnings`` are TEXT columns with no CHECK constraint, so a
    hand-edited row used to raise ``JSONDecodeError`` out of
    ``fetch_hands_by_session`` and take every other hand in the session with it,
    while the evidence blob two columns over degraded cleanly.
    """
    if not value:
        return []
    parsed = _parse_json_object(value, [])
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _parse_json_dict(value: str | None) -> dict[str, str]:
    """Degrade an unreadable stored string-map to an empty one; never raise.

    Same contract as ``_parse_json_list`` and for the same reason: a hand-edited
    ``hand_corrections.before_state`` or ``coaching_reviews.parsed_sections``
    used to raise ``JSONDecodeError`` out of the fetch and take the whole list
    view down with it.
    """
    if not value:
        return {}
    parsed = _parse_json_object(value, {})
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(item) for key, item in parsed.items()}


def _parse_json_object(value: object, default: Any) -> Any:
    """Degrade any unreadable stored blob to ``default``; never raise into a fetch.

    ``ValueError`` covers UnicodeDecodeError as well as JSONDecodeError: a TEXT
    column can still hold a BLOB, and json.loads decodes bytes before parsing, so
    one damaged row must not make every hand in the session unreadable.
    ``RecursionError`` is a RuntimeError rather than a ValueError, so a deeply
    nested blob needs its own clause to degrade instead of escaping into a fetch.
    """
    if not value:
        return default
    try:
        # Sanitised on read as well as on write: a row an older build already
        # stored can hold a bare NaN/Infinity token, which json.loads accepts and
        # which would otherwise flow straight back out through the JSON export.
        return _json_representable(json.loads(value))  # type: ignore[arg-type]
    except (TypeError, ValueError, RecursionError):
        return default


# TODO: Add separate repository modules for CV/OCR-derived hand imports later.
# TODO: Add migration management before this grows beyond the first local schema.
