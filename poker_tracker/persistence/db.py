from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypeVar, get_args

from pydantic import ValidationError

from poker_tracker.math.accounting import (
    LedgerError,
    RakePolicy,
    build_ledger_from_records,
)
from poker_tracker.math.cards import CardParseError, parse_visible_cards
from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence.backup import backup_database
from poker_tracker.persistence.completion import (
    UNREADABLE_CARDS_KEY,
    UNREADABLE_HAND_COLUMNS_KEY,
    confirm_assumption,
    derive_completion_status,
    dump_completion_evidence,
    is_assumption_dependence_code,
    parse_completion_evidence,
    requires_assumption_attestation,
    set_declared_settlement_code,
    strip_derived_evidence_markers,
)
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    CompletionStatus,
    ExtractedFrame,
    Hand,
    HandCorrection,
    HandIssue,
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
    VideoRecord,
)
from poker_tracker.persistence.validation import CardValidationError, normalize_cards
from poker_tracker.ui.roi import validate_roi_bounds

# Anchored to the project root so launching from another directory does not
# silently create a second, empty database.
DEFAULT_DB_PATH = os.environ.get(
    "POKER_DB_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "poker_tracker.db"),
)
SCHEMA_VERSION = 13
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


class PokerDatabase:
    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        *,
        busy_timeout_ms: int = BUSY_TIMEOUT_MS,
    ) -> None:
        self.db_path = str(db_path)
        self._busy_timeout_ms = max(int(busy_timeout_ms), 0)
        # One connection is shared across Streamlit's script-run threads, so every
        # statement goes through _execute() under a re-entrant lock, and grouped
        # writes use transaction() for atomicity.
        self._lock = threading.RLock()
        self._txn_depth = 0
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
        except RuntimeError:
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
            if self._txn_depth == 0:
                self._connection.commit()

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
                f"{backup_module.BACKUPS_DIR}; the migration was not applied and "
                "the database "
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

            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                original_filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
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

    def create_hand(self, hand: Hand) -> Hand:
        _refuse_display_copy(hand, "store")
        payload = hand.model_dump()
        cursor = self._execute(
            """
            INSERT INTO hands (
                session_id, hand_number, game_type, blinds_antes, table_size,
                effective_stack, hero_position, hero_cards, board_cards, pot_size,
                result, hero_bb_won, review_status, confidence_score, source_type,
                tags, notes, created_at, completion_status, completion_evidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _serialize_json(
                    strip_derived_evidence_markers(payload["completion_evidence"])
                ),
            ),
        )
        self._commit()
        return hand.model_copy(update={"id": cursor.lastrowid})

    def update_hand_status(self, hand_id: int, review_status: str) -> None:
        """Set the review status, refusing to promote a hand the store knows is unproven.

        This is the unbypassable floor, not the full readiness rule: db.py can only
        see single-table facts. Accounting, coaching, solver, and per-render user
        confirmation are enforced at the UI choke point, because hand_accounting
        imports db.py and the layering must not invert.
        """
        if review_status not in get_args(ReviewStatus):
            raise ValueError(f"Unknown review status: {review_status!r}")
        if review_status == "reviewed":
            row = self._execute(
                """
                SELECT
                    h.completion_status AS completion_status,
                    h.completion_evidence AS completion_evidence,
                    h.source_type AS source_type,
                    EXISTS(
                        SELECT 1 FROM hand_issues
                        WHERE hand_id = h.id AND status = 'open'
                    ) AS has_open_issue
                FROM hands AS h
                WHERE h.id = ?
                """,
                (hand_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Hand not found.")
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
            if row["has_open_issue"]:
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
        """Write an unreadable card column back verbatim, so a round trip is faithful.

        ``_hand_from_row`` blanks a card column it cannot read and records what it
        held under ``UNREADABLE_CARDS_KEY``. The exporter therefore emits a blank
        column, and the importer strips the marker (it is a derivation, and
        persisting it made INVALID_HERO_OR_BOARD_CARDS permanent). Between them a
        board that "could not be read" silently became a hand with "no board
        recorded" -- a legitimate, unblocked state for a preflop hand -- and the
        text that proved the corruption was gone from the database entirely.

        Restoring the recorded text puts the PRODUCER of the blocker back, so the
        importing database derives it for itself, exactly as the exporting one
        did, and correcting the column still clears it. The value bypasses the
        model deliberately -- ``Hand`` refuses it, which is why it was blanked --
        and it is written only into a column the payload left empty, so it can
        only ever add a blocker, never remove one.
        """
        for column in ("hero_cards", "board_cards"):
            value = recorded.get(column)
            if not isinstance(value, str) or not value.strip():
                continue
            self._execute(
                f"UPDATE hands SET {column} = ? WHERE id = ? AND {column} = ''",
                (value, hand_id),
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
        if stored.completion_status == "partial":
            # Sticky, exactly as in _record_source_correction_in_evidence: no
            # evidence write restores missing footage. A hand whose column was set
            # to `partial` by a source this evidence does not repeat -- the v13
            # migration, or an import that honoured a payload's stronger claim over
            # a weaker re-derivation -- would otherwise be laundered up to
            # `complete` by one acknowledgement.
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
        clauses: list[str] = []
        params: list[object] = []
        if hand_id is not None:
            clauses.append("hand_id = ?")
            params.append(hand_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        rows = self._execute(
            f"""
            SELECT * FROM hand_issues
            {where}
            ORDER BY created_at DESC, id DESC
            """,
            tuple(params),
        ).fetchall()
        return [_hand_issue_from_row(row) for row in rows]

    def resolve_hand_issue(
        self, issue_id: int, *, resolution_notes: str
    ) -> HandIssue:
        notes = resolution_notes.strip()
        if not notes:
            raise ValueError("Resolution notes are required.")
        now = datetime.now(UTC)
        with self.transaction():
            cursor = self._execute(
                """
                UPDATE hand_issues
                SET status = 'resolved', resolution_notes = ?,
                    updated_at = ?, resolved_at = ?
                WHERE id = ? AND status = 'open'
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
        stored = self.fetch_hand(hand_id)
        if stored is None:
            raise ValueError("Hand not found.")
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
        if not is_hero:
            return
        row = self._execute(
            """
            SELECT id FROM hand_players
            WHERE hand_id = ? AND is_hero = 1
              AND (? IS NULL OR id != ?)
            LIMIT 1
            """,
            (hand_id, exclude_player_id, exclude_player_id),
        ).fetchone()
        if row is not None:
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
                is_live_post, pot_before, stack_before, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        updated = parse_completion_evidence(payload)
        status = derive_completion_status(updated, source_type=row["source_type"])
        if row["completion_status"] == "partial":
            # Sticky: no correction restores missing footage, and a hand whose
            # evidence has become unreadable must not lose the stronger claim.
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
        payload = settlement.model_dump()
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
                hand_id, status, dead_money, rake_rate, rake_cap,
                rake_rounding_unit, no_flop_no_drop, gross_pot, rake_amount,
                net_pot, is_balanced, warnings, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hand_id) DO UPDATE SET
                status = excluded.status,
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
        # suspicious fields. `_declared_chips_taken` derives the hand's ledger
        # under the stored policy and again under a neutral one and reports how
        # many chips each declaration actually moves; the code is written exactly
        # when that number is non-zero, and the attestation is bound to the
        # number rather than to the policy tuple.
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
        self._stale_retained_analysis(settlement.hand_id)
        self._demote_reviewed_hand(settlement.hand_id)
        self._commit()
        saved = self.fetch_hand_settlement(settlement.hand_id)
        if saved is None:
            raise RuntimeError("Settlement upsert did not persist a row.")
        return saved

    def fetch_hand_settlement(self, hand_id: int) -> HandSettlement | None:
        row = self._execute(
            "SELECT * FROM hand_settlements WHERE hand_id = ?", (hand_id,)
        ).fetchone()
        return None if row is None else _hand_settlement_from_row(row)

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

    def _insert_settlement_entry(self, entry: SettlementEntry) -> SettlementEntry:
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
        # Both writers run this; only the public one discloses the re-declaration.
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
            before = _declared_award_state(self.fetch_settlement_entries(hand_id))
            self._execute("DELETE FROM settlement_entries WHERE hand_id = ?", (hand_id,))
            # The private insert: this method took its own before/after snapshot
            # above, and the public writer's per-row snapshot would compare each
            # half-rebuilt row set against the last and record a correction per
            # entry for an unchanged declaration.
            saved = [self._insert_settlement_entry(entry) for entry in entries]
            after = _declared_award_state(saved)
            # Also runs when `entries` is empty, which clears every award.
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
        """
        deleted = 0
        for table in ("coaching_reviews", "hand_reviews"):
            cursor = self._execute(
                f"DELETE FROM {table} WHERE hand_id = ? AND is_stale = 1", (hand_id,)
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
                hand_id, status, backend_name, backend_version, input_hash,
                spot, range_ip, range_oop, assumptions, evidence,
                command_path, result_path, log_path, exploitability_pct,
                runtime_seconds, error_message, pid, heartbeat_at, created_at,
                started_at, completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["hand_id"],
                payload["status"],
                payload["backend_name"],
                payload["backend_version"],
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
                payload["error_message"],
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
        row = self._execute(
            """
            SELECT * FROM solver_runs
            WHERE input_hash = ? AND status = 'completed'
            ORDER BY completed_at DESC, id DESC
            LIMIT 1
            """,
            (input_hash,),
        ).fetchone()
        return None if row is None else _solver_run_from_row(row)

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
                duration_seconds, fps, width, height, frame_count, uploaded_at, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["session_id"],
                payload["original_filename"],
                payload["stored_path"],
                payload["file_size_bytes"],
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
                payload["message"],
                payload["error_message"],
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
        pid: int | None = None,
        heartbeat_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        current = self.fetch_processing_job(job_id)
        if current is None:
            raise ValueError(f"Processing job not found: {job_id}")
        self._execute(
            """
            UPDATE processing_jobs
            SET status = ?, progress_percent = ?, message = ?, error_message = ?,
                pid = ?, heartbeat_at = ?, started_at = ?, completed_at = ?
            WHERE id = ?
            """,
            (
                status or current.status,
                current.progress_percent if progress_percent is None else progress_percent,
                current.message if message is None else message,
                current.error_message if error_message is None else error_message,
                current.pid if pid is None else pid,
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
            ),
        )
        self._commit()

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

    def fetch_running_jobs(self) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs WHERE status = 'running' ORDER BY created_at, id"
        ).fetchall()
        return [_processing_job_from_row(row) for row in rows]

    def fetch_active_jobs(self) -> list[ProcessingJob]:
        rows = self._execute(
            "SELECT * FROM processing_jobs WHERE status IN ('queued', 'running') "
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
    data["tags"] = _parse_json_list(data.get("tags", "[]"))
    # _parse_json_object, never _parse_json_dict: the latter str()s every value and
    # would flatten nested evidence into a Python repr.
    evidence = _parse_json_object(data.get("completion_evidence", "{}"), {})
    data["completion_evidence"] = evidence if isinstance(evidence, dict) else {}
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
    _degrade_unreadable_cards(data)
    try:
        return Hand(**{**data, "created_at": _parse_datetime(data["created_at"])})
    except (ValidationError, TypeError, ValueError):
        return _degraded_hand(data)


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
    recorded = {name: repr(data.get(name)) for name in unreadable}
    return hand.model_copy(
        update={
            "review_status": "needs_correction",
            "completion_evidence": {
                **hand.completion_evidence,
                UNREADABLE_HAND_COLUMNS_KEY: recorded,
            },
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
    try:
        data["no_flop_no_drop"] = bool(data["no_flop_no_drop"])
        data["is_balanced"] = bool(data["is_balanced"])
        data["warnings"] = _parse_json_list(data.get("warnings", "[]"))
        data["created_at"] = _parse_datetime(data["created_at"])
        data["updated_at"] = _parse_datetime(data["updated_at"])
        return HandSettlement(**data)
    except (ValidationError, ValueError, TypeError):
        # Every column this row carries, including the timestamps: the whole
        # conversion is inside the guard so no single unreadable cell can raise
        # out of a fetch. See `_degraded_hand_settlement`.
        return _degraded_hand_settlement(data)


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
    for name in (
        "status",
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
    data["before_state"] = _parse_json_dict(data.get("before_state", "{}"))
    data["after_state"] = _parse_json_dict(data.get("after_state", "{}"))
    try:
        return HandCorrection(
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
        return correction


def _hand_issue_from_row(row: sqlite3.Row) -> HandIssue:
    data = _row_dict(row)
    data["issue_types"] = _parse_json_list(data.get("issue_types", "[]"))
    data["evidence_snapshot"] = _parse_json_object(
        data.get("evidence_snapshot", "{}"), {}
    )
    try:
        return HandIssue(
            **{
                **data,
                "created_at": _parse_datetime(data["created_at"]),
                "updated_at": _parse_datetime(data["updated_at"]),
                "resolved_at": _parse_optional_datetime(data.get("resolved_at")),
            }
        )
    except (ValidationError, TypeError, ValueError):
        issue, unreadable = _salvaged_row(
            HandIssue,
            data,
            {
                "hand_id": _coerced_int(data.get("hand_id"), 0),
                "description": _UNREADABLE_LABEL,
                "issue_types": ["other"],
            },
        )
        if not unreadable:
            return issue
        # A resolution recorded in a row this build cannot fully read is not a
        # resolution anyone can verify. Open blocks study readiness
        # (OPEN_DEBUGGING_ISSUE); resolved clears it, so resolved is the one
        # reading a degraded row may not claim.
        return issue.model_copy(update={"status": "open"})


def _coaching_response_from_row(row: sqlite3.Row) -> CoachingResponse:
    data = _row_dict(row)
    data["is_stale"] = bool(data.get("is_stale", 0))
    data["parsed_sections"] = _parse_json_dict(data.get("parsed_sections", "{}"))
    try:
        return CoachingResponse(
            **{**data, "created_at": _parse_datetime(data["created_at"])}
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
        return response.model_copy(
            update={
                "is_stale": True,
                "stale_reason": (
                    "Stored coaching row could not be fully read: "
                    f"{', '.join(unreadable)}."
                ),
            }
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
    for key in ("spot", "range_ip", "range_oop", "evidence"):
        data[key] = _parse_json_object(data.get(key, "{}"), {})
    data["assumptions"] = _parse_json_object(data.get("assumptions", "[]"), [])
    try:
        return SolverRun(
            **{
                **data,
                "created_at": _parse_datetime(data["created_at"]),
                "started_at": _parse_optional_datetime(data.get("started_at")),
                "completed_at": _parse_optional_datetime(data.get("completed_at")),
                "heartbeat_at": _parse_optional_datetime(data.get("heartbeat_at")),
            }
        )
    except (ValidationError, TypeError, ValueError):
        run, unreadable = _salvaged_row(
            SolverRun,
            data,
            {
                "hand_id": _coerced_int(data.get("hand_id"), 0),
                "input_hash": _UNREADABLE_LABEL,
            },
        )
        # `completed` is the status study evidence is granted on, and an
        # unreadable status is not evidence of anything; both degrade to
        # `stale`, which blocks (STALE_SOLVER_EVIDENCE) until re-run or
        # deleted. Queued/running are left alone so job control still sees
        # them.
        if "status" in unreadable or run.status == "completed":
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
    data["issue_types"] = _parse_json_list(data.get("issue_types", "[]"))
    try:
        return ReconstructionFrameReview(
            **{
                **data,
                "created_at": _parse_datetime(data["created_at"]),
                "updated_at": _parse_datetime(data["updated_at"]),
            }
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
        return review


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
