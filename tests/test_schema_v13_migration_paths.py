"""Schema 13 against genuinely old-shaped databases.

The migration tests in ``test_persistence_integrity`` rewind a current-shape
database by dropping the two v13 columns. That proves the v12 -> v13 step but
not the rest of the chain, because every table still has its modern shape. The
databases here are built from raw pre-v6, v5, and v11-era DDL: tables are
missing, columns are missing, and action order is inconsistent, exactly as a
real database written by an older build would be.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import persist_reconciliation
from poker_tracker.services.study_readiness import evaluate_study_readiness

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC).isoformat()

# The shape of the tables that changed between schema 5 and schema 13. Anything
# not listed here is created untouched by _create_base_schema on first open.
_V5_SCHEMA = """
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date_played TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT '',
    stakes TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE hands (
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
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE hand_players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    position TEXT NOT NULL DEFAULT '',
    starting_stack REAL,
    is_hero INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
);

CREATE TABLE actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL,
    street TEXT NOT NULL,
    action_index INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    position TEXT NOT NULL DEFAULT '',
    action_type TEXT NOT NULL,
    amount REAL,
    pot_before REAL,
    stack_before REAL,
    notes TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
);

CREATE TABLE hand_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL,
    hand_summary TEXT NOT NULL,
    theory_coach TEXT NOT NULL,
    exploit_coach TEXT NOT NULL,
    ev_math_notes TEXT NOT NULL DEFAULT '',
    study_lesson TEXT NOT NULL,
    next_review_question TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
);

CREATE TABLE coaching_reviews (
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
    created_at TEXT NOT NULL,
    FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE videos (
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

CREATE TABLE processing_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    video_id INTEGER NOT NULL,
    progress_percent REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
);
"""

# Everything schema 7 through 11 added, so a v11-shaped database only lacks
# hand_issues (v12) and the two completion columns (v13).
_V7_TO_V11_ADDITIONS = """
ALTER TABLE hand_players ADD COLUMN player_key TEXT;
ALTER TABLE hand_players ADD COLUMN seat_index INTEGER;
ALTER TABLE actions ADD COLUMN player_key TEXT;
ALTER TABLE actions ADD COLUMN amount_semantics TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE actions ADD COLUMN forced_bet_type TEXT;
ALTER TABLE actions ADD COLUMN is_live_post INTEGER;
ALTER TABLE hand_reviews ADD COLUMN is_stale INTEGER NOT NULL DEFAULT 0;
ALTER TABLE hand_reviews ADD COLUMN stale_reason TEXT NOT NULL DEFAULT '';
ALTER TABLE coaching_reviews ADD COLUMN is_stale INTEGER NOT NULL DEFAULT 0;
ALTER TABLE coaching_reviews ADD COLUMN stale_reason TEXT NOT NULL DEFAULT '';
ALTER TABLE processing_jobs ADD COLUMN pid INTEGER;
ALTER TABLE processing_jobs ADD COLUMN heartbeat_at TEXT;

CREATE TABLE hand_settlements (
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

CREATE TABLE settlement_entries (
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

CREATE TABLE hand_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL,
    correction_type TEXT NOT NULL,
    before_state TEXT NOT NULL DEFAULT '{}',
    after_state TEXT NOT NULL DEFAULT '{}',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (hand_id) REFERENCES hands(id) ON DELETE CASCADE
);

CREATE TABLE solver_runs (
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

CREATE UNIQUE INDEX idx_actions_hand_street_order
    ON actions(hand_id, street, action_index);
"""

# hand_number -> (source_type, review_status)
#
# The manual rows deliberately carry DIFFERENT review statuses. With only the
# 'reviewed' one, "manual hands keep their review_status untouched" was pinned by
# a tautology: a migration that promoted every manual hand to 'reviewed' wrote the
# same value the fixture already held, so the assertion could not tell the two
# apart and the whole suite stayed green under it. An upgrade must never grant a
# promotion on no evidence.
_LEGACY_HANDS: dict[int, tuple[str, str]] = {
    1: ("manual", "reviewed"),
    2: ("cv_import", "reviewed"),
    3: ("corrected_cv", "unreviewed"),
    4: ("third_party_tool", "reviewed"),
    5: ("manual", "unreviewed"),
    6: ("manual", "needs_correction"),
}


def _write_legacy_database(path: Path, *, stored_version: int | None) -> None:
    """Create a real old-shaped database and populate it with old-shaped rows.

    ``stored_version`` of None means a pre-versioning database: no
    schema_metadata row at all, so init_db also runs the legacy backfill.
    """
    is_v11_shaped = stored_version is not None and stored_version >= 11
    connection = sqlite3.connect(path)
    try:
        connection.executescript(_V5_SCHEMA)
        if is_v11_shaped:
            connection.executescript(_V7_TO_V11_ADDITIONS)
        connection.execute(
            "CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        if stored_version is not None:
            connection.execute(
                "INSERT INTO schema_metadata (key, value) VALUES ('schema_version', ?)",
                (str(stored_version),),
            )
        connection.execute(
            "INSERT INTO sessions (id, name, date_played, created_at) "
            "VALUES (1, 'Legacy session', '2026-01-01', ?)",
            (NOW,),
        )
        for hand_number, (source_type, review_status) in _LEGACY_HANDS.items():
            connection.execute(
                """
                INSERT INTO hands (
                    id, session_id, hand_number, hero_position, hero_cards,
                    board_cards, review_status, source_type, created_at
                )
                VALUES (?, 1, ?, 'BTN', 'Ah Qs', 'Qd 7s 2c', ?, ?, ?)
                """,
                (hand_number, hand_number, review_status, source_type, NOW),
            )
        # Distinct names so the v7 player_key backfill has an unambiguous match.
        for player_id, (name, position, is_hero) in enumerate(
            (("Hero", "BTN", 1), ("Villain", "BB", 0)), start=1
        ):
            connection.execute(
                "INSERT INTO hand_players (id, hand_id, player_name, position, "
                "starting_stack, is_hero) VALUES (?, 2, ?, ?, 100, ?)",
                (player_id, name, position, is_hero),
            )
            if is_v11_shaped:
                connection.execute(
                    "UPDATE hand_players SET player_key = ?, seat_index = ? WHERE id = ?",
                    (name.lower(), player_id - 1, player_id),
                )
        # A pre-v8 database can hold two river actions sharing action_index 9;
        # a v11-shaped one cannot, because v8 already added the unique index.
        for action_id, name in ((1, "Hero"), (2, "Villain")):
            connection.execute(
                "INSERT INTO actions (id, hand_id, street, action_index, player_name, "
                "position, action_type, amount) VALUES (?, 2, 'river', ?, ?, '', 'check', NULL)",
                (action_id, action_id if is_v11_shaped else 9, name),
            )
            if is_v11_shaped:
                connection.execute(
                    "UPDATE actions SET player_key = ?, amount_semantics = 'incremental' "
                    "WHERE id = ?",
                    (name.lower(), action_id),
                )
        connection.execute(
            "INSERT INTO hand_reviews (id, hand_id, hand_summary, theory_coach, "
            "exploit_coach, study_lesson, created_at) "
            "VALUES (1, 2, 'summary', 'theory', 'exploit', 'lesson', ?)",
            (NOW,),
        )
        connection.execute(
            "INSERT INTO coaching_reviews (id, provider_name, model_name, raw_prompt, "
            "raw_response, review_type, hand_id, created_at) "
            "VALUES (1, 'legacy', 'fixture', 'prompt', 'response', 'hand', 2, ?)",
            (NOW,),
        )
        connection.execute(
            "INSERT INTO videos (id, session_id, original_filename, stored_path, "
            "file_size_bytes, uploaded_at) VALUES (1, 1, 'session.mp4', "
            "'videos/session.mp4', 1024, ?)",
            (NOW,),
        )
        connection.execute(
            "INSERT INTO processing_jobs (id, job_type, status, video_id, created_at) "
            "VALUES (1, 'frame_extraction', 'completed', 1, ?)",
            (NOW,),
        )
        connection.commit()
    finally:
        connection.close()


def _open(path: Path) -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def _columns(db: PokerDatabase, table: str) -> list[str]:
    return [row["name"] for row in db._execute(f"PRAGMA table_info({table})").fetchall()]


def _hands_signature(db: PokerDatabase) -> list[tuple]:
    return [
        (row["name"], row["type"], row["notnull"], row["dflt_value"])
        for row in db._execute("PRAGMA table_info(hands)").fetchall()
    ]


# --------------------------------------------------------------------------
# 1. Fresh schema
# --------------------------------------------------------------------------


def test_fresh_schema_declares_completion_columns_last_with_conservative_defaults() -> None:
    db = PokerDatabase(":memory:")
    db.init_db()

    signature = _hands_signature(db)

    assert signature[-2] == ("completion_status", "TEXT", 1, "'not_applicable'")
    assert signature[-1] == ("completion_evidence", "TEXT", 1, "'{}'")
    db.close()


def test_fresh_schema_applies_the_column_defaults_to_a_raw_insert() -> None:
    """A row written outside the model still lands on the conservative defaults."""
    db = PokerDatabase(":memory:")
    db.init_db()
    session = db.create_session(Session(name="Raw insert"))
    db._execute(
        "INSERT INTO hands (session_id, hand_number, created_at) VALUES (?, 1, ?)",
        (session.id, NOW),
    )
    db._commit()

    row = db._execute(
        "SELECT completion_status, completion_evidence FROM hands"
    ).fetchone()

    assert row["completion_status"] == "not_applicable"
    assert row["completion_evidence"] == "{}"
    db.close()


def test_hands_table_carries_no_check_constraint_on_completion() -> None:
    """Enforcement lives at the pydantic and update_hand_status boundaries."""
    db = PokerDatabase(":memory:")
    db.init_db()

    ddl = db._execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'hands'"
    ).fetchone()["sql"]

    assert "CHECK" not in ddl.upper()
    db.close()


# --------------------------------------------------------------------------
# 2. Historical migration paths from real old-shaped databases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "stored_version"),
    [("pre-versioning", None), ("v5", 5), ("v11", 11)],
)
def test_legacy_database_migrates_to_schema_13(
    tmp_path: Path, label: str, stored_version: int | None
) -> None:
    path = tmp_path / f"{label}.sqlite3"
    _write_legacy_database(path, stored_version=stored_version)

    migrated = _open(path)

    assert migrated.schema_version() == SCHEMA_VERSION == 13
    assert {"completion_status", "completion_evidence"} <= set(_columns(migrated, "hands"))
    rows = {
        row["hand_number"]: row
        for row in migrated._execute(
            "SELECT hand_number, source_type, completion_status, completion_evidence, "
            "review_status FROM hands"
        ).fetchall()
    }
    assert len(rows) == len(_LEGACY_HANDS)
    for hand_number, (source_type, review_status) in _LEGACY_HANDS.items():
        row = rows[hand_number]
        assert row["source_type"] == source_type
        # No evidence is ever fabricated for a historical hand.
        assert row["completion_evidence"] == "{}"
        if source_type == "manual":
            assert row["completion_status"] == "not_applicable"
            assert row["review_status"] == review_status
        else:
            assert row["completion_status"] == "uncertain"
            assert row["review_status"] == "needs_correction"
    migrated.close()


@pytest.mark.parametrize("stored_version", [None, 5, 11])
def test_legacy_migration_produces_the_fresh_hands_schema(
    tmp_path: Path, stored_version: int | None
) -> None:
    fresh = PokerDatabase(":memory:")
    fresh.init_db()
    expected = _hands_signature(fresh)
    fresh.close()

    path = tmp_path / f"shape-{stored_version}.sqlite3"
    _write_legacy_database(path, stored_version=stored_version)
    migrated = _open(path)

    assert _hands_signature(migrated) == expected
    migrated.close()


@pytest.mark.parametrize("stored_version", [None, 5])
def test_pre_v7_database_gains_the_whole_chain_not_just_completion(
    tmp_path: Path, stored_version: int | None
) -> None:
    """Proof the old shape is real: v7-v12 artifacts are absent before the open."""
    path = tmp_path / f"chain-{stored_version}.sqlite3"
    _write_legacy_database(path, stored_version=stored_version)
    before = sqlite3.connect(path)
    before_tables = {
        row[0] for row in before.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    before.close()
    assert "hand_settlements" not in before_tables
    assert "hand_issues" not in before_tables
    assert "solver_runs" not in before_tables

    migrated = _open(path)

    # v7: the new identity columns exist and the backfill linked the rows.
    assert "player_key" in _columns(migrated, "hand_players")
    assert "amount_semantics" in _columns(migrated, "actions")
    player_keys = [
        row["player_key"]
        for row in migrated._execute(
            "SELECT player_key FROM hand_players ORDER BY id"
        ).fetchall()
    ]
    assert all(key for key in player_keys)
    action_keys = [
        row["player_key"]
        for row in migrated._execute(
            "SELECT player_key FROM actions ORDER BY id"
        ).fetchall()
    ]
    assert action_keys == player_keys
    # v8: the duplicated river order is repaired and the uniqueness index holds.
    assert [
        row["action_index"]
        for row in migrated._execute(
            "SELECT action_index FROM actions ORDER BY id"
        ).fetchall()
    ] == [1, 2]
    with pytest.raises(sqlite3.IntegrityError):
        migrated._execute(
            "INSERT INTO actions (hand_id, street, action_index, player_name, "
            "action_type, amount_semantics) VALUES (2, 'river', 2, 'Dup', 'check', 'unknown')"
        )
    # v10-v12: staleness columns and the newer tables now exist.
    assert "is_stale" in _columns(migrated, "coaching_reviews")
    assert "is_stale" in _columns(migrated, "hand_reviews")
    for table in ("hand_corrections", "solver_runs", "hand_issues", "hand_settlements"):
        migrated._execute(f"SELECT COUNT(*) FROM {table}")
    migrated.close()


def test_pre_versioning_database_is_backfilled_and_classified(tmp_path: Path) -> None:
    """A database with no schema_metadata row still reaches 13 with real data."""
    path = tmp_path / "no-metadata.sqlite3"
    _write_legacy_database(path, stored_version=None)
    unopened = PokerDatabase(path)
    assert unopened.schema_version() == 0
    unopened.close()

    migrated = _open(path)

    assert migrated.schema_version() == 13
    # Hand 4 carries an unrecognized source_type, which the SourceType literal has
    # always refused, so the rows are read directly rather than through the model.
    rows = migrated._execute(
        "SELECT hand_number, completion_status, completion_evidence FROM hands ORDER BY id"
    ).fetchall()
    assert [row["hand_number"] for row in rows] == sorted(_LEGACY_HANDS)
    assert {row["completion_evidence"] for row in rows} == {"{}"}
    assert [row["completion_status"] for row in rows] == [
        "not_applicable" if source == "manual" else "uncertain"
        for _, (source, _status) in sorted(_LEGACY_HANDS.items())
    ]
    migrated.close()


def test_legacy_migration_retains_every_history_row(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    _write_legacy_database(path, stored_version=5)

    migrated = _open(path)

    assert len(migrated.fetch_reviews_by_hand(2)) == 1
    assert len(migrated.fetch_coaching_reviews_by_hand(2)) == 1
    assert len(migrated.fetch_videos()) == 1
    assert migrated.fetch_actions_by_hand(2)
    assert len(migrated.fetch_players_by_hand(2)) == 2
    migrated.close()


# --------------------------------------------------------------------------
# 3. Conservative CV migration and manual compatibility, through the real store
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hand_id", [2, 3, 4])
def test_migrated_reconstructed_hand_cannot_be_silently_promoted(
    tmp_path: Path, hand_id: int
) -> None:
    path = tmp_path / f"promote-{hand_id}.sqlite3"
    _write_legacy_database(path, stored_version=11)
    migrated = _open(path)

    with pytest.raises(ValueError, match="cannot be marked reviewed"):
        migrated.update_hand_status(hand_id, "reviewed")

    # Read the row directly: hand 4 holds an unrecognized source_type, which the
    # SourceType literal has always refused, independently of completion state.
    row = migrated._execute(
        "SELECT review_status FROM hands WHERE id = ?", (hand_id,)
    ).fetchone()
    assert row["review_status"] == "needs_correction"
    migrated.close()


def test_migrated_reconstructed_hand_blocks_readiness_conservatively(
    tmp_path: Path,
) -> None:
    path = tmp_path / "readiness.sqlite3"
    _write_legacy_database(path, stored_version=11)
    migrated = _open(path)

    readiness = evaluate_study_readiness(migrated.fetch_hand(2), accounting=None)

    assert readiness.is_ready is False
    assert set(readiness.codes()) == {
        "COMPLETION_NOT_COMPLETE",
        "COMPLETION_EVIDENCE_MISSING",
        "UNSUPPORTED_TABLE_LAYOUT",
        "ACCOUNTING_NOT_AUTHORITATIVE",
        "USER_CONFIRMATION_MISSING",
    }
    migrated.close()


def test_migrated_manual_hand_completes_the_full_review_workflow(tmp_path: Path) -> None:
    """End to end on a migrated manual hand: correct, settle, reconcile, review."""
    path = tmp_path / "manual-workflow.sqlite3"
    _write_legacy_database(path, stored_version=5)
    db = _open(path)
    manual = db.fetch_hand(1)
    assert manual.completion_status == "not_applicable"
    assert manual.review_status == "reviewed"

    db.update_hand_facts(
        manual.model_copy(update={"board_cards": "Qd 7s 2c 9h", "pot_size": 20.0}),
        correction_notes="Frame shows the turn.",
    )
    corrected = db.fetch_hand(manual.id)
    # A manual hand is never demoted out of not_applicable by a correction.
    assert corrected.completion_status == "not_applicable"
    assert corrected.review_status == "needs_correction"

    hero = db.create_hand_player(
        HandPlayer(
            hand_id=manual.id,
            player_key="hero",
            player_name="Hero",
            position="BTN",
            starting_stack=100,
            is_hero=True,
        )
    )
    villain = db.create_hand_player(
        HandPlayer(
            hand_id=manual.id,
            player_key="villain",
            player_name="Villain",
            position="BB",
            starting_stack=100,
        )
    )
    for actor, action_type in ((hero, "bet"), (villain, "call")):
        db.create_action(
            Action(
                hand_id=manual.id,
                player_key=actor.player_key,
                player_name=actor.player_name,
                position=actor.position,
                street="river",
                action_type=action_type,
                amount=10,
                amount_semantics="incremental",
            )
        )
    db.upsert_hand_settlement(HandSettlement(hand_id=manual.id, status="settled"))
    db.replace_settlement_entries(
        manual.id,
        [
            SettlementEntry(
                hand_id=manual.id,
                entry_type="award",
                pot_index=0,
                player_key=hero.player_key,
                player_name=hero.player_name,
                amount=20,
                entry_order=1,
            )
        ],
    )
    accounting = persist_reconciliation(db, manual.id)
    assert accounting.is_authoritative is True

    readiness = evaluate_study_readiness(
        db.fetch_hand(manual.id), accounting=accounting, user_confirmed=False
    )
    assert readiness.is_ready is True

    db.update_hand_status(manual.id, "reviewed")
    assert db.fetch_hand(manual.id).review_status == "reviewed"
    db.close()


# --------------------------------------------------------------------------
# 4. Rollback from a legacy database
# --------------------------------------------------------------------------


def test_failed_v13_migration_rolls_back_the_whole_legacy_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure at the last step must undo v7-v12 too, not just v13."""
    path = tmp_path / "legacy-rollback.sqlite3"
    _write_legacy_database(path, stored_version=5)

    def exploding_migration(db: PokerDatabase) -> None:
        db._ensure_column(
            "hands", "completion_status", "TEXT NOT NULL DEFAULT 'not_applicable'"
        )
        raise RuntimeError("v13 exploded")

    monkeypatch.setitem(db_module._MIGRATIONS, 13, exploding_migration)
    failed = PokerDatabase(path)
    with pytest.raises(RuntimeError, match="v13 exploded"):
        failed.init_db()
    failed.close()

    checked = PokerDatabase(path)
    assert checked.schema_version() == 5
    assert "completion_status" not in _columns(checked, "hands")
    assert "completion_evidence" not in _columns(checked, "hands")
    # The earlier steps in the same transaction are gone as well.
    assert "player_key" not in _columns(checked, "hand_players")
    assert "amount_semantics" not in _columns(checked, "actions")
    assert "is_stale" not in _columns(checked, "coaching_reviews")
    assert [
        row["action_index"]
        for row in checked._execute("SELECT action_index FROM actions ORDER BY id").fetchall()
    ] == [9, 9]
    rows = {
        row["hand_number"]: (row["source_type"], row["review_status"])
        for row in checked._execute(
            "SELECT hand_number, source_type, review_status FROM hands"
        ).fetchall()
    }
    assert rows == _LEGACY_HANDS
    checked.close()

    monkeypatch.undo()
    repaired = _open(path)
    assert repaired.schema_version() == 13
    assert repaired.fetch_hand(1).completion_status == "not_applicable"
    assert repaired.fetch_hand(2).completion_status == "uncertain"
    repaired.close()


# --------------------------------------------------------------------------
# 5. Forward compatibility
# --------------------------------------------------------------------------


def test_a_schema_14_file_is_refused_and_left_untouched(
    tmp_path: Path, isolated_backup_dir: Path
) -> None:
    """An older build must refuse a newer database without altering or backing it up."""
    path = tmp_path / "future.sqlite3"
    seeded = _open(path)
    seeded.create_session(Session(name="Future"))
    seeded._execute("UPDATE schema_metadata SET value = '14' WHERE key = 'schema_version'")
    seeded._commit()
    seeded.close()
    before = path.read_bytes()

    newer = PokerDatabase(path)
    with pytest.raises(RuntimeError, match="newer than this app"):
        newer.init_db()
    newer.close()

    assert path.read_bytes() == before
    assert list(isolated_backup_dir.glob("poker_tracker_*.sqlite3")) == []
    reopened = PokerDatabase(path)
    assert reopened.schema_version() == 14
    reopened.close()


def test_a_v13_database_written_now_is_refused_by_a_build_that_knows_12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard is version-relative, so today's file is tomorrow's refusal."""
    path = tmp_path / "current.sqlite3"
    current = _open(path)
    current.close()

    monkeypatch.setattr(db_module, "SCHEMA_VERSION", 12)
    older_build = PokerDatabase(path)
    with pytest.raises(RuntimeError, match="newer than this app"):
        older_build.init_db()
    older_build.close()


def test_a_reconstructed_hand_created_after_migration_is_unproven(tmp_path: Path) -> None:
    """The migration classifies history; the model classifies everything new."""
    path = tmp_path / "post-migration.sqlite3"
    _write_legacy_database(path, stored_version=11)
    db = _open(path)

    fresh_cv = db.create_hand(
        Hand(session_id=1, hand_number=99, source_type="cv_import")
    )

    assert fresh_cv.completion_status == "uncertain"
    assert db.fetch_hand(fresh_cv.id).completion_status == "uncertain"
    db.close()
