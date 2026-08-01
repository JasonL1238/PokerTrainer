from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest

from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence import db as db_module
from poker_tracker.persistence.backup import backups_dir_for
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import import_session
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    ExtractedFrame,
    Hand,
    HandCorrection,
    HandIssue,
    HandPlayer,
    HandReview,
    HandSettlement,
    ProcessingJob,
    ReconstructionFrameReview,
    Session,
    SettlementEntry,
    SolverRun,
    VideoRecord,
)
from poker_tracker.services.regression_promotion import (
    fetch_regression_case,
    promote_issue_to_regression,
    regressions_for_issue,
)


def _make_db(path: str | Path = ":memory:") -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def _make_db_without_init(path: str | Path) -> PokerDatabase:
    """Open an existing file without triggering migrations."""
    return PokerDatabase(path)


def _review(hand_id: int) -> HandReview:
    return HandReview(
        hand_id=hand_id,
        hand_summary="summary",
        theory_coach="theory",
        exploit_coach="exploit",
        study_lesson="lesson",
    )


def _coaching_response(
    *,
    review_type: str,
    hand_id: int | None = None,
    session_id: int | None = None,
) -> CoachingResponse:
    return CoachingResponse(
        provider_name="test",
        model_name="deterministic-fixture",
        raw_prompt="prompt",
        raw_response="response",
        review_type=review_type,
        hand_id=hand_id,
        session_id=session_id,
    )


def test_update_action_rejects_cross_hand_identity_and_invalidates_true_hand() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Action identity"))
    first_hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    second_hand = db.create_hand(Hand(session_id=session.id, hand_number=2))
    action = db.create_action(
        Action(
            hand_id=first_hand.id,
            street="river",
            player_name="Hero",
            action_type="check",
        )
    )
    for hand in (first_hand, second_hand):
        db.upsert_hand_settlement(
            HandSettlement(
                hand_id=hand.id,
                status="reconciled",
                is_balanced=True,
            )
        )
        db.create_hand_review(_review(hand.id))
        # Promoted last: an action, player, or settlement write on a promoted
        # hand now returns it to needs_correction.
        db.update_hand_status(hand.id, "reviewed")

    with pytest.raises(ValueError, match="does not belong"):
        db.update_action(
            action.model_copy(
                update={
                    "hand_id": second_hand.id,
                    "notes": "malicious cross-hand update",
                }
            )
        )

    assert db.fetch_actions_by_hand(first_hand.id)[0].notes == ""
    assert db.fetch_actions_by_hand(second_hand.id) == []
    assert db.fetch_hand_settlement(first_hand.id).status == "reconciled"
    assert db.fetch_hand_settlement(second_hand.id).status == "reconciled"
    assert len(db.fetch_reviews_by_hand(first_hand.id)) == 1
    assert len(db.fetch_reviews_by_hand(second_hand.id)) == 1

    updated = db.update_action(
        action.model_copy(update={"notes": "verified source correction"})
    )
    assert updated.hand_id == first_hand.id
    assert db.fetch_actions_by_hand(first_hand.id)[0].notes == "verified source correction"
    assert db.fetch_hand_settlement(first_hand.id).status == "needs_correction"
    assert db.fetch_hand_settlement(second_hand.id).status == "reconciled"
    first_reviews = db.fetch_reviews_by_hand(first_hand.id)
    assert len(first_reviews) == 1
    assert first_reviews[0].is_stale is True
    assert len(db.fetch_reviews_by_hand(second_hand.id)) == 1
    corrections = db.fetch_hand_corrections(first_hand.id)
    assert len(corrections) == 1
    assert corrections[0].correction_type == "action_update"
    assert db.fetch_hand(first_hand.id).review_status == "needs_correction"
    assert db.fetch_hand(second_hand.id).review_status == "reviewed"
    db.close()


def test_player_and_summary_evidence_edits_retain_and_flag_stale_reviews() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Review invalidation"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, review_status="reviewed")
    )
    player = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            player_name="Hero",
            starting_stack=100,
            is_hero=True,
        )
    )

    def save_reviews() -> None:
        db.create_hand_review(_review(hand.id))
        db.create_coaching_response(
            _coaching_response(review_type="hand", hand_id=hand.id)
        )
        db.create_coaching_response(
            _coaching_response(review_type="session", session_id=session.id)
        )
        db.update_hand_status(hand.id, "reviewed")

    save_reviews()
    db.update_hand_player(player.model_copy(update={"starting_stack": 120}))

    assert all(review.is_stale for review in db.fetch_reviews_by_hand(hand.id))
    assert all(review.is_stale for review in db.fetch_coaching_reviews_by_hand(hand.id))
    assert all(review.is_stale for review in db.fetch_coaching_reviews_by_session(session.id))
    assert db.fetch_hand(hand.id).review_status == "needs_correction"
    assert db.fetch_hand_corrections(hand.id)[0].correction_type == "player_update"

    save_reviews()
    db.update_hand_accounting_evidence(
        hand.id,
        pot_size=24,
        hero_bb_won=4,
    )

    assert len(db.fetch_reviews_by_hand(hand.id)) == 2
    assert all(review.is_stale for review in db.fetch_reviews_by_hand(hand.id))
    assert len(db.fetch_coaching_reviews_by_hand(hand.id)) == 2
    assert all(review.is_stale for review in db.fetch_coaching_reviews_by_hand(hand.id))
    assert len(db.fetch_coaching_reviews_by_session(session.id)) == 2
    assert all(review.is_stale for review in db.fetch_coaching_reviews_by_session(session.id))
    assert db.fetch_hand(hand.id).review_status == "needs_correction"
    assert db.fetch_hand_corrections(hand.id)[0].correction_type == "hand_facts"
    db.close()


def test_action_index_is_unique_per_hand_and_street_on_create_and_update() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Action order"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    first = db.create_action(
        Action(
            hand_id=hand.id,
            street="flop",
            action_index=1,
            player_name="A",
            action_type="check",
        )
    )

    with pytest.raises(ValueError, match="unique"):
        db.create_action(
            Action(
                hand_id=hand.id,
                street="flop",
                action_index=1,
                player_name="B",
                action_type="check",
            )
        )

    second = db.create_action(
        Action(
            hand_id=hand.id,
            street="flop",
            player_name="B",
            action_type="check",
        )
    )
    assert (first.action_index, second.action_index) == (1, 2)

    with pytest.raises(ValueError, match="unique"):
        db.update_action(second.model_copy(update={"action_index": 1}))

    assert [action.action_index for action in db.fetch_actions_by_hand(hand.id)] == [
        1,
        2,
    ]
    db.close()


def test_import_normalizes_duplicate_legacy_action_indexes_in_payload_order() -> None:
    payload = {
        "export_version": 2,
        "session": Session(name="Duplicate import").model_dump(mode="json"),
        "hands": [
            {
                "hand": Hand(session_id=999, hand_number=1).model_dump(mode="json"),
                "players": [],
                "actions": [
                    Action(
                        hand_id=999,
                        street="turn",
                        action_index=4,
                        player_name="First",
                        action_type="check",
                    ).model_dump(mode="json"),
                    Action(
                        hand_id=999,
                        street="turn",
                        action_index=4,
                        player_name="Second",
                        action_type="check",
                    ).model_dump(mode="json"),
                ],
                "settlement": None,
                "settlement_entries": [],
                "reviews": [],
            }
        ],
    }
    db = _make_db()

    imported = import_session(db, payload)
    hand = db.fetch_hands_by_session(imported.id)[0]
    actions = db.fetch_actions_by_hand(hand.id)

    assert [action.player_name for action in actions] == ["First", "Second"]
    assert [action.action_index for action in actions] == [1, 2]
    db.close()


def test_v8_migration_repairs_legacy_duplicate_action_order(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-actions.sqlite3"
    db = _make_db(path)
    session = db.create_session(Session(name="Legacy duplicate"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db._execute("DROP INDEX idx_actions_hand_street_order")
    db._execute(
        """
        INSERT INTO actions (
            hand_id, player_key, street, action_index, player_name, position,
            action_type, amount, amount_semantics, forced_bet_type, is_live_post,
            pot_before, stack_before, notes
        )
        VALUES (?, NULL, 'river', 9, 'First', '', 'check', NULL, 'unknown',
                NULL, NULL, NULL, NULL, '')
        """,
        (hand.id,),
    )
    db._execute(
        """
        INSERT INTO actions (
            hand_id, player_key, street, action_index, player_name, position,
            action_type, amount, amount_semantics, forced_bet_type, is_live_post,
            pot_before, stack_before, notes
        )
        VALUES (?, NULL, 'river', 9, 'Second', '', 'check', NULL, 'unknown',
                NULL, NULL, NULL, NULL, '')
        """,
        (hand.id,),
    )
    # A real schema-7 file predates the v13 columns; see _downgrade_to_v12. A
    # database whose schema is ahead of its own stamp is now refused, because that
    # is what a live v13 database with a lost stamp looks like.
    db._execute("ALTER TABLE hands DROP COLUMN completion_status")
    db._execute("ALTER TABLE hands DROP COLUMN completion_evidence")
    db._execute(
        "UPDATE schema_metadata SET value = '7' WHERE key = 'schema_version'"
    )
    db._commit()
    db.close()

    migrated = _make_db(path)
    actions = migrated.fetch_actions_by_hand(hand.id)
    assert migrated.schema_version() == SCHEMA_VERSION
    assert [action.player_name for action in actions] == ["First", "Second"]
    assert [action.action_index for action in actions] == [1, 2]
    with pytest.raises(sqlite3.IntegrityError):
        migrated._execute(
            """
            INSERT INTO actions (
                hand_id, player_key, street, action_index, player_name, position,
                action_type, amount, amount_semantics, forced_bet_type, is_live_post,
                pot_before, stack_before, notes
            )
            VALUES (?, NULL, 'river', 2, 'Duplicate', '', 'check', NULL, 'unknown',
                    NULL, NULL, NULL, NULL, '')
            """,
            (hand.id,),
        )
    migrated.close()


def _downgrade_to_v12(db: PokerDatabase) -> None:
    """Strip the v13 additions so the next open replays the real migration."""
    db._execute("ALTER TABLE hands DROP COLUMN completion_status")
    db._execute("ALTER TABLE hands DROP COLUMN completion_evidence")
    db._execute("UPDATE schema_metadata SET value = '12' WHERE key = 'schema_version'")
    db._commit()


def _seed_v12_database(path: Path) -> dict[str, int]:
    """Build a populated database and rewind it to schema 12."""
    db = _make_db(path)
    session = db.create_session(Session(name="Legacy completion"))
    manual = db.create_hand(
        Hand(session_id=session.id, hand_number=1, review_status="reviewed")
    )
    cv = db.create_hand(
        Hand(session_id=session.id, hand_number=2, source_type="cv_import")
    )
    corrected = db.create_hand(
        Hand(session_id=session.id, hand_number=3, source_type="corrected_cv")
    )
    unknown = db.create_hand(Hand(session_id=session.id, hand_number=4))
    db.create_hand_player(
        HandPlayer(hand_id=cv.id, player_key="hero", player_name="Hero", is_hero=True)
    )
    db.upsert_hand_settlement(
        HandSettlement(hand_id=cv.id, status="reconciled", is_balanced=True)
    )
    db.create_hand_review(_review(cv.id))
    db.create_coaching_response(_coaching_response(review_type="hand", hand_id=cv.id))
    db.create_hand_issue(
        HandIssue(
            hand_id=corrected.id,
            issue_types=["hand_boundary"],
            description="Boundary is unclear.",
        )
    )
    db.create_solver_run(
        SolverRun(hand_id=cv.id, input_hash="legacy-hash", status="completed")
    )
    # Legacy states written directly: the v13 guard would refuse them today.
    for hand_id in (cv.id, corrected.id, unknown.id):
        db._execute(
            "UPDATE hands SET review_status = 'reviewed' WHERE id = ?", (hand_id,)
        )
    db._execute(
        "UPDATE hands SET source_type = 'third_party_tool' WHERE id = ?", (unknown.id,)
    )
    ids = {
        "session": session.id,
        "manual": manual.id,
        "cv": cv.id,
        "corrected": corrected.id,
        "unknown": unknown.id,
    }
    _downgrade_to_v12(db)
    db.close()
    return ids


def _table_snapshot(db: PokerDatabase, table: str) -> list[tuple]:
    return [
        tuple(row)
        for row in db._execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
    ]


_RETAINED_TABLES = (
    "hand_corrections",
    "hand_issues",
    "hand_reviews",
    "coaching_reviews",
    "hand_settlements",
    "settlement_entries",
    "solver_runs",
    "videos",
)


def test_v13_migration_from_v12_marks_cv_hands_uncertain_and_needs_correction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v12.sqlite3"
    ids = _seed_v12_database(path)

    migrated = _make_db(path)

    assert migrated.schema_version() == SCHEMA_VERSION
    for key in ("cv", "corrected"):
        hand = migrated.fetch_hand(ids[key])
        assert hand.completion_status == "uncertain"
        assert hand.review_status == "needs_correction"
        assert hand.completion_evidence == {}
    migrated.close()


def test_v13_migration_leaves_manual_hands_not_applicable_and_keeps_review_status(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manual.sqlite3"
    ids = _seed_v12_database(path)

    migrated = _make_db(path)

    manual = migrated.fetch_hand(ids["manual"])
    assert manual.completion_status == "not_applicable"
    assert manual.review_status == "reviewed"
    migrated.close()


def test_v13_migration_marks_unknown_source_type_uncertain(tmp_path: Path) -> None:
    path = tmp_path / "unknown-source.sqlite3"
    ids = _seed_v12_database(path)

    migrated = _make_db(path)

    row = migrated._execute(
        "SELECT source_type, completion_status, review_status FROM hands WHERE id = ?",
        (ids["unknown"],),
    ).fetchone()
    assert row["source_type"] == "third_party_tool"
    assert row["completion_status"] == "uncertain"
    assert row["review_status"] == "needs_correction"
    migrated.close()


@pytest.mark.parametrize("stored_version", ["11", "5"])
def test_v13_migration_applies_the_full_chain(tmp_path: Path, stored_version: str) -> None:
    path = tmp_path / f"chain-{stored_version}.sqlite3"
    ids = _seed_v12_database(path)
    older = _make_db_without_init(path)
    older._execute(
        "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
        (stored_version,),
    )
    older._commit()
    older.close()

    migrated = _make_db(path)

    assert migrated.schema_version() == SCHEMA_VERSION
    assert migrated.fetch_hand(ids["cv"]).completion_status == "uncertain"
    assert migrated.fetch_hand(ids["manual"]).completion_status == "not_applicable"
    migrated.close()


def test_v13_migration_from_pre_versioning_database(tmp_path: Path) -> None:
    path = tmp_path / "pre-versioning.sqlite3"
    ids = _seed_v12_database(path)
    older = _make_db_without_init(path)
    older._execute("DELETE FROM schema_metadata WHERE key = 'schema_version'")
    older._commit()
    assert older.schema_version() == 0
    older.close()

    migrated = _make_db(path)

    assert migrated.schema_version() == SCHEMA_VERSION
    assert migrated.fetch_hand(ids["cv"]).completion_status == "uncertain"
    assert migrated.fetch_hand(ids["manual"]).completion_status == "not_applicable"
    migrated.close()


def test_v13_migration_preserves_correction_issue_coaching_settlement_and_solver_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history.sqlite3"
    _seed_v12_database(path)
    before_db = _make_db_without_init(path)
    before = {table: _table_snapshot(before_db, table) for table in _RETAINED_TABLES}
    before_db.close()

    migrated = _make_db(path)

    after = {table: _table_snapshot(migrated, table) for table in _RETAINED_TABLES}
    assert after == before
    migrated.close()


def test_v13_migration_is_idempotent_on_rerun(tmp_path: Path) -> None:
    path = tmp_path / "idempotent.sqlite3"
    ids = _seed_v12_database(path)
    migrated = _make_db(path)
    migrated._execute(
        "UPDATE hands SET completion_status = 'complete' WHERE id = ?", (ids["cv"],)
    )
    migrated._commit()
    migrated.close()

    reopened = _make_db(path)

    # A second open is at the current version, so the backfill must not run again.
    assert reopened.fetch_hand(ids["cv"]).completion_status == "complete"
    assert reopened.schema_version() == SCHEMA_VERSION
    reopened.close()


def test_failed_migration_rolls_back_without_partial_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollback.sqlite3"
    _seed_v12_database(path)
    before_db = _make_db_without_init(path)
    before = {table: _table_snapshot(before_db, table) for table in _RETAINED_TABLES}
    before["hands"] = _table_snapshot(before_db, "hands")
    before_db.close()

    def exploding_migration(db: PokerDatabase) -> None:
        db._ensure_column(
            "hands", "completion_status", "TEXT NOT NULL DEFAULT 'not_applicable'"
        )
        raise RuntimeError("migration exploded")

    monkeypatch.setitem(db_module._MIGRATIONS, 13, exploding_migration)
    failed = PokerDatabase(path)
    with pytest.raises(RuntimeError, match="migration exploded"):
        failed.init_db()
    failed.close()

    checked = _make_db_without_init(path)
    columns = {row["name"] for row in checked._execute("PRAGMA table_info(hands)").fetchall()}
    assert "completion_status" not in columns
    assert "completion_evidence" not in columns
    assert checked.schema_version() == 12
    after = {table: _table_snapshot(checked, table) for table in _RETAINED_TABLES}
    after["hands"] = _table_snapshot(checked, "hands")
    assert after == before
    checked.close()

    monkeypatch.undo()
    repaired = _make_db(path)
    assert repaired.schema_version() == SCHEMA_VERSION
    repaired.close()


def test_executescript_migrations_do_not_break_the_transaction_boundary() -> None:
    db = _make_db()

    with db.transaction(immediate=True):
        db_module._migrate_to_v11(db)
        assert db._connection.in_transaction is True
        db_module._migrate_to_v12(db)
        assert db._connection.in_transaction is True

    db.close()


def test_migration_backup_is_created_for_a_real_file_database(
    tmp_path: Path, isolated_backup_dir: Path
) -> None:
    path = tmp_path / "backed-up.sqlite3"
    _seed_v12_database(path)
    # A snapshot lives with the database it can roll back, so a database that is
    # not the live one is looked for beside itself rather than in the operator's
    # retained set -- which the fixture keeps empty either way.
    snapshot_dir = backups_dir_for(path)
    assert list(snapshot_dir.glob("*.sqlite3")) == []

    migrated = _make_db(path)

    snapshots = list(snapshot_dir.glob("*.sqlite3"))
    assert len(snapshots) == 1
    assert list(isolated_backup_dir.glob("*.sqlite3")) == []
    assert migrated.schema_version() == SCHEMA_VERSION
    migrated.close()


def test_migration_backup_is_skipped_for_a_fresh_file(
    tmp_path: Path, isolated_backup_dir: Path
) -> None:
    fresh = tmp_path / "fresh.sqlite3"
    db = _make_db(fresh)

    assert list(backups_dir_for(fresh).glob("*.sqlite3")) == []
    assert list(isolated_backup_dir.glob("*.sqlite3")) == []
    db.close()


def test_migration_backup_is_skipped_for_memory_database(
    isolated_backup_dir: Path,
) -> None:
    db = _make_db()

    assert list(isolated_backup_dir.glob("*.sqlite3")) == []
    db.close()


def test_migration_aborts_when_the_backup_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "blocked" / "unwritable.sqlite3"
    database.parent.mkdir()
    _seed_v12_database(database)
    # An ordinary file where the snapshot directory belongs: mkdir cannot pass,
    # so the migration meets a directory it genuinely cannot write.
    snapshot_dir = backups_dir_for(database)
    snapshot_dir.write_text("occupied")

    failed = PokerDatabase(database)
    # The raw sqlite3/OS error said "unable to open database file", which reads as
    # "your poker_tracker.db is broken". It names the backup directory now.
    with pytest.raises(RuntimeError) as caught:
        failed.init_db()
    failed.close()
    message = str(caught.value)
    assert "pre-migration backup" in message
    assert str(snapshot_dir) in message
    assert "was not applied" in message

    checked = _make_db_without_init(database)
    assert checked.schema_version() == 12
    columns = {row["name"] for row in checked._execute("PRAGMA table_info(hands)").fetchall()}
    assert "completion_status" not in columns
    checked.close()


def test_execute_script_keeps_comments_literals_and_the_transaction_intact() -> None:
    """The splitter replaces executescript(), so its edge cases are load-bearing.

    sqlite3.Connection.executescript() implicitly COMMITs, which would silently
    defeat migration rollback. _execute_script must behave identically for the
    DDL the migrations actually contain while leaving the transaction open.
    """
    db = _make_db()

    with db.transaction(immediate=True):
        db._execute_script(
            """
            -- a leading comment, as a future migration may well carry
            CREATE TABLE IF NOT EXISTS splitter_probe (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL
            );
            INSERT INTO splitter_probe (label) VALUES ('semi;colon');
            CREATE INDEX IF NOT EXISTS idx_splitter_probe ON splitter_probe(label);
            """
        )
        assert db._connection.in_transaction is True

    assert db._execute("SELECT label FROM splitter_probe").fetchone()["label"] == (
        "semi;colon"
    )
    db.close()


def test_execute_script_rolls_back_every_statement_on_failure() -> None:
    db = _make_db()

    with pytest.raises(sqlite3.OperationalError):
        with db.transaction(immediate=True):
            db._execute_script(
                """
                CREATE TABLE rollback_probe (id INTEGER PRIMARY KEY);
                CREATE TABLE rollback_probe (id INTEGER PRIMARY KEY);
                """
            )

    tables = {
        row["name"]
        for row in db._execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "rollback_probe" not in tables
    db.close()


def test_a_second_open_of_a_current_database_writes_no_further_backup(
    tmp_path: Path, isolated_backup_dir: Path
) -> None:
    """Snapshots are taken before migrating, not on every startup."""
    path = tmp_path / "repeat-open.sqlite3"
    _seed_v12_database(path)
    snapshot_dir = backups_dir_for(path)

    first = _make_db(path)
    first.close()
    assert len(list(snapshot_dir.glob("*.sqlite3"))) == 1

    second = _make_db(path)
    second.close()

    assert len(list(snapshot_dir.glob("*.sqlite3"))) == 1
    assert list(isolated_backup_dir.glob("*.sqlite3")) == []


def test_a_snapshot_is_self_contained_and_read_only_verifiable(tmp_path: Path) -> None:
    """A pre-migration snapshot is the operator's recovery gate, so it must open
    read-only from a directory that cannot be written -- a read-only or archival
    mount, or the isolated copy the restore drill makes."""
    live = tmp_path / "live.sqlite3"
    _seed_v12_database(live)
    backups = tmp_path / "backups"

    snapshot = backup_module.backup_database(live, backups)

    with sqlite3.connect(str(snapshot)) as verify:
        assert verify.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not Path(f"{snapshot}-shm").exists()
    assert not Path(f"{snapshot}-wal").exists()

    backups.chmod(0o500)
    try:
        readonly = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
        assert readonly.execute("SELECT count(*) FROM hands").fetchone()[0] >= 0
        readonly.close()
    finally:
        backups.chmod(0o700)


def test_rotation_removes_a_rotated_snapshot_sidecars(tmp_path: Path) -> None:
    """An orphaned -shm/-wal pair is the only trace a rotated-away backup leaves,
    which made a destroyed retained set hard to even detect."""
    live = tmp_path / "live.sqlite3"
    _seed_v12_database(live)
    backups = tmp_path / "backups"
    backups.mkdir()
    oldest = backups / "poker_tracker_20260101T000000000000Z.sqlite3"
    oldest.write_bytes(b"")
    for suffix in ("-shm", "-wal"):
        Path(f"{oldest}{suffix}").write_bytes(b"")
    os.utime(oldest, (0, 0))

    for _ in range(backup_module.BACKUP_KEEP_COUNT + 1):
        backup_module.backup_database(live, backups)

    assert not oldest.exists()
    assert not Path(f"{oldest}-shm").exists()
    assert not Path(f"{oldest}-wal").exists()


# --------------------------------------------------------------------------
# Connection settings the rest of the phase is built on
# --------------------------------------------------------------------------


def test_the_live_connection_settles_on_the_documented_pragmas(tmp_path: Path) -> None:
    """Read the settings back off the connection, not off the code that sets them.

    Every guarantee in this file rests on these four. The cascade tests below
    prove foreign_keys is on only while the DDL also declares the cascade, so a
    change that dropped ``PRAGMA foreign_keys = ON`` and the cascade together
    would have left the suite green with orphaned rows accumulating.
    """
    db = PokerDatabase(tmp_path / "pragmas.sqlite3", busy_timeout_ms=4321)
    db.init_db()

    def pragma(name: str) -> object:
        return db._execute(f"PRAGMA {name}").fetchone()[0]

    assert pragma("journal_mode") == "wal"
    assert pragma("foreign_keys") == 1
    assert pragma("busy_timeout") == 4321
    assert pragma("synchronous") == 1  # NORMAL
    db.close()


def test_the_default_busy_timeout_is_the_documented_constant(tmp_path: Path) -> None:
    db = PokerDatabase(tmp_path / "default-timeout.sqlite3")
    db.init_db()

    assert db._execute("PRAGMA busy_timeout").fetchone()[0] == db_module.BUSY_TIMEOUT_MS
    db.close()


def test_two_threads_writing_through_one_database_lose_no_row(tmp_path: Path) -> None:
    """Streamlit reruns the script on its own threads against one PokerDatabase.

    The single shared connection is guarded by an RLock, and nothing exercised
    two threads writing through it: sqlite3 with check_same_thread=False will
    happily interleave two statements on one connection and corrupt the cursor
    state or raise "recursive use of cursors not allowed".
    """
    db = PokerDatabase(tmp_path / "threads.sqlite3")
    db.init_db()
    session = db.create_session(Session(name="Two threads"))
    failures: list[BaseException] = []

    def write(first_hand_number: int) -> None:
        try:
            for offset in range(25):
                db.create_hand(
                    Hand(session_id=session.id, hand_number=first_hand_number + offset)
                )
        except BaseException as exc:  # noqa: BLE001 - re-raised in the assertion
            failures.append(exc)

    threads = [
        threading.Thread(target=write, args=(start,)) for start in (1, 1001)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(db.fetch_hands_by_session(session.id)) == 50
    db.close()


def test_a_worker_and_the_ui_writing_as_separate_connections_both_land(
    tmp_path: Path,
) -> None:
    """The CV worker and the app open the same file separately.

    ``test_phase1_adversarial_round7`` proves three processes can OPEN one file
    together; this is the other half, that they can both WRITE. Under WAL a
    second writer gets SQLITE_BUSY immediately, and only the busy timeout turns
    that into a wait rather than a lost write.
    """
    path = tmp_path / "two-writers.sqlite3"
    opener = PokerDatabase(path)
    opener.init_db()
    session = opener.create_session(Session(name="Two writers"))
    opener.close()
    failures: list[BaseException] = []

    def write(first_hand_number: int) -> None:
        writer = PokerDatabase(path, busy_timeout_ms=15000)
        try:
            for offset in range(25):
                writer.create_hand(
                    Hand(session_id=session.id, hand_number=first_hand_number + offset)
                )
        except BaseException as exc:  # noqa: BLE001 - re-raised in the assertion
            failures.append(exc)
        finally:
            writer.close()

    threads = [threading.Thread(target=write, args=(start,)) for start in (1, 1001)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    reader = PokerDatabase(path)
    assert len(reader.fetch_hands_by_session(session.id)) == 50
    reader.close()


def test_an_open_transaction_is_not_visible_to_another_connection_until_commit(
    tmp_path: Path,
) -> None:
    """The transaction boundary has to hold across connections, not just in-process."""
    path = tmp_path / "boundary.sqlite3"
    writer = PokerDatabase(path)
    writer.init_db()
    session = writer.create_session(Session(name="Boundary"))
    observer = PokerDatabase(path)

    with writer.transaction():
        writer.create_hand(Hand(session_id=session.id, hand_number=1))
        assert observer.fetch_hands_by_session(session.id) == []

    assert len(observer.fetch_hands_by_session(session.id)) == 1
    observer.close()
    writer.close()


# --------------------------------------------------------------------------
# Missing tables, columns, indexes and broken references
# --------------------------------------------------------------------------


def test_an_intact_database_reports_nothing_missing(tmp_path: Path) -> None:
    db = _make_db(tmp_path / "intact.sqlite3")

    report = db.schema_integrity()

    assert report.is_intact
    assert report.describe() == "nothing missing"
    db.close()


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        ("DROP TABLE regression_cases", "missing table(s): regression_cases"),
        (
            "ALTER TABLE hands DROP COLUMN study_inclusion",
            "missing column(s): hands.study_inclusion",
        ),
        (
            "DROP INDEX idx_actions_hand_street_order",
            "missing index(es): idx_actions_hand_street_order",
        ),
    ],
)
def test_the_integrity_report_names_what_is_missing(
    tmp_path: Path, damage: str, expected: str
) -> None:
    """An audit that says "incomplete" and stops there is not actionable.

    Each of these passed the schema contract before: it inspected table names and
    column names only, so a database missing the unique index that migration 8
    exists to create -- the one thing standing between the product and silently
    duplicated action order -- reported a clean bill of health.
    """
    db = _make_db(tmp_path / "damaged.sqlite3")
    db._execute(damage)
    db._commit()

    report = db.schema_integrity()

    assert not report.is_intact
    assert expected in report.describe()
    db.close()


def test_the_integrity_report_names_the_rows_whose_parent_is_gone(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path / "orphans.sqlite3")
    session = db.create_session(Session(name="Orphans"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db.create_action(
        Action(hand_id=hand.id, street="river", player_name="Hero", action_type="check")
    )
    # Only reachable with the enforcement off, which is how a file assembled by
    # hand or by another tool arrives.
    db._execute("PRAGMA foreign_keys = OFF")
    db._execute("DELETE FROM hands WHERE id = ?", (hand.id,))
    db._commit()

    report = db.schema_integrity()

    assert not report.is_intact
    assert any("actions row" in violation for violation in report.foreign_key_violations)
    assert "foreign-key violation(s)" in report.describe()
    db.close()


def test_a_column_the_current_ddl_declares_is_restored_and_recorded(
    tmp_path: Path, isolated_backup_dir: Path
) -> None:
    """The silent-corruption case: stamped current, physically incomplete.

    A file assembled from two restores, or edited by hand, keeps its current
    stamp, so no migration runs and ``CREATE TABLE IF NOT EXISTS`` leaves the
    damaged table alone. Every read of that table then failed one at a time with
    a bare sqlite3.OperationalError naming one column.

    The schema is forward-only and additive, so the column comes back carrying
    exactly what the migration that introduced it would have written -- and the
    repair is recorded, because a database that needed one is damaged whether or
    not it now opens.
    """
    path = tmp_path / "incomplete.sqlite3"
    seeded = _make_db(path)
    session = seeded.create_session(Session(name="Incomplete"))
    hand = seeded.create_hand(Hand(session_id=session.id, hand_number=1))
    seeded.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["hand_boundary"],
            description="Boundary is unclear.",
            evidence_snapshot={"hand_number": 1},
        )
    )
    seeded._execute("ALTER TABLE hand_issues DROP COLUMN evidence_snapshot")
    seeded._commit()
    seeded.close()

    reopened = _make_db(path)

    assert reopened.restored_columns == ("hand_issues.evidence_snapshot",)
    assert reopened.schema_integrity().is_intact
    assert (
        reopened._execute("SELECT evidence_snapshot FROM hand_issues").fetchone()[0]
        == "{}"
    )
    reopened.close()


def test_init_db_refuses_a_column_that_cannot_be_declared_back(
    tmp_path: Path, isolated_backup_dir: Path
) -> None:
    """A NOT NULL column with no default cannot be added, so it has to be loud."""
    path = tmp_path / "unrepairable.sqlite3"
    seeded = _make_db(path)
    seeded._execute("ALTER TABLE hand_reviews DROP COLUMN hand_summary")
    seeded._commit()
    seeded.close()

    reopened = PokerDatabase(path)
    with pytest.raises(RuntimeError) as caught:
        reopened.init_db()
    reopened.close()

    message = str(caught.value)
    assert "hand_reviews.hand_summary" in message
    assert "Restore it from a backup" in message


def test_a_lost_index_is_restored_on_the_next_open(tmp_path: Path) -> None:
    """A migration runs once, so an index lost afterwards never came back.

    Nothing failed when it went: the duplicate rows it forbids simply started
    being written. Recreating every index on open is idempotent for a healthy
    database and is the only repair path a file that lost one has.
    """
    path = tmp_path / "lost-index.sqlite3"
    seeded = _make_db(path)
    seeded._execute("DROP INDEX idx_actions_hand_street_order")
    seeded._execute("DROP INDEX idx_hand_players_hand_key")
    seeded._commit()
    assert not seeded.schema_integrity().is_intact
    seeded.close()

    reopened = _make_db(path)

    assert reopened.schema_integrity().is_intact
    reopened.close()


@pytest.mark.parametrize("table", ["hand_settlements", "settlement_entries"])
def test_a_table_that_only_its_migration_created_is_restored_on_the_next_open(
    tmp_path: Path, isolated_backup_dir: Path, table: str
) -> None:
    """These two lived only inside migration 7, unlike every other table.

    A database stamped 7 or later that lost one never got it back: the chain is
    long past the step that creates it, and the base DDL did not declare it. The
    settlement is where a hand's chips are accounted for, so its absence blocks
    every accounting read on every hand -- silently, because a hand with no
    settlement row is also a legitimate state.
    """
    path = tmp_path / f"lost-{table}.sqlite3"
    seeded = _make_db(path)
    seeded._execute(f"DROP TABLE {table}")
    seeded._commit()
    assert table in seeded.schema_integrity().missing_tables
    seeded.close()

    reopened = _make_db(path)

    assert reopened.schema_integrity().is_intact
    session = reopened.create_session(Session(name="After repair"))
    hand = reopened.create_hand(Hand(session_id=session.id, hand_number=1))
    reopened.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
    assert reopened.fetch_hand_settlement(hand.id).status == "settled"
    reopened.close()


def test_rows_that_broke_a_uniqueness_rule_are_reported_not_swallowed(
    tmp_path: Path, isolated_backup_dir: Path
) -> None:
    """Restoring the index cannot be silent when the rows already violate it."""
    path = tmp_path / "duplicate-seats.sqlite3"
    seeded = _make_db(path)
    session = seeded.create_session(Session(name="Duplicates"))
    hand = seeded.create_hand(Hand(session_id=session.id, hand_number=1))
    seeded._execute("DROP INDEX idx_hand_players_hand_key")
    for _ in range(2):
        seeded._execute(
            "INSERT INTO hand_players (hand_id, player_key, player_name) "
            "VALUES (?, 'hero', 'Hero')",
            (hand.id,),
        )
    seeded._commit()
    seeded.close()

    reopened = PokerDatabase(path)
    with pytest.raises(RuntimeError) as caught:
        reopened.init_db()
    reopened.close()

    message = str(caught.value)
    assert "uniqueness rule" in message
    assert "hand_players.player_key" in message
    assert "while the index was missing" in message


def test_the_integrity_reference_is_the_schema_the_product_creates(
    tmp_path: Path,
) -> None:
    """The contract is derived, so a table added tomorrow is covered tomorrow.

    A hand-maintained list of required tables is a list that falls behind the
    DDL, and the audit then reports a database missing the newest table as
    healthy -- which is exactly what happened to every version after 14.
    """
    fresh = _make_db(tmp_path / "reference.sqlite3")
    reference = db_module._reference_schema()
    actual_tables = {
        row["name"]
        for row in fresh._execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }
    actual_indexes = {
        row["name"]
        for row in fresh._execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }

    assert set(reference.tables) == actual_tables
    assert set(reference.indexes) == actual_indexes
    assert set(reference.column_declarations) == actual_tables
    fresh.close()


# --------------------------------------------------------------------------
# Cascade behaviour, for every entity the release plan names
# --------------------------------------------------------------------------


def _seed_full_history(db: PokerDatabase) -> dict[str, int]:
    """One session with every dependent record type attached to one hand."""
    session = db.create_session(Session(name="Cascade"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    other_hand = db.create_hand(Hand(session_id=session.id, hand_number=2))
    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id, player_key="hero", player_name="Hero", is_hero=True
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="hero",
            street="river",
            player_name="Hero",
            action_type="bet",
            amount=10,
        )
    )
    db.create_hand_review(_review(hand.id))
    db.create_coaching_response(_coaching_response(review_type="hand", hand_id=hand.id))
    db.create_coaching_response(
        _coaching_response(review_type="session", session_id=session.id)
    )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="hero",
                player_name="Hero",
                amount=20,
            )
        ],
    )
    correction = db.create_hand_correction(
        HandCorrection(hand_id=hand.id, correction_type="hand_facts", notes="pot")
    )
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["hand_boundary"],
            description="Boundary is unclear.",
        )
    )
    case = promote_issue_to_regression(
        db,
        issue.id,
        kind="cached_state",
        fixture_path="tests/fixtures/boundary.json",
        correction_id=correction.id,
    )
    db.create_solver_run(SolverRun(hand_id=hand.id, input_hash="cascade-hash"))
    video = db.create_video(
        VideoRecord(
            session_id=session.id,
            original_filename="session.mp4",
            stored_path="videos/session.mp4",
            file_size_bytes=1024,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(job_type="cv_reconstruction", video_id=video.id)
    )
    db.create_extracted_frame(
        ExtractedFrame(
            video_id=video.id,
            job_id=job.id,
            timestamp_seconds=1.0,
            frame_index=30,
            image_path="frames/job/frame_000030.png",
        )
    )
    db.upsert_reconstruction_frame_review(
        ReconstructionFrameReview(
            job_id=job.id,
            hand_number=1,
            source_image="frames/job/frame_000030.png",
            timestamp_seconds=1.0,
        )
    )
    return {
        "session": session.id,
        "hand": hand.id,
        "other_hand": other_hand.id,
        "correction": correction.id,
        "issue": issue.id,
        "case": case.id,
        "video": video.id,
        "job": job.id,
    }


def _count(db: PokerDatabase, table: str, where: str, parameter: int) -> int:
    return db._execute(
        f"SELECT COUNT(*) FROM {table} WHERE {where} = ?", (parameter,)
    ).fetchone()[0]


def test_deleting_a_hand_removes_its_dependents_and_nothing_else(
    tmp_path: Path,
) -> None:
    """Every category the release plan names, in one place.

    Individually these were untested for corrections, issues, settlements,
    settlement entries, reviews and regression cases; and in the other direction
    nothing pinned that a recording, a job, its frames or another hand SURVIVE,
    which is the half that loses an operator's data when it goes wrong.
    """
    db = _make_db(tmp_path / "cascade-hand.sqlite3")
    ids = _seed_full_history(db)

    db.delete_hand(ids["hand"])

    for table in (
        "hand_players",
        "actions",
        "hand_reviews",
        "hand_corrections",
        "hand_issues",
        "hand_settlements",
        "settlement_entries",
        "solver_runs",
    ):
        assert _count(db, table, "hand_id", ids["hand"]) == 0, table
    assert _count(db, "coaching_reviews", "hand_id", ids["hand"]) == 0
    assert _count(db, "regression_cases", "id", ids["case"]) == 0
    # Kept: the recording and its derived files outlive the hand they described,
    # and the session review is staled rather than deleted.
    assert db.fetch_video(ids["video"]) is not None
    assert _count(db, "processing_jobs", "id", ids["job"]) == 1
    assert _count(db, "extracted_frames", "job_id", ids["job"]) == 1
    assert _count(db, "reconstruction_frame_reviews", "job_id", ids["job"]) == 1
    assert db.fetch_hand(ids["other_hand"]) is not None
    session_coaching = db.fetch_coaching_reviews_by_session(ids["session"])
    assert len(session_coaching) == 1
    assert session_coaching[0].is_stale is True
    db.close()


def test_deleting_a_session_keeps_the_recording_and_unlinks_it(
    tmp_path: Path,
) -> None:
    """videos.session_id is ON DELETE SET NULL on purpose.

    A recording is a file on the operator's disk that the database only
    references. Cascading it would delete the row that says where that file is,
    leaving the file itself orphaned and unfindable.
    """
    db = _make_db(tmp_path / "cascade-session.sqlite3")
    ids = _seed_full_history(db)

    db.delete_session(ids["session"])

    assert db.fetch_session(ids["session"]) is None
    assert db.fetch_hands_by_session(ids["session"]) == []
    for table in ("hand_settlements", "hand_issues", "hand_corrections", "solver_runs"):
        assert _count(db, table, "hand_id", ids["hand"]) == 0, table
    assert _count(db, "regression_cases", "id", ids["case"]) == 0
    video = db.fetch_video(ids["video"])
    assert video is not None
    assert video.session_id is None
    assert _count(db, "processing_jobs", "id", ids["job"]) == 1
    # Session-scoped coaching describes a session that no longer exists.
    assert db.fetch_coaching_reviews_by_session(ids["session"]) == []
    db.close()


def test_deleting_a_video_takes_its_jobs_and_frames_and_leaves_the_session(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path / "cascade-video.sqlite3")
    ids = _seed_full_history(db)

    db.delete_video(ids["video"])

    assert db.fetch_video(ids["video"]) is None
    assert _count(db, "processing_jobs", "id", ids["job"]) == 0
    assert _count(db, "extracted_frames", "job_id", ids["job"]) == 0
    assert _count(db, "reconstruction_frame_reviews", "job_id", ids["job"]) == 0
    assert db.fetch_session(ids["session"]) is not None
    assert db.fetch_hand(ids["hand"]) is not None
    db.close()


def test_a_regression_case_outlives_its_correction_but_not_its_issue(
    tmp_path: Path,
) -> None:
    """The two foreign keys on regression_cases say different things deliberately.

    The correction is context: which edit was made when the bug was found. The
    issue is the subject: a regression case with no issue is proof of nothing,
    and its issue_id is NOT NULL, so cascade is the only coherent behaviour.
    """
    db = _make_db(tmp_path / "regression-fks.sqlite3")
    ids = _seed_full_history(db)

    db._execute("DELETE FROM hand_corrections WHERE id = ?", (ids["correction"],))
    db._commit()

    case = fetch_regression_case(db, ids["case"])
    assert case.correction_id is None
    assert case.status == "proposed"

    db._execute("DELETE FROM hand_issues WHERE id = ?", (ids["issue"],))
    db._commit()

    assert regressions_for_issue(db, ids["issue"]) == []
    assert _count(db, "regression_cases", "id", ids["case"]) == 0
    db.close()


def test_cascades_leave_no_row_whose_parent_is_gone(tmp_path: Path) -> None:
    """The invariant behind all of the above, stated once.

    Any future table that hangs off a hand or a session is covered by this
    without being named here, which the per-table assertions cannot be.
    """
    db = _make_db(tmp_path / "cascade-invariant.sqlite3")
    ids = _seed_full_history(db)

    db.delete_hand(ids["hand"])
    db.delete_session(ids["session"])
    db.delete_video(ids["video"])

    assert db.schema_integrity().foreign_key_violations == ()
    db.close()


def test_backups_dir_has_exactly_one_definition() -> None:
    """One source of truth: a second copy meant patching the obvious one (the
    historical poker_tracker.ui.video_storage constant) redirected no backup."""
    from poker_tracker.ui import video_storage

    assert not hasattr(video_storage, "BACKUPS_DIR")

    package = Path(backup_module.__file__).resolve().parents[1]
    definitions = [
        str(source.relative_to(package))
        for source in package.rglob("*.py")
        if any(
            line.startswith("BACKUPS_DIR")
            for line in source.read_text().splitlines()
        )
    ]
    assert definitions == ["persistence/backup.py"]
