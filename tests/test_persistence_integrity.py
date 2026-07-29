from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence import db as db_module
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import import_session
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandIssue,
    HandPlayer,
    HandReview,
    HandSettlement,
    Session,
    SolverRun,
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
    assert repaired.schema_version() == 13
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
    assert list(isolated_backup_dir.glob("*.sqlite3")) == []

    migrated = _make_db(path)

    snapshots = list(isolated_backup_dir.glob("*.sqlite3"))
    assert len(snapshots) == 1
    assert migrated.schema_version() == SCHEMA_VERSION
    migrated.close()


def test_migration_backup_is_skipped_for_a_fresh_file(
    tmp_path: Path, isolated_backup_dir: Path
) -> None:
    db = _make_db(tmp_path / "fresh.sqlite3")

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
    path = tmp_path / "unwritable.sqlite3"
    _seed_v12_database(path)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied")
    monkeypatch.setattr(backup_module, "BACKUPS_DIR", blocker / "backups")

    failed = PokerDatabase(path)
    # The raw sqlite3/OS error said "unable to open database file", which reads as
    # "your poker_tracker.db is broken". It names the backup directory now.
    with pytest.raises(RuntimeError) as caught:
        failed.init_db()
    failed.close()
    message = str(caught.value)
    assert "pre-migration backup" in message
    assert str(blocker / "backups") in message
    assert "was not applied" in message

    checked = _make_db_without_init(path)
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

    first = _make_db(path)
    first.close()
    assert len(list(isolated_backup_dir.glob("*.sqlite3"))) == 1

    second = _make_db(path)
    second.close()

    assert len(list(isolated_backup_dir.glob("*.sqlite3"))) == 1


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
