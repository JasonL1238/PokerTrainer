from __future__ import annotations

import sqlite3

from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import (
    export_session,
    import_hands_into_session,
    import_session,
)
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandPlayer,
    HandReview,
    Session,
)


def _db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def _coaching(hand_id: int, session_id: int) -> CoachingResponse:
    return CoachingResponse(
        provider_name="fixture",
        model_name="deterministic",
        raw_prompt="post-session completed hands; do not provide real-time advice",
        raw_response="review",
        review_type="hand",
        hand_id=hand_id,
        session_id=session_id,
    )


def test_current_schema_contains_audit_and_staleness_fields() -> None:
    db = _db()

    assert db.schema_version() == SCHEMA_VERSION == 13
    tables = {
        row["name"]
        for row in db._execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    coaching_columns = {
        row["name"] for row in db._execute("PRAGMA table_info(coaching_reviews)").fetchall()
    }
    hand_review_columns = {
        row["name"] for row in db._execute("PRAGMA table_info(hand_reviews)").fetchall()
    }

    assert "hand_corrections" in tables
    assert {"is_stale", "stale_reason"} <= coaching_columns
    assert {"is_stale", "stale_reason"} <= hand_review_columns
    db.close()


def test_v9_to_current_migration_preserves_existing_coaching(tmp_path) -> None:
    path = tmp_path / "v9.sqlite3"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_metadata (key, value) VALUES ('schema_version', '9');
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
            created_at TEXT NOT NULL
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
            created_at TEXT NOT NULL
        );
        INSERT INTO coaching_reviews (
            provider_name, model_name, raw_prompt, raw_response, review_type,
            hand_id, parsed_sections, created_at
        )
        VALUES ('legacy', 'model', 'prompt', 'response', 'hand', 1, '{}',
                '2026-07-28T00:00:00+00:00');
        """
    )
    legacy.commit()
    legacy.close()

    migrated = PokerDatabase(path)
    migrated.init_db()

    row = migrated._execute(
        "SELECT provider_name, is_stale, stale_reason FROM coaching_reviews"
    ).fetchone()
    assert migrated.schema_version() == SCHEMA_VERSION
    assert dict(row) == {
        "provider_name": "legacy",
        "is_stale": 0,
        "stale_reason": "",
    }
    assert migrated.fetch_hand_corrections(1) == []
    migrated.close()


def test_corrected_cv_facts_are_transactional_audited_and_stale_coaching() -> None:
    db = _db()
    session = db.create_session(Session(name="Correction loop"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            review_status="reviewed",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=12,
        )
    )
    db.create_hand_review(
        HandReview(
            hand_id=hand.id,
            hand_summary="summary",
            theory_coach="theory",
            exploit_coach="exploit",
            study_lesson="lesson",
        )
    )
    db.create_coaching_response(_coaching(hand.id, session.id))

    corrected = Hand(
        **{
            **hand.model_dump(),
            "board_cards": "Qd 7s 6c",
            "pot_size": 14,
        }
    )
    saved = db.update_hand_facts(
        corrected,
        correction_notes="Video frame shows 6c, not 2c.",
    )

    assert saved.board_cards == "Qd 7s 6c"
    assert saved.pot_size == 14
    assert saved.source_type == "corrected_cv"
    assert saved.review_status == "needs_correction"
    correction = db.fetch_hand_corrections(hand.id)[0]
    assert correction.correction_type == "hand_facts"
    assert correction.before_state["board_cards"] == "Qd 7s 2c"
    assert correction.after_state["board_cards"] == "Qd 7s 6c"
    assert correction.notes == "Video frame shows 6c, not 2c."
    assert db.fetch_reviews_by_hand(hand.id)[0].is_stale is True
    assert db.fetch_coaching_reviews_by_hand(hand.id)[0].is_stale is True
    db.close()


def test_action_correction_types_and_export_round_trip_are_retained() -> None:
    source = _db()
    session = source.create_session(Session(name="Portable corrections"))
    hand = source.create_hand(
        Hand(session_id=session.id, hand_number=1, source_type="cv_import")
    )
    hero = source.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            player_name="Hero",
            is_hero=True,
        )
    )
    action = source.create_corrected_action(
        Action(
            hand_id=hand.id,
            player_key=hero.player_key,
            street="flop",
            player_name="Hero",
            action_type="check",
        ),
        correction_notes="Missed action.",
    )
    source.update_action(
        action.model_copy(update={"action_type": "bet", "amount": 4}),
        correction_notes="Frame shows a 4 BB bet.",
    )
    source.delete_action(action.id, correction_notes="Duplicate action.")
    source.create_coaching_response(_coaching(hand.id, session.id))

    payload = export_session(source, session.id)
    target = _db()
    imported_session = import_session(target, payload)
    imported_hand = target.fetch_hands_by_session(imported_session.id)[0]

    assert [
        correction.correction_type
        for correction in reversed(target.fetch_hand_corrections(imported_hand.id))
    ] == ["action_create", "action_update", "action_delete"]
    imported_coaching = target.fetch_coaching_reviews_by_hand(imported_hand.id)
    assert len(imported_coaching) == 1
    assert imported_coaching[0].provider_name == "fixture"
    # Retained coaching survives the round trip, and arrives stale: it describes
    # the hand, ledger and winners of the database that produced it, and nothing
    # here can verify that claim against rows with different ids. The text is
    # kept so it can be re-run, not represented as current.
    assert imported_coaching[0].is_stale is True
    assert imported_coaching[0].stale_reason

    source.close()
    target.close()


def test_importing_hands_into_existing_session_keeps_hand_coaching() -> None:
    source = _db()
    source_session = source.create_session(Session(name="Source"))
    source_hand = source.create_hand(
        Hand(session_id=source_session.id, hand_number=1)
    )
    source.create_coaching_response(
        _coaching(source_hand.id, source_session.id)
    )
    payload = export_session(source, source_session.id)

    target = _db()
    target_session = target.create_session(Session(name="Target"))
    import_hands_into_session(target, payload, target_session.id)
    imported_hand = target.fetch_hands_by_session(target_session.id)[0]
    reviews = target.fetch_coaching_reviews_by_hand(imported_hand.id)

    assert len(reviews) == 1
    assert reviews[0].session_id == target_session.id
    source.close()
    target.close()
