from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.import_export import import_session
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandPlayer,
    HandReview,
    HandSettlement,
    Session,
)


def _make_db(path: str | Path = ":memory:") -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


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
    first_hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, review_status="reviewed")
    )
    second_hand = db.create_hand(
        Hand(session_id=session.id, hand_number=2, review_status="reviewed")
    )
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
