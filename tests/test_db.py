import pytest
from pydantic import ValidationError

from poker_tracker.db import PokerDatabase, SCHEMA_VERSION
from poker_tracker.models import Action, Hand, HandPlayer, HandReview, Session


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def test_database_initialization_is_idempotent() -> None:
    db = make_db()
    db.init_db()

    assert db.fetch_sessions() == []
    assert db.schema_version() == SCHEMA_VERSION

    db.close()


def test_schema_supports_new_hand_fields() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))

    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_position="BTN",
            hero_cards="AhQs",
            board_cards="Qd 7s 2c",
            hero_bb_won=4.5,
            review_status="needs_correction",
            confidence_score=0.72,
            source_type="corrected_cv",
            tags=["MISSED_VALUE", "BIG_POT"],
        )
    )

    saved = db.fetch_hands_by_session(session.id)[0]
    assert hand.id is not None
    assert saved.hero_cards == "Ah Qs"
    assert saved.review_status == "needs_correction"
    assert saved.confidence_score == 0.72
    assert saved.source_type == "corrected_cv"
    assert saved.tags == ["MISSED_VALUE", "BIG_POT"]

    db.close()


def test_create_session_hand_action_review_still_works() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session", stakes="1/2 NL"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1, hero_cards="As Qh"))
    action = db.create_action(
        Action(
            hand_id=hand.id,
            street="preflop",
            player_name="Hero",
            position="BTN",
            action_type="raise",
            amount=2.5,
        )
    )
    review = db.create_hand_review(
        HandReview(
            hand_id=hand.id,
            hand_summary="summary",
            theory_coach="theory",
            exploit_coach="exploit",
            ev_math_notes="math",
            study_lesson="lesson",
            next_review_question="question",
            notes="review note",
        )
    )

    assert session.id is not None
    assert hand.id is not None
    assert action.id is not None
    assert action.action_index == 1
    assert review.id is not None
    assert db.fetch_reviews_by_hand(hand.id)[0].ev_math_notes == "math"

    db.close()


def test_action_order_increments_by_hand_and_street() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))

    first = db.create_action(
        Action(hand_id=hand.id, street="preflop", player_name="A", action_type="raise")
    )
    second = db.create_action(
        Action(hand_id=hand.id, street="preflop", player_name="B", action_type="call")
    )
    flop_first = db.create_action(
        Action(hand_id=hand.id, street="flop", player_name="A", action_type="bet")
    )

    assert first.action_index == 1
    assert second.action_index == 2
    assert flop_first.action_index == 1

    db.close()


def test_create_hand_players_and_status_update() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))

    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_name="Hero",
            position="BTN",
            starting_stack=200,
            is_hero=True,
        )
    )
    db.update_hand_status(hand.id, "reviewed")

    players = db.fetch_players_by_hand(hand.id)
    saved = db.fetch_hand(hand.id)
    assert players[0].player_name == "Hero"
    assert players[0].is_hero is True
    assert saved.review_status == "reviewed"

    db.close()


def test_update_and_delete_action() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    action = db.create_action(
        Action(hand_id=hand.id, street="preflop", player_name="Hero", action_type="call")
    )

    db.update_action(
        Action(
            id=action.id,
            hand_id=hand.id,
            street="preflop",
            action_index=1,
            player_name="Hero",
            action_type="raise",
            amount=8,
        )
    )
    assert db.fetch_actions_by_hand(hand.id)[0].action_type == "raise"

    db.delete_action(action.id)
    assert db.fetch_actions_by_hand(hand.id) == []

    db.close()


def test_delete_hand_cascades_related_rows() -> None:
    db = make_db()
    session = db.create_session(Session(name="Test session"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db.create_action(Action(hand_id=hand.id, street="preflop", player_name="Hero", action_type="win"))

    db.delete_hand(hand.id)

    assert db.fetch_hands_by_session(session.id) == []
    assert db.fetch_actions_by_hand(hand.id) == []

    db.close()


def test_validation_catches_bad_cards_and_action_types() -> None:
    with pytest.raises(ValidationError):
        Hand(session_id=1, hand_number=1, hero_cards="Ax Qs")

    with pytest.raises(ValidationError):
        Hand(session_id=1, hand_number=1, board_cards="Qd Qd 2c")

    with pytest.raises(ValidationError):
        Hand(session_id=1, hand_number=1, hero_cards="Ah Qs", board_cards="Ah 7d 2c")

    with pytest.raises(ValidationError):
        Action(hand_id=1, street="preflop", player_name="Hero", action_type="punt")
