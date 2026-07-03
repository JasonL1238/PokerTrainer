import json

import pytest
from pydantic import ValidationError

from poker_tracker.analytics import compute_session_stats
from poker_tracker.db import PokerDatabase
from poker_tracker.import_export import export_hand, export_session, import_session
from poker_tracker.models import Action, Hand, Session
from poker_tracker.seed_data import create_sample_data


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def test_analytics_calculations() -> None:
    db = make_db()
    session = db.create_session(Session(name="Stats"))
    win = db.create_hand(
        Hand(session_id=session.id, hand_number=1, hero_bb_won=10, tags=["BIG_POT"], review_status="reviewed")
    )
    loss = db.create_hand(
        Hand(session_id=session.id, hand_number=2, hero_bb_won=-4, tags=["MULTIWAY"])
    )
    db.create_action(Action(hand_id=win.id, street="flop", player_name="Hero", action_type="bet"))
    db.create_action(Action(hand_id=loss.id, street="turn", player_name="Hero", action_type="call"))

    stats = compute_session_stats(db, session.id)

    assert stats.hand_count == 2
    assert stats.total_hero_bb == 6
    assert stats.average_hero_bb == 3
    assert stats.biggest_winning_hands[0].hand_number == 1
    assert stats.biggest_losing_hands[0].hand_number == 2
    assert stats.hands_by_tag == {"BIG_POT": 1, "MULTIWAY": 1}
    assert stats.hands_by_review_status["reviewed"] == 1
    assert stats.hands_by_review_status["unreviewed"] == 1
    assert stats.action_counts_by_type == {"bet": 1, "call": 1}
    assert stats.aggression_count == 1
    assert stats.passive_count == 1

    db.close()


def test_seed_data_creation() -> None:
    db = make_db()
    session = create_sample_data(db)
    hands = db.fetch_hands_by_session(session.id)
    stats = compute_session_stats(db, session.id)

    assert len(hands) == 5
    assert {"MISSED_VALUE", "MULTIWAY", "PREFLOP_3BET_SPOT"}.issubset(stats.hands_by_tag)
    assert stats.biggest_losing_hands[0].hero_bb_won == -65

    db.close()


def test_json_export_for_hand_and_session() -> None:
    db = make_db()
    session = create_sample_data(db)
    hand = db.fetch_hands_by_session(session.id)[0]

    hand_payload = export_hand(db, hand.id)
    session_payload = export_session(db, session.id)

    assert hand_payload["hand"]["hand_number"] == 1
    assert hand_payload["actions"]
    assert session_payload["session"]["name"] == "Sample post-session review"
    assert len(session_payload["hands"]) == 5
    json.dumps(session_payload)

    db.close()


def test_json_export_rejects_missing_ids() -> None:
    db = make_db()

    with pytest.raises(ValueError, match="Hand not found"):
        export_hand(db, 9999)

    with pytest.raises(ValueError, match="Session not found"):
        export_session(db, 9999)

    db.close()


def test_json_import_round_trip() -> None:
    source = make_db()
    session = create_sample_data(source)
    payload = export_session(source, session.id)

    target = make_db()
    imported = import_session(target, payload)

    assert imported.id is not None
    assert len(target.fetch_hands_by_session(imported.id)) == 5
    imported_first = target.fetch_hands_by_session(imported.id)[0]
    assert target.fetch_actions_by_hand(imported_first.id)
    assert target.fetch_players_by_hand(imported_first.id)
    assert imported_first.tags

    source.close()
    target.close()


def test_json_import_preserves_review_status_tags_and_action_order() -> None:
    source = make_db()
    session = source.create_session(Session(name="Import details"))
    hand = source.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            review_status="needs_correction",
            tags=["MISSED_VALUE", "RIVER_DECISION"],
        )
    )
    source.create_action(Action(hand_id=hand.id, street="river", player_name="Hero", action_type="check"))
    source.create_action(Action(hand_id=hand.id, street="river", player_name="Villain", action_type="bet", amount=12))
    payload = export_session(source, session.id)

    target = make_db()
    imported = import_session(target, payload)
    imported_hand = target.fetch_hands_by_session(imported.id)[0]
    imported_actions = target.fetch_actions_by_hand(imported_hand.id)

    assert imported_hand.review_status == "needs_correction"
    assert imported_hand.tags == ["MISSED_VALUE", "RIVER_DECISION"]
    assert [action.action_index for action in imported_actions] == [1, 2]
    assert [action.player_name for action in imported_actions] == ["Hero", "Villain"]

    source.close()
    target.close()


def test_json_import_rejects_bad_payload_without_creating_session() -> None:
    db = make_db()

    with pytest.raises(KeyError):
        import_session(db, {"hands": []})

    assert db.fetch_sessions() == []
    db.close()


def test_json_import_rejects_invalid_hand_without_partial_session() -> None:
    db = make_db()
    payload = {
        "session": Session(name="Bad import").model_dump(mode="json"),
        "hands": [
            {
                "hand": Hand(session_id=1, hand_number=1, hero_cards="Ah Qs").model_dump(mode="json")
                | {"board_cards": "Ah 7d 2c"},
                "players": [],
                "actions": [],
                "reviews": [],
            }
        ],
    }

    with pytest.raises(ValidationError):
        import_session(db, payload)

    assert db.fetch_sessions() == []
    db.close()
