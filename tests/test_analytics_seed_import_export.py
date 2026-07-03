import json

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
