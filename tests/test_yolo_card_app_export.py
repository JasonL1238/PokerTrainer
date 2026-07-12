import json

from cv_lab.scripts.export_yolo_card_hands_for_app import (
    _card_to_app,
    apply_hand_corrections,
    export_timeline,
    load_hand_corrections,
    timeline_to_session_payload,
)
from poker_tracker.db import PokerDatabase
from poker_tracker.import_export import import_session


def test_card_to_app_normalizes_detector_labels() -> None:
    assert _card_to_app("AS") == "As"
    assert _card_to_app("10H") == "Th"
    assert _card_to_app("td") == "Td"


def test_timeline_to_session_payload_exports_only_valid_complete_hands() -> None:
    timeline = {
        "hands": [
            {
                "hand_number": 7,
                "t_start": 1.0,
                "t_end": 9.0,
                "hero": ["AS", "10H"],
                "board": ["QD", "7S", "2C"],
                "complete_cards": True,
                "warnings": [],
                "source_images": ["images/val/frame_000001.jpg"],
            },
            {
                "hand_number": 8,
                "t_start": 10.0,
                "t_end": 11.0,
                "hero": ["AS"],
                "board": ["QD", "7S"],
                "complete_cards": False,
                "warnings": ["hero_cards_not_two", "invalid_board_count"],
                "source_images": [],
            },
            {
                "hand_number": 9,
                "t_start": 12.0,
                "t_end": 13.0,
                "hero": ["AS", "10H"],
                "board": ["AS", "7S", "2C"],
                "complete_cards": False,
                "warnings": ["duplicate_visible_cards"],
                "source_images": [],
            },
        ]
    }

    payload = timeline_to_session_payload(
        timeline,
        timeline_path="timeline.json",
        session_name="Draft",
    )

    assert payload["session"]["name"] == "Draft"
    assert len(payload["hands"]) == 1
    assert payload["hands"][0]["hand"]["hero_cards"] == "As Th"
    assert payload["hands"][0]["hand"]["board_cards"] == "Qd 7s 2c"
    assert payload["hands"][0]["hand"]["source_type"] == "cv_import"
    assert payload["hands"][0]["hand"]["review_status"] == "needs_correction"
    assert payload["cv_import_summary"]["skipped_hands"] == 2


def test_timeline_to_session_payload_skips_validation_warning_hands() -> None:
    timeline = {
        "states": [
            {
                "time_s": 1.0,
                "image": "a.jpg",
                "hero_cards": ["AS", "AS"],
                "board_cards": ["QD", "7S", "2C"],
                "other_cards": [],
                "missing": None,
            }
        ],
        "hands": [
            {
                "hand_number": 1,
                "t_start": 1.0,
                "t_end": 1.0,
                "hero": ["AS", "10H"],
                "board": ["QD", "7S", "2C"],
                "complete_cards": True,
                "warnings": [],
                "source_images": ["a.jpg"],
            }
        ],
    }

    payload = timeline_to_session_payload(
        timeline,
        timeline_path="timeline.json",
        session_name="Draft",
    )
    allowed = timeline_to_session_payload(
        timeline,
        timeline_path="timeline.json",
        session_name="Draft",
        allow_validation_warnings=True,
    )

    assert payload["hands"] == []
    assert payload["cv_import_summary"]["skipped"][0]["reason"] == "validation_warnings"
    assert allowed["hands"][0]["hand"]["hero_cards"] == "As Th"


def test_hand_corrections_override_warning_hand_for_export(tmp_path) -> None:
    timeline = {
        "states": [
            {
                "time_s": 1.0,
                "image": "a.jpg",
                "hero_cards": ["AS"],
                "board_cards": ["QD", "7S"],
                "other_cards": [],
                "missing": None,
            }
        ],
        "hands": [
            {
                "hand_number": 3,
                "t_start": 1.0,
                "t_end": 1.0,
                "hero": ["AS"],
                "board": ["QD", "7S"],
                "complete_cards": False,
                "warnings": ["invalid_board_count"],
                "source_images": ["a.jpg"],
            }
        ],
    }
    corrections_path = tmp_path / "hand_corrections.csv"
    corrections_path.write_text(
        "hand_number,hero_cards,board_cards,action,notes\n"
        "3,Ah Qs,Qd 7s 2c,,manual fix\n",
        encoding="utf-8",
    )

    corrected = apply_hand_corrections(timeline, load_hand_corrections(corrections_path))
    payload = timeline_to_session_payload(
        corrected,
        timeline_path="timeline.json",
        session_name="Draft",
    )

    assert payload["hands"][0]["hand"]["hero_cards"] == "Ah Qs"
    assert payload["hands"][0]["hand"]["board_cards"] == "Qd 7s 2c"
    assert "manual_correction=keep" in payload["hands"][0]["hand"]["notes"]


def test_export_payload_imports_into_app_database(tmp_path) -> None:
    timeline_path = tmp_path / "timeline.json"
    out_path = tmp_path / "draft_session.json"
    timeline_path.write_text(
        json.dumps(
            {
                "hands": [
                    {
                        "hand_number": 1,
                        "t_start": 1.0,
                        "t_end": 6.0,
                        "hero": ["AH", "QS"],
                        "board": ["QD", "7S", "2C", "9H", "KC"],
                        "complete_cards": True,
                        "warnings": [],
                        "source_images": ["images/train/frame_000001.jpg"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = export_timeline(timeline_path, out_path, session_name="CV Import")
    db = PokerDatabase(":memory:")
    db.init_db()
    session = import_session(db, payload)
    hands = db.fetch_hands_by_session(session.id)

    assert out_path.exists()
    assert len(hands) == 1
    assert hands[0].hero_cards == "Ah Qs"
    assert hands[0].board_cards == "Qd 7s 2c 9h Kc"
    assert hands[0].source_type == "cv_import"
    assert hands[0].review_status == "needs_correction"
    db.close()


def test_export_imports_full_reconstructed_hand(tmp_path) -> None:
    # A reconstruction-spine hand carries players/actions/pot/winner; the exporter
    # must surface all of them into the app DB, not just the cards.
    timeline_path = tmp_path / "timeline.json"
    out_path = tmp_path / "draft_session.json"
    timeline_path.write_text(
        json.dumps({
            "hands": [{
                "hand_number": 1,
                "t_start": 0.0,
                "t_end": 8.0,
                "hero": ["As", "Kd"],
                "board": ["2c", "7d", "9h", "Ts", "Jc"],
                "complete_cards": True,
                "warnings": [],
                "players": [
                    {"seat": 0, "position": "SB", "player_name": "Hero", "starting_stack": 100, "is_hero": True},
                    {"seat": 4, "position": "BTN", "player_name": "Seat4", "starting_stack": 100, "is_hero": False},
                ],
                "actions": [
                    {"street": "preflop", "action_index": 1, "seat": 4, "position": "BTN",
                     "player_name": "Seat4", "action_type": "raise", "amount": 3.0,
                     "pot_before": 0.0, "stack_before": 100.0},
                    {"street": "flop", "action_index": 1, "seat": 0, "position": "SB",
                     "player_name": "Hero", "action_type": "bet", "amount": 7.0,
                     "pot_before": 6.0, "stack_before": 97.0},
                ],
                "pot": 20.0,
                "winner_seat": 0,
                "result": "Hero wins",
                "hero_bb_won": 10.0,
                "reconciled": True,
                "source_images": ["f.jpg"],
            }]
        }),
        encoding="utf-8",
    )

    payload = export_timeline(timeline_path, out_path, session_name="CV Import")
    db = PokerDatabase(":memory:")
    db.init_db()
    session = import_session(db, payload)
    hands = db.fetch_hands_by_session(session.id)

    assert len(hands) == 1
    hand = hands[0]
    assert hand.pot_size == 20.0
    assert hand.result == "Hero wins"
    assert hand.hero_bb_won == 10.0
    assert hand.hero_position == "SB"
    assert hand.source_type == "cv_import"

    players = db.fetch_players_by_hand(hand.id)
    actions = db.fetch_actions_by_hand(hand.id)
    assert len(players) == 2
    assert any(p.is_hero for p in players)
    assert {a.action_type for a in actions} == {"raise", "bet"}
    assert {a.amount for a in actions} == {3.0, 7.0}
    db.close()
