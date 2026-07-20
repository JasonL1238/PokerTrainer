"""Golden end-to-end test locking the YOLO -> app -> coaching-prompt seam.

This is the one integration test that exercises the whole pipe together:

    YOLO card reconstruction timeline (ground truth, in-test fixture)
        -> export_timeline / timeline_to_session_payload  (cv_lab exporter)
        -> import_session                                 (poker_tracker)
        -> PokerDatabase (sqlite file on tmp_path)
        -> fetch_hand / fetch_actions_by_hand / fetch_players_by_hand
        -> build_hand_review_prompt                       (coaching prompt)

Nothing else covers the reconstructed data flowing all the way into a coaching
prompt, so this test locks that seam. It intentionally uses the "reconstruction
spine" path (a hand carrying `players` and `actions`) so that actions/players
flow through end to end -- the existing tests/test_yolo_card_app_export.py proves
the exporter can carry that data, and here we confirm it survives into the prompt.
"""
from __future__ import annotations

import json

from cv_lab.scripts.export_yolo_card_hands_for_app import export_timeline
from poker_tracker.coaching_prompts import POST_SESSION_SAFETY, build_hand_review_prompt
from poker_tracker.db import PokerDatabase
from poker_tracker.import_export import import_session


# ---------------------------------------------------------------------------
# Ground truth: a YOLO reconstruction-spine timeline.
#
# Cards are given in raw detector-label form (e.g. "AS", "10H") so the test also
# exercises the _card_to_app normalization in the exporter. The app-format cards
# these should become are recorded in GROUND_TRUTH_* below.
# ---------------------------------------------------------------------------
def _spine_timeline() -> dict:
    return {
        "hands": [
            {
                "hand_number": 42,
                "t_start": 0.0,
                "t_end": 30.0,
                "hero": ["AS", "10H"],
                "board": ["QD", "7S", "2C", "9H", "KC"],
                "complete_cards": True,
                "warnings": [],
                "players": [
                    {"seat": 0, "position": "SB", "player_name": "Hero",
                     "starting_stack": 100, "is_hero": True},
                    {"seat": 5, "position": "BTN", "player_name": "Villain",
                     "starting_stack": 100, "is_hero": False},
                ],
                "actions": [
                    {"street": "preflop", "action_index": 1, "seat": 5, "position": "BTN",
                     "player_name": "Villain", "action_type": "raise", "amount": 3.0,
                     "pot_before": 1.5, "stack_before": 100.0},
                    {"street": "preflop", "action_index": 2, "seat": 0, "position": "SB",
                     "player_name": "Hero", "action_type": "call", "amount": 2.5,
                     "pot_before": 4.5, "stack_before": 99.0},
                    {"street": "flop", "action_index": 1, "seat": 0, "position": "SB",
                     "player_name": "Hero", "action_type": "check", "amount": None,
                     "pot_before": 6.5, "stack_before": 96.5},
                    {"street": "flop", "action_index": 2, "seat": 5, "position": "BTN",
                     "player_name": "Villain", "action_type": "bet", "amount": 4.0,
                     "pot_before": 6.5, "stack_before": 96.5},
                    {"street": "turn", "action_index": 1, "seat": 0, "position": "SB",
                     "player_name": "Hero", "action_type": "call", "amount": 4.0,
                     "pot_before": 14.5, "stack_before": 92.5},
                    {"street": "river", "action_index": 1, "seat": 0, "position": "SB",
                     "player_name": "Hero", "action_type": "bet", "amount": 12.0,
                     "pot_before": 18.5, "stack_before": 88.5},
                ],
                "pot": 42.5,
                "winner_seat": 0,
                "result": "Hero wins",
                "hero_bb_won": 21.0,
                "reconciled": True,
                "source_images": ["images/val/frame_000042.jpg"],
            }
        ]
    }


GROUND_TRUTH_HERO = "As Th"
GROUND_TRUTH_BOARD = "Qd 7s 2c 9h Kc"
GROUND_TRUTH_HERO_POSITION = "SB"


def test_yolo_reconstruction_to_coaching_prompt_seam(tmp_path) -> None:
    """Full pipe: spine timeline -> export -> import -> read back -> coaching prompt."""
    timeline_path = tmp_path / "timeline.json"
    out_path = tmp_path / "draft_session.json"
    db_path = tmp_path / "poker.db"
    timeline_path.write_text(json.dumps(_spine_timeline()), encoding="utf-8")

    # 1) Real export/reconstruction.
    payload = export_timeline(timeline_path, out_path, session_name="Golden E2E")
    assert out_path.exists()
    assert payload["cv_import_summary"]["exported_hands"] == 1

    # 2) Real import into a sqlite-file PokerDatabase (mirrors app usage).
    db = PokerDatabase(str(db_path))
    db.init_db()
    session = import_session(db, payload)

    # 3) Read the hand + actions + players BACK from the DB via real read APIs.
    hands = db.fetch_hands_by_session(session.id)
    assert len(hands) == 1
    hand = db.fetch_hand(hands[0].id)
    assert hand is not None
    actions = db.fetch_actions_by_hand(hand.id)
    players = db.fetch_players_by_hand(hand.id)
    session_from_db = db.fetch_session(session.id)
    assert session_from_db is not None

    # --- (a) Reconstructed hand fields match ground truth (app card format). ---
    assert hand.hero_cards == GROUND_TRUTH_HERO
    assert hand.board_cards == GROUND_TRUTH_BOARD
    assert hand.hero_position == GROUND_TRUTH_HERO_POSITION
    assert hand.source_type == "cv_import"
    assert hand.hero_bb_won == 21.0
    assert hand.result == "Hero wins"
    assert hand.pot_size == 42.5

    # --- Confirm actions/players actually flowed through the spine path. ---
    assert len(players) == 2
    assert sum(1 for p in players if p.is_hero) == 1
    assert {p.player_name for p in players} == {"Hero", "Villain"}
    assert len(actions) == 6
    assert {a.action_type for a in actions} == {"raise", "call", "check", "bet"}
    assert {a.street for a in actions} == {"preflop", "flop", "turn", "river"}

    # 4) Build the coaching prompt on the reconstructed data.
    prompt = build_hand_review_prompt(session_from_db, hand, actions, players)

    # --- (b) Prompt contains the reconstructed hero cards and board. ---
    assert GROUND_TRUTH_HERO in prompt
    assert GROUND_TRUTH_BOARD in prompt
    assert GROUND_TRUTH_HERO_POSITION in prompt

    # --- (c) Prompt preserves the post-session safety text. ---
    assert POST_SESSION_SAFETY in prompt

    # --- (d) Since actions flowed through, the prompt's hand history reflects them. ---
    assert "Preflop:" in prompt
    assert "Flop:" in prompt
    assert "Turn:" in prompt
    assert "River:" in prompt
    # Specific reconstructed action lines (see hand_history._format_action).
    assert "BTN Villain raise 3" in prompt
    assert "SB Hero bet 12" in prompt
    # Players block reflects the reconstructed seats.
    assert "Hero Hero: SB" in prompt
    assert "Villain: BTN" in prompt

    db.close()


def test_incomplete_and_duplicate_hands_are_dropped_end_to_end(tmp_path) -> None:
    """Negative case: invalid/incomplete hands never reach the DB or a prompt.

    Mirrors the drop behavior asserted in tests/test_yolo_card_app_export.py:
    a hand with the wrong card count and a hand with duplicate cards are both
    skipped by the exporter, so no saved hand is produced for them. Only the
    single valid spine hand survives to become a coaching prompt.
    """
    timeline_path = tmp_path / "timeline.json"
    out_path = tmp_path / "draft_session.json"
    db_path = tmp_path / "poker.db"

    timeline = _spine_timeline()
    # Bad card count: hero has only one card, board has only two.
    timeline["hands"].append({
        "hand_number": 43,
        "t_start": 31.0,
        "t_end": 40.0,
        "hero": ["AS"],
        "board": ["QD", "7S"],
        "complete_cards": False,
        "warnings": ["hero_cards_not_two", "invalid_board_count"],
        "source_images": [],
    })
    # Duplicate cards: As appears in both hero and board.
    timeline["hands"].append({
        "hand_number": 44,
        "t_start": 41.0,
        "t_end": 50.0,
        "hero": ["AS", "10H"],
        "board": ["AS", "7S", "2C"],
        "complete_cards": False,
        "warnings": ["duplicate_visible_cards"],
        "source_images": [],
    })
    timeline_path.write_text(json.dumps(timeline), encoding="utf-8")

    payload = export_timeline(timeline_path, out_path, session_name="Golden E2E Drop")
    assert payload["cv_import_summary"]["exported_hands"] == 1
    assert payload["cv_import_summary"]["skipped_hands"] == 2

    db = PokerDatabase(str(db_path))
    db.init_db()
    session = import_session(db, payload)

    hands = db.fetch_hands_by_session(session.id)
    # Only the valid spine hand survived; the incomplete/duplicate hands were dropped.
    assert len(hands) == 1
    assert hands[0].hero_cards == GROUND_TRUTH_HERO
    assert hands[0].board_cards == GROUND_TRUTH_BOARD

    db.close()
