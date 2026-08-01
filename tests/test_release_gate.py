"""Phase 3 release-gate scaffolding tests."""

from __future__ import annotations

import json
from pathlib import Path

from poker_tracker.release_gate.evaluate import evaluate_answer_key_against_timeline
from poker_tracker.release_gate.runner import run_release_gate
from poker_tracker.validation.corpus import EXIT_SETUP_INVALID

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "validation" / "clubwpt_v1.json"
TRUTH = REPO / "validation" / "truth" / "fixture_synthetic_hu_01.json"


def test_release_gate_fixture_mode_fails_closed_on_incomplete_corpus(tmp_path: Path):
    result = run_release_gate(
        manifest_path=MANIFEST,
        mode="fixture",
        report_dir=tmp_path / "reports",
    )
    assert result.ok is False
    assert result.exit_code == EXIT_SETUP_INVALID
    assert result.report_path is not None
    assert result.report_path.is_file()
    assert result.report["stages"][0]["name"] == "corpus"
    assert result.report["stages"][0]["ok"] is False


def test_release_gate_full_mode_fails_closed(tmp_path: Path):
    result = run_release_gate(
        manifest_path=MANIFEST,
        mode="full",
        report_dir=tmp_path / "reports",
    )
    assert result.ok is False
    assert result.exit_code == EXIT_SETUP_INVALID


def test_release_gate_report_redacts_secrets(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "super-secret")
    monkeypatch.setenv("POKER_DB_PATH", str(tmp_path / "db.sqlite"))
    result = run_release_gate(
        manifest_path=MANIFEST,
        mode="fixture",
        report_dir=tmp_path / "reports",
    )
    env = result.report["environment"]["env"]
    assert env["APP_PASSWORD"] == "<redacted>"
    assert env["POKER_DB_PATH"].endswith("db.sqlite")


def test_evaluate_fails_closed_on_empty_truth():
    report = evaluate_answer_key_against_timeline({"hands": []}, {"hands": []})
    assert report["ok"] is False
    assert report["fail_closed"] == "empty_truth"


def test_evaluate_missing_predicted_amount_is_not_a_match():
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    timeline = {
        "hands": [
            {
                "hand_number": 1,
                "t_start": 0.0,
                "t_end": 12.0,
                "complete": True,
                "hero": ["As", "Kd"],
                "board": ["2c", "7d", "9h", "Ts", "Jc"],
                "dealer_seat": 1,
                "winner_seat": 0,
                "pot": 16.0,
                "hero_bb_won": 8.0,
                "terminal_event": "showdown",
                "actions": [
                    {"street": "preflop", "seat": 0, "action_type": "raise", "amount": None},
                    {"street": "preflop", "seat": 1, "action_type": "call", "amount": 2.0},
                    {"street": "flop", "seat": 1, "action_type": "check", "amount": None},
                    {"street": "flop", "seat": 0, "action_type": "bet", "amount": 5.0},
                    {"street": "flop", "seat": 1, "action_type": "call", "amount": 5.0},
                    {"street": "turn", "seat": 1, "action_type": "check", "amount": None},
                    {"street": "turn", "seat": 0, "action_type": "check", "amount": None},
                    {"street": "river", "seat": 1, "action_type": "check", "amount": None},
                    {"street": "river", "seat": 0, "action_type": "check", "amount": None},
                ],
            },
            {
                "hand_number": 2,
                "t_start": 20.0,
                "t_end": 28.0,
                "complete": False,
                "hero": ["7h", "2d"],
                "board": [],
                "dealer_seat": 0,
                "winner_seat": None,
                "pot": None,
                "hero_bb_won": -0.5,
                "terminal_event": "unobserved",
                "partial_start": True,
                "actions": [
                    {"street": "preflop", "seat": 1, "action_type": "raise", "amount": 3.0},
                    {"street": "preflop", "seat": 0, "action_type": "fold", "amount": None},
                ],
            },
        ]
    }
    report = evaluate_answer_key_against_timeline(truth, timeline)
    assert report["ok"] is False
    assert any(
        e["category"] == "action_amount"
        for hand in report["per_hand"]
        for e in hand["errors"]
    )


def test_evaluate_wiped_actions_fail_the_gate():
    """Critical fields alone must not pass when the action line is missing."""
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    timeline = {
        "hands": [
            {
                "hand_number": 1,
                "t_start": 0.0,
                "t_end": 12.0,
                "complete": True,
                "hero": ["As", "Kd"],
                "board": ["2c", "7d", "9h", "Ts", "Jc"],
                "dealer_seat": 1,
                "winner_seat": 0,
                "pot": 16.0,
                "hero_bb_won": 8.0,
                "result": "Hero wins",
                "terminal_event": "showdown",
                "actions": [],
            },
            {
                "hand_number": 2,
                "t_start": 20.0,
                "t_end": 28.0,
                "complete": False,
                "hero": ["7h", "2d"],
                "board": [],
                "dealer_seat": 0,
                "winner_seat": None,
                "pot": None,
                "hero_bb_won": -0.5,
                "terminal_event": "unobserved",
                "partial_start": True,
                "actions": [],
            },
        ]
    }
    report = evaluate_answer_key_against_timeline(truth, timeline)
    assert report["ok"] is False
    assert report["critical_errors"] >= 1
    assert any(
        e["category"] == "missing_action"
        for hand in report["per_hand"]
        for e in hand["errors"]
    )


def test_evaluate_null_seat_does_not_abort():
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    timeline = {
        "hands": [
            {
                "hand_number": 1,
                "t_start": 0.0,
                "t_end": 12.0,
                "complete": True,
                "hero": ["As", "Kd"],
                "board": ["2c", "7d", "9h", "Ts", "Jc"],
                "dealer_seat": 1,
                "winner_seat": 0,
                "pot": 16.0,
                "hero_bb_won": 8.0,
                "terminal_event": "showdown",
                "actions": [
                    {"street": "preflop", "seat": None, "action_type": "raise", "amount": 3.0},
                ],
            }
        ]
    }
    report = evaluate_answer_key_against_timeline(truth, timeline)
    assert report["ok"] is False
    assert any(
        e["category"] in {"illegal_action", "missing_action"}
        for hand in report["per_hand"]
        for e in hand["errors"]
    )


def test_partial_unobserved_prediction_matches_partial_truth():
    truth = {
        "hands": [
            {
                "hand_id": "h2",
                "t_first": 20.0,
                "t_last": 28.0,
                "completion_class": "partial",
                "terminal_event": "unobserved",
                "partial_start": True,
                "partial_end": False,
                "actions_complete": False,
                "hero_cards": ["7h", "2d"],
                "final_board": [],
                "dealt_in_seats": [0, 1],
                "actions": [],
                # The recording starts mid-hand and never shows the end, so
                # these facts are unobservable by name rather than by silence.
                "unobservable": [
                    "dealer_seat",
                    "winner_seat",
                    "result",
                    "final_pot",
                    "hero_net",
                ],
            }
        ]
    }
    timeline = {
        "hands": [
            {
                "hand_number": 2,
                "t_start": 20.0,
                "t_end": 28.0,
                "complete": False,
                "hero": ["7h", "2d"],
                "board": [],
                "terminal_event": "unobserved",
                "partial_start": True,
                "partial_end": False,
                "actions": [],
            }
        ]
    }
    report = evaluate_answer_key_against_timeline(truth, timeline)
    assert report["ok"] is True
    assert report["critical_errors"] == 0
    # The skipped checks are reported, not silently dropped — including the
    # action line, which is unscored because the answer key says it is partial.
    assert report["excluded_facts"] == [
        "action_line:answer key declares actions incomplete",
        "dealer_seat",
        "final_pot",
        "hero_net",
        "result",
        "winner_seat",
    ]


def test_evaluate_merged_prediction_fails():
    truth = {
        "hands": [
            {
                "hand_id": "h1",
                "t_first": 0.0,
                "t_last": 10.0,
                "completion_class": "partial",
                "terminal_event": "unobserved",
                "partial_start": True,
                "partial_end": False,
                "actions_complete": False,
                "hero_cards": None,
                "final_board": [],
                "dealt_in_seats": [0],
                "actions": [],
            },
            {
                "hand_id": "h2",
                "t_first": 5.0,
                "t_last": 15.0,
                "completion_class": "partial",
                "terminal_event": "unobserved",
                "partial_start": True,
                "partial_end": False,
                "actions_complete": False,
                "hero_cards": None,
                "final_board": [],
                "dealt_in_seats": [0],
                "actions": [],
            },
        ]
    }
    timeline = {
        "hands": [
            {
                "hand_number": 1,
                "t_start": 0.0,
                "t_end": 15.0,
                "complete": False,
                "terminal_event": "unobserved",
                "partial_start": True,
                "hero": None,
                "board": [],
                "actions": [],
            }
        ]
    }
    report = evaluate_answer_key_against_timeline(truth, timeline)
    assert report["ok"] is False
    assert any(
        e["category"] == "merged_hand"
        for hand in report["per_hand"]
        for e in hand["errors"]
    )
