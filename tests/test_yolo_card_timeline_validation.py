import json

import pytest

from cv_lab.scripts.validate_yolo_card_timeline import MalformedTimeline, main, validate_timeline


def _timeline(states, hand=None):
    if hand is None:
        hand = {
            "hand_number": 1,
            "t_start": states[0]["time_s"],
            "t_end": states[-1]["time_s"],
            "hero": states[0]["hero_cards"],
            "board": states[-1]["board_cards"],
            "streets": [],
            "source_images": [state["image"] for state in states],
        }
    return {"states": states, "hands": [hand]}


def _state(time_s, image, hero=None, board=None, missing=None):
    return {
        "time_s": time_s,
        "image": image,
        "hero_cards": hero if hero is not None else ["AS", "KD"],
        "board_cards": board if board is not None else [],
        "other_cards": [],
        "missing": missing,
    }


def _codes(report):
    return [warning["code"] for warning in report["hands"][0]["warnings"]]


def test_validate_clean_timeline_has_full_confidence():
    report = validate_timeline(_timeline([
        _state(0.0, "a.jpg", board=[]),
        _state(5.0, "b.jpg", board=["2C", "3D", "4H"]),
        _state(10.0, "c.jpg", board=["2C", "3D", "4H", "5S"]),
        _state(15.0, "d.jpg", board=["2C", "3D", "4H", "5S", "9C"]),
    ]))

    assert report["summary"]["total_warnings"] == 0
    assert report["summary"]["confidence_score"] == 1.0
    assert report["hands"][0]["checked_states"] == 4


def test_validate_reports_duplicate_invalid_counts_and_missing_labels():
    report = validate_timeline(_timeline([
        _state(
            0.0,
            "a.jpg",
            hero=["AS"],
            board=["AS", ""],
            missing={"image": "a.jpg", "note": "needs label"},
        ),
    ], hand={
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 0.0,
        "hero": ["AS"],
        "board": ["AS", ""],
        "streets": [],
        "source_images": ["a.jpg"],
    }))

    codes = _codes(report)
    assert "duplicate_visible_cards" in codes
    assert "invalid_hero_count" in codes
    assert "invalid_board_count" in codes
    assert codes.count("missing_label") >= 2
    assert report["summary"]["warning_hands"] == 1
    assert report["summary"]["confidence_score"] < 1.0


def test_validate_reports_board_regression_and_street_order_issue():
    report = validate_timeline(_timeline([
        _state(0.0, "a.jpg", board=[]),
        _state(5.0, "b.jpg", board=["2C", "3D", "4H", "5S"]),
        _state(10.0, "c.jpg", board=["2C", "3D", "4H"]),
    ], hand={
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 10.0,
        "hero": ["AS", "KD"],
        "board": ["2C", "3D", "4H"],
        "streets": [
            {"street": "turn", "time_s": 5.0, "board": ["2C", "3D", "4H", "5S"]},
            {"street": "flop", "time_s": 10.0, "board": ["2C", "3D", "4H"]},
        ],
        "source_images": ["a.jpg", "b.jpg", "c.jpg"],
    }))

    codes = _codes(report)
    assert "board_regression" in codes
    assert "street_order_issue" in codes


def test_validate_reports_reconstruction_warnings():
    states = [_state(0.0, "a.jpg", board=[]), _state(5.0, "b.jpg", board=["2C", "3D", "4H"])]
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 5.0,
        "hero": ["AS", "KD"],
        "board": ["2C", "3D", "4H"],
        "streets": [
            {"street": "flop", "time_s": 5.0, "board": ["2C", "3D", "4H"], "pot": 20},
            {"street": "turn", "time_s": 6.0, "board": ["2C", "3D", "4H", "5S"], "pot": 10},
        ],
        "players": [
            {"seat": 0, "position": "BTN", "player_name": "Hero", "is_hero": True},
            {"seat": 4, "position": "BTN", "player_name": "Seat4", "is_hero": False},
        ],
        "actions": [
            {"street": "flop", "action_index": 1, "action_type": "bet", "amount": 7, "player_name": "Hero"},
            {"street": "preflop", "action_index": 1, "action_type": "call", "amount": 3, "player_name": "Seat4"},
        ],
        "pot": 30,
        "reconciled": False,
        "contributed_est": 10,
        "source_images": ["a.jpg", "b.jpg"],
    }

    report = validate_timeline({"states": states, "hands": [hand]})
    codes = _codes(report)
    assert "pot_regression" in codes
    assert "position_issue" in codes
    assert "action_street_order" in codes
    assert "reconciliation_failed" in codes


def test_card_only_timeline_has_no_reconstruction_warnings():
    # A card-only timeline (no players/actions) must be unaffected by the new checks.
    report = validate_timeline(_timeline([
        _state(0.0, "a.jpg", board=[]),
        _state(5.0, "b.jpg", board=["2C", "3D", "4H"]),
    ]))
    codes = _codes(report)
    for code in ("pot_regression", "position_issue", "action_street_order", "reconciliation_failed"):
        assert code not in codes


def test_malformed_timeline_requires_states_list():
    with pytest.raises(MalformedTimeline):
        validate_timeline({"hands": []})


def test_cli_exit_codes_for_warnings_and_malformed_input(tmp_path):
    timeline_path = tmp_path / "timeline.json"
    timeline_path.write_text(json.dumps(_timeline([
        _state(0.0, "a.jpg", hero=["AS"], board=[]),
    ])), encoding="utf-8")

    assert main([str(timeline_path)]) == 0
    assert main([str(timeline_path), "--fail-on-warnings"]) == 1

    malformed_path = tmp_path / "bad.json"
    malformed_path.write_text("{", encoding="utf-8")
    assert main([str(malformed_path)]) == 2
