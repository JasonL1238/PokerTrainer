import json

import pytest

from cv_lab.scripts.eval.validate_yolo_card_timeline import (
    MalformedTimeline,
    main,
    validate_timeline,
)


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


# --------------------------------------------------------------------------- #
# Confidence: severity-weighted, not warning density.
#
# The old form was 1 - warning_count / (3 * checked_states), so a hand's score
# depended on how many frames it happened to span. Two fatal warnings across 61
# states scored 0.989 on a session in which every board had been destroyed.
# --------------------------------------------------------------------------- #
def _spine_hand(**overrides):
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 1.0,
        "hero": ["AS", "KD"],
        "board": ["2C", "3D", "4H"],
        "streets": [],
        "actions": [],
        "players": [],
        "warnings": [],
        "source_images": [],
    }
    hand.update(overrides)
    return hand


def _n_state_hand(n, prefix, **overrides):
    states = [_state(float(i), f"{prefix}{i}.jpg", board=["2C", "3D", "4H"])
              for i in range(n)]
    hand = _spine_hand(source_images=[s["image"] for s in states], **overrides)
    return {"states": states, "hands": [hand]}


def test_confidence_is_severity_weighted_not_density():
    """Reproduces the exact 06-21 arithmetic: 2 warnings over 61 checked states.

    The density form gave 1 - 2 / (3 * 61) = 0.989 on a session in which every
    board had been destroyed. Two fatal reconstruction facts cannot leave a hand
    98.9% trustworthy however many frames it spanned."""
    report = validate_timeline(_n_state_hand(
        61, "s", warnings=["board_zone_yield_zero", "hero_seat_mismatch"]))
    hand = report["hands"][0]

    assert hand["checked_states"] == 61
    assert hand["warning_count"] == 2
    assert round(1.0 - 2 / (3 * 61), 3) == 0.989      # what it used to score
    assert hand["confidence_score"] <= 0.5


def test_confidence_does_not_depend_on_how_many_frames_a_hand_spans():
    """The same fault over 4 states and over 61 must score identically."""
    def score(n, prefix):
        report = validate_timeline(_n_state_hand(
            n, prefix, warnings=["board_zone_yield_zero"]))
        return report["hands"][0]["confidence_score"]

    assert score(4, "short") == score(61, "long") == 0.4


def test_session_confidence_is_the_worst_hand():
    """A session is only as trustworthy as its worst hand; averaging is what let
    a session with destroyed boards report 0.989."""
    clean = [_state(0.0, "a.jpg", board=["2C", "3D", "4H"])]
    broken = [_state(1.0, "b.jpg", board=["2C", "3D"])]
    report = validate_timeline({
        "states": clean + broken,
        "hands": [
            _spine_hand(hand_number=1, source_images=["a.jpg"]),
            _spine_hand(hand_number=2, source_images=["a.jpg"]),
            _spine_hand(hand_number=3, board=["2C", "3D"], source_images=["b.jpg"]),
        ],
    })
    scores = [h["confidence_score"] for h in report["hands"]]
    assert scores[:2] == [1.0, 1.0]
    assert scores[2] < 1.0
    assert report["summary"]["confidence_score"] == min(scores)
    assert report["summary"]["median_hand_confidence"] == 1.0


# --------------------------------------------------------------------------- #
# Gate asymmetry: the export gate reads validator warnings only, so a spine
# warning that is not mirrored here never blocks anything.
# --------------------------------------------------------------------------- #
def test_spine_fatal_codes_are_mirrored_as_validator_warnings():
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg")],
        "hands": [_spine_hand(warnings=["board_zone_yield_zero"], source_images=["a.jpg"])],
    })
    assert "board_zone_yield_zero" in _codes(report)
    assert report["hands"][0]["warning_count"] >= 1
    assert report["hands"][0]["confidence_score"] <= 0.4


def test_hero_seat_mismatch_now_reaches_the_validator():
    """Latent before: hero_seat_mismatch lowered the exported confidence to
    exactly 0.80 -- not < 0.8, so it did not even earn the LOW_CONFIDENCE tag --
    and never reached the gate at all."""
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg")],
        "hands": [_spine_hand(warnings=["hero_seat_mismatch"], source_images=["a.jpg"])],
    })
    assert "hero_seat_mismatch" in _codes(report)
    assert report["hands"][0]["warning_count"] >= 1


def test_side_pot_unsupported_is_mirrored():
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg")],
        "hands": [_spine_hand(warnings=["side_pot_unsupported"], source_images=["a.jpg"])],
    })
    assert "side_pot_unsupported" in _codes(report)


def test_card_only_hand_without_spine_fields_is_unaffected():
    """The mirror lives in the spine-only branch, so card-only timelines (which
    carry neither players nor actions) still validate exactly as before."""
    report = validate_timeline(_timeline([_state(0.0, "a.jpg", board=["2C", "3D", "4H"])]))
    assert report["hands"][0]["warning_count"] == 0


# --------------------------------------------------------------------------- #
# Cross-field invariants on the hand summary.
# --------------------------------------------------------------------------- #
def test_empty_board_with_multiple_streets_warns():
    """The exact 06-21 hand-1 shape: board_cards="" on a hand whose own street
    summary lists a flop, a turn and a river. An empty board is a legal card
    count everywhere else, so nothing contradicted it."""
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=[])],
        "hands": [_spine_hand(
            board=[],
            streets=[{"street": s} for s in ("preflop", "flop", "turn", "river")],
            source_images=["a.jpg"],
        )],
    })
    assert "board_empty_but_streets_advanced" in _codes(report)
    assert report["hands"][0]["confidence_score"] <= 0.4


def test_empty_board_on_a_preflop_only_hand_does_not_warn():
    """False-positive guard: a hand that ended preflop legitimately has no board."""
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=[])],
        "hands": [_spine_hand(board=[], streets=[{"street": "preflop"}],
                              source_images=["a.jpg"])],
    })
    assert "board_empty_but_streets_advanced" not in _codes(report)


def test_empty_board_with_two_betting_rounds_on_one_street_warns():
    """The street summary is derived from the same community-card readings as the
    board, so when the row is lost the two collapse TOGETHER: the real 06-21 hand
    came out with board=[] and streets=['preflop'], which the check above cannot
    see. The actions are the independent witness -- with no board there can only
    have been one betting round, and a BET pill after a raise means the client had
    already moved to a street the reconstruction never recorded."""
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=[])],
        "hands": [_spine_hand(
            board=[], streets=[{"street": "preflop"}],
            actions=[
                {"street": "preflop", "seat": 5, "action_type": "raise", "amount": 3.6},
                {"street": "preflop", "seat": 0, "action_type": "call", "amount": 3.6},
                {"street": "preflop", "seat": 5, "action_type": "bet", "amount": 6.3},
            ],
            players=[{"seat": 0}, {"seat": 5}],
            source_images=["a.jpg"],
        )],
    })
    assert "board_empty_but_streets_advanced" in _codes(report)
    assert report["hands"][0]["confidence_score"] <= 0.4


def test_empty_board_with_a_check_after_paying_in_warns():
    """The second witness: calling the current bet closes the action to you. It
    reopens only if someone raises -- and then you are facing a bet and cannot
    check. Seat 3 calling and then checking on one street is impossible."""
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=[])],
        "hands": [_spine_hand(
            board=[], streets=[{"street": "preflop"}],
            actions=[
                {"street": "preflop", "seat": 3, "action_type": "call", "amount": 3.6},
                {"street": "preflop", "seat": 3, "action_type": "check"},
            ],
            players=[{"seat": 0}, {"seat": 3}],
            source_images=["a.jpg"],
        )],
    })
    assert "board_empty_but_streets_advanced" in _codes(report)


def test_a_second_betting_round_on_a_hand_that_has_a_board_does_not_warn():
    """The empty-board guard is load-bearing, not decoration. The same action
    shape on a hand whose board and streets are complete describes a much weaker
    claim -- an extra within-street round, which has real reconstructions behind
    it (13 of 164 measured hands) and no measurement supporting rejection."""
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=["2C", "3D", "4H"])],
        "hands": [_spine_hand(
            board=["2C", "3D", "4H"],
            streets=[{"street": "preflop"}, {"street": "flop"}],
            actions=[
                {"street": "flop", "seat": 2, "action_type": "bet", "amount": 8.3},
                {"street": "flop", "seat": 3, "action_type": "call", "amount": 8.3},
                {"street": "flop", "seat": 2, "action_type": "bet", "amount": 24.9},
            ],
            players=[{"seat": 2}, {"seat": 3}],
            source_images=["a.jpg"],
        )],
    })
    assert "board_empty_but_streets_advanced" not in _codes(report)


def test_all_actions_on_one_street_of_a_multi_street_hand_warns():
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=["2C", "3D", "4H"])],
        "hands": [_spine_hand(
            streets=[{"street": s} for s in ("preflop", "flop", "turn", "river")],
            actions=[{"street": "preflop", "seat": s % 4, "action_type": "call"}
                     for s in range(13)],
            players=[{"seat": s} for s in range(4)],
            source_images=["a.jpg"],
        )],
    })
    assert "actions_collapsed_to_one_street" in _codes(report)


def test_actions_spread_across_streets_do_not_warn():
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=["2C", "3D", "4H"])],
        "hands": [_spine_hand(
            streets=[{"street": s} for s in ("preflop", "flop")],
            actions=[{"street": "preflop", "action_type": "call"},
                     {"street": "flop", "action_type": "bet"}],
            source_images=["a.jpg"],
        )],
    })
    assert "actions_collapsed_to_one_street" not in _codes(report)


def test_single_street_hand_with_one_action_street_does_not_warn():
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=[])],
        "hands": [_spine_hand(
            board=[],
            streets=[{"street": "preflop"}],
            actions=[{"street": "preflop", "action_type": "fold"}],
            source_images=["a.jpg"],
        )],
    })
    assert _codes(report) == []


def test_preflop_all_in_with_a_runout_is_not_a_collapsed_hand():
    """False-positive guard, from a real 07-15 hand: hero shoves preflop, one
    villain calls all-in, a third folds, and the board runs out to the river with
    nobody able to act. Every action IS on preflop and that is correct."""
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=["2C", "3D", "4H"])],
        "hands": [_spine_hand(
            streets=[{"street": s} for s in ("preflop", "flop", "turn", "river")],
            actions=[
                {"street": "preflop", "seat": 0, "action_type": "all-in"},
                {"street": "preflop", "seat": 4, "action_type": "call"},
                {"street": "preflop", "seat": 5, "action_type": "fold"},
            ],
            players=[{"seat": s} for s in (0, 4, 5)],
            source_images=["a.jpg"],
        )],
    })
    assert "actions_collapsed_to_one_street" not in _codes(report)


def test_collapse_still_warns_when_two_players_could_have_acted():
    """The same shape with three live, non-all-in players IS the defect."""
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg", board=["2C", "3D", "4H"])],
        "hands": [_spine_hand(
            streets=[{"street": s} for s in ("preflop", "flop", "turn", "river")],
            actions=[{"street": "preflop", "seat": s, "action_type": "call"}
                     for s in (0, 4, 5)],
            players=[{"seat": s} for s in (0, 4, 5)],
            source_images=["a.jpg"],
        )],
    })
    assert "actions_collapsed_to_one_street" in _codes(report)


def test_call_before_any_bet_on_a_street_is_illegal():
    """A call needs something to call. On a post-preflop street a call emitted
    before that street's first bet/raise/all-in is a reconstruction contradicting
    itself -- the shape the green CALL/BET pill ambiguity took when the colour
    fallback forced it to "call". Five of the 21 hands reconstructed from the
    development corpus carried one, every one with spine warnings=[] and, before
    this rule, validator codes=[]."""
    hand = _spine_hand(actions=[
        {"street": "flop", "action_index": 1, "seat": 3, "action_type": "call",
         "amount": 11.5},
    ])
    report = validate_timeline({"states": [], "hands": [hand]})
    codes = [w["code"] for w in report["hands"][0]["warnings"]]
    assert "action_sequence_illegal" in codes
    detail = report["hands"][0]["warnings"][codes.index("action_sequence_illegal")]
    assert "call_with_nothing_to_call" in json.dumps(detail)


def test_call_after_a_bet_on_the_same_street_is_legal():
    """Negative control, plus the preflop exemption: the blinds are a standing
    bet, so a preflop call needs no preceding aggression in the action list."""
    ok = _spine_hand(actions=[
        {"street": "preflop", "action_index": 1, "seat": 3, "action_type": "call",
         "amount": 2.0},
        {"street": "flop", "action_index": 1, "seat": 0, "action_type": "bet",
         "amount": 5.0},
        {"street": "flop", "action_index": 2, "seat": 3, "action_type": "call",
         "amount": 5.0},
    ])
    report = validate_timeline({"states": [], "hands": [ok]})
    assert "action_sequence_illegal" not in [
        w["code"] for w in report["hands"][0]["warnings"]
    ]


def test_amounts_unknown_in_ledger_is_a_spine_fatal_code():
    """Phase 6. The reader now REFUSES rather than guessing, so a hand can carry a
    money action nothing could size, or a transition whose stack delta was
    unmeasurable. PLAN.md requires an incomplete sequence to be marked
    non-authoritative, and this is the code that does it: fatal here, routed into
    rejection_codes at the export, which makes derive_completion_status return
    "uncertain".

    Named LITERALLY rather than iterated over the frozenset -- a loop stops
    testing a code the moment someone deletes it, which is the regression this
    pins (see note 10 on amount_scale_implausible and anchor_unavailable)."""
    from cv_lab.scripts.eval.validate_yolo_card_timeline import (
        SPINE_FATAL_CODES,
        WARNING_SEVERITY,
    )

    assert "amounts_unknown_in_ledger" in SPINE_FATAL_CODES
    assert WARNING_SEVERITY["amounts_unknown_in_ledger"] >= 0.5, (
        "below _REJECTION_SEVERITY it becomes an acknowledgeable warning, and a "
        "hand whose money is unknown would be promotable by pressing a button")
    report = validate_timeline({
        "states": [_state(0.0, "a.jpg")],
        "hands": [_spine_hand(warnings=["amounts_unknown_in_ledger"],
                              source_images=["a.jpg"])],
    })
    assert "amounts_unknown_in_ledger" in _codes(report)
    assert report["hands"][0]["warning_count"] >= 1
