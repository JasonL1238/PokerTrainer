"""Phase 3 evaluator corrections.

Each test here pins one of the PLAN "Evaluator corrections" bullets that the
original scaffold satisfied only by accident: explicit observability, street-aware
amount slack, scored action ordering, and completion precision/recall that keeps
partial edge hands in the denominator.
"""

from __future__ import annotations

import pytest

from poker_tracker.release_gate.evaluate import (
    AMOUNT_TOLERANCE_BY_STREET,
    completion_metrics,
    evaluate_answer_key_against_timeline,
)

ALL_FACTS = ("hero_cards", "final_board", "dealer_seat", "winner_seat", "result",
             "final_pot", "hero_net")


def _truth_hand(**overrides):
    hand = {
        "hand_id": "h1",
        "t_first": 0.0,
        "t_last": 10.0,
        "completion_class": "complete",
        "terminal_event": "showdown",
        "partial_start": False,
        "partial_end": False,
        "actions_complete": True,
        "hero_cards": ["As", "Kd"],
        "final_board": ["2c", "7d", "9h", "Ts", "Jc"],
        "dealer_seat": 1,
        "winner_seat": 0,
        "result": "Hero wins",
        "final_pot": 16.0,
        "hero_net": 8.0,
        "dealt_in_seats": [0, 1],
        "actions": [],
        "unobservable": [],
    }
    hand.update(overrides)
    return hand


def _pred_hand(**overrides):
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 10.0,
        "complete": True,
        "hero": ["As", "Kd"],
        "board": ["2c", "7d", "9h", "Ts", "Jc"],
        "dealer_seat": 1,
        "winner_seat": 0,
        "result": "Hero wins",
        "pot": 16.0,
        "hero_bb_won": 8.0,
        "actions": [],
    }
    hand.update(overrides)
    return hand


def _run(truth_hands, pred_hands):
    return evaluate_answer_key_against_timeline(
        {"hands": truth_hands}, {"hands": pred_hands}
    )


def _categories(report):
    return {
        error["category"]
        for hand in report["per_hand"]
        for error in hand["errors"]
    }


# --- Explicit observability -------------------------------------------------


@pytest.mark.parametrize("fact", ALL_FACTS)
def test_absent_fact_is_an_unfinished_answer_key_not_a_free_pass(fact):
    """A fact nobody annotated must fail rather than score as agreement.

    Under the old implicit-``None`` rule an unfinished answer key scored exactly
    like a perfect one, which is the failure mode that lets a wrong prediction
    be silently accepted.
    """
    hand = _truth_hand()
    hand[fact] = None
    report = _run([hand], [_pred_hand()])
    assert report["ok"] is False
    assert "answer_key_incomplete" in _categories(report)


@pytest.mark.parametrize("fact", ALL_FACTS)
def test_declared_unobservable_fact_is_excluded_and_reported(fact):
    hand = _truth_hand()
    hand[fact] = None
    hand["unobservable"] = [fact]
    report = _run([hand], [_pred_hand()])
    assert report["ok"] is True
    assert fact in report["excluded_facts"]


def test_declared_unobservable_fact_cannot_hide_a_wrong_prediction_elsewhere():
    """Excluding one fact must not excuse the rest of the hand."""
    hand = _truth_hand(final_pot=None, unobservable=["final_pot"])
    report = _run([hand], [_pred_hand(winner_seat=1)])
    assert report["ok"] is False
    assert "winner" in _categories(report)


# --- Street-aware amount tolerance -----------------------------------------


def _action_hand(street, gt_amount, pred_amount):
    truth = _truth_hand(
        actions=[
            {
                "street": street,
                "order": 1,
                "seat": 0,
                "action": "bet",
                "amount": gt_amount,
                "amount_semantics": "incremental",
                "certain": True,
                "observable": True,
            }
        ]
    )
    pred = _pred_hand(
        actions=[{"street": street, "seat": 0, "action_type": "bet", "amount": pred_amount}]
    )
    return truth, pred


def test_preflop_tolerance_is_tighter_than_river_tolerance():
    """The same absolute error passes on the river and fails preflop."""
    assert AMOUNT_TOLERANCE_BY_STREET["preflop"][0] < AMOUNT_TOLERANCE_BY_STREET["river"][0]
    drift = 0.4  # inside the river's 1.0 absolute slack, outside preflop's 0.25

    truth, pred = _action_hand("river", 10.0, 10.0 + drift)
    assert _run([truth], [pred])["ok"] is True

    truth, pred = _action_hand("preflop", 3.0, 3.0 + drift)
    report = _run([truth], [pred])
    assert report["ok"] is False
    assert "action_amount" in _categories(report)


def test_unrecognized_street_gets_the_tightest_tolerance():
    """An unknown street must not buy more slack than any real one."""
    truth, pred = _action_hand("fifth_street", 10.0, 10.4)
    report = _run([truth], [pred])
    assert report["ok"] is False
    assert "action_amount" in _categories(report)


def test_unknown_amount_semantics_excludes_that_action_explicitly():
    truth, pred = _action_hand("flop", None, 5.0)
    truth["actions"][0]["amount_semantics"] = "unknown"
    report = _run([truth], [pred])
    assert report["ok"] is True
    assert any(f.startswith("action_amount:") for f in report["excluded_facts"])


def test_money_action_without_amount_or_unknown_semantics_fails():
    truth, pred = _action_hand("flop", None, 5.0)
    report = _run([truth], [pred])
    assert report["ok"] is False
    assert "answer_key_incomplete" in _categories(report)


def test_fold_and_check_are_not_scored_for_amount():
    truth = _truth_hand(
        actions=[
            {
                "street": "flop",
                "order": 1,
                "seat": 0,
                "action": "check",
                "amount": None,
                "amount_semantics": "incremental",
                "certain": True,
                "observable": True,
            }
        ]
    )
    pred = _pred_hand(
        actions=[{"street": "flop", "seat": 0, "action_type": "check", "amount": None}]
    )
    report = _run([truth], [pred])
    assert report["ok"] is True
    # A structural absence is not a "skipped check" — nothing to report.
    assert report["excluded_facts"] == []


# --- Action ordering --------------------------------------------------------


def _two_action_truth():
    return _truth_hand(
        actions=[
            {
                "street": "flop",
                "order": 1,
                "seat": 0,
                "action": "bet",
                "amount": 5.0,
                "amount_semantics": "incremental",
                "certain": True,
                "observable": True,
            },
            {
                "street": "flop",
                "order": 2,
                "seat": 1,
                "action": "call",
                "amount": 5.0,
                "amount_semantics": "incremental",
                "certain": True,
                "observable": True,
            },
        ]
    )


def test_correct_order_passes():
    pred = _pred_hand(
        actions=[
            {"street": "flop", "seat": 0, "action_type": "bet", "amount": 5.0},
            {"street": "flop", "seat": 1, "action_type": "call", "amount": 5.0},
        ]
    )
    assert _run([_two_action_truth()], [pred])["ok"] is True


def test_reversed_order_is_scored_as_an_error():
    """Every right action in the wrong sequence must not score perfectly."""
    pred = _pred_hand(
        actions=[
            {"street": "flop", "seat": 1, "action_type": "call", "amount": 5.0},
            {"street": "flop", "seat": 0, "action_type": "bet", "amount": 5.0},
        ]
    )
    report = _run([_two_action_truth()], [pred])
    assert report["ok"] is False
    assert "action_order" in _categories(report)


def test_answer_key_order_drives_the_comparison_not_json_layout():
    """Reordering the answer-key list without changing street/order is a no-op."""
    truth = _two_action_truth()
    truth["actions"] = list(reversed(truth["actions"]))
    pred = _pred_hand(
        actions=[
            {"street": "flop", "seat": 0, "action_type": "bet", "amount": 5.0},
            {"street": "flop", "seat": 1, "action_type": "call", "amount": 5.0},
        ]
    )
    assert _run([truth], [pred])["ok"] is True


# --- Completion precision and recall ----------------------------------------


def test_partial_edge_hands_stay_in_the_denominator():
    """A missed complete hand is a false negative, not an omitted row."""
    metrics = completion_metrics(
        [
            {"completion_class": "complete", "predicted_completion_class": "complete"},
            {"completion_class": "complete", "predicted_completion_class": None},
            {"completion_class": "partial", "predicted_completion_class": "partial"},
        ]
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["scored_hands"] == 3


def test_spurious_hand_called_complete_is_a_false_positive():
    metrics = completion_metrics(
        [
            {"completion_class": "complete", "predicted_completion_class": "complete"},
            {"completion_class": None, "predicted_completion_class": "complete"},
        ]
    )
    assert metrics["false_positive"] == 1
    assert metrics["precision"] == 0.5


def test_partial_called_complete_fails_the_gate_through_precision():
    truth = _truth_hand(
        completion_class="partial",
        partial_end=True,
        actions_complete=False,
        winner_seat=None,
        final_pot=None,
        result=None,
        unobservable=["winner_seat", "final_pot", "result"],
    )
    report = _run([truth], [_pred_hand(complete=True)])
    assert report["ok"] is False
    assert report["completion"]["false_positive"] == 1
    assert report["completion"]["precision"] == 0.0


def test_metrics_are_none_rather_than_one_when_nothing_was_scored():
    """An empty denominator must not read as a perfect score."""
    metrics = completion_metrics([])
    assert metrics["precision"] is None
    assert metrics["recall"] is None


# --- Adversarial round 1 findings (Adversary A) ------------------------------


def test_uncertain_actions_are_reported_as_skipped_not_dropped_silently():
    """An answer key could mark every action uncertain and score clean.

    That is the same failure the hand-fact ``unobservable`` rule prevents: the
    check did not happen, so the report has to say so.
    """
    truth = _truth_hand(
        actions=[
            {
                "street": "flop",
                "order": 1,
                "seat": 0,
                "action": "bet",
                "amount": 5.0,
                "amount_semantics": "incremental",
                "certain": False,
                "observable": True,
            },
            {
                "street": "flop",
                "order": 2,
                "seat": 1,
                "action": "call",
                "amount": 5.0,
                "amount_semantics": "incremental",
                "certain": True,
                "observable": False,
            },
        ]
    )
    report = _run([truth], [_pred_hand(actions=[])])
    excluded = report["excluded_facts"]
    assert any(f.startswith("action:") and f.endswith("uncertain") for f in excluded)
    assert any(f.startswith("action:") and f.endswith("unobservable") for f in excluded)


def test_a_partial_hand_says_its_action_line_went_unscored():
    truth = _truth_hand(
        completion_class="partial",
        partial_end=True,
        actions_complete=False,
        winner_seat=None,
        final_pot=None,
        result=None,
        unobservable=["winner_seat", "final_pot", "result"],
    )
    pred = _pred_hand(
        complete=False,
        actions=[{"street": "river", "seat": 0, "action_type": "bet", "amount": 400.0}],
    )
    report = _run([truth], [pred])
    assert "action_line:answer key declares actions incomplete" in report[
        "excluded_facts"
    ]


def test_unknown_semantics_carrying_an_amount_is_rejected_by_the_validator():
    """Excluding a check the key can actually answer is a contradiction."""
    from poker_tracker.validation.truth_validate import validate_answer_key

    document = {
        "schema_version": 2,
        "case_id": "c",
        "annotation": {"passes": 2, "adjudicated": True},
        "hands": [
            {
                **_truth_hand(
                    actions=[
                        {
                            "street": "river",
                            "order": 1,
                            "seat": 0,
                            "action": "bet",
                            "amount": 50.0,
                            "amount_semantics": "unknown",
                            "certain": True,
                            "observable": True,
                        }
                    ]
                ),
                "terminal_event": "showdown",
            }
        ],
    }
    result = validate_answer_key(document)
    assert result.ok is False
    assert any("amount_semantics=unknown" in issue.message for issue in result.issues)
