"""Fail-closed hand scoring against Phase 2 answer keys."""

from __future__ import annotations

from typing import Any

from poker_tracker.validation.schemas import SCORABLE_HAND_FACTS

# Tolerances in displayed BB units.
POT_TOL = 0.5
POT_TOL_FRAC = 0.02
NET_TOL = 1.0

# Street-aware amount tolerance: (absolute BB, fraction of the larger figure).
# Slack widens downstream because the figures themselves do — a river bet is
# read from a larger on-screen number than a preflop blind, so a fixed absolute
# tolerance is simultaneously too loose preflop and too tight on the river.
AMOUNT_TOLERANCE_BY_STREET: dict[str, tuple[float, float]] = {
    "preflop": (0.25, 0.02),
    "flop": (0.50, 0.03),
    "turn": (0.75, 0.04),
    "river": (1.00, 0.05),
    "showdown": (1.00, 0.05),
}
# Used only when a scored action carries a street outside the table above. It is
# the tightest row, so an unrecognized street can never buy extra slack.
AMOUNT_TOLERANCE_FALLBACK = (0.25, 0.02)

# PLAN hard gate: every locked completed hand has at most one noncritical error.
MAX_NONCRITICAL_PER_COMPLETED_HAND = 1

# Street ordering used for both action sequencing and ordering checks.
STREET_RANK: dict[str, int] = {
    "preflop": 0,
    "flop": 1,
    "turn": 2,
    "river": 3,
    "showdown": 4,
}

CRITICAL_CATEGORIES = frozenset(
    {
        "missed_hand",
        "spurious_hand",
        "split_hand",
        "merged_hand",
        "duplicate_hand",
        "completion_class",
        "hero_cards",
        "board_cards",
        "winner",
        "result",
        "illegal_action",
        # Material action-line failures on a complete GT hand.
        "missing_action",
        "spurious_action",
        "action_amount",
        "action_order",
        # A fact that is neither annotated nor declared unobservable cannot be
        # scored at all, so the gate treats the corpus itself as failing.
        "answer_key_incomplete",
    }
)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _amounts_match(expected: float, predicted: float | None, *, street: Any) -> bool:
    """Street-aware amount comparison.

    A missing prediction never matches: the caller only reaches this when the
    answer key supplied a figure, so ``None`` on the prediction side is a real
    failure to read it, not an excused absence.
    """
    if predicted is None:
        return False
    abs_tol, frac_tol = AMOUNT_TOLERANCE_BY_STREET.get(
        str(street), AMOUNT_TOLERANCE_FALLBACK
    )
    return abs(expected - predicted) <= max(
        abs_tol, frac_tol * max(abs(expected), abs(predicted))
    )


# Actions that structurally carry no amount. Their silence is not an annotation
# gap, so they are never scored for amount and never reported as excluded.
NO_AMOUNT_ACTIONS = frozenset({"fold", "check", "show"})

# Severity for each scored hand fact when it disagrees with the prediction.
FACT_SEVERITY: dict[str, str] = {
    "hero_cards": "critical",
    "final_board": "critical",
    "winner_seat": "critical",
    "result": "critical",
    "dealer_seat": "noncritical",
    "final_pot": "noncritical",
    "hero_net": "noncritical",
}

if set(FACT_SEVERITY) != set(SCORABLE_HAND_FACTS):
    # A fact the answer-key schema demands but the evaluator never scores would
    # be annotated by hand and then quietly ignored, which is worse than not
    # collecting it. Refuse to import rather than score an incomplete rubric.
    raise RuntimeError(
        "FACT_SEVERITY and SCORABLE_HAND_FACTS disagree: "
        f"{sorted(set(FACT_SEVERITY) ^ set(SCORABLE_HAND_FACTS))}"
    )


def _declared_unobservable(gt: dict[str, Any]) -> set[str]:
    raw = gt.get("unobservable")
    if not isinstance(raw, list):
        return set()
    return {fact for fact in raw if isinstance(fact, str)}


def _gt_action_sequence(gt: dict[str, Any]) -> list[dict[str, Any]]:
    """Certain, observable answer-key actions in play order.

    Ordering is scored, so the sequence must come from the annotated
    ``street``/``order`` pair rather than from however the JSON happens to be
    laid out on disk.
    """
    actions = [
        action
        for action in (gt.get("actions") or [])
        if isinstance(action, dict)
        and action.get("certain") is True
        and action.get("observable", True) is True
    ]
    return sorted(
        actions,
        key=lambda action: (
            STREET_RANK.get(str(action.get("street")), len(STREET_RANK)),
            action.get("order") if isinstance(action.get("order"), int) else 0,
        ),
    )


def _canon_action(atype: str) -> str:
    return {"all_in": "all-in"}.get(atype, atype)


def _gt_window(hand: dict[str, Any]) -> tuple[float, float]:
    t0 = hand.get("t_first", hand.get("t_first_seen"))
    t1 = hand.get("t_last", hand.get("t_last_seen"))
    if not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
        raise ValueError("answer-key hand missing t_first/t_last")
    return float(t0), float(t1)


def _gt_completion(hand: dict[str, Any]) -> str:
    if "completion_class" in hand:
        return str(hand["completion_class"])
    if hand.get("partial_start") or hand.get("partial_end"):
        return "partial"
    return "complete"


def _pred_completion(hand: dict[str, Any]) -> str:
    if hand.get("complete") is True:
        return "complete"
    if hand.get("partial_start") or hand.get("partial_end"):
        return "partial"
    terminal = hand.get("terminal_event")
    if terminal == "unobserved":
        return "uncertain"
    return "uncertain"


def _error(category: str, detail: str, *, severity: str) -> dict[str, str]:
    return {"category": category, "detail": detail, "severity": severity}


def _pred_illegal_actions(pred: dict[str, Any]) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    folded: set[int] = set()
    for index, action in enumerate(pred.get("actions") or []):
        if not isinstance(action, dict):
            continue
        seat = action.get("seat")
        kind = _canon_action(str(action.get("action_type", action.get("action"))))
        if not isinstance(seat, int):
            continue
        if seat in folded and kind not in {"show", "win"}:
            errors.append(
                _error(
                    "illegal_action",
                    f"pred action[{index}] seat {seat} acts after folding ({kind})",
                    severity="critical",
                )
            )
        if kind == "fold":
            folded.add(seat)
    return errors


def _score_hand(
    gt: dict[str, Any],
    pred: dict[str, Any] | None,
    *,
    fragments: int,
    merged: bool,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    excluded_facts: list[str] = []
    if pred is None:
        errors.append(
            _error("missed_hand", "no overlapping prediction", severity="critical")
        )
        return {
            "errors": errors,
            "n_errors": len(errors),
            "n_critical": len(errors),
            "n_noncritical": 0,
            "excluded_facts": excluded_facts,
            "predicted_completion_class": None,
        }
    if fragments > 1:
        errors.append(
            _error(
                "split_hand",
                f"prediction fragmented across {fragments} hands",
                severity="critical",
            )
        )
    if merged:
        errors.append(
            _error(
                "merged_hand",
                "one prediction window claimed by multiple ground-truth hands",
                severity="critical",
            )
        )
    if _gt_completion(gt) != _pred_completion(pred):
        errors.append(
            _error(
                "completion_class",
                f"gt={_gt_completion(gt)} pred={_pred_completion(pred)}",
                severity="critical",
            )
        )

    unobservable = _declared_unobservable(gt)

    def is_scored(fact: str) -> bool:
        """Whether this fact can be scored, recording why when it cannot.

        A fact the annotator named unobservable is excluded and reported as a
        skipped check. A fact that is simply absent is an unfinished answer key,
        which fails the gate rather than passing silently.
        """
        if fact in unobservable:
            excluded_facts.append(fact)
            return False
        if gt.get(fact) is None:
            errors.append(
                _error(
                    "answer_key_incomplete",
                    f"{fact} is neither annotated nor declared unobservable",
                    severity="critical",
                )
            )
            return False
        return True

    if is_scored("hero_cards"):
        gt_hero = gt.get("hero_cards")
        pred_hero = pred.get("hero")
        if list(pred_hero or []) != list(gt_hero or []):
            errors.append(
                _error("hero_cards", f"gt={gt_hero} pred={pred_hero}", severity="critical")
            )
    if is_scored("final_board"):
        gt_board = gt.get("final_board")
        pred_board = pred.get("board")
        if list(pred_board or []) != list(gt_board or []):
            errors.append(
                _error("board_cards", f"gt={gt_board} pred={pred_board}", severity="critical")
            )
    if is_scored("dealer_seat") and pred.get("dealer_seat") != gt.get("dealer_seat"):
        errors.append(
            _error(
                "dealer_seat",
                f"gt={gt.get('dealer_seat')} pred={pred.get('dealer_seat')}",
                severity=FACT_SEVERITY["dealer_seat"],
            )
        )
    if is_scored("winner_seat") and pred.get("winner_seat") != gt.get("winner_seat"):
        errors.append(
            _error(
                "winner",
                f"gt={gt.get('winner_seat')} pred={pred.get('winner_seat')}",
                severity="critical",
            )
        )
    if is_scored("result") and pred.get("result") != gt.get("result"):
        errors.append(
            _error(
                "result",
                f"gt={gt.get('result')!r} pred={pred.get('result')!r}",
                severity="critical",
            )
        )
    if is_scored("final_pot"):
        gt_pot = float(gt["final_pot"])
        pred_pot = pred.get("pot")
        if not isinstance(pred_pot, (int, float)) or abs(
            gt_pot - float(pred_pot)
        ) > max(POT_TOL, POT_TOL_FRAC * abs(gt_pot)):
            errors.append(
                _error(
                    "final_pot",
                    f"gt={gt_pot} pred={pred_pot}",
                    severity=FACT_SEVERITY["final_pot"],
                )
            )
    if is_scored("hero_net"):
        gt_net = float(gt["hero_net"])
        pred_net = pred.get("hero_bb_won")
        if not isinstance(pred_net, (int, float)) or abs(gt_net - float(pred_net)) > NET_TOL:
            errors.append(
                _error(
                    "hero_net",
                    f"gt={gt_net} pred={pred_net}",
                    severity=FACT_SEVERITY["hero_net"],
                )
            )

    errors.extend(_pred_illegal_actions(pred))

    actions_complete = gt.get("actions_complete") is True
    # Material action mistakes are critical on complete GT lines; pot/net/dealer stay
    # noncritical and are bounded by MAX_NONCRITICAL_PER_COMPLETED_HAND.
    action_severity = "critical" if actions_complete else "noncritical"
    if actions_complete:
        gt_actions = _gt_action_sequence(gt)
        pred_actions = [a for a in (pred.get("actions") or []) if isinstance(a, dict)]
        available = list(enumerate(pred_actions))
        matched_positions: list[tuple[tuple[str, int, str], int]] = []
        for action in gt_actions:
            gt_seat = action.get("seat")
            if not isinstance(gt_seat, int):
                errors.append(
                    _error(
                        "illegal_action",
                        f"answer-key action has non-integer seat {gt_seat!r}",
                        severity="critical",
                    )
                )
                continue
            street = str(action.get("street"))
            want = (street, gt_seat, _canon_action(str(action.get("action"))))
            match_index = None
            for index, (_, candidate) in enumerate(available):
                seat = candidate.get("seat")
                if not isinstance(seat, int):
                    continue
                kind = candidate.get("action_type", candidate.get("action"))
                got = (str(candidate.get("street")), seat, _canon_action(str(kind)))
                if got == want:
                    match_index = index
                    break
            if match_index is None:
                errors.append(
                    _error("missing_action", f"missing {want}", severity=action_severity)
                )
                continue
            position, matched = available.pop(match_index)
            matched_positions.append((want, position))
            _score_action_amount(
                action,
                matched,
                want=want,
                street=street,
                severity=action_severity,
                errors=errors,
                excluded_facts=excluded_facts,
            )
        _score_action_ordering(
            matched_positions, severity=action_severity, errors=errors
        )
        for _, leftover in available:
            seat = leftover.get("seat")
            if not isinstance(seat, int):
                errors.append(
                    _error(
                        "illegal_action",
                        f"prediction action has non-integer seat {seat!r}",
                        severity="critical",
                    )
                )
                continue
            kind = leftover.get("action_type", leftover.get("action"))
            errors.append(
                _error(
                    "spurious_action",
                    f"extra ({leftover.get('street')}, {seat}, {kind})",
                    severity=action_severity,
                )
            )

    n_critical = sum(1 for e in errors if e.get("severity") == "critical")
    n_noncritical = len(errors) - n_critical
    return {
        "errors": errors,
        "n_errors": len(errors),
        "n_critical": n_critical,
        "n_noncritical": n_noncritical,
        "excluded_facts": excluded_facts,
        "predicted_completion_class": _pred_completion(pred),
    }


def _score_action_amount(
    action: dict[str, Any],
    matched: dict[str, Any],
    *,
    want: tuple[str, int, str],
    street: str,
    severity: str,
    errors: list[dict[str, str]],
    excluded_facts: list[str],
) -> None:
    """Compare one matched action's amount under street-aware tolerance.

    Three cases are kept apart deliberately. A fold or check has no amount by
    construction. An amount the annotator could not read is declared through
    ``amount_semantics: "unknown"`` and reported as a skipped check. Anything
    else with no annotated figure is an unfinished answer key, which fails
    rather than matching whatever the prediction happened to say.
    """
    if want[2] in NO_AMOUNT_ACTIONS:
        return
    semantics = action.get("amount_semantics", "incremental")
    if semantics == "unknown":
        excluded_facts.append(f"action_amount:{want[0]}:{want[1]}:{want[2]}")
        return
    expected = action.get("amount")
    if not isinstance(expected, (int, float)):
        errors.append(
            _error(
                "answer_key_incomplete",
                f"{want} has no annotated amount and no amount_semantics=unknown",
                severity="critical",
            )
        )
        return
    if not _amounts_match(float(expected), matched.get("amount"), street=street):
        errors.append(
            _error(
                "action_amount",
                f"{want} gt_amount={expected} pred_amount={matched.get('amount')}",
                severity=severity,
            )
        )


def _score_action_ordering(
    matched_positions: list[tuple[tuple[str, int, str], int]],
    *,
    severity: str,
    errors: list[dict[str, str]],
) -> None:
    """Flag matched actions the prediction placed out of answer-key order.

    Matching is by identity rather than position, so a prediction that contains
    every right action in the wrong sequence otherwise scores perfectly. The
    answer key's ``street``/``order`` pair is the reference sequence.
    """
    previous_position = -1
    previous_want: tuple[str, int, str] | None = None
    for want, position in matched_positions:
        if position < previous_position and previous_want is not None:
            errors.append(
                _error(
                    "action_order",
                    f"{want} appears before {previous_want} in the prediction",
                    severity=severity,
                )
            )
        previous_position = max(previous_position, position)
        previous_want = want


def evaluate_answer_key_against_timeline(
    truth: dict[str, Any],
    timeline: dict[str, Any],
) -> dict[str, Any]:
    """Score a timeline against a Phase 2 answer key. Fail closed on emptiness."""
    gt_hands = truth.get("hands")
    pred_hands = timeline.get("hands")
    if not isinstance(gt_hands, list) or not gt_hands:
        return {
            "ok": False,
            "fail_closed": "empty_truth",
            "hands_scored": 0,
            "total_errors": 0,
            "critical_errors": 0,
            "noncritical_budget_violations": 0,
            "spurious_predicted_hands": [],
            "per_hand": [],
        }
    if not isinstance(pred_hands, list):
        return {
            "ok": False,
            "fail_closed": "missing_predictions",
            "hands_scored": 0,
            "total_errors": 0,
            "critical_errors": 0,
            "noncritical_budget_violations": 0,
            "spurious_predicted_hands": [],
            "per_hand": [],
        }

    # First pass: greedy best-overlap matches, then mark merges.
    matches: list[tuple[dict[str, Any], int | None, int]] = []
    pred_claimants: dict[int, int] = {}
    for gt in gt_hands:
        if not isinstance(gt, dict):
            continue
        t0, t1 = _gt_window(gt)
        overlaps = [
            (i, _overlap(t0, t1, float(p.get("t_start", -1)), float(p.get("t_end", -1))))
            for i, p in enumerate(pred_hands)
            if isinstance(p, dict)
        ]
        fragments = [i for i, ov in overlaps if ov > 0]
        best = max(overlaps, key=lambda item: item[1]) if overlaps else (None, 0.0)
        pred_index = best[0] if best[0] is not None and best[1] > 0 else None
        if pred_index is not None:
            pred_claimants[pred_index] = pred_claimants.get(pred_index, 0) + 1
        matches.append((gt, pred_index, len(fragments)))

    # Duplicate predicted hand_number values are critical segmentation failures.
    seen_numbers: dict[Any, int] = {}
    duplicate_pred_indexes: set[int] = set()
    for index, pred in enumerate(pred_hands):
        if not isinstance(pred, dict):
            continue
        number = pred.get("hand_number")
        if number in seen_numbers:
            duplicate_pred_indexes.add(index)
            duplicate_pred_indexes.add(seen_numbers[number])
        elif number is not None:
            seen_numbers[number] = index

    per_hand: list[dict[str, Any]] = []
    used_pred: set[int] = set()
    budget_violations = 0
    for gt, pred_index, fragment_count in matches:
        t0, t1 = _gt_window(gt)
        pred = pred_hands[pred_index] if pred_index is not None else None
        if pred_index is not None:
            used_pred.add(pred_index)
        merged = pred_index is not None and pred_claimants.get(pred_index, 0) > 1
        scored = _score_hand(gt, pred, fragments=fragment_count, merged=merged)
        if pred_index in duplicate_pred_indexes:
            scored["errors"].append(
                _error(
                    "duplicate_hand",
                    f"duplicate prediction hand_number={pred.get('hand_number') if isinstance(pred, dict) else None}",
                    severity="critical",
                )
            )
            scored["n_critical"] += 1
            scored["n_errors"] += 1
        completion = _gt_completion(gt)
        if (
            completion == "complete"
            and scored["n_noncritical"] > MAX_NONCRITICAL_PER_COMPLETED_HAND
        ):
            budget_violations += 1
            scored["errors"].append(
                _error(
                    "noncritical_budget",
                    (
                        f"completed hand has {scored['n_noncritical']} noncritical "
                        f"errors; max allowed is {MAX_NONCRITICAL_PER_COMPLETED_HAND}"
                    ),
                    severity="critical",
                )
            )
            scored["n_critical"] += 1
            scored["n_errors"] += 1
        per_hand.append(
            {
                "hand_id": gt.get("hand_id"),
                "t": [t0, t1],
                "completion_class": completion,
                **scored,
            }
        )

    spurious_entries = [
        (i, p) for i, p in enumerate(pred_hands) if i not in used_pred and isinstance(p, dict)
    ]
    spurious = [p.get("hand_number") for _, p in spurious_entries]
    for _, pred in spurious_entries:
        per_hand.append(
            {
                "hand_id": None,
                "t": None,
                "completion_class": None,
                "predicted_completion_class": _pred_completion(pred),
                "excluded_facts": [],
                "errors": [
                    _error(
                        "spurious_hand",
                        f"unmatched prediction hand_number={pred.get('hand_number')}",
                        severity="critical",
                    )
                ],
                "n_errors": 1,
                "n_critical": 1,
                "n_noncritical": 0,
            }
        )

    completion = completion_metrics(per_hand)
    total_errors = sum(h["n_errors"] for h in per_hand)
    critical_errors = sum(h["n_critical"] for h in per_hand)
    hands_scored = len([h for h in per_hand if h.get("hand_id") is not None])
    excluded_facts = sorted(
        {fact for h in per_hand for fact in (h.get("excluded_facts") or [])}
    )
    ok = (
        hands_scored > 0
        and critical_errors == 0
        and budget_violations == 0
        and not spurious
        and completion["precision"] in (None, 1.0)
        and completion["recall"] in (None, 1.0)
    )
    fail_closed = None
    if hands_scored <= 0:
        fail_closed = "zero_scored_hands"
    elif not ok:
        fail_closed = "threshold_failures"
    return {
        "ok": ok,
        "fail_closed": fail_closed,
        "hands_scored": hands_scored,
        "total_errors": total_errors,
        "critical_errors": critical_errors,
        "noncritical_budget_violations": budget_violations,
        "spurious_predicted_hands": spurious,
        "completion": completion,
        "excluded_facts": excluded_facts,
        "per_hand": per_hand,
    }


def completion_metrics(per_hand: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision and recall for the completed-versus-partial decision.

    ``complete`` is the positive class, because calling a partial hand complete
    is what silently admits an unfinished hand into study. Every answer-key hand
    contributes, including the partial first and last hands of a recording: a
    missed hand counts as a false negative rather than dropping out of the
    denominator, and a spurious predicted hand called complete counts as a false
    positive. Omitting either is how a recording's own edges disappear from the
    metric that exists to police them.
    """
    true_positive = false_positive = false_negative = true_negative = 0
    for hand in per_hand:
        expected = hand.get("completion_class")
        predicted = hand.get("predicted_completion_class")
        if expected is None:
            # Spurious prediction: no answer-key hand behind it.
            if predicted == "complete":
                false_positive += 1
            continue
        if expected == "complete":
            if predicted == "complete":
                true_positive += 1
            else:
                # Includes a missed hand, whose predicted class is None.
                false_negative += 1
        else:
            if predicted == "complete":
                false_positive += 1
            else:
                true_negative += 1
    precision_den = true_positive + false_positive
    recall_den = true_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": (true_positive / precision_den) if precision_den else None,
        "recall": (true_positive / recall_den) if recall_den else None,
        "scored_hands": true_positive + false_negative + true_negative,
    }
