"""Fail-closed hand scoring against Phase 2 answer keys."""

from __future__ import annotations

from typing import Any

# Tolerances in displayed BB units (street-aware amount slack is deferred; these
# are explicit so missing predicted amounts never match by accident).
POT_TOL = 0.5
POT_TOL_FRAC = 0.02
NET_TOL = 1.0
AMT_TOL = 0.5
AMT_TOL_FRAC = 0.05
# PLAN hard gate: every locked completed hand has at most one noncritical error.
MAX_NONCRITICAL_PER_COMPLETED_HAND = 1

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
    }
)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _amounts_match(expected: float | None, predicted: float | None) -> bool:
    if expected is None:
        return True
    if predicted is None:
        return False
    return abs(expected - predicted) <= max(
        AMT_TOL, AMT_TOL_FRAC * max(abs(expected), abs(predicted))
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
    if pred is None:
        errors.append(
            _error("missed_hand", "no overlapping prediction", severity="critical")
        )
        return {
            "errors": errors,
            "n_errors": len(errors),
            "n_critical": len(errors),
            "n_noncritical": 0,
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
    gt_hero = gt.get("hero_cards")
    pred_hero = pred.get("hero")
    if gt_hero is not None and list(pred_hero or []) != list(gt_hero):
        errors.append(
            _error(
                "hero_cards",
                f"gt={gt_hero} pred={pred_hero}",
                severity="critical",
            )
        )
    gt_board = gt.get("final_board")
    pred_board = pred.get("board")
    if isinstance(gt_board, list) and list(pred_board or []) != list(gt_board):
        errors.append(
            _error(
                "board_cards",
                f"gt={gt_board} pred={pred_board}",
                severity="critical",
            )
        )
    if gt.get("dealer_seat") is not None and pred.get("dealer_seat") != gt.get("dealer_seat"):
        errors.append(
            _error(
                "dealer_seat",
                f"gt={gt.get('dealer_seat')} pred={pred.get('dealer_seat')}",
                severity="noncritical",
            )
        )
    if gt.get("winner_seat") is not None and pred.get("winner_seat") != gt.get("winner_seat"):
        errors.append(
            _error(
                "winner",
                f"gt={gt.get('winner_seat')} pred={pred.get('winner_seat')}",
                severity="critical",
            )
        )
    gt_result = gt.get("result")
    if isinstance(gt_result, str) and gt_result.strip():
        pred_result = pred.get("result")
        if pred_result != gt_result:
            errors.append(
                _error(
                    "result",
                    f"gt={gt_result!r} pred={pred_result!r}",
                    severity="critical",
                )
            )

    gt_pot = gt.get("final_pot")
    pred_pot = pred.get("pot")
    if isinstance(gt_pot, (int, float)):
        if not isinstance(pred_pot, (int, float)):
            errors.append(
                _error("final_pot", f"gt={gt_pot} pred={pred_pot}", severity="noncritical")
            )
        elif abs(float(gt_pot) - float(pred_pot)) > max(
            POT_TOL, POT_TOL_FRAC * abs(float(gt_pot))
        ):
            errors.append(
                _error("final_pot", f"gt={gt_pot} pred={pred_pot}", severity="noncritical")
            )
    gt_net = gt.get("hero_net")
    pred_net = pred.get("hero_bb_won")
    if isinstance(gt_net, (int, float)):
        if not isinstance(pred_net, (int, float)) or abs(float(gt_net) - float(pred_net)) > NET_TOL:
            errors.append(
                _error("hero_net", f"gt={gt_net} pred={pred_net}", severity="noncritical")
            )

    errors.extend(_pred_illegal_actions(pred))

    actions_complete = gt.get("actions_complete") is True
    # Material action mistakes are critical on complete GT lines; pot/net/dealer stay
    # noncritical and are bounded by MAX_NONCRITICAL_PER_COMPLETED_HAND.
    action_severity = "critical" if actions_complete else "noncritical"
    if actions_complete:
        gt_actions = [
            a
            for a in (gt.get("actions") or [])
            if isinstance(a, dict)
            and a.get("certain") is True
            and a.get("observable", True) is True
        ]
        pred_actions = [a for a in (pred.get("actions") or []) if isinstance(a, dict)]
        remaining = list(pred_actions)
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
            want = (
                str(action.get("street")),
                gt_seat,
                _canon_action(str(action.get("action"))),
            )
            match_index = None
            for index, candidate in enumerate(remaining):
                seat = candidate.get("seat")
                if not isinstance(seat, int):
                    continue
                kind = candidate.get("action_type", candidate.get("action"))
                got = (
                    str(candidate.get("street")),
                    seat,
                    _canon_action(str(kind)),
                )
                if got == want:
                    match_index = index
                    break
            if match_index is None:
                errors.append(
                    _error("missing_action", f"missing {want}", severity=action_severity)
                )
                continue
            matched = remaining.pop(match_index)
            if not _amounts_match(action.get("amount"), matched.get("amount")):
                errors.append(
                    _error(
                        "action_amount",
                        (
                            f"{want} gt_amount={action.get('amount')} "
                            f"pred_amount={matched.get('amount')}"
                        ),
                        severity=action_severity,
                    )
                )
        for leftover in remaining:
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

    n_critical = sum(
        1
        for e in errors
        if e.get("severity") == "critical" or e["category"] in CRITICAL_CATEGORIES
    )
    n_noncritical = sum(1 for e in errors if e.get("severity") == "noncritical")
    # Prefer explicit severity when both apply.
    n_critical = sum(1 for e in errors if e.get("severity") == "critical")
    n_noncritical = sum(1 for e in errors if e.get("severity") != "critical")
    return {
        "errors": errors,
        "n_errors": len(errors),
        "n_critical": n_critical,
        "n_noncritical": n_noncritical,
    }


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

    spurious = [
        p.get("hand_number")
        for i, p in enumerate(pred_hands)
        if i not in used_pred and isinstance(p, dict)
    ]
    for hand_number in spurious:
        per_hand.append(
            {
                "hand_id": None,
                "t": None,
                "completion_class": None,
                "errors": [
                    _error(
                        "spurious_hand",
                        f"unmatched prediction hand_number={hand_number}",
                        severity="critical",
                    )
                ],
                "n_errors": 1,
                "n_critical": 1,
                "n_noncritical": 0,
            }
        )

    total_errors = sum(h["n_errors"] for h in per_hand)
    critical_errors = sum(h["n_critical"] for h in per_hand)
    hands_scored = len([h for h in per_hand if h.get("hand_id") is not None])
    ok = (
        hands_scored > 0
        and critical_errors == 0
        and budget_violations == 0
        and not spurious
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
        "per_hand": per_hand,
    }
