"""Hand-level accuracy eval: reconstructed timeline vs a ground-truth answer key.

Scores the CV -> hand-reconstruction bridge by counting discrete errors per hand
("things wrong"): segmentation (missed / split / spurious hands), hero cards,
board cards, dealer seat, winner, final pot, hero net, and the action sequence.
The project bar is: given CV info, at most ONE thing wrong per hand.

Ground truth format (built by VLM-assisted annotation, cross-checked with pot
arithmetic -- see cv_lab/results/ground_truth/):
    {"video": "v00", "hands": [{
        "t_first_seen": s, "t_last_seen": s,
        "partial_start": bool, "partial_end": bool,
        "hero_cards": ["5d","6s"]|null, "final_board": [...],
        "dealer_seat": int|null, "players_dealt_in": [ints],
        "final_pot": float|null, "winner_seat": int|null, "hero_net": float|null,
        "actions": [{"street","order","seat","action","amount","certain"}],
        "actions_complete": bool
    }, ...]}

Usage:
    python -m cv_lab.scripts.eval_hand_reconstruction \
        --timeline cv_lab/results/hand_timeline_v00_gtbox.json \
        --truth cv_lab/results/ground_truth/v00_hands.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

STREETS = ("preflop", "flop", "turn", "river")

# Tolerances for numeric comparisons (amounts are in displayed BB units).
POT_TOL = 0.5          # absolute BB tolerance on final pot
POT_TOL_FRAC = 0.02    # ... or 2% relative, whichever is larger
NET_TOL = 1.0          # hero net tolerance (settled-stack reads can lag a beat)
AMT_TOL = 0.5          # per-action amount tolerance
AMT_TOL_FRAC = 0.05


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _amounts_match(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return True  # unreadable amount is scored by the action match itself
    return abs(a - b) <= max(AMT_TOL, AMT_TOL_FRAC * max(abs(a), abs(b)))


def _canon_action(atype: str) -> str:
    return {"all_in": "all-in"}.get(atype, atype)


def _action_key(a: dict[str, Any]) -> tuple:
    return (a["street"], int(a["seat"]), _canon_action(str(a["action_type" if "action_type" in a else "action"])))


def _types_compatible(gt: str, pred: str) -> bool:
    """all-in is a size statement, not a distinct move: it matches bet/raise/call."""
    gt, pred = _canon_action(gt), _canon_action(pred)
    if gt == pred:
        return True
    aggressive = {"bet", "raise", "call", "all-in"}
    return "all-in" in (gt, pred) and gt in aggressive and pred in aggressive


def _match_actions(gt_actions: list[dict], pred_actions: list[dict]) -> dict[str, Any]:
    """Per-street MULTISET alignment on (seat, compatible type).

    Intra-street display order is not scored -- poker order is recoverable from
    seats + the ring, and 2s sampling makes observed order approximate. Only GT
    actions marked certain participate, and 'win' pseudo-rows are skipped.
    Errors: missing (GT action absent), spurious (predicted action with no GT
    support), wrong_amount (matched pair with irreconcilable sizes).
    """
    errors: dict[str, list] = {"missing": [], "spurious": [], "wrong_amount": []}
    for street in STREETS:
        gt = [a for a in gt_actions
              if a["street"] == street and a.get("certain", True) and a["action"] != "win"]
        pred = [a for a in pred_actions if a["street"] == street]
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        # two passes: exact type equality first, then all-in-compatible matches
        for exact in (True, False):
            for i, g in enumerate(gt):
                if i in matched_gt:
                    continue
                for j, p in enumerate(pred):
                    if j in matched_pred or int(p["seat"]) != int(g["seat"]):
                        continue
                    g_t, p_t = _canon_action(str(g["action"])), _canon_action(str(p["action_type"]))
                    ok = (g_t == p_t) if exact else _types_compatible(g_t, p_t)
                    if not ok:
                        continue
                    matched_gt.add(i)
                    matched_pred.add(j)
                    if not _amounts_match(g.get("amount"), p.get("amount")):
                        errors["wrong_amount"].append(
                            {"street": street, "seat": g["seat"], "gt": g.get("amount"),
                             "pred": p.get("amount")})
                    break
        errors["missing"].extend(
            {"street": street, "seat": gt[k]["seat"], "action": gt[k]["action"]}
            for k in range(len(gt)) if k not in matched_gt)
        errors["spurious"].extend(
            {"street": street, "seat": pred[k]["seat"], "action": pred[k]["action_type"]}
            for k in range(len(pred)) if k not in matched_pred)
    return errors


def _score_hand(gt: dict[str, Any], pred: dict[str, Any] | None,
                n_fragments: int) -> dict[str, Any]:
    """Count discrete errors for one GT hand against its matched prediction."""
    errs: list[dict[str, Any]] = []

    def err(category: str, detail: Any) -> None:
        errs.append({"category": category, "detail": detail})

    if pred is None:
        err("missed_hand", f"no predicted hand overlaps t={gt['t_first_seen']}-{gt['t_last_seen']}")
        return {"errors": errs, "n_errors": len(errs)}

    if n_fragments > 1:
        err("split_hand", f"GT hand covered by {n_fragments} predicted hands")

    # hero cards (order-insensitive)
    gt_hero = set(gt.get("hero_cards") or [])
    pred_hero = set(pred.get("hero") or [])
    if gt_hero:
        for c in sorted(gt_hero - pred_hero):
            err("hero_card", f"missing/wrong {c} (pred={sorted(pred_hero)})")
        for c in sorted(pred_hero - gt_hero):
            if len(gt_hero - pred_hero) < len(pred_hero - gt_hero):
                err("hero_card", f"spurious {c}")

    # board (order-insensitive per card; length mismatches surface as set diffs)
    gt_board = set(gt.get("final_board") or [])
    pred_board = set(pred.get("board") or [])
    if not gt.get("partial_end"):
        for c in sorted(gt_board - pred_board):
            err("board_card", f"missing/wrong {c} (pred={sorted(pred_board)})")
        for c in sorted(pred_board - gt_board):
            if len(gt_board - pred_board) < len(pred_board - gt_board):
                err("board_card", f"spurious {c}")

    # dealer seat
    if gt.get("dealer_seat") is not None and pred.get("dealer_seat") != gt["dealer_seat"]:
        err("dealer_seat", f"gt={gt['dealer_seat']} pred={pred.get('dealer_seat')}")

    # final pot
    gt_pot, pred_pot = gt.get("final_pot"), pred.get("pot")
    if gt_pot is not None and not gt.get("partial_end"):
        if pred_pot is None or abs(pred_pot - gt_pot) > max(POT_TOL, POT_TOL_FRAC * gt_pot):
            err("final_pot", f"gt={gt_pot} pred={pred_pot}")

    # winner
    if gt.get("winner_seat") is not None and not gt.get("partial_end"):
        if pred.get("winner_seat") != gt["winner_seat"]:
            err("winner", f"gt={gt['winner_seat']} pred={pred.get('winner_seat')}")

    # hero net
    gt_net = gt.get("hero_net")
    if gt_net is not None and not gt.get("partial_end") and not gt.get("partial_start"):
        pred_net = pred.get("hero_bb_won")
        if pred_net is None or abs(pred_net - gt_net) > NET_TOL:
            err("hero_net", f"gt={gt_net} pred={pred_net}")

    # actions
    a_errs = _match_actions(gt.get("actions") or [], pred.get("actions") or [])
    for e in a_errs["missing"]:
        err("action_missing", e)
    if gt.get("actions_complete", False):
        for e in a_errs["spurious"]:
            err("action_spurious", e)
    for e in a_errs["wrong_amount"]:
        err("action_amount", e)

    return {"errors": errs, "n_errors": len(errs)}


def evaluate(truth: dict[str, Any], timeline: dict[str, Any],
             include_partial: bool = False) -> dict[str, Any]:
    gt_hands = [h for h in truth["hands"]
                if include_partial or not (h.get("partial_start") or h.get("partial_end"))]
    pred_hands = timeline["hands"]

    # match each GT hand to overlapping predictions
    per_hand = []
    used_pred: set[int] = set()
    for gt in gt_hands:
        overlaps = [
            (i, _overlap(gt["t_first_seen"], gt["t_last_seen"], p["t_start"], p["t_end"]))
            for i, p in enumerate(pred_hands)
        ]
        frags = [i for i, ov in overlaps if ov > 0]
        best = max(overlaps, key=lambda t: t[1]) if overlaps else (None, 0.0)
        pred = pred_hands[best[0]] if best[0] is not None and best[1] > 0 else None
        used_pred.update(frags)
        result = _score_hand(gt, pred, n_fragments=len(frags))
        per_hand.append({
            "t": [gt["t_first_seen"], gt["t_last_seen"]],
            "hero_gt": gt.get("hero_cards"),
            "matched_pred": pred["hand_number"] if pred else None,
            "fragments": len(frags),
            **result,
        })

    spurious_hands = [p["hand_number"] for i, p in enumerate(pred_hands)
                      if i not in used_pred]

    by_category: dict[str, int] = {}
    for h in per_hand:
        for e in h["errors"]:
            by_category[e["category"]] = by_category.get(e["category"], 0) + 1

    n = len(per_hand)
    total = sum(h["n_errors"] for h in per_hand)
    return {
        "hands_scored": n,
        "total_errors": total,
        "errors_per_hand": round(total / n, 2) if n else None,
        "max_errors_in_hand": max((h["n_errors"] for h in per_hand), default=0),
        "hands_at_most_1_error": sum(1 for h in per_hand if h["n_errors"] <= 1),
        "by_category": dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
        "spurious_predicted_hands": spurious_hands,
        "per_hand": per_hand,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", default="", help="optional JSON report path")
    ap.add_argument("--include-partial", action="store_true",
                    help="also score GT hands cut off by the recording window")
    args = ap.parse_args()

    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    timeline = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
    report = evaluate(truth, timeline, include_partial=args.include_partial)

    print(f"hands scored:        {report['hands_scored']}")
    print(f"total errors:        {report['total_errors']}")
    print(f"errors per hand:     {report['errors_per_hand']}")
    print(f"max errors (1 hand): {report['max_errors_in_hand']}")
    print(f"hands with <=1 err:  {report['hands_at_most_1_error']}/{report['hands_scored']}")
    print(f"by category:         {report['by_category']}")
    if report["spurious_predicted_hands"]:
        print(f"spurious pred hands: {report['spurious_predicted_hands']}")
    for h in report["per_hand"]:
        flag = "OK " if h["n_errors"] <= 1 else "BAD"
        print(f"  [{flag}] t={h['t'][0]:.0f}-{h['t'][1]:.0f} hero={h['hero_gt']} "
              f"pred#{h['matched_pred']} frags={h['fragments']} errors={h['n_errors']}")
        for e in h["errors"]:
            print(f"        - {e['category']}: {e['detail']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
