"""Validate an offline YOLO card-state timeline.

This script reads a saved completed-session card timeline. It does not capture
live tables or provide current-hand advice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


VALID_BOARD_COUNTS = {0, 3, 4, 5}
STREET_ORDER = {0: 0, 3: 1, 4: 2, 5: 3}
# Action-street ordering for reconstructed hands (spine output).
ACTION_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}


class MalformedTimeline(ValueError):
    """Raised when the input is not a readable YOLO card timeline."""


def _cards(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise MalformedTimeline("card fields must be lists")
    return value


def _warn(code: str, message: str, **details: Any) -> dict[str, Any]:
    warning = {"code": code, "message": message}
    warning.update(details)
    return warning


def _card_warning_context(state: dict[str, Any], hand: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {
        "time_s": state.get("time_s"),
        "image": state.get("image"),
    }
    if hand.get("hand_number") is not None:
        context["hand_number"] = hand["hand_number"]
    return context


def _validate_cards_are_strings(cards: list[Any], field: str, state: dict[str, Any], hand: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for idx, card in enumerate(cards):
        if not isinstance(card, str) or not card.strip():
            warnings.append(_warn(
                "missing_label",
                f"{field} contains a missing card label",
                field=field,
                index=idx,
                **_card_warning_context(state, hand),
            ))
    return warnings


def _state_warnings(state: dict[str, Any], hand: dict[str, Any]) -> list[dict[str, Any]]:
    hero = _cards(state.get("hero_cards", []))
    board = _cards(state.get("board_cards", []))
    other = _cards(state.get("other_cards", []))
    warnings: list[dict[str, Any]] = []

    if len(hero) not in {0, 2}:
        warnings.append(_warn(
            "invalid_hero_count",
            "hero card count must be 0 or 2",
            hero_count=len(hero),
            hero=hero,
            **_card_warning_context(state, hand),
        ))
    if len(board) not in VALID_BOARD_COUNTS:
        warnings.append(_warn(
            "invalid_board_count",
            "board card count must be 0, 3, 4, or 5",
            board_count=len(board),
            board=board,
            **_card_warning_context(state, hand),
        ))

    visible = hero + board
    duplicates = sorted({card for card in visible if isinstance(card, str) and card and visible.count(card) > 1})
    if duplicates:
        warnings.append(_warn(
            "duplicate_visible_cards",
            "duplicate cards appear across hero and board",
            duplicates=duplicates,
            hero=hero,
            board=board,
            **_card_warning_context(state, hand),
        ))

    for field, cards in (("hero_cards", hero), ("board_cards", board), ("other_cards", other)):
        warnings.extend(_validate_cards_are_strings(cards, field, state, hand))

    if state.get("missing"):
        warnings.append(_warn(
            "missing_label",
            "state carries missing label metadata from review",
            missing=state["missing"],
            **_card_warning_context(state, hand),
        ))

    return warnings


def _ordered_hand_states(hand: dict[str, Any], states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_images = hand.get("source_images")
    if isinstance(source_images, list) and source_images:
        wanted = {image for image in source_images if isinstance(image, str)}
        return [state for state in states if state.get("image") in wanted]

    t_start = hand.get("t_start")
    t_end = hand.get("t_end")
    if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)):
        return [
            state for state in states
            if isinstance(state.get("time_s"), (int, float)) and t_start <= state["time_s"] <= t_end
        ]

    return []


def _hand_from_all_states(states: list[dict[str, Any]]) -> dict[str, Any]:
    times = [state.get("time_s") for state in states if isinstance(state.get("time_s"), (int, float))]
    return {
        "hand_number": 1,
        "t_start": min(times) if times else None,
        "t_end": max(times) if times else None,
        "source_images": [state.get("image") for state in states if state.get("image")],
    }


def _hand_sequence_warnings(hand: dict[str, Any], hand_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    previous_count: int | None = None
    previous_order: int | None = None
    previous_board: list[str] = []

    for state in hand_states:
        board = _cards(state.get("board_cards", []))
        count = len(board)
        context = _card_warning_context(state, hand)

        if previous_count is not None and count < previous_count:
            warnings.append(_warn(
                "board_regression",
                "board card count shrank within the hand",
                previous_count=previous_count,
                board_count=count,
                previous_board=previous_board,
                board=board,
                **context,
            ))
        elif previous_board and count >= previous_count and not set(previous_board).issubset(set(board)):
            warnings.append(_warn(
                "board_regression",
                "board cards changed without preserving earlier board cards",
                previous_board=previous_board,
                board=board,
                **context,
            ))

        order = STREET_ORDER.get(count)
        if order is not None:
            if previous_order is not None and order < previous_order:
                warnings.append(_warn(
                    "street_order_issue",
                    "street order moved backward within the hand",
                    previous_board_count=previous_count,
                    board_count=count,
                    **context,
                ))
            elif previous_order is not None and order > previous_order + 1:
                warnings.append(_warn(
                    "street_order_issue",
                    "street order skipped an expected board state",
                    previous_board_count=previous_count,
                    board_count=count,
                    **context,
                ))
            previous_order = order

        previous_count = count
        previous_board = board

    streets = hand.get("streets", [])
    if isinstance(streets, list):
        seen_order: int | None = None
        for street in streets:
            if not isinstance(street, dict):
                continue
            board = _cards(street.get("board", []))
            order = STREET_ORDER.get(len(board))
            if order is None:
                continue
            if seen_order is not None and order < seen_order:
                warnings.append(_warn(
                    "street_order_issue",
                    "hand street summary is out of order",
                    street=street.get("street"),
                    board_count=len(board),
                    hand_number=hand.get("hand_number"),
                    time_s=street.get("time_s"),
                ))
            seen_order = order

    return warnings


def _hand_summary_warnings(hand: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    hero = _cards(hand.get("hero") or [])
    board = _cards(hand.get("board", []))
    context = {
        "hand_number": hand.get("hand_number"),
        "time_s": hand.get("t_start"),
        "image": None,
    }

    if len(hero) not in {0, 2}:
        warnings.append(_warn(
            "invalid_hero_count",
            "hand summary hero card count must be 0 or 2",
            hero_count=len(hero),
            hero=hero,
            **context,
        ))
    if len(board) not in VALID_BOARD_COUNTS:
        warnings.append(_warn(
            "invalid_board_count",
            "hand summary board card count must be 0, 3, 4, or 5",
            board_count=len(board),
            board=board,
            **context,
        ))

    visible = hero + board
    duplicates = sorted({card for card in visible if isinstance(card, str) and card and visible.count(card) > 1})
    if duplicates:
        warnings.append(_warn(
            "duplicate_visible_cards",
            "duplicate cards appear across hand summary hero and board",
            duplicates=duplicates,
            hero=hero,
            board=board,
            **context,
        ))

    for field, cards in (("hero", hero), ("board", board)):
        for idx, card in enumerate(cards):
            if not isinstance(card, str) or not card.strip():
                warnings.append(_warn(
                    "missing_label",
                    f"{field} contains a missing card label",
                    field=field,
                    index=idx,
                    **context,
                ))

    return warnings


def _hand_reconstruction_warnings(hand: dict[str, Any]) -> list[dict[str, Any]]:
    """Warnings for the spine's extended fields (players / actions / pot / winner).

    Card-only timelines carry none of these keys and are skipped, so this is purely
    additive over the existing card checks.
    """
    warnings: list[dict[str, Any]] = []
    if not any(key in hand for key in ("players", "actions")):
        return warnings

    context = {"hand_number": hand.get("hand_number"), "time_s": hand.get("t_start"), "image": None}

    # Per-street pot must not shrink across streets.
    streets = hand.get("streets")
    if isinstance(streets, list):
        prev_pot: float | None = None
        for street in streets:
            if not isinstance(street, dict):
                continue
            pot = street.get("pot")
            if isinstance(pot, (int, float)):
                if prev_pot is not None and pot < prev_pot - 1e-6:
                    warnings.append(_warn(
                        "pot_regression",
                        "per-street pot shrank across streets",
                        street=street.get("street"), pot=pot, previous_pot=prev_pot, **context,
                    ))
                prev_pot = pot

    # Actions must not move to an earlier street.
    actions = hand.get("actions")
    if isinstance(actions, list):
        prev_order: int | None = None
        for action in actions:
            if not isinstance(action, dict):
                continue
            order = ACTION_STREET_ORDER.get(action.get("street"))
            if order is None:
                warnings.append(_warn(
                    "action_street_invalid", "action has an unknown street",
                    street=action.get("street"), **context,
                ))
                continue
            if prev_order is not None and order < prev_order:
                warnings.append(_warn(
                    "action_street_order", "actions moved to an earlier street",
                    street=action.get("street"), **context,
                ))
            prev_order = max(order, prev_order) if prev_order is not None else order

    # Positions must be unique across players.
    players = hand.get("players")
    if isinstance(players, list):
        positions = [p.get("position") for p in players if isinstance(p, dict) and p.get("position")]
        dupes = sorted({pos for pos in positions if positions.count(pos) > 1})
        if dupes:
            warnings.append(_warn(
                "position_issue", "duplicate positions across players",
                positions=dupes, **context,
            ))

    # Soft arithmetic reconciliation flag from the spine.
    if hand.get("pot") is not None and hand.get("reconciled") is False:
        warnings.append(_warn(
            "reconciliation_failed",
            "sum of stack contributions does not match the final pot",
            contributed=hand.get("contributed_est"), pot=hand.get("pot"), **context,
        ))

    return warnings


def _confidence_score(warning_count: int, checked_states: int) -> float:
    if checked_states <= 0:
        return 0.0
    score = 1.0 - (warning_count / max(checked_states * 3, 1))
    return round(max(0.0, min(1.0, score)), 3)


def validate_timeline(timeline: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(timeline, dict):
        raise MalformedTimeline("timeline JSON must be an object")

    states = timeline.get("states")
    if not isinstance(states, list):
        raise MalformedTimeline("timeline must contain a states list")
    for state in states:
        if not isinstance(state, dict):
            raise MalformedTimeline("states must contain objects")

    hands = timeline.get("hands")
    if hands is None:
        hands = [_hand_from_all_states(states)] if states else []
    if not isinstance(hands, list):
        raise MalformedTimeline("hands must be a list when present")
    for hand in hands:
        if not isinstance(hand, dict):
            raise MalformedTimeline("hands must contain objects")

    reports: list[dict[str, Any]] = []
    total_warnings = 0
    checked_states = 0

    for index, hand in enumerate(hands, start=1):
        hand_states = _ordered_hand_states(hand, states)
        if not hand_states and len(hands) == 1:
            hand_states = states

        warnings = _hand_summary_warnings(hand)
        for state in hand_states:
            warnings.extend(_state_warnings(state, hand))
        warnings.extend(_hand_sequence_warnings(hand, hand_states))
        warnings.extend(_hand_reconstruction_warnings(hand))

        checked_states += len(hand_states)
        total_warnings += len(warnings)
        reports.append({
            "hand_number": hand.get("hand_number", index),
            "t_start": hand.get("t_start"),
            "t_end": hand.get("t_end"),
            "checked_states": len(hand_states),
            "warning_count": len(warnings),
            "confidence_score": _confidence_score(len(warnings), max(len(hand_states), 1)),
            "warnings": warnings,
        })

    return {
        "summary": {
            "malformed": False,
            "hands": len(reports),
            "checked_states": checked_states,
            "warning_hands": sum(1 for report in reports if report["warning_count"]),
            "total_warnings": total_warnings,
            "confidence_score": _confidence_score(total_warnings, max(checked_states, 1)),
        },
        "hands": reports,
    }


def _load_timeline(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise MalformedTimeline(f"invalid JSON: {exc}") from exc
    except OSError as exc:
        raise MalformedTimeline(str(exc)) from exc
    if not isinstance(data, dict):
        raise MalformedTimeline("timeline JSON must be an object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline", help="Path to a YOLO card timeline JSON")
    parser.add_argument("--out", help="Optional path for the validation report JSON")
    parser.add_argument("--fail-on-warnings", action="store_true", help="Exit nonzero when validation warnings are found")
    args = parser.parse_args(argv)

    try:
        report = validate_timeline(_load_timeline(Path(args.timeline)))
    except MalformedTimeline as exc:
        report = {
            "summary": {
                "malformed": True,
                "error": str(exc),
                "hands": 0,
                "checked_states": 0,
                "warning_hands": 0,
                "total_warnings": 0,
                "confidence_score": 0.0,
            },
            "hands": [],
        }
        output = json.dumps(report, indent=2)
        if args.out:
            Path(args.out).write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return 2

    output = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    if args.fail_on_warnings and report["summary"]["total_warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
