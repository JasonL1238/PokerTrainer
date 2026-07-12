"""Reconstruction spine: 7-class YOLO region detections -> full hand histories.

This is the fuller sibling of build_yolo_card_timeline.py on the SAME YOLO path.
Where the card-only builder handles just `face_card` (hero/board/streets), this
consumes all seven region classes and reconstructs complete hands:

    positions   <- dealer_button + seat ring
    dealt-in/folds <- card_back (appear/disappear)
    streets     <- face_card board count
    actions     <- action_pill (type) ordered by active_turn_indicator,
                   with bet SIZES from stack_text deltas, reconciled vs pot_text
    pot/winner  <- pot_text series + stack recovery

It emits the same timeline JSON shape validate_yolo_card_timeline / the app exporter
already read, EXTENDED per hand with players / actions / pot / result / winner_seat.

Input is the decoupled Frame contract (region_detections). No detector, OCR, or
model is required to run it: synthetic fixtures and labeled ground-truth boxes both
work today, and the trained 7-class detector plugs in via the same contract later.
Completed-session data only; never live capture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cv_lab.scripts.build_yolo_card_timeline import (
    _best_board,
    _stage,
    _street_events,
)
from cv_lab.scripts import region_detections as rd

DEFAULT_OUT = "cv_lab/results/yolo_hand_timeline.json"

_STREET_BY_COUNT = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
_POSITION_NAMES = ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"]
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Pass 1: per-frame table state + collapse to distinct states
# --------------------------------------------------------------------------- #
def _frame_state(frame: rd.Frame) -> dict[str, Any]:
    view = rd.assign_regions(frame)
    seats = view["seats"]
    dealt_in = sorted(i for i, info in seats.items() if info["card_back"])
    stacks = {i: info["stack"] for i, info in seats.items() if info["stack"] is not None}
    pills = {i: info["pill_action"] for i, info in seats.items() if info["pill_action"]}
    board = view["board"]
    return {
        "time_s": frame.time_s,
        "image": frame.image,
        "stage": _stage(len(board)),
        "hero_cards": view["hero"],
        "board_cards": board,
        "other_cards": [],
        "pot": view["pot"],
        "dealt_in": dealt_in,
        "stacks": stacks,
        "pills": pills,
        "dealer_seat": view["dealer_seat"],
        "active_seat": view["active_seat"],
        "missing": None,
    }


def _signature(state: dict[str, Any]) -> tuple:
    return (
        tuple(state["hero_cards"]),
        tuple(state["board_cards"]),
        tuple(state["dealt_in"]),
        tuple(sorted(state["stacks"].items())),
        tuple(sorted(state["pills"].items())),
        state["pot"],
        state["dealer_seat"],
        state["active_seat"],
    )


def build_states(frames: list[rd.Frame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (distinct states, events). States collapse consecutive identical frames."""
    states: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_sig: tuple | None = None
    prev: dict[str, Any] | None = None

    for frame in frames:
        state = _frame_state(frame)
        sig = _signature(state)
        if sig == last_sig:
            continue
        state["state_index"] = len(states)
        states.append(state)

        if prev is not None:
            if state["hero_cards"] != prev["hero_cards"]:
                events.append({"type": "hero_cards_changed", "time_s": state["time_s"],
                               "from": prev["hero_cards"], "to": state["hero_cards"]})
            if state["board_cards"] != prev["board_cards"]:
                events.append({"type": "board_changed", "time_s": state["time_s"],
                               "stage": state["stage"], "to": state["board_cards"]})
            if state["pot"] != prev["pot"]:
                events.append({"type": "pot_changed", "time_s": state["time_s"],
                               "from": prev["pot"], "to": state["pot"]})
            folded = set(prev["dealt_in"]) - set(state["dealt_in"])
            for seat in sorted(folded):
                events.append({"type": "fold", "time_s": state["time_s"], "seat": seat})
            for seat, action in state["pills"].items():
                if prev["pills"].get(seat) != action:
                    events.append({"type": "action", "time_s": state["time_s"],
                                   "seat": seat, "action": action})
            if state["active_seat"] != prev["active_seat"]:
                events.append({"type": "turn_changed", "time_s": state["time_s"],
                               "seat": state["active_seat"]})

        last_sig = sig
        prev = state
    return states, events


# --------------------------------------------------------------------------- #
# Pass 2: segment into hands (hero-cards change is the boundary)
# --------------------------------------------------------------------------- #
def _segment(states: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    hands: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_hero: list[str] = []
    for state in states:
        hero = state["hero_cards"]
        boundary = False
        if current:
            prev = current[-1]
            hero_changed = len(hero) == 2 and hero != current_hero
            board_reset = bool(prev["board_cards"]) and not state["board_cards"]
            large_gap = state["time_s"] - prev["time_s"] > 30
            boundary = hero_changed and (board_reset or bool(prev["board_cards"]) or large_gap)
        if boundary:
            hands.append(current)
            current, current_hero = [], []
        current.append(state)
        if len(hero) == 2 and not current_hero:
            current_hero = hero
    if current:
        hands.append(current)
    return hands


# --------------------------------------------------------------------------- #
# Pass 3: reconstruct one hand
# --------------------------------------------------------------------------- #
def _street_for_count(count: int, last: str) -> str:
    return _STREET_BY_COUNT.get(count, last)


def _positions(players: list[int], dealer_seat: int | None) -> dict[int, str]:
    """Assign BTN/SB/BB/... walking the seat ring from the dealer over dealt-in seats."""
    if not players:
        return {}
    ring = [s for s in rd.SEAT_RING if s in players]
    if not ring:
        ring = sorted(players)
    start = 0
    if dealer_seat in ring:
        start = ring.index(dealer_seat)
    ordered = ring[start:] + ring[:start]
    return {seat: (_POSITION_NAMES[k] if k < len(_POSITION_NAMES) else f"P{k}")
            for k, seat in enumerate(ordered)}


def _stack_series(hand: list[dict[str, Any]], seat: int) -> list[float]:
    return [s["stacks"][seat] for s in hand if seat in s["stacks"]]


def _mode(values: list[Any]) -> Any:
    from collections import Counter
    values = [v for v in values if v is not None]
    return Counter(values).most_common(1)[0][0] if values else None


def _reconstruct_actions(
    hand: list[dict[str, Any]], positions: dict[int, str], player_name: dict[int, str]
) -> list[dict[str, Any]]:
    """Derive ordered actions from state-to-state deltas: folds from card_back
    disappearing, bet/raise/call sizes from stack decreases (reconciled against the
    pot delta), checks from a check pill with no stack change."""
    actions: list[dict[str, Any]] = []
    street_index: dict[str, int] = {}
    street_has_bet: dict[str, bool] = {}

    for prev, cur in zip(hand, hand[1:]):
        street = _street_for_count(len(cur["board_cards"]), cur["stage"])
        if street not in _STREET_BY_COUNT.values():
            continue
        pot_before = prev["pot"]

        # folds first (card_back disappeared)
        for seat in sorted(set(prev["dealt_in"]) - set(cur["dealt_in"])):
            street_index[street] = street_index.get(street, 0) + 1
            actions.append(_action(street, street_index[street], seat, positions, player_name,
                                    "fold", None, pot_before, prev["stacks"].get(seat)))

        # stack-decrease actions (bet / raise / call / all-in)
        for seat in sorted(cur["stacks"]):
            before = prev["stacks"].get(seat)
            after = cur["stacks"][seat]
            if before is None or after is None or after >= before - _EPS:
                continue
            amount = round(before - after, 2)
            pill = cur["pills"].get(seat)
            if after <= _EPS:
                atype = "all-in"
            elif pill in {"raise", "bet", "call"}:
                atype = pill
            else:
                atype = "call" if street_has_bet.get(street) else "bet"
            if atype in {"bet", "raise", "all-in"}:
                street_has_bet[street] = True
            street_index[street] = street_index.get(street, 0) + 1
            actions.append(_action(street, street_index[street], seat, positions, player_name,
                                    atype, amount, pot_before, before))

        # explicit checks (check pill, no stack change)
        for seat, pill in cur["pills"].items():
            if pill == "check" and prev["pills"].get(seat) != "check":
                before, after = prev["stacks"].get(seat), cur["stacks"].get(seat)
                if before is not None and after is not None and abs(before - after) <= _EPS:
                    street_index[street] = street_index.get(street, 0) + 1
                    actions.append(_action(street, street_index[street], seat, positions,
                                            player_name, "check", None, pot_before, before))
    return actions


def _action(street, index, seat, positions, player_name, atype, amount, pot_before, stack_before):
    return {
        "street": street,
        "action_index": index,
        "seat": seat,
        "player_name": player_name[seat],
        "position": positions.get(seat, ""),
        "action_type": atype,
        "amount": amount,
        "pot_before": pot_before,
        "stack_before": stack_before,
    }


def reconstruct(hand: list[dict[str, Any]], hand_number: int) -> dict[str, Any]:
    hero_candidates = [s["hero_cards"] for s in hand if len(s["hero_cards"]) == 2]
    hero = hero_candidates[0] if hero_candidates else []
    board = _best_board(hand)
    dealer_seat = _mode([s["dealer_seat"] for s in hand])

    players = sorted({seat for s in hand for seat in s["dealt_in"]})
    positions = _positions(players, dealer_seat)
    player_name = {seat: ("Hero" if seat == 0 else f"Seat{seat}") for seat in players}

    player_rows = []
    for seat in players:
        series = _stack_series(hand, seat)
        player_rows.append({
            "seat": seat,
            "position": positions.get(seat, ""),
            "player_name": player_name[seat],
            "starting_stack": series[0] if series else None,
            "is_hero": seat == 0,
        })

    actions = _reconstruct_actions(hand, positions, player_name)

    # per-street end pot (last stable pot at each board count) + final pot
    street_pot: dict[str, float] = {}
    for s in hand:
        street = _STREET_BY_COUNT.get(len(s["board_cards"]))
        if street and s["pot"] is not None:
            street_pot[street] = s["pot"]
    streets = _street_events(hand)
    for street in streets:
        if street["street"] in street_pot:
            street["pot"] = street_pot[street["street"]]
    pots = [s["pot"] for s in hand if s["pot"] is not None]
    final_pot = pots[-1] if pots else None

    # winner + hero result from stack recovery
    contributed = 0.0
    winner_seat, win_gain = None, 0.0
    hero_bb_won = None
    for seat in players:
        series = _stack_series(hand, seat)
        if not series:
            continue
        low = min(series)
        contributed += max(series[0] - low, 0.0)
        gain = series[-1] - low
        if gain > win_gain + _EPS:
            win_gain, winner_seat = gain, seat
        if seat == 0:
            hero_bb_won = round(series[-1] - series[0], 2)
    contributed = round(contributed, 2)

    if winner_seat == 0:
        result = "Hero wins"
    elif winner_seat is not None:
        result = "Villain wins"
    else:
        result = ""

    cards = hero + board
    cards_unique = len(cards) == len(set(cards))
    reconciled = (
        final_pot is not None and contributed > 0
        and abs(contributed - final_pot) <= max(3.0, 0.25 * final_pot)
    )
    complete_cards = len(hero) == 2 and len(board) in {0, 3, 4, 5} and cards_unique
    complete = bool(hero) and cards_unique and final_pot is not None and bool(actions) and winner_seat is not None

    warnings: list[str] = []
    if len(hero) != 2:
        warnings.append("hero_cards_not_two")
    if board and len(board) not in {3, 4, 5}:
        warnings.append("invalid_board_count")
    if not cards_unique:
        warnings.append("duplicate_visible_cards")
    if final_pot is not None and not reconciled:
        warnings.append("pot_not_reconciled")

    return {
        "hand_number": hand_number,
        "t_start": hand[0]["time_s"],
        "t_end": hand[-1]["time_s"],
        "n_states": len(hand),
        "hero": hero or None,
        "board": board,
        "dealer_seat": dealer_seat,
        "players": player_rows,
        "streets": streets,
        "actions": actions,
        "pot": final_pot,
        "winner_seat": winner_seat,
        "win_gain": round(win_gain, 2),
        "result": result,
        "hero_bb_won": hero_bb_won,
        "contributed_est": contributed,
        "reconciled": reconciled,
        "complete_cards": complete_cards,
        "complete": complete,
        "warnings": warnings,
        "source_images": [s["image"] for s in hand],
    }


# --------------------------------------------------------------------------- #
# Top-level build
# --------------------------------------------------------------------------- #
def build_hand_timeline(frames: list[rd.Frame]) -> dict[str, Any]:
    states, events = build_states(frames)
    hands = [reconstruct(hand, i) for i, hand in enumerate(_segment(states), start=1)]
    return {
        "metadata": {
            "source": "yolo_region_detections",
            "classes": rd.CLASSES,
            "notes": [
                "Offline completed-session reconstruction from 7-class region detections.",
                "Seat zones are coarse geometry; the anchored seat model plugs in later.",
                "Attribute reads (rank/suit, amounts, pill colour) are pluggable stubs.",
            ],
        },
        "summary": {
            "frames": len(frames),
            "states": len(states),
            "events": len(events),
            "hands": len(hands),
            "complete_hands": sum(1 for h in hands if h["complete"]),
            "card_complete_hands": sum(1 for h in hands if h["complete_cards"]),
        },
        "hands": hands,
        "states": states,
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Frames fixture JSON (region_detections contract)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    frames = rd.load_frames(args.frames)
    timeline = build_hand_timeline(frames)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    print(f"frames={timeline['summary']['frames']}")
    print(f"states={timeline['summary']['states']}")
    print(f"hands={timeline['summary']['hands']}")
    print(f"complete_hands={timeline['summary']['complete_hands']}")


if __name__ == "__main__":
    main()
