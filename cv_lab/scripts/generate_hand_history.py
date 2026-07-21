"""Render reconstructed timeline hands as PokerStars-style hand histories.

Input is the same `two_model_timeline_<video>.json` the eval harness and
manual-verification viewer already consume -- this only changes how the
per-hand `actions`/`streets`/`players` data already in that file is
*presented*. No new CV signal is read.

Known limitations (inherent to what the CV read can see, not a bug here):
  - Villain hole cards are only known when the poker client actually flips them
    face-up on screen (an all-in showdown, or a voluntary show) -- most hands
    never reach that, so most non-hero seats still show as "mucked".
  - Stakes/date are synthetic (bb-denominated, keyed to video + t_start)
    since no real currency or wall-clock timestamp exists.
  - "raises A to B" totals are derived by accumulating each seat's own
    per-action chip delta across the street; there is no independent
    "amount to call" signal to cross-check against, so a street with a
    dropped/misread action can throw off every later total on that street.
  - Uncalled-bet-returned lines are not emitted; the summary only reports
    the total pot and the collecting seat.
  - The summary's "Total pot" is the pipeline's already-reconciled `pot`
    field (validated elsewhere against text/contributed-chips/win-sweep
    consensus); the street-by-street action amounts above it are a
    best-effort replay of raw per-action deltas and are not guaranteed to
    sum to exactly that total when a street had a dropped/misread action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SB = 0.5
BB = 1.0

STREET_HEADER = {"flop": "FLOP", "turn": "TURN", "river": "RIVER"}

POSITION_FULL = {
    "BTN": "Button",
    "SB": "Small Blind",
    "BB": "Big Blind",
    "UTG": "UTG",
    "UTG+1": "UTG+1",
    "MP": "Lojack",
    "HJ": "Hijack",
    "CO": "Cutoff",
}


def _fmt(amount: float) -> str:
    return f"{amount:.2f}".rstrip("0").rstrip(".") or "0"


def _seat_label(p: dict) -> str:
    return "Hero" if p["is_hero"] else p["player_name"]


def _pos_label(position: str | None) -> str:
    if not position:
        return ""
    return POSITION_FULL.get(position, position)


def render_hand(hand: dict, video: str) -> str:
    players = {p["seat"]: p for p in hand["players"]}
    lines: list[str] = []

    hand_id = f"{video.upper()}-{hand['hand_number']}"
    lines.append(
        f"PokerTrainer Hand #{hand_id}: Hold'em No Limit ({_fmt(SB)}bb/{_fmt(BB)}bb) "
        f"- Session {video}, t={hand['t_start']:.0f}s"
    )
    btn_seat = hand.get("dealer_seat")
    n_seats = max(6, len(players))
    lines.append(f"Table '{video}' {n_seats}-max Seat #{btn_seat} is the button")

    for seat in sorted(players):
        p = players[seat]
        lines.append(f"Seat {seat}: {_seat_label(p)} ({_fmt(p['starting_stack'])}bb in chips)")

    sb_seat = next((s for s, p in players.items() if p["position"] == "SB"), None)
    bb_seat = next((s for s, p in players.items() if p["position"] == "BB"), None)
    if sb_seat is not None:
        lines.append(f"{_seat_label(players[sb_seat])}: posts small blind {_fmt(SB)}")
    if bb_seat is not None:
        lines.append(f"{_seat_label(players[bb_seat])}: posts big blind {_fmt(BB)}")

    hero = next((p for p in players.values() if p["is_hero"]), None)
    if hero is not None and hand.get("hero"):
        lines.append(f"*** HOLE CARDS ***")
        lines.append(f"Dealt to Hero [{' '.join(hand['hero'])}]")

    board = hand.get("board") or []
    committed = {s: 0.0 for s in players}
    if sb_seat is not None:
        committed[sb_seat] = SB
    if bb_seat is not None:
        committed[bb_seat] = BB
    current_bet = BB if bb_seat is not None else 0.0
    folded: set[int] = set()

    street_order = ["preflop", "flop", "turn", "river"]
    actions_by_street: dict[str, list[dict]] = {s: [] for s in street_order}
    for a in hand.get("actions", []):
        actions_by_street.setdefault(a["street"], []).append(a)

    def board_brackets(street: str) -> str:
        if street == "flop":
            return f"[{' '.join(board[:3])}]"
        if street == "turn":
            return f"[{' '.join(board[:3])}] [{board[3]}]" if len(board) > 3 else "[]"
        if street == "river":
            return f"[{' '.join(board[:4])}] [{board[4]}]" if len(board) > 4 else "[]"
        return ""

    # Which streets were actually reached is ground-truthed off the final
    # board length, not off whether any action happened to be captured on
    # it -- a street can be checked through with nothing recorded and still
    # have been played (e.g. river dealt but no action data survived).
    streets_reached = {"preflop"}
    if len(board) >= 3:
        streets_reached.add("flop")
    if len(board) >= 4:
        streets_reached.add("turn")
    if len(board) == 5:
        streets_reached.add("river")

    for street in street_order:
        if street not in streets_reached:
            continue
        acts = actions_by_street.get(street, [])
        if street != "preflop":
            lines.append(f"*** {STREET_HEADER[street]} *** {board_brackets(street)}")
            committed = {s: 0.0 for s in players}
            current_bet = 0.0

        for a in acts:
            seat = a["seat"]
            label = _seat_label(players[seat]) if seat in players else a.get("player_name", f"Seat{seat}")
            atype = a["action_type"]
            amt = a.get("amount")

            if atype == "fold":
                # folds are tracked internally (for the summary's collected/
                # mucked logic) but never printed -- only calls/raises/bets
                # matter here.
                folded.add(seat)
            elif atype == "check":
                lines.append(f"{label}: checks")
            elif atype == "call":
                if amt is not None:
                    committed[seat] = committed.get(seat, 0.0) + amt
                lines.append(f"{label}: calls {_fmt(amt or 0.0)}")
            elif atype in ("bet", "raise", "all-in"):
                prior = committed.get(seat, 0.0)
                new_total = prior + (amt or 0.0)
                committed[seat] = new_total
                suffix = " and is all in" if atype == "all-in" else ""
                if new_total <= current_bet:
                    # short all-in (or noisy delta) that doesn't clear the bet
                    # to call is a call, not a raise, however the pill read it.
                    lines.append(f"{label}: calls {_fmt(amt or 0.0)}{suffix}")
                elif current_bet <= 0:
                    lines.append(f"{label}: bets {_fmt(amt or 0.0)}{suffix}")
                else:
                    incr = new_total - current_bet
                    lines.append(f"{label}: raises {_fmt(incr)} to {_fmt(new_total)}{suffix}")
                current_bet = max(current_bet, new_total)
            else:
                lines.append(f"{label}: {atype} {_fmt(amt) if amt is not None else ''}".rstrip())

    lines.append("*** SUMMARY ***")
    pot = hand.get("pot")
    if pot is not None:
        lines.append(f"Total pot {_fmt(pot)}bb")
    if board:
        lines.append(f"Board [{' '.join(board)}]")

    winner_seat = hand.get("winner_seat")
    hero_folded = hand.get("hero_folded")
    for seat in sorted(players):
        p = players[seat]
        label = _seat_label(p)
        pos = f" ({_pos_label(p.get('position'))})" if p.get("position") else ""
        is_folded = seat in folded or (p["is_hero"] and hero_folded)
        if is_folded:
            continue  # folds are irrelevant here; only calls/raises are shown
        shown = p.get("shown_cards")
        if seat == winner_seat:
            shown_bit = f" showed [{' '.join(shown)}] and" if shown else ""
            lines.append(f"Seat {seat}: {label}{pos}{shown_bit} collected ({_fmt(pot)}bb)")
        elif p["is_hero"] and hand.get("hero"):
            lines.append(f"Seat {seat}: {label}{pos} showed [{' '.join(hand['hero'])}] and lost")
        elif shown:
            lines.append(f"Seat {seat}: {label}{pos} showed [{' '.join(shown)}] and lost")
        else:
            lines.append(f"Seat {seat}: {label}{pos} mucked")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True, help="path to two_model_timeline_<video>.json")
    ap.add_argument("--video", required=True, help="short video id, e.g. v00")
    ap.add_argument("--hand", type=int, default=None, help="single hand number; omit for all")
    ap.add_argument("--out", default=None, help="output .txt path; default stdout")
    args = ap.parse_args()

    tl = json.load(open(args.timeline))
    hands = tl["hands"]
    if args.hand is not None:
        hands = [h for h in hands if h["hand_number"] == args.hand]

    rendered = "\n\n\n".join(render_hand(h, args.video) for h in hands)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
        print(f"wrote {len(hands)} hand(s) to {args.out}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
