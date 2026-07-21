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
    bets = {i: info.get("bet") for i, info in seats.items() if info.get("bet") is not None}
    pills = {i: info["pill_action"] for i, info in seats.items() if info["pill_action"]}
    board = view["board"]
    # A pot text of exactly 0 is never real once a hand is in progress (blinds and
    # antes are posted before the first observed state); it is an OCR dropout on the
    # pot region. Treat it as unread so it can't anchor initial_pot / final_pot.
    pot = view["pot"]
    if pot is not None and pot <= 0.0:
        pot = None
    return {
        "time_s": frame.time_s,
        "image": frame.image,
        "stage": _stage(len(board)),
        "hero_cards": view["hero"],
        "board_cards": board,
        "other_cards": [],
        "villain_cards": view.get("villain_cards", {}),
        "hero_dim": view.get("hero_dim", False),
        "pot": pot,
        "dealt_in": dealt_in,
        "stacks": stacks,
        "bets": bets,
        "pills": pills,
        "dealer_seat": view["dealer_seat"],
        "active_seat": view["active_seat"],
        "hero_seat_mismatch": view.get("hero_seat_mismatch", False),
        "missing": None,
    }


def _signature(state: dict[str, Any]) -> tuple:
    return (
        tuple(state["hero_cards"]),
        tuple(state["board_cards"]),
        tuple(sorted((seat, tuple(cards)) for seat, cards in state["villain_cards"].items())),
        state["hero_dim"],
        tuple(state["dealt_in"]),
        tuple(sorted(state["stacks"].items())),
        tuple(sorted(state["bets"].items())),
        tuple(sorted(state["pills"].items())),
        state["pot"],
        state["dealer_seat"],
        state["active_seat"],
    )


def _debounce_cards(raw: list[dict[str, Any]], key: str) -> None:
    """Debounce card-list fields (hero_cards / board_cards) in place over the raw
    per-frame states. A NON-EMPTY reading that differs from the accepted one must
    be confirmed by the next non-empty reading, else it is replaced by the carried
    value -- or dropped entirely when there is nothing to carry. An empty reading
    RESETS the carry: cards leaving the table (sweep / new deal) must not leak the
    previous hand's cards into the next one."""
    accepted: tuple | None = None
    for idx, state in enumerate(raw):
        cards = tuple(state[key])
        if not cards:
            accepted = None
            continue
        if accepted is not None and cards == accepted:
            continue
        confirmed = True
        for nxt in raw[idx + 1:]:
            nxt_cards = tuple(nxt[key])
            if nxt_cards:
                # confirmed when the next reading repeats the candidate OR
                # extends it (boards only grow: a one-state turn immediately
                # followed by the river that contains it is real)
                confirmed = (nxt_cards == cards
                             or nxt_cards[: len(cards)] == cards)
                break
        if confirmed:
            accepted = cards
        else:
            state[key] = list(accepted) if accepted is not None else []


def _debounce_bool_confirm(raw: list[dict[str, Any]], key: str) -> None:
    """A True reading must be confirmed by the VERY NEXT reading also being
    True, else it's a single-frame blip (e.g. a card-flip/bet animation
    transiently darkening the hero-card crop) and is rejected. A real greyed-
    out fold holds for many consecutive samples, so this costs it nothing;
    a one-off dip loses its only reading. Mirrors _debounce_cards' confirm
    rule but for a plain boolean rather than a card list."""
    n = len(raw)
    accepted = [False] * n
    for i, state in enumerate(raw):
        if state[key] and i + 1 < n and raw[i + 1][key]:
            accepted[i] = True
    for i, state in enumerate(raw):
        state[key] = accepted[i]


def build_states(frames: list[rd.Frame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (distinct states, events). Raw per-frame states are debounced first
    (cards, pot, stacks, bets -- single-frame OCR/classifier blips are rejected
    unless the next reading confirms them), then collapsed to distinct states."""
    raw = [_frame_state(f) for f in frames]
    _debounce_cards(raw, "hero_cards")
    _debounce_cards(raw, "board_cards")
    _debounce_bool_confirm(raw, "hero_dim")
    for state in raw:
        state["stage"] = _stage(len(state["board_cards"]))

    states: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_sig: tuple | None = None
    prev: dict[str, Any] | None = None

    for state in raw:
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
    """A new hand starts when the hero's hole cards change AND there is table
    evidence of a fresh deal: the board reset, the pot dropped (back to the
    blinds+antes), the dealer button moved, or a long recording gap. A hero-card
    change alone -- with a board still showing, or mid-preflop with the same pot
    and button -- is read noise (suit misreads, showdown reveals landing in the
    hero card zone), never a real boundary."""
    hands: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_hero: list[str] = []
    anchor: dict[str, Any] | None = None  # last state showing the current hero's cards
    for state in states:
        hero = state["hero_cards"]
        boundary = False
        if current:
            ref = anchor or current[-1]
            hero_changed = len(hero) == 2 and bool(current_hero) and hero != current_hero
            if hero_changed and not state["board_cards"]:
                # Compare against the anchor, not the immediately preceding state:
                # the table often passes through a swept interstitial state (no
                # cards, pot cleared) that would otherwise hide the reset.
                board_reset = bool(ref["board_cards"])
                pot_dropped = (state["pot"] is not None and ref["pot"] is not None
                               and state["pot"] < ref["pot"] - _EPS)
                dealer_moved = (state["dealer_seat"] is not None
                                and ref["dealer_seat"] is not None
                                and state["dealer_seat"] != ref["dealer_seat"])
                large_gap = state["time_s"] - ref["time_s"] > 30
                boundary = board_reset or pot_dropped or dealer_moved or large_gap
        if boundary:
            hands.append(current)
            current, current_hero, anchor = [], [], None
        current.append(state)
        if len(hero) == 2 and not current_hero:
            current_hero = hero
        if hero == current_hero and current_hero:
            anchor = state
    if current:
        hands.append(current)
    return hands


# --------------------------------------------------------------------------- #
# Temporal cleanup: OCR jitter debounce (a value must survive two consecutive
# states to be believed; otherwise the previous accepted value carries forward)
# --------------------------------------------------------------------------- #
def _debounce_series(hand: list[dict[str, Any]], key: str) -> None:
    """Debounce per-seat numeric dicts (stacks / bets) in place over ONE hand's
    states, with revert-only rejection: a reading B is a blip only in an
    A -> B -> A pattern (the next reading reverts to the previous accepted
    value). Directional sequences (A -> B -> C) are kept -- a call followed
    immediately by the winner's pot award must not be eaten."""
    accepted: dict[int, float] = {}
    for idx, state in enumerate(hand):
        cur = state[key]
        for seat in list(cur):
            val = cur[seat]
            prev_val = accepted.get(seat)
            if prev_val is None or abs(val - prev_val) <= _EPS:
                accepted[seat] = val
                continue
            nxt_val = None
            for nxt in hand[idx + 1:]:
                if seat in nxt[key]:
                    nxt_val = nxt[key][seat]
                    break
            if nxt_val is not None and abs(nxt_val - prev_val) <= _EPS:
                cur[seat] = prev_val  # A -> B -> A: B was an OCR blip
            else:
                accepted[seat] = val
    # done in place


def _debounce_pot(hand: list[dict[str, Any]]) -> None:
    """Debounce ONE hand's pot series in place: revert-only blip rejection plus
    a floor -- a mid-hand pot can never fall below the antes already in it."""
    accepted: float | None = None
    for idx, state in enumerate(hand):
        pot = state["pot"]
        if pot is None:
            continue
        if accepted is not None and pot < 1.0 <= accepted:
            state["pot"] = accepted
            continue
        if accepted is None or abs(pot - accepted) <= _EPS:
            accepted = pot
            continue
        nxt_val = None
        for nxt in hand[idx + 1:]:
            if nxt["pot"] is not None:
                nxt_val = nxt["pot"]
                break
        if nxt_val is not None and abs(nxt_val - accepted) <= _EPS:
            state["pot"] = accepted  # A -> B -> A blip
        else:
            accepted = pot


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


def _trim_trailing_next_deal(hand: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop trailing states that already belong to the NEXT deal.

    Segmentation cuts a hand only when the hero's hole cards change to a new pair,
    so an interstitial frame between hands -- button already advanced, pot reset to
    the blinds, hero not yet re-dealt -- trails the current hand and corrupts its
    final pot, hero net, and reconciliation (e.g. a 240 BB pot 'ending' at 1.5).
    The button only moves between hands, so a trailing state whose dealer differs
    from the hand's modal dealer, or whose pot has collapsed far below the hand's
    peak, while the hero holds no cards, is next-deal noise."""
    if len(hand) <= 1:
        return hand
    modal_dealer = _mode([s["dealer_seat"] for s in hand])
    max_pot = max((s["pot"] for s in hand if s["pot"] is not None), default=None)
    end = len(hand)
    while end > 1:
        s = hand[end - 1]
        dealer_moved = (modal_dealer is not None and s["dealer_seat"] is not None
                        and s["dealer_seat"] != modal_dealer)
        pot_collapsed = (max_pot is not None and max_pot > 6.0 and s["pot"] is not None
                         and s["pot"] < 0.3 * max_pot)
        no_hero = len(s["hero_cards"]) != 2
        if no_hero and (dealer_moved or pot_collapsed):
            end -= 1
        else:
            break
    return hand[:end]


def _settle_index(hand: list[dict[str, Any]]) -> int:
    """Index of the state where the pot is swept to the winner, or the last index
    if never seen. The sweep is the single LARGEST qualifying stack jump in the
    hand -- during play stacks only fall (chips go in), so any increase is a pot
    award and the real sweep dwarfs any pre-sweep noise bump. Picking the first
    jump above a low threshold is wrong when a spurious sub-pot bump on a folded
    seat precedes the true sweep by a frame (it truncates the window before the
    winner is paid, flipping the result). States after the sweep are next-deal
    noise: the next hand's antes tick every stack down and its blinds post while
    the old hero cards linger."""
    n = len(hand)
    # Suffix max of the pot from each index onward: a real settlement is TERMINAL
    # -- the pot is swept and only the next deal's small blinds follow -- so a
    # candidate jump is rejected if the pot later grows well beyond its own level.
    # This stops a mid-hand phantom (a stack OCR blip on a seat that then folds)
    # from truncating the hand before the real sweep.
    suffix_max_pot = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        p = hand[i]["pot"] or 0.0
        suffix_max_pot[i] = max(p, suffix_max_pot[i + 1])
    pot_so_far: float | None = None
    best_idx, best_jump = n - 1, 0.0
    for idx, (prev, cur) in enumerate(zip(hand, hand[1:]), start=1):
        if prev["pot"] is not None:
            pot_so_far = prev["pot"] if pot_so_far is None else max(pot_so_far, prev["pot"])
        threshold = max(3.0, 0.4 * (pot_so_far or 0.0))
        cur_pot = cur["pot"] if cur["pot"] is not None else (pot_so_far or 0.0)
        if suffix_max_pot[idx + 1] > cur_pot + threshold:
            continue  # pot keeps growing after this -> not the terminal sweep
        for seat, after in cur["stacks"].items():
            before = prev["stacks"].get(seat)
            if before is None:
                continue
            jump = after - before
            # >= so that, on a tie, the later (true post-sweep) frame wins.
            if jump >= threshold and jump >= best_jump:
                best_idx, best_jump = idx, jump
    return best_idx


def _reconstruct_actions(
    hand: list[dict[str, Any]], positions: dict[int, str], player_name: dict[int, str]
) -> list[dict[str, Any]]:
    """Derive ordered actions from state-to-state deltas.

    Sources, in order of trust:
      folds   <- fold pills (the ONLY signal for the hero, whose grayed cards keep
                 a card_back on screen) plus card_back disappearance, latched so a
                 later pill misread cannot un-fold a seat;
      amounts <- bet_text deltas (per-street contributions rendered on the felt),
                 falling back to debounced stack decreases;
      checks  <- a fresh check pill with no money delta (readable stacks NOT
                 required -- pill evidence stands on its own).

    Every action is attributed to the PREVIOUS state's street: chips go in (and
    seats fold) before the next board card appears, so on the transition where a
    street closes, the new state already shows the next street's board.
    """
    actions: list[dict[str, Any]] = []
    street_index: dict[str, int] = {}
    # Preflop starts True: the blinds are a live bet, so limps are calls and
    # preflop folds are always legitimate.
    street_has_bet: dict[str, bool] = {"preflop": True}
    folded: set[int] = set()
    acted: dict[str, set[int]] = {}          # street -> seats seen acting
    street_raised: dict[str, bool] = {}      # street -> saw raise/bet/all-in

    def emit(street, seat, atype, amount, pot_before, stack_before):
        street_index[street] = street_index.get(street, 0) + 1
        if atype in {"bet", "raise", "all-in"}:
            street_has_bet[street] = True
            street_raised[street] = True
        acted.setdefault(street, set()).add(seat)
        actions.append(_action(street, street_index[street], seat, positions, player_name,
                               atype, amount, pot_before, stack_before))

    # ---- pre-observed actions standing in the hand's FIRST state ----
    # A hand often enters view mid-preflop: pills and bet_texts already on the
    # table are completed actions we never saw happen. Bets at or below the big
    # blind with no pill are blind posts, not actions. (This table renders
    # amounts in BB units, so the big blind is 1.0.)
    first = hand[0]
    if not first["board_cards"]:
        for seat in sorted(set(first["pills"]) | set(first["bets"])):
            pill = first["pills"].get(seat)
            bet = first["bets"].get(seat)
            if pill == "fold":
                folded.add(seat)
                emit("preflop", seat, "fold", None, None, first["stacks"].get(seat))
            elif pill in {"raise", "bet", "call"}:
                emit("preflop", seat, pill if pill != "bet" else "raise",
                     bet, None, first["stacks"].get(seat))
            elif pill == "check":
                emit("preflop", seat, "check", None, None, first["stacks"].get(seat))
            elif pill is None and bet is not None and bet > 1.0 + _EPS:
                emit("preflop", seat, "call", bet, None, first["stacks"].get(seat))

    settle = _settle_index(hand)
    for prev, cur in zip(hand[: settle + 1], hand[1: settle + 1]):
        street = _street_for_count(len(prev["board_cards"]), prev["stage"])
        if street not in _STREET_BY_COUNT.values():
            continue
        pot_before = prev["pot"]

        # ---- money actions (bet_text delta, corroborated by the stack) ----
        money_seats: dict[int, tuple[float, float | None]] = {}  # seat -> (amount, stack_before)
        for seat in sorted(set(cur["bets"]) | set(cur["stacks"])):
            if seat in folded or seat not in positions:
                continue
            before = prev["stacks"].get(seat)
            after = cur["stacks"].get(seat)
            stack_dropped = before is not None and after is not None and after < before - _EPS
            stack_flat = before is not None and after is not None and abs(after - before) <= _EPS
            amount = None
            if stack_dropped:
                # the debounced stack delta is the most reliable size
                amount = round(before - after, 2)
            elif not stack_flat:
                # stack unreadable this transition: fall back to the bet_text
                # delta. (A bet_text rising while the stack is provably unchanged
                # is rendering lag of an action already emitted from the stack.)
                cur_bet = cur["bets"].get(seat)
                prev_bet = prev["bets"].get(seat)
                same_street = len(prev["board_cards"]) == len(cur["board_cards"])
                if cur_bet is not None and cur_bet >= 0.5 - _EPS:
                    base = prev_bet if (same_street and prev_bet is not None) else 0.0
                    if cur_bet > base + _EPS:
                        amount = round(cur_bet - base, 2)
            if amount is not None and amount >= 0.5 - _EPS:
                money_seats[seat] = (amount, before)

        for seat, (amount, stack_before) in sorted(money_seats.items()):
            pill = cur["pills"].get(seat)
            after = cur["stacks"].get(seat)
            if after is not None and after <= _EPS:
                atype = "all-in"
            elif pill in {"raise", "bet", "call"}:
                atype = pill
            else:
                atype = "call" if street_has_bet.get(street) else "bet"
            emit(street, seat, atype, amount, pot_before, stack_before)

        # ---- folds: card_back disappeared OR a fresh fold pill (hero: pill only) ----
        # A fold is only possible facing a bet; "folds" detected on a street where
        # nobody has bet are showdown reveals/sweeps, not actions.
        gone = set(prev["dealt_in"]) - set(cur["dealt_in"])
        pill_folds = {seat for seat, pill in cur["pills"].items()
                      if pill == "fold" and prev["pills"].get(seat) != "fold"}
        for seat in sorted((gone | pill_folds) - folded):
            if seat in money_seats or seat not in positions:
                continue  # money and a fold can't both happen; unknown seats are noise
            if not street_has_bet.get(street):
                continue
            if seat in gone and seat not in pill_folds:
                if len(gone) >= 2:
                    continue  # multi-seat sweep after the pot is awarded
                before = prev["stacks"].get(seat)
                if before is not None and before <= _EPS:
                    continue  # an all-in player's cards flip over; they can't fold
            folded.add(seat)
            emit(street, seat, "fold", None, pot_before, prev["stacks"].get(seat))

        # ---- checks: fresh check pill, no money this transition ----
        # A fresh check pill arriving together WITH a new board card belongs to
        # the new street (its first check), unlike money, which closes streets.
        check_street = street
        if len(cur["board_cards"]) > len(prev["board_cards"]):
            check_street = _street_for_count(len(cur["board_cards"]), cur["stage"])
        for seat, pill in sorted(cur["pills"].items()):
            if pill != "check" or prev["pills"].get(seat) == "check":
                continue
            if seat in money_seats or seat in folded or seat not in positions:
                continue
            emit(check_street, seat, "check", None, pot_before, prev["stacks"].get(seat))

    # ---- synthesized closing checks ----
    # 2s sampling and instant street closes hide some checks: a street that ended
    # with no bet was checked through by every live seat, and a preflop with no
    # raise gives the big blind a free check. Emit the checks we know happened
    # but never saw.
    streets_seen = {_STREET_BY_COUNT[n] for n in
                    {len(s["board_cards"]) for s in hand} if n in _STREET_BY_COUNT}
    folded_at: dict[int, str] = {a["seat"]: a["street"] for a in actions
                                 if a["action_type"] == "fold"}
    order = [s for s in ("preflop", "flop", "turn", "river") if s in streets_seen]
    all_in_at: dict[int, int] = {a["seat"]: order.index(a["street"]) for a in actions
                                 if a["action_type"] == "all-in" and a["street"] in order}
    for si, street in enumerate(order):
        live = [p for p in positions
                if p not in folded_at or order.index(folded_at[p]) > si]
        # seats already all-in on an earlier street cannot act; with fewer than
        # two actionable seats there is no betting round to check through
        actionable = [p for p in live if all_in_at.get(p, 99) >= si]
        if len(actionable) < 2:
            continue
        for seat in sorted(actionable):
            if seat in acted.get(street, set()):
                continue
            if not street_has_bet.get(street):
                # on the last street, only a showdown (>=2 seats never folded)
                # proves the unseen players actually got to check
                if street != order[-1] or len(actionable) >= 2:
                    emit(street, seat, "check", None, None, None)
            elif street == "preflop" and positions.get(seat) == "BB" \
                    and not street_raised.get("preflop"):
                emit(street, seat, "check", None, None, None)

    # Emission interleaves sources (transitions, cur-street checks, synthesis),
    # so impose global street order; within a street the emit order stands.
    street_rank = {s: k for k, s in enumerate(("preflop", "flop", "turn", "river"))}
    actions.sort(key=lambda a: street_rank.get(a["street"], 9))
    return actions


def _action(street, index, seat, positions, player_name, atype, amount, pot_before, stack_before):
    return {
        "street": street,
        "action_index": index,
        "seat": seat,
        # A pill can be assigned (by the coarse stub seat model) to a seat that was
        # never seen dealt in; fall back to a synthetic name rather than crashing.
        "player_name": player_name.get(seat, "Hero" if seat == 0 else f"Seat{seat}"),
        "position": positions.get(seat, ""),
        "action_type": atype,
        "amount": amount,
        "pot_before": pot_before,
        "stack_before": stack_before,
    }


def _vote_board(hand: list[dict[str, Any]]) -> list[str]:
    """Majority-vote the final board: among observed board tuples, take the longest
    one that either repeats (>=2 states) or extends a repeating shorter tuple as a
    prefix. A single-frame misread tuple loses to the stable reading."""
    from collections import Counter

    counts = Counter(tuple(s["board_cards"]) for s in hand if s["board_cards"])
    if not counts:
        return []
    stable = [t for t, n in counts.items() if n >= 2]
    stable.sort(key=lambda t: (len(t), counts[t]))
    best = stable[-1] if stable else ()
    for t, n in counts.items():
        if n == 1 and len(t) > len(best) and tuple(t[: len(best)]) == tuple(best):
            best = t  # a once-seen river that extends the stable turn is real
    return list(best) if best else _best_board(hand)


def reconstruct(hand: list[dict[str, Any]], hand_number: int) -> dict[str, Any]:
    # Shed any next-deal interstitial frames that segmentation left on the tail
    # before anything measures pot / stacks / winner from them.
    hand = _trim_trailing_next_deal(hand)
    # Numeric debounce is PER HAND: bets/stacks/pot must never carry an accepted
    # value across a hand boundary (the next hand's first readings are fresh).
    _debounce_series(hand, "stacks")
    _debounce_series(hand, "bets")
    _debounce_pot(hand)
    hero = _mode([tuple(s["hero_cards"]) for s in hand if len(s["hero_cards"]) == 2])
    hero = list(hero) if hero else []
    board = _vote_board(hand)
    dealer_seat = _mode([s["dealer_seat"] for s in hand])

    # A seat is a player with two states of card_back evidence, or with any
    # evidence in the hand's opening states (instant folders show exactly one
    # frame of cards). A single mid-hand misdetection must not conjure a
    # phantom player.
    from collections import Counter

    dealt_counts = Counter(seat for s in hand for seat in s["dealt_in"])
    opening = {seat for s in hand[:2] for seat in s["dealt_in"]}
    players = sorted(seat for seat, n in dealt_counts.items()
                     if n >= 2 or seat in opening)
    if not players:
        players = sorted(dealt_counts)
    positions = _positions(players, dealer_seat)
    player_name = {seat: ("Hero" if seat == 0 else f"Seat{seat}") for seat in players}

    # Villain showdown reveals: a non-hero seat's cards flip face-up only when
    # the client actually shows them (all-in showdown, or a voluntary show), so
    # this is empty for the vast majority of hands. Mode over every state's
    # reading (like hero/board) rather than trusting a single frame.
    shown_cards: dict[int, list[str]] = {}
    known_cards = set(hero) | set(board)
    for seat in players:
        if seat == 0:
            continue
        votes = [tuple(s["villain_cards"].get(seat, [])) for s in hand]
        votes = [v for v in votes if len(v) == 2]
        best = _mode(votes)
        # A reveal sharing a card with hero/board is a misread (board/hero
        # card bleeding across zones), not a real villain reveal -- drop it.
        if best and not (set(best) & known_cards):
            shown_cards[seat] = list(best)

    player_rows = []
    for seat in players:
        series = _stack_series(hand, seat)
        player_rows.append({
            "seat": seat,
            "position": positions.get(seat, ""),
            "player_name": player_name[seat],
            "starting_stack": series[0] if series else None,
            "is_hero": seat == 0,
            "shown_cards": shown_cards.get(seat),
        })

    actions = _reconstruct_actions(hand, positions, player_name)

    # per-street end pot (last stable pot at each board count) + final pot.
    # Only states up to the settlement (pot swept to the winner) count: after it
    # the display is already showing the next deal's antes/blinds.
    settle = _settle_index(hand)
    settled_states = hand[: settle + 1]
    street_pot: dict[str, float] = {}
    for s in settled_states:
        street = _STREET_BY_COUNT.get(len(s["board_cards"]))
        if street and s["pot"] is not None:
            street_pot[street] = s["pot"]
    streets = _street_events(hand)
    for street in streets:
        if street["street"] in street_pot:
            street["pot"] = street_pot[street["street"]]
    pots = [s["pot"] for s in settled_states if s["pot"] is not None]
    final_pot = pots[-1] if pots else None

    # winner + hero result from stack recovery. Contributions are measured
    # against the last PRE-settlement stack, not the minimum: an over-shove's
    # uncalled portion is returned before the sweep and never enters the pot.
    # Gains are measured on the SETTLED window only: the next-deal states that
    # trail a hand contain auto top-ups (stack refills to the buy-in) and the
    # next hand's antes/blinds, either of which corrupts a full-series delta.
    # Phantom-winner guard: a seat whose LAST action is a fold, showing a "gain"
    # that exceeds the pot it could possibly rake, is a stack OCR blip (a folded
    # short stack misreading upward), not a sweep -- disqualify it from winning.
    # Both conditions are required so real winners survive: a transient fold pill
    # misdetected on a seat that then wins still shows a gain consistent with the
    # pot (that seat is kept), and an all-in winner who is never marked folded is
    # never touched however large the side pot.
    last_action: dict[int, str] = {}
    for a in actions:
        last_action[a["seat"]] = a["action_type"]
    folded_seats = {s for s, act in last_action.items() if act == "fold"}
    contributed = 0.0
    winner_seat, win_gain = None, 0.0
    hero_bb_won = None
    for seat in players:
        series = _stack_series(settled_states, seat) or _stack_series(hand, seat)
        if not series:
            continue
        low = min(series)
        pre_settle = [s["stacks"][seat] for s in settled_states[:-1] if seat in s["stacks"]]
        if pre_settle:
            contributed += max(series[0] - pre_settle[-1], 0.0)
        gain = series[-1] - low
        phantom = (seat in folded_seats and final_pot is not None
                   and gain > 1.5 * final_pot + 3.0)
        if not phantom and gain > win_gain + _EPS:
            win_gain, winner_seat = gain, seat
        if seat == 0:
            hero_bb_won = round(series[-1] - series[0], 2)
    contributed = round(contributed, 2)

    # Did the hero fold? Signals, strongest first: a hero fold pill, a
    # reconstructed hero fold action, the hero's own cards rendered greyed-out
    # (the client's persistent in-place fold indicator -- unlike the pill,
    # which flashes for under a second, this holds for the rest of the hand,
    # so it survives sparse sampling that would otherwise miss the pill), or
    # -- for a blind/checked hero who never commits chips (no pill sampled) --
    # the hero's cards mucking away while the pot is still contested multiway.
    # The last is gated on winner_seat is None and >=2 villains still holding
    # cards so a hero-WIN sweep (everyone else's cards clear at once) can
    # never be mistaken for a fold.
    hero_had_cards = len(hero) == 2
    hero_pill_fold = any(s["pills"].get(0) == "fold" for s in hand)
    hero_action_fold = any(a["seat"] == 0 and a["action_type"] == "fold" for a in actions)
    hero_dim_fold = any(s["hero_dim"] for s in hand)
    last = hand[-1]
    villains_live_at_end = len([s for s in last["dealt_in"] if s != 0]) >= 2
    hero_mucked = (
        hero_had_cards and winner_seat is None and villains_live_at_end
        and len(hand[0]["hero_cards"]) == 2 and len(last["hero_cards"]) != 2
    )
    hero_folded = hero_had_cards and (
        hero_pill_fold or hero_action_fold or hero_dim_fold or hero_mucked
    )

    if winner_seat == 0:
        result = "Hero wins"
    elif winner_seat is not None:
        result = "Villain wins"
    elif hero_folded:
        result = "Hero folds"
    else:
        result = ""

    # A hand the hero folded out of, whose villain resolution falls outside the
    # tracked window (no pot sweep observed), is a COMPLETE record of the hero's
    # decision -- cards, fold, net loss -- even without an observed winner. The
    # full-pot reconciliation does not apply: we never see the pot close.
    hero_fold_only = hero_folded and winner_seat is None

    cards = hero + board
    cards_unique = len(cards) == len(set(cards))
    # Blinds + antes are already in the pot before the first observed state
    # (stacks are pre-debited), so observed stack contributions reconcile
    # against the pot GROWTH from the first reading, not the whole pot.
    initial_pot = pots[0] if pots else None

    # Three INDEPENDENT estimates of the final pot, each with a distinct failure
    # mode, so a wrong one is outvoted rather than trusted:
    #   text    -- the pot-region OCR; can drop out / freeze stale mid-betting
    #              (a 52 BB river sweep still displaying "12").
    #   contrib -- initial pot + chips every seat put in; breaks if a stack
    #              misreads (one bad seat inflates the sum).
    #   win     -- chips swept to the winner; OVERCOUNTS by any uncalled bet
    #              returned (an over-shove's excess is refunded pre-sweep).
    # Report the estimate in the largest mutually-agreeing cluster, preferring
    # the pot text on ties (it needs no arithmetic). A hand reconciles when >=2
    # estimates independently agree -- money conservation the single text read
    # can't fake. Two stack estimates agreeing against the text means the text
    # is the misread, so the text is correctly overridden.
    pot_text = final_pot
    contrib_pot = round(initial_pot + contributed, 2) if initial_pot is not None else None
    win_pot = round(win_gain, 2) if (winner_seat is not None and win_gain > _EPS) else None
    candidates = [(k, v) for k, v in
                  (("text", pot_text), ("contrib", contrib_pot), ("win", win_pot))
                  if v is not None]

    def _pot_agree(a: float, b: float) -> bool:
        return abs(a - b) <= max(3.0, 0.15 * max(a, b))

    pref = {"text": 0, "contrib": 1, "win": 2}  # tie-break order
    best = None  # (support, -pref, value)
    for k, v in candidates:
        support = sum(1 for _, u in candidates if _pot_agree(v, u))
        key = (support, -pref[k])
        if best is None or key > best[0]:
            best = (key, v, k)
    if best is not None and not hero_fold_only:
        final_pot = best[1]
    best_support = best[0][0] if best is not None else 0
    pot_text_dropped = bool(best is not None and best[2] != "text"
                            and pot_text is not None and not _pot_agree(final_pot, pot_text))

    if hero_fold_only:
        reconciled = True  # not applicable: villain resolution unobserved
    else:
        reconciled = best_support >= 2

    # Phantom-winner demotion: an UNRECONCILED villain-winner on a hand the hero
    # folded is not corroborated by the pot or the contributions -- it is a
    # phantom sweep from a trailing junk frame (the table replaced by the lobby,
    # or a frozen end-of-recording frame the stack OCR misreads). Reporting it
    # would assert a winner we cannot stand behind. The reliable, coaching-
    # relevant fact is the hero fold; drop the unverifiable winner and record the
    # villain resolution as unobserved. (A reconciled winner is kept: v01#1's
    # real river sweep still reconciles via the contrib/win consensus.)
    if hero_folded and winner_seat is not None and not reconciled:
        winner_seat, win_gain = None, 0.0
        result = "Hero folds"
        final_pot = pot_text  # the consensus/derived pot rode on the phantom sweep
        hero_fold_only = True
        pot_text_dropped = False
        reconciled = True

    complete_cards = len(hero) == 2 and len(board) in {0, 3, 4, 5} and cards_unique
    complete = (bool(hero) and cards_unique and final_pot is not None and bool(actions)
                and (winner_seat is not None or hero_fold_only))

    warnings: list[str] = []
    if len(hero) != 2:
        warnings.append("hero_cards_not_two")
    if board and len(board) not in {3, 4, 5}:
        warnings.append("invalid_board_count")
    if not cards_unique:
        warnings.append("duplicate_visible_cards")
    if final_pot is not None and not reconciled and not hero_fold_only:
        warnings.append("pot_not_reconciled")
    if any(s.get("hero_seat_mismatch") for s in hand):
        # The hero zone's cards sat nearer another seat's card anchor: the
        # "hero = seat 0" convention is suspect, so is_hero / hero_position /
        # hero_bb_won attribution below can't be trusted for this layout.
        warnings.append("hero_seat_mismatch")

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
        "pot_text_dropped": pot_text_dropped,
        "winner_seat": winner_seat,
        "win_gain": round(win_gain, 2),
        "result": result,
        "hero_folded": hero_folded,
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
                "Offline completed-session reconstruction from 8-class region detections.",
                "Seats assigned via per-class anchors learned from the labeled boxes.",
                "Attribute reads: Model 2 rank/suit + deterministic template OCR "
                "(amounts, pill words); fixtures may pass attrs through directly.",
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
