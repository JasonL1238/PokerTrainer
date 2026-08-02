"""Validate an offline YOLO card-state timeline.

This script reads a saved completed-session card timeline. It does not capture
live tables or provide current-hand advice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

VALID_BOARD_COUNTS = {0, 3, 4, 5}
STREET_ORDER = {0: 0, 3: 1, 4: 2, 5: 3}
# Reconstruction-spine warning codes that must BLOCK export, mirrored into the
# validator report below. The export gate reads validator warnings only, so a
# spine warning that is not mirrored here is latent: it lowers the confidence
# score and lands in the notes string, but the hand still exports. That was true
# of hero_seat_mismatch, which exported at exactly 0.80 -- not < 0.8, so it did
# not even earn the LOW_CONFIDENCE tag.
SPINE_FATAL_CODES = frozenset({
    "board_zone_yield_zero",
    "board_zone_yield_partial",
    "board_empty_but_streets_advanced",
    "actions_collapsed_to_one_street",
    # Retained as GATE VOCABULARY although the spine no longer produces it: its
    # producer (`_reject_stack_outliers`, a sibling-median magnitude net) is
    # deleted, but a timeline built by an earlier build, hand-edited, or produced
    # elsewhere can still carry the code, and an unrecognised code fails closed
    # into rejection_codes with no name attached. The failure it described -- a
    # dropped decimal -- is now refused at the reader, where the evidence is.
    "amount_scale_implausible",
    "side_pot_unsupported",
    "hero_seat_mismatch",
    "anchor_unavailable",
    "stack_ledger_incoherent",
    # Chip conservation: the action ledger sums ABOVE the pot by more than any
    # uncalled shove could explain, so the record books money the pot does not
    # contain (the measured cause was one raise emitted twice, from the stack
    # delta and then again from the bet_text delta).
    "contributions_exceed_pot",
    # The published result and the hero's net disagree in sign -- e.g. "Villain
    # wins" alongside hero_bb_won=+65.2. Two fields off the same stack ledger.
    "result_contradicts_hero_net",
    # The hand's MONEY LEDGER carries an UNKNOWN: a money action the spine proved
    # happened but could not size, or a transition whose stack delta was
    # unmeasurable because the reader refused a read. PLAN.md requires an
    # incomplete sequence to be marked non-authoritative, and this is the code
    # that does it -- routed through _split_source_codes into rejection_codes,
    # which makes derive_completion_status return "uncertain".
    "amounts_unknown_in_ledger",
    # Mid-hand stretch where the table was not visible (tab-in-front / lobby /
    # modal) and board/pot/stacks/roster/dealer changed underneath. The spine
    # recovers what is visible on either side and refuses to invent the hole;
    # export must not promote the hand to complete.
    "mid_hand_coverage_gap",
    # A player row whose starting stack the reader REFUSED. Accounting rejects a
    # None starting stack outright (LedgerError), so the hand can never become
    # an authoritative money record; before this code existed the fact travelled
    # only as CompletionEvidence.extra and the hand exported as "complete", then
    # dead-ended downstream with no clearing action.
    "starting_stack_unknown",
})
# Every warning code build_yolo_hand_timeline.reconstruct (or the exporter's
# manual-correction path) can emit. A spine warning OUTSIDE this set is a code
# this build cannot assess at all, so it is mirrored as fatal rather than
# ignored. Without that arm the export gate -- which reads this report and
# nothing else -- failed OPEN on an unrecognised code while the same exported
# record listed it under rejection_codes.
RECOGNISED_SPINE_CODES = SPINE_FATAL_CODES | frozenset({
    # The card classifier read one board/hero slot two different ways while the
    # card sat on the table; the debounce absorbs the excursion and publishes the
    # majority as fact. Measured over the five development recordings, 4 of 21
    # hands carry such a contest and ONE of them is confirmed wrong on the pixels
    # ('Js 6h 4c 8d' exported for a real 'Js 6h 4c 8h'); the other three settle on
    # the reading the frames actually show. Deliberately NOT fatal: discarding
    # three good hands to hold one bad one is the wrong trade at a 1-in-4 hit
    # rate. It is severe enough (0.35) to put the hand under LOW_CONFIDENCE and
    # into review, which is the signal that was missing entirely.
    "board_card_identity_split",
    "hero_card_identity_split",
    "hero_cards_not_two",
    "invalid_board_count",
    "duplicate_visible_cards",
    "pot_not_reconciled",
    "pot_text_dropped",
    "manual_hand_correction",
})
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
        # A transient per-state misread that the hand's final (voted) hero+board
        # already resolved is noise, not a data problem: suppress when the hand
        # summary exists and carries no duplicate among these cards.
        summary = _cards(hand.get("hero") or []) + _cards(hand.get("board") or [])
        resolved = bool(summary) and all(summary.count(c) <= 1 for c in duplicates)
        if not resolved:
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


def _time_ordered(states: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """The hand's states in the order the recording produced them, and whether the
    input disagreed with that.

    Every sequence rule below -- board append-only, street order, the terminal
    sweep -- walks its argument in LIST order, and nothing anywhere read
    ``time_s``. So the verdict was a function of how the producer happened to
    emit the list rather than of what the recording shows, and the same three
    observations could be assessed two ways: a board of four cards at t=5
    followed by three at t=10 is a board_regression at confidence 0.1, but the
    identical states listed 0, 10, 5 walk 0 -> 3 -> 4 cards and score a clean
    1.0. The export gate reads this report and nothing else, so that is a
    destroyed hand promoted to study-ready by list order alone.

    Ordering is applied only when every state carries a numeric time, because
    there is no order to restore otherwise; the sort is stable, so states
    sharing a timestamp keep the order they arrived in and the report stays
    deterministic. Out-of-order input is also reported in its own right: it is
    evidence about the PRODUCER, and the spine that built this timeline
    (build_states, the debounces, fold detection, action attribution) walks the
    same list in the same way, so its output rests on the same wrong order.
    """
    times = [state.get("time_s") for state in states]
    if not all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in times):
        return list(states), False
    ordered = sorted(states, key=lambda state: state["time_s"])
    return ordered, any(a is not b for a, b in zip(ordered, states, strict=True))


def _hand_sequence_warnings(hand: dict[str, Any], hand_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    previous_count: int | None = None
    previous_order: int | None = None
    previous_board: list[str] = []

    hand_states, out_of_order = _time_ordered(hand_states)
    if out_of_order:
        warnings.append(_warn(
            "state_time_order",
            "hand states are not listed in the order their timestamps record",
            hand_number=hand.get("hand_number"),
            time_s=hand.get("t_start"),
            image=None,
        ))

    # The pot sweep at the end of every hand clears the board: an empty board
    # that never comes back within the hand is settlement, not a regression.
    last_nonempty = -1
    for idx, state in enumerate(hand_states):
        if _cards(state.get("board_cards", [])):
            last_nonempty = idx

    for idx, state in enumerate(hand_states):
        board = _cards(state.get("board_cards", []))
        count = len(board)
        context = _card_warning_context(state, hand)

        if count == 0 and idx > last_nonempty:
            break  # terminal sweep: nothing after this is street data

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


def _further_betting_was_possible(hand: dict[str, Any], actions: list[Any]) -> bool:
    """Could anyone still have acted after the last observed street?

    A hand in which every action lands on one street is only suspicious if the
    later streets HAD a betting round to miss. When all but one player is folded
    or all-in, the board runs out with no action possible and one-street action is
    the correct reconstruction -- a preflop all-in three-way pot is the ordinary
    case, not a defect. Without this test the collapse check rejects every
    preflop all-in hand.
    """
    folded: set[Any] = set()
    all_in: set[Any] = set()
    seats: set[Any] = set()
    for action in actions:
        if not isinstance(action, dict):
            continue
        seat = action.get("seat", action.get("player_key"))
        seats.add(seat)
        if action.get("action_type") == "fold":
            folded.add(seat)
        elif action.get("action_type") == "all-in":
            all_in.add(seat)
    players = hand.get("players")
    if isinstance(players, list) and players:
        seats |= {p.get("seat") for p in players if isinstance(p, dict)}
    return len(seats - folded - all_in) >= 2


def _betting_round_reopened(actions: list[Any]) -> list[dict[str, Any]]:
    """Action-sequence evidence that a street label covers MORE than one betting
    round -- i.e. a street boundary the reconstruction never recorded.

    Two structural impossibilities, both threshold-free:

    * a pill-derived ``bet`` after a ``bet``/``raise``/``all-in`` already went in
      on that street. Facing a bet the client shows RAISE, never BET, so a second
      BET pill on one street means the client had started a new street;
    * a ``check`` by a seat that already paid into that street. Calling the
      current bet closes the action to you; it reopens only if someone raises,
      and then you are facing a bet and cannot check.

    Returns one entry per violation (empty when the sequence is consistent).
    """
    aggressed: set[Any] = set()
    paid: set[tuple[Any, Any]] = set()
    hits: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        street = action.get("street")
        seat = action.get("seat", action.get("player_key"))
        kind = action.get("action_type")
        if kind == "bet" and street in aggressed:
            hits.append({"reason": "bet_after_aggression", "street": street, "seat": seat,
                         "action_index": action.get("action_index")})
        elif kind == "check" and (street, seat) in paid:
            hits.append({"reason": "check_after_paying_in", "street": street, "seat": seat,
                         "action_index": action.get("action_index")})
        if kind in {"bet", "raise", "all-in"}:
            aggressed.add(street)
        if kind in {"bet", "raise", "all-in", "call"}:
            paid.add((street, seat))
    return hits


def _action_sequence_illegal(actions: list[Any]) -> list[dict[str, Any]]:
    """Sequences that no poker client can produce, whatever the board looks like.

    This is the check that was missing from the REACHABLE path.
    ``_betting_round_reopened`` above is gated behind ``board_is_empty``, which
    makes it dead code on every hand that has a board -- and 5 of the 15 hands
    reconstructed from the development corpus carry sequences that contradict
    themselves, every one exported at confidence 1.0 with no warning and three of
    them at completion_status "complete".

    Each rule is a physical impossibility, threshold-free, and evaluated only over
    fields the exporter already publishes:

    * a seat acts again after going all-in -- it has no chips left to act with;
    * a seat folds immediately after its own call with no intervening aggression
      -- a call closes the action to you, and it reopens only if someone raises;
    * a seat checks while facing an unmatched bet on that street;
    * a seat CALLS on a post-preflop street before anyone has bet on it -- there
      is nothing to call. (Preflop is exempt: the blinds are a standing bet.)
      This is the shape the green CALL/BET pill ambiguity took when the colour
      fallback forced it to "call": 5 of the 21 hands reconstructed from the
      development corpus carried one, every one with spine warnings=[] and no
      validator code. It is fixed at the source in region_detections, and this
      is the permanent net under it -- a mislabelled aggressive action does not
      just publish the wrong word, it makes the spine drop the folds that follow.

    The two rules of ``_betting_round_reopened`` are deliberately NOT repeated
    here: they support the stronger "a street boundary was missed" claim, which
    this file only has the measurement to make on an empty board.
    """
    hits: list[dict[str, Any]] = []
    all_in: set[Any] = set()
    aggressor: dict[Any, Any] = {}            # street -> seat of the last aggressor
    aggressions: dict[Any, int] = {}          # street -> count of bet/raise/all-in
    committed: set[tuple[Any, Any]] = set()   # (street, seat) put chips in this street
    called_at: dict[tuple[Any, Any], int] = {}  # aggression count when this seat called

    def hit(reason: str, action: dict[str, Any]) -> None:
        hits.append({"reason": reason, "street": action.get("street"),
                     "seat": action.get("seat", action.get("player_key")),
                     "action_index": action.get("action_index")})

    for action in actions:
        if not isinstance(action, dict):
            continue
        street = action.get("street")
        seat = action.get("seat", action.get("player_key"))
        kind = action.get("action_type")
        key = (street, seat)
        if seat in all_in:
            hit("acts_after_all_in", action)
        elif kind == "fold" and key in called_at \
                and called_at[key] == aggressions.get(street, 0):
            # Nobody has aggressed since this seat's own call, so there is nothing
            # left for it to fold to.
            hit("fold_after_own_call", action)
        elif kind == "check" and street in aggressor \
                and aggressor[street] != seat and key not in committed:
            hit("check_facing_a_bet", action)
        elif kind == "call" and street != "preflop" and street not in aggressor:
            hit("call_with_nothing_to_call", action)
        if kind in {"bet", "raise", "all-in"}:
            aggressor[street] = seat
            aggressions[street] = aggressions.get(street, 0) + 1
        if kind in {"bet", "raise", "all-in", "call"}:
            committed.add(key)
        if kind == "call":
            called_at[key] = aggressions.get(street, 0)
        if kind == "all-in":
            all_in.add(seat)
    return hits


def _hand_reconstruction_warnings(hand: dict[str, Any]) -> list[dict[str, Any]]:
    """Warnings for the spine's extended fields (players / actions / pot / winner).

    Card-only timelines carry none of these keys and are skipped, so this is purely
    additive over the existing card checks.
    """
    warnings: list[dict[str, Any]] = []
    if not any(key in hand for key in ("players", "actions")):
        return warnings

    context = {"hand_number": hand.get("hand_number"), "time_s": hand.get("t_start"), "image": None}

    # Mirror the spine's own fatal reconstruction warnings so the export gate,
    # which reads this report and nothing else, actually rejects them.
    for code in hand.get("warnings", []) or []:
        if code in SPINE_FATAL_CODES:
            warnings.append(_warn(
                code, f"reconstruction spine raised {code}", **context,
            ))
        elif code not in RECOGNISED_SPINE_CODES:
            warnings.append(_warn(
                code, f"reconstruction spine raised unrecognised code {code}", **context,
            ))

    # A hand whose streets advanced past preflop must have a board. An empty board
    # is a legal card count everywhere else in this file (VALID_BOARD_COUNTS
    # contains 0), which is exactly why a hand that lost its whole community row
    # exported clean: nothing anywhere contradicted board_cards="" on a hand whose
    # own street summary listed a flop, a turn and a river.
    streets = hand.get("streets")
    board_is_empty = not _cards(hand.get("board") or [])
    if isinstance(streets, list) and board_is_empty:
        named = {s.get("street") for s in streets if isinstance(s, dict) and s.get("street")}
        if len(named) > 1:
            warnings.append(_warn(
                "board_empty_but_streets_advanced",
                "hand summary has no board but reports more than one street",
                streets=sorted(named), **context,
            ))

    # Second, INDEPENDENT trigger for the same fact, read off the ACTIONS.
    #
    # The street summary above is derived from the same board readings as `board`
    # itself, so when the community row is lost the two collapse together and the
    # check cannot fire on the failure it was written for. Measured: the real AR
    # 1.750 hand exported with board_cards='', streets=['preflop'] and 13 actions
    # all labelled preflop -- and scored 1.0 with no warning anywhere.
    #
    # The action sequence is an independent witness, because with no board there
    # can only ever have been ONE betting round; more than one is impossible
    # rather than merely unusual. Over 164 reconstructed hands from 13 recordings
    # -- 19 of them legitimate empty-board preflop folds -- this fires 0 times; on
    # the real destroyed hand it fires 4 times.
    #
    # The board_is_empty guard is load-bearing, not decoration: without it the
    # same evidence also fires on 13 of those 164 hands whose board and streets
    # are complete, where an extra within-street betting round is a much weaker
    # claim that this file has no measurement to support.
    reopened = (_betting_round_reopened(hand["actions"])
                if board_is_empty and isinstance(hand.get("actions"), list) else [])
    if reopened:
        warnings.append(_warn(
            "board_empty_but_streets_advanced",
            "hand has no board but its actions span more than one betting round",
            evidence=reopened[:4], **context,
        ))

    # Action-sequence legality, on the REACHABLE path this time: the check above
    # only runs on an empty board, so it is dead code on every hand that has one.
    illegal = (_action_sequence_illegal(hand["actions"])
               if isinstance(hand.get("actions"), list) else [])
    if illegal:
        warnings.append(_warn(
            "action_sequence_illegal",
            "hand contains an action sequence no client can produce",
            evidence=illegal[:4], **context,
        ))

    # Per-street pot must not shrink across streets.
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
    if isinstance(actions, list) and actions and isinstance(streets, list):
        named = {s.get("street") for s in streets if isinstance(s, dict) and s.get("street")}
        acted = {a.get("street") for a in actions if isinstance(a, dict) and a.get("street")}
        if len(named) > 1 and len(acted) == 1 and _further_betting_was_possible(hand, actions):
            # Every action of a multi-street hand landed on one street: the street
            # attribution collapsed. Observed for real as 13 actions all labelled
            # preflop on a four-street showdown, with no warning anywhere.
            warnings.append(_warn(
                "actions_collapsed_to_one_street",
                "every action is on one street but the hand reports several",
                streets=sorted(named), action_street=sorted(acted)[0], **context,
            ))
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


# Severity per warning code: the fraction of a hand's trustworthiness the fact
# destroys. A fact that makes the reconstruction WRONG (rather than merely
# incomplete) is >= 0.5, so any one of them alone puts the hand under the 0.8
# LOW_CONFIDENCE line.
WARNING_SEVERITY: dict[str, float] = {
    "board_zone_yield_zero": 0.60,
    "board_zone_yield_partial": 0.60,
    "board_empty_but_streets_advanced": 0.60,
    "side_pot_unsupported": 0.60,
    "actions_collapsed_to_one_street": 0.50,
    "amount_scale_implausible": 0.50,
    # Money of unknown size in the ledger. 0.50 is the "the reconstruction is
    # WRONG rather than merely incomplete" band: _split_source_codes routes any
    # code at or above _REJECTION_SEVERITY into rejection_codes, which is what
    # blocks the promotion to completion_status "complete".
    "amounts_unknown_in_ledger": 0.50,
    # Tab/lobby covering the table across a critical mid-hand change. Same
    # band as amounts_unknown_in_ledger: the reconstruction cannot claim the
    # unobserved actions, so export routes this into rejection_codes.
    "mid_hand_coverage_gap": 0.50,
    # An unknown starting stack is the same class of ledger unknown: every
    # derived money fact for that seat (effective stack, SPR, contributions)
    # rests on a number the reader declined to supply, and accounting fails
    # closed on it permanently. >= 0.5 routes it into rejection_codes.
    "starting_stack_unknown": 0.50,
    # The hand's own action list contradicts itself, so every derived number
    # (pot, contributions, hero net) rests on a sequence that never happened.
    "action_sequence_illegal": 0.50,
    # The states this hand was reconstructed from are not listed in the order
    # their own timestamps record. This report re-orders them before judging the
    # board and the streets, so the verdict beside this code is sound -- but the
    # SPINE that produced the hand walked the list as given (build_states, the
    # run-length debounces, fold detection, action attribution all consume it in
    # list order), so its players, actions, pot and winner describe a sequence
    # the recording did not have. That is the "wrong rather than merely
    # incomplete" band, so it routes into rejection_codes.
    "state_time_order": 0.50,
    "stack_ledger_incoherent": 0.50,
    "anchor_unavailable": 0.40,
    "reconciliation_failed": 0.35,
    # See RECOGNISED_SPINE_CODES: 1 of the 4 contested hands measured on the
    # development corpus has a confirmed-wrong card. 0.35 lands a contested hand
    # at 0.65 -- tagged LOW_CONFIDENCE and held for review, not shipped as fact.
    "board_card_identity_split": 0.35,
    "hero_card_identity_split": 0.35,
    "hero_seat_mismatch": 0.30,
    "board_regression": 0.30,
    "street_order_issue": 0.30,
    "invalid_board_count": 0.30,
    "invalid_hero_count": 0.30,
    "duplicate_visible_cards": 0.30,
    "pot_regression": 0.25,
    "action_street_invalid": 0.25,
    "action_street_order": 0.25,
    "position_issue": 0.20,
    "missing_label": 0.10,
}
UNKNOWN_WARNING_SEVERITY = 0.20


def _confidence_score(warnings: list[dict[str, Any]]) -> float:
    """Severity-weighted confidence in the RECONSTRUCTION.

    The previous form was a warning DENSITY -- ``1 - warnings / (3 * states)`` --
    which made a hand's score depend on how many frames it happened to span. Two
    fatal warnings over 61 checked states scored 0.989 on a session in which every
    board had been destroyed. Warning counts do not divide by frame count; a fact
    that invalidates a hand invalidates it however long the hand was.
    """
    return round(max(0.0, 1.0 - sum(
        WARNING_SEVERITY.get(w.get("code"), UNKNOWN_WARNING_SEVERITY) for w in warnings
    )), 3)


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
            "confidence_score": _confidence_score(warnings),
            "warnings": warnings,
        })

    hand_scores = [report["confidence_score"] for report in reports]
    return {
        "summary": {
            "malformed": False,
            "hands": len(reports),
            "checked_states": checked_states,
            "warning_hands": sum(1 for report in reports if report["warning_count"]),
            "total_warnings": total_warnings,
            # A session is only as trustworthy as its WORST hand. Averaging (which
            # the density form effectively did across the whole timeline) is what
            # let a session containing destroyed boards report 0.989. Reported for
            # operators; never used as a gate -- the gate is per hand.
            "confidence_score": round(min(hand_scores), 3) if hand_scores else 0.0,
            "median_hand_confidence": (
                round(median(hand_scores), 3) if hand_scores else 0.0
            ),
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
