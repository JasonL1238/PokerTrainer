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
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any

from cv_lab.scripts.pipeline import region_detections as rd
from cv_lab.scripts.pipeline.build_yolo_card_timeline import (
    _best_board,
    _stage,
    _street_events,
)
from cv_lab.scripts.pipeline.landmark_anchor import ANCHOR_MIN_SESSION_FITS

DEFAULT_OUT = "cv_lab/results/yolo_hand_timeline.json"

_STREET_BY_COUNT = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
_EPS = 1e-6

# Run-length debounce thresholds are calibrated at this sampling interval. When
# frames are sampled FINER than this, fixed-duration display transients (the
# fold grey-out glow, card-flip / bet animations) span proportionally MORE
# samples, so a fixed sample-count threshold would let a longer transient slip
# through. Sample-count thresholds are therefore scaled up in inverse proportion
# to the interval to hold the same wall-clock rejection window. Coarser sampling
# keeps the calibrated minimums (never scaled below 1.0x), so existing 2.0s
# fixtures reconstruct byte-identically.
_CALIB_INTERVAL_S = 2.0
_FOLD_MIN_RUN_CALIB = 3   # hero greyed-out fold: reject True-runs < 3 samples @2s

# Layout supportedness facts -- anchor-health based, not window-size based.
# The old W x H lower bound (1272x896) was anti-correlated with reality on the
# first two sessions it judged: it stamped a 1052x732 recording that
# reconstructed at a 4.5% operator-flagged error rate "-unsupported", and a
# 2114x1414 recording that failed for game-structure reasons "supported".
# What actually bounds the readers is the fitted table scale:
#   * _REF_GLYPH_H_PX: HUD glyph height at reference scale. The corpus's
#     native run heights span 12..31px at fitted scales ~0.62..1.34; both
#     extremes divide back to ~19..23px, i.e. ~21px at s=1.
#   * _MIN_RENORMALIZABLE_GLYPH_PX: the smallest native glyph the renormalized
#     OCR re-read has been VALIDATED on. The pinned 1052x732 crops (s=0.512,
#     glyphs 10-11px) read correctly; the adversarial sweep's confident-wrong
#     upscales live at 7-9px. 9px is the boundary between "renormalizable"
#     and "unvalidated".
_REF_GLYPH_H_PX = 21.0
_MIN_RENORMALIZABLE_GLYPH_PX = 9.0

# Action types that MOVE MONEY. Only these can carry "an amount of unknown size";
# a check or a fold legitimately has amount None and always did, which is exactly
# why None on the Action model cannot be overloaded to mean unknown money.
_MONEY_ACTIONS = frozenset({"bet", "raise", "call", "all-in"})

# Winner plausibility. A seat that swept the pot gained approximately the pot, so
# a candidate "gain" far below it is stack OCR jitter rather than a sweep.
# Measured over every winner the spine reconstructs on the 5 development
# geometries (19 hands): gain/pot lands in [0.962, 1.034], tightly clustered
# because the sweep IS the pot. The one measured defect sits at 0.029 -- 0.5 BB
# of jitter between two consecutive reads on a 17.5 BB pot. 0.5 is 0.52x the
# smallest real ratio and 17x the defect, and it deliberately tolerates a
# main-pot winner taking only half of a main+side total (the corpus's one
# side-pot hand still measures 1.0, so that headroom is unexercised margin).
# Inert when the pot is unknown: a relative floor needs a denominator.
_WIN_GAIN_MIN_POT_RATIO = 0.5

# Chip conservation slack, in BB. See _contribution_residual. A negative residual
# is arithmetically impossible under incremental amount semantics, so the only
# tolerance needed is for rounding on the two-decimal reads that feed both sides.
# Measured over the development corpus after the duplicate-action fix, every
# healthy hand's residual is >= 0; the smallest defect is -3.0 BB and the largest
# -122.9 BB. 1.0 BB is well under the smallest defect and well above any rounding.
_CONTRIBUTION_RESIDUAL_SLACK_BB = 1.0

# Stack-ledger coherence (rejection signal C). Chips only ever LEAVE a player
# until the pot is awarded, so no pre-settlement stack can exceed that player's
# own stack at the start of the hand. That is exact and threshold-free, and it is
# deliberately NOT "the stack must never rise": a rise IS legal -- the client
# returns the uncalled part of an over-shove before the sweep, which on the
# development corpus shows up as 0.0 -> 18.9 (g0723a hand 2, seat 0 shoved 167.0
# into a 148.1 call) and 0.0 -> 128.4 (hand 5, seat 5). Both are real, and a
# "never rises" rule flags both. What cannot happen is a seat holding MORE than
# it started with: a truncated "218 BB" read as 21.0 made seat 3 start the hand
# on 21.0 and hold 210.5 two actions later, and that hand exported at confidence
# 1.0 with warnings=none.
# The +1.0 BB slack absorbs read jitter without admitting any measured defect
# (the smallest is a full order of magnitude).
_STACK_RISE_SLACK_BB = 1.0
# A mid-hand coverage gap is a stretch where the table was not observed (tab
# in front, lobby, modal) long enough that consecutive *table* states are
# farther apart than a normal sample. Gaps at or below this multiple of the
# sampling interval are treated as brief HUD blips the existing debounce
# already handles; longer gaps with a critical state change are unobserved.
_COVERAGE_GAP_INTERVAL_MULT = 2.5
_COVERAGE_GAP_MIN_S = 2.5


def _sampling_interval(frames: list[rd.Frame]) -> float:
    """Median positive gap (s) between consecutive sampled frame times.

    The spine self-calibrates its run-length debounces to whatever rate the
    frames were sampled at, so 2.0s and 1.0s captures both reconstruct correctly
    with no caller change. Falls back to the calibration interval when it can't
    be measured (0/1 frame, or non-increasing times)."""
    ts = [f.time_s for f in frames if f.time_s is not None]
    deltas = sorted(b - a for a, b in zip(ts, ts[1:], strict=False) if b - a > _EPS)
    return deltas[len(deltas) // 2] if deltas else _CALIB_INTERVAL_S


def _scaled_run(calib_count: int, interval_s: float) -> int:
    """Scale a sample-count threshold from the 2.0s calibration to the actual
    sampling interval, never below the calibrated value (finer only tightens)."""
    scale = _CALIB_INTERVAL_S / interval_s if interval_s > _EPS else 1.0
    return max(calib_count, round(calib_count * scale))


def _is_table_frame(frame: rd.Frame) -> bool:
    return getattr(frame, "screen", "table") == "table"


def _coverage_gap_threshold_s(interval_s: float) -> float:
    return max(_COVERAGE_GAP_INTERVAL_MULT * interval_s, _COVERAGE_GAP_MIN_S)


def _critical_unobserved_change(prev: dict[str, Any], cur: dict[str, Any]) -> bool:
    """True when a mid-hand time gap likely hid a state change we must not invent.

    Brief overlays that leave board/stacks/roster/dealer/hero unchanged are safe
    to stitch. A board advance, pot move, roster change, dealer move, hero-card
    change, or meaningful stack debit across an unobserved stretch means the
    actions in the hole were not seen -- record the gap, do not fabricate them.
    """
    if prev.get("board_cards") != cur.get("board_cards"):
        return True
    if prev.get("hero_cards") != cur.get("hero_cards") and (
        prev.get("hero_cards") or cur.get("hero_cards")
    ):
        return True
    if set(prev.get("dealt_in") or []) != set(cur.get("dealt_in") or []):
        return True
    prev_dealer, cur_dealer = prev.get("dealer_seat"), cur.get("dealer_seat")
    if (
        prev_dealer is not None
        and cur_dealer is not None
        and prev_dealer != cur_dealer
    ):
        return True
    prev_pot, cur_pot = prev.get("pot"), cur.get("pot")
    if (
        prev_pot is not None
        and cur_pot is not None
        and abs(cur_pot - prev_pot) > 1.0 + _EPS
    ):
        return True
    seats = set(prev.get("stacks") or []) | set(cur.get("stacks") or [])
    for seat in seats:
        before = (prev.get("stacks") or {}).get(seat)
        after = (cur.get("stacks") or {}).get(seat)
        if before is not None and after is not None and abs(before - after) > 0.5 + _EPS:
            return True
    return False


# --------------------------------------------------------------------------- #
# Pass 1: per-frame table state + collapse to distinct states
# --------------------------------------------------------------------------- #
def _session_anchor(frames: list[rd.Frame]) -> rd.TableAnchor | None:
    """Median table transform over every frame that anchors on its own.

    Used only as the fallback for a frame whose own ``stack_text`` constellation
    is too sparse to fit. Safe because the table does not move within a recording:
    the session-median transform reproduces the per-frame transform to within
    0.0038 reference-y units at worst (p50 0.0001-0.0005) over the 5 development
    geometries -- an order of magnitude inside the card-zone band guards.
    """
    fits = [
        a
        for a in (rd.anchor_for_frame(f) for f in frames if _is_table_frame(f))
        if a is not None
    ]
    if len(fits) < ANCHOR_MIN_SESSION_FITS:
        return None
    return rd.TableAnchor(
        s=median(a.s for a in fits),
        tx=median(a.tx for a in fits),
        ty=median(a.ty for a in fits),
        resid=median(a.resid for a in fits),
        n_points=0,
        source="session",
    )


# DELETED IN THIS PHASE: `_reject_stack_outliers`, `_STACK_OUTLIER_RATIO`,
# `_STACK_OUTLIER_MIN_READS` and `_DECIMAL_NOT_LOCATED` -- the sibling-median net
# that dropped a stack read at least 6.0x the frame's other stacks when the read
# had located no decimal point.
#
# Its premise was "no field in this corpus legitimately exceeds ~400 BB", measured
# on five recordings, and it is FALSE on the sixth development recording:
# clubwpt_session_01 carries 541 reads at or above 1000 BB (max 1157.10), every
# one of them provable under the reader's new contract and legible on screen.
# Instrumented on that recording the net fires on 83 frames and drops a legible
# 1110.0 BB stack against a ~150 sibling median every single time, then raises
# `amount_scale_implausible` -- a SPINE_FATAL code at severity 0.50 -- on three of
# its hands. Measured effect of deleting it under the new reader contract: exports
# rise from 16 to 19 (clubwpt_session_01 6 -> 9) with no other recording changed.
#
# The failure it was built for -- a decimal dropped by the OCR, "79.7" read as
# "797.0" -- is now refused AT THE READER by P5 (`integer_over_decimal_band`),
# where the evidence for it exists: the run's own inter-digit spacing. A
# magnitude comparison against unrelated seats never had that evidence; it had a
# ratio, and the ratio does not separate a broken read from a deep stack.
#
# `amounts_rejected` / `stack_outlier_check_skipped` went with it. A counter that
# can only ever report 0 reads as "checked and clean", which is the exact failure
# mode this file keeps flagging elsewhere.


def _stack_ledger_violations(
    pre_settlement: list[dict[str, Any]], players: Iterable[int]
) -> list[dict[str, Any]]:
    """Seats holding MORE than they started the hand with, before settlement.

    Evaluated on the pre-settlement window only, so the winner's sweep -- the one
    way a player legitimately ends up above its starting stack -- is out of scope
    by construction. See _STACK_RISE_SLACK_BB for why this is not "the stack must
    never rise".

    A seat is only in the ledger while it still holds cards. A seat that has
    folded is out of the hand and the client will top it back up to the buy-in
    without waiting for the hand to finish -- measured on the baseline recording,
    seat 6 folds on the turn and refills 164.0 -> 200.0 while the river is still
    being played.
    """
    hits: list[dict[str, Any]] = []
    for seat in players:
        series = [(s["time_s"], s["stacks"][seat]) for s in pre_settlement
                  if seat in s.get("stacks", {}) and s["stacks"][seat] is not None
                  and seat in (s.get("dealt_in") or [])]
        if not series:
            continue
        start = series[0][1]
        for time_s, value in series[1:]:
            if value > start + _STACK_RISE_SLACK_BB:
                hits.append({"seat": seat, "time_s": time_s, "start": start, "to": value})
                break
    return hits


def _contribution_residual(hand: dict[str, Any]) -> float | None:
    """pot - sum(action amounts), or None when either side is unknown.

    Action amounts are INCREMENTAL (each is the chips that seat added at that
    moment), so every chip in an action is a chip in the pot and the sum can never
    EXCEED it. A POSITIVE residual is expected and uninteresting -- blinds and
    antes are already posted before the first observed state. A NEGATIVE residual
    means the reconstruction booked money the pot does not contain: a duplicated
    action, or an uncalled over-shove recorded at its full size against a shorter
    caller. Four of 15 exported development hands were negative, one by 122.9 BB
    on a 212.5 BB pot, all at confidence 1.0 with warnings=none.
    """
    pot = hand.get("pot")
    if pot is None:
        return None
    known = [a["amount"] for a in hand.get("actions") or [] if a.get("amount") is not None]
    if not known:
        return None
    return round(pot - sum(known), 2)


def _uncalled_shove_headroom(actions: Iterable[dict[str, Any]]) -> float:
    """The largest amount an all-in seat could have had RETURNED to it uncalled.

    A shove above what any single opponent can match is refunded before the pot is
    swept, so its excess is money the seat pushed and the pot correctly never
    contained. That makes a negative contribution residual legal to exactly this
    depth and no further. Measured on the development corpus, both negative
    residuals are covered: -13.4 against 18.9 of headroom, and -122.9 against
    128.4. The duplicate-action defect this net exists to catch had a residual of
    -3.0 with no all-in anywhere, i.e. zero headroom.
    """
    committed: dict[int, float] = {}
    all_in: set[int] = set()
    for a in actions:
        seat = a.get("seat")
        if seat is None:
            continue
        if a.get("amount") is not None:
            committed[seat] = committed.get(seat, 0.0) + a["amount"]
        if a.get("action_type") == "all-in":
            all_in.add(seat)
    headroom = 0.0
    for seat in all_in:
        others = max((v for s, v in committed.items() if s != seat), default=0.0)
        headroom = max(headroom, committed.get(seat, 0.0) - others)
    return round(headroom, 2)


# The sign a published `result` FORBIDS the hero's net from taking. Stated as a
# table over the CLOSED vocabulary the producer emits (see the winner_seat /
# hero_folded branch: "Hero wins", "Villain wins", "Hero folds", "") so that the
# check covers the space instead of enumerating two of its members.
#
# "" is absent deliberately and is not an oversight: an unresolved hand asserts
# nothing about who won, so no net contradicts it. "Hero folds" IS a claim -- a
# hero who folded forfeits what it committed and cannot come out ahead. (A hero
# who bets everyone off the pot is winner_seat == 0, i.e. "Hero wins"; the fold
# branch is only reached when the hero itself surrendered.) Its omission is the
# round-2 finding: the check could not fire on a folded hero with a positive net,
# which is the same two-fields-off-one-ledger contradiction as the other two.
_RESULT_FORBIDS_NET_SIGN = {
    "Hero wins": -1,      # a hero who won the pot cannot net negative
    "Villain wins": +1,   # a hero who lost the pot cannot net positive
    "Hero folds": +1,     # a hero who folded cannot net positive
}


def _result_contradicts_hero_net(result: str, hero_bb_won: float | None) -> bool:
    """True when the published result and the hero's net disagree in SIGN.

    Two fields derived from the same stack ledger, never compared: a hand shipped
    result="Villain wins" alongside hero_bb_won=+65.2 (the hero had in fact bet
    both opponents off the turn). Zero is not a contradiction -- a hero who folded
    for free, or whose net is unobserved, nets 0 under either result.

    Latent on the development corpus in the "Hero folds" direction: all 5 folded
    hands net -0.5, 0.0, or unknown. It is covered because the 0.5 BB read jitter
    that motivated _WIN_GAIN_MIN_POT_RATIO is enough to make a folded hero's net
    read positive, and there was no check to catch it.
    """
    if hero_bb_won is None or abs(hero_bb_won) <= _EPS:
        return False
    forbidden = _RESULT_FORBIDS_NET_SIGN.get(result)
    if forbidden is None:
        return False
    return hero_bb_won > 0 if forbidden > 0 else hero_bb_won < 0


def _frame_state(frame: rd.Frame, session_anchor: rd.TableAnchor | None = None) -> dict[str, Any]:
    view = rd.assign_regions(frame, anchor=rd.anchor_for_frame(frame) or session_anchor)
    seats = view["seats"]
    dealt_in = sorted(i for i, info in seats.items() if info["card_back"])
    stacks = {i: info["stack"] for i, info in seats.items() if info["stack"] is not None}
    # UNKNOWN, carried as a first-class per-seat fact instead of being dropped.
    # `stacks` omits a refused seat exactly as it omits a seat with no box, and
    # note 10 recorded what that costs: "an unknown stack is dropped, not carried;
    # unknown is neither zero nor unknown, it is invisible". Every consumer of
    # `stacks` must consult this map before falling back to another state.
    stacks_unknown = dict(view.get("stack_unknown") or {})
    unknown_by_code = dict(view.get("amounts_unknown_by_code") or {})
    # Unknowns MINTED HERE (pot_zero_impossible / bet_zero_impossible) increment
    # BOTH the by-code map and the count. They used to go into the map only, so
    # `amounts_unknown` and "the per-reason breakdown of the count above" (the
    # exporter's own words) were computed from different populations and the
    # breakdown could sum past the count -- the two fields disagreed by
    # construction, and a pot dropout never reached the unread-amount rate that
    # caps confidence.
    extra_unknown = 0

    def _bump(code: str) -> None:
        nonlocal extra_unknown
        unknown_by_code[code] = unknown_by_code.get(code, 0) + 1
        extra_unknown += 1

    # A bet_text of exactly 0 is never real: the client renders no bet row at all
    # when a seat has nothing in front of it, so a "0 BB" bet is a crop holding
    # only chip sprites, whose white annulus matches the '0' template. That makes
    # it evidence of NO BET (the seat has nothing on the felt), not of an
    # unreadable bet -- so it is excluded from `bets` exactly as before and is NOT
    # a refusal. It is named on the record rather than applied silently, so a
    # zero-storm on the bet region is visible instead of looking like "no bet".
    bets = {i: info.get("bet") for i, info in seats.items()
            if info.get("bet") is not None and info["bet"] > 0.0}
    bets_unknown = dict(view.get("bet_unknown") or {})
    bet_zero_impossible = sorted(i for i, info in seats.items()
                                 if info.get("bet") is not None and info["bet"] <= 0.0)
    for _ in bet_zero_impossible:
        _bump("bet_zero_impossible")
    pills = {i: info["pill_action"] for i, info in seats.items() if info["pill_action"]}
    board = view["board"]
    # A pot text of exactly 0 is never real once a hand is in progress (blinds and
    # antes are posted before the first observed state). Unlike the bet row, the
    # client ALWAYS renders a pot, so a zero here is a dropout on the pot region
    # and the amount is UNKNOWN -- named, and carried, so nothing substitutes a
    # neighbouring state's pot for it.
    pot = view["pot"]
    pot_unknown = view.get("pot_unknown")
    if pot is not None and pot <= 0.0:
        pot = None
        pot_unknown = "pot_zero_impossible"
        _bump("pot_zero_impossible")
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
        "pot_unknown": pot_unknown,
        # A detected second pot. The spine does NOT reconstruct side-pot
        # arithmetic; carrying the value is what lets the hand be rejected
        # explicitly instead of the side pot being silently discarded.
        "side_pot": view.get("side_pot"),
        "side_pot_unknown": view.get("side_pot_unknown"),
        "dealt_in": dealt_in,
        "stacks": stacks,
        "stacks_unknown": stacks_unknown,
        "bets": bets,
        "bets_unknown": bets_unknown,
        "bet_zero_impossible": bet_zero_impossible,
        "pills": pills,
        "dealer_seat": view["dealer_seat"],
        "active_seat": view["active_seat"],
        "hero_seat_mismatch": view.get("hero_seat_mismatch", False),
        # POSITIVE confirmation, not the absence of a contradiction: see
        # region_detections.assign_regions.
        "hero_seat_confirmed": view.get("hero_seat_confirmed", False),
        "pot_text_off_column": view.get("pot_text_off_column", 0),
        "stack_conflicts": view.get("stack_conflicts", 0),
        "bet_conflicts": view.get("bet_conflicts", 0),
        # Anchor health. Deliberately NOT part of _signature: a residual that
        # drifts by a thousandth between frames must not create a distinct state.
        "anchor_ok": view.get("anchor_ok", False),
        "anchor_source": view.get("anchor_source"),
        "unanchored_cards": view.get("unanchored_cards", 0),
        # Read-quality signals. Also deliberately outside _signature: a rejected
        # amount already changed `stacks`, which IS in the signature.
        "board_row_missed": view.get("board_row_missed", False),
        "board_row_partial": view.get("board_row_partial", False),
        "pot_collapses_repaired": 0,
        "amounts_unknown": view.get("amounts_unknown", 0) + extra_unknown,
        "amounts_unknown_by_code": unknown_by_code,
        # Seats whose money movement across a transition could not be MEASURED
        # because a stack read was refused. Filled by _reconstruct_actions, which
        # is where transitions are walked; declared here so the key always exists.
        "unmeasured_transitions": [],
        # Decoder-proven static span: the table provably showed THIS state until
        # here (a later sample resolved to the same decoded frame). Observation
        # continuity for the gap math, not a distinct reading.
        "observed_static_until_s": getattr(frame, "observed_static_until_s", None),
        # Mid-hand coverage: seconds the table was UNOBSERVED before this state
        # (since the last table observation, including same-signature frames
        # and proven-static spans, not merely the last distinct state). Filled
        # in build_states / action walk.
        "prior_gap_s": 0.0,
        "sampling_interval_s": 0.0,
        "coverage_gap": False,
        # Set by _settle_index when a settlement-scan transition is skipped
        # because this state's pot read was refused (the old code substituted a
        # STALE pot from an earlier state for the unread one).
        "settle_scan_skipped": False,
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
        # A side pot appearing or clearing is a genuine table change, not jitter.
        state.get("side_pot"),
        state["dealer_seat"],
        state["active_seat"],
    )


def _flag_identity_splits(raw: list[dict[str, Any]], key: str) -> None:
    """Mark states where a card slot's IDENTITY is contested, before the debounce
    absorbs the disagreement.

    A same-length change in a card list is not board growth and not a partial
    read: it is the classifier reading one slot two different ways while the card
    sits on the table. ``_debounce_cards`` is built to swallow exactly that, and
    afterwards nothing downstream can tell "the classifier agreed with itself"
    from "the classifier was overruled". The majority is then published as fact --
    and measured on the 07-23 3.21 PM recording the majority is WRONG: the turn
    card reads 8d eight times and 8h twice, the card on screen is unmistakably the
    eight of hearts, and 'Js 6h 4c 8d' exported at confidence 1.0 with tags [].
    The real board has two hearts and a live flush draw; the exported one does not.

    Compared against the previous non-empty READING rather than against the
    debounce's accepted value: when the excursion is the second reading of a
    street there is nothing accepted yet, and the contest would be invisible.

    Measured over the five development recordings: 4 of 21 hands carry a contest.
    Different-length disagreements (board growth, partial reads) are NOT contests
    and are excluded -- 8 of those occur.
    """
    prev: tuple | None = None
    for state in raw:
        cards = tuple(state[key])
        if not cards:
            prev = None
            continue
        if prev is not None and len(cards) == len(prev) and cards != prev:
            state[f"{key}_identity_split"] = True
        prev = cards


def _debounce_cards(raw: list[dict[str, Any]], key: str) -> None:
    """Debounce card-list fields (hero_cards / board_cards) in place over the raw
    per-frame states. A NON-EMPTY reading that differs from the accepted one must
    be confirmed by the next non-empty reading, else it is replaced by the carried
    value -- or dropped entirely when there is nothing to carry. An empty reading
    RESETS the carry: cards leaving the table (sweep / new deal) must not leak the
    previous hand's cards into the next one.

    A REPLACEMENT (a change that is not board growth) is additionally rejected as
    a transient if the accepted value reappears before the table is next swept.
    Real new hands are always preceded by a sweep (an empty reading), so the prior
    cards never legitimately reappear mid-stream: a Tc->Ts->Tc suit misread is a
    read excursion, not a boundary. This is interval-independent -- it rejects the
    excursion however many frames it spans -- and at 2s (where such excursions are
    single-frame) it is equivalent to the next-reading confirmation below, so 2s
    reconstructions are unchanged. Left unrejected, a multi-frame hero excursion at
    1s fabricates a hand boundary (dealer/hero misreading together).

    See _flag_identity_splits, which runs first and records the disagreements this
    function is about to absorb.
    """
    accepted: tuple | None = None
    for idx, state in enumerate(raw):
        cards = tuple(state[key])
        if not cards:
            accepted = None
            continue
        if accepted is not None and cards == accepted:
            continue
        # revert check: a non-growth change whose accepted value returns before a
        # sweep is a misread excursion, regardless of how long it persists.
        is_growth = (accepted is not None and len(cards) > len(accepted)
                     and cards[: len(accepted)] == accepted)
        if accepted is not None and not is_growth:
            reverts = False
            for nxt in raw[idx + 1:]:
                nxt_cards = tuple(nxt[key])
                if not nxt_cards:
                    break  # table swept: a genuine boundary window, not a revert
                if nxt_cards == accepted:
                    reverts = True
                    break
                if len(nxt_cards) > len(cards) and nxt_cards[: len(cards)] == cards:
                    break  # `cards` itself grew (committed as real) before any revert
            if reverts:
                state[key] = list(accepted)
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


def _resolve_identity_by_majority(hand: list[dict[str, Any]]) -> None:
    """Give each card slot ONE identity per hand: the reading most states saw.

    ``_debounce_cards`` is sequential -- it accepts a change the NEXT reading
    confirms, and reverts an excursion only if the accepted value comes back
    before the table is swept. That leaves one shape unhandled: a misread that
    starts partway through a hand and never reverts because the hand ENDS first.
    Measured on a 1344x836 recording, hero held Kd Tc for 31 consecutive states
    and Ts for the last 8 before the sweep. Nothing reverted, so the tail was
    confirmed and published, and one hand carried two different hole-card pairs.
    Hole cards are dealt once and do not change, so that is not low confidence,
    it is incoherent -- and no consumer of the hand can tell which half to trust.

    Runs per SEGMENT, on the boundaries ``_segment`` already found, and never on
    a raw span of its own devising. An earlier revision segmented on "a maximal
    run of non-empty readings", reasoning that cards leaving the table end the
    hand. They do not always: the sweep can fall between samples or be collapsed
    away, so that rule glued several deals together and gave one hand the hole
    cards of a hand two earlier -- which then duplicated a board card, a strictly
    worse defect than the one being fixed. A deal boundary has exactly one
    definition in this module and this is not the place to add a second.

    Board growth survives untouched because each state keeps its own LENGTH; only
    the identity at a slot is rewritten, and slot 0 of a flop is the same card on
    the river. Ties go to the reading seen first, so the result never depends on
    dict ordering.

    Deliberately does NOT silence _flag_identity_splits: the majority is what to
    publish, not proof it is right. The 07-23 turn card read 8d eight times and
    8h twice with the eight of hearts on screen, so a hand whose slots were
    contested still exports contested, at reduced confidence, for review.
    """
    for key in ("hero_cards", "board_cards"):
        readings = [state for state in hand if state[key]]
        if not readings:
            continue
        winners: dict[int, str] = {}
        for slot in range(max(len(state[key]) for state in readings)):
            counts: dict[str, int] = {}
            first_seen: dict[str, int] = {}
            for index, state in enumerate(readings):
                if slot >= len(state[key]):
                    continue
                card = state[key][slot]
                counts[card] = counts.get(card, 0) + 1
                first_seen.setdefault(card, index)
            if counts:
                winners[slot] = max(counts, key=lambda c: (counts[c], -first_seen[c]))
        for state in readings:
            resolved = [winners[slot] for slot in range(len(state[key]))]
            # Never MANUFACTURE a duplicate. Two slots whose majorities collide
            # would publish one card twice -- and across hero and board it would
            # deal a card that is already on the table. Both are worse claims
            # than the disagreement they came from, so such a state is left as
            # the debounce left it.
            others = set(state["board_cards" if key == "hero_cards" else "hero_cards"])
            if len(set(resolved)) == len(resolved) and not (set(resolved) & others):
                state[key] = resolved


def _debounce_bool_confirm(raw: list[dict[str, Any]], key: str, min_run: int = 3,
                            reset_key: str | None = None) -> None:
    """Only accept a True reading that's part of a run of >= min_run
    consecutive True raw readings; shorter runs are rejected as a blip (e.g.
    a card-flip/bet animation, or a brief glow/highlight effect, transiently
    darkening the hero-card crop for a couple of samples). A real greyed-out
    fold holds for the rest of the hand -- empirically dozens of samples --
    so this costs it nothing.

    ON THE VALUE 3 (via _FOLD_MIN_RUN_CALIB), stated honestly. The figure was
    originally chosen against a false positive observed on a recording later
    recognised as belonging to the held-out split, so that observation is not
    usable evidence and is not cited here. Re-measured on the six DEVELOPMENT
    recordings (1s sampling, threshold scaled to 6): confirmed hero folds run
    6-64 samples, every raw dim run terminates at a hand boundary or the
    recording end, and no run reverts to bright mid-hand -- so the development
    corpus contains no false positive at all and cannot distinguish 1 from 3.
    3 is RETAINED as the conservative resolution of that open interval: the
    error mode of a threshold set too high is a missed fold (result stays
    "unobserved" -- a coverage loss), while too low fabricates a hero fold (a
    wrong result), and only the second is a release blocker. Lowering it would
    need positive evidence no permitted recording supplies.

    ``reset_key``, when given, breaks a run wherever that field's value
    changes (e.g. hero_cards) -- a run must not bridge a hand boundary: the
    previous hand's real terminal fold-darkening reads as a persistent run
    that would otherwise still be "confirmed" one sample into the NEXT hand,
    before that hand's own cards have been read even once (seen for real in
    v05#5's hero_dim bleeding a stale True into the new hand's first state)."""
    n = len(raw)
    accepted = [False] * n
    i = 0
    while i < n:
        if raw[i][key]:
            j = i
            while j < n and raw[j][key] and (
                reset_key is None or j == i or raw[j][reset_key] == raw[j - 1][reset_key]
            ):
                j += 1
            if j - i >= min_run:
                for k in range(i, j):
                    accepted[k] = True
            i = j
        else:
            i += 1
    for i, state in enumerate(raw):
        state[key] = accepted[i]


def build_states(
    frames: list[rd.Frame],
    *,
    sampling_interval_s: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (distinct states, events). Raw per-frame states are debounced first
    (cards, pot, stacks, bets -- single-frame OCR/classifier blips are rejected
    unless the next reading confirms them), then collapsed to distinct states.

    Nontable frames (tab-in-front / lobby / modal) are omitted from inference
    state: their timestamps create a ``prior_gap_s`` on the next table state so
    action reconstruction can refuse to invent what happened underneath.

    ``sampling_interval_s`` is the cadence the caller actually sampled at. Pass
    it when known: on a variable-rate recording the duplicate-dropping sampler
    emits frames at the recording's own change spacing, so measuring the
    cadence from frame times over-estimates it and loosens every
    interval-scaled debounce and gap threshold. When None it is measured from
    the frame times, which is exact for dense fixtures.
    """
    table_frames = [f for f in frames if _is_table_frame(f)]
    session_anchor = _session_anchor(table_frames)
    interval_s = (
        float(sampling_interval_s)
        if sampling_interval_s and sampling_interval_s > _EPS
        else _sampling_interval(table_frames if table_frames else frames)
    )
    raw = [_frame_state(f, session_anchor) for f in table_frames]
    for state in raw:
        state["sampling_interval_s"] = interval_s
    # Record the identity contests BEFORE the debounce absorbs them.
    _flag_identity_splits(raw, "hero_cards")
    _flag_identity_splits(raw, "board_cards")
    _debounce_cards(raw, "hero_cards")
    _debounce_cards(raw, "board_cards")
    _debounce_bool_confirm(raw, "hero_dim",
                           min_run=_scaled_run(_FOLD_MIN_RUN_CALIB, interval_s),
                           reset_key="hero_cards")
    for state in raw:
        state["stage"] = _stage(len(state["board_cards"]))

    states: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    last_sig: tuple | None = None
    prev: dict[str, Any] | None = None
    # The latest time the table was provably OBSERVED, advanced by every table
    # frame -- including same-signature frames the collapse below skips, and
    # decoder-proven static spans. ``prior_gap_s`` is the unobserved stretch
    # before a state, so it must be measured from here, not from the previous
    # DISTINCT state: a table that provably sat still for a minute and then
    # changed was watched the whole time, and reading that minute as a coverage
    # hole would refuse a perfectly observed hand.
    observed_until: float | None = None

    for state in raw:
        sig = _signature(state)
        static_until = state.get("observed_static_until_s")
        cur_observed = state["time_s"] if static_until is None else max(
            state["time_s"], static_until
        )
        if sig == last_sig:
            if observed_until is not None:
                observed_until = max(observed_until, cur_observed)
            continue
        if prev is not None:
            since = prev["time_s"] if observed_until is None else observed_until
            state["prior_gap_s"] = max(0.0, state["time_s"] - since)
        else:
            state["prior_gap_s"] = 0.0
        observed_until = cur_observed
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
# Pass 2: segment into hands (button movement / board reset / fresh hero cards)
# --------------------------------------------------------------------------- #
def _board_continues(states: list[dict[str, Any]], i: int, ref_board: list[str]) -> bool:
    """After an empty board at states[i], does the SAME deal's board come back?

    True when the next non-empty board reading is the reference board or that
    board grown by later streets -- a detection dropout mid-hand, not a
    teardown. The card debounce deliberately never repairs empty readings (an
    empty must reset the carry so one hand's cards cannot leak into the next),
    so a one-state board dropout reaches segmentation and must be told apart
    from a real reset here, by whether the board it interrupted resumes."""
    ref = tuple(ref_board)
    for nxt in states[i + 1:]:
        cards = tuple(nxt["board_cards"])
        if not cards:
            continue
        return cards == ref or (len(cards) > len(ref) and cards[: len(ref)] == ref)
    return False


def _segment(states: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """A new hand starts at the first state, no longer showing the old board,
    with table evidence of a fresh deal. Three CO-EQUAL signals, any of which
    cuts -- none depends on the hero being dealt in, so a session the hero sits
    out entirely still segments:

    - the dealer button MOVED (it moves exactly once per deal), confirmed by
      the next dealer reading so a one-state marker misread cannot split a hand;
    - the board RESET after showing cards, confirmed by the old board never
      resuming (see _board_continues -- a mid-hand detection dropout resumes);
    - the hero's hole cards changed to a NEW pair after passing through empty.
      The empty passage is the teardown evidence: cards leave the hero zone
      only at a fold or the sweep, so a "new pair" with no empty reading in
      between is read noise (suit misreads, showdown reveals landing in the
      hero zone), never a real boundary.

    A signal only arms the boundary; the cut itself lands on the first state
    that shows the NEW deal (hole-card backs at any seat, two hero cards, or a
    fresh board). The client tears a hand down in stages -- sweep, pot reset,
    button advance, cards cleared -- and those teardown states carry the old
    hand's settlement, so they must trail it (where _trim_trailing_next_deal
    prunes them), not head the next hand and pollute its opening roster and
    starting stacks. Signals are additionally gated on the state not still
    showing the old board, so a lingering board can never be cut into the next
    hand's street logic."""
    n = len(states)
    # next_dealer[i]: the first dealer reading strictly after states[i].
    next_dealer: list[int | None] = [None] * n
    last_read: int | None = None
    for j in range(n - 1, -1, -1):
        next_dealer[j] = last_read
        if states[j]["dealer_seat"] is not None:
            last_read = states[j]["dealer_seat"]

    hands: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_hero: list[str] = []
    hero_swept = False        # hero zone read empty since current_hero appeared
    ref_dealer: int | None = None   # this hand's confirmed button seat
    ref_board: list[str] = []       # this hand's last non-empty (debounced) board
    pending = False           # teardown seen; cut at the next deal evidence
    for i, state in enumerate(states):
        hero = state["hero_cards"]
        board = state["board_cards"]
        if current:
            if not pending and not board:
                d = state["dealer_seat"]
                dealer_moved = (
                    d is not None and ref_dealer is not None and d != ref_dealer
                    and next_dealer[i] in (d, None)
                )
                board_reset = bool(ref_board) and not _board_continues(states, i, ref_board)
                hero_fresh = (len(hero) == 2 and bool(current_hero)
                              and hero != current_hero and hero_swept)
                pending = dealer_moved or board_reset or hero_fresh
            deal_here = bool(state["dealt_in"]) or len(hero) == 2 or bool(board)
            stale_board = bool(board) and board == ref_board
            if pending and deal_here and not stale_board:
                hands.append(current)
                current, current_hero = [], []
                hero_swept, pending = False, False
                ref_dealer, ref_board = None, []
        current.append(state)
        d = state["dealer_seat"]
        if ref_dealer is None and d is not None and next_dealer[i] in (d, None):
            ref_dealer = d
        if state["board_cards"]:
            ref_board = state["board_cards"]
        if len(hero) == 2 and not current_hero:
            current_hero = hero
        if current_hero and not hero:
            hero_swept = True
    if current:
        hands.append(current)
    return hands


def _observed_forced_posts(
    hand: list[dict[str, Any]],
    players: list[int],
    dealer_seat: int | None,
) -> tuple[float, ...] | None:
    """Read one hand's forced-post chain off its deal-open state, or None.

    A forced post is a standing bet, present at the deal-open state BEFORE
    anyone has acted, on a seat in the chain clockwise of the button. The chain
    is read in ring order -- SB, BB, then straddles -- and must be
    nondecreasing, with each straddle at least doubling the post before it
    (that is what a straddle is; it is also what tells a 2.0 straddle from a
    seat's ante or a stray read). Anything that shows action already happened
    -- a board, a non-POST pill, standing money beyond the chain -- makes the
    open unobservable and returns None: the session-level vote decides from
    the hands whose opens WERE seen. Amounts are in BB display units, so the
    result is effectively (0.5, 1.0) plus any straddles; nothing here assumes
    that, it is simply what the client renders.
    """
    if not hand or dealer_seat is None:
        return None
    first = hand[0]
    if first["board_cards"] or not first["dealt_in"]:
        return None
    ring = [s for s in rd.SEAT_RING if s in players]
    if dealer_seat not in ring or len(ring) < 3:
        # Heads-up posting is positional (the button IS the small blind) and
        # has no straddle slot; nothing to learn beyond the default.
        return None
    start = ring.index(dealer_seat)
    ring = ring[start:] + ring[:start]
    chain = ring[1:]
    bets = first.get("bets") or {}
    unknown = first.get("bets_unknown") or {}
    posts: list[float] = []
    chain_seats: list[int] = []
    for seat in chain:
        if seat in unknown:
            return None            # a refused read cannot certify the chain
        amount = bets.get(seat)
        if amount is None:
            break
        if posts and amount < posts[-1] - _EPS:
            break
        if len(posts) >= 2 and amount < 2 * posts[-1] - _EPS:
            break
        posts.append(amount)
        chain_seats.append(seat)
    if len(posts) < 2:
        return None
    for seat, pill in (first.get("pills") or {}).items():
        # The only pills tolerated at a true open are the green POST pills on
        # the posting seats themselves (the client paints POST on the same
        # green as CALL/BET, so the word template often yields bet_or_call).
        if seat in chain_seats and pill in {"post_blind", rd.PILL_BET_OR_CALL}:
            continue
        return None
    if any(seat not in chain_seats for seat in bets):
        return None                # standing money beyond the chain = action
    return tuple(posts)


def _session_forced_posts(
    segments: list[list[dict[str, Any]]],
    session_dealt_seats: set[int],
) -> tuple[tuple[float, ...], list[bool]]:
    """The table's forced-post structure, voted across every observable
    deal-open in the session. The structure -- how many posts, of what size --
    is a property of the TABLE, not of one hand, so the session mode decides
    and individual unobservable opens simply don't vote. Falls back to plain
    blinds (0.5, 1.0 in the client's BB display units) when no open was ever
    observed; that fallback is a last resort, not a model of the game.

    Also returns, per segment, whether that hand's OWN deal open was observed
    -- the fact _player_seats needs to know card backs are authoritative for
    that hand's roster.
    """
    observed: list[tuple[float, ...]] = []
    open_seen: list[bool] = []
    for segment in segments:
        dealer_seat = _mode([s["dealer_seat"] for s in segment])
        # The ring for the chain read comes from CARD BACKS alone (plus the
        # button's own seat), never from the full roster heuristics: at a deal
        # open the card backs are authoritative by definition, and a seat the
        # stack recruitment would wrongly admit (a sitting-out player) lands
        # inside the chain and severs it -- measured on the 2026-08-03
        # session, where the waiting hero broke the chain between the small
        # and big blinds on every hand it was recruited into.
        players = sorted(
            {seat for state in segment for seat in state["dealt_in"]}
            | ({dealer_seat} if dealer_seat is not None else set())
        )
        posts = _observed_forced_posts(segment, players, dealer_seat)
        open_seen.append(posts is not None)
        if posts is not None:
            observed.append(posts)
    return (_mode(observed) or (0.5, 1.0)), open_seen


def _has_deal_evidence(segment: list[dict[str, Any]]) -> bool:
    """A segment is a HAND only if a deal was ever visible in it: hole-card
    backs at any seat, a board, or two hero cards. Teardown interstitials and
    idle stretches between deals segment off on the boundary signals above and
    must not be reconstructed as hands of nothing."""
    return any(
        s["dealt_in"] or s["board_cards"] or len(s["hero_cards"]) == 2
        for s in segment
    )


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
    """Debounce ONE hand's pot series in place with revert-only blip rejection:
    a reading B is repaired only in an A -> B -> A pattern (the next reading
    reverts to the accepted value) -- the same rule every other numeric series
    gets. There is deliberately NO collapse-repair arm any more: the old
    "a pot far below the accepted pot is the next deal's blind pot showing
    through" carry existed because segmentation used to leave next-deal states
    inside the hand, and on a session it misjudged it erased the true hand
    boundary 35 times in one merged mega-hand. Segmentation now cuts on the
    button / board-reset evidence itself, so a persistent pot collapse inside
    one hand is real by construction -- repairing it would be inventing money."""
    accepted: float | None = None
    for idx, state in enumerate(hand):
        pot = state["pot"]
        if pot is None:
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


def _position_names(n_straddles: int) -> list[str]:
    """Position names for the button-anchored ring, straddle slots included.

    Generated, not a fixed table: a straddle game inserts ST (then ST2, ST3...)
    between the big blind and UTG, and every later name shifts one seat. The
    old fixed 8-name list could not represent that, so a straddled table's
    every position came out one seat wrong."""
    return (["BTN", "SB", "BB"]
            + [("ST" if k == 0 else f"ST{k + 1}") for k in range(n_straddles)]
            + ["UTG", "UTG+1", "LJ", "HJ", "CO"])


def _is_straddle_position(name: str) -> bool:
    return name == "ST" or (name.startswith("ST") and name[2:].isdigit())


def _positions(
    players: list[int], dealer_seat: int | None, *, n_straddles: int = 0
) -> dict[int, str]:
    """Assign BTN/SB/BB/... walking the seat ring from the dealer over dealt-in
    seats. ``n_straddles`` comes from the session's observed forced-post
    structure; it is capped by the ring size (a 3-handed ring has no seat left
    to straddle)."""
    if not players:
        return {}
    ring = [s for s in rd.SEAT_RING if s in players]
    if not ring:
        ring = sorted(players)
    start = 0
    if dealer_seat in ring:
        start = ring.index(dealer_seat)
    ordered = ring[start:] + ring[:start]
    names = _position_names(min(max(0, n_straddles), max(0, len(ring) - 3)))
    return {seat: (names[k] if k < len(names) else f"P{k}")
            for k, seat in enumerate(ordered)}


def _street_seat_order(positions: dict[int, str], street: str) -> list[int]:
    """Seats in legal action order, derived only from the button-anchored ring.

    ``positions`` preserves the ring order produced by ``_positions``:
    BTN, SB, BB, then the remaining seats through CO.  Preflop begins left of
    the big blind; later streets begin left of the button.  This is used only to
    order actions that first become visible in the same sampled state.  Action
    detections never feed back into position assignment.
    """
    ring = list(positions)
    if len(ring) <= 1:
        return ring
    if street == "preflop":
        # Heads-up is the exception: the button is also the small blind and acts
        # first preflop.  Multiway action begins left of the LAST forced post --
        # the big blind, or the last straddle when the table straddles (the
        # position names already encode how many straddle slots exist).
        if len(ring) == 2:
            return ring
        n_straddles = sum(
            1 for name in positions.values() if _is_straddle_position(name)
        )
        k = (3 + n_straddles) % len(ring)
        return ring[k:] + ring[:k]
    return ring[1:] + ring[:1]


def _player_seats(
    hand: list[dict[str, Any]],
    dealer_seat: int | None,
    *,
    session_dealt_seats: set[int] | None = None,
    deal_open_observed: bool = False,
) -> list[int]:
    """Recover the hand roster, including players who folded before capture.

    Card backs and action pills remain the primary participation evidence.  When
    the first captured state is already mid-hand, however, early folders have
    neither: the client has removed their cards and their short-lived FOLD pills
    have expired.  Stable stack HUDs are then the surviving occupancy evidence.
    The dealer button independently proves its own seat participated.

    Opening stack occupancy is used only when the first state itself proves the
    hand was already under way (board, action pill, or a bet above a blind), and
    a stack seat must repeat.  This keeps a one-frame stack false positive from
    creating a player while allowing a recording that begins mid-action to retain
    the seats whose folds occurred before frame zero.

    ``session_dealt_seats``, when provided, is the set of seats that were EVER
    dealt cards anywhere in the session, and it caps the stack recruitment: a
    stack HUD proves a seat is OCCUPIED, not that it is in this hand. A player
    sitting out shows a stable stack all session and no cards ever, and
    recruiting them shifts every blind and position by one seat -- measured on
    the 2026-08-03 session, where a waiting hero was exported as the small
    blind of every hand.

    ``deal_open_observed`` disables stack recruitment entirely: when the
    hand's own deal-open state was seen (its forced-post chain was readable),
    every participant's card back was on screen, so the card backs are
    authoritative and a stack HUD adds nothing. Without this, a straddled
    table defeats the mid-hand-open test on EVERY hand -- the straddler's
    green POST pill and above-blind standing bet at the deal open read as
    "action already happened" -- and a seat that sat out THIS hand but played
    elsewhere in the session (so the session cap admits it) is recruited into
    a hand whose open plainly showed it holding no cards.
    """
    if not hand:
        return []
    from collections import Counter

    dealt_counts = Counter(seat for state in hand for seat in state["dealt_in"])
    opening_dealt = {seat for state in hand[:2] for seat in state["dealt_in"]}
    pill_counts = Counter(
        seat for state in hand for seat in (state.get("pills") or {})
    )
    players = {
        seat for seat, count in dealt_counts.items()
        if count >= 2 or seat in opening_dealt
    } | {
        seat for seat, count in pill_counts.items()
        if count >= 2
    }

    first = hand[0]
    mid_hand_open = not deal_open_observed and bool(
        first["board_cards"]
        or first.get("pills")
        or any(
            amount is not None and amount > 1.0 + _EPS
            for amount in (first.get("bets") or {}).values()
        )
    )
    if mid_hand_open:
        stack_counts = Counter(
            seat for state in hand for seat in (state.get("stacks") or {})
        )
        players.update(
            seat for seat in first.get("stacks", {})
            if stack_counts[seat] >= 2
            and (session_dealt_seats is None or seat in session_dealt_seats)
        )

    if dealer_seat is not None:
        players.add(dealer_seat)
    if not players:
        players.update(dealt_counts)
    return sorted(players)


def _opening_live_seats(first: dict[str, Any]) -> set[int]:
    """Seats with evidence they were still live when the first state was seen.

    Hero hole cards stay on screen after a fold (they grey out in place), so a
    ``hero_dim`` first state must NOT count seat 0 as live -- otherwise the
    opening roster keeps a folded hero and later inferred checks invent actions
    for a player who already left.
    """
    first_bets_unknown = first.get("bets_unknown") or {}
    hero_live = (
        len(first["hero_cards"]) == 2 and not first.get("hero_dim")
    )
    live = (
        set(first["dealt_in"])
        | set(first.get("villain_cards") or {})
        | ({0} if hero_live else set())
        | {
            seat for seat, pill in (first.get("pills") or {}).items()
            if pill != "fold"
        }
        | set(first.get("bets") or {})
        | set(first_bets_unknown)
    )
    if first.get("hero_dim"):
        live.discard(0)
    return live


def _seat_owes_call_on_street(
    actions: list[dict[str, Any]], street: str, seat: int
) -> bool:
    """True when a later raise/bet on ``street`` has not been answered by ``seat``.

    Used when the board advances while the seat is still dealt-in: remaining in
    the hand after facing an unanswered raise means they called (operator:
    "because SB is still in you should assume they called").
    """
    street_actions = [action for action in actions if action["street"] == street]
    if any(
        action["seat"] == seat and action["action_type"] == "all-in"
        for action in street_actions
    ):
        # An all-in seat stays dealt-in but cannot answer a later raise.
        return False
    last_aggressor_index = max(
        (
            index
            for index, action in enumerate(street_actions)
            if action["action_type"] in {"raise", "bet", "all-in"}
        ),
        default=-1,
    )
    if last_aggressor_index < 0:
        return False
    aggressor = street_actions[last_aggressor_index]["seat"]
    if aggressor == seat:
        return False
    return not any(
        action["seat"] == seat
        for action in street_actions[last_aggressor_index + 1 :]
    )


def _seat_checks_preflop_later(
    hand: list[dict[str, Any]], seat: int, *, start_index: int = 0
) -> bool:
    """True when ``seat`` shows a preflop check after ``start_index``.

    A non-blind seat that posts a live missed blind can check when action
    returns; a limper who already called cannot. Used to disambiguate green
    CALL/BET pills that are actually POST.
    """
    for state in hand[start_index:]:
        if state.get("board_cards"):
            break
        if (state.get("pills") or {}).get(seat) == "check":
            return True
    return False


def _is_live_post_action(
    street: str,
    seat: int,
    positions: dict[int, str],
    pill: str | None,
    amount: float | None,
    *,
    later_preflop_check: bool = False,
) -> bool:
    """Classify a preflop chip commitment as a live missed-blind post.

    Operator flag: "utg+1 is actually a post, not a call". ClubWPT paints POST
    on the same green as CALL/BET, so OCR often yields ``bet_or_call``.
    """
    if street != "preflop":
        return False
    if pill == "post_blind":
        return True
    position = positions.get(seat, "")
    if position in {"SB", "BB"}:
        return False
    if pill not in {rd.PILL_BET_OR_CALL, "call", "bet"}:
        return False
    if amount is not None and amount <= 1.0 + _EPS:
        return True
    return bool(amount is None and later_preflop_check)


def _opening_bet_amount(
    hand: list[dict[str, Any]], seat: int, first: dict[str, Any]
) -> float | None:
    """Standing bet size for a first-state action, or None when unknown.

    Prefers the stack-constancy window (``_committed_at_start``). A same-street
    bet_text scan is allowed ONLY when the seat has no stack series at all --
    the mid-street-open case where HUDs never read -- and never past a refused
    stack read (that break is exactly what ``_committed_at_start`` protects).
    """
    if (first.get("bets_unknown") or {}).get(seat):
        return None
    direct = first.get("bets", {}).get(seat)
    if direct is not None and direct > _EPS:
        return float(direct)
    standing = _committed_at_start(hand, seat)
    if standing is not None and standing > _EPS:
        return float(standing)
    # Stacks were readable somewhere: trust the constancy window's answer
    # (including 0.0 / None after a refusal). Do not second-guess with a later
    # bet_text that may postdate a hidden debit.
    if _stack_series(hand, seat):
        return None
    # No stack series at all: take the first agreeing same-street bet_text.
    first_board = len(first.get("board_cards") or [])
    seen: float | None = None
    for state in hand:
        if len(state.get("board_cards") or []) != first_board:
            break
        if (state.get("stacks_unknown") or {}).get(seat):
            break
        if (state.get("bets_unknown") or {}).get(seat):
            return None
        cand = (state.get("bets") or {}).get(seat)
        if cand is None:
            continue
        if seen is not None and abs(cand - seen) > _EPS:
            return None
        seen = float(cand)
    return seen


def _stack_series(hand: list[dict[str, Any]], seat: int) -> list[float]:
    return [s["stacks"][seat] for s in hand if seat in s["stacks"]]


def _observed_starting_stack(hand: list[dict[str, Any]], seat: int) -> float | None:
    """The seat's stack at the START of the hand as the table state proves it:
    the FIRST observed state's read PLUS the chips already standing in front of
    the seat at that moment. None when either half is unknown.

    ``_stack_series(...)[0]`` -- the first SURVIVING reading -- is not the same
    quantity. A seat whose stack_text failed to read early in the hand is dropped
    from those states entirely (an unknown and an absent box are indistinguishable
    in ``state["stacks"]``), so the first surviving reading is taken after the seat
    has already put money in, and the field named ``starting_stack`` published a
    mid-hand stack as fact. Measured by blanking one seat's reads over a real
    hand: BTN 123.4 -> 115.4, SB 218.0 -> 210.5, with warnings=none either time.
    Understating a starting stack corrupts effective stack and SPR directly.

    The committed half is not optional. The client renders stacks PRE-DEBITED
    (see _committed_at_start), so the raw first-state read EXCLUDES the seat's
    standing bet while the action ledger books that same money as an action --
    which understated 31 player rows across the 12 exported development hands by
    exactly the seat's first-state bet, and on one all-in seat crossed into
    arithmetic impossibility: a seat contributing 99.5 BB from a published
    starting stack of 97.5, at confidence 1.0 with warnings=[].

    Measured cost of refusing to guess: 3 of 148 player rows across the five
    development recordings (their first surviving reading is 4-5 states in).
    """
    if not hand:
        return None
    base = hand[0]["stacks"].get(seat)
    if base is None:
        return None
    committed = _committed_at_start(hand, seat)
    if committed is None:
        # A refused bet read inside the committed-scan window: the amount already
        # on the felt is unknown, so the starting stack is too.
        return None
    return round(base + committed, 2)


def _committed_at_start_unknown_reason(
    hand: list[dict[str, Any]], seat: int
) -> str | None:
    """Distinguish OCR refusal from contradictory standing-bet evidence.

    ``committed_at_start_conflict``: disagreeing bet reads / box conflict under a
    constant stack -- structural blinds must NOT paper over this.
    ``committed_at_start_unknown``: every bet read in the window refused (or the
    scan broke) with no successful size -- SB/BB may use the structural blind.
    """
    series = _stack_series(hand, seat)
    if not series:
        return None
    start = series[0]
    saw_refusal = False
    committed: float | None = None
    for state in hand:
        if (state.get("stacks_unknown") or {}).get(seat):
            break
        stack = state["stacks"].get(seat)
        if stack is None:
            continue
        if abs(stack - start) > _EPS:
            break
        bet_code = (state.get("bets_unknown") or {}).get(seat)
        if bet_code == rd.AMOUNT_BET_BOXES_DISAGREE:
            return "committed_at_start_conflict"
        if bet_code:
            saw_refusal = True
        bet = state["bets"].get(seat)
        if bet is not None:
            if committed is not None and abs(bet - committed) > _EPS:
                return "committed_at_start_conflict"
            committed = bet
    if committed is not None:
        return None
    return "committed_at_start_unknown" if saw_refusal else None


def _observed_starting_stack_unknown(hand: list[dict[str, Any]], seat: int) -> str | None:
    """Why the seat's starting stack is unknown, or None (it is known, or absent).

    Same states ``_observed_starting_stack`` reads, so the pair never disagrees.
    """
    if not hand:
        return None
    if hand[0]["stacks"].get(seat) is None:
        return (hand[0].get("stacks_unknown") or {}).get(seat)
    if _committed_at_start(hand, seat) is None:
        return _committed_at_start_unknown_reason(hand, seat) or (
            "committed_at_start_unknown"
        )
    return None


def _series_has_hole(states: list[dict[str, Any]], seat: int) -> bool:
    """True when any state in `states` REFUSED this seat's stack read.

    A series with a hole is not a shorter series -- it is a series whose gaps are
    of unknown size, so every quantity derived by differencing it (contribution,
    sweep gain, hero net) rests on an unmeasured step. Estimators built on one
    abstain rather than widening their window to fill it from elsewhere.
    """
    return any(seat in (s.get("stacks_unknown") or {}) for s in states)


def _committed_at_start(hand: list[dict[str, Any]], seat: int) -> float | None:
    """Chips already standing in front of `seat` when the hand came into view.

    The client renders stacks PRE-DEBITED: chips pushed into the pot leave the
    displayed stack immediately. So a stack delta measured across the observed
    window silently excludes everything committed before the first sample, and
    that is not a rare edge -- it covers the blinds on every hand and the whole
    preflop action on any hand the segmenter picks up late.

    Measured consequence: the 07-23 3.33.54 PM session's first hand shows the
    hero's 24.0 BB raise standing in bet_text for 18 consecutive states while the
    hero's stack never moves, so `series[-1] - series[0]` reported +52.8 for a hand
    whose true net was +28.8 -- an 83% overstatement published as a bare number
    with warnings=[].

    Scan forward only while the seat's stack still reads its first value: once the
    stack moves, later bet_text belongs to actions the delta already accounts for.
    bet_text flickers between states (it is absent on 3 of the 19 states above),
    so take the largest reading in that span rather than the first.

    WHY A LATER READING IS A MEASUREMENT AND NOT A BACKFILL. The client
    pre-debits: chips joining the felt leave the displayed stack in the same
    render. So while the seat's stack READS its starting value, the standing bet
    is a CONSTANT -- it cannot have grown (that debits the stack and ends the
    window) and it cannot have shrunk (a street close clears it to nothing, it
    never shows a smaller number). Every successful bet read inside the
    proven-constant window therefore measures the same standing amount, and a
    refusal on one state is answered by a read on another.

    Four consequences, each load-bearing:
      * the scan STOPS at the first state whose stack read for this seat is
        REFUSED -- constancy is proven by the run of equal stack reads, and a
        refusal breaks the proof (a debit could hide inside it: measured on a
        synthetic arm, scanning past a refused stack read booked a 24.0 raise
        whose true first-state size was 4.0);
      * a bet reading counts only from a state whose OWN stack read equals the
        start -- an absent-stack state neither extends the proof nor supplies a
        reading (its bet may postdate a hidden debit);
      * every reading inside the window must AGREE. The constancy argument cuts
        both ways: a constant stack means the standing bet cannot have changed,
        so two reads of DIFFERENT values inside one window falsify the premise
        the window rests on -- at least one of the boxes is not this seat's
        standing bet (a neighbouring seat's box misattributed by the anchors,
        or next-deal leakage). Taking the max was a choice among contradictory
        evidence: measured on g0621 hand 2, a folded seat whose stack never
        moved carried a 3.0 read preflop and a 15.0 read on the turn, and the
        max published starting_stack 115.7 for a seat whose true committed
        chips were 0 -- and would have booked a phantom "call 15.0". A window
        with disagreeing reads returns None: the standing amount is UNKNOWN;
      * when the window contains a REFUSED bet and NO successful read, the
        standing amount is UNKNOWN (None), not 0.0: a bet box was detected and
        declined to read, and the maximum over nothing is not a measurement.
        Callers treat None as unknown -- the pre-observed action branch emits
        amount=None, and hero_bb_won / starting_stack become unknown.
    """
    series = _stack_series(hand, seat)
    if not series:
        return 0.0
    start = series[0]
    committed: float | None = None
    saw_refusal = False
    for state in hand:
        if (state.get("stacks_unknown") or {}).get(seat):
            break  # constancy proof broken: nothing past here is in evidence
        stack = state["stacks"].get(seat)
        if stack is None:
            continue  # no stack read: skip -- neither proof nor reading
        if abs(stack - start) > _EPS:
            break  # the stack moved: the window is over
        bet_code = (state.get("bets_unknown") or {}).get(seat)
        if bet_code == rd.AMOUNT_BET_BOXES_DISAGREE:
            # A BOX CONFLICT IS A DISAGREEING READING, not an unreadable one, and
            # the two are answered differently. The constancy argument tolerates
            # a reader refusal because the standing bet cannot have changed while
            # the stack is constant, so any successful read in the window
            # measures it. A conflict says something stronger: two boxes resolved
            # to this seat and read DIFFERENT numbers, so either the standing bet
            # is not what the window's other reads say or the anchors
            # misattributed a neighbouring seat's box -- and the third bullet
            # above already holds that a window with disagreeing readings returns
            # UNKNOWN. Measured: g0621 hand 3, seat 3 reads 0.5 nine times and
            # then carries a conflict between a "15 BB" box (conf 0.967) and a
            # "0.50 BB" box (conf 0.937); before the bet channel had a conflict
            # rule at all, detector list order published 6.0 there and the
            # disagreement clause below fired. Falling back to 0.5 because the
            # contradiction now arrives as a code would silently un-flag the hand.
            return None
        if bet_code:
            saw_refusal = True
        bet = state["bets"].get(seat)
        if bet is not None:
            if committed is not None and abs(bet - committed) > _EPS:
                # Two different values inside one proven-constant window:
                # contradictory evidence, not a measurement to rank.
                return None
            committed = bet
    if committed is not None:
        return round(committed, 2)
    return None if saw_refusal else 0.0


def _mode(values: list[Any]) -> Any:
    from collections import Counter
    values = [v for v in values if v is not None]
    return Counter(values).most_common(1)[0][0] if values else None


def _trim_trailing_next_deal(hand: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop trailing states that already belong to the NEXT deal.

    Segmentation arms its cut on the teardown signals but lands it on the first
    state showing the new DEAL, so the teardown interstitial -- button already
    advanced, pot reset, cards cleared -- deliberately trails the current hand
    (it carries this hand's settlement) and is pruned here, exactly as before.
    The pot-collapse arm this function used to carry is gone with the collapse
    machinery (see _debounce_pot): a next-deal blind pot state either trails
    here dealer-marked or heads the next segment, and in a segmented hand a
    persistent pot collapse is real by construction."""
    if len(hand) <= 1:
        return hand
    modal_dealer = _mode([s["dealer_seat"] for s in hand])
    end = len(hand)
    while end > 1:
        s = hand[end - 1]
        no_hero = len(s["hero_cards"]) != 2
        # A tail state with no hero cards, no pot read, no dealer read and no
        # stack reads carries zero information -- it can neither corroborate nor
        # refute anything. SKIPPING it (rather than stopping on it, which is what
        # the loop used to do the moment a state failed both tests) is what lets
        # the next-deal state BEHIND it be trimmed: an unreadable frame must not
        # shield the next deal. Observed for real on 07-15, where a fully-dropped
        # final frame hid the next deal's blind pot and made a 240.9 BB hand
        # report a final pot of 1.0.
        uninformative = (no_hero and s["pot"] is None
                         and s["dealer_seat"] is None and not s["stacks"])
        dealer_moved = (modal_dealer is not None and s["dealer_seat"] is not None
                        and s["dealer_seat"] != modal_dealer)
        if uninformative or (no_hero and dealer_moved):
            end -= 1
        else:
            break
    return hand[:end]


def _settle_index(hand: list[dict[str, Any]], players: set[int] | None = None) -> int:
    """Index of the state where the pot is swept to the winner, or the last index
    if never seen. The sweep is the single LARGEST qualifying stack jump in the
    hand -- during play stacks only fall (chips go in), so any increase is a pot
    award and the real sweep dwarfs any pre-sweep noise bump. Picking the first
    jump above a low threshold is wrong when a spurious sub-pot bump on a folded
    seat precedes the true sweep by a frame (it truncates the window before the
    winner is paid, flipping the result). States after the sweep are next-deal
    noise: the next hand's antes tick every stack down and its blinds post while
    the old hero cards linger.

    ``players`` restricts the sweep search to seats that were actually dealt into
    THIS hand. Without it any seat on the table can set the cut, including one
    that was never in the hand -- on 07-15 a seat that sat out the whole hand was
    dealt into the NEXT one, and its buy-in showing up as a stack jump truncated
    the hand seven states past its real settlement."""
    n = len(hand)
    # Suffix max of the pot from each index onward: a real settlement is TERMINAL
    # -- the pot is swept and only the next deal's small blinds follow -- so a
    # candidate jump is rejected if the pot later grows well beyond its own level.
    # This stops a mid-hand phantom (a stack OCR blip on a seat that then folds)
    # from truncating the hand before the real sweep.
    #
    # A state whose pot is unread (refused OR absent) contributes NOTHING to the
    # suffix max -- it is skipped, not coerced to 0.0 (`hand[i]["pot"] or 0.0`
    # was the one place this function still substituted a number for an unread
    # pot, sixteen lines above its own refusal to do exactly that). The result is
    # a maximum over the KNOWN future pots, i.e. a LOWER BOUND, and the guard
    # built on it is one-sided by construction: it only ever REJECTS a candidate,
    # and only on proven evidence (a later pot that is genuinely larger). An
    # unread future pot can neither create nor suppress a rejection here; it
    # simply supplies no evidence. Measured against the pre-Option-A behaviour
    # (never skipping): the chosen settle index is identical on all 31
    # development hands.
    suffix_max_pot = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        p = hand[i]["pot"]
        suffix_max_pot[i] = max(p if p is not None else 0.0, suffix_max_pot[i + 1])
    pot_so_far: float | None = None
    best_idx, best_jump = n - 1, 0.0
    for idx, (prev, cur) in enumerate(zip(hand, hand[1:], strict=False), start=1):
        if prev["pot"] is not None:
            pot_so_far = prev["pot"] if pot_so_far is None else max(pot_so_far, prev["pot"])
        threshold = max(3.0, 0.4 * (pot_so_far or 0.0))
        if cur["pot"] is None and cur.get("pot_unknown"):
            # The pot at this state was REFUSED, and the terminal-sweep test below
            # needs this state's own pot. `cur["pot"] or pot_so_far` substituted a
            # STALE pot carried from an earlier state for the unread one -- a
            # backfill from a neighbouring state, which is the one thing an UNKNOWN
            # must never receive. An unmeasurable transition is skipped, not
            # guessed; the scan simply looks elsewhere for the sweep.
            cur["settle_scan_skipped"] = True
            continue
        # Reached only when cur["pot"] is a value OR the pot box was ABSENT
        # (pot_unknown is None) -- the refused case was skipped above. For an
        # absent box, `pot_so_far` (the running max of the pots actually read)
        # stands in, and the substitution feeds ONLY the one-sided guard below:
        # it can reject a candidate, never produce a value, and understating
        # cur_pot only makes the guard stricter. Fixture states carrying no pot
        # data at all resolve to 0.0 against a 0.0 suffix max, leaving the stack
        # scan untouched -- which is why absence, unlike refusal, is not a
        # settle_scan_skipped fact: nothing here declined to read it.
        cur_pot = cur["pot"] if cur["pot"] is not None else (pot_so_far or 0.0)
        if suffix_max_pot[idx + 1] > cur_pot + threshold:
            continue  # pot keeps growing after this -> not the terminal sweep
        for seat, after in cur["stacks"].items():
            if players is not None and seat not in players:
                continue
            before = prev["stacks"].get(seat)
            if before is None:
                continue
            jump = after - before
            # >= so that, on a tie, the later (true post-sweep) frame wins.
            if jump >= threshold and jump >= best_jump:
                best_idx, best_jump = idx, jump
    return best_idx


def _reconstruct_actions(
    hand: list[dict[str, Any]],
    positions: dict[int, str],
    player_name: dict[int, str],
    forced_posts_bb: tuple[float, ...] = (0.5, 1.0),
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

    def emit(
        street,
        seat,
        atype,
        amount,
        pot_before,
        stack_before,
        *,
        source: dict[str, Any] | None,
        derivation: str,
        is_live_post: bool | None = None,
        forced_bet_type: str | None = None,
    ):
        street_index[street] = street_index.get(street, 0) + 1
        if atype in {"bet", "raise", "all-in"}:
            street_has_bet[street] = True
            street_raised[street] = True
        # A live missed-blind post is a live bet to the posted level.
        if atype == "post_blind" and is_live_post:
            street_has_bet[street] = True
        acted.setdefault(street, set()).add(seat)
        actions.append(
            _action(
                street,
                street_index[street],
                seat,
                positions,
                player_name,
                atype,
                amount,
                pot_before,
                stack_before,
                source=source,
                derivation=derivation,
                is_live_post=is_live_post,
                forced_bet_type=forced_bet_type,
            )
        )

    # Initialized BEFORE the pre-observed scan so that scan can record a
    # standing bet nothing may size (a refused first-state read on a pill-less
    # seat) as an unmeasured transition instead of dropping it. It used to be
    # initialized just before the transition loop, which would have wiped any
    # entry the scan wrote.
    for state in hand:
        state["unmeasured_transitions"] = []

    # ---- pre-observed actions standing in the hand's FIRST state ----
    # A hand often enters view mid-preflop: pills and bet_texts already on the
    # table are completed actions we never saw happen. Bets at or below the big
    # blind with no pill are blind posts, not actions. (This table renders
    # amounts in BB units, so the big blind is 1.0.)
    #
    # THE SEAT SET is every player, not merely the seats visible in the first
    # state's pill/bet maps. A seat present only in `bets_unknown` -- a standing
    # bet whose read was REFUSED -- was never iterated at all, and a seat whose
    # bet box FLICKERED off the very first state (while its pill, which flashes
    # for under a second, had already expired) had no entry anywhere: in both
    # arms a call the client rendered on the felt for the whole window, and
    # whose amount the constancy proof publishes into starting_stack, was
    # silently absent from the ledger with every unknown channel reading clean.
    # An observed money action is never silently dropped: it is booked when the
    # constancy window proves its size, and recorded as an unmeasured
    # transition when a refusal leaves it unprovable.
    first = hand[0]
    first_bets_unknown = first.get("bets_unknown") or {}
    # A recovered opening-roster seat with no cards, non-fold pill, or standing
    # bet has already folded before capture began.  Keep it in the positional
    # ring, but never synthesize later checks or actions for it.
    preobserved_folded = set(positions) - _opening_live_seats(first)
    folded.update(preobserved_folded)
    if (
        0 in preobserved_folded
        and first.get("hero_dim")
        and len(first.get("hero_cards") or []) == 2
        and first.get("pills", {}).get(0) != "fold"
    ):
        # Greyed hero cards at hand open: book the fold so later streets cannot
        # synthesize checks for a hero who already left (operator flag: "hero
        # folded a long time ago"). Skip when a fold pill is already present --
        # the first-state pill loop books that fold once.
        emit(
            "preflop",
            0,
            "fold",
            None,
            None,
            first["stacks"].get(0),
            source=first,
            derivation="hero_dim",
        )
    # Forced straddle posts are structural, like the blinds: they are on the
    # felt before anyone acts, their POST pill is frequently expired or
    # unreadable by the time a state is sampled, and the session-observed
    # forced-post structure -- not a pill -- is the evidence they happened.
    # Booked first, so the ledger sees the money exactly where it entered.
    # (On the development corpus the straddle used to be booked as an UTG
    # "call 2.0" off its green pill, which kept the arithmetic right only
    # while the pill was readable; on the 2026-08-03 session the pill was
    # gone and the whole ledger came apart.)
    posted_straddles: set[int] = set()
    straddle_names = sorted(
        (name, seat) for seat, name in positions.items()
        if _is_straddle_position(name)
    )
    for k, (_name, seat) in enumerate(straddle_names):
        amount = forced_posts_bb[2 + k] if 2 + k < len(forced_posts_bb) else None
        emit(
            "preflop", seat, "post_blind", amount, None,
            hand[0]["stacks"].get(seat),
            source=hand[0], derivation="forced_post_structure",
            is_live_post=True, forced_bet_type="straddle",
        )
        posted_straddles.add(seat)

    # Book actions already standing in the first state, including mid-street
    # opens (operator: "missed where lj bet" when capture begins on the flop).
    opening_street = _street_for_count(
        len(first["board_cards"]), first.get("stage") or "preflop"
    )
    if opening_street in _STREET_BY_COUNT.values():
        first_seats = (
            set(positions) | set(first["pills"])
            | set(first["bets"]) | set(first_bets_unknown)
        )
        street_order = _street_seat_order(positions, opening_street)
        ordered_first_seats = (
            [seat for seat in street_order if seat in first_seats]
            + sorted(first_seats - set(street_order))
        )
        for seat in ordered_first_seats:
            pill = first["pills"].get(seat)
            bet = first["bets"].get(seat)
            if seat in posted_straddles:
                # The post is already booked from the structure. The seat's
                # green POST pill and its standing post-sized bet are that same
                # post, not a second action -- but a bet ABOVE the post level
                # (or an explicit raise/fold pill) is the straddler genuinely
                # acting and falls through to the normal booking below.
                expected = next(
                    (forced_posts_bb[2 + k]
                     for k, (_n, s) in enumerate(straddle_names) if s == seat),
                    None,
                )
                post_shaped_pill = pill in {None, "post_blind", rd.PILL_BET_OR_CALL}
                post_sized = (bet is None or expected is None
                              or abs(bet - expected) <= _EPS)
                if post_shaped_pill and post_sized:
                    continue
            if pill == "fold":
                folded.add(seat)
                emit(
                    opening_street, seat, "fold", None, None,
                    first["stacks"].get(seat),
                    source=first, derivation="action_pill",
                )
            elif pill in {
                "raise", "bet", "call", "post_blind", rd.PILL_BET_OR_CALL
            }:
                if bet is None and not (first.get("bets_unknown") or {}).get(seat):
                    # bet_text flickers: absent on the first state, present on
                    # the next. ONLY fill an ABSENT box; a REFUSED read stays
                    # amount=None (see prior corpus regressions).
                    bet = _opening_bet_amount(hand, seat, first)
                later_check = (
                    opening_street == "preflop"
                    and _seat_checks_preflop_later(hand, seat, start_index=1)
                )
                if _is_live_post_action(
                    opening_street, seat, positions, pill, bet,
                    later_preflop_check=later_check,
                ):
                    emit(
                        opening_street, seat, "post_blind", bet, None,
                        first["stacks"].get(seat),
                        source=first, derivation="action_pill",
                        is_live_post=True,
                    )
                elif opening_street == "preflop":
                    # Preflop blinds are already a standing bet, so green
                    # CALL/BET is a call and a "bet" pill is a raise.
                    atype = {
                        "bet": "raise", rd.PILL_BET_OR_CALL: "call"
                    }.get(pill, pill)
                    emit(
                        opening_street, seat, atype, bet, None,
                        first["stacks"].get(seat),
                        source=first, derivation="action_pill",
                    )
                else:
                    # Mid-street open: standing aggression is a bet; green is
                    # bet when nothing is yet booked on this street, else call.
                    if pill in {"raise", "bet"}:
                        atype = pill
                    elif pill == rd.PILL_BET_OR_CALL:
                        atype = (
                            "call" if street_has_bet.get(opening_street) else "bet"
                        )
                    else:
                        atype = pill
                    emit(
                        opening_street, seat, atype, bet, None,
                        first["stacks"].get(seat),
                        source=first, derivation="action_pill",
                    )
            elif pill == "check":
                emit(
                    opening_street, seat, "check", None, None,
                    first["stacks"].get(seat),
                    source=first, derivation="action_pill",
                )
            elif pill is None and bet is not None:
                if opening_street == "preflop" and bet > 1.0 + _EPS:
                    emit(
                        opening_street, seat, "call", bet, None,
                        first["stacks"].get(seat),
                        source=first, derivation="bet_text",
                    )
                elif opening_street != "preflop" and bet > _EPS:
                    atype = (
                        "call" if street_has_bet.get(opening_street) else "bet"
                    )
                    emit(
                        opening_street, seat, atype, bet, None,
                        first["stacks"].get(seat),
                        source=first, derivation="bet_text",
                    )
            elif pill is None and bet is None and opening_street == "preflop":
                # The pill-less arms the old iteration could never reach.
                if first_bets_unknown.get(seat):
                    # A REFUSED first-state bet: money of unknown size is
                    # standing; record the hole without inventing an action.
                    first["unmeasured_transitions"].append(seat)
                elif first["stacks"].get(seat) is not None:
                    standing = _committed_at_start(hand, seat)
                    if standing is None:
                        first["unmeasured_transitions"].append(seat)
                    elif standing > 1.0 + _EPS:
                        emit(
                            opening_street, seat, "call", standing, None,
                            first["stacks"].get(seat),
                            source=first, derivation="bet_text",
                        )

    settle = _settle_index(hand, set(positions))
    window = hand[: settle + 1]
    # High-water mark of board length seen so far this hand: a real board only
    # grows, so any observed SHRINK is a detection blip (board_cards has the
    # exact same single-frame-empty-reading weakness _debounce_cards already
    # has for hero_cards -- confirmed for real: v05#12 had two states deep
    # into the flop misread the board back to [] and back, which mapped
    # straight to "preflop" via _street_for_count(0, ...) and mislabeled a
    # late flop double-fold as two extra preflop actions). Using the running
    # max instead of the raw (possibly-blipped) reading neutralizes that
    # without touching the shared debounce fn or hand segmentation.
    board_hwm = 0
    # Per-(street, seat) high-water bet_text, carried across single-frame read
    # dropouts. Dense (1s) sampling lands on brief overlay/animation frames where
    # a seat's whole HUD reads empty for one sample; without carrying the standing
    # bet across that gap, the felt's still-showing raise reads as a fresh bet the
    # next frame and duplicates the action (see run-length debounce note at top).
    carried_bet: dict[tuple[str, int], float] = {}
    interval_s = float(hand[0].get("sampling_interval_s") or 0.0) or _CALIB_INTERVAL_S
    gap_threshold_s = _coverage_gap_threshold_s(interval_s)
    for i, (prev, cur) in enumerate(zip(window, window[1:], strict=False)):
        nxt = window[i + 2] if i + 2 < len(window) else None
        board_hwm = max(board_hwm, len(prev["board_cards"]))
        street = _street_for_count(board_hwm, prev["stage"])
        if street not in _STREET_BY_COUNT.values():
            continue
        pot_before = prev["pot"]
        prior_gap_s = float(cur.get("prior_gap_s") or 0.0)
        coverage_gap = (
            prior_gap_s > gap_threshold_s
            and _critical_unobserved_change(prev, cur)
        )
        if coverage_gap:
            cur["coverage_gap"] = True

        # ---- hero_dim rising mid-hand: persistent fold indicator ----
        if (
            cur.get("hero_dim")
            and not prev.get("hero_dim")
            and 0 in positions
            and 0 not in folded
        ):
            folded.add(0)
            emit(
                street,
                0,
                "fold",
                None,
                pot_before,
                prev["stacks"].get(0),
                source=cur,
                derivation="hero_dim",
            )

        # ---- money actions (bet_text delta, corroborated by the stack) ----
        money_seats: dict[int, tuple[float | None, float | None, str]] = {}
        prev_unknown = prev.get("stacks_unknown") or {}
        cur_unknown = cur.get("stacks_unknown") or {}
        cur_bet_unknown = cur.get("bets_unknown") or {}
        prev_bet_unknown = prev.get("bets_unknown") or {}
        for seat in sorted(set(cur["bets"]) | set(cur["stacks"]) | set(cur_unknown)):
            if seat in folded or seat not in positions:
                continue
            before = prev["stacks"].get(seat)
            after = cur["stacks"].get(seat)
            stack_dropped = before is not None and after is not None and after < before - _EPS
            # A REFUSED read on either end of the transition: the delta is not
            # merely missing, it is unmeasurable.
            stack_refused = bool(prev_unknown.get(seat) or cur_unknown.get(seat))
            amount = None
            derivation = ""
            unmeasurable = False
            if coverage_gap and (
                stack_dropped
                or stack_refused
                or before is None
                or after is None
                or (before is not None and after is not None
                    and abs(before - after) > 0.5 + _EPS)
            ):
                # Tab/lobby hid the table across a material money change. Do not
                # invent a sized action from the post-gap stack delta.
                unmeasurable = True
            elif stack_dropped:
                # the debounced stack delta is the most reliable size
                amount = round(before - after, 2)
                derivation = "stack_delta"
            elif stack_refused:
                # THE REFUSAL BRANCH, split off from "the stack is absent" below.
                # It used to share it: `before is None or after is None` made "the
                # reader could not read this stack" the TRIGGER for a weaker
                # estimator, so a refusal did not reduce what the spine claimed,
                # it changed which evidence the claim rested on. Measured on
                # g0723b hand 1: blanking the hero's stack reads made this fall
                # through to the bet_text delta, which re-emitted a raise that was
                # merely still standing on the felt -- `preflop 0 raise 24.0`
                # booked TWICE, 92.3 BB of action against a 73.8 BB pot. The spine
                # got MORE wrong, not merely less complete, as reads became
                # unknown.
                #
                # An estimator may fill a gap whose edges it can see; it may not
                # fill a hole a refusal punched. So: no money action is derived
                # here, and the transition is recorded as unmeasured.
                unmeasurable = True
            elif before is None or after is None:
                # stack ABSENT this transition (no box, or the fixture path):
                # fall back to the bet_text delta. (A bet_text rising while the
                # stack is provably unchanged is rendering lag of an action
                # already emitted from the stack.)
                # The guard used to be `not stack_flat`, i.e. the fallback was
                # skipped only when the stack was PROVABLY unchanged -- so a stack
                # that RISED between two reads (190.5 -> 182.0 -> 182.5, half a BB
                # of OCR jitter) let it run, and the seat's raise, already emitted
                # from the stack delta on the previous transition, was emitted a
                # second time at a slightly different size ("raise 8.5" then
                # "raise 8.0"). That double-counts the money and is the direct
                # cause of an exported hand whose contributions exceed its pot.
                # A seat whose stack is readable and did not FALL put no chips in;
                # the fallback exists for the unreadable case only.
                cur_bet = cur["bets"].get(seat)
                prev_bet = prev["bets"].get(seat)
                same_street = len(prev["board_cards"]) == len(cur["board_cards"])
                if cur_bet is not None and cur_bet >= 0.5 - _EPS:
                    # Base = the seat's standing bet this street. Prefer the prior
                    # state, but when it dropped out (prev_bet None) fall back to
                    # the carried high-water bet so an already-standing raise is
                    # not re-emitted as fresh. A real re-raise still exceeds the
                    # high-water mark, so this only suppresses re-displays.
                    prior = carried_bet.get((street, seat))
                    if same_street and prev_bet is not None:
                        base = prev_bet
                    elif same_street and prior is not None:
                        base = prior
                    elif same_street and prev_bet_unknown.get(seat):
                        # The previous state's bet was REFUSED and the high-water
                        # mark was dropped with it (see the carry loop below), so
                        # there is no base to subtract. Falling through to 0.0
                        # would attribute the seat's whole standing bet to this
                        # transition -- a backfill of the missing base with a
                        # number no state supplied.
                        base = None
                    else:
                        base = 0.0
                    if base is None:
                        unmeasurable = True
                    elif cur_bet > base + _EPS:
                        amount = round(cur_bet - base, 2)
                        derivation = "bet_text_delta"
            if amount is not None and amount >= 0.5 - _EPS:
                money_seats[seat] = (amount, before, derivation)
            elif unmeasurable:
                # AN UNMEASURED TRANSITION IS NOT "a read was refused here". It is
                # "money is KNOWN to have moved and its size is unmeasurable",
                # which is the only shape that damages the ledger. A refusal on an
                # idle seat -- no pill, no bet on the felt -- costs the hand
                # nothing but the read itself, and that is already carried as a
                # soft signal (`amounts_unknown`, which caps the confidence
                # score). Conflating the two would let one unreadable crop on a
                # seat that never acted reject an otherwise sound hand: a coverage
                # cost with no correctness benefit.
                pill = cur["pills"].get(seat)
                fresh_pill = (pill in {"raise", "bet", "call", "all-in",
                                       "post_blind", rd.PILL_BET_OR_CALL}
                              and prev["pills"].get(seat) != pill)
                cur_bet = cur["bets"].get(seat)
                prev_bet = prev["bets"].get(seat)
                same_street = len(prev["board_cards"]) == len(cur["board_cards"])
                known_base = (prev_bet if (same_street and prev_bet is not None)
                              else (carried_bet.get((street, seat)) if same_street
                                    else 0.0))
                bet_grew = (cur_bet is not None and cur_bet >= 0.5 - _EPS
                            and (known_base is None or cur_bet > known_base + _EPS))
                stack_moved = (
                    before is not None
                    and after is not None
                    and abs(before - after) > 0.5 + _EPS
                )
                # Across a coverage gap, refuse stack-delta invention but still
                # size from a readable current-frame bet_text (operator: "bb
                # bets 20" with the felt showing 20 BB after a short hole).
                felt_amount = None
                if (
                    coverage_gap
                    and cur_bet is not None
                    and cur_bet >= 0.5 - _EPS
                    and (fresh_pill or bet_grew)
                ):
                    base = 0.0 if known_base is None else known_base
                    if cur_bet > base + _EPS:
                        felt_amount = round(cur_bet - base, 2)
                    elif fresh_pill:
                        felt_amount = round(cur_bet, 2)
                if felt_amount is not None:
                    money_seats[seat] = (felt_amount, before, "bet_text")
                else:
                    if fresh_pill or bet_grew or (coverage_gap and stack_moved):
                        cur["unmeasured_transitions"].append(seat)
                    # A pill PROVES the act happened even when nothing can size
                    # it. Emit amount=None -- never drop, never fabricate from
                    # post-gap stacks. Freshness avoids re-booking a standing pill.
                    if fresh_pill:
                        money_seats[seat] = (None, before, "amount_unknown")

        street_order = _street_seat_order(positions, street)
        money_order = {seat: index for index, seat in enumerate(street_order)}
        for seat, (amount, stack_before, derivation) in sorted(
            money_seats.items(),
            key=lambda item: (money_order.get(item[0], len(money_order)), item[0]),
        ):
            pill = cur["pills"].get(seat)
            after = cur["stacks"].get(seat)
            later_check = (
                street == "preflop"
                and _seat_checks_preflop_later(hand, seat, start_index=i + 1)
            )
            live_post = _is_live_post_action(
                street, seat, positions, pill, amount,
                later_preflop_check=later_check,
            )
            if after is not None and after <= _EPS:
                atype = "all-in"
                live_post = False
            elif live_post:
                atype = "post_blind"
            elif pill in {"raise", "bet", "call", "post_blind"}:
                atype = pill
            else:
                # Reached for an unreadable pill AND for the green CALL/BET
                # ambiguity (rd.PILL_BET_OR_CALL). Structure resolves it: a seat
                # putting chips in on a street where nobody has bet is BETTING;
                # only a standing bet makes "call" a possible action at all.
                #
                # Forcing green -> "call" instead published `turn: seat:3 call
                # 11.5` on a turn the flop had checked through, and the knock-on
                # was worse than the wrong label: street_has_bet stayed False, so
                # the fold handler (which correctly refuses folds on a street with
                # nothing to fold to) DISCARDED the villain's observed FOLD pill,
                # and the round-completion pass then synthesised a CHECK for that
                # seat. Five of the 21 reconstructed hands carried such a call,
                # every one with spine warnings=[].
                atype = "call" if street_has_bet.get(street) else "bet"
            emit(
                street, seat, atype, amount, pot_before, stack_before,
                source=cur, derivation=derivation,
                is_live_post=True if atype == "post_blind" else None,
            )

        # ---- folds: card_back disappeared OR a fresh fold pill (hero: pill only) ----
        # A fold is only possible facing a bet; "folds" detected on a street where
        # nobody has bet are showdown reveals/sweeps, not actions.
        # Hero's own cards never actually disappear on fold (they grey out in
        # place -- see hero_dim), so a hero_cards read dropping to empty is a
        # detection blip (an animation briefly covering the crop), not a real
        # card_back-style disappearance. Excluded here per this function's own
        # "hero: pill only" contract, which the `gone` computation used to
        # silently violate -- a one-frame hero_cards blip on a street that
        # already had a bet was enough to fabricate a hero fold action, even
        # on hands the hero went on to win (confirmed against source video).
        # A villain's card_back disappearing is only trusted as a fold if it's
        # STILL gone next state too -- a lone-frame detection blip (the same
        # failure mode fixed for the hero above, just via a different signal)
        # would otherwise fabricate a fold action, and the seat's real later
        # actions that state (bet/call/raise) then read as "acting after
        # folding" (confirmed for real: v05#12, seat2 and seat5 both "folded"
        # preflop on a one-state card_back blip, then legitimately bet/called
        # the flop).
        gone = {
            s for s in set(prev["dealt_in"]) - set(cur["dealt_in"])
            if s != 0 and (nxt is None or s not in nxt["dealt_in"])
        }
        pill_folds = {seat for seat, pill in cur["pills"].items()
                      if pill == "fold" and prev["pills"].get(seat) != "fold"}
        for seat in sorted(
            (gone | pill_folds) - folded,
            key=lambda item: (money_order.get(item, len(money_order)), item),
        ):
            if seat in money_seats or seat not in positions:
                continue  # money and a fold can't both happen; unknown seats are noise
            if not street_has_bet.get(street):
                continue
            if coverage_gap and seat in gone and seat not in pill_folds:
                # Roster shrunk across an unobserved stretch. A card_back
                # disappearance may be a fold -- or a lobby sweep -- and we did
                # not see which. Do not invent the fold; mark the hole.
                cur["unmeasured_transitions"].append(seat)
                continue
            if seat in gone and seat not in pill_folds:
                if len(gone) >= 2:
                    continue  # multi-seat sweep after the pot is awarded
                before = prev["stacks"].get(seat)
                if before is not None and before <= _EPS:
                    continue  # an all-in player's cards flip over; they can't fold
                shown = (cur.get("villain_cards") or {}).get(seat) or []
                if len(shown) >= 2:
                    # Showdown reveal removes the card back while adding hole
                    # cards (operator: "no LJ called and showed the card").
                    continue
            folded.add(seat)
            emit(
                street, seat, "fold", None, pot_before, prev["stacks"].get(seat),
                source=cur,
                derivation=(
                    "action_pill" if seat in pill_folds else "card_back_disappeared"
                ),
            )

        # ---- still-in calls when the street closes ----
        # After money/fold evidence for this transition: if the board advanced
        # and a live seat still owes a response to the last raise, remaining
        # dealt-in is proof they called (amount may still be unknown).
        # Never across a coverage_gap: both endpoints being dealt-in does not
        # prove continuous presence through an unobserved stretch.
        cur_board_hwm = max(board_hwm, len(cur["board_cards"]))
        if (
            cur_board_hwm > board_hwm
            and street_has_bet.get(street)
            and not coverage_gap
        ):
            still_in = (
                (set(prev.get("dealt_in") or []) & set(cur.get("dealt_in") or []))
                - folded
            )
            if cur.get("hero_dim"):
                still_in.discard(0)
            for seat in sorted(
                still_in,
                key=lambda item: (money_order.get(item, len(money_order)), item),
            ):
                if seat not in positions or seat in money_seats:
                    continue
                stack_now = prev["stacks"].get(seat)
                if stack_now is not None and stack_now <= _EPS:
                    # Already all-in: remains dealt-in but cannot answer a raise.
                    continue
                if not _seat_owes_call_on_street(actions, street, seat):
                    continue
                emit(
                    street,
                    seat,
                    "call",
                    None,
                    pot_before,
                    prev["stacks"].get(seat),
                    source=cur,
                    derivation="inferred_still_in",
                )

        # ---- checks: fresh check pill, no money this transition ----
        # A fresh check pill arriving together WITH a new board card belongs to
        # the new street (its first check), unlike money, which closes streets.
        check_street = street
        cur_hwm = max(board_hwm, len(cur["board_cards"]))
        if cur_hwm > board_hwm:
            check_street = _street_for_count(cur_hwm, cur["stage"])
        check_order = _street_seat_order(positions, check_street)
        check_rank = {seat: index for index, seat in enumerate(check_order)}
        for seat, pill in sorted(
            cur["pills"].items(),
            key=lambda item: (check_rank.get(item[0], len(check_rank)), item[0]),
        ):
            if pill != "check" or prev["pills"].get(seat) == "check":
                continue
            if seat in money_seats or seat in folded or seat not in positions:
                continue
            emit(
                check_street, seat, "check", None, pot_before, prev["stacks"].get(seat),
                source=cur, derivation="action_pill",
            )

        # carry each seat's standing bet_text forward as a per-street high-water
        # mark so a later single-frame dropout can't reset its base to 0
        for seat, bv in cur["bets"].items():
            if bv is not None:
                key = (street, seat)
                carried_bet[key] = max(carried_bet.get(key, 0.0), bv)
        # ...but never ACROSS a refusal. The high-water mark is a reading from a
        # DIFFERENT state, and carrying it over a state whose bet read the reader
        # positively refused is exactly the backfill this phase removes: the mark
        # would stand in for a number the pixels declined to supply. Dropping it
        # makes the base unknown, which the fallback above turns into an
        # unmeasured transition rather than a guess.
        for seat in cur_bet_unknown:
            carried_bet.pop((street, seat), None)

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
        live = [
            p for p in positions
            if p not in preobserved_folded
            and (p not in folded_at or order.index(folded_at[p]) > si)
        ]
        # seats already all-in on an earlier street cannot act; with fewer than
        # two actionable seats there is no betting round to check through
        actionable = [p for p in live if all_in_at.get(p, 99) >= si]
        if len(actionable) < 2:
            continue
        action_order = _street_seat_order(positions, street)
        action_rank = {seat: index for index, seat in enumerate(action_order)}
        for seat in sorted(
            actionable,
            key=lambda item: (action_rank.get(item, len(action_rank)), item),
        ):
            if seat in acted.get(street, set()):
                continue
            if not street_has_bet.get(street):
                # on the last street, only a showdown (>=2 seats never folded)
                # proves the unseen players actually got to check
                if street != order[-1] or len(actionable) >= 2:
                    source = next(
                        (
                            state
                            for state in reversed(window)
                            if _STREET_BY_COUNT.get(len(state["board_cards"])) == street
                        ),
                        window[-1],
                    )
                    emit(
                        street, seat, "check", None, None, None,
                        source=source, derivation="inferred_round_complete",
                    )
            elif street == "preflop" and positions.get(seat) == "BB" \
                    and not street_raised.get("preflop"):
                source = next(
                    (
                        state
                        for state in reversed(window)
                        if not state["board_cards"]
                    ),
                    window[0],
                )
                emit(
                    street, seat, "check", None, None, None,
                    source=source, derivation="inferred_round_complete",
                )

    # Emission interleaves sources (transitions, cur-street checks, synthesis),
    # so impose global street order; within a street the emit order stands.
    street_rank = {s: k for k, s in enumerate(("preflop", "flop", "turn", "river"))}
    actions.sort(key=lambda a: street_rank.get(a["street"], 9))
    return actions


def _action(
    street,
    index,
    seat,
    positions,
    player_name,
    atype,
    amount,
    pot_before,
    stack_before,
    *,
    source: dict[str, Any] | None,
    derivation: str,
    is_live_post: bool | None = None,
    forced_bet_type: str | None = None,
):
    payload = {
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
        "source_time_s": None if source is None else source.get("time_s"),
        "source_image": None if source is None else source.get("image"),
        "source_state_index": None if source is None else source.get("state_index"),
        "derivation": derivation,
    }
    if atype == "post_blind":
        if forced_bet_type is not None:
            # Structure-derived posts (straddles) name their own type; the
            # amount heuristic below only ever knew about the two blinds.
            payload["forced_bet_type"] = forced_bet_type
        elif amount is not None and amount <= 0.5 + _EPS:
            # Missed-blind POST on ClubWPT is a live big-blind post unless sized as SB.
            payload["forced_bet_type"] = "small_blind"
        else:
            payload["forced_bet_type"] = "big_blind"
        payload["is_live_post"] = True if is_live_post is None else bool(is_live_post)
    return payload


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


# ClubWPT timelines render amounts in BB units. SB/BB posts are structural
# priors for preflop commitment tracking -- not invented voluntary actions.
_SB_BLIND_BB = 0.5
_BB_BLIND_BB = 1.0
_LEDGER_INFER = "ledger_infer"


def _blind_priors(positions: dict[int, str]) -> dict[int, float]:
    """Street commitment already out before voluntary preflop action."""
    priors: dict[int, float] = {}
    for seat, position in positions.items():
        if position == "SB":
            priors[seat] = _SB_BLIND_BB
        elif position == "BB":
            priors[seat] = _BB_BLIND_BB
    return priors


def _mark_ledger_infer(action: dict[str, Any]) -> None:
    prev = str(action.get("derivation") or "")
    if not prev or prev == _LEDGER_INFER:
        action["derivation"] = _LEDGER_INFER
    elif _LEDGER_INFER not in prev.split("+"):
        action["derivation"] = f"{prev}+{_LEDGER_INFER}"


def _infer_raise_to_from_later_call(
    street_actions: list[dict[str, Any]],
    raise_index: int,
    seat: int,
    commit_before: dict[int, float],
) -> float | None:
    """Back-solve an unsized raise/bet from later sized calls.

    Two redundant bridges (either is enough; both must agree when both exist):
      1. Callers of THIS raise, before the next aggression: each sized call C
         from a seat at prior P implies raise-to level P+C.
      2. The raiser's own later sized call under a known facing F: post-raise
         commitment is F-C, so raise size is (F-C) - commit_before[R].

    Never invents when levels conflict or evidence is short.
    """
    raiser_prior = float(commit_before.get(seat, 0.0))
    levels_from_callers: list[float] = []
    commit = dict(commit_before)
    facing = max(commit.values()) if commit else 0.0
    facing_known = False  # this raise dirties facing
    self_bridge: float | None = None

    for later in street_actions[raise_index + 1 :]:
        later_type = later.get("action_type")
        later_seat = later.get("seat")
        later_amount = later.get("amount")
        if later_seat is None or later_type not in _MONEY_ACTIONS:
            continue
        if later_amount is None:
            if later_type in {"bet", "raise", "all-in"}:
                # Next aggression ends the "callers of this raise" window.
                break
            if later_seat == seat:
                return None
            continue
        prior = float(commit.get(later_seat, 0.0))
        if later_type in {"bet", "raise", "all-in"}:
            if later_seat == seat:
                return None
            # Next sized aggression ends caller-bridge collection; still allow
            # a later self-call bridge under the new facing.
            commit[later_seat] = round(prior + float(later_amount), 2)
            facing = commit[later_seat]
            facing_known = True
            levels_from_callers = list(levels_from_callers)  # freeze caller levels
            # Stop collecting callers; keep scanning for a self-call bridge.
            continue
        # sized call
        if not facing_known:
            # Still the window facing this unsized raise.
            level = round(prior + float(later_amount), 2)
            if later_seat != seat and level > raiser_prior + _EPS:
                levels_from_callers.append(level)
            commit[later_seat] = level
            continue
        # Facing re-established by a later known raise.
        if later_seat == seat and self_bridge is None:
            level_after_raise = round(facing - float(later_amount), 2)
            fill = round(level_after_raise - raiser_prior, 2)
            if fill >= 0.5 - _EPS and level_after_raise > raiser_prior + _EPS:
                self_bridge = fill
            break
        commit[later_seat] = facing

    candidates: list[float] = []
    if levels_from_callers:
        level0 = levels_from_callers[0]
        if all(abs(level - level0) <= _EPS for level in levels_from_callers):
            fill = round(level0 - raiser_prior, 2)
            if fill >= 0.5 - _EPS:
                candidates.append(fill)
    if self_bridge is not None:
        candidates.append(self_bridge)
    if not candidates:
        return None
    if any(abs(c - candidates[0]) > _EPS for c in candidates):
        return None
    return candidates[0]

def _backfill_ledger_amounts(
    actions: list[dict[str, Any]], positions: dict[int, str]
) -> list[dict[str, Any]]:
    """Fill the minimum money sizes the ledger still needs, from poker rules.

    Contract (redundant, never invent when underdetermined):
      * A call's size is facing_level - seat_prior when both are known.
      * SB=0.5 / BB=1.0 are preflop commitment priors (BB units).
      * An unsized raise/bet is filled only when this seat's later sized call
        under a known facing uniquely implies the raise-to level.
      * An unsized raise dirties facing: later calls are NOT filled as limps
        to the blind while that raise is unresolved.
      * Leave amount=None when evidence conflicts or is short.

    This is NOT OCR backfill of a refused crop from a later frame of the same
    box. It uses independently measured later actions (stack deltas, sized
    calls) and structural blinds. Derivation gains ``ledger_infer`` so the
    channel is auditable.
    """
    if not actions:
        return actions

    for _ in range(8):
        changed = False
        for street in ("preflop", "flop", "turn", "river"):
            street_actions = [a for a in actions if a.get("street") == street]
            if not street_actions:
                continue
            commitment = _blind_priors(positions) if street == "preflop" else {}
            facing = max(commitment.values()) if commitment else 0.0
            facing_known = True

            for index, action in enumerate(street_actions):
                atype = action.get("action_type")
                seat = action.get("seat")
                if seat is None or atype not in _MONEY_ACTIONS:
                    continue
                prior = float(commitment.get(seat, 0.0))
                amount = action.get("amount")

                if atype in {"bet", "raise", "all-in"} and amount is None:
                    inferred = _infer_raise_to_from_later_call(
                        street_actions, index, seat, commitment
                    )
                    if inferred is not None:
                        action["amount"] = inferred
                        _mark_ledger_infer(action)
                        amount = inferred
                        changed = True
                    else:
                        # Unresolved aggression: do not treat blind level as facing.
                        facing_known = False
                        continue

                if atype == "call" and amount is None:
                    if facing_known and facing > prior + _EPS:
                        fill = round(facing - prior, 2)
                        if fill >= 0.5 - _EPS:
                            action["amount"] = fill
                            _mark_ledger_infer(action)
                            amount = fill
                            changed = True

                if amount is None:
                    if atype in {"bet", "raise", "all-in"}:
                        facing_known = False
                    continue

                if atype in {"bet", "raise", "all-in"}:
                    new_total = round(prior + float(amount), 2)
                    commitment[seat] = new_total
                    facing = new_total
                    facing_known = True
                elif atype == "call":
                    if facing_known:
                        # A call matches the facing level; use absolute commitment
                        # so a prior unknown raise does not leave prior=0.
                        commitment[seat] = facing
                    else:
                        commitment[seat] = round(prior + float(amount), 2)

        if not changed:
            break

    return actions


def _opening_commitments_from_ledger(
    actions: list[dict[str, Any]], positions: dict[int, str]
) -> dict[int, float]:
    """Chips already committed at the first observed state, from the filled ledger.

    Only actions booked on the opening state (source_state_index matching the
    minimum observed index, typically 0) update the opening commitment. Later
    sized actions must not rewrite what was already on the felt at hand open.
    """
    commit = _blind_priors(positions)
    opening_indices = {
        a.get("source_state_index")
        for a in actions
        if a.get("source_state_index") is not None
    }
    if not opening_indices:
        return commit
    opening_index = min(opening_indices)
    for action in actions:
        if action.get("street") != "preflop":
            break
        if action.get("source_state_index") != opening_index:
            continue
        atype = action.get("action_type")
        seat = action.get("seat")
        amount = action.get("amount")
        if seat is None or atype not in _MONEY_ACTIONS or amount is None:
            continue
        prior = float(commit.get(seat, 0.0))
        commit[seat] = round(prior + float(amount), 2)
    return commit


def _apply_ledger_starting_stacks(
    player_rows: list[dict[str, Any]],
    hand: list[dict[str, Any]],
    opening_commit: dict[int, float],
    positions: dict[int, str],
    actions: list[dict[str, Any]],
) -> None:
    """Resolve ``committed_at_start_unknown`` when the ledger pins the standing bet.

    Only fills when the raw first-state stack is readable and the unknown reason
    was specifically the standing-bet half (not an unreadable stack box). Never
    invents a stack that was never observed.

    Structural blinds alone may clear SB/BB only for pure OCR refusal
    (``committed_at_start_unknown``), never for contradictory standing-bet
    evidence (``committed_at_start_conflict``). They must also NOT clear a seat
    whose standing raise/call is still unsized.
    """
    if not hand:
        return
    first_stacks = hand[0].get("stacks") or {}
    bare_blinds = _blind_priors(positions)
    opening_indices = {
        a.get("source_state_index")
        for a in actions
        if a.get("source_state_index") is not None
    }
    opening_index = min(opening_indices) if opening_indices else None
    for row in player_rows:
        seat = row.get("seat")
        if seat is None:
            continue
        reason = row.get("starting_stack_unknown")
        if reason not in {
            "committed_at_start_unknown",
            "committed_at_start_conflict",
        }:
            continue
        if row.get("starting_stack") is not None:
            continue
        base = first_stacks.get(seat)
        committed = opening_commit.get(seat)
        if base is None or committed is None:
            continue
        # Never paper over contradictory standing-bet evidence with blinds or
        # an inferred open -- the window said the amount is not unique.
        if reason == "committed_at_start_conflict":
            continue
        unsized_opening = any(
            a.get("seat") == seat
            and a.get("street") == "preflop"
            and a.get("action_type") in _MONEY_ACTIONS
            and a.get("amount") is None
            and (
                opening_index is None
                or a.get("source_state_index") == opening_index
            )
            for a in actions
        )
        bare = bare_blinds.get(seat)
        if unsized_opening:
            continue
        if (
            bare is not None
            and abs(float(committed) - float(bare)) <= _EPS
            and positions.get(seat) not in {"SB", "BB"}
        ):
            continue
        row["starting_stack"] = round(float(base) + float(committed), 2)
        row["starting_stack_unknown"] = None

def _clear_blind_only_unmeasured(
    hand: list[dict[str, Any]], positions: dict[int, str]
) -> None:
    """Drop first-state unmeasured markers that are only unread SB/BB posts.

    A pill-less refused bet on the blind seat is the blind OCR failing, not a
    voluntary money action of unknown size. Structural blinds already size it.
    """
    if not hand:
        return
    first = hand[0]
    kept: list[int] = []
    for seat in first.get("unmeasured_transitions") or []:
        position = positions.get(seat, "")
        pill = (first.get("pills") or {}).get(seat)
        forced_poster = position in {"SB", "BB"} or _is_straddle_position(position)
        if forced_poster and pill in {None, "post_blind"}:
            continue
        kept.append(seat)
    first["unmeasured_transitions"] = kept


def _clear_settlement_sweep_unmeasured(hand: list[dict[str, Any]]) -> None:
    """Drop unmeasured markers that are only a winner collecting the pot.

    Across a coverage gap the winner's stack jumps by about the pot with no
    voluntary money pill. That is settlement, not an unsized put-in.
    """
    for prev, cur in zip(hand, hand[1:], strict=False):
        um = list(cur.get("unmeasured_transitions") or [])
        if not um:
            continue
        kept: list[int] = []
        for seat in um:
            before = (prev.get("stacks") or {}).get(seat)
            after = (cur.get("stacks") or {}).get(seat)
            pill = (cur.get("pills") or {}).get(seat)
            fresh_money = pill in {
                "raise", "bet", "call", "all-in", "post_blind", rd.PILL_BET_OR_CALL
            } and (prev.get("pills") or {}).get(seat) != pill
            if (
                before is not None
                and after is not None
                and after > before + 0.5 + _EPS
                and not fresh_money
            ):
                continue
            kept.append(seat)
        cur["unmeasured_transitions"] = kept


def _clear_unmeasured_for_ledger_inferred_seats(
    hand: list[dict[str, Any]], actions: list[dict[str, Any]]
) -> None:
    """Drop unmeasured markers for seats whose money was sized by ledger_infer.

    ``unmeasured_transitions`` often records the same hole ``amount_unknown``
    already booked on an action. After ledger inference sizes that action, the
    marker would otherwise keep ``amounts_unknown_in_ledger`` fatal forever.
    Seats that still have an unsized money action keep their markers.
    """
    still_unsized = {
        a.get("seat")
        for a in actions
        if a.get("action_type") in _MONEY_ACTIONS and a.get("amount") is None
    }
    inferred_seats = {
        a.get("seat")
        for a in actions
        if a.get("action_type") in _MONEY_ACTIONS
        and a.get("amount") is not None
        and _LEDGER_INFER in str(a.get("derivation") or "").split("+")
        and a.get("seat") not in still_unsized
    }
    if not inferred_seats:
        return
    for state in hand:
        um = list(state.get("unmeasured_transitions") or [])
        if not um:
            continue
        state["unmeasured_transitions"] = [s for s in um if s not in inferred_seats]


def _lethal_coverage_gap_count(hand: list[dict[str, Any]]) -> int:
    """Coverage gaps that still imply a missing street or unsized money.

    Sparse keyframes often set ``coverage_gap`` on a transition where the far
    side still shows the sized action (pill + bet). Those are not fatal once the
    ledger is sound. A board-length advance across the gap, or an unmeasured
    marker that survived settlement/blind cleanup, still is.
    """
    lethal = 0
    for prev, cur in zip(hand, hand[1:], strict=False):
        if not cur.get("coverage_gap"):
            continue
        board_advanced = len(cur.get("board_cards") or []) > len(
            prev.get("board_cards") or []
        )
        if board_advanced or (cur.get("unmeasured_transitions") or []):
            lethal += 1
    return lethal

def reconstruct(
    hand: list[dict[str, Any]],
    hand_number: int,
    *,
    session_dealt_seats: set[int] | None = None,
    forced_posts_bb: tuple[float, ...] = (0.5, 1.0),
    deal_open_observed: bool = False,
) -> dict[str, Any]:
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

    # The roster combines direct card/pill evidence with stable opening
    # occupancy when capture begins after early folds.  Positions are then
    # assigned exclusively from the dealer button and this physical seat ring.
    players = _player_seats(hand, dealer_seat,
                            session_dealt_seats=session_dealt_seats,
                            deal_open_observed=deal_open_observed)
    positions = _positions(players, dealer_seat,
                           n_straddles=max(0, len(forced_posts_bb) - 2))
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
        player_rows.append({
            "seat": seat,
            "position": positions.get(seat, ""),
            "player_name": player_name[seat],
            "starting_stack": _observed_starting_stack(hand, seat),
            # WHY it is None, when the reader refused it. `starting_stack` already
            # refuses to guess (note 12); this says so out loud, so an unknown
            # starting stack is distinguishable from a seat whose box was simply
            # not detected on the hand's first state.
            "starting_stack_unknown": _observed_starting_stack_unknown(hand, seat),
            "is_hero": seat == 0,
            "shown_cards": shown_cards.get(seat),
        })

    actions = _reconstruct_actions(hand, positions, player_name, forced_posts_bb)
    # Minimum-bar amount repair: calls are determined by the facing level,
    # blinds are structural, and unsized raises are back-solved only when later
    # sized calls uniquely pin the level. Runs before the unknown-ledger gate.
    actions = _backfill_ledger_amounts(actions, positions)
    _clear_blind_only_unmeasured(hand, positions)
    _clear_settlement_sweep_unmeasured(hand)
    _clear_unmeasured_for_ledger_inferred_seats(hand, actions)
    opening_commit = _opening_commitments_from_ledger(actions, positions)
    _apply_ledger_starting_stacks(
        player_rows, hand, opening_commit, positions, actions
    )

    # UNKNOWN money reaching the ledger. Two shapes, both fatal to "complete":
    #   * a money action the spine PROVED happened but could not SIZE, and
    #   * a transition whose stack delta was unmeasurable because a read was
    #     refused, so no action could be derived from it at all.
    # Neither is visible in `amounts_unknown`, which counts crops, not ledger
    # entries: a hand can have many unread crops and a sound ledger, or one unread
    # crop sitting exactly on the transition that carried the money.
    # Computed HERE, before the pot estimators, because `contrib` must consult it:
    # it used to be computed after them, so the contribution estimator voted in
    # the pot consensus with a null-amount action in the very ledger it was
    # summing, and 6 of the 10 development hands carrying an unknown in their
    # ledger published `reconciled: true`.
    unmeasured_transitions = sum(len(s.get("unmeasured_transitions") or []) for s in hand)
    unknown_money_actions = sum(1 for a in actions
                                if a["action_type"] in _MONEY_ACTIONS
                                and a.get("amount") is None)
    amounts_unknown_in_ledger = bool(unmeasured_transitions or unknown_money_actions)
    coverage_gaps = sum(1 for s in hand if s.get("coverage_gap"))
    # Fatal coverage: board advanced across the hole, or an unmeasured marker
    # remains after blind/settlement cleanup. Soft keyframe gaps that still
    # published a sized far-side action do not reject the hand.
    lethal_coverage_gaps = _lethal_coverage_gap_count(hand)

    # per-street end pot (last stable pot at each board count) + final pot.
    # Only states up to the settlement (pot swept to the winner) count: after it
    # the display is already showing the next deal's antes/blinds.
    settle = _settle_index(hand, set(players))
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
    # Estimators whose inputs carry a REFUSED read abstain; they are removed from
    # the pot consensus below rather than voting with a hole in them.
    #
    # `contrib` starts holed whenever the MONEY LEDGER itself carries an unknown,
    # not only when a stack SERIES does: an action of unknown size, or a
    # transition nothing could measure, is money the contribution sum cannot see
    # (a standing bet whose read was refused is debited from no stack and posted
    # to no pot text). "AN ESTIMATOR WITH AN UNKNOWN INPUT IS REMOVED FROM THE
    # CANDIDATE LIST" -- the consensus comment below states the rule; this makes
    # the contribution estimator actually obey it.
    contrib_holed = amounts_unknown_in_ledger
    win_holed_by_seat: dict[int, bool] = {}
    hero_net_holed = False
    for seat in players:
        holed = _series_has_hole(settled_states, seat)
        win_holed_by_seat[seat] = holed
        if holed:
            contrib_holed = True
            if seat == 0:
                hero_net_holed = True
        series = _stack_series(settled_states, seat)
        if not series and not holed:
            # Widening the window to the WHOLE hand is only legitimate when the
            # settled window genuinely observed nothing. When it observed a
            # refusal, widening substitutes post-settlement states -- the next
            # deal's antes and auto top-ups -- for a read the pixels declined to
            # supply, which is a backfill from a different state.
            series = _stack_series(hand, seat)
        if not series:
            continue
        low = min(series)
        pre_settle = [s["stacks"][seat] for s in settled_states[:-1] if seat in s["stacks"]]
        if pre_settle:
            contributed += max(series[0] - pre_settle[-1], 0.0)
        gain = series[-1] - low
        phantom = (seat in folded_seats and final_pot is not None
                   and gain > 1.5 * final_pot + 3.0)
        # ...and the floor on the other side. A seat that swept the pot gained
        # approximately the pot; a gain that is a small FRACTION of it is stack
        # OCR jitter, not a sweep. The guard above only ever rejected gains that
        # were too LARGE, so 0.5 BB of jitter between two consecutive reads on a
        # 17.5 BB pot (0.03x) named a winner, published result="Villain wins" and
        # -- through terminal_event -- promoted a recording-truncated hand to
        # completion_status "complete" at confidence 1.0 with no warning.
        if final_pot is not None and gain < _WIN_GAIN_MIN_POT_RATIO * final_pot:
            phantom = True
        if not phantom and gain > win_gain + _EPS:
            win_gain, winner_seat = gain, seat
        if seat == 0 and not hero_net_holed:
            # Net over the hand, not over the observed window: chips already in
            # front of the hero at the first sample left its displayed stack
            # before the series began, so the raw delta cannot see them. See
            # _committed_at_start.
            #
            # ABSTAINS on a holed series. `series[0]` is the first SURVIVING
            # reading, so a refusal anywhere before it silently moves the baseline
            # forward past chips the hero had already committed -- the same defect
            # note 12 fixed for `starting_stack`, on the field a reviewer reads as
            # the hand's bottom line. An unknown net is published as unknown.
            # ... and abstains the same way when the COMMITTED half is unknown (a
            # refused bet read inside the committed-scan window): the net cannot
            # be computed from a lower bound.
            hero_committed = _committed_at_start(hand, 0)
            if hero_committed is not None:
                hero_bb_won = round(series[-1] - series[0] - hero_committed, 2)
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
    # ... but `pots` silently SKIPS refused states, so when the pot read at the
    # start of the window was REFUSED, pots[0] is a RE-BASED value from later in
    # the hand -- money committed between the window start and the first
    # surviving read is then counted twice (once inside pots[0], once in
    # `contributed`), and the inflated sum can corroborate an equally inflated
    # win estimate and outvote the correct pot text 2:1, publishing a wrong pot
    # as reconciled with zero visible signal. A refused pot read at or before
    # the first surviving one is therefore an UNKNOWN INPUT to the contribution
    # estimator, and "AN ESTIMATOR WITH AN UNKNOWN INPUT IS REMOVED FROM THE
    # CANDIDATE LIST, not down-weighted" (the consensus rule below). Refusals
    # AFTER the first surviving read do not touch initial_pot and cost nothing.
    for s in settled_states:
        if s.get("pot_unknown"):
            initial_pot_refused = True
            break
        if s["pot"] is not None:
            initial_pot_refused = False
            break
    else:
        initial_pot_refused = False

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
    #
    # AN ESTIMATOR WITH AN UNKNOWN INPUT IS REMOVED FROM THE CANDIDATE LIST, not
    # down-weighted. `contrib` sums stack differences across the settled window,
    # so a refused read inside that window makes its sum wrong by an unmeasured
    # amount while looking exactly like a clean one -- and it would then be one of
    # the two "independent estimates that agree" that `reconciled` means. With
    # fewer than two survivors `reconciled` is False and pot_not_reconciled fires,
    # which is the correct, visible outcome.
    pot_text = final_pot
    contrib_pot = (round(initial_pot + contributed, 2)
                   if (initial_pot is not None and not contrib_holed
                       and not initial_pot_refused) else None)
    win_pot = (round(win_gain, 2)
               if (winner_seat is not None and win_gain > _EPS
                   and not win_holed_by_seat.get(winner_seat, False)) else None)
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
    # villain resolution as unobserved. (A reconciled winner is kept: 13 of the
    # 31 development hands are hero-folds whose villain sweep reconciles via the
    # contrib/win consensus -- e.g. g0723a hand 3, seat 7 sweeping 135.8 -- and
    # every one keeps its winner through this demotion.)
    if hero_folded and winner_seat is not None and not reconciled:
        winner_seat, win_gain = None, 0.0
        result = "Hero folds"
        final_pot = pot_text  # the consensus/derived pot rode on the phantom sweep
        hero_fold_only = True
        pot_text_dropped = False
        reconciled = True

    complete_cards = len(hero) == 2 and len(board) in {0, 3, 4, 5} and cards_unique
    complete = (bool(hero) and cards_unique and final_pot is not None and bool(actions)
                and (winner_seat is not None or hero_fold_only)
                # A hand carrying money of unknown size, or a transition nothing
                # could measure, is not a COMPLETE record of what happened,
                # whatever else holds. PLAN.md: "Mark an incomplete sequence
                # non-authoritative."
                and not amounts_unknown_in_ledger
                # A tab/lobby covering the table across a critical state change
                # leaves actions unobserved even when stacks happened to look
                # continuous afterward. Soft keyframe gaps that still publish a
                # sized far-side action are not lethal (see lethal_coverage_gaps).
                and not lethal_coverage_gaps)

    # Which terminal event the spine actually OBSERVED, for the app's completion
    # evidence. Additive: no existing consumer reads this key. Anything the spine
    # could not see stays "unobserved", which blocks study readiness downstream.
    if winner_seat is not None and win_holed_by_seat.get(winner_seat, False):
        # The sweep was read off a stack series with a refusal inside the settled
        # window, so "this seat gained the pot" rests on a step nothing measured.
        # A terminal event must not be PROMOTED off an unmeasured sweep -- naming
        # a showdown or a fold-win here is what turns an unreadable frame into a
        # completion claim downstream.
        terminal_event = "unobserved"
    elif winner_seat is not None:
        # A showdown is two or more seats CONTESTING a COMPLETED board. Both halves
        # were missing. `len(last["dealt_in"]) >= 2` alone is satisfied on nearly
        # every hand, because assign_regions marks hero dealt-in whenever the hero's
        # hole cards are visible and the client leaves a folded hero's cards on
        # screen greyed out for the rest of the hand -- so a hero who folded preflop
        # still counts. And the board was never consulted at all: 9 of the 21
        # development hands claimed "showdown", 8 of them with fewer than 5 board
        # cards and one with no board whatsoever. Seats whose last reconstructed
        # action is a fold are excluded, which is what removes the folded hero.
        #
        # `last["dealt_in"]` cannot supply the count: on 19 of the 21 development
        # hands the last state of the segment is already the NEXT deal, where every
        # seat is dealt in and the board is empty, so the old test was true on
        # essentially every hand with a winner. The action ledger is the witness --
        # a seat whose last reconstructed action is a fold is out; a seat that never
        # acted counts as live, which biases toward NOT asserting a fold.
        # Seats recovered only from stable opening occupancy had already folded
        # before capture.  They belong in the positional ring, but cannot turn a
        # one-player fold win into a fabricated multiway showdown merely because
        # no reconstructable FOLD action exists for their pre-capture fold.
        preobserved_folded = set(players) - _opening_live_seats(hand[0])
        contenders = [
            seat for seat in players
            if seat not in preobserved_folded and last_action.get(seat) != "fold"
        ]
        if len(contenders) <= 1:
            terminal_event = "fold_win"
        elif len(board) == 5:
            terminal_event = "showdown"
        else:
            # Two or more seats still contesting a pot that was swept on a board
            # that never completed. Nobody shows down an incomplete board, so the
            # later streets simply were not observed -- say so, rather than assert
            # a fold that did not happen or a showdown that could not have.
            terminal_event = "unobserved"
    elif hero_fold_only:
        terminal_event = "hero_fold"
    else:
        terminal_event = "unobserved"

    warnings: list[str] = []
    if len(hero) != 2 and 0 in players:
        # Only a defect when the hero IS a player in this hand. A hand the
        # hero sat out has no hero cards by construction, and flagging that
        # would mark every correctly-reconstructed sat-out hand defective.
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

    # ---- reconstruction-correctness nets -------------------------------------
    # Everything above warns about a value the spine could not COMPUTE. The four
    # below warn about a reconstruction that computed cleanly and is still wrong:
    # each corresponds to a real failure that produced a confident export with no
    # warning at all.
    side_pot = max((s["side_pot"] for s in hand if s.get("side_pot") is not None),
                   default=None)
    # A REFUSED side-pot read is a DETECTED second pot whose amount is unknown --
    # not the same fact as "no side pot". `side_pot_unknown` was plumbed
    # end-to-end (region_detections sets it on every state) and read by nothing,
    # so a refusal on the side band silenced side_pot_unsupported -- a SPINE_FATAL
    # code -- and the hand was indistinguishable from a single-pot hand. Latent on
    # the development corpus (0 of 716 states carry it; the one side-pot recording
    # reads both boxes), which is exactly why it needs consuming before the
    # geometry that trips it arrives.
    side_pot_refused = any(s.get("side_pot_unknown") for s in hand)
    if side_pot is not None or side_pot_refused:
        # The detector reads both pots correctly, but the spine models the pot as
        # a single scalar and Hand.pot_size cannot hold a main/side pair. Rather
        # than discard the side pot silently (which is what happened before, via
        # a max-confidence pick between the two boxes), say so and let the hand be
        # rejected. See PLAN.md: side pots are supported once held-out truth exists.
        #
        # KNOWN LIMIT, stated rather than papered over: this refusal rests entirely
        # on the detector finding a second pot_text box, and on the one development
        # recording that has a side pot those boxes score 0.382-0.758 -- mostly
        # under 0.6. Suppress them (a higher detector threshold, an occlusion, a
        # different render) and the same unrepresentable hand exports clean at
        # confidence 1.0. The obvious corroborating evidence -- two seats all-in
        # with a third still live -- is NOT usable as a substitute on this corpus:
        # the side-pot recording yields only ONE all-in action, while two baseline
        # hands show two all-in seats each and have no side pot. A rule built on
        # either signal would be tuned against material that does not support it.
        # Closing this needs a recording with both a measured side pot and a
        # measured all-in ledger, which the development corpus does not contain.
        warnings.append("side_pot_unsupported")
    if any(s.get("board_row_missed") for s in hand):
        # A community row was on screen and not one of its cards was zoned as
        # board. An empty board is a legal value everywhere downstream, so without
        # this the failure is invisible by construction.
        #
        # Neither the old `>= 3 states` run nor the old `and not board` conjunct
        # survived measurement. State collapse compresses a hand to its DISTINCT
        # states, and exported development hands hold a street for as few as one
        # (g0711 hand 6 holds its river board for exactly 1 distinct state; g0723a
        # hand 5 for 2), so a three-state run asked for more evidence than a river
        # supplies. And `not board` silenced the net entirely on a hand whose flop
        # WAS captured and whose later rows were all missed -- the exact shape a
        # mid-hand zoning drift produces.
        #
        # A state's board_row_missed is a GEOMETRIC fact about the frame, not a
        # numeric read, so it needs no run to be believed. Measured cost of firing
        # on one state: 0 of 470 states across the five development recordings
        # raise the flag at all.
        warnings.append("board_zone_yield_zero")
    for field, code in (("board_cards", "board_card_identity_split"),
                        ("hero_cards", "hero_card_identity_split")):
        if any(s.get(f"{field}_identity_split") for s in hand):
            # The classifier read the SAME card slot two different ways while the
            # card sat on the table. The debounce picks the majority and reports
            # it as fact; measured on the 07-23 3.21 PM recording it picked wrong,
            # and 'Js 6h 4c 8d' exported for a real Js 6h 4c 8h at confidence 1.0
            # with tags [] -- a board with one heart standing in for a board with
            # two, i.e. a different hand for every equity and texture conclusion.
            # A contested identity is unknown, not a vote.
            warnings.append(code)
    if any(s.get("board_row_partial") for s in hand):
        # The row was on screen and only PART of it was zoned board. Every
        # surviving count is legal downstream, so this is exactly as silent as
        # losing the whole row: a 5-card river board reported as a completed
        # 4-card turn board, at confidence 1.0, with a showdown result asserted on
        # a board missing its river card. Gated on one state for the same reason
        # as board_zone_yield_zero above -- a river street does not supply three.
        warnings.append("board_zone_yield_partial")
    # NOTE: two sibling invariants -- "board empty while streets advanced" and
    # "every action on one street of a multi-street hand" -- are checked in
    # validate_yolo_card_timeline, NOT here. Inside the spine both halves of each
    # pair derive from the same `board_cards` readings (`streets` from
    # _street_events, the voted board from _vote_board, an action's street from
    # the board high-water mark), so they cannot disagree and the checks would be
    # dead code. Across the fields of an already-serialized hand -- a hand-edited
    # timeline, a manual correction, a different producer -- they can, and that is
    # where they belong.
    pot_text_off_column = sum(s.get("pot_text_off_column", 0) for s in hand)
    stack_conflicts = sum(s.get("stack_conflicts", 0) for s in hand)
    bet_conflicts = sum(s.get("bet_conflicts", 0) for s in hand)
    hero_seat_confirmed = any(s.get("hero_seat_confirmed") for s in hand)
    amounts_unknown_by_code: dict[str, int] = {}
    for s in hand:
        for code, n in (s.get("amounts_unknown_by_code") or {}).items():
            amounts_unknown_by_code[code] = amounts_unknown_by_code.get(code, 0) + n
    if amounts_unknown_in_ledger:
        # The hand's MONEY LEDGER -- not merely its crops -- carries an unknown.
        # Either an action whose size no evidence supplies, or a transition whose
        # stack delta was unmeasurable. Both make every derived total (pot
        # consensus, contributions, hero net) rest on a step nothing measured, so
        # the hand must not be presented as an authoritative record of what
        # happened. Fatal at the export gate; see SPINE_FATAL_CODES.
        warnings.append("amounts_unknown_in_ledger")
    if lethal_coverage_gaps:
        # The table was unobserved across a board advance or a money transition
        # nothing could size. Soft keyframe gaps that still show a sized action
        # on the far side are kept in coverage_gaps for audit but are not fatal.
        warnings.append("mid_hand_coverage_gap")
    if any(row.get("starting_stack_unknown") for row in player_rows):
        # A seat's starting stack was REFUSED (as opposed to a box that was never
        # detected). Accounting downstream rejects a None starting stack
        # outright, so the hand can never become an authoritative money record --
        # yet the fact used to travel only as CompletionEvidence.extra
        # (cv_starting_stack_unknown), which nothing read, and the hand exported
        # as completion_status "complete" with warning_codes=[] and then failed
        # PERMANENTLY at the accounting layer with no clearing action. Same
        # class of unknown as amounts_unknown_in_ledger; same fatal treatment.
        warnings.append("starting_stack_unknown")
    ledger_hits = _stack_ledger_violations(settled_states[:-1], players)
    if ledger_hits:
        # A stack that multiplies before the pot is awarded is arithmetically
        # impossible, so every number derived from that seat's ledger -- its
        # contribution, its action sizes, and the hero net computed from them --
        # rests on a read that cannot be true.
        warnings.append("stack_ledger_incoherent")
    residual = _contribution_residual({"pot": final_pot, "actions": actions})
    if residual is not None and -residual > (_uncalled_shove_headroom(actions)
                                             + _CONTRIBUTION_RESIDUAL_SLACK_BB):
        # Chip conservation. See _contribution_residual: with incremental amount
        # semantics the action ledger cannot sum ABOVE the pot, so a negative
        # residual says the reconstruction booked money the pot does not contain.
        # The one legal source of one -- an over-shove refunded before the sweep --
        # is measured and subtracted, so this fires only on money that has no
        # explanation (the duplicate-action defect, at -3.0 with zero headroom).
        warnings.append("contributions_exceed_pot")
    if _result_contradicts_hero_net(result, hero_bb_won):
        # Two fields off the same stack ledger, published together and never
        # compared: "Villain wins" alongside hero_bb_won=+65.2.
        warnings.append("result_contradicts_hero_net")
    unanchored_cards = sum(s.get("unanchored_cards", 0) for s in hand)
    anchor_missing_states = sum(1 for s in hand if not s.get("anchor_ok", True))
    if unanchored_cards:
        # Cards were detected and then dropped because no table transform could be
        # fitted. Failing closed is correct, but the record is now incomplete.
        warnings.append("anchor_unavailable")

    return {
        "hand_number": hand_number,
        "t_start": hand[0]["time_s"],
        "t_end": hand[-1]["time_s"],
        # The forced-post structure this hand was reconstructed under (BB
        # units, button-relative: sb, bb, then any straddles). Additive key;
        # advisory evidence for the operator's blind attestation downstream.
        "forced_posts_bb": list(forced_posts_bb),
        "n_states": len(hand),
        "hero": hero or None,
        "board": board,
        "dealer_seat": dealer_seat,
        "players": player_rows,
        "streets": streets,
        "actions": actions,
        # "pot" keeps its exact meaning: the MAIN pot scalar. "side_pot" is
        # additive -- every existing consumer (thresholds, export, validation)
        # reads a value whose semantics never changed.
        "pot": final_pot,
        "side_pot": side_pot,
        "pot_text_dropped": pot_text_dropped,
        "pot_text_off_column": pot_text_off_column,
        "stack_conflicts": stack_conflicts,
        "bet_conflicts": bet_conflicts,
        "hero_seat_confirmed": hero_seat_confirmed,
        "pot_collapses_repaired": sum(s.get("pot_collapses_repaired", 0) for s in hand),
        "amounts_unknown": sum(s.get("amounts_unknown", 0) for s in hand),
        # Per-reason breakdown of the count above. "18 unread amounts" cannot tell
        # a systematic reader failure on one region from scattered occlusions.
        "amounts_unknown_by_code": amounts_unknown_by_code,
        # Transitions whose money movement nothing could measure, and money
        # actions the spine proved happened but could not size.
        "unmeasured_transitions": unmeasured_transitions,
        "unknown_money_actions": unknown_money_actions,
        "coverage_gaps": coverage_gaps,
        "settle_scan_skipped": sum(1 for s in hand if s.get("settle_scan_skipped")),
        "anchor_missing_states": anchor_missing_states,
        "winner_seat": winner_seat,
        "win_gain": round(win_gain, 2),
        "result": result,
        "terminal_event": terminal_event,
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
def layout_support_statement() -> str:
    """One sentence for the Settings/diagnostics readout. Lives beside the
    facts it describes so a drifted duplicate cannot advertise support the
    readers don't have."""
    return (
        "Supported when the table anchor fits the frame (a similarity fit over "
        "the seat landmarks) and HUD glyphs render at a normalizable size "
        f"(fitted scale x {_REF_GLYPH_H_PX:.0f}px >= "
        f"{_MIN_RENORMALIZABLE_GLYPH_PX:.0f}px). Window width x height alone "
        "does not decide; a recording failing the anchor or glyph test is "
        "stamped '-unsupported' on the hand's layout profile."
    )


def _layout_profile(frames: list[rd.Frame], session_anchor=None) -> str:
    """The render geometry this session was reconstructed from, as a fact on the
    record: "<W>x<H>" of the modal frame, plus "-unsupported" when the session
    is outside what the calibrated readers can handle.

    "Unsupported" is decided by what actually bounds the readers -- can a table
    anchor be fitted at all, and do HUD glyphs render at a size the OCR's
    renormalized re-read is validated for -- not by the raw window dimensions.
    The old W x H lower bound was one-sided and misleading (see the constants'
    comment above): it flagged a small window that read fine and said nothing
    about any other failure mode. The string format is unchanged; every
    consumer parses only the "-unsupported" suffix.
    """
    sizes = [(f.width, f.height) for f in frames if f.width and f.height]
    if not sizes:
        return ""
    w, h = _mode(sizes)
    supported = (
        session_anchor is not None
        and session_anchor.s * _REF_GLYPH_H_PX >= _MIN_RENORMALIZABLE_GLYPH_PX
    )
    return f"{w}x{h}" if supported else f"{w}x{h}-unsupported"


def build_hand_timeline(
    frames: list[rd.Frame],
    *,
    sampling_interval_s: float | None = None,
) -> dict[str, Any]:
    states, events = build_states(frames, sampling_interval_s=sampling_interval_s)
    table_frames_list = [f for f in frames if _is_table_frame(f)]
    session_anchor = _session_anchor(table_frames_list)
    segments = [seg for seg in _segment(states) if _has_deal_evidence(seg)]
    # A card slot holds one identity for the whole deal. Settling it on the
    # reading most states saw is only safe once the deal's boundaries are known,
    # which is here and not in build_states.
    for segment in segments:
        _resolve_identity_by_majority(segment)
    # Seats ever dealt cards anywhere in the session. Caps the mid-hand-open
    # stack recruitment in _player_seats: an occupied-but-sitting-out seat must
    # never enter any hand's roster.
    session_dealt_seats = {
        seat for state in states for seat in state["dealt_in"]
    }
    forced_posts_bb, open_seen = _session_forced_posts(segments, session_dealt_seats)
    hands = [
        reconstruct(hand, i, session_dealt_seats=session_dealt_seats,
                    forced_posts_bb=forced_posts_bb,
                    deal_open_observed=observed)
        for i, (hand, observed) in enumerate(
            zip(segments, open_seen, strict=True), start=1
        )
    ]
    table_frames = sum(1 for frame in frames if _is_table_frame(frame))
    nontable_frames = len(frames) - table_frames
    return {
        "metadata": {
            "source": "yolo_region_detections",
            "classes": rd.CLASSES,
            "layout_profile": _layout_profile(frames, session_anchor),
            # Session-voted forced-post chain (BB units: sb, bb, straddles...).
            "forced_post_structure": list(forced_posts_bb),
            "notes": [
                "Offline completed-session reconstruction from 8-class region detections.",
                "Seats assigned via per-class anchors learned from the labeled boxes.",
                "Attribute reads: Model 2 rank/suit + deterministic template OCR "
                "(amounts, pill words); fixtures may pass attrs through directly.",
            ],
        },
        "summary": {
            "frames": len(frames),
            "table_frames": table_frames,
            "nontable_frames": nontable_frames,
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
