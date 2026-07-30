"""Detector-output contract + region assignment for the 7-class YOLO pipeline.

This is the input layer of the reconstruction spine (see build_yolo_hand_timeline).
It is deliberately decoupled from *how* detections were produced so the spine can be
exercised today with synthetic fixtures and labeled ground-truth boxes, and switched
to the trained 7-class detector unchanged later. It reads saved / completed-session
data only; it never captures live tables.

Pipeline position:
    raw detections (per frame) -> assign_regions (seats/zones) + attribute readers
    -> per-frame table state -> [build_yolo_hand_timeline]

The 7 classes and their reconstruction jobs (cv_lab/labeling_poker/config.py, README):
    face_card, card_back, dealer_button, pot_text, stack_text, action_pill,
    active_turn_indicator.

Attribute reads (rank/suit, OCR amounts, pill colour) and the real anchored seat
model are SEPARATE sub-parts; here they are pluggable interfaces with stub
implementations that read straight from each detection's ``attr`` field, so
ground-truth boxes and synthetic fixtures work with no OCR/model.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Canonical rank+suit normaliser shared with the labeler ("KD"/"10C" -> "Kd"/"Tc").
from cv_lab.labeling_poker.config import CLASSES, normalize_card_label

# Card zones are ANCHORED: every face_card center is mapped into the reference
# table basis by a per-frame similarity fit before it is zoned. The old raw
# normalized rectangles (build_yolo_card_timeline._zone_for_box) are deliberately
# NOT used here -- they lost 100% of the community row on a 1.750 aspect ratio.
from cv_lab.scripts.pipeline.landmark_anchor import (
    ANCHOR_MAX_RESID,
    ANCHOR_MIN_TRUSTED_POINTS,
    REF_DET_H,
    REF_DET_W,
    REF_POT_RX,
    REF_POT_SIDE_RY_MIN,
    TableAnchor,
    anchor_from_points,
    zone_for_ref,
)

CLASS_SET = set(CLASSES)

# Coarse per-seat avatar centroids for the current 8-max ClubWPT view, in
# normalized (cx, cy) coords. Seat 0 is the hero (bottom-center). Used as the
# fallback for classes without a learned anchor table below.
SEAT_CENTROIDS: dict[int, tuple[float, float]] = {
    0: (0.50, 0.86),
    1: (0.16, 0.80),
    2: (0.05, 0.50),
    3: (0.18, 0.20),
    4: (0.50, 0.14),
    5: (0.82, 0.20),
    6: (0.95, 0.50),
    7: (0.84, 0.80),
}

# Learned per-CLASS seat anchors (k-means over the human-labeled boxes in
# labels.sqlite3, v00 frames, normalized by true image dims). Each HUD element
# renders at its own per-seat position -- card backs sit above/inside of the
# avatar, bet texts toward the table center -- so nearest-avatar assignment
# flaps between adjacent seats; nearest-class-anchor does not. Regenerate by
# re-running the k-means when the table layout changes.
SEAT_ANCHORS_BY_CLASS: dict[str, dict[int, tuple[float, float]]] = {
    "card_back": {0: (0.500, 0.860), 1: (0.194, 0.623), 2: (0.117, 0.368), 3: (0.174, 0.164),
                  4: (0.480, 0.125), 5: (0.810, 0.164), 6: (0.868, 0.369), 7: (0.791, 0.618)},
    "stack_text": {0: (0.528, 0.809), 1: (0.180, 0.717), 2: (0.106, 0.458), 3: (0.162, 0.255),
                   4: (0.470, 0.195), 5: (0.833, 0.257), 6: (0.889, 0.461), 7: (0.815, 0.714)},
    "action_pill": {0: (0.521, 0.842), 1: (0.182, 0.754), 2: (0.109, 0.499), 3: (0.165, 0.291),
                    4: (0.468, 0.234), 5: (0.826, 0.296), 6: (0.881, 0.501), 7: (0.807, 0.750)},
    "dealer_button": {0: (0.424, 0.640), 1: (0.308, 0.680), 2: (0.154, 0.539), 3: (0.289, 0.245),
                      4: (0.584, 0.263), 5: (0.725, 0.227), 6: (0.787, 0.511), 7: (0.676, 0.671)},
    "active_turn_indicator": {0: (0.447, 0.788), 1: (0.259, 0.689), 2: (0.183, 0.437), 3: (0.239, 0.233),
                              4: (0.546, 0.177), 5: (0.752, 0.239), 6: (0.804, 0.445), 7: (0.733, 0.691)},
    "bet_text": {0: (0.467, 0.579), 1: (0.311, 0.591), 2: (0.256, 0.441), 3: (0.313, 0.313),
                 4: (0.491, 0.312), 5: (0.659, 0.314), 6: (0.714, 0.438), 7: (0.661, 0.581)},
}

# Seat ring in action order starting at the hero seat: on the ClubWPT layout the
# action moves hero (bottom-center) -> bottom-left -> up the left side -> across
# the top -> down the right side, i.e. ascending seat index (dealer -> SB=dealer+1
# -> BB=dealer+2 ... verified against live blind posts). Used for positions.
SEAT_RING: list[int] = [0, 1, 2, 3, 4, 5, 6, 7]

# Detections whose zone is a table seat (as opposed to the shared board/pot).
_SEATED_CLASSES = {
    "card_back",
    "stack_text",
    "bet_text",
    "action_pill",
    "dealer_button",
    "active_turn_indicator",
}

# Classes whose attr is a numeric amount readable by the template OCR.
_AMOUNT_CLASSES = {"pot_text", "stack_text", "bet_text"}

# Mean grayscale crop value below which hero's own hole cards read as the
# client's greyed-out "folded" rendering rather than the lit/active one.
# Calibrated against real frames (ClubWPT session, v05#32): live cards ~172,
# folded cards ~50 -- a wide gap, so this sits with margin on both sides.
_HERO_DIM_BRIGHTNESS_THRESHOLD = 100.0


@dataclass
class Detection:
    """One YOLO region box. ``attr`` is the attribute-read payload:

    - face_card: rank+suit label (e.g. "As"), or the detector's own "AS"/"10C"
    - pot_text / stack_text: the numeric value (float or numeric string)
    - action_pill: the action or colour ("raise"/"call"/"bet"/"check"/"gray"/None)
    - card_back / dealer_button / active_turn_indicator: unused (None)

    ``brightness`` (face_card only): mean grayscale value of the card crop,
    0-255. The client renders a folded seat's own hole cards greyed-out in
    place rather than removing them, so this is a second, more durable fold
    signal than the action_pill (which flashes "FOLD" for under a second --
    easy to miss at a few-second sampling rate -- while the greyed cards stay
    that way for the rest of the hand). None when not computed (fixture/test
    paths that never had pixel access).
    """

    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]
    attr: Any = None
    brightness: float | None = None
    # Amount-read provenance (pot_text / stack_text / bet_text only), following the
    # ``brightness`` precedent: filled wherever there is pixel access, None on the
    # fixture path. ``attr_source`` is ocr_readers.AmountRead.decimal_source, a
    # CLOSED vocabulary with two disjoint halves: DECIMAL_EVIDENCE ("dot" /
    # "integer") always accompanies a value, REFUSAL_CODES always accompanies
    # ``attr is None``.
    #
    # THE THREE-WAY DISTINCTION THIS FIELD CARRIES, which every consumer must
    # keep apart (use ``amount_state`` rather than reading the pair by hand):
    #   attr is not None                     -> a PROVEN value
    #   attr is None, attr_source is not None -> the reader RAN and REFUSED.
    #                                            UNKNOWN: not zero, not absent
    #   attr is None, attr_source is None     -> no read was attempted (fixture
    #                                            path, or no box at all): ABSENT
    # The channel already existed and was unread; ``frame_from_models`` now sets
    # it on EVERY amount detection, including when no template bank is calibrated
    # ("reader_unavailable"), so the middle case is never silently merged into the
    # last one.
    attr_source: str | None = None
    attr_score: float | None = None


@dataclass
class Frame:
    """One sampled frame's detections plus the geometry needed to normalize them."""

    image: str
    time_s: float
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    video_frame: int = 0


# --------------------------------------------------------------------------- #
# Attribute readers (STUBS). Real OCR / rank-suit / pill-colour classifiers
# replace these behind the same signatures.
# --------------------------------------------------------------------------- #
def read_card_label(det: Detection) -> str | None:
    """face_card attr -> canonical rank+suit ("As"), or None if unreadable."""
    try:
        return normalize_card_label("face_card", det.attr)
    except ValueError:
        return None


# Refusal codes raised by THIS layer rather than by the reader. They join
# ocr_readers.REFUSAL_CODES in the same namespace; the tokens are duplicated here
# as literals for the same reason PILL_BET_OR_CALL is (region_detections is the
# contract boundary and must stay importable without cv2/numpy).
AMOUNT_UNREADABLE_ATTR = "unparseable_amount"      # attr present but not a number
AMOUNT_STACK_BOXES_DISAGREE = "stack_boxes_disagree"  # two stack_text boxes, one seat
AMOUNT_BET_BOXES_DISAGREE = "bet_boxes_disagree"      # two bet_text boxes, one seat
AMOUNT_POT_ZERO_IMPOSSIBLE = "pot_zero_impossible"    # see _frame_state / note 10
AMOUNT_READER_UNAVAILABLE = "reader_unavailable"      # no template bank calibrated
# An amount box carrying no value and no reason -- the fixture path, where no read
# was attempted at all. It is counted as unknown (the hand still has no number for
# that box) but it is NOT a positive refusal, so it never enters ``stack_unknown``
# / ``bet_unknown`` and the whole synthetic suite stays inert w.r.t. the
# refusal-propagation rules below.
AMOUNT_UNSPECIFIED = "unspecified"

# ``amount_state`` verdicts.
AMOUNT_VALUE = "value"
AMOUNT_UNKNOWN = "unknown"
AMOUNT_ABSENT = "absent"


def amount_state(det: Detection) -> tuple[str, Any]:
    """The THREE-WAY amount verdict for one detection. Use this, not ``read_amount``.

    Returns one of:
      ``("value", float)``   -- the reader proved a number (0.0 included: an
                                all-in seat genuinely shows "0 BB")
      ``("unknown", code)``  -- the reader RAN and REFUSED; ``code`` names which
                                condition failed. UNKNOWN is a first-class value:
                                it is not 0.0 and it is not "no box here"
      ``("absent", "")``     -- no read was attempted (the fixture path, or a
                                class that carries no amount)

    ``read_amount`` collapses the first two, which is precisely the conflation
    PLAN.md's "treat absent amounts as unknown, not zero" forbids downstream.
    """
    if det.attr is not None:
        try:
            return AMOUNT_VALUE, float(str(det.attr).replace(",", "").strip())
        except (TypeError, ValueError):
            return AMOUNT_UNKNOWN, AMOUNT_UNREADABLE_ATTR
    if det.attr_source:
        return AMOUNT_UNKNOWN, det.attr_source
    return AMOUNT_ABSENT, ""


def read_amount(det: Detection) -> float | None:
    """pot_text / stack_text attr -> float value, or None.

    THE VALUE-ONLY PROJECTION of ``amount_state``, and it must not be used to
    DECIDE anything: its None means "unknown OR absent OR unreadable", and a
    consumer that branches on it has already lost the distinction. Kept for the
    call sites that genuinely only want the number.
    """
    kind, payload = amount_state(det)
    return payload if kind == AMOUNT_VALUE else None


# Re-exported so the spine can name the ambiguity without importing the OCR layer
# (region_detections is the contract boundary; ocr_readers needs cv2/numpy).
PILL_BET_OR_CALL = "bet_or_call"

_PILL_ALIASES = {
    "raise": "raise",
    "bet": "bet",
    "call": "call",
    "check": "check",
    "fold": "fold",
    "all-in": "all-in",
    "all_in": "all-in",
    "allin": "all-in",
    "post_blind": "post_blind",
    "bb": "post_blind",
    "sb": "post_blind",
}


def read_pill_action(det: Detection, *, dealt_in: bool) -> str | None:
    """action_pill attr -> action type. A gray pill is check when the seat still
    has cards, fold otherwise (README: "gray check-vs-fold is resolved by card
    backs disappearing"). Returns None when the pill carries no readable action.

    A GREEN pill whose word is unreadable resolves to ``PILL_BET_OR_CALL``, not to
    "call". The client paints CALL and BET on the same green, so the colour cannot
    choose between them, and "call is the safe default" was not a default at all --
    it is a positive claim that somebody had already bet on that street. Measured
    on the 5 development geometries: 964 of 2439 action_pill reads (39.5%) fall
    back to colour because the word template misses, 412 of them green. On the
    baseline recording that forced "call" opened the turn with no prior
    aggression; the spine's fold handler discards folds on a street it believes
    has no bet, so the villain's observed FOLD was dropped and a CHECK was
    synthesised in its place -- exported with warnings=[] at confidence 0.8.

    The ambiguity is resolvable, but by STRUCTURE rather than by colour: on a
    street where nobody has bet yet, a green pill can only be a bet. That
    resolution belongs to the spine, which knows the betting history; this layer's
    job is to stop asserting what it cannot see.
    """
    if det.attr is None:
        return None
    text = str(det.attr).strip().lower()
    if text in _PILL_ALIASES:
        return _PILL_ALIASES[text]
    # Colour fallback emitted by the OCR reader when the pill word is unreadable:
    # gray = check/fold (resolved by whether the seat still holds cards), green =
    # call OR bet (unresolved here), orange = raise.
    if text in {"gray", "grey", "neutral"}:
        return "check" if dealt_in else "fold"
    if text in {"green", PILL_BET_OR_CALL}:
        return PILL_BET_OR_CALL
    if text == "orange":
        return "raise"
    return None


# --------------------------------------------------------------------------- #
# Seat / zone assignment (coarse STUB geometry).
# --------------------------------------------------------------------------- #
def _center(det: Detection, frame: Frame) -> tuple[float, float]:
    x0, y0, x1, y1 = det.xyxy
    return ((x0 + x1) / 2.0 / frame.width, (y0 + y1) / 2.0 / frame.height)


def _nearest_seat(cx: float, cy: float, cls: str = "") -> int:
    anchors = SEAT_ANCHORS_BY_CLASS.get(cls, SEAT_CENTROIDS)
    return min(
        anchors,
        key=lambda s: (cx - anchors[s][0]) ** 2 + (cy - anchors[s][1]) ** 2,
    )


def _center_px(det: Detection) -> tuple[float, float]:
    """Box center in FRAME PIXELS (``_center`` is the normalized twin)."""
    x0, y0, x1, y1 = det.xyxy
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# Reference constellation for the detector-side anchor: the same seat table
# _nearest_seat uses, expressed in reference pixels. ONE table, two consumers.
# stack_text is the anchor class because it is the most reliably present: 8 of 8
# points on 358/361 frames (g0723a), 358/361 (g0621), 195/197 (g0723b), and 0 of
# 1309 card-bearing frames across the 5 development geometries carried fewer than
# 3 -- so failing closed on a missing anchor costs nothing on measured material.
_REF_STACK_PTS = [(x * REF_DET_W, y * REF_DET_H)
                  for x, y in SEAT_ANCHORS_BY_CLASS["stack_text"].values()]


def anchor_for_frame(frame: Frame) -> TableAnchor | None:
    """Fit the table transform from this frame's own ``stack_text`` boxes.

    Returns None when too few landmarks were detected, when the fit rests on too
    few of them to be believed (ANCHOR_MIN_TRUSTED_POINTS -- the residual is a
    shape test and shape needs redundancy), or when the fit's shape is implausible
    (median reprojection error above ANCHOR_MAX_RESID of the fitted table width).
    A wrong transform is worse than no transform, because every card zone
    downstream is derived from it.
    """
    pts = [_center_px(d) for d in frame.detections if d.cls == "stack_text"]
    fit = anchor_from_points(pts, _REF_STACK_PTS)
    if fit is None or fit.resid > ANCHOR_MAX_RESID:
        return None
    if fit.n_points < ANCHOR_MIN_TRUSTED_POINTS:
        return None
    return fit


# Board-row detector (rejection signal A). A community row is a set of cards
# sharing a reference-y within _BOARD_ROW_RY_TOL whose span is board-SHAPED.
#
# The shape test is expressed in units of the row's OWN median card width, not in
# reference-x units. That matters: reference coordinates are the frame divided by
# the fitted anchor scale, so a wrong scale inflates or deflates a reference span
# straight out of any fixed window -- and disables the net at exactly the moment
# the anchor is worst. Measured, a real 5-card board spans 0.2447 of reference-x,
# so the old [0.08, 0.30] window switched itself off below 0.816x true scale; at
# 0.80x every board card on a whole recording was lost with the net silent on all
# 39 frames and a mathematically perfect residual. A ratio of two quantities that
# both carry the same scale factor cannot be moved by that factor at all.
#
# Measured over 562 community rows on the 5 development geometries: 3-card rows
# span 3.45-4.58 card widths, 4-card 5.79-6.95, 5-card 7.49-8.88. The only
# non-board same-row card strip anywhere (the g0723a mid-deal top strip, 4 cards)
# spans 26.99-27.21. The window below clears the widest board by 3.1 and sits
# 15.0 below the confuser.
#
# ROW TOLERANCE. This bounds how far a card may sit from its row-mates in
# reference-y and still BE one, and it is what decides whether the partial net
# can see a card the zone test rejected. At 0.02 it could not: the board band is
# 0.061 tall, so a card leaving it through the top or bottom edge is 0.030-0.042
# from row-mates sitting where real rows sit (measured reference-y 0.4379-0.4503
# on all five geometries), the zone test therefore REMOVED the lost card from the
# row, and the survivors were a complete, board-shaped, fully-zoned row -- so
# `0 < zoned < len(row)` was False by construction on 557 of 562 real rows.
#
# Re-derived from two independent measurements over those 562 rows:
#   * the tolerance REQUIRED so that a card exiting either band edge still groups
#     with every row-mate is at most 0.0423 (p50 0.0316);
#   * the CLOSEST any non-board card comes to a board row's edge is 0.0621
#     (n=1227; p1 0.0727).
# 0.05 sits inside that window at 1.18x the requirement and 1.24x under the
# nearest confuser. It is also 3.9x the largest observed WITHIN-row scatter
# (0.0129; p50 0.0014, p99 0.0032), so no real row can split itself.
_BOARD_ROW_RY_TOL = 0.05
_BOARD_ROW_MIN_CARDS = 3
_BOARD_ROW_SPAN_CARD_WIDTHS = (2.5, 12.0)

# Largest centre-to-centre gap, in card widths, between adjacent cards of one
# row. Same-y is not the same fact as same-row: two villains' revealed pairs sit
# on one line at opposite ends of the table. Measured over the 562 rows, adjacent
# board cards sit 1.71-2.32 card widths apart (p50 2.02); a board with a middle
# card undetected would read about 4.6. The g0723a showdown strip's two reveal
# pairs are 26 card widths apart. 5.0 clears the widest real board rhythm by 2.2x
# and sits 5.2x under the confuser.
_BOARD_ROW_MAX_ADJACENT_GAP = 5.0

_Card = tuple[float, float, str, float]


def _median_card_width(row: list[_Card]) -> float:
    widths = sorted(c[3] for c in row)
    return widths[len(widths) // 2]


def _rx_runs(group: list[_Card]) -> list[list[_Card]]:
    """Split a same-y card group into horizontally CONTIGUOUS runs."""
    card_w = _median_card_width(group)
    if card_w <= 0:
        return []
    ordered = sorted(group, key=lambda c: c[0])
    runs: list[list[_Card]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur[0] - prev[0] > _BOARD_ROW_MAX_ADJACENT_GAP * card_w:
            runs.append([])
        runs[-1].append(cur)
    return runs


def _is_board_shaped(row: list[_Card]) -> bool:
    if len(row) < _BOARD_ROW_MIN_CARDS:
        return False
    card_w = _median_card_width(row)
    if card_w <= 0:
        return False
    ratio = (max(c[0] for c in row) - min(c[0] for c in row)) / card_w
    return _BOARD_ROW_SPAN_CARD_WIDTHS[0] <= ratio <= _BOARD_ROW_SPAN_CARD_WIDTHS[1]


def _community_rows(card_ref: list[_Card]) -> list[list[_Card]]:
    """EVERY board-shaped card row on screen, not just the largest one.

    Testing only the largest same-y set was a silent switch-off, and it fired
    under the CORRECT anchor: on the real g0723a t=265 showdown frame a four-card
    villain-reveal strip outnumbered the three-card flop, failed the span test,
    and the flop was then never examined at all -- both board nets returned False
    with the whole community row sitting on screen. Ties were broken by detection
    iteration order, so which row got tested was not even deterministic.
    """
    if len(card_ref) < _BOARD_ROW_MIN_CARDS:
        return []
    rows: list[list[_Card]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for _rx, ry, _zone, _w in card_ref:
        group = [c for c in card_ref if abs(c[1] - ry) <= _BOARD_ROW_RY_TOL]
        if len(group) < _BOARD_ROW_MIN_CARDS:
            continue
        for run in _rx_runs(group):
            key = tuple(sorted((c[0], c[1]) for c in run))
            if key in seen:
                continue
            seen.add(key)
            if _is_board_shaped(run):
                rows.append(run)
    return rows


def _community_row(card_ref: list[_Card]) -> list[_Card]:
    """The LARGEST board-shaped card row on screen, or [].

    The single-row view of ``_community_rows``, kept for the hard-example miner,
    which zones cards from the constellation alone and needs one board row rather
    than every candidate. The nets themselves must never use it: reducing the
    candidates to one is the defect ``_community_rows`` exists to fix.
    """
    rows = _community_rows(card_ref)
    return max(rows, key=len) if rows else []


def _board_row_missed(card_ref: list[_Card]) -> bool:
    """True when a board-shaped community row is on screen and NONE of it was
    zoned as board.

    This is the permanent net under the anchor: the failure it catches is silent
    by construction, because an empty board is a legal value everywhere
    downstream. It fired on 104 of 352 card-bearing frames of the AR 1.750
    recording whose every board the old hardcoded zones destroyed, and on 0 of
    the 872 card-bearing frames of the four healthy geometries -- with the anchor
    in place it is silent on all five.
    """
    return any(all(c[2] != "board" for c in row) for row in _community_rows(card_ref))


def _board_row_partial(card_ref: list[_Card]) -> bool:
    """True when a community row is on screen and only SOME of it was zoned board.

    All-or-nothing was not the failure mode. A card zone is a hard rectangle, so a
    row that straddles a band edge yields most of its cards and drops the rest --
    and every surviving count (3, 4, 5) is legal downstream. Measured: a 5-card
    river board whose last card sat 0.0001 of reference-y below the band edge
    exported as a completed FOUR-card turn board, at confidence 1.0, with no
    warning, asserting a showdown result on a board missing its river card.

    See _BOARD_ROW_RY_TOL: this net is only as wide as the row grouping under it,
    and the grouping used to be narrower than the band the zone test applies.
    """
    for row in _community_rows(card_ref):
        zoned = sum(1 for c in row if c[2] == "board")
        if 0 < zoned < len(row):
            return True
    return False


def assign_regions(frame: Frame, *, anchor: TableAnchor | None = None) -> dict[str, Any]:
    """Assign a frame's detections to hero/board zones and table seats.

    Returns a normalized table view:
        {
          "hero": [card,...], "board": [card,...],
          "villain_cards": { seat_index: [card,...] },  # non-hero face_card boxes,
              # i.e. cards a villain flipped face-up (showdown reveal) -- only
              # present in the rare frames where the client actually shows them.
          "hero_dim": bool,  # hero's own cards are rendered greyed-out (folded)
          "pot": float | None,        # the MAIN pot, unchanged meaning
          "side_pot": float | None,   # a second pot_text below the main pot row
          "pots": [float, ...],       # every pot read, main first
          "pot_unknown": str | None,       # refusal code when the pot was REFUSED
          "side_pot_unknown": str | None,  # ... same for the side pot
          "stack_unknown": { seat_index: code },  # REFUSED stack reads only
          "bet_unknown": { seat_index: code },    # REFUSED bet reads only
          "amounts_unknown_by_code": { code: count },
          "board_row_missed": bool,   # a community row was on screen, none zoned board
          "board_row_partial": bool,  # ... and only some of it was zoned board
          "seats": { seat_index: {"card_back","stack","pill_action","dealer","turn"} },
          "dealer_seat": int | None, "active_seat": int | None,
          "anchor_ok": bool, "anchor_resid": float | None,
          "anchor_source": str | None, "unanchored_cards": int,
        }

    ``anchor`` is the table transform to zone cards through. When omitted it is
    fitted from this frame alone. When no anchor is available at all the frame
    FAILS CLOSED: every face_card is dropped (counted in ``unanchored_cards``)
    and hero/board/villain_cards come back empty. There is deliberately no
    fallback to a fixed normalized window -- that fallback is the defect this
    replaced, and it silently mis-zoned every board card at aspect ratio 1.750.
    """
    anchor = anchor or anchor_for_frame(frame)
    unanchored_cards = 0
    amounts_unknown = 0
    # Per-refusal-code tally of every UNKNOWN amount on this frame. A bare count
    # cannot tell a systematic reader failure on one region from scattered
    # occlusions, and it is the count alone that reached the export before.
    amounts_unknown_by_code: dict[str, int] = {}

    def note_unknown(code: str) -> None:
        nonlocal amounts_unknown
        amounts_unknown += 1
        amounts_unknown_by_code[code] = amounts_unknown_by_code.get(code, 0) + 1
    # (rx, ry, zone, reference width) for every anchored face_card -- the input to
    # the board-row nets. The width is what makes their shape test scale-free.
    card_ref: list[tuple[float, float, str, float]] = []
    stack_sources: dict[int, str | None] = {}
    hero_dets: list[tuple[float, Detection]] = []
    board_dets: list[tuple[float, Detection]] = []
    # face_card boxes outside the hero/board zones -- a villain's cards flipped
    # face-up at showdown -- keyed by nearest seat (same anchors as card_back,
    # so a seat's revealed cards line up with where its card_back sat).
    villain_dets: dict[int, list[tuple[float, Detection]]] = {}
    pot_candidates: list[Detection] = []
    pot_off_column = 0
    seat_conflicts = {"stack_text": 0, "bet_text": 0}
    # (class, seat) -> [(value, refusal code, box confidence, attr source), ...]
    seen_seat_amounts: dict[tuple[str, int], list[tuple[float | None, str | None,
                                                        float, str | None]]] = {}
    seats: dict[int, dict[str, Any]] = {}
    # Nearest card-anchor seat for each card in the STRICT hero zone ("other"-zone
    # strays like villain showdown reveals don't vote). The spine's hero identity
    # is the convention hero zone == seat 0; these votes cross-check it.
    hero_zone_seat_votes: list[int] = []

    def seat(i: int) -> dict[str, Any]:
        return seats.setdefault(
            i, {"card_back": False, "stack": None, "stack_unknown": None,
                "bet": None, "bet_unknown": None, "pill_action": None,
                "dealer": False, "turn": False}
        )

    def resolve_seat_amounts() -> None:
        """Resolve every seated money box into one reading per (class, seat).

        ONE RULE FOR BOTH SEATED MONEY CHANNELS, because there is one problem.
        Two boxes of the same class can resolve to one seat, and until this was
        shared the two channels handled that differently: stack_text called it a
        conflict and published UNKNOWN, while bet_text was a bare
        `seat(i)["bet"] = value` -- last-write-wins, decided by the order the
        detector happened to list its boxes.

        Measured on the six development recordings, 65 frames carry a bet_text
        seat conflict (g0621 59, g0723a 3, g0711 2, g0723b 1) and in ALL 65 the
        winner under list order is NOT the highest-confidence box. Both failure
        directions were real and both reached the ledger: at g0723b t=0 a
        conf-0.49 box containing only a chip sprite (refused: no_digit_run)
        overwrote a conf-0.942 box reading a plainly legible "24 BB", flipping
        hand 1's `preflop/0/raise` between amount=24.0 and amount=None -- i.e.
        toggling the input to the fatal `amounts_unknown_in_ledger` code; and at
        g0621 t=116 two different seats' bets collided on one seat and published
        a confident 0.5 over a legible "15 BB" with no counter and no code.

        Two contradictory readings of one seat are a conflict, and PLAN.md is
        explicit that an absent amount is unknown, not a guess. A positive
        REFUSAL is one of the contradicting readings: it must not be erased by a
        second box, and it must not erase a legible one either -- which is why
        `None` participates in the comparison rather than being skipped.

        RESOLVING AFTER the detection loop rather than inside it is what makes
        the result a function of the box SET instead of its order. Writing each
        box as it arrived meant the seat carried an interim answer, `amounts
        unknown` was tallied once per box against a growing history, and a
        three-box seat produced a different count in each direction (measured:
        3 of g0723a's 361 frames).
        """
        for (cls, i), boxes in sorted(seen_seat_amounts.items()):
            distinct = {("none" if b[0] is None else round(b[0], 2)) for b in boxes}
            if len(distinct) > 1:
                seat_conflicts[cls] += len(boxes) - 1
                # The conflict gets its OWN code and never masks the reader's.
                # `stack_sources[i] = ... else "conflict"` used to overwrite the
                # refusal reason with the word "conflict", so a seat whose box
                # the reader had positively refused became indistinguishable
                # from a seat whose two boxes disagreed.
                value, code = None, (AMOUNT_STACK_BOXES_DISAGREE
                                     if cls == "stack_text"
                                     else AMOUNT_BET_BOXES_DISAGREE)
                source = None
            else:
                # Every box agrees. WHICH refusal reason (or which read's
                # provenance) is published was still decided by detector order
                # when a seat had more than one box -- measured at g0723b t=0
                # seat 0, `no_digit_run` as detected and
                # `unexplained_ink_in_numeral` reversed. The value is the same
                # either way; the reason is diagnostic, so it comes from the box
                # the detector was most confident about, with the payload itself
                # as a deterministic tiebreak.
                best = max(boxes, key=lambda b: (b[2], str(b[1]), str(b[3])))
                value, code, _, source = best
            if cls == "stack_text":
                seat(i)["stack"] = value
                seat(i)["stack_unknown"] = code
                stack_sources[i] = source if value is not None else None
            else:
                seat(i)["bet"] = value
                seat(i)["bet_unknown"] = code
            if value is None:
                note_unknown(code or AMOUNT_UNSPECIFIED)

    for det in frame.detections:
        if det.cls not in CLASS_SET:
            continue
        cx, cy = _center(det, frame)

        if det.cls == "face_card":
            if anchor is None:
                unanchored_cards += 1
                continue  # FAIL CLOSED: never guess a zone
            rx, ry = anchor.to_ref(*_center_px(det))
            zone = zone_for_ref(rx, ry)
            width_ref = (det.xyxy[2] - det.xyxy[0]) / anchor.s / REF_DET_W
            card_ref.append((rx, ry, zone, width_ref))
            if zone == "board":
                board_dets.append((cx, det))
            elif zone == "hero":
                hero_dets.append((cx, det))
                hero_zone_seat_votes.append(_nearest_seat(cx, cy, "card_back"))
            else:
                i = _nearest_seat(cx, cy, "card_back")
                villain_dets.setdefault(i, []).append((cx, det))
        elif det.cls == "pot_text":
            pot_candidates.append(det)
        elif det.cls in _SEATED_CLASSES:
            i = _nearest_seat(cx, cy, det.cls)
            if det.cls == "card_back":
                seat(i)["card_back"] = True
            elif det.cls in ("stack_text", "bet_text"):
                # Two boxes of one class can map to one seat (measured: 3 of 1309
                # frames on stack_text across the 5 development geometries, all 3
                # DISAGREEING, and 65 frames on bet_text). Collected here and
                # resolved once per seat below -- see resolve_seat_amounts.
                kind, payload = amount_state(det)
                seen_seat_amounts.setdefault((det.cls, i), []).append(
                    (payload if kind == AMOUNT_VALUE else None,
                     payload if kind == AMOUNT_UNKNOWN else None,
                     det.conf, det.attr_source))
                seat(i)  # the seat exists even if every box refuses
            elif det.cls == "action_pill":
                seat(i)["_pill_det"] = det  # resolved after dealt-in is known
            elif det.cls == "dealer_button":
                seat(i)["dealer"] = True
            elif det.cls == "active_turn_indicator":
                seat(i)["turn"] = True

    resolve_seat_amounts()

    hero_cards = [c for c in (read_card_label(d) for _, d in sorted(hero_dets)) if c][:2]
    board_cards = [c for c in (read_card_label(d) for _, d in sorted(board_dets)) if c][:5]

    # Hero fold, second signal: the client greys out hero's own hole cards in
    # place on fold rather than removing them (mean crop brightness ~50/255,
    # vs. ~170+ while still live) -- and unlike the "FOLD" action_pill, which
    # flashes for well under a second, the greyed-out cards stay that way for
    # the rest of the hand, so a sparse sampling rate can still catch it.
    hero_brightness_vals = [d.brightness for _, d in hero_dets if d.brightness is not None]
    hero_dim = bool(hero_brightness_vals) and (
        sum(hero_brightness_vals) / len(hero_brightness_vals) < _HERO_DIM_BRIGHTNESS_THRESHOLD
    )
    villain_cards: dict[int, list[str]] = {}
    for seat_i, dets in villain_dets.items():
        cards = [c for c in (read_card_label(d) for _, d in sorted(dets)) if c][:2]
        if cards:
            villain_cards[seat_i] = cards

    # Hero seat is dealt in when hero hole cards are visible.
    if hero_cards:
        seat(0)["card_back"] = True

    # Resolve pills now that dealt-in status per seat is known.
    for _i, info in seats.items():
        pill_det = info.pop("_pill_det", None)
        if pill_det is not None:
            info["pill_action"] = read_pill_action(pill_det, dealt_in=info["card_back"])

    # Pot: the main pot renders in a band well above any side pot (reference ry
    # 0.3410-0.3810 over 1258 single-pot frames on all 5 geometries; the only
    # observed side pot renders at 0.5419-0.5460). Splitting on the anchored row
    # keeps the main pot's meaning EXACTLY as before -- max-confidence among the
    # main-band candidates -- while stopping a detected side pot from silently
    # replacing it. max(conf) alone is not even a consistent selector when both
    # are present: on the one recording that has a side pot it picks the main at
    # t=8.0 (0.921 vs 0.421) and the SIDE at t=6.0 (0.758 vs 0.558).
    pot = None
    side_pot = None
    pot_unknown: str | None = None
    side_pot_unknown: str | None = None

    def _read_pot(dets: list[Detection]) -> tuple[float | None, str | None]:
        """(value, refusal code) for the highest-confidence box of one pot band.

        A REFUSED pot is not the same fact as "no pot box was detected", and the
        two used to be one None. It must also never fall through to the other
        band's slot: main and side are read from their own candidates only.
        """
        kind, payload = amount_state(max(dets, key=lambda d: d.conf))
        if kind == AMOUNT_VALUE:
            return payload, None
        return None, (payload if kind == AMOUNT_UNKNOWN else AMOUNT_UNSPECIFIED)

    if pot_candidates and anchor is not None:
        rows = [(anchor.to_ref(*_center_px(det)), det) for det in pot_candidates]
        # Both pots live in the table's centre column. A pot_text box outside it
        # is a misclassified HUD element, not a pot, and is dropped rather than
        # allowed to compete for either slot.
        kept = [(rxy[1], det) for rxy, det in rows
                if REF_POT_RX[0] <= rxy[0] <= REF_POT_RX[1]]
        # A dropped pot box is not the same fact as "no pot was detected", and it
        # used to be indistinguishable: pot came back None with every other field
        # unchanged, so downstream saw FEWER signals than a mis-reconciled pot
        # (pot_not_reconciled is itself gated on final_pot is not None). Counted
        # now. The reject branch is unexercised on the development corpus -- 0 of
        # 1267 pot_text boxes sit off-column -- so this counter is the only way a
        # future geometry that does trip it can be told apart from a missing pot.
        pot_off_column = len(rows) - len(kept)
        rows = kept
        mains = [det for ry, det in rows if ry < REF_POT_SIDE_RY_MIN]
        sides = [det for ry, det in rows if ry >= REF_POT_SIDE_RY_MIN]
        if mains:
            pot, pot_unknown = _read_pot(mains)
        if sides:
            side_pot, side_pot_unknown = _read_pot(sides)
    elif pot_candidates:
        # Unanchored: the main/side row test is unavailable, so report the main
        # pot only rather than guess which band a box came from.
        pot, pot_unknown = _read_pot(pot_candidates)
    for d in pot_candidates:
        kind, payload = amount_state(d)
        if kind != AMOUNT_VALUE:
            note_unknown(payload if kind == AMOUNT_UNKNOWN else AMOUNT_UNSPECIFIED)
    pots = [v for v in (pot, side_pot) if v is not None]

    dealer_seat = next((i for i, info in seats.items() if info["dealer"]), None)
    active_seat = next((i for i, info in seats.items() if info["turn"]), None)

    # Hero-seat cross-check: if most hero-zone cards sit nearer another seat's
    # card anchor than seat 0's, the layout/anchors have drifted and every
    # downstream "hero = seat 0" attribution (is_hero, hero_position, hero net)
    # is suspect. Majority vote so one flapped assignment doesn't warn.
    # ... and it is only "confirmed" when there IS evidence. With an empty vote
    # list the majority test evaluates `0 > 0` and reports "hero seat confirmed"
    # on the strength of nothing -- which is precisely the state a layout drift
    # produces, because the drift is what empties the hero zone. Measured: a 1.24x
    # vertical stretch pushes hero's own cards 0.0021 of reference-y past the hero
    # band, they are re-attributed to two different villain seats as showdown
    # reveals, the board still zones 5/5, the anchor residual stays inside
    # tolerance -- and hero_seat_mismatch stayed False throughout.
    off_seat = sum(1 for v in hero_zone_seat_votes if v != 0)
    hero_seat_mismatch = off_seat * 2 > len(hero_zone_seat_votes)
    hero_seat_confirmed = bool(hero_zone_seat_votes) and not hero_seat_mismatch

    return {
        "hero": hero_cards,
        "board": board_cards,
        "villain_cards": villain_cards,
        "hero_dim": hero_dim,
        "pot": pot,
        "side_pot": side_pot,
        "pots": pots,
        "seats": seats,
        "stack_sources": stack_sources,
        # UNKNOWN, per slot, with the reason. Present iff the read was REFUSED --
        # an absent box leaves the slot out entirely, so a consumer can tell
        # "unreadable" from "not there" without re-deriving it.
        "pot_unknown": pot_unknown,
        "side_pot_unknown": side_pot_unknown,
        "stack_unknown": {i: info["stack_unknown"] for i, info in seats.items()
                          if info.get("stack_unknown")},
        "bet_unknown": {i: info["bet_unknown"] for i, info in seats.items()
                        if info.get("bet_unknown")},
        "amounts_unknown": amounts_unknown,
        "amounts_unknown_by_code": amounts_unknown_by_code,
        "board_row_missed": _board_row_missed(card_ref),
        "board_row_partial": _board_row_partial(card_ref),
        "dealer_seat": dealer_seat,
        "active_seat": active_seat,
        "hero_seat_mismatch": hero_seat_mismatch,
        "hero_seat_confirmed": hero_seat_confirmed,
        "hero_seat_votes": len(hero_zone_seat_votes),
        "pot_text_off_column": pot_off_column,
        "stack_conflicts": seat_conflicts["stack_text"],
        "bet_conflicts": seat_conflicts["bet_text"],
        "anchor_ok": anchor is not None,
        "anchor_resid": None if anchor is None else round(anchor.resid, 5),
        "anchor_source": None if anchor is None else anchor.source,
        "unanchored_cards": unanchored_cards,
    }


# --------------------------------------------------------------------------- #
# Adapters: produce Frame objects from each source.
# --------------------------------------------------------------------------- #
def frames_from_fixture(data: Iterable[dict[str, Any]]) -> list[Frame]:
    """Parse a list of plain dicts (JSON fixtures / tests) into Frames."""
    frames: list[Frame] = []
    for row in data:
        dets = [
            Detection(
                cls=str(d["cls"]),
                conf=float(d.get("conf", 1.0)),
                xyxy=tuple(float(v) for v in d["xyxy"]),  # type: ignore[arg-type]
                attr=d.get("attr"),
                brightness=d.get("brightness"),
                # Absent on every fixture written before the read-quality channel
                # existed, which is what keeps the plausibility rules gated on
                # attr_source inert across the whole synthetic suite. A fixture
                # may still set it explicitly to exercise those rules.
                attr_source=d.get("attr_source"),
                attr_score=d.get("attr_score"),
            )
            for d in row.get("detections", [])
        ]
        frames.append(
            Frame(
                image=str(row["image"]),
                time_s=float(row["time_s"]),
                width=int(row["width"]),
                height=int(row["height"]),
                detections=dets,
                video_frame=int(row.get("video_frame", 0)),
            )
        )
    return frames


def load_frames(path: str | Path) -> list[Frame]:
    """Load a frames fixture JSON file (a list of frame dicts)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("frames", [])
    return frames_from_fixture(raw)


def frame_from_yolo_rows(
    image: str,
    time_s: float,
    width: int,
    height: int,
    rows: Iterable[dict[str, Any]],
    *,
    video_frame: int = 0,
) -> Frame:
    """Adapt one image's YOLO detection rows (cv_lab.labeling_poker.inference.predict_cards
    shape: class/label/confidence/x1..y2) into a Frame. Downstream wiring for the
    trained 7-class detector; unlike the old card model, ``class`` is trusted here."""
    dets: list[Detection] = []
    for row in rows:
        cls = str(row.get("class", "")).strip()
        if cls not in CLASS_SET:
            continue
        attr = row.get("label")
        if cls in {"pot_text", "stack_text", "action_pill"}:
            attr = row.get("attr", row.get("value"))
        dets.append(
            Detection(
                cls=cls,
                conf=float(row.get("confidence", row.get("conf", 0.0)) or 0.0),
                xyxy=(
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ),
                attr=attr,
            )
        )
    return Frame(image=image, time_s=time_s, width=width, height=height, detections=dets, video_frame=video_frame)


def _pad_crop_xyxy(image, x0: float, y0: float, x1: float, y1: float, pad: float):
    """Crop [x0,y0,x1,y1] (pixels) expanded by ``pad`` each side, clamped to image.

    Returns the sub-array, or None if degenerate/out of bounds. Uses plain array
    slicing so region_detections stays importable without cv2/numpy on the
    fixture path.
    """
    if image is None:
        return None
    h, w = image.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * pad
    x1 += bw * pad
    y0 -= bh * pad
    y1 += bh * pad
    xi0, yi0 = max(int(round(x0)), 0), max(int(round(y0)), 0)
    xi1, yi1 = min(int(round(x1)), w), min(int(round(y1)), h)
    if xi1 <= xi0 or yi1 <= yi0:
        return None
    return image[yi0:yi1, xi0:xi1]


def frame_from_models(
    image,
    time_s: float,
    rows: Iterable[dict[str, Any]],
    *,
    classifier,
    image_name: str = "",
    pad: float = 0.12,
    video_frame: int = 0,
    ocr: bool = True,
) -> Frame:
    """Build a Frame from Model 1's region detections + Model 2's card classifier.

    This is the Design-A wiring: the region detector (Model 1) only *localizes*
    ``face_card`` boxes -- it does not name them. For each ``face_card`` box we
    crop the region out of ``image`` and hand it to ``classifier`` (Model 2,
    duck-typed: any object with ``classify(bgr_crop) -> (label, conf)``) to get
    the rank+suit, which becomes the detection's ``attr``. Amount classes
    (pot_text / stack_text / bet_text) and action_pill are read by the
    deterministic template OCR (ocr_readers); pass ``ocr=False`` to skip it
    (attrs then stay whatever the row provides, as on the fixture path).

    ``rows`` are Model 1 detection dicts with keys: class, confidence|conf,
    x1, y1, x2, y2. ``image`` is an HxWx3 BGR array (Model 1's input frame);
    width/height are taken from it.
    """
    h, w = (int(image.shape[0]), int(image.shape[1])) if image is not None else (0, 0)
    ocr_readers = None
    if ocr and image is not None:
        # Lazy import keeps the fixture path importable without cv2/numpy.
        from cv_lab.scripts.pipeline import ocr_readers as _ocr

        ocr_readers = _ocr
    dets: list[Detection] = []
    for row in rows:
        cls = str(row.get("class", "")).strip()
        if cls not in CLASS_SET:
            continue
        xyxy = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
        attr = row.get("attr")
        brightness = None
        attr_source: str | None = None
        attr_score: float | None = None
        if cls == "face_card" and classifier is not None:
            crop = _pad_crop_xyxy(image, *xyxy, pad)
            if crop is not None and getattr(crop, "size", 1) > 0:
                label, _conf = classifier.classify(crop)
                attr = label
            # Brightness is read off the TIGHT (unpadded) box -- the folded/
            # active grey-vs-white contrast is on the card face itself, and
            # padding pulls in felt background that would dilute it.
            tight = _pad_crop_xyxy(image, *xyxy, 0.0)
            if tight is not None and getattr(tight, "size", 1) > 0:
                import cv2

                brightness = float(cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY).mean())
        elif ocr_readers is not None and attr is None and cls in _AMOUNT_CLASSES:
            # attr_source is set UNCONDITIONALLY on this path, which is what makes
            # "the reader refused" distinguishable from "no read was attempted"
            # everywhere downstream. It used to be set only inside the
            # `detail is not None` branch, so a run with no calibrated template
            # bank produced attr=None / attr_source=None -- byte-identical to a
            # fixture detection carrying no amount at all.
            detail = ocr_readers.read_amount_detail_from_image(image, xyxy)
            if detail is None:
                attr_source = AMOUNT_READER_UNAVAILABLE
            else:
                attr = detail.value
                attr_source = detail.decimal_source
                attr_score = detail.score
        elif ocr_readers is not None and attr is None and cls == "action_pill":
            attr = ocr_readers.read_pill_attr(image, xyxy)
        dets.append(
            Detection(
                cls=cls,
                conf=float(row.get("confidence", row.get("conf", 0.0)) or 0.0),
                xyxy=xyxy,
                attr=attr,
                brightness=brightness,
                attr_source=attr_source,
                attr_score=attr_score,
            )
        )
    return Frame(image=image_name, time_s=time_s, width=w, height=h,
                 detections=dets, video_frame=video_frame)
