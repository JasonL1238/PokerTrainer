"""Tests for the detector-side table anchor (landmark_anchor).

The reconstruction spine derives every card zone from this similarity fit, so a
silent drift here re-creates the defect it replaced: on a 1.750-aspect recording
the hardcoded normalized board window rejected 100% of the community row while
the detector and card classifier were both correct.
"""
from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from cv_lab.scripts.pipeline import landmark_anchor as la  # noqa: E402
from cv_lab.scripts.pipeline import region_detections as rd  # noqa: E402

REF_PTS = [(x * la.REF_DET_W, y * la.REF_DET_H)
           for x, y in rd.SEAT_ANCHORS_BY_CLASS["stack_text"].values()]


def _transform(pts, s, tx, ty):
    return [(s * x + tx, s * y + ty) for x, y in pts]


def test_anchor_from_points_recovers_scale_and_translation():
    a = la.anchor_from_points(_transform(REF_PTS, 1.37, 120.0, -260.0), REF_PTS)
    assert a is not None
    assert abs(a.s - 1.37) / 1.37 < 0.005
    assert abs(a.tx - 120.0) < 2.0
    assert abs(a.ty - (-260.0)) < 2.0
    assert a.resid < 1e-6
    assert a.source == "frame"
    assert a.n_points == 8


def test_anchor_from_points_rejects_fewer_than_three():
    assert la.anchor_from_points(_transform(REF_PTS, 1.0, 0.0, 0.0)[:2], REF_PTS) is None


def test_anchor_survives_two_missing_and_one_outlier_point():
    det = _transform(REF_PTS, 1.10, 40.0, 15.0)[:6]
    det.append((det[0][0] + 400.0, det[0][1] + 400.0))  # a stray box, 400px off
    a = la.anchor_from_points(det, REF_PTS)
    assert a is not None
    assert abs(a.s - 1.10) / 1.10 < 0.01


def test_to_ref_inverts_the_fitted_transform():
    a = la.anchor_from_points(_transform(REF_PTS, 0.60, -30.0, 90.0), REF_PTS)
    assert a is not None
    px = 0.60 * (0.50 * la.REF_DET_W) - 30.0
    py = 0.60 * (0.45 * la.REF_DET_H) + 90.0
    rx, ry = a.to_ref(px, py)
    assert abs(rx - 0.50) < 1e-6
    assert abs(ry - 0.45) < 1e-6


def test_zone_for_ref_bands_match_measured_extremes():
    # Board row observed at ref ry 0.4372..0.4583 over the 5 DEVELOPMENT
    # geometries. The upper extreme is the one the previous revision of this test
    # had wrong (it recorded 0.4513), which is how a band edge at 0.466 came to
    # sit 0.0077 above the real maximum with nothing pinning it.
    assert la.zone_for_ref(0.45, 0.4372) == "board"
    assert la.zone_for_ref(0.45, 0.4583) == "board"
    # Nearest confusers: pot_text below at ry <= 0.3810, a stray reveal above at
    # 0.4811 (g0723a t=143). Both edges are their midpoints with the board.
    assert la.zone_for_ref(0.50, 0.381) == "other"
    assert la.zone_for_ref(0.50, 0.4811) == "other"
    # Hero hole cards observed at ref ry 0.6737..0.6874.
    assert la.zone_for_ref(0.49, 0.6737) == "hero"
    assert la.zone_for_ref(0.51, 0.6874) == "hero"
    # rx guards: outside the observed x span is not a card zone. Probed AT the
    # band edges, not far outside them -- 0.20/0.40 alone left the board rx band
    # free to move in either direction with the suite green, which is how
    # REF_BOARD_RX came to sit at (0.295, 0.640) instead of the +-0.030 the comment
    # beside it claims. Board rx observed 0.3497..0.6043 over 2073 anchored
    # detections; hero rx observed 0.4700..0.5135.
    assert la.zone_for_ref(0.3497, 0.45) == "board"
    assert la.zone_for_ref(0.6043, 0.45) == "board"
    assert la.zone_for_ref(0.3196, 0.45) == "other"
    assert la.zone_for_ref(0.6344, 0.45) == "other"
    assert la.zone_for_ref(0.4700, 0.68) == "hero"
    assert la.zone_for_ref(0.5135, 0.68) == "hero"
    assert la.zone_for_ref(0.20, 0.45) == "other"
    assert la.zone_for_ref(0.40, 0.68) == "other"


def test_similarity_fit_is_unchanged_by_the_pure_python_rewrite():
    """Golden values produced by the previous numpy implementation."""
    ref = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0),
           (50.0, 25.0), (25.0, 10.0), (75.0, 40.0), (10.0, 45.0)]
    det = [(11.0, 7.0), (231.5, 6.0), (230.0, 116.0), (9.0, 114.0),
           (120.0, 61.0), (65.0, 29.0), (175.0, 95.0), (32.0, 106.0)]
    s, t = la._similarity_fit(ref, det)
    assert s == 2.20141065830721
    assert t[0] == 10.124020376175537
    assert t[1] == 6.211206896551722


def test_anchor_rejects_a_grossly_wrong_aspect_ratio():
    """A 1000x1000 square frame cannot be the 1.397-aspect reference table: the
    fit residual must clear ANCHOR_MAX_RESID so the frame fails closed."""
    square = [(x * 1000.0, y * 1000.0)
              for x, y in rd.SEAT_ANCHORS_BY_CLASS["stack_text"].values()]
    a = la.anchor_from_points(square, REF_PTS)
    assert a is not None
    assert a.resid > la.ANCHOR_MAX_RESID


def test_the_residual_is_blind_to_a_rigid_offset():
    """ANCHOR_MAX_RESID cannot bound zoning error, and the code must not be
    written as if it could. A translation of the whole landmark constellation IS
    a similarity, so the fit absorbs it exactly: the residual is unchanged while
    every card's reference-y moves by the full offset. Pinned so the comment
    above the constant stays honest."""
    base = la.anchor_from_points(_transform(REF_PTS, 1.0, 0.0, 0.0), REF_PTS)
    shifted = la.anchor_from_points(_transform(REF_PTS, 1.0, 0.0, 40.0), REF_PTS)
    assert base is not None and shifted is not None
    assert abs(base.resid - shifted.resid) < 1e-12
    assert abs(shifted.ty - base.ty - 40.0) < 1e-6
    # ... and the offset lands entirely on the zoning, undetected by the gate.
    assert shifted.resid <= la.ANCHOR_MAX_RESID
    ry_base = base.to_ref(0.5 * la.REF_DET_W, 0.44 * la.REF_DET_H)[1]
    ry_shift = shifted.to_ref(0.5 * la.REF_DET_W, 0.44 * la.REF_DET_H)[1]
    assert abs(ry_base - ry_shift) > 0.02


def test_board_band_edges_sit_at_the_confuser_midpoints():
    """The band edges are DERIVED, and the derivation is the thing to pin -- the
    observed-extreme assertions above are satisfied by any ceiling at or above
    0.4583, which is how a ceiling computed from a stale maximum (0.4513) sat
    0.0077 above the real board with nothing complaining.

    Measured on the 5 development recordings: board ry 0.4372..0.4583, nearest
    confuser below pot_text at 0.3810, nearest above a stray reveal at 0.4811.
    Both edges are the midpoints, so the margin is symmetric on each side."""
    board_lo, board_hi = 0.4372, 0.4583
    confuser_below, confuser_above = 0.3810, 0.4811

    assert la.zone_for_ref(0.45, board_lo) == "board"
    assert la.zone_for_ref(0.45, board_hi) == "board"
    assert la.REF_BOARD_RY[0] == pytest.approx((confuser_below + board_lo) / 2, abs=0.002)
    assert la.REF_BOARD_RY[1] == pytest.approx((board_hi + confuser_above) / 2, abs=0.002)


def _stack_frame(points):
    """A Frame whose only detections are stack_text boxes at `points` (pixels)."""
    return rd.frames_from_fixture([{
        "image": "f.jpg", "time_s": 0.0, "width": 2000, "height": 1400,
        "detections": [
            {"cls": "stack_text", "conf": 0.9, "attr": None,
             "xyxy": [x - 40, y - 15, x + 40, y + 15]}
            for x, y in points
        ],
    }])[0]


def test_anchor_for_frame_applies_the_residual_gate():
    """The gate as APPLIED had no test. test_anchor_rejects_a_grossly_wrong_aspect
    _ratio asserts a property of anchor_from_points' return value; nothing asserted
    that anchor_for_frame acts on it, so deleting `or fit.resid > ANCHOR_MAX_RESID`
    from region_detections left the whole 886-test suite green. The gate is also
    inert on the development corpus (0 of 1299 real fits exceed 0.040), which is
    exactly why only a unit test can reach it.

    A wrong transform is worse than no transform: every card zone downstream is
    derived from it, so the frame must fail closed to the session anchor."""
    square = [(x * 1000.0, y * 1000.0)
              for x, y in rd.SEAT_ANCHORS_BY_CLASS["stack_text"].values()]
    assert la.anchor_from_points(square, REF_PTS).resid > la.ANCHOR_MAX_RESID
    assert rd.anchor_for_frame(_stack_frame(square)) is None

    # Negative control: the same constellation at the reference aspect fits, and
    # anchor_for_frame returns it.
    good = [(x * 2000.0, y * 1431.6)
            for x, y in rd.SEAT_ANCHORS_BY_CLASS["stack_text"].values()]
    fit = rd.anchor_for_frame(_stack_frame(good))
    assert fit is not None and fit.resid <= la.ANCHOR_MAX_RESID


def _row(ry_by_zone):
    """(rx, ry, zone, card_w) tuples forming one board-shaped community row."""
    return [(0.40 + 0.05 * i, ry, zone, 0.045)
            for i, (ry, zone) in enumerate(ry_by_zone)]


def test_board_row_partial_detects_a_row_that_straddles_the_band_edge():
    """The producer of board_zone_yield_partial had no test at all: stubbing
    _board_row_partial to `return False` left the whole 886-test suite green,
    because the only references anywhere set the flag by hand on synthetic states
    and exercise the CONSUMER. The sibling _community_row is covered (stubbing it
    fails two tests), so this was the one uncovered arm.

    The failure it exists for is silent by construction: 3, 4 and 5 are all legal
    board counts downstream, so a 5-card river row whose last card fell 0.0001 of
    reference-y outside the band exported as a completed FOUR-card turn board at
    confidence 1.0."""
    all_board = _row([(0.445, "board")] * 5)
    assert rd._board_row_partial(all_board) is False
    assert rd._board_row_missed(all_board) is False

    # The river card drifted past the band edge and zoned "other": the row is on
    # screen, four of its five cards are board.
    straddling = _row([(0.445, "board")] * 4 + [(0.447, "other")])
    assert rd._board_row_partial(straddling) is True
    assert rd._board_row_missed(straddling) is False

    # Whole row lost -> the zero-yield net owns it, not the partial one.
    none_board = _row([(0.445, "other")] * 5)
    assert rd._board_row_partial(none_board) is False
    assert rd._board_row_missed(none_board) is True

    # No board-shaped row on screen at all -> neither net may fire.
    assert rd._board_row_partial([]) is False
    assert rd._board_row_partial(_row([(0.445, "board"), (0.445, "other")])) is False


def test_board_row_partial_fires_at_the_MEASURED_board_row_y():
    """Round 4, adversary A: the partial net had a dead zone covering 99.1% of
    real board rows.

    ``_board_row_partial`` is defined on the row ``_community_row`` returns, and
    that row was grouped by a reference-y tolerance (0.02) SMALLER than half the
    board band's own height (0.061). A card leaving the band through its top or
    bottom edge is therefore >= 0.030 from its row-mates whenever the row sits
    where real rows sit -- measured reference-y 0.4377..0.4408 on all five
    development geometries -- so the zone test silently REMOVED the lost card from
    the row, the survivors were a complete, board-shaped, fully-zoned row, and
    `0 < zoned < len(row)` was False by construction.

    Measured dead-zone coverage before the fix: 557 of 562 real board rows.

    The invariant this pins, both sides measured over those 562 rows:
      * the tolerance must be at least 0.0423, the largest distance from a real
        board row-mate to either band edge -- below that the zone test decides row
        membership and the net is undefined;
      * it must stay under 0.0621, the closest any non-board card comes to a
        board row's edge -- above that a stray joins the row and the net
        false-positives.
    """
    assert 0.0423 <= rd._BOARD_ROW_RY_TOL <= 0.0621, (
        "row tolerance must cover the whole band-edge distance of a real board "
        "row while staying under the nearest measured non-board card"
    )
    row_ry = 0.4393                        # measured median board row y (g0711)
    river_ry = la.REF_BOARD_RY[1] + 0.0002  # 0.0002 past the upper band edge
    straddling = [(0.3588, row_ry, "board", 0.030),
                  (0.4197, row_ry, "board", 0.030),
                  (0.4805, row_ry, "board", 0.030),
                  (0.5394, row_ry, "board", 0.030),
                  (0.6001, river_ry, "other", 0.030)]
    assert rd._board_row_partial(straddling) is True
    assert rd._board_row_missed(straddling) is False


def test_board_row_nets_survive_a_larger_non_board_card_row():
    """Round 4, adversary A: a bigger same-y card set switched both nets OFF.

    ``_community_row`` kept only the LARGEST same-y set and returned [] when that
    set failed the board-shape test, so a four-card villain showdown strip (4 > a
    3-card flop) suppressed the examination of the flop entirely. Measured on the
    real g0723a t=265 frame with the correct 8-point anchor in place: both nets
    reported False while the flop was on screen.
    """
    W = 0.030
    flop_lost = [(0.3589, 0.4395, "other", W), (0.4176, 0.4403, "other", W),
                 (0.4791, 0.4392, "other", W)]
    assert rd._board_row_missed(flop_lost) is True
    # A wider, LARGER non-board strip must not suppress it.
    strip4 = [(0.10 + 0.25 * i, 0.1770, "other", W) for i in range(4)]
    assert rd._board_row_missed(flop_lost + strip4) is True

    river_partial = [(0.3588, 0.4393, "board", W), (0.4197, 0.4393, "board", W),
                     (0.4805, 0.4393, "board", W), (0.5394, 0.4393, "board", W),
                     (0.6001, 0.4394, "other", W)]
    strip6 = [(0.10 + 0.16 * i, 0.1770, "other", W) for i in range(6)]
    assert rd._board_row_partial(river_partial) is True
    assert rd._board_row_partial(river_partial + strip6) is True


def test_a_three_point_frame_fit_is_not_trusted_to_zone_cards():
    """Round 4, adversary A: the residual gate admits fits that re-zone hero's own
    hole cards as the community board, with every net silent.

    ANCHOR_MAX_RESID is a SHAPE test, and shape needs redundancy. The similarity
    has three degrees of freedom (uniform scale + two translations), so a
    three-landmark fit has none: some transform always puts three points near
    three of the eight reference slots, and the residual it reports is a tautology
    rather than a measurement.

    Measured by enumerating every stack_text subset of every card-bearing frame of
    the 07-23 3.21 PM recording and re-zoning through each gate-passing fit:

        landmarks   subsets   pass the gate   silently mis-zone
            3          3206        1773              15
            4          3990        2883              15
            5          3178        3008               0
            6+         2088        2088               0

    Worst case at 3: hero's own 8c 6c become the BOARD, the real flop zones
    "other", hero comes back empty, and board_row_missed, board_row_partial and
    hero_seat_mismatch are all False.

    Costs nothing measured: the fewest stack_text boxes on any card-bearing frame
    across the five development recordings is 6."""
    pts = _transform(REF_PTS, 1.0, 0.0, 0.0)
    assert rd.anchor_for_frame(_stack_frame(pts)) is not None
    assert la.ANCHOR_MIN_TRUSTED_POINTS >= 5

    sparse = rd.anchor_for_frame(_stack_frame(pts[:4]))
    assert sparse is None, (
        "a four-landmark fit carries one degree of redundancy and still mis-zones")
    # ... and anchor_from_points itself is unchanged: it still produces the fit,
    # so the coin-based path and the session median keep their own semantics.
    assert la.anchor_from_points(pts[:4], REF_PTS) is not None


def _paint_reference_coins(img, *, radius: int, pot_radius: int) -> None:
    import cv2

    height, width = img.shape[:2]
    for nx, ny in la.REF_SEAT_COINS:
        cv2.circle(
            img,
            (int(nx * width), int(ny * height)),
            radius,
            (0, 220, 0),
            -1,
        )
    cv2.circle(
        img,
        (int(la.REF_POT_COIN[0] * width), int(la.REF_POT_COIN[1] * height)),
        pot_radius,
        (0, 220, 0),
        -1,
    )


def test_detect_coins_keeps_half_scale_seat_constellation() -> None:
    """Fixed 3x3 open + area>=40 erased ClubWPT coins on ~1052x732 windows."""
    np = pytest.importorskip("numpy")
    scale = 0.5
    width = int(la.REF_W * scale)
    height = int(la.REF_H * scale)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    _paint_reference_coins(img, radius=max(3, int(11 * scale)), pot_radius=max(4, int(14 * scale)))
    coins = la.detect_coins(img)
    assert len(coins) >= 8, f"expected seat+pot coins at half scale, got {len(coins)}"


def test_detect_coins_reference_scale_still_finds_constellation() -> None:
    np = pytest.importorskip("numpy")
    img = np.zeros((la.REF_H, la.REF_W, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    _paint_reference_coins(img, radius=11, pot_radius=14)
    assert len(la.detect_coins(img)) >= 8


# The ClubWPT coin is a thin green RING, and the client antialiases it, so the
# only pixels that clear the saturation gate are the stroke's core: measured on
# real captures that core is ~1px on a 1344x836 client and ~2px at reference
# size. _paint_reference_coins paints FILLED discs instead, which is why the
# whole existing coin suite passed while a real 1344x836 recording had every one
# of its 1021 frames rejected -- a 3x3 open leaves a filled disc alone and
# erases a 1-2px stroke.
_COIN_CORE_STROKE_PX = 2.0


def _ring_table_frame(scale: float, *, with_pot: bool = True):
    """A table frame whose coins are drawn the way the client draws them."""
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    width, height = int(la.REF_W * scale), int(la.REF_H * scale)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)
    radius = max(3, round(11 * scale))
    stroke = max(1, round(_COIN_CORE_STROKE_PX * scale))
    slots = list(la.REF_SEAT_COINS)
    if with_pot:
        # Seat-sized on purpose: the client does not draw the pot coin 1.6x a
        # seat coin at every window size (see POT_SLOT_MAX_OFFSET).
        slots.append(la.REF_POT_COIN)
    for nx, ny in slots:
        cv2.circle(img, (int(nx * width), int(ny * height)), radius,
                   (0, 220, 0), stroke)
    return img


def test_scaled_odd_kernel_shrinks_instead_of_rounding_back_up() -> None:
    """The morphology gate must fall with scale over the WHOLE range.

    Rounding an even proportional size up to the next odd handed a 0.5-0.833
    scale client the full calibrated 3x3 open, which erases the 2px ring stroke
    the seat coin is drawn with. Both reference geometries keep the kernels they
    were measured with; only the band in between changes.
    """
    reference = la._coin_detection_scale(2114, 1414)     # the 2114x1414 recording
    half = la._coin_detection_scale(1052, 732)           # the 1052x732 recording
    middle = la._coin_detection_scale(1344, 836)         # the recording that broke

    assert (la._scaled_odd_kernel(3, 1.0), la._scaled_odd_kernel(7, 1.0)) == (3, 7)
    assert (la._scaled_odd_kernel(3, reference), la._scaled_odd_kernel(7, reference)) == (3, 7)
    assert (la._scaled_odd_kernel(3, half), la._scaled_odd_kernel(7, half)) == (1, 3)
    assert (la._scaled_odd_kernel(3, middle), la._scaled_odd_kernel(7, middle)) == (1, 5)
    # Never larger than the calibrated kernel, however big the capture.
    assert (la._scaled_odd_kernel(3, 2.5), la._scaled_odd_kernel(7, 2.5)) == (3, 7)
    # Always an odd, usable structuring element.
    for scale in (0.05, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 2.0):
        for calibrated in (3, 7):
            size = la._scaled_odd_kernel(calibrated, scale)
            assert size >= 1 and size % 2 == 1


def test_detect_coins_keeps_ring_drawn_coins_at_intermediate_scale() -> None:
    """1344x836: the geometry whose every frame read as non-table."""
    coins = la.detect_coins(_ring_table_frame(la._coin_detection_scale(1344, 836)))
    assert len(coins) >= 8, (
        f"ring-drawn coins must survive the scaled open, got {len(coins)}")


def test_anchor_fits_ring_drawn_intermediate_scale_table() -> None:
    scale = la._coin_detection_scale(1344, 836)
    fitted = la.anchor(_ring_table_frame(scale))
    assert fitted is not None
    # The fit recovers the true scale rather than collapsing onto the few blobs
    # that survived: the broken build reported s ~= 0.32 for this frame.
    assert abs(fitted["s"] - scale) / scale < 0.05
    assert fitted["resid"] <= la.ANCHOR_MAX_RESID


def test_anchor_finds_a_seat_sized_pot_coin_by_position() -> None:
    """The pot coin is not 1.6x a seat coin on every client size.

    On 1344x836 the client draws it seat-sized, so the area rule filed it as a
    seat and ``has_pot_coin`` came back False on frames with a pot plainly on
    the table. Position is the scale-free property.
    """
    scale = la._coin_detection_scale(1344, 836)
    img = _ring_table_frame(scale)          # pot coin painted at seat radius
    coins = la.detect_coins(img)
    _seats, by_area = la._classify(coins)
    assert by_area is None, "this frame is the case the area rule cannot see"

    fitted = la.anchor(img)
    assert fitted is not None
    assert fitted["has_pot_coin"], "a coin sits exactly on the projected pot slot"
    # Recovered by REclassification, not by counting the same blob twice.
    assert fitted["n_seats"] == len(la.REF_SEAT_COINS)


def test_anchor_reports_no_pot_coin_when_the_table_has_none() -> None:
    """The positional fallback must not borrow a seat coin as the pot."""
    fitted = la.anchor(_ring_table_frame(la._coin_detection_scale(1344, 836),
                                         with_pot=False))
    assert fitted is not None
    assert not fitted["has_pot_coin"]
    assert fitted["n_seats"] == len(la.REF_SEAT_COINS)
