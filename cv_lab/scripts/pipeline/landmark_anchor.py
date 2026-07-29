"""Landmark-anchored table localization for ClubWPT Gold recordings.

Separate from poker_tracker/.

PROBLEM (user constraint): screen recordings vary at the edges -- sometimes more
or less non-WPT screen content is captured, so the table sits at a different
offset/scale in the frame. Absolute pixel/fraction ROIs (Findings 04) therefore
break. We must locate the table by ON-SCREEN LANDMARKS and derive ROIs relative
to it.

KEY FINDING (see color analysis in Findings 05): the entire ClubWPT UI is dark
and COOL (hue ~110, saturation < 90) EXCEPT the little green chip-coin icons,
which are fully saturated (sat=255). They are the only saturated pixels in the
UI, so they are trivially and robustly detectable regardless of surrounding crop.
They form a RIGID CONSTELLATION: 8 seat coins at fixed table positions + one
larger POT coin. We fit a similarity transform (uniform scale + translation; no
rotation in a screen recording) from the detected coins to a reference
constellation, then map any ROI through it.

Reference constellation + canonical ROIs are measured from t0360 of
data/videos/clubwpt_session_01.mov (a frame where the table fills the frame, so
its whole-frame-normalized coords == canonical table coords).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# canonical reference frame dims (t0360) and coin constellation (normalized 0..1)
REF_W, REF_H = 2138, 1402
# 8 seat coins (nx, ny) measured from the reference frame
REF_SEAT_COINS = [
    (0.436, 0.199), (0.132, 0.260), (0.800, 0.265), (0.075, 0.464),
    (0.851, 0.467), (0.150, 0.721), (0.778, 0.724), (0.492, 0.820),
]
REF_POT_COIN = (0.481, 0.379)
# canonical ROIs (x0, x1, y0, y1) in reference-normalized coords
CANON_ROIS = {
    "pot": (0.47, 0.57, 0.345, 0.405),
    "board": (0.33, 0.63, 0.39, 0.55),
}

# --------------------------------------------------------------------------- #
# DETECTOR-side anchoring.
#
# The coin constellation above needs an HSV+contour pass over every frame to
# recover landmarks the 7-class region detector already emits. The reconstruction
# spine therefore anchors off detector boxes instead (region_detections feeds
# ``stack_text`` centers into anchor_from_points), and only the ICP/fit machinery
# is shared. The two reference bases below and above were measured on DIFFERENT
# frames and must never be mixed.
# --------------------------------------------------------------------------- #

# Reference basis for DETECTOR-side anchoring. This is the normalized coordinate
# frame of the baseline 2054x1470 recording, i.e. the basis region_detections'
# SEAT_ANCHORS_BY_CLASS was k-means'd in. It is deliberately NOT the coin-based
# REF_W/REF_H above.
REF_DET_W, REF_DET_H = 2054.0, 1470.0

# CORPUS DISCIPLINE FOR EVERY CONSTANT BELOW. All of them are measured on the 5
# DEVELOPMENT recordings only (AR 1.397 / 1.420 / 1.486 / 1.547 / 1.750, fitted
# scale 0.59x-1.34x). The archived fixture sets in cv_lab/results/ are NOT usable
# as measurement material: by geometry, 7 of the 8 belong to locked-test or
# validation recordings, and only frames_v05 (2722x1832) is a development one.
# An earlier revision of this block cited "9 geometries (the 5 development
# recordings plus the 8 archived fixture sets)" and justified REF_POT_RX from an
# off-column HUD element on a 2796x1914 archive -- a geometry that exists only in
# the held-out split. Those numbers have been re-measured below on development
# material and the held-out justification removed.

# Card zones in reference-normalized coords. Measured over 4642 face_card
# detections across the 5 development geometries.
#   board ry observed 0.4372..0.4583; nearest confusers: pot_text at ry<=0.3810
#     below, a stray reveal at ry 0.4811 above (g0723a t=143) -> both band edges
#     are the CONFUSER MIDPOINTS. The upper edge used to be 0.466, computed
#     against a stale board maximum of 0.4513; the real maximum is 0.4583 (g0711
#     t=329, 9d), which left 0.0077 of margin -- a third of the +-0.025 rigid
#     offset the zones tolerate, and small enough that one client resize drops a
#     river card out of the row. Applying the documented midpoint rule to the
#     corrected measurement gives 0.470, symmetric at 0.0114 either side. The
#     margin is still thin by construction, which is why the partial-yield net
#     (region_detections._board_row_partial) now sits under it: a row that loses
#     SOME of its cards to a band edge is no longer silent.
#   hero ry observed 0.6737..0.6874; no confuser within +-0.06 on any geometry
#     -> band is observed extremes +- 0.030 (about 8x the session-median fit error).
#   rx bands are observed extremes +- 0.030 (board 0.3497..0.6043, hero 0.4700..0.5135).
#     REF_BOARD_RX used to read (0.295, 0.640) -- margins of -0.0547 and +0.0357,
#     i.e. it did NOT follow the rule stated on the line above it, and the 0.640
#     was a round number rather than a derivation. REF_HERO_RX did follow it. The
#     rx extremes have been re-measured over 2073 anchored board-card detections on
#     the 5 development recordings and confirm 0.3497..0.6043 exactly, so the
#     constant is now what the rule produces from that measurement. Both edges are
#     pinned in tests/test_landmark_anchor.py; neither had a confuser pin before.
REF_BOARD_RY = (0.409, 0.470)
REF_BOARD_RX = (0.3197, 0.6343)
REF_HERO_RY = (0.644, 0.717)
REF_HERO_RX = (0.440, 0.544)

# Both pots render in the table's CENTRE COLUMN, the main one directly above the
# side one. Measured over 1267 anchored pot_text detections on the 5 development
# recordings: every genuine pot -- main or side -- sits at rx 0.4916..0.5076. The
# band below is those extremes with ~0.05 of margin.
#
# The guard is structural rather than empirical on this corpus: no development
# recording carries an off-column pot_text at all (0 of 1267), so it costs
# nothing here and keeps a misclassified HUD element out of the side-pot slot.
REF_POT_RX = (0.44, 0.56)

# Within that column, the main pot renders at ry 0.3410..0.3810 (n=1258) and a
# side pot at ry 0.5419..0.5460 (n=9, one recording -- 07-15 is the only
# development recording with a side pot). 0.46 splits them with +0.079 above the
# highest main and -0.082 below the lowest side.
REF_POT_SIDE_RY_MIN = 0.46

# A fit is rejected when its median reprojection error exceeds this fraction of
# the fitted table width. Worst residual on any real table frame over 969 fitted
# frames = 0.0240 (07-15); a grossly wrong aspect ratio (1000x1000 against the
# 1.397 reference) produces 0.0537.
#
# WHAT IT CANNOT DO. This is a SHAPE-plausibility gate and nothing more. Any
# translation or uniform scaling of the landmark constellation IS a similarity,
# so anchor_from_points absorbs it exactly and the residual does not move at all:
# sweeping a rigid dy offset from -0.040 to +0.040 of frame height on a real
# frame leaves the residual identical to 13 significant figures while every
# card's reference-y moves by the full offset. Card zoning tolerates about
# +-0.025 of rigid offset, and the residual contributes nothing to that budget.
# The nets that DO bound zoning error are elsewhere and are deliberately
# independent of the fit: region_detections._board_row_missed and
# ._board_row_partial (shape-of-the-row, in scale-free card-width units) and the
# hero-seat cross-check. Do not read this constant as a precision guarantee.
ANCHOR_MAX_RESID = 0.040
ANCHOR_MIN_POINTS = 3

# Landmarks a PER-FRAME fit must rest on before its zoning is trusted.
#
# The residual is a shape test, and shape needs redundancy. This similarity has
# three degrees of freedom (uniform scale + two translations), so a three-point
# fit has none: some transform always drops three points near three of the eight
# reference slots, and the residual it reports is a tautology rather than a
# measurement. ANCHOR_MIN_POINTS = 3 is the algebraic floor for producing a fit at
# all -- it is not a floor for believing one.
#
# Measured by enumerating every stack_text subset of every card-bearing frame of
# the 07-23 3.21 PM recording and re-zoning the frame through each gate-passing
# fit, then comparing against the 8-point truth zoning:
#     landmarks   subsets   pass the gate   silently mis-zone
#         3          3206        1773              15
#         4          3990        2883              15
#         5          3178        3008               0
#         6+         2088        2088               0
# "Silently" meaning board_row_missed, board_row_partial and hero_seat_mismatch
# all False. The worst three-point case zones hero's own hole cards as the BOARD
# and the real flop as "other", with hero coming back empty.
#
# Costs nothing measured: the fewest stack_text boxes on any card-bearing frame
# across the five development recordings is 6 (g0711). Below the floor
# region_detections falls back to the session-median anchor, and failing that
# fails closed -- which is what it already does when no fit exists at all.
ANCHOR_MIN_TRUSTED_POINTS = 5
ANCHOR_MIN_SESSION_FITS = 5


@dataclass(frozen=True)
class TableAnchor:
    """Similarity transform reference-pixels -> frame-pixels (uniform scale +
    translation; a screen recording never rotates)."""

    s: float
    tx: float
    ty: float
    resid: float
    n_points: int
    source: str  # "frame" | "session"

    def to_ref(self, px: float, py: float) -> tuple[float, float]:
        """Pixel center -> reference-normalized (rx, ry)."""
        return ((px - self.tx) / self.s / REF_DET_W,
                (py - self.ty) / self.s / REF_DET_H)


def zone_for_ref(rx: float, ry: float) -> str:
    """Anchored card zone. THE single definition -- region_detections is the only
    caller. build_yolo_card_timeline._zone_for_box is the legacy unanchored
    CSV-path test and is not used by the spine."""
    if REF_BOARD_RY[0] <= ry <= REF_BOARD_RY[1] and REF_BOARD_RX[0] <= rx <= REF_BOARD_RX[1]:
        return "board"
    if REF_HERO_RY[0] <= ry <= REF_HERO_RY[1] and REF_HERO_RX[0] <= rx <= REF_HERO_RX[1]:
        return "hero"
    return "other"


def anchor_from_points(det_pts, ref_pts, *, max_iter: int = 3) -> TableAnchor | None:
    """Similarity fit from a reference constellation onto detected landmark
    centers, by ICP with outlier rejection.

    This is the body of ``anchor()`` re-parameterized to take already-detected
    landmark centers instead of HSV coin blobs, so there is exactly one ICP
    implementation. ``det_pts`` and ``ref_pts`` are pixel-space (x, y) sequences
    and need not be the same length or in correspondence.

    Returns None when fewer than ANCHOR_MIN_POINTS landmarks were detected.
    """
    det_all = [(float(x), float(y)) for x, y in det_pts]
    ref_all = [(float(x), float(y)) for x, y in ref_pts]
    if len(det_all) < ANCHOR_MIN_POINTS or len(ref_all) < ANCHOR_MIN_POINTS:
        return None

    # init: centroid + RMS-radius scale (occupancy-robust)
    dc = _centroid(det_all)
    rc = _centroid(ref_all)
    d_rms = _rms_radius(det_all, dc)
    r_rms = _rms_radius(ref_all, rc)
    if r_rms <= 0.0 or d_rms <= 0.0:
        return None
    s = d_rms / r_rms
    t = (dc[0] - s * rc[0], dc[1] - s * rc[1])

    # ICP-lite: assign each detected landmark to the nearest transformed ref slot, refit.
    for _ in range(max_iter):
        proj = [(s * rx + t[0], s * ry + t[1]) for (rx, ry) in ref_all]
        pairs = []
        for dp in det_all:
            j = min(range(len(proj)),
                    key=lambda k: (dp[0] - proj[k][0]) ** 2 + (dp[1] - proj[k][1]) ** 2)
            pairs.append((ref_all[j], dp))
        dists = [(((s * rp[0] + t[0] - dp[0]) ** 2 +
                   (s * rp[1] + t[1] - dp[1]) ** 2) ** 0.5) for rp, dp in pairs]
        med = _median(dists)
        good = [pair for pair, d in zip(pairs, dists, strict=True) if d <= max(med * 3, 30)]
        if len(good) < ANCHOR_MIN_POINTS:
            break
        s, t = _similarity_fit([g[0] for g in good], [g[1] for g in good])

    # normalized fit residual: median reprojection error / fitted table width (px).
    proj = [(s * rx + t[0], s * ry + t[1]) for (rx, ry) in ref_all]
    errs = [min(((dp[0] - p[0]) ** 2 + (dp[1] - p[1]) ** 2) ** 0.5 for p in proj)
            for dp in det_all]
    resid = _median(errs) / (s * REF_DET_W)
    return TableAnchor(s=float(s), tx=float(t[0]), ty=float(t[1]),
                       resid=float(resid), n_points=len(det_all), source="frame")


def _centroid(pts) -> tuple[float, float]:
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def _rms_radius(pts, c) -> float:
    n = float(len(pts))
    return (sum((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for p in pts) / n) ** 0.5


def _median(vals: list[float]) -> float:
    xs = sorted(vals)
    n = len(xs)
    if not n:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def detect_coins(img_bgr):
    """Return list of dicts: {cx, cy, area} for saturated-green coin blobs (pixels)."""
    import cv2

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (45, 120, 50), (95, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 40:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.5 < w / h < 2.0):        # roughly round
            continue
        out.append({"cx": x + w / 2.0, "cy": y + h / 2.0, "area": float(a),
                    "w": w, "h": h})
    return out


def _classify(coins):
    """Split blobs into seat coins vs pot coin vs bet-chip (by area). Areas at
    reference scale: seat ~340, pot ~555, bet-chips >800. Scale-invariant ratios
    are recovered later, so here we use scale-robust *relative* areas: the median
    round-coin area defines the seat size."""
    if not coins:
        return [], None
    areas = np.array([c["area"] for c in coins])
    seat_area = float(np.median(areas))           # most coins are seat coins
    seats, pot_candidates = [], []
    for c in coins:
        r = c["area"] / seat_area
        if 0.6 <= r <= 1.5:
            seats.append(c)
        elif 1.4 <= r <= 2.4:                      # pot coin ~1.6x a seat coin
            pot_candidates.append(c)
        # r > 2.4 => bet chip, ignored as anchor
    # pot coin = the pot_candidate nearest the top-center of the seat cluster
    pot = None
    if pot_candidates and seats:
        cx = np.mean([s["cx"] for s in seats])
        top = min(s["cy"] for s in seats)
        pot = min(pot_candidates,
                  key=lambda c: (abs(c["cx"] - cx) + abs(c["cy"] - top)))
    elif pot_candidates:
        pot = pot_candidates[0]
    return seats, pot


def _similarity_fit(ref_pts, det_pts):
    """Least-squares similarity (uniform scale s + translation t, no rotation):
    minimize sum |s*ref + t - det|^2.  ref_pts, det_pts: (N,2) matched.

    Pure Python (5-line closed form) so the detector-side anchor path carries no
    array dependency; the arithmetic is identical to the previous numpy form.
    """
    rc = _centroid(ref_pts)
    dc = _centroid(det_pts)
    num = sum((rp[0] - rc[0]) * (dp[0] - dc[0]) + (rp[1] - rc[1]) * (dp[1] - dc[1])
              for rp, dp in zip(ref_pts, det_pts, strict=True))
    den = sum((rp[0] - rc[0]) ** 2 + (rp[1] - rc[1]) ** 2 for rp in ref_pts)
    s = float(num / den)
    return s, (dc[0] - s * rc[0], dc[1] - s * rc[1])


def anchor(img_bgr, max_iter=3):
    """Locate the table. Returns dict with similarity (s, tx, ty), the number of
    matched coins, and a map() closure turning canonical ROIs into pixel boxes.
    Returns None if too few coins detected."""
    coins = detect_coins(img_bgr)
    seats, pot = _classify(coins)
    ref_seat_px = [(u * REF_W, v * REF_H) for (u, v) in REF_SEAT_COINS]
    ref_pot_px = (REF_POT_COIN[0] * REF_W, REF_POT_COIN[1] * REF_H)

    det = [(c["cx"], c["cy"]) for c in seats]
    if pot is not None:
        det_pot = (pot["cx"], pot["cy"])
    else:
        det_pot = None
    if len(det) < 3:
        return None

    # ONE ICP implementation, shared with the detector-side path. The coin
    # constellation is fitted in its own REF_W/REF_H basis, so the residual is
    # re-normalized below by the coin table width rather than REF_DET_W.
    ref_all = ref_seat_px + ([ref_pot_px] if det_pot else [])
    det_all = det + ([det_pot] if det_pot else [])
    fit = anchor_from_points(det_all, ref_all, max_iter=max_iter)
    if fit is None:
        return None
    s, t = fit.s, (fit.tx, fit.ty)
    resid = fit.resid * REF_DET_W / REF_W

    def map_roi(box):
        x0, x1, y0, y1 = box
        px0 = s * (x0 * REF_W) + t[0]
        px1 = s * (x1 * REF_W) + t[0]
        py0 = s * (y0 * REF_H) + t[1]
        py1 = s * (y1 * REF_H) + t[1]
        return int(px0), int(px1), int(py0), int(py1)

    return {"s": s, "tx": float(t[0]), "ty": float(t[1]),
            "n_seats": len(det), "has_pot_coin": det_pot is not None,
            "resid": resid, "map_roi": map_roi}


if __name__ == "__main__":
    import argparse

    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    img = cv2.imread(args.image)
    a = anchor(img)
    if a is None:
        print("anchor FAILED (too few coins)")
    else:
        print(f"s={a['s']:.4f} tx={a['tx']:.1f} ty={a['ty']:.1f} "
              f"seats={a['n_seats']} pot_coin={a['has_pot_coin']}")
        for k, box in CANON_ROIS.items():
            print(f"  {k}: {a['map_roi'](box)}")
