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

import numpy as np
import cv2

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


def detect_coins(img_bgr):
    """Return list of dicts: {cx, cy, area} for saturated-green coin blobs (pixels)."""
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
    minimize sum |s*ref + t - det|^2.  ref_pts, det_pts: (N,2) matched."""
    ref = np.asarray(ref_pts, float)
    det = np.asarray(det_pts, float)
    rc, dc = ref.mean(0), det.mean(0)
    rr, dd = ref - rc, det - dc
    s = float((rr * dd).sum() / (rr * rr).sum())
    t = dc - s * rc
    return s, t


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

    # init: centroid + RMS-radius scale (occupancy-robust)
    det_arr = np.array(det)
    ref_arr = np.array(ref_seat_px)
    dc, rc = det_arr.mean(0), ref_arr.mean(0)
    d_rms = np.sqrt(((det_arr - dc) ** 2).sum(1).mean())
    r_rms = np.sqrt(((ref_arr - rc) ** 2).sum(1).mean())
    s = d_rms / r_rms
    t = dc - s * rc

    # ICP-lite: assign each detected seat coin to nearest transformed ref slot, refit
    ref_all = ref_seat_px + ([ref_pot_px] if det_pot else [])
    det_all = det + ([det_pot] if det_pot else [])
    for _ in range(max_iter):
        proj = [(s * rx + t[0], s * ry + t[1]) for (rx, ry) in ref_all]
        pairs = []
        for dp in det_all:
            j = int(np.argmin([(dp[0] - p[0]) ** 2 + (dp[1] - p[1]) ** 2 for p in proj]))
            pairs.append((ref_all[j], dp))
        # reject correspondences that are absurdly far (bet chips etc.)
        med = np.median([((s * rp[0] + t[0] - dp[0]) ** 2 +
                          (s * rp[1] + t[1] - dp[1]) ** 2) ** 0.5 for rp, dp in pairs])
        good = [(rp, dp) for rp, dp in pairs
                if (((s * rp[0] + t[0] - dp[0]) ** 2 +
                     (s * rp[1] + t[1] - dp[1]) ** 2) ** 0.5) <= max(med * 3, 30)]
        if len(good) < 3:
            break
        s, t = _similarity_fit([g[0] for g in good], [g[1] for g in good])

    # normalized fit residual: mean reprojection error / table width (px).
    proj = [(s * rx + t[0], s * ry + t[1]) for (rx, ry) in ref_all]
    errs = []
    for dp in det_all:
        errs.append(min(((dp[0] - p[0]) ** 2 + (dp[1] - p[1]) ** 2) ** 0.5 for p in proj))
    resid = float(np.median(errs)) / (s * REF_W)

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
