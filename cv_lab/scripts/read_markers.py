"""Deterministic table-marker readers (no VLM at runtime): active seat + dealer.

Separate from poker_tracker/. Part of the deterministic READ step (Findings 06).

ACTIVE SEAT: the player to act has a bright BLUE countdown ring around their
avatar. Blue is (like the green coins) a uniquely saturated colour in the
otherwise dark/cool UI, so we mask blue, take the largest ring blob, and assign
it to the nearest seat-coin landmark.

DEALER BUTTON: a small near-white 'D' disc sits on the felt beside the button
seat. We mask bright low-saturation circular blobs of button size on the felt
and assign the best one to the nearest seat-coin landmark. (The button is also
derivable from the blind posts, so this is a cross-check.)
"""
from __future__ import annotations

import numpy as np
import cv2

from landmark_anchor import REF_SEAT_COINS, REF_W, REF_H


def _seat_pixels(anchor_map):
    """Canonical seat-coin positions mapped to pixel coords via the anchor."""
    pts = []
    for (u, v) in REF_SEAT_COINS:
        x0, x1, y0, y1 = anchor_map((u, u, v, v))
        pts.append((x0, y0))
    return pts


def _nearest_seat(pt, seat_px):
    d = [((pt[0] - sx) ** 2 + (pt[1] - sy) ** 2) for (sx, sy) in seat_px]
    j = int(np.argmin(d))
    return j, float(np.sqrt(d[j]))


def active_seat(img_bgr, anchor_map, scale):
    """Return (seat_idx|None, info). Detect the blue timer ring."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (100, 120, 120), (130, 255, 255))
    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seat_px = _seat_pixels(anchor_map)
    best = None
    min_area = 2000 * scale * scale            # enclosed disc area of the ring
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.80 <= w / max(h, 1) <= 1.20):  # the ring is essentially circular
            continue
        # HOLLOW test: a thin ring has few blue px relative to its bbox; filled
        # card-back pairs (~0.40) and avatar silhouettes fill much more of it.
        fill = float((blue[y:y + h, x:x + w] > 0).mean())
        if not (0.05 <= fill <= 0.30):
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        if best is None or a > best[0]:
            best = (a, cx, cy)
    if best is None:
        return None, {}
    j, dist = _nearest_seat((best[1], best[2]), seat_px)
    # ring must be near a seat (avatar sits close to the coin)
    if dist > 0.12 * REF_W * scale:
        return None, {"reason": "blue blob far from any seat", "dist": dist}
    return j, {"area": best[0], "dist": dist}


def dealer_seat(img_bgr, anchor_map, scale):
    """Return (seat_idx|None, info). Detect the near-white 'D' disc."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    white = ((hsv[:, :, 1] < 60) & (g > 180)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seat_px = _seat_pixels(anchor_map)
    lo, hi = 500 * scale * scale, 2600 * scale * scale     # solid disc area band
    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (lo <= a <= hi):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.75 < w / max(h, 1) < 1.35):    # round disc
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (peri * peri + 1e-6)
        if circ < 0.7:
            continue
        fill = float((white[y:y + h, x:x + w] > 0).mean())
        if fill < 0.55:                          # solid disc, not a hollow glyph
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        cands.append((a, cx, cy, circ))          # rank by size (disc > stray text)
    if not cands:
        return None, {}
    cands.sort(reverse=True)                     # largest disc first
    cands = [(circ, cx, cy, a) for (a, cx, cy, circ) in cands]
    # button sits between two seats on the felt; assign to nearest seat.
    for circ, cx, cy, a in cands:
        j, dist = _nearest_seat((cx, cy), seat_px)
        if dist < 0.22 * REF_W * scale:
            return j, {"circ": circ, "area": a, "dist": dist, "pos": (cx, cy)}
    return None, {"reason": "no disc near a seat"}
