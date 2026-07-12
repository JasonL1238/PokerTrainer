"""Deterministic per-seat reader (no VLM at runtime): stacks + bets.

Separate from poker_tracker/. Part of the deterministic READ step (Findings 06).

Every seat panel shows "<green coin> <stack> BB". The green coin sits at exactly
the landmark constellation position (landmark_anchor.REF_SEAT_COINS), and the
stack number is immediately to its RIGHT. So we anchor a small ROI to each
canonical seat-coin position and OCR it with the SHARED pot digit bank
(read_pot) -- same font, so no new templates needed.

Bets (chips pushed toward the pot on a street) show a second "<coin> <amount>"
cluster between the seat and the table center. We anchor a bet ROI along the
seat->center line and OCR it the same way. Bets are a cross-check; the reconciler
primarily uses street-to-street STACK deltas.

Seat order matches landmark_anchor.REF_SEAT_COINS index (0..7).
"""
from __future__ import annotations

import numpy as np
import cv2

from landmark_anchor import REF_SEAT_COINS, REF_POT_COIN
from read_pot import glyph_boxes, _norm_glyph, _recognize

# stack number box relative to a seat coin, in reference-normalized full-frame
# coords: from just right of the coin, extending right, centered on coin height.
STACK_DX0, STACK_DX1 = 0.006, 0.082      # x offset from coin center (fraction of W)
STACK_DY = 0.017                          # half-height (fraction of H)

# table center = centroid of the 8 seat coins; bets sit on a ring around it.
_CENTER = (float(np.mean([u for u, _ in REF_SEAT_COINS])),
           float(np.mean([v for _, v in REF_SEAT_COINS])))
BET_T = 0.46                              # fraction seat->center for the bet box
BET_DX = 0.058                            # half-width of bet box (fraction of W)
BET_DY = 0.032                            # half-height (absorbs radial variation)


def _seat_stack_box(idx):
    u, v = REF_SEAT_COINS[idx]
    return (u + STACK_DX0, u + STACK_DX1, v - STACK_DY, v + STACK_DY)


def _seat_bet_box(idx):
    u, v = REF_SEAT_COINS[idx]
    cu, cv = _CENTER
    mx = u + BET_T * (cu - u)
    my = v + BET_T * (cv - v)
    return (mx - BET_DX, mx + BET_DX, my - BET_DY, my + BET_DY)


def _read_number(crop_bgr, bank):
    """OCR a small green-coin+number crop -> (value|None, raw)."""
    bw, digits, dots = glyph_boxes(crop_bgr)
    if not digits:
        return None, ""
    items = [(b[0], _recognize(_norm_glyph(bw, b), bank)[0]) for b in digits]
    for d in dots:
        items.append((d[0], "."))
    items.sort(key=lambda z: z[0])
    raw = "".join(c for _, c in items)
    try:
        v = float(raw)
    except ValueError:
        return None, raw
    # a stack/bet of exactly 0 is never displayed (busted seats vanish); treat
    # 0.0 as an all-in/animation misread rather than a real value.
    return (v, raw) if v > 0 else (None, raw)


def read_seats(img_bgr, anchor_map, bank, with_bets=True):
    """Return list of 8 dicts: {seat, stack, stack_raw, bet, bet_raw}.
    stack/bet are floats or None."""
    out = []
    for idx in range(len(REF_SEAT_COINS)):
        x0, x1, y0, y1 = anchor_map(_seat_stack_box(idx))
        crop = img_bgr[max(y0, 0):y1, max(x0, 0):x1]
        stack, sraw = _read_number(crop, bank) if crop.size else (None, "")
        rec = {"seat": idx, "stack": stack, "stack_raw": sraw,
               "bet": None, "bet_raw": ""}
        if with_bets:
            bx0, bx1, by0, by1 = anchor_map(_seat_bet_box(idx))
            bcrop = img_bgr[max(by0, 0):by1, max(bx0, 0):bx1]
            if bcrop.size:
                bet, braw = _read_number(bcrop, bank)
                rec["bet"], rec["bet_raw"] = bet, braw
        out.append(rec)
    return out
