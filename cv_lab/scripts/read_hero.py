"""Deterministic HERO hole-card reader (no VLM at runtime).

Separate from poker_tracker/. Part of the deterministic READ step (Findings 06).

Hero's two cards are fanned/tilted at bottom-center, overlapping. Unlike the
axis-aligned board slots, each must be DESKEWED before the rank/suit glyph lines
up with the shared card templates (read_cards). Pipeline:

  1. anchor a hero ROI above the hero seat coin.
  2. mask the bright white card faces on the dark felt.
  3. take the up-to-2 largest card blobs; for each, minAreaRect -> rotation.
  4. rotate the card upright (deskew) and crop its face.
  5. read rank+suit from the deskewed face's index corner using the SAME
     suit-colour + shape + rank templates as the board reader.

Reuses read_cards templates (same card art/font; template match is scale-norm).
"""
from __future__ import annotations

import numpy as np
import cv2

from landmark_anchor import REF_SEAT_COINS
from read_cards import (_match, RANK_SIZE, SUIT_SIZE, _tight)

HERO_SEAT = 7  # bottom-center seat coin (Mochi / hero)
# hero-card ROI relative to the hero seat coin (reference-normalized full-frame).
HERO_ROI = (-0.052, 0.052, -0.205, -0.028)   # dx0,dx1,dy0,dy1 from coin

# The fan geometry is fixed in this UI. Each card's FACE is given in hero-ROI-
# normalized coords as (cx, cy, w, h, angle_deg). angle_deg is the CCW rotation
# to apply to bring the card upright (left card tilts CW-negative, right CW+).
LEFT_FACE = (0.30, 0.47, 0.42, 0.60, -8.0)
RIGHT_FACE = (0.62, 0.52, 0.42, 0.62, 11.0)
FACES = [LEFT_FACE, RIGHT_FACE]

# on a deskewed upright card, the index corner (rank over suit), top-left.
IDX_X0, IDX_X1 = 0.05, 0.50
RANK_Y0, RANK_Y1 = 0.03, 0.44
SUIT_Y0, SUIT_Y1 = 0.44, 0.82
# The right fanned card overlaps the left card. A lower/tighter suit crop avoids
# pulling the left card's club pip into black-suit matching.
SUIT_SUBS = [
    (IDX_X0, IDX_X1, SUIT_Y0, SUIT_Y1),
    (0.20, 0.50, 0.42, 0.96),
]


def hero_roi_box():
    u, v = REF_SEAT_COINS[HERO_SEAT]
    dx0, dx1, dy0, dy1 = HERO_ROI
    return (u + dx0, u + dx1, v + dy0, v + dy1)


def card_present(roi_bgr):
    """Are hero cards showing? (enough bright card pixels in the ROI)."""
    if roi_bgr.size == 0:
        return False
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return float((g > 140).mean()) > 0.12


def _deskew(roi_bgr, face):
    """Rotate the ROI by the card's fixed angle and crop its upright face."""
    H, W = roi_bgr.shape[:2]
    cxf, cyf, wf, hf, ang = face
    cx, cy = cxf * W, cyf * H
    cw, ch = wf * W, hf * H
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(roi_bgr, M, (W, H))
    x0 = int(max(cx - cw / 2, 0)); y0 = int(max(cy - ch / 2, 0))
    return rot[y0:int(cy + ch / 2), x0:int(cx + cw / 2)]


def _sub(crop, x0f, x1f, y0f, y1f):
    h, w = crop.shape[:2]
    return crop[int(y0f * h):int(y1f * h), int(x0f * w):int(x1f * w)]


def _glyph(sub_bgr, size):
    g = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY_INV)
    return _tight(bw, size)


def _suit_color(sub_bgr):
    hsv = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255)) | \
          cv2.inRange(hsv, (168, 90, 60), (180, 255, 255))
    return "red" if int((red > 0).sum()) > 15 else "black"


def read_hero_card(face_bgr, tmpl, suit_frac=(IDX_X0, IDX_X1, SUIT_Y0, SUIT_Y1)):
    if face_bgr.size == 0 or min(face_bgr.shape[:2]) < 10:
        return None
    rank_sub = _sub(face_bgr, IDX_X0, IDX_X1, RANK_Y0, RANK_Y1)
    suit_sub = _sub(face_bgr, *suit_frac)
    color = _suit_color(suit_sub)
    cand = ["h", "d"] if color == "red" else ["s", "c"]
    rank = _match(_glyph(rank_sub, RANK_SIZE), tmpl["ranks"])
    suit = _match(_glyph(suit_sub, SUIT_SIZE), tmpl["suits"], candidates=cand)
    return rank + suit


def read_hero(img_bgr, anchor_map, tmpl, debug_dir=None):
    x0, x1, y0, y1 = anchor_map(hero_roi_box())
    roi = img_bgr[max(y0, 0):y1, max(x0, 0):x1]
    if roi.size == 0 or not card_present(roi):
        return []
    out = []
    for i, face_spec in enumerate(FACES):
        face = _deskew(roi, face_spec)
        if debug_dir is not None:
            cv2.imwrite(f"{debug_dir}/hero_face{i}.png", face)
        c = read_hero_card(face, tmpl, SUIT_SUBS[i])
        if c:
            out.append(c)
    return out
