"""Deterministic CARD reader (no VLM at runtime).

Separate from poker_tracker/. Part of the deterministic READ step (Findings 06).
Also the deterministic fix for the spade-vs-club ambiguity the VLM kept flagging.

Per anchored card slot:
  1. present? -> mean brightness of the slot (white card vs dark felt).
  2. suit COLOR -> red vs black by HSV red mask in the suit sub-region (trivial,
     robust). This alone splits {h,d} from {s,c}.
  3. suit SHAPE -> template-match the binarized suit pip against the two same-color
     candidates (red: heart vs diamond; black: spade vs club). This is what
     deterministically resolves spade-vs-club.
  4. rank -> template-match the rank glyph (top of card).

Board slots are axis-aligned and evenly spaced (easy). Hero cards are fanned/
tilted -> handled separately later.

Templates are built from cards whose identity is known (VLM answer key used only
to LABEL at build time; runtime is pure template matching).
"""
from __future__ import annotations

import numpy as np
import cv2

# canonical board slots (x0, x1, y0, y1) in reference-normalized coords
BOARD_SLOTS = [
    (0.344, 0.398, 0.403, 0.518),
    (0.403, 0.457, 0.403, 0.518),
    (0.463, 0.517, 0.403, 0.518),
    (0.522, 0.576, 0.403, 0.518),
    (0.581, 0.635, 0.403, 0.518),
]
# sub-regions within a card slot (fractions of the slot)
RANK_SUB = (0.06, 0.94, 0.04, 0.46)
SUIT_SUB = (0.06, 0.94, 0.48, 0.96)
RANK_SIZE = (28, 34)
SUIT_SIZE = (30, 30)


def _sub(crop, frac):
    h, w = crop.shape[:2]
    x0, x1, y0, y1 = frac
    return crop[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def slot_crop(img_bgr, anchor_map, slot):
    x0, x1, y0, y1 = anchor_map(slot)
    return img_bgr[y0:y1, x0:x1]


def card_present(slot_bgr):
    g = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean()) > 90.0     # white card is bright; empty felt is dark


def suit_color(slot_bgr):
    """'red' or 'black' from the suit sub-region."""
    s = _sub(slot_bgr, SUIT_SUB)
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 90, 60), (12, 255, 255)) | \
          cv2.inRange(hsv, (168, 90, 60), (180, 255, 255))
    return "red" if int((red > 0).sum()) > 25 else "black"


def _tight(bw, size):
    """Crop a binary glyph to its foreground bbox, then resize. Consistent
    framing regardless of surrounding margin (board vs hero index corner)."""
    ys, xs = np.where(bw > 0)
    if len(xs) >= 8:
        bw = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(bw, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _rank_glyph(slot_bgr):
    r = _sub(slot_bgr, RANK_SUB)
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY_INV)  # dark glyph on white card
    return _tight(bw, RANK_SIZE)


def _suit_glyph(slot_bgr):
    s = _sub(slot_bgr, SUIT_SUB)
    g = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY_INV)
    return _tight(bw, SUIT_SIZE)


def build_templates(labeled):
    """labeled: list of (slot_bgr, rank_char, suit_char). Returns dict."""
    ranks: dict[str, list] = {}
    suits: dict[str, list] = {}
    for slot, rk, su in labeled:
        ranks.setdefault(rk, []).append(_rank_glyph(slot))
        suits.setdefault(su, []).append(_suit_glyph(slot))
    return {"ranks": ranks, "suits": suits}


def _match(glyph, bank, candidates=None):
    best, bestc = 1e9, "?"
    for ch, temps in bank.items():
        if candidates and ch not in candidates:
            continue
        for t in temps:
            d = float(np.mean((glyph - t) ** 2))
            if d < best:
                best, bestc = d, ch
    return bestc


def read_card(slot_bgr, tmpl):
    if not card_present(slot_bgr):
        return None
    color = suit_color(slot_bgr)
    cand = ["h", "d"] if color == "red" else ["s", "c"]
    rank = _match(_rank_glyph(slot_bgr), tmpl["ranks"])
    suit = _match(_suit_glyph(slot_bgr), tmpl["suits"], candidates=cand)
    return rank + suit


def read_board(img_bgr, anchor_map, tmpl):
    out = []
    for slot in BOARD_SLOTS:
        c = read_card(slot_crop(img_bgr, anchor_map, slot), tmpl)
        if c:
            out.append(c)
    return out


def save_templates(tmpl, path):
    flat = {}
    for kind in ("ranks", "suits"):
        for ch, temps in tmpl[kind].items():
            for i, t in enumerate(temps):
                flat[f"{kind}:{ch}:{i}"] = t
    np.savez_compressed(path, **flat)


def load_templates(path):
    data = np.load(path)
    tmpl = {"ranks": {}, "suits": {}}
    for k in data.files:
        kind, ch, _ = k.split(":")
        tmpl[kind].setdefault(ch, []).append(data[k])
    return tmpl
