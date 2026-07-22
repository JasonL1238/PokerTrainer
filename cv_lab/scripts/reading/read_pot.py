"""Deterministic POT-number reader (no VLM at runtime).

Separate from poker_tracker/. Part of the deterministic READ step (Findings 06):
the pot number is the linchpin of the arithmetic reconciler.

Pipeline per key-frame:
  1. landmark_anchor -> pot ROI (edge/scale tolerant).
  2. mask the green coin, binarize bright text.
  3. connected components -> classify by HEIGHT band:
       digits  h~21 (big font)   |  'BB' letters h~16 (ignored)
       decimal point h~4 (small, low)   |  panel edge lines h<=4 & wide (ignored)
  4. recognize each digit glyph by nearest-template (normalized SSD).
  5. insert '.' by x-position, parse to float.

Templates are built ONCE from frames whose pot value is known (the VLM answer key
is used only to LABEL glyphs at build time; recognition at runtime is pure
template matching). Font is shared with stacks/bets, so the same digit bank
generalizes to those readers later.
"""
from __future__ import annotations

import numpy as np
import cv2

GLYPH_SIZE = (20, 28)   # (w, h) normalized digit


def glyph_boxes(crop_bgr, thr=160):
    """Return (digit_boxes sorted by x, dot_boxes). Boxes are (x,y,w,h)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.dilate(cv2.inRange(hsv, (45, 120, 50), (95, 255, 255)),
                       np.ones((11, 11), np.uint8))
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)
    bw[green > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    digits, dots = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h <= 4 and w > 40:                       # panel edge line
            continue
        if 18 <= h <= 26 and area >= 40:            # number digit
            digits.append((int(x), int(y), int(w), int(h)))
        elif 3 <= h <= 9 and w <= 9 and area >= 5:  # decimal point
            dots.append((int(x), int(y), int(w), int(h)))
        # h ~13-17 => 'BB' letters, ignored
    digits.sort(key=lambda b: b[0])
    return bw, digits, dots


def _norm_glyph(bw, box):
    x, y, w, h = box
    g = bw[y:y + h, x:x + w]
    return cv2.resize(g, GLYPH_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def build_templates(labeled_frames):
    """labeled_frames: list of (pot_crop_bgr, value_string like '162.20').
    Returns {digit_char: [glyph_arrays]}."""
    bank: dict[str, list] = {str(d): [] for d in range(10)}
    for crop, s in labeled_frames:
        bw, digits, dots = glyph_boxes(crop)
        chars = [c for c in s if c.isdigit()]
        if len(chars) != len(digits):
            continue  # segmentation mismatch -> skip for template building
        for box, ch in zip(digits, chars):
            bank[ch].append(_norm_glyph(bw, box))
    return bank


def _recognize(glyph, bank):
    best, bestc = 1e9, "?"
    for ch, temps in bank.items():
        for t in temps:
            d = float(np.mean((glyph - t) ** 2))
            if d < best:
                best, bestc = d, ch
    return bestc, best


def read_pot(crop_bgr, bank):
    """Return (value_float_or_None, raw_string)."""
    bw, digits, dots = glyph_boxes(crop_bgr)
    if not digits:
        return None, ""
    items = [(b[0], _recognize(_norm_glyph(bw, b), bank)[0]) for b in digits]
    for d in dots:
        items.append((d[0], "."))
    items.sort(key=lambda z: z[0])
    raw = "".join(c for _, c in items)
    try:
        return float(raw), raw
    except ValueError:
        return None, raw


def save_bank(bank, path):
    flat = {}
    for ch, temps in bank.items():
        for i, t in enumerate(temps):
            flat[f"{ch}_{i}"] = t
    np.savez_compressed(path, **flat)


def load_bank(path):
    data = np.load(path)
    bank: dict[str, list] = {str(d): [] for d in range(10)}
    for k in data.files:
        ch = k.split("_")[0]
        bank[ch].append(data[k])
    return bank
