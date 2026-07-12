"""Deterministic action-pill reader (no VLM at runtime).

Separate from poker_tracker/. Part of the deterministic READ step (Findings 06).

Under each seat panel a small rounded pill shows the player's last action, e.g.
RAISE (orange), CALL/BET (green), CHECK/FOLD/BB (neutral gray/white). We anchor a
pill ROI to each canonical seat-coin position, then determine:
  - COLOR: median hue of the saturated fill  -> {orange, green, neutral}
  - LABEL: binarize the white text and template-match a short word bank.
Color alone separates raise (orange) from call/bet (green) from passive
(check/fold/bb, neutral). Word templates resolve within the neutral group.

Combined with the seat bet amount (read_seats), color+amount reconstructs the
action: bet>0 & orange -> raise; bet>0 & green -> call/bet; bet==0 & neutral ->
check/fold; small forced bet & neutral -> blind post.
"""
from __future__ import annotations

import numpy as np
import cv2

from landmark_anchor import REF_SEAT_COINS

# pill box relative to a seat coin (reference-normalized full-frame coords).
PILL_DX0, PILL_DX1 = -0.020, 0.078
PILL_DY0, PILL_DY1 = 0.028, 0.072

TEXT_SIZE = (72, 20)   # (w, h) normalized pill-word image


def _pill_box(idx):
    u, v = REF_SEAT_COINS[idx]
    return (u + PILL_DX0, u + PILL_DX1, v + PILL_DY0, v + PILL_DY1)


def _colored_frac(crop_bgr):
    """(fraction of saturated 'fill' pixels, their median hue)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    colored = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 70)
    n = int(colored.sum())
    frac = n / max(crop_bgr[:, :, 0].size, 1)
    hue = float(np.median(hsv[:, :, 0][colored])) if n else -1.0
    return frac, hue


def _fill_color(crop_bgr):
    """Colour class of the pill fill -> {orange, green, neutral}. Requires a
    substantial coloured fill (>5% of the crop) to avoid anti-alias noise."""
    frac, h = _colored_frac(crop_bgr)
    if frac < 0.05:
        return "neutral"          # white/gray pill (CHECK/FOLD/BB) or noise
    if 8 <= h <= 28:
        return "orange"
    if 35 <= h <= 95:
        return "green"
    return "neutral"


def pill_present(crop_bgr):
    """Present if there's a coloured fill (call/raise) OR a bright bar with white
    text (neutral pill). Fraction-based so it's scale-invariant."""
    if crop_bgr.size == 0:
        return False
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    frac, _ = _colored_frac(crop_bgr)
    return frac > 0.05 or float((g > 70).mean()) > 0.22


def _text_glyph(crop_bgr):
    """Binarized white pill text, cropped to its bbox and resized."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 150, 255, cv2.THRESH_BINARY)  # white text
    ys, xs = np.where(bw > 0)
    if len(xs) < 10:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = bw[y0:y1, x0:x1]
    return cv2.resize(crop, TEXT_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def build_word_templates(labeled):
    """labeled: list of (pill_crop_bgr, word). Returns {word: [glyphs]}."""
    bank: dict[str, list] = {}
    for crop, word in labeled:
        gl = _text_glyph(crop)
        if gl is not None:
            bank.setdefault(word, []).append(gl)
    return bank


def _match_word(glyph, bank):
    best, bestw = 1e9, "?"
    for w, temps in bank.items():
        for t in temps:
            d = float(np.mean((glyph - t) ** 2))
            if d < best:
                best, bestw = d, w
    return bestw, best


def read_pill(crop_bgr, word_bank=None):
    """Return {present, color, word} for one seat pill crop."""
    if not pill_present(crop_bgr):
        return {"present": False, "color": None, "word": None}
    color = _fill_color(crop_bgr)
    word = None
    if word_bank:
        gl = _text_glyph(crop_bgr)
        if gl is not None:
            word, _ = _match_word(gl, word_bank)
    return {"present": True, "color": color, "word": word}


def read_pills(img_bgr, anchor_map, word_bank=None):
    out = []
    for idx in range(len(REF_SEAT_COINS)):
        x0, x1, y0, y1 = anchor_map(_pill_box(idx))
        crop = img_bgr[max(y0, 0):y1, max(x0, 0):x1]
        r = read_pill(crop, word_bank)
        r["seat"] = idx
        out.append(r)
    return out


def save_word_bank(bank, path):
    flat = {}
    for w, temps in bank.items():
        for i, t in enumerate(temps):
            flat[f"{w}|{i}"] = t
    np.savez_compressed(path, **flat)


def load_word_bank(path):
    data = np.load(path)
    bank: dict[str, list] = {}
    for k in data.files:
        w = k.split("|")[0]
        bank.setdefault(w, []).append(data[k])
    return bank
