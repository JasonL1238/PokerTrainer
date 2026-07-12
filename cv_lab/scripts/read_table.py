"""Unified deterministic table reader (no VLM at runtime).

Separate from poker_tracker/. This is the pipeline's per-frame READ facade
(Findings 06 step 5): given one BGR frame it classifies the screen, anchors the
table, and runs every deterministic extractor, returning a structured snapshot.
Nothing here calls a vision model.
"""
from __future__ import annotations

import cv2

from classify_screen import classify
from read_pot import load_bank, read_pot
from read_cards import load_templates, read_board
from read_seats import read_seats
from read_pills import read_pills, load_word_bank
from read_markers import active_seat, dealer_seat
from read_hero import read_hero
from landmark_anchor import CANON_ROIS

_DIGITS = None
_CARDS = None
_WORDS = None


def load_models(models_dir):
    global _DIGITS, _CARDS, _WORDS
    _DIGITS = load_bank(f"{models_dir}/pot_digits.npz")
    _CARDS = load_templates(f"{models_dir}/card_templates.npz")
    try:
        _WORDS = load_word_bank(f"{models_dir}/pill_words.npz")
    except Exception:
        _WORDS = None


def read_table(img_bgr, a=None):
    """Return a snapshot dict, or {'screen':'nontable'} for skipped frames."""
    label, a = classify(img_bgr, a)
    if label != "table":
        return {"screen": label}
    m = a["map_roi"]
    px0, px1, py0, py1 = m(CANON_ROIS["pot"])
    pot, pot_raw = read_pot(img_bgr[max(py0, 0):py1, max(px0, 0):px1], _DIGITS)
    board = read_board(img_bgr, m, _CARDS)
    hero = read_hero(img_bgr, m, _CARDS)
    seats = read_seats(img_bgr, m, _DIGITS, with_bets=True)
    pills = read_pills(img_bgr, m, word_bank=_WORDS)
    act, _ = active_seat(img_bgr, m, a["s"])
    dlr, _ = dealer_seat(img_bgr, m, a["s"])
    return {
        "screen": "table",
        "scale": a["s"], "resid": a["resid"],
        "pot": pot, "pot_raw": pot_raw,
        "board": board, "hero": hero,
        "seats": [{"seat": s["seat"], "stack": s["stack"], "bet": s["bet"]} for s in seats],
        "pills": [{"seat": p["seat"], "present": p["present"], "color": p["color"],
                   "word": p.get("word")} for p in pills],
        "active_seat": act, "dealer_seat": dlr,
    }
