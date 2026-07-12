"""Deterministic screen classifier (no VLM at runtime).

Separate from poker_tracker/. First stage of the pipeline (Findings 06 step 2):
skip lobby/menu/transition frames so the reader only runs on real table frames.

A table frame is the ONLY screen where the saturated-green coins form the rigid
seat constellation with a pot coin at low fit residual. The lobby also shows
scattered green chip icons, but they don't fit the constellation (no pot coin,
wrong count, high residual). So we classify directly off the anchor:

  table   -> anchor fits: 5<=seats<=9, scale in plausible band, low residual.
  else    -> lobby / transition (skip).
"""
from __future__ import annotations

from landmark_anchor import anchor

SCALE_LO, SCALE_HI = 0.45, 1.8
MAX_RESID = 0.02          # median reprojection error < 2% of table width
MIN_SEATS, MAX_SEATS = 5, 10


def classify(img_bgr, a=None):
    """Return (label, anchor_dict_or_None). label in {'table','nontable'}."""
    if a is None:
        a = anchor(img_bgr)
    if a is None:
        return "nontable", None
    ok = (MIN_SEATS <= a["n_seats"] <= MAX_SEATS
          and SCALE_LO <= a["s"] <= SCALE_HI
          and a["resid"] <= MAX_RESID
          and a["has_pot_coin"])
    return ("table" if ok else "nontable"), a


def is_table(img_bgr, a=None):
    return classify(img_bgr, a)[0] == "table"
