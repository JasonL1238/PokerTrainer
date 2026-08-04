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

# --- repo-root on sys.path so cv_lab.scripts.* absolute imports resolve ---
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))

from cv_lab.scripts.pipeline.landmark_anchor import anchor

# Scale band vs the coin-constellation reference (2138x1402). LO: below ~0.45x
# the coin blobs fall under the detection morphology's minimum area. HI: 2.5
# admits a 2x Retina/HiDPI capture of the reference-sized client (which fits at
# s ~= 2.0 and used to black the whole recording out at the old 1.8 ceiling),
# with the same relative headroom the low side carries over its measured
# minimum (0.45 vs 0.512 observed).
SCALE_LO, SCALE_HI = 0.45, 2.5
MAX_RESID = 0.02          # median reprojection error < 2% of table width
MIN_SEATS, MAX_SEATS = 5, 10


def classify(img_bgr, a=None, reasons=None):
    """Return (label, anchor_dict_or_None). label in {'table','nontable'}.

    ``reasons`` is an optional dict tally: each nontable verdict increments the
    check that failed. A recording rejected wholesale used to be
    indistinguishable from a recording of the lobby -- "all frames nontable"
    with no trace of WHY -- and the difference between "the operator recorded
    the wrong screen" and "the capture scale is outside the classifier band"
    decides what the operator should do about it.
    """
    if a is None:
        a = anchor(img_bgr)
    if a is None:
        if reasons is not None:
            reasons["no_coin_constellation"] = reasons.get("no_coin_constellation", 0) + 1
        return "nontable", None
    checks = (
        ("seat_count_out_of_range", MIN_SEATS <= a["n_seats"] <= MAX_SEATS),
        ("scale_outside_band", SCALE_LO <= a["s"] <= SCALE_HI),
        ("residual_too_high", a["resid"] <= MAX_RESID),
        ("no_pot_coin", bool(a["has_pot_coin"])),
    )
    failed = [name for name, ok in checks if not ok]
    if failed:
        if reasons is not None:
            for name in failed:
                reasons[name] = reasons.get(name, 0) + 1
        return "nontable", a
    return "table", a
