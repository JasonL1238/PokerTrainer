"""Deterministic CV readers for the ClubWPT HUD: numeric amounts + action pills.

This is the attribute-reader layer the reconstruction spine calls (region_detections
read_amount / read_pill_action). It is intentionally NOT a vision model: the ClubWPT
client renders pot/stack/bet text and action pills in a fixed font at a fixed style,
so a template-matched glyph reader is exact, fast, and offline. (Per project rule:
the production read is deterministic CV; a VLM is used only to *calibrate* these
templates, never at runtime -- see calibrate_ocr.py.)

Pipeline:
    crop (BGR) --binarize_text--> white-on-black mask
      numbers: --segment_glyphs--> per-digit template match --> longest numeric run
      pills:   --whole-word mask--> word-template match (+ colour tiebreak)

Templates live in cv_lab/models/ocr_templates.npz, built by calibrate_ocr.py. If the
file is absent the readers degrade to returning None (the spine then leaves the
amount/action unfilled, exactly as with the earlier attr=None stubs).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "cv_lab" / "models" / "ocr_templates.npz"

# Normalized glyph / word canvas sizes (rows, cols).
DIGIT_SIZE = (22, 16)
WORD_SIZE = (24, 96)

# Pill background colour -> candidate actions (HSV hue ranges). Text is definitive;
# colour only breaks ties / fills gaps when the word template is uncalibrated.
PILL_VOCAB = ("check", "call", "bet", "raise", "fold", "all-in", "post_blind")


# --------------------------------------------------------------------------- #
# Binarization + segmentation
# --------------------------------------------------------------------------- #
def binarize_text(crop_bgr: np.ndarray, v_min: int = 150, s_max: int = 90) -> np.ndarray:
    """Isolate near-white text (high value, low saturation) -> uint8 {0,255} mask.

    Drops the green chip icon, coloured pill fills, and dark backgrounds, keeping
    the white glyph strokes shared by pot/stack/bet/pill text."""
    if crop_bgr.size == 0:
        return np.zeros((0, 0), np.uint8)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    return (((v > v_min) & (s < s_max)).astype(np.uint8)) * 255


@dataclass
class Glyph:
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray  # bool, cropped to bbox


def segment_glyphs(mask: np.ndarray, min_area: int = 5, min_h_px: int = 4) -> list[Glyph]:
    """Connected-component glyphs, left-to-right. Filters only specks (absolute area
    / height floors); the decimal dot and full-height glyphs both survive. Callers do
    the height-relative and token filtering, since box padding varies by HUD element."""
    if mask.size == 0 or mask.max() == 0:
        return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[Glyph] = []
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[k])
        if area < min_area or h < min_h_px:
            continue
        out.append(Glyph(x, y, w, h, (lab[y : y + h, x : x + w] == k)))
    out.sort(key=lambda g: g.x)
    return out


def tokenize(glyphs: list[Glyph], gap_factor: float = 0.9, min_gap: float = 6.0) -> list[list[Glyph]]:
    """Group left-to-right glyphs into tokens separated by wide horizontal gaps
    (the chip icon and inter-word spaces that separate POT:, the value, and BB)."""
    if not glyphs:
        return []
    med_w = float(np.median([g.w for g in glyphs]))
    tokens: list[list[Glyph]] = [[glyphs[0]]]
    for prev, g in zip(glyphs, glyphs[1:]):
        gap = g.x - (prev.x + prev.w)
        if gap > max(min_gap, gap_factor * med_w):
            tokens.append([])
        tokens[-1].append(g)
    return tokens


def _norm(bitmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a bool/uint8 glyph bitmap to `size` and return a flat unit vector."""
    src = (bitmap.astype(np.uint8)) * 255
    r = cv2.resize(src, (size[1], size[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    r -= r.mean()
    nrm = np.linalg.norm(r)
    return (r / nrm).ravel() if nrm > 1e-6 else r.ravel()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# --------------------------------------------------------------------------- #
# Template bank
# --------------------------------------------------------------------------- #
class TemplateOCR:
    """Holds averaged digit templates and whole-word pill templates."""

    def __init__(self, digits: dict[str, np.ndarray], words: dict[str, np.ndarray]):
        self.digits = digits  # char -> unit vector (DIGIT_SIZE)
        self.words = words    # word -> unit vector (WORD_SIZE)

    # ---- persistence ----
    @classmethod
    def load(cls, path: Path | str = DEFAULT_TEMPLATE_PATH) -> "TemplateOCR | None":
        path = Path(path)
        if not path.is_file():
            return None
        z = np.load(path, allow_pickle=False)
        digits = {k[len("d_") :]: z[k] for k in z.files if k.startswith("d_")}
        words = {k[len("w_") :]: z[k] for k in z.files if k.startswith("w_")}
        return cls(digits, words)

    def save(self, path: Path | str = DEFAULT_TEMPLATE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"d_{k}": v for k, v in self.digits.items()}
        payload.update({f"w_{k}": v for k, v in self.words.items()})
        np.savez_compressed(path, **payload)

    # ---- glyph classification ----
    def classify_digit(self, glyph: np.ndarray) -> tuple[str, float]:
        """Nearest template over digits AND affix glyphs (B/P/T letters, 'c' chip
        icon). Returning a non-digit lets read_number reject POT:/BB/chip glyphs
        instead of misreading B as 8 or the chip's suit highlight as 0."""
        vec = _norm(glyph, DIGIT_SIZE)
        best, bs = "", -1.0
        for ch, tpl in self.digits.items():
            sc = _cos(vec, tpl)
            if sc > bs:
                best, bs = ch, sc
        return best, bs

    # ---- number reading ----
    def read_number(self, crop_bgr: np.ndarray, min_score: float = 0.55) -> tuple[float | None, str]:
        """Return (value, debug_string). Isolates the value from POT:/BB/chip/name:
        keep full-height glyphs, split into gap-separated tokens, and pick the token
        that reads as an all-confident-digit run. The decimal point is a short glyph on
        the baseline; since ClubWPT always renders fractions with exactly 2 places, a
        detected dot inside the value maps to a 2-decimal split rather than being
        template-matched (a tiny dot matches unreliably)."""
        mask = binarize_text(crop_bgr)
        comps = segment_glyphs(mask)
        if not comps:
            return None, ""
        max_h = max(c.h for c in comps)
        tall = [c for c in comps if c.h >= 0.55 * max_h]        # digits + same-height letters
        dots = [c for c in comps if c.h < 0.5 * max_h and c.w <= 0.55 * max_h]  # ., colon, specks
        if not tall:
            return None, ""

        # A stack box can include the player's NAME on the row above the value; its
        # digits (e.g. "Lord5699") would otherwise interleave by x with the value.
        # Group glyphs into vertical row-bands by y-center first, then read each row.
        tall.sort(key=lambda g: g.y + g.h / 2.0)
        bands: list[list[Glyph]] = [[tall[0]]]
        for prev, g in zip(tall, tall[1:]):
            if (g.y + g.h / 2.0) - (prev.y + prev.h / 2.0) > 0.6 * max_h:
                bands.append([])
            bands[-1].append(g)

        # Within each row, take the longest contiguous run of confident DIGITS. Affix
        # letters (P/T/B of POT:/BB, or name letters) classify as letters and break it.
        candidates: list[tuple] = []  # (n_digits, mean_score, row_y, run)
        for band in bands:
            band.sort(key=lambda g: g.x)
            # Value digits are the full-height glyphs of the row; the "BB" suffix caps
            # are ~0.67x shorter. Excluding short glyphs drops BB even when a B is
            # mis-scored as 8 (B and 8 are otherwise easily confused).
            band_h = max(g.h for g in band)
            labeled = [(g, *self.classify_digit(g.mask)) for g in band]
            runs: list[list[tuple]] = [[]]
            for item in labeled:
                g, ch, sc = item
                if ch.isdigit() and sc >= min_score and g.h >= 0.78 * band_h:
                    runs[-1].append(item)
                elif runs[-1]:
                    runs.append([])
            for r in runs:
                if r:
                    row_y = float(np.mean([g.y for g, _, _ in r]))
                    candidates.append((len(r), float(np.mean([s for _, _, s in r])), row_y, r))
        if not candidates:
            return None, ""
        # Prefer the longest digit run; ties -> the lower row (the value sits below the
        # name) -> higher mean match.
        run = max(candidates, key=lambda c: (c[0], c[2], c[1]))[3]

        digits = "".join(ch for _, ch, _ in run)
        gx = [(g.x, g.x + g.w) for g, _, _ in run]
        x0, x1 = gx[0][0], gx[-1][1]
        # Detect the decimal two ways (ClubWPT fractions are always 2 places, so a
        # detected decimal -> split the last two digits):
        #  1. a short dot glyph sitting strictly between value digits, or
        #  2. a scale-invariant gap: the inter-digit space at the decimal is markedly
        #     wider than the others (the faint dot blob often falls below the speck
        #     floor in the smaller stack font, but its gap always survives).
        has_dot = any(x0 < (d.x + d.w / 2.0) < x1 for d in dots)
        if not has_dot and len(digits) >= 3:
            gaps = [gx[i + 1][0] - gx[i][1] for i in range(len(gx) - 1)]
            if gaps and max(gaps) >= 5 and max(gaps) >= 1.7 * float(np.median(gaps)):
                has_dot = True
        raw = f"{digits[:-2]}.{digits[-2:]}" if (has_dot and len(digits) > 2) else digits
        try:
            return float(raw), raw
        except ValueError:
            return None, raw

    # ---- pill reading ----
    def read_word(self, crop_bgr: np.ndarray, min_score: float = 0.40) -> tuple[str | None, float]:
        mask = binarize_text(crop_bgr)
        glyphs = segment_glyphs(mask)
        if glyphs:
            max_h = max(g.h for g in glyphs)
            glyphs = [g for g in glyphs if g.h >= 0.5 * max_h]  # drop underscores/specks
        if not glyphs or not self.words:
            return None, 0.0
        x0 = min(g.x for g in glyphs)
        y0 = min(g.y for g in glyphs)
        x1 = max(g.x + g.w for g in glyphs)
        y1 = max(g.y + g.h for g in glyphs)
        word_mask = mask[y0:y1, x0:x1]
        vec = _norm(word_mask > 0, WORD_SIZE)
        best, bs = None, -1.0
        for w, tpl in self.words.items():
            sc = _cos(vec, tpl)
            if sc > bs:
                best, bs = w, sc
        if bs < min_score:
            return None, bs
        return best, bs


def pill_color(crop_bgr: np.ndarray) -> str:
    """Coarse pill background colour -> 'green' | 'orange' | 'gray'."""
    if crop_bgr.size == 0:
        return "gray"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sat = s > 70
    if sat.mean() < 0.08:
        return "gray"
    hue = float(np.median(h[sat]))
    if 35 <= hue <= 90:
        return "green"
    if 5 <= hue <= 25:
        return "orange"
    return "gray"


# --------------------------------------------------------------------------- #
# Public reader entrypoints (used by region_detections).
# --------------------------------------------------------------------------- #
_CACHE: dict[str, TemplateOCR | None] = {}


def _bank(path: Path | str = DEFAULT_TEMPLATE_PATH) -> TemplateOCR | None:
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = TemplateOCR.load(path)
    return _CACHE[key]


def _crop(img_bgr: np.ndarray, xyxy: Sequence[float]) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    return img_bgr[y1:y2, x1:x2]


def read_amount_from_image(img_bgr: np.ndarray, xyxy: Sequence[float]) -> float | None:
    bank = _bank()
    if bank is None:
        return None
    val, _ = bank.read_number(_crop(img_bgr, xyxy))
    return val


def read_pill_attr(img_bgr: np.ndarray, xyxy: Sequence[float]) -> str | None:
    """Read a pill's attr WITHOUT resolving the gray ambiguity: the action word when
    the template matches, else the background colour ('green'/'orange'/'gray').
    region_detections.read_pill_action resolves colours once dealt-in is known."""
    crop = _crop(img_bgr, xyxy)
    if crop.size == 0:
        return None
    bank = _bank()
    if bank is not None:
        word, _ = bank.read_word(crop)
        if word:
            return word
    return pill_color(crop)


def read_pill_from_image(img_bgr: np.ndarray, xyxy: Sequence[float], *, dealt_in: bool) -> str | None:
    """Read the pill's action word; fall back to colour when the word is unreadable.
    A gray pill with no readable word is check (still holding cards) else fold."""
    crop = _crop(img_bgr, xyxy)
    bank = _bank()
    word = None
    if bank is not None:
        word, _ = bank.read_word(crop)
    if word:
        return word
    color = pill_color(crop)
    if color == "orange":
        return "raise"
    if color == "green":
        return "call"  # green = call/bet; call is the safe default when text is lost
    return "check" if dealt_in else "fold"
