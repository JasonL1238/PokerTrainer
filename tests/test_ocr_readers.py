"""Offline tests for the deterministic template OCR (cv_lab/scripts/ocr_readers.py).

The production template bank (ocr_templates.npz) is a local calibration artifact,
so these tests build a synthetic bank from cv2-rendered glyphs instead: digits are
drawn with putText and the chip icon is a filled wide ellipse (its binarized white
suit-highlight is squat and wider than tall, unlike every digit). This locks the
regression where the chip icon next to bet amounts classified as a confident '0',
joined the digit run, and turned "12" into 0.12 via the gap-inferred decimal.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from cv_lab.scripts.ocr_readers import (
    DIGIT_SIZE,
    TemplateOCR,
    _norm,
    binarize_text,
    segment_glyphs,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.9
THICK = 2
WHITE = (255, 255, 255)


def _render(draw) -> np.ndarray:
    img = np.zeros((48, 320, 3), np.uint8)
    draw(img)
    return img


def _glyphs(img: np.ndarray):
    return segment_glyphs(binarize_text(img))


def _digit_template(ch: str) -> np.ndarray:
    img = _render(lambda im: cv2.putText(im, ch, (8, 34), FONT, SCALE, WHITE, THICK))
    glyphs = _glyphs(img)
    assert len(glyphs) == 1, f"digit {ch!r} rendered {len(glyphs)} glyphs"
    return _norm(glyphs[0].mask, DIGIT_SIZE)


def _draw_chip(img: np.ndarray, cx: int, cy: int = 24) -> None:
    # Squat filled ellipse, wider than tall - same silhouette class as the chip
    # icon's white suit highlight that survives binarize_text at HUD scale.
    cv2.ellipse(img, (cx, cy), (13, 9), 0, 0, 360, WHITE, -1)


@pytest.fixture(scope="module")
def digit_templates() -> dict[str, np.ndarray]:
    return {ch: _digit_template(ch) for ch in "0123456789"}


@pytest.fixture(scope="module")
def chip_template() -> np.ndarray:
    img = _render(lambda im: _draw_chip(im, 20))
    glyphs = _glyphs(img)
    assert len(glyphs) == 1
    return _norm(glyphs[0].mask, DIGIT_SIZE)


def _chip_then_12(img: np.ndarray) -> None:
    # chip, wide gap, then "12" with tight digit spacing: the layout whose wide
    # chip gap used to satisfy the gap-inferred-decimal rule.
    _draw_chip(img, 20)
    cv2.putText(im := img, "1", (60, 34), FONT, SCALE, WHITE, THICK)
    cv2.putText(im, "2", (78, 34), FONT, SCALE, WHITE, THICK)


def test_plain_integer_reads_without_chip(digit_templates) -> None:
    bank = TemplateOCR(dict(digit_templates), {})
    img = _render(lambda im: cv2.putText(im, "12", (60, 34), FONT, SCALE, WHITE, THICK))
    val, raw = bank.read_number(img)
    assert (val, raw) == (12.0, "12")


def test_chip_breaks_out_of_digit_run(digit_templates, chip_template) -> None:
    bank = TemplateOCR({**digit_templates, "c": chip_template}, {})
    img = _render(_chip_then_12)
    val, raw = bank.read_number(img)
    assert (val, raw) == (12.0, "12")


def test_without_chip_template_chip_would_join_run(digit_templates, chip_template) -> None:
    """Documents why the 'c' affix exists: with digits only, the chip's best
    match is a digit and it joins the run (misreading the value)."""
    bank = TemplateOCR(dict(digit_templates), {})
    ch, score = bank.classify_digit(_glyphs(_render(lambda im: _draw_chip(im, 20)))[0].mask)
    assert ch.isdigit()
    if score < 0.55:
        pytest.skip("synthetic chip fell below the confidence floor; run not joined")
    val, raw = bank.read_number(_render(_chip_then_12))
    assert raw != "12"  # chip polluted the run


def test_genuine_decimal_still_reads(digit_templates, chip_template) -> None:
    def draw(im: np.ndarray) -> None:
        _draw_chip(im, 20)
        cv2.putText(im, "0", (60, 34), FONT, SCALE, WHITE, THICK)
        cv2.circle(im, (82, 33), 2, WHITE, -1)  # decimal dot on the baseline
        cv2.putText(im, "5", (90, 34), FONT, SCALE, WHITE, THICK)
        cv2.putText(im, "0", (108, 34), FONT, SCALE, WHITE, THICK)

    bank = TemplateOCR({**digit_templates, "c": chip_template}, {})
    val, raw = bank.read_number(_render(draw))
    assert (val, raw) == (0.5, "0.50")
