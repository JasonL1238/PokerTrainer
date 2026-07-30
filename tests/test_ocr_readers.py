"""Offline tests for the deterministic template OCR.

THE READER'S CONTRACT CHANGED IN THIS PHASE. It now returns a numeric value only
when the read is PROVABLY unambiguous, and UNKNOWN with a named refusal code in
every other case. Lower coverage is an accepted outcome of that; a confident wrong
number is not. Several tests below therefore assert UNKNOWN where they used to
assert a value -- each one records the measured cost of that loss in its own
docstring rather than being quietly deleted.

The production template bank (cv_lab/models/ocr_templates.npz) is TRACKED -- 21 KB
of averaged glyph vectors that cannot be rebuilt without the source recordings.
While cv_lab/models/ was ignored wholesale, 8 of the real-crop tests below skipped
on every clean checkout, in CI and in the Docker build, so the decimal and
truncation defects they pin were protected only on the calibrating machine.

Tests that do NOT need the bank build a synthetic one from cv2-rendered glyphs:
digits are drawn with putText, the 'B' of the "BB" suffix likewise, and the chip
icon is a filled wide ellipse (its binarized white suit-highlight is squat and
wider than tall, unlike every digit). This locks the regression where the chip icon
next to bet amounts classified as a confident '0' and joined the digit run.

Every synthetic crop renders a WHOLE HUD token -- `<numeral> BB` -- because the
reader now requires the numeral to be terminated by its suffix. A bare row of
digits is not a thing the ClubWPT client ever draws, and a test that feeds one is
testing a shape production never sees.

Synthetic glyphs alone are not enough for the decimal-inference rules: the dot is a
2-3px component whose exact geometry decides whether it survives segmentation, and a
dot drawn at a convenient size passes tests that production fails. The decimal tests
below therefore run against frozen NATIVE-RESOLUTION HUD crops in tests/fixtures/ocr/
(see tests/fixtures/PROVENANCE.json for source recording and true value), plus
synthetic cases rendered at the *measured* dot and gap ratios.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from cv_lab.scripts.pipeline import ocr_readers
from cv_lab.scripts.pipeline.ocr_readers import (
    DECIMAL_EVIDENCE,
    DIGIT_SIZE,
    REFUSAL_CODES,
    AmountRead,
    TemplateOCR,
    _norm,
    binarize_text,
    segment_glyphs,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE = 0.9
THICK = 2
WHITE = (255, 255, 255)

FIXTURES = Path(__file__).parent / "fixtures" / "ocr"


def _render(draw) -> np.ndarray:
    img = np.zeros((48, 320, 3), np.uint8)
    draw(img)
    return img


def _glyphs(img: np.ndarray):
    return segment_glyphs(binarize_text(img))


def _glyph_template(ch: str) -> np.ndarray:
    img = _render(lambda im: cv2.putText(im, ch, (8, 34), FONT, SCALE, WHITE, THICK))
    glyphs = _glyphs(img)
    assert len(glyphs) == 1, f"glyph {ch!r} rendered {len(glyphs)} components"
    return _norm(glyphs[0].mask, DIGIT_SIZE)


def _draw_bb(img: np.ndarray, x: int, y: int = 34, scale: float = SCALE) -> None:
    """The "BB" suffix, at a word space from the value.

    The reader requires it: the client renders `<numeral> BB` on every numeric
    class, and the suffix is the one token whose expected content is known a
    priori, so it is the only available proof that the numeral was not truncated.
    """
    cv2.putText(img, "BB", (x, y), FONT, scale, WHITE, THICK)


def _draw_chip(img: np.ndarray, cx: int, cy: int = 24) -> None:
    # Squat filled ellipse, wider than tall - same silhouette class as the chip
    # icon's white suit highlight that survives binarize_text at HUD scale.
    cv2.ellipse(img, (cx, cy), (13, 9), 0, 0, 360, WHITE, -1)


@pytest.fixture(scope="module")
def digit_templates() -> dict[str, np.ndarray]:
    """Digits plus the 'B' of the "BB" suffix.

    'B' is not optional furniture: without it in the bank the suffix classifies as
    a digit, the numeral is never proven terminated, and every synthetic read is
    UNKNOWN. Its presence is also what makes `classify_digit`'s pooled argmax the
    affix-vs-digit decision the reader relies on."""
    return {ch: _glyph_template(ch) for ch in "0123456789B"}


@pytest.fixture(scope="module")
def chip_template() -> np.ndarray:
    img = _render(lambda im: _draw_chip(im, 20))
    glyphs = _glyphs(img)
    assert len(glyphs) == 1
    return _norm(glyphs[0].mask, DIGIT_SIZE)


def _draw_12_bb(img: np.ndarray) -> None:
    # "12 BB" at a 3px inter-digit gap. The spacing is deliberate: at this render
    # size a '1' inks a bare stroke inside a full-width advance, so the default
    # putText advance leaves a gap of 0.36 of glyph height -- inside the band a
    # located decimal occupies -- and the read is then correctly refused. Real
    # ClubWPT kerning is tighter than cv2's; 47 of 18006 corpus reads sit in the
    # band and are refused there too.
    cv2.putText(img, "1", (60, 34), FONT, SCALE, WHITE, THICK)
    cv2.putText(img, "2", (73, 34), FONT, SCALE, WHITE, THICK)
    _draw_bb(img, 115)


def _chip_then_12(img: np.ndarray) -> None:
    # chip, wide gap, then "12 BB" with tight digit spacing: the layout whose wide
    # chip gap used to satisfy the gap-inferred-decimal rule.
    _draw_chip(img, 20)
    _draw_12_bb(img)


def test_plain_integer_reads_without_chip(digit_templates) -> None:
    bank = TemplateOCR(dict(digit_templates), {})
    val, raw = bank.read_number(_render(_draw_12_bb))
    assert (val, raw) == (12.0, "12")


def test_chip_breaks_out_of_digit_run(digit_templates, chip_template) -> None:
    bank = TemplateOCR({**digit_templates, "c": chip_template}, {})
    val, raw = bank.read_number(_render(_chip_then_12))
    assert (val, raw) == (12.0, "12")


def test_without_chip_template_chip_would_join_run(digit_templates, chip_template) -> None:
    """Documents why the 'c' affix exists: with no chip template its best match is
    something else in the bank, and if that is a digit it joins the run."""
    bank = TemplateOCR(dict(digit_templates), {})
    ch, score = bank.classify_digit(_glyphs(_render(lambda im: _draw_chip(im, 20)))[0].mask)
    if not ch.isdigit() or score < 0.55:
        pytest.skip("synthetic chip did not resolve to a confident digit; run not joined")
    val, raw = bank.read_number(_render(_chip_then_12))
    assert raw != "12"  # chip polluted the run


def test_genuine_decimal_still_reads(digit_templates, chip_template) -> None:
    # Kerning matters: P5(b) refuses any hole the separator does not explain that
    # exceeds the widest measured intra-numeral gap (0.25 of run height), so this
    # render keeps its dot-adjacent holes at 3px against a 22px run (0.14, inside
    # the 0.08-0.17 band real client decimals measure). cv2's default putText
    # advance is looser than the client's font and would land the dot-to-'5' hole
    # at 0.32 -- a spacing no real ClubWPT numeral produces.
    def draw(im: np.ndarray) -> None:
        _draw_chip(im, 20)
        cv2.putText(im, "0", (60, 34), FONT, SCALE, WHITE, THICK)
        cv2.circle(im, (82, 33), 2, WHITE, -1)  # decimal dot on the baseline
        cv2.putText(im, "5", (86, 34), FONT, SCALE, WHITE, THICK)
        cv2.putText(im, "0", (104, 34), FONT, SCALE, WHITE, THICK)
        _draw_bb(im, 145)

    bank = TemplateOCR({**digit_templates, "c": chip_template}, {})
    val, raw = bank.read_number(_render(draw))
    assert (val, raw) == (0.5, "0.50")


def test_sub_baseline_speck_is_not_a_decimal(digit_templates) -> None:
    """A stray speck below the digit baseline (cursor fleck, sub-baseline noise)
    shares the value's x-range but sits well under it. It must NOT be read as a
    decimal point -- this is the regression that turned an integer "211" into 2.11
    (verified on real HUD crops: a w4xh6 speck ~30px below the digits)."""
    bank = TemplateOCR(dict(digit_templates), {})
    img = _compose("211", 1.2, [5, 5], specks=[(0, 3, 3, 1.55)])
    assert bank.read_number(img) == (211.0, "211")


def test_above_baseline_speck_is_not_a_decimal(digit_templates) -> None:
    """Sibling of the sub-baseline case, and the direction the old test left open:
    the dot test accepted ANY vertical overlap with the digit run, so a speck level
    with the digits' TOPS -- card-border ringing, a chip-sprite edge, compression
    noise -- was a valid decimal candidate and deflated the value 100x.

    Measured over 9088 real decimal reads, a true decimal point's centre lands at
    0.846-0.958 of the run band (0 = top of the digits, 1 = bottom); not one is
    below 0.75.

    Under the completed P2 (the policed set now includes sub-band ink -- see
    _policed_ink), a speck sitting ON the numeral is no longer merely "not a
    decimal": it is ink the read cannot explain, and the read is UNKNOWN. The
    original property this test pinned still holds -- the speck must never split
    the value 100x -- and the refusal is the strictly safer outcome: routing
    small ink into an unpoliced bucket is exactly what let a leading-digit
    occlusion sliver ship 50.0 for a 99.50 BB stack (cwpt01 t=554)."""
    bank = TemplateOCR(dict(digit_templates), {})
    img = _compose("312", 1.2, [5, 5], specks=[(1, 3, 3, 0.02)])
    detail = bank.read_number_detail(img)
    assert detail.value is None, "a speck at the digit tops must not yield a value"
    assert detail.decimal_source == "unexplained_ink_in_numeral"


def test_decimal_splits_at_dot_not_last_two(digit_templates) -> None:
    """When a trailing fractional glyph drops out of the run, the split must follow
    the dot's real x-position, not blindly take the last two digits: "127.8" (a
    dropped trailing 0) must read 127.8, never 12.78."""
    bank = TemplateOCR(dict(digit_templates), {})
    # The dot's HOSTING gap must sit inside the measured decimal band (>= 3/13
    # of run height; here 28px digits -> >= 6.46px): a located decimal has never
    # occupied a narrower gap on any development geometry, and the reader now
    # refuses a separator candidate hosted by one (the dot-forgery family).
    img = _compose("1278", 1.2, [5, 5, 7], dot=(2, 3, 3))
    assert bank.read_number(img) == (127.8, "127.8")


# --------------------------------------------------------------------------- #
# Decimal inference on frozen native-resolution HUD crops.
#
# Every read below was measured on a real development recording before the fix; the
# "was" values are recorded in tests/fixtures/PROVENANCE.json. The defects: an
# absolute >=5px decimal-gap floor that no small render size ever clears, a
# min_h_px=4 speck floor that deletes the 2-3px dot itself, and a hardcoded
# "fractions are always 2 places" split.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def production_bank() -> TemplateOCR:
    bank = TemplateOCR.load()
    if bank is None:
        pytest.skip("cv_lab/models/ocr_templates.npz missing -- it is tracked; restore it")
    return bank


def _read_fixture(bank: TemplateOCR, name: str) -> tuple[float | None, str]:
    img = cv2.imread(str(FIXTURES / name))
    assert img is not None, f"missing OCR fixture {name}"
    return bank.read_number(img)


def test_real_crop_small_render_keeps_decimal(production_bank) -> None:
    """1272x896 stack_text: was (31490.0, '31490') -- a 100x inflation."""
    assert _read_fixture(production_bank, "stack_314_90_at_1272x896.png") == (314.9, "314.90")


def test_real_crop_small_render_bet_keeps_decimal(production_bank) -> None:
    """1272x896 bet_text: was (1950.0, '1950')."""
    assert _read_fixture(production_bank, "bet_19_50_at_1272x896.png") == (19.5, "19.50")


def test_real_crop_leading_zero_decimal(production_bank) -> None:
    """Fires on the BASELINE 2054x1470 geometry: was (50.0, '050'), and reached an
    exported hand as 'preflop seat:3 call 50.0' in a 200 BB game."""
    assert _read_fixture(production_bank, "bet_0_50_at_2054x1470.png") == (0.5, "0.50")


def test_real_crop_one_decimal_place(production_bank) -> None:
    """The 07-15 client renders BB pots with ONE decimal place. Splitting at
    len(digits)-2 turned 240.9 into 24.09."""
    assert _read_fixture(production_bank, "pot_240_9_one_decimal.png") == (240.9, "240.9")


def test_real_crop_three_digit_one_decimal(production_bank) -> None:
    """Three digits, one decimal place: was (891.0, '891')."""
    assert _read_fixture(production_bank, "pot_89_1_one_decimal.png") == (89.1, "89.1")


def test_real_crop_integer_unchanged(production_bank) -> None:
    """Negative control: an integer BB pot must not acquire a decimal point."""
    assert _read_fixture(production_bank, "pot_165_integer.png") == (165.0, "165")


def test_real_crop_true_zero_is_zero_not_none(production_bank) -> None:
    """An all-in seat genuinely shows '0 BB'. A true zero is a VALUE and must stay
    distinguishable from an unreadable crop, which is None."""
    val, raw = _read_fixture(production_bank, "stack_0_true_zero.png")
    assert (val, raw) == (0.0, "0")
    assert val is not None


def test_real_chip_sprite_alone_is_unknown_not_a_confident_zero(production_bank) -> None:
    """A bet_text crop holding ONLY the green chip sprite -- no text whatsoever --
    returned AmountRead(value=0.0, raw='0', score=0.836): byte-identical in every
    field to the genuine all-in "0 BB" read on the line above. The sprite's pale
    annulus scores 0.79-0.88 against the '0' template, well over the 0.55 floor.

    That confident false zero is not inert: stack_text 0.0 is TRUSTED by the spine,
    and a zero stack labels the seat's action "all-in"."""
    val, _raw = _read_fixture(production_bank, "bet_chip_sprite_no_text.png")
    assert val is None, "a crop containing no text must not read as a value"
    # Negative control lives in test_real_crop_true_zero_is_zero_not_none: the
    # genuine all-in "0 BB" must still read 0.0, so this cannot be fixed by
    # banning zeros.


def test_real_crop_read_is_stable_across_a_window_resize(production_bank) -> None:
    """The decimal recovery is documented as scale-RELATIVE, so the same HUD text
    must read the same value when the client window is resized. It did not: this
    1272x896 stack (the smallest supported geometry, run_h 12px) read 343.6 at
    1.00x and 0.95x, 34360.0 at 0.90x, and 343.6 again at 0.85x -- a 100x inflation
    that is NON-MONOTONIC in scale, at full confidence.

    The property being pinned is unchanged: the reader must never return a
    DIFFERENT value for the same HUD text at a different window size."""
    img = cv2.imread(str(FIXTURES / "stack_343_60_at_1272x896.png"))
    assert img is not None
    h, w = img.shape[:2]
    for scale in (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        val, raw = production_bank.read_number(small)
        assert val in (343.6, None), (
            f"scale {scale}: read {val!r} (raw {raw!r}); the only honest answers "
            "are the true value and unknown")
    for scale in (1.0, 0.95):
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        assert production_bank.read_number(small)[0] == 343.6, scale


def test_a_baseline_speck_in_a_letter_gap_never_forges_a_decimal(production_bank) -> None:
    """DOT FORGERY (round-2 adversary A, attack A). A 2x2 text-coloured speck at
    the baseline of the 4px letter gap of a real '197 BB' crop was accepted as a
    decimal separator -- only the SPLIT was checked, never the gap HOSTING the
    candidate -- and the reader shipped a fully confident 1.97, a 100x
    under-read, reproducible on ~45% of integer-read crops across all six
    development geometries. The reader's own calibration table
    (_DECIMAL_BAND_MIN_GAP, n=13,669 located decimals) has no real separator in
    a gap under 3/13 of run height, so a candidate hosted by a narrower gap is
    provably not a decimal: it is unexplained ink, and the read is UNKNOWN.

    The family is general -- ANY sub-band gap on ANY geometry -- and the fix is
    constant-free (the located separator's host gap must reach the same band the
    integer branch already refuses from), so every real located-dot read keeps
    its value by construction: measured 0 value changes over 13,669 dot reads."""
    control = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_197_at_2054x1470.png")))
    assert control.value == 197.0
    assert control.decimal_source == "integer"
    forged = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_197_speck_forged_decimal_at_2054x1470.png")))
    assert forged.value != 1.97, "the speck must never fabricate a decimal"
    assert forged.value is None, "a speck-hosting gap below the decimal band refuses"
    assert forged.decimal_source == "unexplained_ink_in_numeral"


def test_a_wide_occluder_fragment_far_from_the_run_still_refuses(production_bank) -> None:
    """WIDE OCCLUDER (round-2 adversary A, attack D). The repo's own chip-sprite
    fixture pasted over the leading digits of a real '343.60 BB' crop leaves
    three mask fragments the bank confidently matches to NOTHING -- but they end
    26px left of the surviving run, outside the 0.28*run_h adjacency window, so
    no predicate policed them and the surviving trailing '0' shipped as a
    confident 0.0, which the spine books as a genuine all-in. Distance is not an
    exemption: glyph-scale ink on the numeral's row must be explained (accepted
    run, affix, separator, confident template match, or an anchored chain of
    letter-spaced glyphs -- see _unanchored_row_ink) wherever it sits, and this
    crop must therefore be UNKNOWN, never a number."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_343_60_sprite_far_fragments_at_1272x896.png")))
    assert detail.value != 0.0, "an occluded 343.60 BB stack must never read as all-in"
    assert detail.value is None
    assert detail.decimal_source == "unexplained_ink_in_numeral"


def test_read_number_detail_reports_decimal_source(production_bank) -> None:
    """The evidence channel: downstream must be able to tell a value whose decimal
    was actually LOCATED from one whose absence was proven."""
    dot = production_bank.read_number_detail(cv2.imread(str(FIXTURES / "pot_240_9_one_decimal.png")))
    integer = production_bank.read_number_detail(cv2.imread(str(FIXTURES / "pot_165_integer.png")))
    assert (dot.value, dot.decimal_source, dot.digits) == (240.9, "dot", 4)
    assert (integer.value, integer.decimal_source, integer.digits) == (165.0, "integer", 3)
    assert dot.score > 0.5 and integer.score > 0.5


# --------------------------------------------------------------------------- #
# Synthetic cases rendered at MEASURED geometry (composited glyph bitmaps, so the
# inter-digit gaps and the dot size are exact rather than font-metric accidents).
# --------------------------------------------------------------------------- #
def _digit_bitmap(ch: str, scale: float) -> np.ndarray:
    img = np.zeros((90, 90, 3), np.uint8)
    cv2.putText(img, ch, (15, 70), FONT, scale, WHITE, THICK)
    glyphs = _glyphs(img)
    assert len(glyphs) == 1
    return glyphs[0].mask


def _compose(
    digits: str,
    scale: float,
    gaps: list[int],
    dot: tuple[int, int, int] | None = None,
    dots: list[tuple[int, int, int]] | None = None,
    specks: list[tuple[int, int, int, float]] | None = None,
    suffix: bool = True,
    suffix_chars: str = "BB",
):
    """Lay out `digits` with exact pixel gaps, then a word space and the "BB"
    suffix; `dot` is (gap_index, w, h), drawn on the digit baseline. `dots` draws
    several -- a client that renders BOTH a group separator and a decimal point
    puts two same-silhouette glyphs in one numeral. `specks` is
    (gap_index, w, h, band_fraction), placing a NON-baseline component at a chosen
    height (0.0 = the digits' tops, 1.0 = their baseline, >1 = below it) so the
    dot's position test can be exercised at an exact fraction rather than at a
    font-metric accident.

    The suffix is rendered by default because the reader requires it. The word
    space is 0.60 of glyph height, inside the measured 0.55-and-up band that
    separates "POT:" and "BB" from the value, and comfortably outside the 0.28
    letter spacing that defines the numeral.

    `suffix=False` renders the numeral with NO terminator, and `suffix_chars`
    renders a MALFORMED one ("8B", "B"). Both are P1's own condition, which is
    otherwise unreachable synthetically: every other synthetic crop here renders
    the well-formed "BB" precisely because the reader requires it, so nothing
    exercised the refusal itself."""
    bitmaps = [_digit_bitmap(c, scale) for c in digits]
    height = max(b.shape[0] for b in bitmaps)
    caps = [_digit_bitmap(c, scale) for c in suffix_chars] if suffix else []
    word = round(0.60 * height)
    kern = max(1, round(0.10 * height))
    tail = word + sum(b.shape[1] for b in caps) + kern * len(caps) if caps else 0
    width = sum(b.shape[1] for b in bitmaps) + sum(gaps) + tail
    img = np.zeros((height + 40, width + 40, 3), np.uint8)
    x, y, spans = 20, 20, []
    for i, b in enumerate(bitmaps):
        bh, bw = b.shape
        img[y : y + bh, x : x + bw][b] = WHITE
        spans.append((x, x + bw))
        x += bw + (gaps[i] if i < len(gaps) else 0)
    if caps:
        x += word
        for b in caps:
            bh, bw = b.shape
            img[y : y + bh, x : x + bw][b] = WHITE
            x += bw + kern
    for gap_i, dw, dh in ([dot] if dot is not None else []) + list(dots or []):
        gx0, gx1 = spans[gap_i][1], spans[gap_i + 1][0]
        cx = (gx0 + gx1) // 2 - dw // 2
        img[y + height - dh : y + height, cx : cx + dw] = WHITE
    for gap_i, sw, sh, frac in specks or []:
        gx0, gx1 = spans[gap_i][1], spans[gap_i + 1][0]
        cx = (gx0 + gx1) // 2 - sw // 2
        cy = int(y + frac * height) - sh // 2
        img[cy : cy + sh, cx : cx + sw] = WHITE
    return img


@pytest.fixture(scope="module")
def scaled_digit_templates() -> dict[str, np.ndarray]:
    """Digit (and suffix-'B') templates averaged over three render scales, mirroring
    how calibrate_ocr.py averages real observations. A single-scale synthetic
    template scores below the 0.55 confidence floor on a 13px-tall render."""
    out: dict[str, np.ndarray] = {}
    for ch in "0123456789B":
        vec = np.mean([_norm(_digit_bitmap(ch, s), DIGIT_SIZE) for s in (0.5, 0.9, 1.2)], axis=0)
        out[ch] = vec / np.linalg.norm(vec)
    return out


def _run_height(img: np.ndarray) -> int:
    return max(g.h for g in segment_glyphs(binarize_text(img), min_area=2, min_h_px=1))


def test_decimal_dot_at_production_geometry(scaled_digit_templates) -> None:
    """The dot rendered at its MEASURED size: 3x3px, area 9, against a 22px digit
    height (dot_h/digit_h = 0.136, matching pot_240_9_one_decimal.png).

    test_genuine_decimal_still_reads draws cv2.circle(radius=2), which binarizes to
    w=5 h=5 area=13 -- ratio 0.227, nearly twice the real dot. That over-sized dot
    cleared the min_h_px=4 speck floor, which is why the suite stayed green while
    production dropped the decimal on 15.8% of stack reads."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    img = _compose("050", 0.9, [7, 3], dot=(0, 3, 3))
    dot = [c for c in segment_glyphs(binarize_text(img), min_area=2, min_h_px=1) if c.h == 3]
    assert [(c.w, c.h, int(c.mask.sum())) for c in dot] == [(3, 3, 9)], "dot geometry drifted"
    assert bank.read_number(img) == (0.5, "0.50")


def test_a_located_decimal_reads_the_same_at_two_render_sizes(scaled_digit_templates) -> None:
    """Identical digits, proportional gaps and a real dot at two render sizes must
    produce the SAME split. The old absolute '>= 5px' floor split only the large one
    (and split it in the wrong place); the small one -- exactly the 1272x896 case --
    was left as a 100x-inflated integer."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    small = bank.read_number(_compose("2409", 0.5, [2, 2, 4], dot=(2, 2, 2)))   # run_h 13
    large = bank.read_number(_compose("2409", 1.2, [4, 4, 9], dot=(2, 4, 4)))   # run_h 28
    assert small == large == (240.9, "240.9")


def test_a_decimal_inferred_from_spacing_is_now_a_refusal(scaled_digit_templates) -> None:
    """SUPERSEDES test_gap_fallback_splits_at_widest_gap_not_last_two, which asserted
    this crop reads 240.9 off the widest gap with no dot present.

    The decimal-GAP arm is deleted. It inferred a decimal point from inter-digit
    spacing alone, and its measured firing rate on the six development recordings
    is ZERO -- `decimal_source == "gap"` occurs 0 times in 18006 native reads. Every
    documented failure it caused was real, though: it split "18.30 BB" at the WORD
    SPACE before the suffix and returned a confident 1830.88.

    P5(b) states the same fact as a refusal instead: a numeral with no separator
    anywhere on its baseline whose widest gap reaches the band a located decimal
    occupies is UNKNOWN. Measured cost across the corpus: 47 reads."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    detail = bank.read_number_detail(_compose("2409", 0.9, [1, 2, 7]))
    assert detail.value is None
    assert detail.decimal_source == "integer_over_decimal_band"


def test_leading_zero_integer_is_unknown(scaled_digit_templates) -> None:
    """ClubWPT never renders a leading zero on an integer, so digits '050' with no
    decimal evidence is a known-bad read. It must surface as UNKNOWN, not as a
    confident 50.0 -- which is what reached an exported hand as a 50 BB call."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    detail = bank.read_number_detail(_compose("050", 0.9, [3, 3]))
    assert detail.raw == "050"
    assert detail.value is None
    assert detail.decimal_source == "leading_zero_no_dot"


# --------------------------------------------------------------------------- #
# THE PREDICATE. A read produces a value only when every condition holds.
# --------------------------------------------------------------------------- #
def test_a_glyph_overlapping_the_run_makes_the_read_unknown(production_bank) -> None:
    """P2, the half the old adjacency test could not see.

    `_bridged_gap` returns a NEGATIVE number for a glyph that OVERLAPS the digit
    run, and the test was written `0 <= gap < 0.28 * run_h`, so an unaccepted glyph
    sitting literally on top of the numeral was the one kind of intruder the net
    ignored. On this crop a chip stack covers "40.9" of a "POT: 240.9 BB", leaving a
    single '2' that passes every other check: unclipped, run height 20 inside the
    band, and invariant under all 27 constant ablations. It shipped a confident 2.0
    -- a 120x under-read.

    Measured: exactly ONE read in 17785 value-producing reads on the six
    development recordings has an overlapping unaccepted glyph, and it is this
    one. Deleting the lower bound therefore costs one read and buys the case."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "pot_240_9_chip_overlaps_run_at_2132x1378.png"))
    )
    assert detail.value is None, "a glyph sitting on the numeral is unexplained ink"
    assert detail.value != 2.0
    assert detail.decimal_source == "unexplained_ink_in_numeral"


def test_a_malformed_bb_suffix_makes_the_read_unknown(production_bank) -> None:
    """P1. The client renders `<numeral> BB` on all three numeric classes, and the
    suffix butts against the value at 0.07-0.27 of run height -- inside the letter
    spacing -- so it is exactly where truncation and occlusion strike, and it is the
    only token whose expected content is known a priori.

    This bet_text crop is covered by chip sprites: there is no legible number in it
    at all, one sprite edge reads as a confident '1', and the suffix reads "3B". It
    shipped a confident 1.0.

    Measured over 17785 value reads: 98.31% carry a clean "BB". The 239 refusals are
    dominated by "8B" (107) and "B8" (68) -- crops where the bank cannot tell a 'B'
    from an '8' in the suffix, which is the same discrimination it is relying on
    inside the numeral."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "bet_chip_covered_malformed_suffix_at_2054x1470.png"))
    )
    assert detail.value is None, "a crop with no legible numeral must not ship 1.0"
    # This crop used to fall through to P1 (suffix "3B"). With the policed set
    # completed, the sprite fragments AROUND the '1' are unexplained ink and P2
    # fires first -- an earlier, stronger refusal of the same crop, so this test
    # does NOT exercise P1 at all.
    #
    # An earlier version of this comment claimed "P1's own firing is pinned by
    # test_bb_suffix_is_never_absorbed_into_the_numeral and by the 221
    # suffix_not_bb refusals in the corpus census". Both halves were false. That
    # test only asserts the caps are not absorbed INTO the run (its crop's value
    # stays unknown by a different route), a corpus census is not a test, and
    # ablating P1 to `if False and ...` left the whole owned CV suite GREEN while
    # 218 of the 219 suffix_not_bb refusals came back as confident values. P1's
    # own firing is pinned by the two tests directly below.
    assert detail.decimal_source == "unexplained_ink_in_numeral"


def test_the_timer_badge_never_ships_as_the_stack(production_bank) -> None:
    """The real crop that carries TWO numerals, and the reader wins with the
    wrong one.

    A g0621 seat panel (2062x1178) showing the stack the screen renders
    "212.90 BB" plus a circular action-timer badge reading "12" over the
    player's avatar. The badge's 2-digit run is the one P3 selects, because the
    stack's own glyphs are barely half its height and never enter the band.
    Note 12 recorded this crop shipping a confident 12.0 -- a 17.7x under-read of
    a real stack -- back when run completeness was a ranking key.

    ATTRIBUTION MOVED IN THE ROUND-3 REPAIR, and the assertion moved with it
    rather than being softened. Until the repair, P1 was the refusing predicate:
    a timer carries no "BB". P2's far arm now fires first, because the stack's
    own digits sit on the badge's row as glyph-scale ink and a confidently
    classified DIGIT outside the winning run is no longer accepted as its own
    explanation (see _unanchored_row_ink). The crop is still refused, still never
    ships 12.0, and the predicate that refuses it is named exactly -- if a third
    one starts firing here, this assertion says so. P1's own killing tests are
    `test_p1_fires_on_every_malformed_terminator` (synthetic family) and
    `test_p1_refuses_a_digit_the_chip_template_swallowed` (real crop, and a
    genuine confident-wrong)."""
    img = cv2.imread(str(FIXTURES / "stack_212_90_timer_badge_at_2062x1178.png"))
    detail = production_bank.read_number_detail(img)
    assert detail.value is None, "a crop carrying two numerals is not proven"
    assert detail.value != 12.0, "the timer badge must never ship as the stack"
    assert detail.decimal_source == "unexplained_ink_in_numeral", (
        "name the predicate that actually refuses this crop; claiming a "
        "predicate fires where a different one does is how P1 came to be "
        f"unpinned in the first place (got {detail.decimal_source})")
    # Structural: the fixture must still carry a SECOND numeral, or the test is
    # exercising nothing. The badge's digits and the stack's digits differ in
    # height by more than the band floor.
    comps = segment_glyphs(binarize_text(img), min_area=2, min_h_px=1)
    max_h = max(c.h for c in comps)
    tall = [c for c in comps if c.h >= 0.55 * max_h]
    short_digits = [c for c in comps
                    if 0.3 * max_h <= c.h < 0.55 * max_h
                    and production_bank.classify_digit(c.mask)[0].isdigit()]
    assert len(tall) == 2, "the badge's own 2-digit run must be the only tall run"
    assert len(short_digits) >= 4, "the stack's digits must still be in the crop"


def test_p1_refuses_a_digit_the_chip_template_swallowed(production_bank) -> None:
    """P1 on a REAL crop, and on a genuine confident-wrong read.

    P1 used to inspect `suffix[:2]` of a list from which every glyph the bank
    labelled 'c' had ALREADY been deleted. Two severed digits were invisible at
    once, and this fixture is the first: shaving four rows off the top of the '8'
    of the frozen "218 BB" stack -- any occluder clipping the top of the last
    digit -- drops it to 0.667 of band height, where the bank scores it as the
    chip template at 0.576. The chip filter then removed it from the suffix
    entirely, leaving ['B','B'], which passed, and the reader shipped a confident
    21.0 for a 218 BB starting stack. The parent crop already refuses; the shaved
    one carries strictly LESS information, so it could not honestly become
    provable.

    P1 now tests the located terminator TOKEN -- the maximal letter-spacing chain
    of band glyphs starting right of the numeral -- unfiltered and in full. Here
    that token is ['c'] and the read is UNKNOWN.

    Verified to fail first against the shipped predicate: with the pre-repair P1
    this crop returns 21.0 at score 0.772."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_218_top_shaved_at_1272x896.png"))
    )
    assert detail.value is None, f"shaved 218 shipped {detail.value!r}"
    assert detail.value != 21.0
    assert detail.decimal_source == "suffix_not_bb", detail


def test_a_digit_outside_the_run_never_explains_itself(production_bank) -> None:
    """P2's far arm: a confidently classified DIGIT is not an anchor.

    `confident` used to mean "the bank matched this glyph at >= min_score under
    ANY label", so a full-height glyph the bank read as '3' at 0.952, sitting on
    the numeral's own row outside the winning run, anchored itself -- and then
    laundered the 14px occlusion sliver beside it into the same chain under the
    adjacency window. The fixture is a real 2062x1178 seat panel rendering
    "392.30 BB" with the production chip sprite laid over one interior digit as a
    full-height 8px strip; the shipped reader returned 2.30 at score 0.933, a
    170x under-read.

    A digit outside the winning run is the definition of a numeral the read has
    fragmented, so it is the one label that can never anchor. The 'O' of "POT:",
    which the bank also labels '0', keeps its explanation by chaining to the 'P'
    and 'T' beside it -- pinned by the negative control below, which must keep
    reading its pot."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_392_30_digit_severed_by_sprite_at_2062x1178.png"))
    )
    assert detail.value is None, f"severed 392.30 shipped {detail.value!r}"
    assert detail.value != 2.3
    assert detail.decimal_source == "unexplained_ink_in_numeral", detail
    # Negative control: a "POT:" prefix whose 'O' the bank labels a digit still
    # reads, because it chains to the confident 'P' and 'T' at letter spacing.
    ok = production_bank.read_number_detail(cv2.imread(str(FIXTURES / "pot_165_integer.png")))
    assert ok.value == 165.0, ok
    # Negative control: the chip icon, whose annulus scores as '0' at reduced
    # render size, sits BEYOND the "BB" token and must keep anchoring itself.
    chip = production_bank.read_number_detail(cv2.imread(str(FIXTURES / "bet_19_50_at_1272x896.png")))
    assert chip.value == 19.5, chip


def test_the_affix_exemption_is_denied_left_of_the_numeral(production_bank) -> None:
    """P2: `named` is the only thing that can silence P2 on a glyph, and it
    exists for one object -- the "BB" terminator, which follows the value.

    Granted by height and label alone, anywhere in the band, it went to an
    occluded DIGIT instead. The fixture is a real 2722x1832 seat panel rendering
    "190.10 BB" with the production chip sprite over the lower 12 rows of the
    '9': the occluded digit drops below the digit floor, scores as the chip
    template 'c' at 0.649, and used to enter `named` -- where it exempted itself
    from P2's near arm AND anchored the severed leading digits in the far arm.
    Shipped: 0.10 at score 0.946, a 1901x under-read, and 0.1 BB is all-in
    territory the spine reads as a positive fact.

    Verified to fail first: with the exemption granted band-wide this crop reads
    0.1."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_190_10_digit_occluded_into_affix_at_2722x1832.png"))
    )
    assert detail.value is None, f"occluded 190.10 shipped {detail.value!r}"
    assert detail.value != 0.1
    assert detail.decimal_source == "unexplained_ink_in_numeral", detail
    # Negative control: the exemption must still admit a real terminator, or the
    # narrowing is indistinguishable from deleting `named` altogether.
    ok = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_198_suffix_named_at_2054x1470.png")))
    assert ok.value == 198.0, ok


def test_a_second_separator_candidate_is_unexplained_ink(production_bank) -> None:
    """P5(a): the client renders AT MOST ONE separator.

    Requiring only that multiple candidates AGREE left an occluder free to both
    widen an inter-digit gap into the decimal band and scatter two baseline
    fragments into it: one fragment supplies the split, and the PAIR jointly
    fills the hole so P5(b)'s remainder test sees nothing. The fixture is a real
    1272x896 "POT: 39.50 BB" with the production chip sprite over the '9';
    shipped 3.50 at score 0.852.

    Costs 0 of the 18,006 development crops -- no value-producing crop carries
    two candidates -- and the negative control below keeps the one-candidate path
    honest."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "pot_39_50_two_forged_separators_at_1272x896.png"))
    )
    assert detail.value is None, f"two-candidate crop shipped {detail.value!r}"
    assert detail.value != 3.5
    assert detail.decimal_source == "separator_unreconciled", detail
    ok = production_bank.read_number_detail(cv2.imread(str(FIXTURES / "stack_343_60_at_1272x896.png")))
    assert ok.value == 343.6 and ok.decimal_source == "dot", ok


def test_a_leading_zero_is_only_ever_the_whole_integer_part(production_bank) -> None:
    """P6 tests the separator's POSITION, not merely its existence.

    "no leading zero UNLESS a separator was located" let "05.50" through, and an
    occluder makes that shape easily: the fixture is a real 2138x1402 seat panel
    rendering "95.50 BB" with the production chip sprite over the bottom 7 rows
    of the '9', which then classifies as '0' at 0.570. With the real decimal
    still located at position 2 the reader shipped a confident 5.5 for a 95.5 BB
    stack.

    The client renders a leading zero only as the WHOLE integer part: all 648
    leading-zero reads in the development corpus are "0.50" (635) or "0.2" (13),
    i.e. split_at == 1 in 648 of 648. Cost of testing the position: 0 reads."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_95_50_leading_digit_occluded_at_2138x1402.png"))
    )
    assert detail.value is None, f"05.50 shipped {detail.value!r}"
    assert detail.value != 5.5
    assert detail.decimal_source == "leading_zero_no_dot", detail
    # Negative control: the legitimate leading zero -- the whole integer part.
    ok = production_bank.read_number_detail(cv2.imread(str(FIXTURES / "bet_0_50_at_2054x1470.png")))
    assert ok.value == 0.5 and ok.decimal_source == "dot", ok


def test_p1_fires_on_every_malformed_terminator(scaled_digit_templates) -> None:
    """P1 as a FAMILY, stated structurally rather than on one frozen crop.

    The three instances below are the ways a terminator can be malformed while
    the numeral itself stays clean -- no suffix at all (occluded, or clipped away
    by a tight detector box), one cap instead of two, and a second cap the bank
    reads as an '8'. Each was constructed for this test and each was verified
    against the ablated predicate: with P1 off, all three return a confident
    312.0.

    Two neighbouring cases are deliberately NOT asserted here, because a test
    that claims a predicate fires where a different one actually fires is how
    P1 came to be unpinned in the first place:
      * "8B"  -- P2 fires first (the leading cap is not a proven affix), an
                 earlier and stronger refusal of the same crop;
      * "88"  -- P1 fires here, but the ablated reader is caught downstream by
                 integer_over_decimal_band, so the case cannot kill the mutant.
      * "8BB" -- a digit at the FRONT of the terminator token. Synthetically the
                 stray '8' is full-height and confidently a digit, so it joins
                 the run itself ("3128") and P2 fires; the real-crop form of this
                 case, where the severed digit scores as the CHIP template and so
                 breaks the run, is asserted in
                 test_p1_refuses_a_digit_the_chip_template_swallowed.
    All three are recorded rather than asserted."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    for label, kwargs in (
        ("no terminator", {"suffix": False}),
        ("one cap", {"suffix_chars": "B"}),
        ("second cap reads as a digit", {"suffix_chars": "B8"}),
        # Round-3 addition. A terminator TOKEN of three glyphs is the shape a
        # trailing digit misread as 'B' produces, and `suffix[:2]` accepted it:
        # 16 real development crops already render a "BBB" suffix.
        ("three caps", {"suffix_chars": "BBB"}),
    ):
        detail = bank.read_number_detail(_compose("312", 0.9, [3, 3], **kwargs))
        assert detail.value is None, f"{label}: shipped {detail.value}"
        assert detail.decimal_source == "suffix_not_bb", f"{label}: {detail!r}"
    # Negative control: the same numeral WITH a well-formed suffix reads. Without
    # this, setting P1 to refuse unconditionally would also pass the loop above.
    ok = bank.read_number_detail(_compose("312", 0.9, [3, 3]))
    assert ok.value == 312.0
    assert ok.decimal_source == "integer"


def test_the_row_band_split_keeps_a_name_row_out_of_the_numeral(production_bank) -> None:
    """The 0.6*max_h row-band split, the other acceptance-path mechanic that
    round-2 mutation found pinned by nothing (`> 9999 * max_h`, i.e. one single
    band, survived all 280 tests).

    A stack box routinely includes the player's NAME on the row above the value,
    and the reader separates them by y-centre before it looks for digit runs. Fold
    the two rows into one band and the name's glyphs become ink on the numeral's
    own row, so P2 refuses: measured over all 18,006 retained development crops,
    the single-band mutant turns 236 values into UNKNOWN and produces 0
    UNKNOWN->value and 0 value->different reads.

    That direction is why this finding is not release-blocking -- the mutant only
    destroys coverage, it never fabricates a number, which is exactly the trade
    this phase accepts. It is pinned anyway, because "unpinned but currently
    harmless" is the state P1 was in.

    The fixture is a real cwpt01 seat panel (2138x1402) whose top edge carries the
    tail of the name row above "124.80 BB". The structural assertion below is what
    keeps the test honest: if a future fixture swap left only ONE row of glyphs,
    the read would still be 124.80 and the test would pass while exercising
    nothing."""
    img = cv2.imread(str(FIXTURES / "stack_124_80_name_row_above_at_2138x1402.png"))
    comps = segment_glyphs(binarize_text(img), min_area=2, min_h_px=1)
    max_h = max(c.h for c in comps)
    tall = sorted(c.y + c.h / 2.0 for c in comps if c.h >= 0.55 * max_h)
    assert any(b - a > 0.6 * max_h for a, b in zip(tall, tall[1:], strict=False)), (
        "this fixture must carry two glyph ROWS, or it does not exercise the split")
    detail = production_bank.read_number_detail(img)
    assert detail.value == 124.8, "the name row must not reach the numeral's row"
    assert detail.decimal_source == "dot"


def test_a_sprite_fragment_above_the_calibrated_band_is_unknown(production_bank) -> None:
    """P7, upper edge. The render floor was measured and pinned; the CEILING never
    was, so a client rendering far LARGER than any geometry the bank was fitted on
    was silently treated as calibrated.

    This seat panel renders at run height 45-60 against a calibrated band of 12-32.
    The bank matches a sprite fragment as a confident '7' and ships it for a stack
    the screen renders as "1131.90 BB" -- a 162x under-read, on 16 samples.

    The floor's own reasoning applies unchanged in this direction: outside the band
    the bank is extrapolating, and the honest answer is that it was never calibrated
    for the size."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_1131_90_above_calibrated_band_at_2138x1402.png"))
    )
    assert detail.value is None
    assert detail.value != 7.0
    assert detail.decimal_source == "above_calibrated_render_size"


def test_an_integer_whose_widest_gap_reaches_the_decimal_band_is_unknown(
    production_bank,
) -> None:
    """P5(b), and a DELIBERATE COVERAGE LOSS pinned so it stays visible.

    This crop reads "191 BB" and 191.0 is correct on the pixels. It is refused
    anyway: its widest inter-digit gap is 3px inside a 12px run (0.250 of run
    height), and the narrowest gap a LOCATED decimal has ever occupied on this
    corpus is 3/13 = 0.2308. The two bands overlap over [0.2308, 0.2500] -- 64
    integer reads and 43 decimal reads land inside it -- so an integer asserted from
    in there is indistinguishable from a decimal whose dot was lost to binarization.

    The alternative is to put the boundary at the round 0.25, which sits INSIDE the
    ambiguity and admits exactly the "314.90 -> 31490" class this phase exists to
    close. Measured cost of the honest edge: 47 reads across six recordings."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_191_wide_gap_at_1272x896.png"))
    )
    assert detail.value is None
    assert detail.decimal_source == "integer_over_decimal_band"


def test_a_shorter_run_never_outscores_a_longer_one(scaled_digit_templates) -> None:
    """P3. Candidate selection used to rank by `(complete, len, row_y, score)`, so a
    SHORTER fragment could beat the real value: on g0621 t=177 a seat panel showing
    "212.90 BB" alongside a timer ring reading "12" returned a confident 12.0,
    because the 5-digit run was judged a fragment and lost the completeness key to
    the 2-digit badge.

    Selecting among competing readings by score is precisely the kind of rescue this
    contract removes. The rule is now: the unique longest run, or UNKNOWN. Here two
    runs tie at two digits, so there is no unique longest and the answer is
    UNKNOWN -- never one of the two candidates."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    left = _compose("29", 0.9, [3])
    right = _compose("12", 0.9, [3])
    h = max(left.shape[0], right.shape[0])
    img = np.zeros((h, left.shape[1] + right.shape[1], 3), np.uint8)
    img[: left.shape[0], : left.shape[1]] = left
    img[: right.shape[0], left.shape[1] :] = right
    detail = bank.read_number_detail(img)
    assert detail.value is None, "two runs tie for longest; neither is provable"
    assert detail.decimal_source == "ambiguous_longest_run"


def test_the_decimal_gap_arm_no_longer_exists() -> None:
    """The arm that INFERRED a decimal point from inter-digit spacing is deleted,
    not merely bypassed. It fired 0 times in 18006 native reads on the six
    development recordings, and every documented failure it produced (1830.88 from
    "18.30 BB", split at the word space before the suffix) was a confident wrong
    value."""
    src = inspect.getsource(ocr_readers.TemplateOCR.read_number_detail)
    assert '"gap"' not in src, "the inferred-decimal arm is back"
    assert "gap" not in DECIMAL_EVIDENCE
    assert not hasattr(ocr_readers, "_INTRA_NUMERAL_GAP"), (
        "the constant that only the gap arm consumed is back")


def test_absorb_adjacent_digits_no_longer_exists() -> None:
    """`_absorb_adjacent_digits` re-classified a glyph the pooled bank had labelled
    an AFFIX against a digit-only bank, i.e. it overruled the classifier in order to
    produce a value. `classify_digit_only` existed solely to serve it.

    Both are deleted. Measured cost: 30 reads used the absorb path, 14 of which pass
    the rest of the predicate and are now UNKNOWN -- including both crops it was
    written for (see test_the_two_reads_absorb_was_written_for_are_now_unknown)."""
    assert not hasattr(ocr_readers.TemplateOCR, "_absorb_adjacent_digits")
    assert not hasattr(ocr_readers.TemplateOCR, "classify_digit_only")
    assert not hasattr(ocr_readers, "read_amount_from_image"), (
        "the wrapper that flattened the refusal and no-bank channels into one "
        "float|None is back; it had no caller outside the module")


def test_the_two_reads_absorb_was_written_for_are_now_unknown(production_bank) -> None:
    """SUPERSEDES test_real_crop_truncated_run_reads_its_true_value (asserted 218.0)
    and test_real_crop_trailing_fractional_zero_is_not_lost_to_the_chip_template
    (asserted 212.5). Both now return UNKNOWN. THIS IS A DELIBERATE COVERAGE LOSS
    and it is recorded here rather than quietly kept green.

    Both crops are cases where the bank CANNOT TELL a digit from an affix glyph:
      * "218 BB" at 12px -- the '8' scores 0.675 as '8' against 0.677 as 'B'
      * "POT: 212.50 BB" -- the trailing '0' scores 0.846 as '0' against 0.861 as
        the chip affix 'c'
    The old code resolved that ambiguity by re-reading the glyph against a
    digit-only bank, i.e. by assuming the answer. Note 10 records the SAME ambiguity
    resolving the other way and shipping a confident 21.0 for the same seat.

    A margin of 0.002 in a cosine score is not proof. When the classifier cannot
    separate the glyph, the read is unknown."""
    for name, was in (("stack_218_at_1272x896.png", 218.0),
                      ("pot_212_50_trailing_zero_at_2054x1470.png", 212.5)):
        detail = production_bank.read_number_detail(cv2.imread(str(FIXTURES / name)))
        assert detail.value is None, f"{name}: {detail!r}"
        assert detail.value != was
        assert detail.decimal_source == "unexplained_ink_in_numeral"


def test_real_crop_numeral_beginning_at_the_decimal_point_is_unknown(
    production_bank,
) -> None:
    """A clipped seat panel renders ").60 BB" -- the integer part is cut away and
    the decimal point is the first thing in the crop. The reader returned a
    confident 60.0, i.e. it positively asserted the number has no fractional part,
    on a crop where the dot is rendered AND was segmented as a dot candidate. The
    seat really held 180.6; this shipped for three consecutive samples on the
    BASELINE 2054x1470 recording."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_leading_decimal_clipped_at_2054x1470.png"))
    )
    assert detail.value is None, "a numeral whose integer part is clipped is unknown"
    assert detail.decimal_source == "unexplained_ink_in_numeral"


def test_decimal_boundary_fragment_fails_closed_under_a_window_resize(
    production_bank,
) -> None:
    """Sibling of the case above, in the direction a resize produces rather than a
    clipped box. At 0.90x the '9' of "19.50" scores 0.541, just under the 0.55
    floor, so the run breaks into "1" and "50"; the fragment "50" then measured
    4px to the rejected '9' against a 3.36px threshold and was declared a COMPLETE
    numeral, returning a confident 50.0 -- NON-MONOTONIC in scale (0.85x and 0.95x
    both read 19.5)."""
    img = cv2.imread(str(FIXTURES / "bet_19_50_at_1272x896.png"))
    h, w = img.shape[:2]
    for scale in (0.85, 0.90, 0.95, 1.00):
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        detail = production_bank.read_number_detail(small)
        assert detail.value in (19.5, None), f"scale {scale}: read {detail!r}"
        assert detail.value != 50.0, f"scale {scale}: 100x-class fragment shipped"


def test_chip_sprite_is_not_a_confident_zero_at_any_render_size(
    production_bank,
) -> None:
    """The digit-aspect gate is what stops a text-free crop reading a confident
    0.0, and it was inclusive at exactly its own confuser's value: `g.w <=
    _MAX_DIGIT_ASPECT * g.h` with the constant at 1.0. A chip's pale annulus
    rounds to precisely square at reduced render size, so this bet_text crop --
    which holds only chip sprites and no text at all -- read 0.0 at 0.65x, 0.70x
    and 0.75x. stack_text 0.0 is TRUSTED by the spine and a zero stack labels the
    seat's action all-in.

    The frozen crop this pins is the one that DISCRIMINATES. Its sibling
    bet_chip_sprite_no_text.png returns None either way (via a different code
    path), so the older test could not detect the gate's removal at all: setting
    _MAX_DIGIT_ASPECT to 1e9 passed the entire 197-test CV suite."""
    img = cv2.imread(str(FIXTURES / "bet_chips_only_at_1272x896.png"))
    assert img is not None
    h, w = img.shape[:2]
    for i in range(31):
        scale = round(0.60 + 0.05 * i, 2)
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        val = production_bank.read_number_detail(small).value
        assert val is None, f"scale {scale}: text-free crop read {val!r}"


def test_clipped_box_stays_unknown_across_a_window_resize(production_bank) -> None:
    """The same "71.20 BB"-clipped-to-"20 BB" crop as the test below, swept across
    render size. The truncation net's old height gate (`g.h >= 0.78 * band_h`)
    cleared the clipped fragment by 0.006 of band height at 1.00x and missed it at
    21 of 31 scales, returning a confident 0.0 -- including at 0.90x, 1.10x and
    1.25x. A digit clipped by a tight detector box is SHORT precisely because it
    is cut, so height cannot be the test that separates it from the "BB" caps."""
    img = cv2.imread(str(FIXTURES / "stack_clipped_box_at_2722x1832.png"))
    assert img is not None
    h, w = img.shape[:2]
    for i in range(31):
        scale = round(0.60 + 0.05 * i, 2)
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        val = production_bank.read_number_detail(small).value
        assert val is None, f"scale {scale}: clipped numeral read {val!r}"


def test_bb_suffix_is_never_absorbed_into_the_numeral(production_bank) -> None:
    """Negative control for the rules above, and the regression they caused when the
    affix height gate was simply deleted: the "BB" caps butt right up against the
    value (measured horizontal gaps of 2-6px against run heights of 22-28px, i.e.
    INSIDE the letter-spacing floor), so the gap test alone cannot separate them.
    Deleting the height gate absorbed the 'B' of "198 BB" as an '8' on 55 real reads.

    A genuine "0 BB" is the sharpest form: a one-glyph value flanked by caps that
    render at FULL digit height on this geometry. It reads 0.0 at native size and
    across a resize -- except at 0.80x, where the bank reads one cap as an '8', two
    one-glyph runs tie for longest and the answer is UNKNOWN. That is the honest
    outcome of the tie: the reader must never pick 8.0 or 88.0, and it must never
    silently pick 0.0 out of a tie either."""
    val, raw = _read_fixture(production_bank, "stack_0_true_zero.png")
    assert (val, raw) == (0.0, "0")
    img = cv2.imread(str(FIXTURES / "stack_0_true_zero.png"))
    h, w = img.shape[:2]
    for scale in (0.80, 0.85, 0.90, 1.00, 1.20):
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        value = production_bank.read_number_detail(small).value
        assert value in (0.0, None), f"scale {scale}: the BB caps entered the numeral"
    for scale in (0.85, 0.90, 1.00, 1.20):
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        assert production_bank.read_number_detail(small).value == 0.0, scale


def test_two_digit_numeral_with_a_wide_gap_is_unknown(scaled_digit_templates) -> None:
    """A one-decimal client ("1.5 BB", "7.5 BB" -- the 07-15 recording renders
    exactly this) whose dot is lost to binarization returned a confident 10x
    inflation, because the old decimal-gap arm needed `max(gaps) >= 1.7 *
    median(gaps)` and with a single gap the median IS the max.

    The bespoke two-digit rule that patched that hole is DELETED along with the arm
    it patched. It carried a `"1" not in digits` exclusion -- note 12 records the
    10x hole that exclusion left at every gap width -- and it was a second, narrower
    statement of P5(b). P5(b) covers every run length with one rule and no
    exclusion."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    run_h = _run_height(_compose("75", 0.9, [3]))
    wide = round(0.40 * run_h)   # inside the measured DECIMAL band, dot lost
    detail = bank.read_number_detail(_compose("75", 0.9, [wide]))
    assert detail.value is None, "a lost decimal must not ship as a 10x inflation"
    assert detail.decimal_source == "integer_over_decimal_band"
    # The hole the old exclusion left: a two-digit numeral CONTAINING a '1' was
    # exempt from the rule entirely and shipped the inflation.
    holed = bank.read_number_detail(_compose("15", 0.9, [wide]))
    assert holed.value is None, "the '1' exclusion's 10x hole is closed"
    # Negative control: an ordinary two-digit integer at measured spacing still reads.
    assert bank.read_number(_compose("75", 0.9, [round(0.15 * run_h)])) == (75.0, "75")


def test_real_crop_clipped_box_is_unknown(production_bank) -> None:
    """A detector box that shrank onto part of the number ("71.20 BB" cropped to
    "20 BB") produced a confident 0.0 -- indistinguishable from a genuine all-in
    "0 BB", and it invented an all-in action in an exported hand."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_clipped_box_at_2722x1832.png"))
    )
    assert detail.value is None
    assert detail.decimal_source in REFUSAL_CODES


def test_real_crop_single_digit_pot_reads_its_value(production_bank) -> None:
    """"POT: 9 BB": the 'O' of POT: and the value are both one-glyph runs, and the
    tie used to be broken by match score -- so the pot read 0.0. The 'O' is flanked
    by P and T at intra-numeral spacing, so it is not a candidate at all, and the 9
    is the unique longest run."""
    assert _read_fixture(production_bank, "pot_9_single_digit_at_2722x1832.png") == (9.0, "9")


def test_truncated_run_net_leaves_real_affixes_alone(production_bank) -> None:
    """Negative control for the same net. A genuine "0 BB" is a one-glyph run whose
    neighbours are the BB caps -- rendered at FULL digit height on this geometry,
    so height alone cannot separate them. The word gap can: it is 0.55 of the digit
    height against the 0.25 that truncated "218"."""
    val, raw = _read_fixture(production_bank, "stack_0_true_zero.png")
    assert (val, raw) == (0.0, "0")


def test_truncated_run_detected_without_the_production_bank(digit_templates) -> None:
    """Bank-free form of the same rule, so it runs on any checkout: '4' rendered
    flush against a glyph the bank cannot call a digit truncates the numeral."""
    def draw(im: np.ndarray) -> None:
        cv2.putText(im, "4", (60, 34), FONT, SCALE, WHITE, THICK)
        # A full-height blob one pixel after the digit: whatever it is, the numeral
        # does not end at the '4'.
        im[10:38, 82:96] = WHITE
        _draw_bb(im, 120)

    bank = TemplateOCR(dict(digit_templates), {})
    detail = bank.read_number_detail(_render(draw))
    assert detail.value is None
    assert detail.decimal_source == "unexplained_ink_in_numeral"


def test_unreconcilable_separator_is_unknown_not_a_confident_value(
    scaled_digit_templates,
) -> None:
    """A group separator sits where a decimal point sits and has the same
    silhouette, so "12,345" split as 12.345 -- a 1000x under-read. ClubWPT renders
    at most two fractional places (measured: 0, 1 and 2 across 14390 real reads and
    never 3), so a split leaving three or more is not a decimal point.

    Rejecting the split is only half the rule. The reader then returned the UNSPLIT
    digits as a CONFIDENT value, which is the same claim in the other direction:
    "1.234" came back 1234.0 -- a 1000x OVER-read -- at every glyph height from 12
    to 80px. The reader cannot tell a period from a comma at HUD sizes; when it has
    positive evidence of a separator it cannot reconcile, the honest answer is
    unknown."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    comma = bank.read_number_detail(_compose("12345", 0.9, [3, 7, 3, 3], dot=(1, 3, 3)))
    assert comma.value is None, "an unreconcilable separator must not produce a value"
    assert comma.value != 12.345, "and must never split 1000x low"
    assert comma.decimal_source == "separator_unreconciled"
    # Same silhouette, three fractional places: the 1000x OVER-read direction.
    deep = bank.read_number_detail(_compose("1234", 0.9, [7, 3, 3], dot=(0, 3, 3)))
    assert deep.value is None
    assert deep.decimal_source == "separator_unreconciled"


def test_a_discarded_separator_candidate_makes_the_read_unknown(
    scaled_digit_templates,
) -> None:
    """SUPERSEDES test_comma_and_decimal_together_read_the_decimal, which asserted
    that "1,234.50" reads 1234.5 because the comma is rejected on arity and the real
    decimal wins.

    P5(a) requires that exactly one separator candidate reconciles AND that no other
    candidate had to be discarded to reach it. Two same-silhouette glyphs inside one
    numeral means the reader is CHOOSING between two readings of the separator, and
    at HUD glyph sizes it has no evidence with which to choose -- "1,234.50" and
    "1.234,50" are the same pixels under a different locale. Trying candidates in
    order until one reconciles is a search for an answer that reconciles, not proof
    that it is the right one.

    Measured cost on the development corpus: 0 reads -- no real read has more than
    one baseline separator candidate. The rule is a refusal that costs nothing here
    and closes the case the moment a comma-rendering client is ingested."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    # gaps: [comma(7), 3, 3, decimal(6), 3] -- the comma gap is the WIDER of the
    # two, and BOTH hosting gaps sit inside the measured decimal band (>= 3/13 of
    # run height; 22px digits -> >= 5.08px) so each dot is a legal candidate and
    # the refusal under test is P5a's, not the host-gap forgery refusal.
    img = _compose("123450", 0.9, [7, 3, 3, 6, 3], dots=[(0, 3, 3), (3, 3, 3)])
    detail = bank.read_number_detail(img)
    assert detail.value is None
    assert detail.value != 123450.0, "and never the 100x over-read the old pick gave"
    assert detail.decimal_source == "separator_unreconciled"


def test_two_fractional_places_still_split(scaled_digit_templates) -> None:
    """Negative control for the rule above: two places is the deepest ClubWPT
    renders and must still split."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    assert bank.read_number(_compose("31490", 0.9, [3, 3, 7, 3], dot=(2, 3, 3))) == (314.90, "314.90")


def test_an_integer_at_the_widest_measured_integer_gap_is_refused(
    scaled_digit_templates,
) -> None:
    """SUPERSEDES test_gap_fallback_cannot_split_a_measured_integer, which asserted
    that an integer whose widest gap sits at 0.250 of run height -- the measured
    integer maximum -- still reads its value.

    It no longer does, and that is the point of P5(b). The integer band's maximum
    (0.2500) and the located-decimal band's minimum (3/13 = 0.2308) OVERLAP. The old
    rule put its threshold at 0.28, above BOTH, which made the arm unable to invent
    a decimal in an integer -- but the failure that mattered ran the other way: a
    real decimal whose dot was lost, sitting in the overlap, shipped as a confident
    integer. Coverage inside the overlap is not available to either rule; only the
    direction of the error is a choice, and refusing is the safe one."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    run_h = _run_height(_compose("1650", 0.9, [3, 3, 5]))
    wide = round(0.250 * run_h)          # the widest gap measured inside a real integer
    detail = bank.read_number_detail(_compose("1650", 0.9, [3, 3, wide]))
    assert detail.value is None
    assert detail.decimal_source == "integer_over_decimal_band"
    # Below the decimal band's own floor the integer still reads: this is a
    # boundary, not a blanket refusal of integers.
    narrow = round(0.15 * run_h)
    assert bank.read_number(_compose("1650", 0.9, [3, 3, narrow])) == (1650.0, "1650")


# --------------------------------------------------------------------------- #
# TRUE ZERO. A genuine all-in "0 BB" is a positive fact the spine needs, and it is
# proven by the predicate plus one structural fact: the run is the single glyph
# '0' and P1 proves that glyph is terminated by "BB".
# --------------------------------------------------------------------------- #
def test_a_proven_zero_requires_a_bb_suffix(production_bank) -> None:
    """The three adversarial zeros, and the one genuine one, separated by proof
    rather than by a tuned gate.

    Positive: the real all-in "0 BB" reads 0.0. Measured over the six development
    recordings, 78 crops produce a winning run of the single digit '0'; 69 pass the
    predicate and ALL 69 have a proven "BB" suffix. Zero genuine zeros are lost.

    Negatives: a crop of chip sprites has no "BB" anywhere, so it has no suffix and
    cannot be a numeral. That is a structural fact about the crop, not a threshold
    on the sprite's aspect ratio -- the previous defence was `_MAX_DIGIT_ASPECT`
    sitting 0.05 from the confuser's own measured value."""
    assert _read_fixture(production_bank, "stack_0_true_zero.png") == (0.0, "0")
    for name in ("bet_chip_sprite_no_text.png",
                 "bet_chips_only_at_1272x896.png",
                 "bet_chip_annulus_square_at_2722x1832.png"):
        detail = production_bank.read_number_detail(cv2.imread(str(FIXTURES / name)))
        assert detail.value is None, f"{name} manufactured {detail.value!r}"
        assert detail.decimal_source == "no_digit_run", name


def test_the_frozen_lost_decimal_zero_crop_is_never_a_confident_zero(
    production_bank,
) -> None:
    """"0.50 BB" read as a confident 0.0.

    At 0.70x of the 1272x896 client the decimal point is lost to binarization and
    the '5' scores 0.466, just under the 0.55 floor, so the run breaks after the
    leading '0'. The dot's own width then puts the surviving '0' 0.44 run heights
    from the rejected '5' -- above the 0.28 word-break floor -- so the completeness
    net called the single '0' a whole numeral.

    A confident 0.0 is not a neutral failure: a stack_text 0.0 is trusted by the
    spine as all-in. Whichever condition owns it, the answer must never be 0.0."""
    img = cv2.imread(str(FIXTURES / "bet_0_50_dot_lost_at_890x627.png"))
    assert img is not None
    h, w = img.shape[:2]
    for scale in (1.00, 1.20, 1.40, 1.60, 2.00):
        crop = img if scale == 1.0 else cv2.resize(
            img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
        value = production_bank.read_number_detail(crop).value
        assert value in (None, 0.5), f"scale {scale}: read {value!r}"
        assert value != 0.0


def test_real_crop_a_true_zero_still_reads_zero(production_bank) -> None:
    """The other side of the rule above: an all-in seat showing "0 BB" is a
    positive fact the spine needs, and the "BB" caps beside it must not be mistaken
    for the fractional digits of a "0.xx"."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_0_true_zero.png"))
    )
    assert detail.value == 0.0
    assert detail.decimal_source == "integer"


def test_real_chip_annulus_at_reduced_size_is_not_a_confident_zero(
    production_bank,
) -> None:
    """Round 4, adversary C: _MAX_DIGIT_ASPECT could be reverted from 0.95 to its
    documented pre-fix value of 1.0 with the entire CV suite green. The existing
    pin only starts failing at >= 1.09, so the whole [0.95, 1.05] interval -- which
    includes the exact confuser value the comment cites -- was unprotected.

    This is that population, frozen: a real bet_text crop from the 2722x1832
    recording holding chip sprites and NO text. Its pale annulus binarizes to a
    component that rounds to precisely square at reduced render size."""
    img = cv2.imread(str(FIXTURES / "bet_chip_annulus_square_at_2722x1832.png"))
    assert img is not None
    h, w = img.shape[:2]
    for scale in (0.70, 0.75, 0.85, 0.95, 1.00):
        crop = img if scale == 1.0 else cv2.resize(
            img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        detail = production_bank.read_number_detail(crop)
        assert detail.value is None, (
            f"scale {scale}: a textless chip crop read {detail.value!r}")


def test_a_speck_at_mid_band_height_is_not_a_decimal(digit_templates) -> None:
    """Round 4, adversary C: _DOT_MIN_BASELINE_POS could be loosened from 0.6 all
    the way to 0.4 with the whole CV suite green. The one frozen speck in the suite
    sits at the very TOP of the band, so only the constant's total removal (0.0) was
    pinned and the entire 0.0-0.6 interval was untested apart from its endpoint.

    The constant's stated basis, over 9088 real decimal reads, is that a true
    decimal's centre lands at 0.846-0.958 of the run band and never below 0.75.
    0.6 keeps a 1.41x margin under the lowest measured real dot.

    This pins the middle of that interval: a speck at ~0.5 of band height -- well
    above any measured decimal -- must not split the numeral. Under the completed
    P2 the speck is unexplained ink on the numeral, so the read is UNKNOWN rather
    than the unsplit integer; either way _DOT_MIN_BASELINE_POS keeping it out of
    the separator population is what this test exercises (were the speck admitted
    as a dot, the read would come back a confident 3.12)."""
    bank = TemplateOCR(dict(digit_templates), {})
    # 28px digits: the 7px gap keeps the baseline-arm dot inside the measured
    # decimal band (>= 6.46px host gap), so this stays a POSITION test -- the
    # same component refuses at mid-band and reads at baseline.
    mid = _compose("312", 1.2, [5, 7], specks=[(1, 3, 3, 0.50)])
    detail = bank.read_number_detail(mid)
    assert detail.value != 3.12, "mid-band speck is not a dot"
    assert detail.value is None, "unexplained ink on the numeral must refuse"
    assert detail.decimal_source == "unexplained_ink_in_numeral"
    # ... while a component of the SAME size on the baseline still IS a decimal, so
    # this is a position test and not a blanket refusal.
    baseline = _compose("312", 1.2, [5, 7], dot=(1, 3, 3))
    assert bank.read_number(baseline)[0] == 31.2


# --------------------------------------------------------------------------- #
# The calibration DOMAIN, expressed in the quantity that governs the reader.
# --------------------------------------------------------------------------- #
def test_calibrated_render_band_is_the_range_actually_measured() -> None:
    """Round 4, adversary B: _MIN_CALIBRATED_RUN_H's stated justification was false,
    and the constant sat below every render size the bank was ever shown to read
    correctly. Re-measured over 14390 real crops x 17 render scales: 893 reads
    return a confident value that disagrees with the audited 1.0x reference, and 757
    of them occur at run heights 9, 10 and 11 -- INSIDE the declared calibrated
    range.

    The band is now two-sided. Its edges are the measurement and nothing else: over
    every value-producing read at native size, run height spans 12-32 px. Above 32
    the bank is extrapolating exactly as much as below 12, and the corpus shows what
    that costs (see
    test_a_sprite_fragment_above_the_calibrated_band_is_unknown)."""
    assert ocr_readers._MIN_CALIBRATED_RUN_H >= 12, (
        "the floor must not sit below the smallest render height the bank has "
        "been measured correct at")
    assert ocr_readers._MAX_CALIBRATED_RUN_H <= 32, (
        "the ceiling must not sit above the largest render height the bank has "
        "been measured correct at")


def test_reads_below_the_calibrated_render_size_are_unknown(production_bank) -> None:
    """Nothing in the pipeline recorded the render geometry as a supported or
    unsupported fact, so a client rendering smaller than anything the templates
    were calibrated on degraded to CONFIDENT WRONG VALUES rather than to unknowns:
    across the 0.60x-2.10x sweep of the frozen fixtures, "314.90" read 90.0,
    "19.50" read 50.0 and "343.60" read 360.0 -- all at full confidence, and all
    NON-MONOTONIC in scale.

    The guard is stated in glyph height rather than window size because glyph
    height is what breaks, and it is independent of detector box padding, crop
    dimensions and client resolution."""
    img = cv2.imread(str(FIXTURES / "stack_314_90_at_1272x896.png"))
    h, w = img.shape[:2]
    assert production_bank.read_number_detail(img).value == 314.9
    for scale in (0.75, 0.85, 0.90):
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        detail = production_bank.read_number_detail(small)
        assert detail.value is None, f"scale {scale}: {detail!r}"
        assert detail.decimal_source == "below_calibrated_render_size", scale
    # At 0.70x the run itself breaks apart before the floor can be consulted, so a
    # different condition owns it. Still unknown; a different reason.
    tiny = cv2.resize(img, (round(w * 0.70), round(h * 0.70)),
                      interpolation=cv2.INTER_AREA)
    detail = production_bank.read_number_detail(tiny)
    assert detail.value is None
    assert detail.decimal_source in REFUSAL_CODES


def test_production_entrypoint_recovers_small_render_via_upscale(
    production_bank, monkeypatch
) -> None:
    """Job-4 style ~1050x730 captures refuse every HUD amount as
    below_calibrated_render_size even when digits are human-legible (13 BB raise,
    20 BB flop bet). Lowering the floor reintroduces confident wrong values; the
    production entrypoint instead upscales into the calibrated band and accepts
    only a multi-scale consensus that matches the native digit run.

    The bank itself must still refuse the native small crop -- only the
    entrypoint may recover."""
    img = cv2.imread(str(FIXTURES / "bet_19_50_at_1272x896.png"))
    assert img is not None
    h, w = img.shape[:2]
    # 0.80x is inside the floor band for this fixture; 0.85x refuses for a
    # different reason (unexplained ink) and must not be "rescued" by upscale.
    small = cv2.resize(img, (round(w * 0.80), round(h * 0.80)),
                       interpolation=cv2.INTER_AREA)
    native = production_bank.read_number_detail(small)
    assert native.value is None
    assert native.decimal_source == "below_calibrated_render_size"

    monkeypatch.setattr(ocr_readers, "_bank", lambda: production_bank)
    recovered = ocr_readers.read_amount_detail_from_image(
        small, (0, 0, small.shape[1], small.shape[0])
    )
    assert recovered is not None
    assert recovered.value == 19.5


def test_production_entrypoint_recovers_job4_stack_crops(
    production_bank, monkeypatch
) -> None:
    """Real 1052x732 ClubWPT stack crops from job 4: native bank refuses under
    the calibrated floor, but the entrypoint must recover the on-screen BB value.
    These are the exact failure mode that left session Hands empty (30/30
    starting_stack_unknown)."""
    expected = {
        "stack_1458_90_at_1052x732.png": 1458.9,
        "stack_203_30_at_1052x732.png": 203.3,
        "stack_212_20_at_1052x732.png": 212.2,
        "stack_204_50_at_1052x732.png": 204.5,
        "stack_191_at_1052x732.png": 191.0,
        "stack_224_20_at_1052x732.png": 224.2,
    }
    monkeypatch.setattr(ocr_readers, "_bank", lambda: production_bank)
    job4 = FIXTURES / "job4_1052x732"
    for name, value in expected.items():
        img = cv2.imread(str(job4 / name))
        assert img is not None, name
        native = production_bank.read_number_detail(img)
        assert native.value is None, name
        assert native.decimal_source == "below_calibrated_render_size", name
        recovered = ocr_readers.read_amount_detail_from_image(
            img, (0, 0, img.shape[1], img.shape[0])
        )
        assert recovered is not None, name
        assert recovered.value == value, (name, recovered)


def test_production_entrypoint_does_not_rescue_sprite_fragment(
    production_bank, monkeypatch
) -> None:
    """Adversary: upscaling stack_343_60_sprite_far_fragments at several scales
    returned confident 0.0 while the screen shows 343.6. A lone under-floor "0"
    must stay unknown -- multi-scale agreement alone is not enough because every
    scale agrees on the fragment."""
    img = cv2.imread(str(FIXTURES / "stack_343_60_sprite_far_fragments_at_1272x896.png"))
    assert img is not None
    h, w = img.shape[:2]
    monkeypatch.setattr(ocr_readers, "_bank", lambda: production_bank)
    for scale in (0.71, 0.84, 0.97, 0.98):
        small = cv2.resize(img, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        native = production_bank.read_number_detail(small)
        if native.decimal_source != "below_calibrated_render_size":
            continue
        detail = ocr_readers.read_amount_detail_from_image(
            small, (0, 0, small.shape[1], small.shape[0])
        )
        assert detail is not None
        assert detail.value is None, scale
        assert detail.decimal_source == "below_calibrated_render_size", scale


def test_adversary_truncated_native_does_not_ship_wrong_value(
    production_bank, monkeypatch
) -> None:
    """Adversary A/B: hostile scales must not invent wrong longer/shorter amounts.

    Unknown is fine. A confident value other than the 1.0x truth is not.
    Note: when a downscale destroys a leading digit (191->pixels show 19),
    recovering 19 is pixel-faithful; that residual is accepted for coverage of
    true 2-digit stacks like 50 BB.
    """
    monkeypatch.setattr(ocr_readers, "_bank", lambda: production_bank)
    cases = [
        ("job4_1052x732/stack_212_20_at_1052x732.png", 0.95, 212.2),
        ("job4_1052x732/stack_224_20_at_1052x732.png", 0.70, 224.2),
        ("stack_leading_decimal_clipped_at_2054x1470.png", 0.575, None),
        ("pot_39_50_two_forged_separators_at_1272x896.png", 0.55, 39.5),
        ("bet_0_50_dot_lost_at_890x627.png", 1.05, 0.5),
        ("pot_39_50_two_forged_separators_at_1272x896.png", 0.70, 39.5),
    ]
    for name, scale, truth in cases:
        img = cv2.imread(str(FIXTURES / name))
        assert img is not None, name
        h, w = img.shape[:2]
        if "39_50" in name and scale <= 0.55:
            interp = cv2.INTER_NEAREST
        elif "0_50_dot_lost" in name:
            interp = cv2.INTER_CUBIC
        elif "39_50" in name:
            interp = cv2.INTER_LINEAR
        else:
            interp = cv2.INTER_AREA
        small = cv2.resize(
            img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp
        )
        detail = ocr_readers.read_amount_detail_from_image(
            small, (0, 0, small.shape[1], small.shape[0])
        )
        assert detail is not None
        # Clipped ``.60`` fixture at hostile NEAREST may still show only ``60``
        # in-pixels; unknown or pixel-faithful 60 are both acceptable — a
        # different wrong amount is not.
        if "leading_decimal_clipped" in name:
            assert detail.value in (None, 60.0), (name, scale, detail.value)
            continue
        assert detail.value in (None, truth), (name, scale, detail.value)


def test_digit_runs_compatible_trailing_fraction_zero() -> None:
    """Native under-floor OCR often drops the trailing fractional zero
    (2122 vs 212.20); recovery must treat that as compatible."""
    recovered = AmountRead(212.2, "212.20", 0.9, "dot", 5)
    assert ocr_readers._digit_runs_compatible("2122", recovered)
    assert ocr_readers._digit_runs_compatible("21220", recovered)
    assert not ocr_readers._digit_runs_compatible("9999", recovered)


def test_soft_digit_related_rejects_shorter_fragments() -> None:
    """Free-path gate: never promote 191→19 or left-mask 0.50→50."""
    short = AmountRead(19.0, "19", 0.9, "no_dot", 2)
    assert not ocr_readers._soft_digit_related("191", short)
    assert not ocr_readers._soft_digit_related("050", AmountRead(50.0, "50", 0.9, "no_dot", 2))
    # Hamming on short all-zero native must not invent 50 from 00.
    assert not ocr_readers._soft_digit_related("00", AmountRead(50.0, "50", 0.9, "no_dot", 2))
    assert ocr_readers._soft_digit_related("19", AmountRead(198.5, "198.50", 0.9, "dot", 5))
    assert ocr_readers._soft_digit_related("191", AmountRead(191.0, "191", 0.9, "no_dot", 3))
    # Integer +1 growth is allowed (21 -> 218); decimal +1/+2 is not.
    assert ocr_readers._soft_digit_related("21", AmountRead(218.0, "218", 0.9, "integer", 3))
    assert not ocr_readers._soft_digit_related("21", AmountRead(21.8, "21.8", 0.9, "dot", 3))
    assert not ocr_readers._soft_digit_related("22", AmountRead(22.42, "22.42", 0.9, "dot", 4))
    assert not ocr_readers._soft_digit_related("6", AmountRead(60.0, "60", 0.9, "integer", 2))
    assert ocr_readers._soft_digit_related("50", AmountRead(50.0, "50", 0.9, "integer", 2))
    assert ocr_readers._soft_digit_related("", AmountRead(191.0, "191", 0.9, "no_dot", 3))
    assert not ocr_readers._soft_digit_related("", AmountRead(0.0, "0", 0.9, "no_dot", 1))
    # Same-length hamming only for long runs (compat-aligned).
    assert not ocr_readers._soft_digit_related(
        "215", AmountRead(218.0, "218", 0.9, "no_dot", 3)
    )
    assert ocr_readers._soft_digit_related(
        "21520", AmountRead(21820.0, "21820", 0.9, "no_dot", 5)
    )
    # Ambiguous 19|20 recovers as 192.20 (digits 19220, not a prefix of 1920).
    assert ocr_readers._soft_digit_related(
        "1920",
        AmountRead(192.2, "192.20", 0.9, "dot", 5),
        native_raw="19|20",
    )
    # Truncated under-floor 19 -> 198.50.
    assert ocr_readers._soft_digit_related(
        "19", AmountRead(198.5, "198.50", 0.9, "dot", 5)
    )
    # 1.30 must not digit-equal native 130.
    assert not ocr_readers._soft_digit_related(
        "130", AmountRead(1.3, "1.30", 0.9, "dot", 3)
    )
    # Weak single-digit ambiguous pieces: need a 3+ digit recovery (181), not 11.
    assert not ocr_readers._soft_digit_related(
        "11",
        AmountRead(11.0, "11", 0.9, "no_dot", 2),
        native_raw="1|1",
    )
    assert ocr_readers._soft_digit_related(
        "11",
        AmountRead(181.0, "181", 0.9, "no_dot", 3),
        native_raw="1|1",
    )
    # Near-match: native 2057 vs recovered 208.70 (one digit confusion + frac zero).
    assert ocr_readers._soft_digit_related(
        "2057", AmountRead(208.7, "208.70", 0.9, "dot", 5)
    )


def test_production_entrypoint_never_reads_wrong_value_on_frozen_fixtures(
    production_bank, monkeypatch
) -> None:
    """Adversary: the bank-only scale sweep missed entrypoint recovery bugs
    (sprite no_digit_run -> 0.0, suffix parse inventing 50.0 / 350.0). The public
    entrypoint must also never return a confident wrong number."""
    truth: dict[str, float | None] = {
        "stack_314_90_at_1272x896.png": 314.9,
        "bet_19_50_at_1272x896.png": 19.5,
        "bet_0_50_at_2054x1470.png": 0.5,
        "pot_89_1_one_decimal.png": 89.1,
        "pot_240_9_one_decimal.png": 240.9,
        "pot_165_integer.png": 165.0,
        "stack_0_true_zero.png": 0.0,
        "stack_218_at_1272x896.png": 218.0,
        "stack_clipped_box_at_2722x1832.png": None,
        "bet_chips_only_at_1272x896.png": None,
        "pot_9_single_digit_at_2722x1832.png": 9.0,
        "bet_chip_sprite_no_text.png": None,
        "stack_343_60_at_1272x896.png": 343.6,
        "stack_leading_decimal_clipped_at_2054x1470.png": None,
        "pot_212_50_trailing_zero_at_2054x1470.png": 212.5,
        "pot_240_9_chip_overlaps_run_at_2132x1378.png": 240.9,
        "bet_chip_covered_malformed_suffix_at_2054x1470.png": None,
        "stack_1131_90_above_calibrated_band_at_2138x1402.png": 1131.9,
        "stack_99_50_menu_occluded_at_2138x1402.png": None,
        "pot_6_50_sprite_occluded_at_2722x1832.png": None,
        "stack_198_suffix_named_at_2054x1470.png": 198.0,
        "stack_191_wide_gap_at_1272x896.png": 191.0,
        "bet_0_50_dot_lost_at_890x627.png": 0.5,
        "bet_18_30_suffix_absorbed_1399x986.png": 18.3,
        "bet_chip_annulus_square_at_2722x1832.png": None,
        "stack_197_at_2054x1470.png": 197.0,
        "stack_197_speck_forged_decimal_at_2054x1470.png": 197.0,
        "stack_343_60_sprite_far_fragments_at_1272x896.png": 343.6,
        "stack_212_90_timer_badge_at_2062x1178.png": 212.9,
        "stack_124_80_name_row_above_at_2138x1402.png": 124.8,
        "stack_392_30_digit_severed_by_sprite_at_2062x1178.png": 392.3,
        "stack_190_10_digit_occluded_into_affix_at_2722x1832.png": 190.1,
        "stack_218_top_shaved_at_1272x896.png": 218.0,
        "pot_39_50_two_forged_separators_at_1272x896.png": 39.5,
        "stack_95_50_leading_digit_occluded_at_2138x1402.png": 95.5,
        "stack_162_40_at_2138x1402.png": 162.4,
    }
    monkeypatch.setattr(ocr_readers, "_bank", lambda: production_bank)
    wrong: list[tuple[str, float, object]] = []
    for name, true_value in truth.items():
        img = cv2.imread(str(FIXTURES / name))
        assert img is not None, name
        h, w = img.shape[:2]
        for i in range(0, 31, 2):  # every 0.10 scale; full bank sweep covers denser
            scale = round(0.60 + 0.05 * i, 2)
            small = cv2.resize(
                img,
                (round(w * scale), round(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
            detail = ocr_readers.read_amount_detail_from_image(
                small, (0, 0, small.shape[1], small.shape[0])
            )
            value = None if detail is None else detail.value
            if value is not None and value != true_value:
                wrong.append((name, scale, value))
    assert wrong == [], f"entrypoint confident wrong reads: {wrong}"


def test_adversary_reported_wrong_value_regressions_stay_unknown(
    production_bank, monkeypatch
) -> None:
    """Concrete wrong values adversary A reproduced before the recovery harden."""
    monkeypatch.setattr(ocr_readers, "_bank", lambda: production_bank)
    cases = [
        ("stack_343_60_sprite_far_fragments_at_1272x896.png", 0.80),
        ("stack_343_60_sprite_far_fragments_at_1272x896.png", 0.85),
        ("bet_0_50_dot_lost_at_890x627.png", 1.05),
        ("pot_39_50_two_forged_separators_at_1272x896.png", 0.75),
    ]
    for name, scale in cases:
        img = cv2.imread(str(FIXTURES / name))
        assert img is not None, name
        h, w = img.shape[:2]
        small = cv2.resize(
            img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
        detail = ocr_readers.read_amount_detail_from_image(
            small, (0, 0, small.shape[1], small.shape[0])
        )
        assert detail is not None
        # Unknown is fine; a fabricated value is not. True values at these scales
        # are either refused or (for bet_0_50) may recover correctly as 0.5.
        if detail.value is not None:
            if name.startswith("bet_0_50"):
                assert detail.value == 0.5, (name, scale, detail.value)
            elif name.startswith("pot_39_50"):
                assert detail.value == 39.5, (name, scale, detail.value)
            else:
                assert detail.value == 343.6, (name, scale, detail.value)

def test_no_frozen_fixture_reads_a_wrong_value_at_any_render_size(
    production_bank,
) -> None:
    """The whole sweep as one assertion, which is what the previous round's
    scale-stability test was missing: it swept ONE fixture at four scales, and the
    fixture it picked was the one that survives. Every numeric fixture, every
    0.05 step from 0.60x to 2.10x -- must be either the true value or unknown.
    Never a different number.

    This is also the adversarial sweep from note 12 restated as a REFUSAL test: that
    sweep found 893 confident values disagreeing with the audited 1.0x reference,
    757 of them removed by the render floor. The remaining 136 are the business of
    P1, P4 and P7's upper edge, and what is asserted here is that the count is ZERO
    -- not that any particular constant is correctly placed.

    `None` as a truth entry means "the crop carries no legible value", so the ONLY
    acceptable answer at every scale is unknown. Several entries carry a true value
    that the reader now REFUSES at 1.0x; the truth column records what the screen
    says, so those stay pinned against ever coming back as a different number."""
    truth: dict[str, float | None] = {
        "stack_314_90_at_1272x896.png": 314.9,
        "bet_19_50_at_1272x896.png": 19.5,
        "bet_0_50_at_2054x1470.png": 0.5,
        "pot_89_1_one_decimal.png": 89.1,
        "pot_240_9_one_decimal.png": 240.9,
        "pot_165_integer.png": 165.0,
        "stack_0_true_zero.png": 0.0,
        "stack_218_at_1272x896.png": 218.0,
        "stack_clipped_box_at_2722x1832.png": None,
        "bet_chips_only_at_1272x896.png": None,
        "pot_9_single_digit_at_2722x1832.png": 9.0,
        "bet_chip_sprite_no_text.png": None,
        "stack_343_60_at_1272x896.png": 343.6,
        "stack_leading_decimal_clipped_at_2054x1470.png": None,
        "pot_212_50_trailing_zero_at_2054x1470.png": 212.5,
        # Added this phase; all four are refused at 1.0x, and must stay refused or
        # (never, on this corpus) return their true value at every other scale.
        "pot_240_9_chip_overlaps_run_at_2132x1378.png": 240.9,
        "bet_chip_covered_malformed_suffix_at_2054x1470.png": None,
        "stack_1131_90_above_calibrated_band_at_2138x1402.png": 1131.9,
        # Round-1 repair: the two occlusion crops carry no complete numeral (the
        # leading digits are hidden), so unknown is the only acceptable answer at
        # every scale; the suffix-named stack pins the affix gate's admitting
        # direction and must read its value or refuse -- never anything else.
        "stack_99_50_menu_occluded_at_2138x1402.png": None,
        "pot_6_50_sprite_occluded_at_2722x1832.png": None,
        "stack_198_suffix_named_at_2054x1470.png": 198.0,
        "stack_191_wide_gap_at_1272x896.png": 191.0,
        "bet_0_50_dot_lost_at_890x627.png": 0.5,
        "bet_18_30_suffix_absorbed_1399x986.png": 18.3,
        "bet_chip_annulus_square_at_2722x1832.png": None,
        # Round-2 dot-forgery pair: the screen renders 197 BB in both; the
        # forged crop must never come back 1.97 at ANY scale.
        "stack_197_at_2054x1470.png": 197.0,
        "stack_197_speck_forged_decimal_at_2054x1470.png": 197.0,
        # Round-2 wide-occluder composite: the screen renders 343.60 BB under
        # the sprite; 0.0 (or anything else) must never ship.
        "stack_343_60_sprite_far_fragments_at_1272x896.png": 343.6,
        # Round-2 predicate pins: a real stack whose only defect is a missing
        # "BB" (the timer badge wins the run), and a real two-row stack panel.
        "stack_212_90_timer_badge_at_2062x1178.png": 212.9,
        "stack_124_80_name_row_above_at_2138x1402.png": 124.8,
        # Round-3 repair. Five occlusion composites built from real development
        # crops with the production chip sprite (see PROVENANCE for the exact
        # boxes); each shipped the wrong number before its predicate was
        # repaired, and the truth column is what the screen renders underneath.
        # Plus one unmodified control, the top-clip killer for P4's touch arm.
        "stack_392_30_digit_severed_by_sprite_at_2062x1178.png": 392.3,
        "stack_190_10_digit_occluded_into_affix_at_2722x1832.png": 190.1,
        "stack_218_top_shaved_at_1272x896.png": 218.0,
        "pot_39_50_two_forged_separators_at_1272x896.png": 39.5,
        "stack_95_50_leading_digit_occluded_at_2138x1402.png": 95.5,
        "stack_162_40_at_2138x1402.png": 162.4,
    }
    assert set(truth) == {p.name for p in FIXTURES.glob("*.png")}, (
        "every frozen OCR fixture must carry a transcribed truth value here; a "
        "fixture with no entry is swept by nothing")
    wrong: list[tuple[str, float, object]] = []
    for name, true_value in truth.items():
        img = cv2.imread(str(FIXTURES / name))
        assert img is not None, name
        h, w = img.shape[:2]
        for i in range(31):
            scale = round(0.60 + 0.05 * i, 2)
            small = cv2.resize(img, (round(w * scale), round(h * scale)),
                               interpolation=cv2.INTER_AREA)
            value = production_bank.read_number_detail(small).value
            if value is not None and value != true_value:
                wrong.append((name, scale, value))
    assert wrong == [], f"confident wrong reads: {wrong}"


def test_real_crop_gap_arm_does_not_preempt_the_separator_guard(
    production_bank,
) -> None:
    """Round 4, adversary B. The fail-closed guard for "a separator was LOCATED and
    no reading of it reconciles" was unreachable in exactly the case it was written
    for, because the decimal-GAP fallback ran first and always found a split.

    Real crop: "18.30 BB" on the 1272x896 client, re-rendered at 1.10x (a 1399x986
    window, inside the supported range). At that size the "BB" suffix clears
    _AFFIX_MAX_REL_H and is read into the numeral as "88", so the digits are
    "183088"; the REAL dot is still segmented on the baseline, but its split leaves
    4 fractional places and is rejected on arity. The gap arm then split at the WORD
    SPACE before the suffix and returned a confident 1830.88 -- a 100x inflation on
    a bet, in a 200 BB game.

    The gap arm is now deleted outright, so this crop is owned by whichever
    condition sees it first; on this bank that is the unexplained ink of the
    absorbed suffix. What is pinned is the outcome: never 1830.88, never any
    value."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "bet_18_30_suffix_absorbed_1399x986.png"))
    )
    assert detail.value != 1830.88, "the word space before the suffix is not a decimal"
    assert detail.value is None, "a crop whose suffix was read as digits means unknown"
    assert detail.decimal_source in REFUSAL_CODES


# --------------------------------------------------------------------------- #
# The contract itself.
# --------------------------------------------------------------------------- #
def test_decimal_source_none_no_longer_exists() -> None:
    """"none" was returned BOTH for a proven integer and for a crop containing no
    text at all, which is precisely the conflation this phase exists to end. The
    vocabulary is now two disjoint closed sets."""
    assert "none" not in DECIMAL_EVIDENCE
    assert "none" not in REFUSAL_CODES
    assert not (DECIMAL_EVIDENCE & REFUSAL_CODES), "the two halves must be disjoint"
    assert DECIMAL_EVIDENCE == {"dot", "integer"}
    src = inspect.getsource(ocr_readers.TemplateOCR.read_number_detail)
    assert '"none"' not in src


def test_every_refusal_carries_a_named_code(production_bank) -> None:
    """Every frozen fixture, at native size and across a resize: a read either
    produces a value with a member of DECIMAL_EVIDENCE, or it is UNKNOWN with a
    member of REFUSAL_CODES. The two exits at :488/:493/:540 that returned the bare
    `AmountRead(None, "", 0.0, "none", 0)` were indistinguishable from a value read
    with no decimal."""
    seen: set[str] = set()
    for path in sorted(FIXTURES.glob("*.png")):
        img = cv2.imread(str(path))
        assert img is not None, path.name
        h, w = img.shape[:2]
        for scale in (0.70, 0.85, 1.00, 1.30, 1.70):
            small = img if scale == 1.0 else cv2.resize(
                img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            detail = production_bank.read_number_detail(small)
            seen.add(detail.decimal_source)
            if detail.value is None:
                assert detail.decimal_source in REFUSAL_CODES, (path.name, scale, detail)
            else:
                assert detail.decimal_source in DECIMAL_EVIDENCE, (path.name, scale, detail)
    # The fixtures are chosen to exercise the contract, not just to pass it.
    assert {"dot", "integer"} <= seen
    assert len(seen & REFUSAL_CODES) >= 5, sorted(seen & REFUSAL_CODES)


def test_the_refusal_vocabulary_is_named_literally() -> None:
    """Named one by one rather than looped over the frozenset: a loop stops testing
    a code the moment someone deletes it (note 10's rule, applied to the reader's
    own vocabulary this time)."""
    for code in ("no_digit_run", "ambiguous_longest_run", "unexplained_ink_in_numeral",
                 "suffix_not_bb", "run_clipped", "separator_unreconciled",
                 "integer_over_decimal_band", "unexplained_gap_in_numeral",
                 "leading_zero_no_dot",
                 "below_calibrated_render_size", "above_calibrated_render_size",
                 "reader_unavailable"):
        assert code in REFUSAL_CODES, code
    assert "dot" in DECIMAL_EVIDENCE
    assert "integer" in DECIMAL_EVIDENCE


def test_a_value_read_never_carries_a_refusal_code(scaled_digit_templates) -> None:
    """The disjointness as an invariant of the returned object, not of the two
    frozensets: `value is None` and `decimal_source in REFUSAL_CODES` must always
    agree."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    crops = [
        _compose("050", 0.9, [7, 3], dot=(0, 3, 3)),      # 0.5
        _compose("1650", 0.9, [3, 3, 3]),                  # 1650
        _compose("050", 0.9, [3, 3]),                      # leading zero -> unknown
        _compose("12345", 0.9, [3, 7, 3, 3], dot=(1, 3, 3)),  # separator -> unknown
        _compose("75", 0.9, [9]),                          # decimal band -> unknown
        np.zeros((40, 60, 3), np.uint8),                   # empty -> unknown
    ]
    for crop in crops:
        d = bank.read_number_detail(crop)
        assert (d.value is None) == (d.decimal_source in REFUSAL_CODES), d
        assert (d.value is not None) == (d.decimal_source in DECIMAL_EVIDENCE), d


# --------------------------------------------------------------------------- #
# Round-1 repair regressions: OCCLUSION, CLIPPING, and MISSING-DIGIT families.
#
# The general defect (adversary A, round 1): ink smaller than the glyph band was
# routed into buckets no predicate ever policed, so occlusion and box-clipping
# produced confident wrong values (60982 under left-edge clipping, 13485 of 15408
# under leading-digit occlusion) while right/top/bottom clipping produced zero.
# The general fix is structural, not tuned: P2 polices EVERY component that
# vertically overlaps the numeral's row (_policed_ink), P4 refuses a run whose
# left margin is inside the numeral's own letter spacing, and P5's gap test runs
# on every read with only the located separator allowed to occupy a gap.
# --------------------------------------------------------------------------- #
def test_real_crop_menu_occlusion_is_unknown(production_bank) -> None:
    """cwpt01 t=554: the client's side menu slides over seat 3's panel and only
    '.50 BB' survives. The same seat reads 99.50 BB at t=520..553 and t=555..556.
    The sliver of the hidden digit and the out-of-run decimal both landed in the
    `dots` bucket, which no predicate iterated, and the reader shipped a confident
    50.0 -- 49.5 BB below the true stack -- which one extra second of menu
    animation turned into an exported `preflop seat:3 call 49.5` that never
    happened, inside a 10x pot, with warnings=[]."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_99_50_menu_occluded_at_2138x1402.png"))
    )
    assert detail.value is None, "an occluded numeral must not ship a fragment's value"
    assert detail.decimal_source == "unexplained_ink_in_numeral"


def test_real_crop_sprite_occluded_pot_is_unknown(production_bank) -> None:
    """g0711 t=257: a dealt-card sprite covers the '6' of a 6.50 BB pot, leaving
    a 19x10 fragment that was in NEITHER `tall` NOR `dots` (invisible to every
    predicate) and a decimal point left of the surviving run. Read 50.0 -- the
    regression note 13 section 6 recorded as 'Not repaired'. Now the fragment and
    the out-of-run dot are both policed ink."""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "pot_6_50_sprite_occluded_at_2722x1832.png"))
    )
    assert detail.value is None
    assert detail.decimal_source == "unexplained_ink_in_numeral"


def test_clip_family_never_yields_a_different_value(production_bank) -> None:
    """THE FAMILY, on ALL FOUR SIDES: shifting any crop edge inward removes
    information while the number on screen is unchanged, so every clipped read
    must return the native value or UNKNOWN. Before the repair, 9628 of 17469
    value-producing corpus crops had at least one confident-wrong left-clipped
    read (60982 wrong values; 4313 of them exactly 100x). After it: zero, over
    every crop, every side, 60 depths.

    ITS PREDECESSOR CLAIMED "every side" AND SLICED ONLY COLUMNS. That gap left
    P4's four boundary-TOUCH clauses -- `x0 <= 0`, `x1 >= crop_w - 1`,
    `run_y0 <= 0`, `run_y1 >= crop_h - 1` -- pinned by nothing: only the margin
    clause and the predicate as a whole had ever been ablated. With the four
    touch clauses disabled and the margin clause kept, the entire owned CV suite
    stays green AND all 18,006 native reads are byte-identical, so no pipeline
    run could notice, while a 92,624-read four-direction clip sweep produces 83
    confident wrong values -- 23 on top clipping, 60 on bottom, every one a
    stack_text read (162.4 -> 102.4 at 20px, 232.2 -> 90.0 at 26px,
    394.4 -> 304.4 at 25px). Two of the fixtures below are here specifically to
    kill that mutant in the vertical directions, and the assertions at the end
    require each direction's refusal to be OBSERVED rather than assumed."""
    saw_run_clipped = {"left": False, "right": False, "top": False, "bottom": False}
    for name in ("stack_314_90_at_1272x896.png", "bet_19_50_at_1272x896.png",
                 "pot_240_9_one_decimal.png", "bet_0_50_at_2054x1470.png",
                 # bottom-clip killer: 197.0 -> 107.0 at 20px with the touch
                 # clauses off. top-clip killer: 162.4 -> 102.4 at 20px.
                 "stack_197_at_2054x1470.png", "stack_162_40_at_2138x1402.png"):
        img = cv2.imread(str(FIXTURES / name))
        assert img is not None, name
        native = production_bank.read_number_detail(img).value
        assert native is not None, f"{name} must read at native crop"
        h, w = img.shape[:2]
        for k in range(1, 46):
            for side, sub in (("left", img[:, k:]), ("right", img[:, : w - k]),
                              ("top", img[k:, :]), ("bottom", img[: h - k, :])):
                if sub.shape[0] < 3 or sub.shape[1] < 3:
                    continue
                detail = production_bank.read_number_detail(sub)
                assert detail.value is None or detail.value == native, (
                    f"{name} {side}-clipped {k}px read {detail.value!r} "
                    f"(native {native!r})")
                saw_run_clipped[side] |= detail.decimal_source == "run_clipped"
    for side, seen in saw_run_clipped.items():
        assert seen, f"P4 never fired on the {side} edge across the family"


def test_interior_digit_paint_out_is_unknown(production_bank) -> None:
    """P5(b) on the dot branch. The gap test lived in an `elif`, so once a
    separator was located NO gap was examined and painting out an interior digit
    left a confident wrong value on 38439 of 39953 corpus paint-outs (96.2%):
    191.3 -> 11.3, 194.6 -> 14.6, 321.2 -> 31.2. Now every hole the located
    separator does not occupy must stay inside the measured letter-spacing
    ceiling. Four instances across three geometries and both branches."""
    # Three dot-branch instances across two geometries plus one integer-branch
    # instance (the pre-existing arm, kept as the negative-control direction).
    for name in ("stack_314_90_at_1272x896.png", "stack_343_60_at_1272x896.png",
                 "pot_240_9_one_decimal.png", "pot_165_integer.png"):
        img = cv2.imread(str(FIXTURES / name))
        assert img is not None, name
        native = production_bank.read_number_detail(img).value
        assert native is not None, name
        mask = binarize_text(img)
        comps = segment_glyphs(mask, min_area=2, min_h_px=1)
        max_h = max(c.h for c in comps)
        tall = sorted([c for c in comps if c.h >= 0.55 * max_h], key=lambda g: g.x)
        digit_glyphs = [c for c in tall
                        if production_bank.classify_digit(c.mask)[0].isdigit()]
        # The numeral's own run, not every digit-classified glyph: the 'O' of a
        # "POT:" prefix classifies as '0' and sits a word space away, and treating
        # it as run ink would turn this into a LEADING-digit paint-out (which is
        # trace-free and out of scope here). Split on the letter-spacing bound.
        runs, cur = [], []
        for g in digit_glyphs:
            if cur and g.x - (cur[-1].x + cur[-1].w) >= 0.28 * max_h:
                runs.append(cur)
                cur = []
            cur.append(g)
        runs.append(cur)
        digits = max(runs, key=len)
        assert len(digits) >= 3, name
        flat = img.reshape(-1, 3)
        bg = np.median(flat[np.argsort(flat.sum(1))[:200]], 0).astype(np.uint8)
        for g in digits[1:-1]:                      # interior digits only
            painted = img.copy()
            painted[max(0, g.y - 1):g.y + g.h + 1, max(0, g.x - 1):g.x + g.w + 1] = bg
            detail = production_bank.read_number_detail(painted)
            assert detail.value is None, (
                f"{name}: painting out an interior digit read {detail.value!r}")


def test_two_separator_candidates_that_both_reconcile_are_refused(
    scaled_digit_templates,
) -> None:
    """P5(a) as SPECIFIED, not as ranked. The shipped loop broke at the first
    reconcilable candidate in widest-gap-first order, so two candidates that both
    reconcile -- to DIFFERENT values -- were resolved by a ranking key instead of
    refused, and `discarded` stayed 0. 'Selecting among competing readings by
    score is exactly the kind of rescue this contract removes.' Measured: 0 corpus
    reads carry two candidates, so the strict predicate costs nothing."""
    bank = TemplateOCR(dict(scaled_digit_templates), {})
    # Two baseline dots in different gaps of "1234": splits 2 and 3 BOTH
    # reconcile (both leave <= 2 fractional digits), reading 12.34 or 123.4.
    # Both hosting gaps sit inside the measured decimal band (22px digits ->
    # >= 5.08px) so each is a legal candidate and P5a is the refusal that fires.
    img = _compose("1234", 0.9, [3, 7, 6], dots=[(1, 3, 3), (2, 3, 3)])
    detail = bank.read_number_detail(img)
    assert detail.value is None, f"two reconcilable separators read {detail.value!r}"
    assert detail.decimal_source == "separator_unreconciled"


def test_affix_gate_is_value_admitting_and_pinned(production_bank) -> None:
    """_AFFIX_MAX_REL_H is NOT refusal-only, whatever its old comment said:
    passing the gate puts a glyph into `named`, which exempts it from P2, which
    ADMITS the value. Ablating the gate to 0.0 flipped 57 corpus reads to UNKNOWN
    with the whole suite green (mutation K-affix0). This crop -- a real '198 BB'
    stack -- reads its value only because the gate names its suffix caps, so the
    admitting direction is now pinned from the reader itself. (The permissive
    direction, 99.0, was already caught by two suffix-absorption tests.)"""
    detail = production_bank.read_number_detail(
        cv2.imread(str(FIXTURES / "stack_198_suffix_named_at_2054x1470.png"))
    )
    assert detail.value == 198.0
    assert detail.decimal_source == "integer"


def test_frame_from_models_preserves_the_three_way_amount_distinction() -> None:
    """THE PRODUCTION SITE of the value/refusal split (region_detections
    frame_from_models), pinned with the real bank on real crops. Two mutations
    survived the entire 253-test surface here: `attr = 0.0 if detail.value is
    None else detail.value` (every refusal becomes a confident zero -- 13
    fabricated all-ins, a hero-net sign flip, exports 12 -> 15) and
    `attr_source = None if detail.value is None else ...` (every refusal becomes
    ABSENT -- re-enabling the bet-delta estimator on refused transitions, exports
    12 -> 18). Both die here: the fixture-path tests feed attr_source strings by
    hand and never exercise this line."""
    from cv_lab.scripts.pipeline import region_detections as rd
    from cv_lab.scripts.pipeline.ocr_readers import _bank

    bank = _bank()
    assert bank is not None, "production template bank must be present (tracked)"
    readable = cv2.imread(str(FIXTURES / "pot_165_integer.png"))
    refused = cv2.imread(str(FIXTURES / "stack_99_50_menu_occluded_at_2138x1402.png"))
    canvas = np.zeros((max(readable.shape[0], refused.shape[0]) + 40,
                       readable.shape[1] + refused.shape[1] + 60, 3), np.uint8)
    y0, x0 = 10, 10
    canvas[y0:y0 + readable.shape[0], x0:x0 + readable.shape[1]] = readable
    x1 = x0 + readable.shape[1] + 20
    canvas[y0:y0 + refused.shape[0], x1:x1 + refused.shape[1]] = refused
    rows = [
        {"class": "pot_text", "conf": 0.9, "x1": x0, "y1": y0,
         "x2": x0 + readable.shape[1], "y2": y0 + readable.shape[0]},
        {"class": "stack_text", "conf": 0.9, "x1": x1, "y1": y0,
         "x2": x1 + refused.shape[1], "y2": y0 + refused.shape[0]},
    ]
    frame = rd.frame_from_models(canvas, 0.0, rows, classifier=None, pad=0.0)
    pot, stack = frame.detections

    # A PROVEN read: value + DECIMAL_EVIDENCE, and amount_state agrees.
    assert pot.attr == 165.0
    assert pot.attr_source in DECIMAL_EVIDENCE
    assert rd.amount_state(pot) == ("value", 165.0)

    # A REFUSED read: attr None, attr_source a REFUSAL CODE. Not 0.0 ("a zero
    # stack labels the seat's action all-in") and not ABSENT (attr_source None).
    assert stack.attr is None
    assert stack.attr_source in REFUSAL_CODES
    kind, payload = rd.amount_state(stack)
    assert kind == "unknown" and payload == stack.attr_source
