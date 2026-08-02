"""Property tests for OCR numeric parsing (PLAN Phase 14).

``tests/test_ocr_readers.py`` is deep but entirely example-based: ~50 frozen
real HUD crops plus hand-laid synthetic renders. Nothing GENERATES amounts,
render sizes and hostile decorations and asserts the invariant the whole reader
contract exists for:

    read_number_detail(render(v)) is v, or it is UNKNOWN.
    It is never a different number.

That is the release-blocking distinction stated in this project's own terms: a
refused read is a coverage limitation, and a confident wrong stack is a study
record built on a number nobody typed. The severe defects this reader has
actually shipped were all of the second kind -- 314.90 read as 31490 (a 100x
inflation, frozen at tests/test_ocr_readers.py), "0.50" read as 50.0 into an
exported call, an empty chip-sprite crop read as a confident 0.0 that the spine
books as an all-in.

WHAT IS RENDERED AND WHY IT IS SYNTHETIC. The production template bank is
calibrated to the ClubWPT client's own font; feeding it cv2 Hershey glyphs would
measure the font mismatch rather than the reader. So the bank here is built the
way ``tests/test_ocr_readers.py`` already builds one -- averaged over three
render scales, mirroring how calibrate_ocr.py averages real observations -- and
every crop renders a WHOLE HUD token, ``<numeral> BB``, because a bare row of
digits is not something the client ever draws. Scales 0.45 to 1.4 span glyph
heights 12 to 32, which is exactly the bank's calibrated band
(``_MIN_CALIBRATED_RUN_H`` .. ``_MAX_CALIBRATED_RUN_H``).

WHAT IS NOT ASSERTED, deliberately. "Adding ink never changes the value" is not
a property this or any template reader can have: the mask IS the text, so
painting white into it can genuinely turn a 5 into a 6, and a test asserting
otherwise would be asserting that the reader must ignore what the pixels show.
The degradation properties below therefore add ink OFF the numeral's row (P2's
far arm) and CLIP the crop (P4) -- transformations that remove or displace
evidence without forging a different glyph.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cv_lab.scripts.pipeline.ocr_readers import (
    _MAX_CALIBRATED_RUN_H,
    _MIN_CALIBRATED_RUN_H,
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
THICK = 2
WHITE = (255, 255, 255)
PAD = 20

# 0.45 -> glyph height 12 (the calibrated floor); 1.4 -> 32 (the ceiling).
IN_BAND_SCALES = (0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
BELOW_BAND_SCALES = (0.25, 0.3, 0.35, 0.4)
ABOVE_BAND_SCALES = (1.5, 1.7, 2.0, 2.4)

SETTINGS = settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)

_BITMAPS: dict[tuple[str, float], np.ndarray] = {}


def _glyph_bitmap(char: str, scale: float) -> np.ndarray:
    """One binarized glyph, cached: rendering dominates the runtime of this file."""
    key = (char, scale)
    if key not in _BITMAPS:
        canvas = np.zeros((160, 160, 3), np.uint8)
        cv2.putText(canvas, char, (24, 110), FONT, scale, WHITE, THICK)
        glyphs = segment_glyphs(binarize_text(canvas))
        assert len(glyphs) == 1, f"{char!r} at {scale} rendered {len(glyphs)} components"
        _BITMAPS[key] = glyphs[0].mask
    return _BITMAPS[key]


@pytest.fixture(scope="module")
def bank() -> TemplateOCR:
    """Digits plus the 'B' of the "BB" suffix, averaged over three render scales.

    'B' is not furniture: without it the suffix classifies as a digit, the
    numeral is never proven terminated (P1), and every read below is UNKNOWN for
    a reason that has nothing to do with the property under test.
    """
    templates: dict[str, np.ndarray] = {}
    for char in "0123456789B":
        vector = np.mean(
            [_norm(_glyph_bitmap(char, s), DIGIT_SIZE) for s in (0.5, 0.9, 1.2)], axis=0
        )
        templates[char] = vector / np.linalg.norm(vector)
    return TemplateOCR(templates, {})


def _render(digits: str, scale: float, split: int | None = None) -> np.ndarray:
    """``<digits> BB`` at MEASURED geometry, with the decimal point where the
    client puts it.

    Gaps are proportional to glyph height so the layout is the same numeral at
    every scale: intra-numeral letter spacing at 0.10 of run height (inside the
    measured band), the separator's host gap at 0.32 (inside the band a located
    decimal occupies, ``_DECIMAL_BAND_MIN_GAP`` .. ), the dot itself at 0.13 of
    run height (the measured dot_h/digit_h of real crops), and a 0.60 word space
    before the suffix.
    """
    bitmaps = [_glyph_bitmap(c, scale) for c in digits]
    height = max(b.shape[0] for b in bitmaps)
    caps = [_glyph_bitmap("B", scale), _glyph_bitmap("B", scale)]
    tight = max(1, round(0.10 * height))
    separator_gap = max(3, round(0.32 * height))
    dot_side = max(2, round(0.13 * height))
    gaps = [tight] * max(0, len(digits) - 1)
    if split is not None:
        gaps[split - 1] = separator_gap
    word = round(0.60 * height)
    kern = max(1, round(0.10 * height))

    width = (
        sum(b.shape[1] for b in bitmaps)
        + sum(gaps)
        + word
        + sum(b.shape[1] for b in caps)
        + kern * len(caps)
    )
    image = np.zeros((height + 2 * PAD, width + 2 * PAD, 3), np.uint8)
    x, y = PAD, PAD
    spans: list[tuple[int, int]] = []
    for index, bitmap in enumerate(bitmaps):
        bh, bw = bitmap.shape
        image[y : y + bh, x : x + bw][bitmap] = WHITE
        spans.append((x, x + bw))
        x += bw + (gaps[index] if index < len(gaps) else 0)
    x += word
    for cap in caps:
        bh, bw = cap.shape
        image[y : y + bh, x : x + bw][cap] = WHITE
        x += bw + kern
    if split is not None:
        left, right = spans[split - 1][1], spans[split][0]
        centre = (left + right) // 2 - dot_side // 2
        image[y + height - dot_side : y + height, centre : centre + dot_side] = WHITE
    return image


def _run_height(image: np.ndarray) -> int:
    glyphs = segment_glyphs(binarize_text(image), min_area=2, min_h_px=1)
    return max((g.h for g in glyphs), default=0)


@st.composite
def hud_amount(draw) -> tuple[str, int | None, float]:
    """A numeral the ClubWPT client can render, and the value it means.

    Constrained to what the client actually draws, because a render the client
    cannot produce is not evidence about the reader: at most two fractional
    places (``_MAX_FRACTIONAL_DIGITS``), and a leading zero only ever in front
    of the separator (P6 -- "050" is a "0.50" whose dot was lost).
    """
    length = draw(st.integers(min_value=1, max_value=5))
    splits = [None] + [s for s in range(1, length) if length - s <= 2]
    split = draw(st.sampled_from(splits))
    first = draw(st.sampled_from("0123456789" if split == 1 else "123456789"))
    rest = draw(st.text(alphabet="0123456789", min_size=length - 1, max_size=length - 1))
    return first + rest, split, draw(st.sampled_from(IN_BAND_SCALES))


def _truth(digits: str, split: int | None) -> float:
    return float(digits if split is None else f"{digits[:split]}.{digits[split:]}")


def _assert_contract(read: AmountRead) -> None:
    """The reader's own contract, which every assertion below rests on: the two
    halves of ``decimal_source`` are disjoint closed sets, and UNKNOWN is a
    first-class answer rather than a missing one."""
    if read.value is None:
        assert read.decimal_source in REFUSAL_CODES, read
    else:
        assert read.decimal_source in DECIMAL_EVIDENCE, read


# --------------------------------------------------------------------------- #
# The value property.
# --------------------------------------------------------------------------- #
@given(amount=hud_amount())
@SETTINGS
def test_a_rendered_amount_reads_as_itself_or_as_unknown(
    amount: tuple[str, int | None, float], bank: TemplateOCR
) -> None:
    """THE property. Across every numeral shape the client draws and every render
    size inside the calibrated band, the only two honest answers are the value
    and UNKNOWN."""
    digits, split, scale = amount
    image = _render(digits, scale, split)
    read = bank.read_number_detail(image)
    _assert_contract(read)
    assert read.value in (_truth(digits, split), None), (
        f"{digits!r} split={split} at scale {scale} (run_h {_run_height(image)}) "
        f"read {read.value!r} raw={read.raw!r} code={read.decimal_source}"
    )


@given(amount=hud_amount())
@SETTINGS
def test_the_same_numeral_never_reads_two_different_values_across_render_sizes(
    amount: tuple[str, int | None, float], bank: TemplateOCR
) -> None:
    """Scale invariance, generated. The reader's constants are ratios against run
    height precisely so a window resize cannot move a value, and the defect that
    motivated them was non-monotonic in scale: one real stack read 343.6 at 1.00x
    and 0.95x, 34360.0 at 0.90x, and 343.6 again at 0.85x."""
    digits, split, _ = amount
    values = {
        bank.read_number_detail(_render(digits, scale, split)).value
        for scale in IN_BAND_SCALES
    }
    values.discard(None)
    assert len(values) <= 1, f"{digits!r} split={split} read {sorted(values)} across scales"


@given(amount=hud_amount(), scale=st.sampled_from(BELOW_BAND_SCALES + ABOVE_BAND_SCALES))
@SETTINGS
def test_a_render_outside_the_calibrated_band_is_refused_by_name(
    amount: tuple[str, int | None, float], scale: float, bank: TemplateOCR
) -> None:
    """P7. Outside the band the bank is extrapolating, and the measured cost of
    letting it is 10x-100x values at full confidence (and one 0.0 the spine books
    as an all-in). Refusal there must be unconditional and must say which edge.

    The band is stated as literals rather than read from the module, because the
    module's constants are what is under test: classifying with them would make
    this test agree with any band the reader declares, including the 9 the floor
    used to be -- an extrapolation under which 757 confident wrong values sat
    inside the range the constant called calibrated. Re-measuring the bank is a
    legitimate change; it just has to come here too.
    """
    assert (_MIN_CALIBRATED_RUN_H, _MAX_CALIBRATED_RUN_H) == (12, 32), (
        "the calibrated render band moved; this test's literals must be "
        "re-derived from the new measurement, not from the constants"
    )
    digits, split, _ = amount
    image = _render(digits, scale, split)
    height = _run_height(image)
    if 12 <= height <= 32:
        return  # the scale landed back inside the band; nothing to assert
    read = bank.read_number_detail(image)
    _assert_contract(read)
    assert read.value is None, f"{digits!r} at run_h {height} returned {read.value!r}"
    assert read.decimal_source in {
        "below_calibrated_render_size",
        "above_calibrated_render_size",
        "no_digit_run",
        "ambiguous_longest_run",
        "run_clipped",
        "unexplained_ink_in_numeral",
        "suffix_not_bb",
    }, read


# --------------------------------------------------------------------------- #
# Hostile HUD text. Everything a real table can put in a numeric crop that is
# not the numeral: a thousands separator that is the same silhouette as the
# decimal point, a currency mark, a magnitude suffix, a minus sign, an occluder
# where a digit should be, and a numeral with no terminator at all.
# --------------------------------------------------------------------------- #
def _decorated(
    digits: str,
    scale: float,
    *,
    prefix: tuple[int, int] | None = None,
    infix_after: int | None = None,
    infix: tuple[int, int] | None = None,
    suffix_block: tuple[int, int] | None = None,
    terminator: bool = True,
    separator_after: int | None = None,
) -> np.ndarray:
    """``<decoration><digits><decoration> BB`` laid out glyph by glyph.

    ``prefix`` / ``infix`` / ``suffix_block`` are (width, height) rectangles --
    a currency mark, an occluder over a digit, a 'K'. ``separator_after`` draws a
    baseline dot in a wide gap, which is what a THOUSANDS separator is: the same
    ink in the same place as a decimal point.
    """
    bitmaps = [_glyph_bitmap(c, scale) for c in digits]
    height = max(b.shape[0] for b in bitmaps)
    tight = max(1, round(0.10 * height))
    wide = max(3, round(0.32 * height))
    dot_side = max(2, round(0.13 * height))
    word = round(0.60 * height)
    kern = max(1, round(0.10 * height))

    pieces: list[tuple[str, object]] = []
    if prefix is not None:
        pieces.append(("block", (prefix[0], prefix[1], 0)))
        pieces.append(("gap", tight))
    for index, char in enumerate(digits):
        pieces.append(("glyph", char))
        if index == len(digits) - 1:
            continue
        if separator_after == index + 1:
            # One gap of `wide` with the dot centred in it, exactly as _render
            # lays a real decimal out. Padding on BOTH sides instead would leave
            # sub-holes of 0.32 of run height once the dot is subtracted, above
            # the widest gap a real numeral shows inside itself, and the read
            # would then refuse for a hole in the numeral rather than for the
            # separator this test is about.
            pad = max(1, (wide - dot_side) // 2)
            pieces.append(("gap", pad))
            pieces.append(("dot", (dot_side, dot_side)))
            pieces.append(("gap", wide - dot_side - pad if wide > dot_side + pad else 1))
        elif infix_after == index + 1 and infix is not None:
            pieces.append(("gap", tight))
            pieces.append(("block", (infix[0], infix[1], 0)))
            pieces.append(("gap", tight))
        else:
            pieces.append(("gap", tight))
    if suffix_block is not None:
        pieces.append(("gap", tight))
        pieces.append(("block", (suffix_block[0], suffix_block[1], 0)))
    if terminator:
        pieces.append(("gap", word))
        pieces.append(("glyph", "B"))
        pieces.append(("gap", kern))
        pieces.append(("glyph", "B"))

    width = 2 * PAD
    for kind, payload in pieces:
        if kind == "glyph":
            width += _glyph_bitmap(payload, scale).shape[1]  # type: ignore[arg-type]
        elif kind == "gap":
            width += payload  # type: ignore[operator]
        else:
            width += payload[0]  # type: ignore[index]
    image = np.zeros((height + 2 * PAD, width, 3), np.uint8)
    x, y = PAD, PAD
    for kind, payload in pieces:
        if kind == "glyph":
            bitmap = _glyph_bitmap(payload, scale)  # type: ignore[arg-type]
            bh, bw = bitmap.shape
            image[y : y + bh, x : x + bw][bitmap] = WHITE
            x += bw
        elif kind == "gap":
            x += payload  # type: ignore[operator]
        elif kind == "dot":
            dw, dh = payload  # type: ignore[misc]
            image[y + height - dh : y + height, x : x + dw] = WHITE
            x += dw
        else:
            bw, bh, top = payload  # type: ignore[misc]
            image[y + top : y + top + bh, x : x + bw] = WHITE
            x += bw
    return image


@given(
    digits=st.text(alphabet="123456789", min_size=4, max_size=5),
    split_at=st.integers(min_value=1, max_value=2),
    scale=st.sampled_from(IN_BAND_SCALES),
)
@SETTINGS
def test_a_thousands_separator_is_never_read_as_a_decimal_point(
    digits: str, split_at: int, scale: float, bank: TemplateOCR
) -> None:
    """"12,345" and "12.345" are the same pixels in the same place, and splitting
    at the wrong one under-reads by 1000x. The client renders at most two
    fractional places, so a split leaving more than two is a group separator and
    the read must be UNKNOWN -- never the deeper number, and never the integer
    either, because nothing proves the ink was not a decimal."""
    image = _decorated(digits, scale, separator_after=split_at)
    read = bank.read_number_detail(image)
    _assert_contract(read)
    if len(digits) - split_at <= 2:
        return  # a legal decimal position, covered by the value property above
    assert read.value is None, (
        f"{digits[:split_at]},{digits[split_at:]} read {read.value!r} "
        f"({read.decimal_source})"
    )


@given(
    digits=st.text(alphabet="123456789", min_size=2, max_size=4),
    scale=st.sampled_from(IN_BAND_SCALES),
    mark=st.sampled_from(["currency", "magnitude", "minus"]),
)
@SETTINGS
def test_a_non_numeral_mark_beside_the_value_makes_the_read_unknown(
    digits: str, scale: float, mark: str, bank: TemplateOCR
) -> None:
    """A currency symbol, a K/M magnitude suffix, a minus sign: ink the bank
    cannot name, sitting on the numeral's own row. Each one changes what the
    numeral MEANS, so a reader that ignores it ships a number the screen does not
    show. P2 makes the whole read unknown instead."""
    height = _glyph_bitmap("8", scale).shape[0]
    if mark == "currency":
        image = _decorated(digits, scale, prefix=(max(2, round(0.30 * height)), height))
    elif mark == "magnitude":
        image = _decorated(
            digits, scale, suffix_block=(max(2, round(0.35 * height)), height)
        )
    else:
        image = _decorated(
            digits, scale, prefix=(max(2, round(0.35 * height)), max(2, round(0.12 * height)))
        )
    read = bank.read_number_detail(image)
    _assert_contract(read)
    assert read.value is None, f"{mark} beside {digits!r} read {read.value!r}"


@given(
    digits=st.text(alphabet="123456789", min_size=3, max_size=5),
    scale=st.sampled_from(IN_BAND_SCALES),
    data=st.data(),
)
@SETTINGS
def test_an_occluder_where_a_digit_should_be_never_yields_the_shorter_number(
    digits: str, scale: float, data, bank: TemplateOCR
) -> None:
    """The 10x-100x family. A chip stack or a card sprite over one interior digit
    leaves a numeral the reader can still segment, and the surviving fragment is
    a perfectly plausible smaller number -- 240.9 shipped as a confident 2.0,
    343.60 as a confident 0.0. The occluded numeral must be UNKNOWN, and in
    particular must never equal the value with that digit deleted."""
    # An INTERIOR digit: the occluder has to sit between two surviving digits, so
    # the reader is looking at a numeral with a hole in it rather than at a
    # shorter numeral with something after it.
    position = data.draw(st.integers(min_value=1, max_value=len(digits) - 2))
    height = _glyph_bitmap("8", scale).shape[0]
    covered = digits[:position] + digits[position + 1 :]
    image = _decorated(
        covered,
        scale,
        infix_after=position,
        infix=(_glyph_bitmap(digits[position], scale).shape[1], height),
    )
    read = bank.read_number_detail(image)
    _assert_contract(read)
    assert read.value != float(covered), (
        f"an occluded {digits!r} read as the fragment {covered!r}"
    )
    assert read.value != float(digits), "the occluder cannot be read through"


@given(amount=hud_amount())
@SETTINGS
def test_a_numeral_with_no_bb_terminator_is_refused(
    amount: tuple[str, int | None, float], bank: TemplateOCR
) -> None:
    """P1. The suffix is the only token whose content is known a priori, so it is
    the only available proof that the numeral was not truncated on its right."""
    digits, _, scale = amount
    read = bank.read_number_detail(_decorated(digits, scale, terminator=False))
    _assert_contract(read)
    assert read.value is None, f"an unterminated {digits!r} read {read.value!r}"


# --------------------------------------------------------------------------- #
# Degradation. Evidence removed or displaced, never forged: clipping the crop
# and ink OFF the numeral's row. See the module docstring for why painting into
# the numeral itself is not a property this reader can have.
# --------------------------------------------------------------------------- #
@given(
    amount=hud_amount(),
    side=st.sampled_from(["left", "right", "top", "bottom"]),
    # Up to 60px, because the render carries 20px of padding: a cut that stops
    # at 30 only ever grazes the first glyph, and the failures P4 exists for --
    # a leading digit removed cleanly, leaving no fragment behind -- start
    # further in. Ablating P4 and sweeping this range produces 44 confident
    # wrong values in 1,200 clips (742 -> 2.0, 81.347 -> 3.47); stopping at 30
    # produces none, so the bound is what makes this test able to fail.
    cut=st.integers(min_value=1, max_value=60),
)
@SETTINGS
def test_clipping_the_crop_on_any_side_never_yields_a_different_value(
    amount: tuple[str, int | None, float], side: str, cut: int, bank: TemplateOCR
) -> None:
    """P4, generated on all four sides. A detector box that lands slightly wrong
    is the ordinary case, not the exotic one: shifting every crop's left edge
    inward produced 60,982 confident wrong values on the development corpus
    before the margin rule, and the vertical directions produced 83 more."""
    digits, split, scale = amount
    image = _render(digits, scale, split)
    height, width = image.shape[:2]
    if side == "left":
        clipped = image[:, min(cut, width - 2) :]
    elif side == "right":
        clipped = image[:, : max(2, width - cut)]
    elif side == "top":
        clipped = image[min(cut, height - 2) :, :]
    else:
        clipped = image[: max(2, height - cut), :]
    read = bank.read_number_detail(clipped)
    _assert_contract(read)
    assert read.value in (_truth(digits, split), None), (
        f"{digits!r} clipped {cut}px on the {side} read {read.value!r} "
        f"({read.decimal_source})"
    )


@given(
    amount=hud_amount(),
    band=st.sampled_from(["above", "below"]),
    box=st.tuples(
        st.integers(min_value=1, max_value=14),
        st.integers(min_value=1, max_value=8),
    ),
    offset=st.integers(min_value=0, max_value=400),
)
@SETTINGS
def test_ink_off_the_numerals_row_never_changes_the_value(
    amount: tuple[str, int | None, float],
    band: str,
    box: tuple[int, int],
    offset: int,
    bank: TemplateOCR,
) -> None:
    """Compression speck, cursor fleck, the edge of a neighbouring HUD element.
    Ink that is not on the numeral's row is not evidence about the numeral, and
    it may cost coverage but must never move the number."""
    digits, split, scale = amount
    image = _render(digits, scale, split).copy()
    tall, wide = image.shape[:2]
    numeral_top, numeral_bottom = PAD, tall - PAD
    bw, bh = box
    x = offset % max(1, wide - bw - 1)
    if band == "above":
        y = max(0, (numeral_top - bh - 1))
    else:
        y = min(tall - bh - 1, numeral_bottom + 1)
    if y < 0 or y + bh > tall:
        return
    image[y : y + bh, x : x + bw] = WHITE
    read = bank.read_number_detail(image)
    _assert_contract(read)
    assert read.value in (_truth(digits, split), None), (
        f"{digits!r} with a {bw}x{bh} speck {band} the row read {read.value!r}"
    )


# --------------------------------------------------------------------------- #
# Totality, on inputs that are not a HUD crop at all.
# --------------------------------------------------------------------------- #
@given(
    height=st.integers(min_value=0, max_value=60),
    width=st.integers(min_value=0, max_value=200),
    kind=st.sampled_from(["black", "white", "noise", "grey", "blob"]),
    seed=st.integers(min_value=0, max_value=2**16),
)
@SETTINGS
def test_a_crop_containing_no_text_never_returns_a_value(
    height: int, width: int, kind: str, seed: int, bank: TemplateOCR
) -> None:
    """The defect that cost the most. A ``bet_text`` crop holding only the green
    chip sprite -- no text whatsoever -- returned ``AmountRead(value=0.0,
    raw='0', score=0.836)``, byte-identical in every field to the genuine all-in
    "0 BB" read, and a zero stack labels the seat's action all-in.

    Zero is also what a crop of nothing decodes to most naturally, so it is
    called out separately: refusing must not be achieved by banning zeros, and
    the real "0 BB" read is pinned in ``tests/test_ocr_readers.py``.
    """
    rng = np.random.default_rng(seed)
    if kind == "black":
        image = np.zeros((height, width, 3), np.uint8)
    elif kind == "white":
        image = np.full((height, width, 3), 255, np.uint8)
    elif kind == "grey":
        image = np.full((height, width, 3), 40, np.uint8)
    elif kind == "noise":
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    else:
        image = np.zeros((height, width, 3), np.uint8)
        if height > 4 and width > 4:
            cv2.ellipse(
                image,
                (width // 2, height // 2),
                (max(2, width // 6), max(2, height // 4)),
                0,
                0,
                360,
                WHITE,
                -1,
            )
    read = bank.read_number_detail(image)
    _assert_contract(read)
    assert read.value is None, f"a textless {kind} {width}x{height} crop read {read.value!r}"
    assert read.value != 0.0
