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

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "cv_lab" / "models" / "ocr_templates.npz"

# Normalized glyph / word canvas sizes (rows, cols).
DIGIT_SIZE = (22, 16)
WORD_SIZE = (24, 96)

# Pill background colour -> candidate actions (HSV hue ranges). Text is definitive;
# colour only breaks ties / fills gaps when the word template is uncalibrated.
PILL_VOCAB = ("check", "call", "bet", "raise", "fold", "all-in", "post_blind")

# The pill action a GREEN background carries when the word template misses. The
# client paints CALL and BET on the same green, so the colour alone cannot choose
# between them and this token says exactly that. Resolving it to "call" -- the old
# "safe default" -- is not safe: it is a positive claim of "somebody bet before
# me", and on the development corpus it deleted an observed FOLD and fabricated a
# CHECK in its place (see region_detections.read_pill_action).
PILL_BET_OR_CALL = "bet_or_call"


# --------------------------------------------------------------------------- #
# Binarization + segmentation
# --------------------------------------------------------------------------- #
def binarize_text(crop_bgr: np.ndarray, v_min: int = 150, s_max: int = 90) -> np.ndarray:
    """Isolate near-white text (high value, low saturation) -> uint8 {0,255} mask.

    Drops the green chip icon, coloured pill fills, and dark backgrounds, keeping
    the white glyph strokes shared by pot/stack/bet/pill text.

    `v_min` / `s_max` ARE ABSOLUTE PHOTOMETRIC THRESHOLDS and they gate every
    predicate in read_number_detail's contract, which is why they are disclosed
    here rather than left as bare defaults. They are neither structural nor a
    ratio of two like-scaled quantities, so they are a SECOND domain precondition
    on the template bank alongside P7's render band: the bank is calibrated for
    the ClubWPT client's own near-white HUD text as captured by a screen
    recorder, and a recording graded, tone-mapped or re-encoded away from that is
    outside its calibration.
    Measured direction of the exposure over 1,736 value-producing crops x 11
    monotone photometric transforms (gains 0.75-1.30, gammas 0.8-1.25) = 19,096
    reads: CONFIDENT WRONG 0, in every cell. The reader fails CLOSED and the cost
    is coverage, steeply and monotonically in distance from the calibrated
    render: 17 refusals at gain 0.95, 95 at 1.05, 1,090 at 1.15, 1,215 at 1.30,
    1,104 at gamma 0.8. Retained unchanged and disclosed: the failure direction
    is the intended one, and there is no ratio available that could express an
    absolute brightness precondition."""
    if crop_bgr.size == 0:
        return np.zeros((0, 0), np.uint8)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    return (((v > v_min) & (s < s_max)).astype(np.uint8)) * 255


# The reader's CONTRACT, as a closed vocabulary. `AmountRead.decimal_source` is
# always exactly one of these tokens and the two halves are DISJOINT: a member of
# DECIMAL_EVIDENCE always accompanies a value, a member of REFUSAL_CODES always
# accompanies `value is None`. Nothing else is a legal value of the field.
#
# The reader returns a number only when the read is PROVABLY unambiguous; every
# other outcome is UNKNOWN with a named reason. UNKNOWN is a first-class value --
# it is not 0.0, not "None meaning zero", and must not be silently dropped
# downstream (PLAN.md: "Treat absent amounts as unknown, not zero").
DECIMAL_EVIDENCE = frozenset({
    "dot",       # a baseline separator was LOCATED inside the run and reconciled
    "integer",   # no separator candidate exists AND the run's widest inter-digit
                 # gap is below the band a real decimal has ever occupied
})

REFUSAL_CODES = frozenset({
    "no_digit_run",                 # nothing in the crop is a run of confident digits
    "ambiguous_longest_run",        # two or more runs tie for longest (P3)
    "unexplained_ink_in_numeral",   # ink on the numeral's own row that the read
                                    # cannot explain: inside the numeral's letter
                                    # spacing, or glyph-scale ink anywhere on the
                                    # row with no anchored explanation (P2)
    "suffix_not_bb",                # the numeral is not terminated by "BB" (P1)
    "run_clipped",                  # the run touches a crop edge, or begins closer
                                    # to the left edge than its own letter spacing
                                    # with no surviving ink to prove the edge did
                                    # not sever a digit (P4)
    "separator_unreconciled",       # a baseline separator candidate exists whose
                                    # reading does not reconcile, or two candidates
                                    # reconcile to different values (P5a)
    "integer_over_decimal_band",    # no separator, but the widest gap reaches the
                                    # band a located decimal occupies (P5b)
    "unexplained_gap_in_numeral",   # a separator was located, but the run carries
                                    # ANOTHER gap wide enough that a located decimal
                                    # has occupied it -- a digit-sized hole the read
                                    # cannot explain (P5b, dot arm)
    "leading_zero_no_dot",          # "050" is a "0.50" whose dot was lost (P6)
    "below_calibrated_render_size",  # run height under the calibrated band (P7)
    "above_calibrated_render_size",  # run height over the calibrated band (P7)
    "reader_unavailable",           # no template bank is calibrated; set by the
                                    # Detection layer, never returned from here
})


@dataclass(frozen=True)
class AmountRead:
    """A numeric HUD read, or a NAMED refusal to read.

    `value is None` is the UNKNOWN channel: the read could not be PROVEN, and
    `decimal_source` says which condition failed. It is distinct from a confident
    `0.0`, which is what an all-in seat showing "0 BB" genuinely reads --
    downstream must never conflate the two (a missing stack is unknown, not zero).

    `decimal_source` is drawn from one of two disjoint closed sets:
      * DECIMAL_EVIDENCE ("dot" | "integer")  <-> `value is not None`
      * REFUSAL_CODES                          <-> `value is None`
    There is no token that can mean both, which is the conflation this contract
    exists to end: the retired "none" was returned both for a proven integer and
    for a crop containing no text at all."""

    value: float | None
    raw: str
    score: float          # mean digit-template score over the accepted run; 0.0 when none
    decimal_source: str   # DECIMAL_EVIDENCE when value is not None, else REFUSAL_CODES
    digits: int


@dataclass
class Glyph:
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray  # bool, cropped to bbox


# The module's own speck-vs-glyph boundary: components below EITHER floor are
# too small to be a rendered glyph at any supported HUD size and are treated as
# noise by default segmentation. `read_number_detail` deliberately segments
# BELOW these floors (min_area=2, min_h_px=1) so the decimal dot survives, which
# is why the same floors reappear inside P2's far arm: ink that fails them is
# compression noise / border ringing the row can legitimately carry anywhere,
# while ink that MEETS them is glyph-scale and must be explained (see
# _unanchored_row_ink). These are the values segment_glyphs has always used;
# they are named so the two call sites provably share one boundary.
_GLYPH_MIN_AREA = 5
_GLYPH_MIN_H_PX = 4


def segment_glyphs(mask: np.ndarray, min_area: int = _GLYPH_MIN_AREA,
                   min_h_px: int = _GLYPH_MIN_H_PX) -> list[Glyph]:
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


# RUN ADJACENCY (P2). Largest inter-glyph gap that is still INSIDE a numeral, as
# a fraction of the digit run's own glyph height -- a ratio of two quantities that
# carry the same scale factor, so it cannot be moved by render size. Measured over
# 3843 real >=3-digit reads: every intra-numeral gap, including the decimal
# point's own (the widest of them), lands at or below 0.250, while the "POT:"
# prefix and the "BB" suffix are separated by a word space of 0.55 and up. On the
# six-recording development corpus the integer band's own maximum is exactly
# 0.2500, so 0.28 is 1.12x the widest measured intra-numeral gap.
#
# Ink closer than this to the run belongs to the numeral. If it was not accepted
# into the run and is not a PROVEN affix, the read is UNKNOWN -- there is no arm
# that rescues it. This constant only ever refuses.
#
# It used to be one half of a shared constant whose other half -- a decimal
# INFERRED from spacing -- is deleted. The two wanted opposite margins, and the
# inference arm was the mechanism behind "18.30 BB" -> 1830.88 (it split at the
# WORD SPACE before the suffix). P5(b) below states the same fact as a refusal.
_RUN_ADJACENCY_GAP = 0.28

# DECIMAL BAND FLOOR (P5b). An integer read asserts "this numeral has no
# fractional part". That assertion is only admissible when the run's widest
# inter-digit gap is NARROWER than any gap a located decimal has ever occupied.
#
# Measured over the reader's own record of all six development recordings
# (18006 crops), max_gap / run_h -- again a ratio in which render scale cancels.
# REPRODUCED against the shipped reader of this round (the previous table's
# n=13900/n=1612 was measured at a code state before P1 existed and could not be
# regenerated from the code it sat beside):
#
#                                          n     min      p1     p50     p99     max
#   gap occupied by a LOCATED decimal   13669  0.2308  0.2857  0.4091  0.5238  0.5833
#   widest gap of a VALUE-PRODUCING
#   no-separator integer                 1541  0.0000  0.0000  0.1667  0.2143  0.2273
#
# The RAW bands overlap over [0.2308, 0.2500] -- pre-refusal, 47 integer reads
# landed inside it and are exactly the integer_over_decimal_band refusals, which
# is why the surviving integer population now tops out at 0.2273, strictly below
# the boundary. Inside the overlap an integer with no located separator is
# indistinguishable from a decimal whose dot was lost to binarization, so the
# honest boundary is the LOWER edge of the decimal band, not the round 0.25 that
# would sit inside the ambiguity.
#
# The constant is the extremum ITSELF, kept as the exact ratio 3/13 rather than as
# its rounded print 0.2308: the narrowest gap a located decimal has ever occupied
# is a 3px gap inside a 13px run on the 1272x896 client, and one integer read
# (g0723b stack_text t=122) sits on exactly that ratio. Transcribing the boundary
# as 0.2308 admits that read by 4e-5 of run height, which is a rounding artefact
# rather than evidence. The comparison is `>=`, so a gap AT the extremum -- a
# measurement a decimal has been seen to produce -- refuses.
_DECIMAL_BAND_MIN_GAP = 3 / 13

# INTRA-NUMERAL GAP CEILING (P5b, dot arm). The widest inter-glyph gap a real
# numeral has ever shown inside itself -- INCLUDING the decimal point's own gap,
# which is the widest of them -- is exactly 0.250 of run height (the same
# measurement _RUN_ADJACENCY_GAP is derived from, over 3843 real >=3-digit
# reads). A read whose separator is already LOCATED has no lost-dot ambiguity
# left: the client renders at most one separator, so the remaining holes can
# only be letter spacing (measured <= 0.250) or a MISSING DIGIT -- occlusion or
# dropout -- whose measured holes start at a digit's own width plus two letter
# gaps, ~2x this ceiling. The comparison is strict (`>`): the extremum itself is
# a measurement a real numeral produced, so a hole AT it is admissible, and a
# hole above it has no measured producer other than a hole in the numeral.
# Refusal-only, like every constant in this file.
#
# Why the integer branch does NOT use this ceiling: with no separator located, a
# gap in [_DECIMAL_BAND_MIN_GAP, 0.250] is inside the band where a legitimate
# letter space and a decimal whose dot was lost to binarization are the same
# measurement, so the integer claim refuses from the decimal band's lower edge.
# With the separator located, that ambiguity is spent.
_INTRA_NUMERAL_MAX_GAP = 0.25

# Deepest fractional part ClubWPT renders. Measured over 14390 real reads across
# the 5 development geometries: 0, 1 and 2 places occur, 3 never does. A split
# that would leave more is not a decimal point -- it is a thousands separator,
# which sits in the same place and has the same silhouette ("12,345" -> 12.345,
# a 1000x under-read).
_MAX_FRACTIONAL_DIGITS = 2

# A digit is taller than it is wide at every HUD render size. Without this test a
# crop containing NO TEXT AT ALL returned a confident 0.0, byte-identical in every
# field to the genuine all-in "0 BB" read; stack_text 0.0 is trusted by the spine
# and a zero stack labels the seat's action "all-in".
#
# Re-measured over 276307 glyphs that were accepted into the WINNING run of a
# value-producing read -- 11996 real crops from the 5 development geometries, each
# re-read at 0.70x/0.80x/0.85x/0.90x/1.00x/1.20x. That is the population the gate
# actually governs; the earlier 47121-glyph figure pooled every band in the crop,
# including the 'O' of "POT:" and chip sprites, and so measured its own confusers
# as if they were digits. In the winning-run population w/h has p99 = 0.800,
# p99.9 = 0.875 and exactly SEVEN glyphs above 0.900 -- all at 1.000, all of them
# chip annuli or the 'O' of "POT:", five of which produce the false 0.0 this gate
# exists to stop.
#
# So the old 1.0 was inclusive at exactly the confuser's own value: a chip's pale
# annulus rounds to precisely square at reduced render size and passed. 0.95 is
# 1.056x the widest real digit measured anywhere in the sweep and strictly below
# the 1.000 floor of every measured confuser.
_MAX_DIGIT_ASPECT = 0.95

# Smallest rendered digit height, in pixels, at which this template bank has been
# shown to read correctly. THE CALIBRATION RANGE, expressed in the quantity that
# actually governs the reader rather than in client window size: glyph height is
# independent of detector box padding, of the crop's own dimensions and of the
# recording's resolution, so one floor covers every geometry.
#
# Measured over all 14193 value-producing reads at NATIVE size on the 5
# development geometries: run height is 12-14 on the 1272x896 client (the smallest
# supported one) and 20-31 on every other geometry. 12 is therefore the smallest
# height at which this bank has read correctly ON A NATIVE CLIENT, and the
# constant is that measurement and nothing else.
#
# TWO HONESTY CAVEATS on what that measurement licenses:
#   * "in-band" does not imply "sound". On RESAMPLED crops (0.55x-0.60x of
#     larger clients) the bank returns confident wrong values at run height
#     11-12 -- the ':' of "POT:" collapses to a stroke that classifies as '1'
#     (141.5 for a pot of 41.5) and an '8' scores as '5' (162.5 for 162.8). The
#     smallest NATIVE client keeps the colon as two separate dots and reads
#     correctly, so 12 stands as measured -- but it was validated on exactly one
#     client's rendering, and at 12px the bank's glyph discrimination is thin.
#     A new client rendering at 12-14px is calibration work, not a free pass.
#     Round-2 adversarial measurement of the same zone: 8 confident-wrong
#     reads across a 523,770-read AREA/LINEAR scale sweep plus a 2,500-crop
#     interpolation sample, every one on a RESAMPLED crop at run height 12-13
#     (162.8 -> 162.50 at AREA 0.60x with '8' scoring 0.683 as '5';
#     189.1 -> 159.10 at LINEAR 0.60x; NEAREST 0.90x/0.55x/0.60x cases), all
#     single-scale knife edges whose neighbouring scales read correctly. The
#     precondition (native hinted rendering) is not enforceable from the
#     pixels, and raising the floor to exclude the zone would delete the
#     entire calibrated 1272x896 native client (1,138 reads at exactly 12)
#     against zero native failures -- so the residual is RETAINED and
#     disclosed: an operator-resampled or transcoded recording is outside
#     this bank's calibration whatever P7 measures, and coverage claims made
#     inside the band do not cover it.
#   * the band has ZERO measured margin below (1138 g0723b reads sit exactly on
#     the floor) and one pixel above (max measured 31 against a 32 ceiling), so
#     the corpus coverage figures measured inside it say nothing about any
#     geometry outside the six development recordings. A client rendering
#     slightly smaller than 1272x896 loses most of its reads to this floor
#     (measured: 15.18% refused at 0.95x, 73.58% at 0.90x), which is the
#     intended fail-closed direction but must be disclosed next to any coverage
#     number.
#
# It used to be 9, on the claim that "every read stays correct or fails closed
# down to run height 9, and EVERY confident wrong value in the whole 0.60x-2.10x
# sweep sits at 7 or 8". That claim was false. Re-measured over 14390 real crops x
# 17 render scales, 893 reads return a confident value disagreeing with the
# audited 1.0x reference, and 757 of them sit at run heights 9, 10 and 11 -- i.e.
# inside the range the constant declared calibrated. 9 was 0.75x the smallest
# measured render, an extrapolation nothing supported.
#
# Raising it to 12 fails closed on 0 of those 14193 native reads and removes 757
# of the 893 confident-wrong values from the sweep. Below the floor the honest
# answer is that this bank was never calibrated for the size, so the read is
# UNKNOWN rather than a guess: the failures down there are not fail-closed by
# nature, they are 10x-100x values with full confidence, and one of them (0.0) is
# a positive fact the spine reads as all-in.
#
# THIS IS THE ONE CONDITION THAT IS NOT SCALE-RELATIVE, deliberately, and it is
# not dressed up as if it were. Resolution is an absolute-pixel fact about the
# TEMPLATE ARTEFACT, not about the image's geometry: run_h / anchor_scale,
# run_h / crop_h and run_h / frame_h are all invariant to exactly the quantity
# that breaks, so none of them can express it. It is a domain precondition on the
# bank, not a claim that the pixels prove something -- and like everything else
# retained here, it only ever refuses.
_MIN_CALIBRATED_RUN_H = 12

# The other edge of the same band. 32 is the largest run height the constants in
# this module were fitted on (note 11: run height spans 12-32 px over the
# value-producing corpus). Above it the bank is extrapolating just as much as it
# is below 12, and the development corpus shows what that costs: 16 `cwpt01`
# reads at run heights 45-60 are a SPRITE FRAGMENT rather than text, and they ship
# a confident 7.0 for a stack the screen renders as 1131.90 BB.
_MAX_CALIBRATED_RUN_H = 32

# --------------------------------------------------------------------------- #
# THE SEGMENTATION MECHANICS' OWN RATIOS. These four were inline literals with no
# name and no measurement, one of them (0.78) numerically equal to
# _AFFIX_MAX_REL_H five lines under a comment asserting that constant was
# "retained NOWHERE ELSE" -- true of the name, false of the value. They are named
# and tabulated here so a future change to one does not silently need to be
# mirrored in an anonymous twin. All four are ratios against a height measured in
# the same crop, so render scale cancels. Populations below are the 17,334
# value-producing crops of the six development recordings unless stated.
# --------------------------------------------------------------------------- #

# Row-band membership: a component at least this tall (relative to the crop's
# tallest) is a candidate glyph -- a digit or a same-height letter. Measured
# h/max_h over all 156,946 components: this floor sits INSIDE a populated region
# (21,072 components land in [0.45, 0.65)), so it is not a gap in the data and is
# not presented as one. It is a bound on what can join a digit RUN, and a
# component below it is not thereby ignored: it stays visible to P2 as policed
# ink (_policed_ink adds every component overlapping the numeral's row) and to
# P5 as a separator candidate.
_BAND_MIN_REL_H = 0.55

# The separator-candidate bucket: short AND narrow relative to the crop's
# tallest component. Measured over the 36,963 bucket members on decimal-reading
# crops, h/max_h tops out at 0.4762 and w/max_h at 0.5455, i.e. both bounds sit
# at their own measured extremum and neither is approached by a real decimal
# point (a located separator measures 0.0769-0.2381 of RUN height; see
# _DOT_MIN_BASELINE_POS). Widening either bound admits glyph-scale ink into the
# population P5 splits on, which is the dot-forgery direction.
_SHORT_MAX_REL_H = 0.5
_SHORT_MAX_REL_W = 0.55

# Row-band split: consecutive glyph y-centres further apart than this are
# different TEXT ROWS (a stack box can carry the player's name above the value).
# Measured over 96,234 consecutive pairs in the value-producing crops, split by
# vertical overlap: within one row the y-centre scatter never exceeds 0.3182 of
# max_h, and two genuine text rows are never closer than 1.0 (the two pairs
# between those figures, at 0.50 and 0.52, are a bet pill's chip icon against
# its digits, not a second text row). 0.6 is 1.89x the widest measured
# within-row scatter and 0.6x the tightest measured row separation, i.e. it sits
# in an empty window three times wider than either margin.
_ROW_BAND_MAX_REL_GAP = 0.6

# Digit-height floor inside a row band: a glyph shorter than this fraction of its
# band cannot join the digit run. Measured h/band_h over the 70,463 glyphs
# actually ACCEPTED into a winning run: min 0.8462, p1 0.9091, median 1.0000 --
# so the floor clears the smallest real digit by 0.066 of band height. The
# population it excludes is real: 1,674 glyphs the bank labels a digit at
# >= min_score sit outside a winning run with h/band_h as low as 0.6923, and a
# digit CLIPPED or OCCLUDED is short precisely because it is cut (that is what
# makes it a fragment rather than a value).
#
# It is the numerical complement of _AFFIX_MAX_REL_H over glyph height, and that
# is a coincidence of two independently measured populations, not a shared
# constant: this floor is the bottom of the accepted-digit distribution, that one
# is the top of the "BB" cap distribution on five of six recordings. Changing
# either does NOT imply changing the other, which is exactly why both now have
# names.
_MIN_DIGIT_REL_H = 0.78

# A decimal point sits ON the digit run's baseline. Measured over 9088 real
# decimal reads, the dot's centre lands at 0.846-0.958 of the run band height
# (0 = the digits' tops, 1 = their baseline); not one is below 0.75. The old test
# accepted ANY vertical overlap with the run, so a speck level with the digit tops
# -- card-border ringing, a chip-sprite edge, compression noise -- was a valid
# decimal candidate and deflated the value 100x. 0.6 keeps a 1.41x margin under
# the lowest measured real dot.
_DOT_MIN_BASELINE_POS = 0.6


def _largest_hole(lo: float, hi: float, blockers: Sequence[Glyph]) -> float:
    """Width of the widest EMPTY span in [lo, hi] once `blockers` are subtracted."""
    if hi <= lo:
        return hi - lo
    spans = sorted((max(lo, b.x), min(hi, b.x + b.w))
                   for b in blockers if b.x < hi and b.x + b.w > lo)
    widest, cursor = 0.0, float(lo)
    for s0, s1 in spans:
        widest = max(widest, s0 - cursor)
        cursor = max(cursor, float(s1))
    return max(widest, hi - cursor)


def _baseline_shorts(run: list[tuple], shorts: Sequence[Glyph]) -> list[Glyph]:
    """The short glyphs that plausibly belong to `run` -- i.e. sit on its baseline.

    Same test the decimal-point search uses, and for the same reason: a speck that
    merely shares the numeral's x-range while sitting at the digit TOPS or above
    the crop is not part of it. Bridging with every short component in the crop
    instead let a 1px speck on the frozen "218 BB" crop's top edge bridge the word
    space to the "BB" caps and truncate a complete numeral.
    """
    if not run:
        return []
    run_y0 = min(g.y for g, _, _ in run)
    run_y1 = max(g.y + g.h for g, _, _ in run)
    span = max(1.0, float(run_y1 - run_y0))
    return [d for d in shorts
            if d.y < run_y1 and (d.y + d.h / 2.0 - run_y0) >= _DOT_MIN_BASELINE_POS * span]


def _bridged_gap(g: Glyph, x0: float, x1: float, shorts: Sequence[Glyph]) -> float:
    """Gap from `g` to the digit run spanning [x0, x1], with the numeral's own
    short glyphs (the decimal point) bridged out.

    Measuring the raw span is what made the truncation net blind at the decimal
    boundary. A digit lost immediately BESIDE the decimal leaves the dot sitting in
    the space between the rejected glyph and the surviving run, so the raw gap
    carries the dot's width plus BOTH of its sub-gaps: measured 0.333-0.429 of
    run_h on real crops, above the 0.28 word-break floor, so the net called the
    fragment a complete numeral. Real examples, both confident and both wrong:
    "19.50" -> 50.0 at 0.90x of the smallest supported geometry, and ".60 BB" (a
    clipped seat panel) -> 60.0 for three consecutive samples on the BASELINE
    2054x1470 recording. Bridging the dot puts the same two cases at 0.08-0.17 of
    run_h, i.e. unambiguously inside the numeral.

    WHAT IS *NOT* BRIDGED, stated because the near arm's scope is easy to
    misread: only the numeral's own baseline separator candidates. An OCCLUDER
    lying between `g` and the run is NOT bridged, so its own width counts toward
    the measured gap and a component screened by one reads as "far". That is
    deliberate — the near arm's question is "does this sit within the numeral's
    letter spacing", and a component a digit's width away from the run does not,
    whatever is in between. The screened component is covered by P2's FAR arm
    (_unanchored_row_ink), which polices glyph-scale ink anywhere on the row.
    Measured: subtracting every occupied span here instead, so that an occluder
    is bridged too, changes 0 of 18,006 native development reads and 0 of
    187,261 real-sprite occlusion variants -- the far arm already refuses every
    case it would newly catch, and inert machinery behind a load-bearing comment
    is exactly what this file is trying to stop carrying.
    """
    if g.x < x0:
        return _largest_hole(g.x + g.w, x0, shorts)
    return _largest_hole(x1, g.x, shorts)


# Relative height below which a neighbour is not the numeral's own glyph. The
# "BB" suffix butts right up against the value: measured horizontal gaps of 2-6px
# against run heights of 22-28px, i.e. 0.07-0.27 of run_h, which is INSIDE the
# letter-spacing floor. So the gap test alone cannot separate the suffix from the
# numeral, and dropping the height test absorbed the "BB" of "198 BB" as an '8' on
# 55 real reads.
#
# WHAT THIS CONSTANT DOES NOT DO, stated because the previous comment claimed the
# opposite. It read "the 'BB' suffix renders at 0.59-0.77 of the digits' height",
# and that is false on one of the development recordings. Re-measured over 35185
# glyphs the bank itself labels 'B' at >= 0.55 confidence, on the current
# retained samples of ALL SIX development recordings (the previous table's g0621
# row, n=8808, came from an older 361-frame window, and cwpt01 was missing from
# a table presented as the development corpus):
#   rec      n_B     min    p50    p99    max   >= 0.78
#   g0723a  7933   0.591  0.619  0.773  1.000       5  (0.06%)
#   g0621   4020   0.591  0.636  0.762  0.762       0
#   g0711   6944   0.586  0.643  0.759  0.786       2  (0.03%)
#   g0715    435   0.667  1.000  1.000  1.000     383  (88.05%)
#   g0723b  3959   0.615  0.667  0.769  1.000      16  (0.40%)
#   cwpt01 11894   0.591  0.636  0.773  1.000       7  (0.06%)
# On g0715 the suffix renders at the FULL band height, so this gate excludes it
# from `named` on essentially every crop there and contributes nothing at all; the
# separation on that geometry rests entirely on the 0.28 adjacency gap.
#
# AND WHAT IT DOES DO, stated because a prior comment called it "refusal-only",
# which is false in the direction that matters: passing this gate puts a glyph
# into `named`, `named` exempts the glyph from P2, and that exemption is what
# ADMITS the read. Ablating the gate to 0.0 turns 57 development reads (55 on
# g0723a) from a value into UNKNOWN -- those values exist only because this
# threshold admitted their suffix. What remains true is the direction of its
# failure modes: set too LOW it only costs coverage; set too HIGH it absorbs
# real digits into `named` and silences P2, which is why the permissive end is
# pinned by the suffix-absorption tests and the admitting end by
# test_affix_gate_is_value_admitting_and_pinned (a real "198 BB" crop).
#
# The consequence when both defences are thin is a suffix absorbed into the digit
# run as "88" -- measured on the real "18.30 BB" crop at 1.10x of the 1272x896
# client, where the B's h/band_h crosses 0.769 -> 0.786. That read no longer ships
# a value: the digits become "183088", the true dot's split leaves 4 fractional
# places, and the fail-closed guard for a located-but-unreconcilable separator now
# runs BEFORE the decimal-gap arm and returns unknown (see read_number_detail).
# Raising this constant is not the fix -- above ~1.0 it starts absorbing real
# digits -- so the honest statement is that height alone does not separate the
# suffix on every supported client, and the reader must fail closed when it cannot.
_AFFIX_MAX_REL_H = 0.78


def _policed_ink(run: list[tuple], band: list[Glyph], comps: Sequence[Glyph]) -> list[Glyph]:
    """Every component P2 must be able to explain: the numeral's own row.

    THIS IS THE SET, and getting it wrong is what made P2 structurally blind.
    `segment_glyphs` output was split three ways at the top of read_number_detail
    -- `tall` (h >= 0.55*max_h), `dots` (h < 0.5*max_h AND w <= 0.55*max_h), and an
    unnamed remainder that is NEITHER -- and P2 iterated only the row band built
    from `tall`. So ink that was merely SMALL, or small-and-wide, could not be seen
    by any predicate at all: `dots` was consulted solely as a separator candidate
    strictly inside the run and as a bridge, and the remainder by nothing.

    That is exactly the shape occlusion and box-clipping produce. A leading digit
    75% covered leaves a sliver that lands in `dots`; a card sprite over a digit
    leaves a wide low fragment that lands in the remainder. Measured on the six
    development recordings by removing information from real crops -- which can
    only ever move a read from a value to UNKNOWN -- the old set returned 60982
    confident WRONG values under left-edge clipping (9628 of 17469 value-producing
    crops) and 13485 of 15408 under leading-digit occlusion, while right, top and
    bottom clipping produced zero. The asymmetry localised the hole precisely:
    everything the numeral loses on its left survives as sub-band ink.

    So the policed set is the row band PLUS every non-`tall` component that
    vertically overlaps the run. `tall` glyphs outside the run's own band are still
    excluded, deliberately and unchanged: a stack box can carry the player's NAME
    on the row above, and those glyphs are a different row, not unexplained ink.
    """
    if not run:
        return []
    run_y0 = min(g.y for g, _, _ in run)
    run_y1 = max(g.y + g.h for g, _, _ in run)
    banded = {id(g) for g in band}
    extras = [c for c in comps
              if id(c) not in banded and c.y < run_y1 and c.y + c.h > run_y0]
    return list(band) + extras


def _numeral_intruders(run: list[tuple], policed: Sequence[Glyph],
                       shorts: Sequence[Glyph],
                       named: frozenset[int] = frozenset(),
                       explained: frozenset[int] = frozenset()) -> list[Glyph]:
    """Components that sit INSIDE `run`'s numeral but were not accepted into it --
    ink the read cannot explain. Any one of them makes the read UNKNOWN (P2).

    A component belongs to the numeral when it sits within the numeral's own
    letter-spacing, measured with the decimal point bridged out (see _bridged_gap).
    Exactly two things explain a component that is not in the run:

      `named`     -- the template bank matched it confidently as a NON-DIGIT and it
                     is short relative to the row band: the "BB" caps. This is the
                     only available way to tell the suffix from a numeral glyph,
                     since the suffix is not separated from the value by a word
                     space at all.
      `explained` -- it is a legal baseline separator candidate STRICTLY INSIDE the
                     run, i.e. a member of the same population P5 reconciles. The
                     numeral's own decimal point is ink the read does explain; a
                     baseline dot OUTSIDE the run's span is not, because a decimal
                     preceding the first accepted digit means the integer part is
                     missing (".50 BB" -> a confident 50.0 for a stack of 99.50).

    Height alone is not the affix test. A leading digit CLIPPED by a tight detector
    box is short precisely because it is cut, and it is the glyph the net most
    needs to see: the frozen "71.20 BB"-clipped-to-"20 BB" crop clears a fixed 0.78
    height gate at 1.00x by 0.006 of band height and fails it at 1.10x, where the
    reader returns a confident 0.0 for a stack that is really 71.2. Such a fragment
    has no confident template match, so it is never in `named`.

    The gap test has NO lower bound. `_bridged_gap` returns a NEGATIVE number for a
    glyph that OVERLAPS the run, and the old `0 <= gap` form therefore never saw
    one -- an unaccepted glyph literally on top of the numeral was the one kind of
    intruder the net ignored. Measured over 17785 value-producing reads on the six
    development recordings, exactly ONE read has an overlapping unaccepted glyph,
    and it is g0715 pot_text t=6: a confident 2.0 for a pot the screen renders as
    240.9, a 120x under-read that survived every other check and every one of 27
    constant ablations.
    """
    if not run:
        return []
    run_h = max(g.h for g, _, _ in run)
    x0 = run[0][0].x
    x1 = run[-1][0].x + run[-1][0].w
    accepted = {id(g) for g, _, _ in run}
    bridge = _baseline_shorts(run, shorts)
    out: list[Glyph] = []
    for g in policed:
        if id(g) in accepted or id(g) in named or id(g) in explained:
            continue
        if _bridged_gap(g, x0, x1, bridge) < _RUN_ADJACENCY_GAP * run_h:
            out.append(g)
    return out


def _run_is_truncated(run: list[tuple], policed: Sequence[Glyph],
                      shorts: Sequence[Glyph] = (),
                      named: frozenset[int] = frozenset(),
                      explained: frozenset[int] = frozenset()) -> bool:
    """True when the numeral carries ink the read cannot explain (P2) -- because
    `run` is a FRAGMENT of a longer numeral, or because something is sitting on it.

    ``classify_digit`` pools the affix glyphs (B/P/T, chip icon) with the digits so
    that a real affix BREAKS the digit run instead of joining it. That is correct
    for an affix and silently destructive for a digit: at 12px glyph height the '8'
    of "218 BB" scores 0.675 as '8' against 0.677 as 'B', so the run truncates to
    "21" and a confident 21.0 is returned -- a 10x under-read that shipped into an
    export as a player's starting stack.

    This is a HARD REFUSAL, not a ranking key. It used to be the leading term of a
    `(complete, len, row_y, score)` sort over competing runs, which is how a
    2-digit seat-timer badge beat the 5-digit stack beside it ("212.90 BB" + a
    timer ring reading "12" -> a confident 12.0). Selecting among competing
    readings by score is exactly the kind of rescue this contract removes; P3
    replaces it with "the unique longest run, or UNKNOWN".

    Deliberately makes no exception for the chip icon: a leading digit the chip
    template out-scores is exactly the 10x error this rule exists to stop.
    """
    return bool(_numeral_intruders(run, policed, shorts, named, explained))


def _xspan_gap(a: Glyph, b: Glyph) -> float:
    """Horizontal gap between two components' x-spans; 0 when they overlap."""
    if a.x + a.w <= b.x:
        return float(b.x - (a.x + a.w))
    if b.x + b.w <= a.x:
        return float(a.x - (b.x + b.w))
    return 0.0


def _host_gap(center_x: float, gx: list[tuple[int, int]]) -> float:
    """Width of the run's inter-digit gap CONTAINING `center_x`, or 0.0 when the
    point sits inside an accepted digit's own x-span (no gap hosts it)."""
    for (_, a1), (b0, _) in zip(gx, gx[1:], strict=False):
        if a1 <= center_x <= b0:
            return float(b0 - a1)
    return 0.0


def _unanchored_row_ink(run: list[tuple], policed: Sequence[Glyph],
                        named: frozenset[int] = frozenset(),
                        explained: frozenset[int] = frozenset(),
                        confident: frozenset[int] = frozenset()) -> list[Glyph]:
    """Glyph-scale ink on the numeral's row that NO anchored chain explains --
    the FAR arm of P2, and any one of them makes the read UNKNOWN.

    _numeral_intruders polices ink within the numeral's own letter spacing; this
    polices the REST of the row, because distance used to be an exemption: a chip
    sprite sliding over the leading digits leaves fragments that end well outside
    the adjacency window of the surviving fragment-run, and with those fragments
    invisible to every predicate the surviving trailing digit shipped as a
    confident 0.0 (booked by the spine as an all-in) for a stack the screen
    rendered as 343.60 BB. The refusing evidence -- mask components the bank
    confidently matches to NOTHING -- was present and unused.

    What legitimately sits far from the run on a real row, measured over all
    17,459 value-producing development reads: the "POT:" prefix (letters the bank
    matches confidently, plus the colon's dots), the "BB" suffix, the chip icon,
    and scattered 1-2px compression specks. So a far component is explained by
    exactly one of:

      * being SUB-GLYPH-SCALE -- it fails the module's own speck floors
        (_GLYPH_MIN_AREA / _GLYPH_MIN_H_PX, the segment_glyphs defaults): noise
        the row carries everywhere, dangerous only as a separator candidate or
        inside the numeral, both of which are policed elsewhere; or
      * lying in the transitive closure, under the numeral's own letter-spacing
        gap (_RUN_ADJACENCY_GAP, no new constant), of an ANCHOR: an accepted run
        glyph, a proven affix, a separator candidate, or a glyph the bank matches
        at >= min_score as a NON-DIGIT. That is what a rendered word IS -- a
        chain of glyphs at letter spacing -- and it explains the colon dots
        (chained to the confident 'T' beside them) and the chip sprite's ragged
        pieces (chained to its confidently-matched core) without explaining an
        isolated occluder fragment, which chains to nothing.

    A CONFIDENTLY-CLASSIFIED DIGIT IS NOT AN EXPLANATION, and admitting one was
    a hole big enough to ship a 170x under-read. `confident` used to mean "the
    bank matched it under ANY label", so a full-height glyph the bank read as
    '3' at 0.952, sitting on the numeral's own row outside the winning run, was
    its own explanation -- and it further laundered the occlusion sliver beside
    it into the chain. Measured on the real chip sprite laid over one interior
    digit: "392.30 BB" shipped 2.30 at score 0.933 and "249.30 BB" shipped 9.30
    at 0.911. A digit outside the winning run is the DEFINITION of a numeral the
    read has fragmented (P3 already refuses when two runs tie for longest), so
    it is the one label that can never anchor. The 'O' of "POT:" -- which the
    bank does label '0' -- keeps its explanation by chaining to the 'P' and 'T'
    beside it at 2-3px, well inside the letter-spacing window.

    Measured cost of the whole arm on the six development recordings: 125 of
    17,393 value reads (0.72%) refuse -- 66 before the digit-anchor repair and
    59 added by it -- every one carrying glyph-scale ink on the numeral's row
    that the bank cannot name, which is exactly the ambiguity (an occluded or
    severed glyph) this arm exists to refuse. Confident-anchor membership does
    NOT weaken the near arm: a confident full-height 'B' inside the adjacency
    window is an intruder there regardless of what it anchors here.
    """
    if not run:
        return []
    run_h = max(g.h for g, _, _ in run)
    window = _RUN_ADJACENCY_GAP * run_h
    accepted = {id(g) for g, _, _ in run}
    anchor_ids = accepted | set(named) | set(explained) | set(confident)
    anchored = [g for g in policed if id(g) in anchor_ids]
    pending = [g for g in policed if id(g) not in anchor_ids]
    changed = True
    while changed and pending:
        changed = False
        for g in list(pending):
            if any(_xspan_gap(g, e) <= window for e in anchored):
                anchored.append(g)
                pending.remove(g)
                changed = True
    return [g for g in pending
            if g.w * g.h >= _GLYPH_MIN_AREA and g.h >= _GLYPH_MIN_H_PX]


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
    def load(cls, path: Path | str = DEFAULT_TEMPLATE_PATH) -> TemplateOCR | None:
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
    def read_number_detail(
        self,
        crop_bgr: np.ndarray,
        min_score: float = 0.55,
        *,
        skip_unexplained_ink: bool = False,
        allow_b8_suffix: bool = False,
    ) -> AmountRead:
        """Read a HUD amount, or NAME the reason the read cannot be proven.

        Recovery-only flags (small-render entrypoint):
          * ``skip_unexplained_ink`` -- waive P2 when multi-scale consensus agrees
          * ``allow_b8_suffix`` -- treat B/8 terminator confusions as "BB" so P5
            can still prove the decimal; never invent a split from digit count
        P3/P4/P5/P6/P7 stay mandatory.

        THE CONTRACT. A value is returned only when every condition below holds.
        In every other case the answer is UNKNOWN with a refusal code. Lower
        coverage is an accepted outcome; a confident wrong number is not. Each
        condition below is either structural or a ratio of two quantities carrying
        the same scale factor -- with TWO stated exceptions, both domain
        preconditions on the template bank rather than claims about the pixels:
        P7 (the calibrated render band) and `binarize_text`'s absolute
        `v_min` / `s_max` photometric gate, which runs upstream of every
        condition here and is documented at that function.

        P1  the numeral is TERMINATED BY "BB". The glyphs of the winning run's own
            row band lying to its right begin with exactly B, B, and the only
            thing that may follow them is the chip icon. ClubWPT renders
            `<numeral> BB` on all
            three numeric classes, and the suffix butts against the value at
            0.07-0.27 of run height -- inside the letter spacing -- so it is where
            truncation and occlusion strike, and it is the only token whose
            expected content is known a priori. Measured over 17785 value reads:
            98.31% carry a clean "BB"; the 239 refusals are dominated by "8B"/"B8",
            i.e. crops where the bank cannot tell a 'B' from an '8' in the suffix,
            which is the same discrimination it is relying on inside the numeral.

        P2  NO UNEXPLAINED INK on the numeral's row. Near arm: nothing inside
            the numeral's own letter spacing that is not a proven affix or a
            separator candidate (_run_is_truncated). Far arm: no glyph-scale
            ink anywhere else on the row without an anchored-chain explanation
            (_unanchored_row_ink) -- an occluder's fragments do not stop being
            evidence because they landed far from the surviving run.

        P3  the winner is the UNIQUE LONGEST digit run in the crop. Structural, and
            it replaces the old `(complete, len, row_y, score)` ranking under which
            a shorter fragment could outscore the real value.

        P4  the run does NOT TOUCH OR GRAZE THE CROP BOUNDARY. A clipped run is
            by construction a fragment of unknown length, and a crop edge closer
            to the run's start than the numeral's own letter spacing could have
            severed a leading digit without leaving a fragment. Unconditional:
            no ink inside the crop can prove what lies beyond its edge.

        P5  the decimal is PROVEN, not assumed. (a) there is AT MOST ONE baseline
            separator candidate inside the run -- the client renders one
            separator, so a second candidate is unexplained ink, not a second
            opinion -- and it reconciles; and (b) once the located separator (if
            any) is subtracted, every remaining inter-digit hole is below
            _DECIMAL_BAND_MIN_GAP -- a digit-sized hole with or without a dot
            beside it refuses.

        P6  A LEADING ZERO ONLY EVER PRECEDES THE SEPARATOR: the client renders no
            leading zero on an integer part longer than one digit, so "050" is a
            "0.50" whose dot was lost and "05.50" is a numeral whose leading digit
            was destroyed.

        P7  the run height lies in the bank's CALIBRATED BAND.

        There is no arm anywhere below that produces a value by overruling,
        rescuing or re-scoring a glyph decision. The three that did are gone: the
        digit-only re-classification of an affix glyph (_absorb_adjacent_digits),
        the decimal INFERRED from inter-digit spacing (which fired 0 times in
        18006 native reads), and the candidate ranking.

        Mechanics unchanged: keep full-height glyphs, group them into vertical
        row-bands (a stack box can include the player's NAME on the row above the
        value), and within each band take contiguous runs of confident digits. The
        decimal point is a short glyph on the baseline; it is located by position
        rather than template-matched (a 2-3px dot matches unreliably). The number
        of fractional places is read off the separator's real position, never
        assumed to be 2 -- the 07-15 client renders BB pots with 0 or 1."""
        mask = binarize_text(crop_bgr)
        # The decimal point is a 2-3px component at HUD digit heights of 13-22px
        # (measured dot_h/digit_h = 0.136-0.154 on real ClubWPT crops). The default
        # floors (min_h_px=4, min_area=5) delete it at every render size below
        # ~2050px wide, which is what turned 314.90 into 31490 and 0.50 into 050.
        # min_area=2 is the smallest value that still excludes single-pixel noise.
        # calibrate_ocr.py keeps the defaults; only this reader loosens them.
        comps = segment_glyphs(mask, min_area=2, min_h_px=1)
        if not comps:
            return AmountRead(None, "", 0.0, "no_digit_run", 0)
        max_h = max(c.h for c in comps)
        tall = [c for c in comps
                if c.h >= _BAND_MIN_REL_H * max_h]              # digits + same-height letters
        dots = [c for c in comps
                if c.h < _SHORT_MAX_REL_H * max_h
                and c.w <= _SHORT_MAX_REL_W * max_h]            # ., colon, specks
        if not tall:
            return AmountRead(None, "", 0.0, "no_digit_run", 0)

        # Group glyphs into vertical row-bands by y-center, then read each row: a
        # stack box can include the player's NAME on the row above the value, whose
        # digits ("Lord5699") would otherwise interleave by x with it.
        tall.sort(key=lambda g: g.y + g.h / 2.0)
        bands: list[list[Glyph]] = [[tall[0]]]
        for prev, g in zip(tall, tall[1:], strict=False):
            if (g.y + g.h / 2.0) - (prev.y + prev.h / 2.0) > _ROW_BAND_MAX_REL_GAP * max_h:
                bands.append([])
            bands[-1].append(g)

        # Within each row, take contiguous runs of confident DIGITS. Affix letters
        # (P/T/B of POT:/BB, or name letters) classify as letters and break a run.
        candidates: list[tuple] = []   # (run, band, band_h, labeled, named)
        for band in bands:
            band.sort(key=lambda g: g.x)
            band_h = max(g.h for g in band)
            labeled = [(g, *self.classify_digit(g.mask)) for g in band]
            # The "BB" suffix caps: glyphs whose best template match is a confident
            # NON-DIGIT and which are short relative to the row band. Both halves
            # are load-bearing. Short alone silences P2 on a leading digit clipped
            # by a tight detector box (short because it is cut). Confident alone
            # silences it on the geometries where the caps render at full digit
            # height. A glyph whose best match is a DIGIT is never the suffix.
            #
            # _AFFIX_MAX_REL_H is retained here and NOWHERE ELSE -- and, since a
            # previous form of this comment implied more than it should, its
            # numerical twin has a name of its own: the digit-height floor two
            # lines below is _MIN_DIGIT_REL_H, independently measured, and the
            # two are not one constant. FAILING _AFFIX_MAX_REL_H is
            # refusal-only -- the glyph is no longer routed into a digit-only
            # re-read; it is unexplained ink and the read is UNKNOWN. PASSING it
            # is value-ADMITTING: membership in `named` exempts the glyph from
            # P2, and 57 development reads hold their value only through that
            # exemption (see the constant's own comment).
            # Candidate affixes only -- POSITION is applied once the numeral's
            # right edge is known, because a terminator by definition follows
            # the value (see the `named` narrowing below).
            named = frozenset(id(g) for g, ch, sc in labeled
                              if sc >= min_score and not ch.isdigit()
                              and g.h < _AFFIX_MAX_REL_H * band_h)
            runs: list[list[tuple]] = [[]]
            for item in labeled:
                g, ch, sc = item
                if (ch.isdigit() and sc >= min_score
                        and g.h >= _MIN_DIGIT_REL_H * band_h
                        and g.w <= _MAX_DIGIT_ASPECT * g.h):
                    runs[-1].append(item)
                elif runs[-1]:
                    runs.append([])
            for r in runs:
                if r:
                    candidates.append((r, band, band_h, labeled, named))
        if not candidates:
            return AmountRead(None, "", 0.0, "no_digit_run", 0)

        # ---- P3: the unique longest run, or UNKNOWN --------------------------- #
        longest = max(len(c[0]) for c in candidates)
        at_max = [c for c in candidates if len(c[0]) == longest]
        if len(at_max) > 1:
            tied = "|".join("".join(ch for _, ch, _ in c[0]) for c in at_max)
            return AmountRead(None, tied, 0.0, "ambiguous_longest_run", longest)
        run, band, band_h, labeled, named = at_max[0]
        digits = "".join(ch for _, ch, _ in run)
        score = float(np.mean([s for _, _, s in run]))
        n = len(digits)

        # ---- P7: the bank's calibrated render band ---------------------------- #
        run_h = max(g.h for g, _, _ in run)
        if run_h < _MIN_CALIBRATED_RUN_H:
            return AmountRead(None, digits, score, "below_calibrated_render_size", n)
        if run_h > _MAX_CALIBRATED_RUN_H:
            return AmountRead(None, digits, score, "above_calibrated_render_size", n)

        gx = [(g.x, g.x + g.w) for g, _, _ in run]
        x0, x1 = gx[0][0], gx[-1][1]
        run_y0 = min(g.y for g, _, _ in run)
        run_y1 = max(g.y + g.h for g, _, _ in run)

        # ---- P4: the run does not touch (or graze) the crop boundary ---------- #
        # Touching is not the only way a crop edge severs a digit: a cut landing
        # in the inter-digit GAP removes the leading digit and leaves no fragment
        # at all, and the surviving run then starts a gap's width from the edge --
        # which is AT MOST one intra-numeral letter space (measured max 0.2500 of
        # run_h; see _RUN_ADJACENCY_GAP). So a run whose left margin is inside the
        # letter-spacing band is indistinguishable from one whose leading digit
        # was cut away, and the margin refuses UNCONDITIONALLY. Nothing inside
        # the crop can waive it: every visible component is right of the cut by
        # construction, so surviving ink -- however far left -- proves nothing
        # about what the cut removed (a border speck 3px from the edge "proved"
        # crop intactness on a clip that had just deleted the leading '1' of
        # 191.30, and the read shipped 91.3).
        # Measured on the six development recordings by shifting every crop's left
        # edge inward 1..60px (which leaves the on-screen number unchanged): the
        # old boundary-touch test alone allowed 60982 confident wrong values. The
        # margin is refusal-only and reuses the P2 constant; there is no new
        # threshold here.
        #
        # ALL FIVE CLAUSES ARE LOAD-BEARING, and a previous form of this comment
        # said otherwise: it claimed "right/top/bottom clipping produced zero --
        # the right is defended by P1 and a vertical cut degrades every glyph's
        # template score at once", which credited the wrong mechanism. Disabling
        # only the four boundary-TOUCH clauses (keeping the margin clause, and
        # keeping every other predicate) leaves the 18,006 native reads
        # byte-identical -- so no pipeline run can notice -- while producing 83
        # confident wrong values over a 92,624-read four-direction clip sweep: 23
        # on top clipping, 60 on bottom (162.4 -> 102.4 at 20px, 232.2 -> 90.0 at
        # 26px, 394.4 -> 304.4 at 25px), every one a stack_text read. Right
        # clipping is indeed zero even with the touch test off, but the vertical
        # directions are defended by these clauses and by nothing else.
        # `test_clip_family_never_yields_a_different_value` now sweeps all four
        # sides (its predecessor's docstring claimed "every side" while its body
        # only sliced columns) and pins the vertical arms on frozen crops.
        crop_h, crop_w = crop_bgr.shape[0], crop_bgr.shape[1]
        if (x0 <= 0 or x1 >= crop_w - 1 or run_y0 <= 0 or run_y1 >= crop_h - 1
                or x0 < _RUN_ADJACENCY_GAP * run_h):
            return AmountRead(None, digits, score, "run_clipped", n)

        # The separator-candidate population, needed by BOTH P2 and P5: a real
        # decimal point sits ON the run's baseline, strictly inside the run's
        # span, AND inside an inter-digit gap wide enough that a located decimal
        # has ever occupied it. This is the one kind of non-run ink inside the
        # numeral the read can explain; a baseline dot OUTSIDE the span means the
        # integer part is missing (".50 BB" -> a confident 50.0 for a stack of
        # 99.50) and is policed by P2 like any other intruder.
        #
        # THE HOST-GAP TEST is what makes a candidate a candidate. Without it,
        # any baseline speck strictly inside the run was accepted and only the
        # SPLIT was checked, so a 2x2 compression speck sitting in a 0.18-rel
        # LETTER gap of "197 BB" fabricated "1.97" -- a fully confident 100x
        # under-read, reproducible on ~45% of integer-read crops across all six
        # development geometries. The reader's own calibrated table
        # (_DECIMAL_BAND_MIN_GAP: n=13,669 located decimals, min host gap
        # exactly 3/13) says no real decimal has ever occupied a narrower gap,
        # so a dot in one is provably not a separator; it is unexplained ink and
        # P2 refuses the read. `>=` keeps the measured extremum itself, so every
        # real located-dot read in the corpus keeps its value by construction.
        # No new constant: this is the SAME band the integer branch refuses
        # from, applied to the object whose existence defines the band.
        #
        # FOUR CLAUSES ON THIS PATH CANNOT BE PINNED BY ANY TEST, and that is a
        # measured property, not an oversight: the strictly-inside test below
        # (`x0 < centre < x1`), the `dots` bucket's width bound, `_reconcilable`'s
        # `0 < split` lower bound, and P1's chip-skip are each subsumed by a later
        # predicate on every input reachable today. Ablating each one
        # independently leaves the owned CV suite green AND leaves all 18,006
        # native reads plus a 14,400-read scale/clip sweep byte-identical, so no
        # regression can fail against them. They are retained because each is the
        # PRIMARY statement of a rule the docstrings rely on (a decimal left of
        # the first digit means the integer part is missing; a split must leave
        # digits on both sides), and a subsumption that holds today is not a
        # specification. Note 15's claim that all nineteen acceptance-path
        # predicates are killed by a regression is corrected in note 16 on this
        # point: nineteen are, these four cannot be.
        band_span = max(1.0, float(run_y1 - run_y0))
        sep_dots = [d for d in dots
                    if x0 < (d.x + d.w / 2.0) < x1
                    and d.y < run_y1
                    and (d.y + d.h / 2.0 - run_y0) >= _DOT_MIN_BASELINE_POS * band_span
                    and _host_gap(d.x + d.w / 2.0, gx)
                    >= _DECIMAL_BAND_MIN_GAP * run_h]

        # ---- LOCATE THE TERMINATOR TOKEN -------------------------------------- #
        # P1 tests it and P2 uses its right edge, so it is located once, here,
        # before either. The terminator is the maximal run of band glyphs
        # starting immediately right of the numeral and held together by the
        # numeral's own letter spacing (_RUN_ADJACENCY_GAP -- no new constant):
        # that is what a rendered TOKEN is. Everything past the first word-space
        # break is a different token -- on this client, the chip icon.
        #
        # Locating the token instead of slicing `suffix[:2]` is what closes two
        # severed-digit holes at once. The old form deleted every glyph the bank
        # labelled 'c' and then inspected two survivors, so a trailing digit
        # clipped at the top -- which scores as the chip template -- vanished
        # before the test: shaving 4 rows off the '8' of the frozen "218 BB" crop
        # left ['c','B','B'], filtered to ['B','B'], and shipped a confident 21.0
        # for a 218 BB starting stack. And a trailing digit that scores as 'B'
        # left ['B','B','B'], which `suffix[:2]` accepted. Both are now a
        # three-glyph (or wrong-first-glyph) terminator token and both refuse.
        # Cost on the 18,006-crop development corpus: 0 -- the 43 reads whose
        # third right-hand glyph is the chip icon MISLABELLED as a digit keep
        # their value, because the chip sits a word space away and is not part
        # of the token.
        label = {id(g): ch for g, ch, _ in labeled}
        accepted = {id(g) for g, _, _ in run}
        right_of_run = [g for g in band if g.x >= x1 and id(g) not in accepted]
        terminator: list[Glyph] = []
        for g in right_of_run:
            if terminator and _xspan_gap(terminator[-1], g) > _RUN_ADJACENCY_GAP * run_h:
                break
            terminator.append(g)
        term_x1 = terminator[-1].x + terminator[-1].w if terminator else x1

        # ---- P2: no unexplained ink on the numeral's row ---------------------- #
        # Two arms, both refusal-only. NEAR: ink within the numeral's own letter
        # spacing that is not a proven affix or separator candidate
        # (_numeral_intruders). FAR: glyph-scale ink anywhere else on the row
        # that no anchored chain explains (_unanchored_row_ink) -- distance used
        # to be an exemption, and a wide occluder's fragments beyond the
        # adjacency window were invisible to every predicate.
        # THE AFFIX EXEMPTION IS POSITIONAL. `named` is the one thing that can
        # silence P2 on a glyph, and it exists for exactly one object: the "BB"
        # terminator, which the client renders AFTER the value. Granting it to a
        # glyph left of, or inside, the numeral hands the exemption to whatever
        # the bank happens to label a non-digit there -- and an occluded digit is
        # precisely that. Measured with the real chip sprite over the lower rows
        # of one digit: the occluded '9' of "190.10 BB" dropped below the digit
        # floor, was labelled 'c' (chip) at 0.649, entered `named`, exempted
        # itself from P2's near arm, and then ANCHORED the severed leading digits
        # in the far arm -- shipping 0.10 at score 0.946 for a 190.1 BB stack,
        # and "POT: 30 BB" as a confident 0.0. Restricting the exemption to the
        # located terminator token costs 0 reads on the 18,006-crop development
        # corpus: no legitimate affix has ever been anywhere else.
        term_ids = {id(g) for g in terminator}
        named = frozenset(i for i in named if i in term_ids)
        policed = _policed_ink(run, band, comps)
        sep_ids = frozenset(id(d) for d in sep_dots)
        if not skip_unexplained_ink:
            if _run_is_truncated(run, policed, dots, named, sep_ids):
                return AmountRead(None, digits, score, "unexplained_ink_in_numeral", n)
            # Anchors for the far arm's chain. A confident NON-DIGIT explains itself
            # ("POT:", "BB", the chip icon); a confident DIGIT outside the winning
            # run does not -- it is a fragment of the numeral the read failed to
            # keep whole. See _unanchored_row_ink.
            #
            # The one place the digit label carries no such warning is BEYOND THE
            # TERMINATOR: the client renders `<numeral> BB`, so nothing right of the
            # "BB" token can be a digit of the numeral, whatever the bank calls it.
            # The chip icon lands there and its pale annulus scores as '0' at reduced
            # render size on 43 development reads (the frozen "19.50 BB" bet crop
            # among them). Granting those glyphs an anchor cannot launder a numeral
            # fragment, because P1 below refuses unless the token really is "BB".
            conf_by_id = {id(g): (ch, sc) for g, ch, sc in labeled}
            confident: set[int] = set()
            for g in policed:
                got = conf_by_id.get(id(g))
                if got is None:
                    got = self.classify_digit(g.mask)
                ch, sc = got
                if sc >= min_score and (not ch.isdigit() or g.x >= term_x1):
                    confident.add(id(g))
            if _unanchored_row_ink(run, policed, named, sep_ids, frozenset(confident)):
                return AmountRead(None, digits, score, "unexplained_ink_in_numeral", n)

        # ---- P1: the numeral is terminated by a well-formed "BB" token -------- #
        # The whole token, located above, must be exactly two glyphs and both
        # must be 'B'. No filtering, no prefix slice: see the terminator's own
        # comment for the two severed-digit holes those two shortcuts opened.
        term_labels = [label[id(g)] for g in terminator]
        if term_labels != ["B", "B"]:
            # Small-render recovery only: at low glyph height the bank confuses
            # suffix B with 8 and P (~1.7%+ of development reads). Allowing the
            # measured confusion set keeps P5's decimal proof intact.
            if not (
                allow_b8_suffix
                and len(term_labels) == 2
                and all(ch in {"B", "8", "P"} for ch in term_labels)
            ):
                return AmountRead(None, digits, score, "suffix_not_bb", n)

        # ---- P5: the decimal is proven, not assumed --------------------------- #
        def _reconcilable(split: int) -> bool:
            """A split is a decimal point only if it leaves digits on both sides and
            no more fractional places than this client renders. Too deep means the
            glyph is a thousands separator, which sits in the same slot with the
            same silhouette; splitting there under-reads by 1000x. This is a reason
            to REFUSE a split -- never a reason to choose one."""
            return 0 < split < n and n - split <= _MAX_FRACTIONAL_DIGITS

        split_at: int | None = None
        if len(sep_dots) > 1:
            # THE CLIENT RENDERS AT MOST ONE SEPARATOR, so a second candidate is
            # not a second opinion about where the decimal is -- it is ink the
            # read cannot explain, and one of the two is not a decimal point.
            # Requiring uniqueness rather than agreement costs 0 of the 18,006
            # development crops (no value-producing crop carries two candidates),
            # and it closes a real fabrication: an occluder that both widens an
            # inter-digit gap into the decimal band AND scatters two baseline
            # specks into it gets a split from one speck while the pair jointly
            # fills the hole, so P5(b)'s remainder test sees nothing. Measured on
            # the real chip sprite over "POT: 39.50 BB": a confident 3.50.
            return AmountRead(None, digits, score, "separator_unreconciled", n)
        if sep_dots:
            # The candidate must reconcile. This replaces a widest-gap-first
            # ordering that broke at the first reconcilable candidate: under that
            # loop two candidates reconciling to different values were resolved
            # by a ranking key ("widest gap wins") instead of refused, and a
            # non-reconcilable candidate behind the winner was never examined at
            # all -- while the docstring claimed "no other candidate was
            # discarded". The loop is kept in agreement form so the predicate
            # stays correct if the uniqueness rule above is ever revisited.
            splits: set[int] = set()
            for d in sep_dots:
                # Split at the separator's REAL position, not blindly at the last
                # two digits: a dropped trailing glyph ("127.80" read as "1278")
                # would otherwise become 12.78 instead of 127.8.
                dot_x = d.x + d.w / 2.0
                candidate = sum(1 for a, b in gx if (a + b) / 2.0 < dot_x)
                if not _reconcilable(candidate):
                    # A separator whose reading does not reconcile -- "12,345" and
                    # "12.345" are the same pixels -- and the old fallbacks turned
                    # that into 1234.0, 34360.0 and 1830.88.
                    return AmountRead(None, digits, score, "separator_unreconciled", n)
                splits.add(candidate)
            if len(splits) > 1:
                return AmountRead(None, digits, score, "separator_unreconciled", n)
            split_at = splits.pop()

        # The gap test runs on EVERY read, not only when no separator exists. It
        # used to live in an `elif`, so the moment a separator was located no gap
        # was examined anywhere in the run -- and 78.3% of production reads take
        # the dot branch. That left a missing INTERIOR digit invisible: painting
        # one non-edge digit out of every value-producing crop produced 38439
        # confident wrong values out of 39953 (96.2%), e.g. 191.3 -> 11.3 and
        # 194.6 -> 14.6, every one through the dot branch. The located separator
        # is the one thing allowed to occupy a gap, so it is subtracted
        # (_largest_hole) and every remaining hole must stay inside the measured
        # letter-spacing of a real numeral. The boundary differs by branch --
        # _DECIMAL_BAND_MIN_GAP without a separator, _INTRA_NUMERAL_MAX_GAP with
        # one -- because the ambiguity differs: see both constants. Real decimals
        # measure 0.08-0.17 of run_h on their separator gap's remainders,
        # comfortably inside either.
        holes = [_largest_hole(gx[i][1], gx[i + 1][0], sep_dots)
                 for i in range(len(gx) - 1)]
        widest_hole = (max(holes) / run_h) if holes else 0.0
        if split_at is None and widest_hole >= _DECIMAL_BAND_MIN_GAP:
            # The old P5(b): an integer claim made from inside the band where a
            # lost dot and a wide letter space are the same measurement. See
            # _DECIMAL_BAND_MIN_GAP.
            return AmountRead(None, digits, score, "integer_over_decimal_band", n)
        if split_at is not None and widest_hole > _INTRA_NUMERAL_MAX_GAP:
            # A hole the located separator does not explain and letter spacing
            # has never produced: a digit is missing from the numeral. See
            # _INTRA_NUMERAL_MAX_GAP.
            return AmountRead(None, digits, score, "unexplained_gap_in_numeral", n)

        # ---- P6: a leading zero only ever precedes the separator -------------- #
        # The client never renders a leading zero on an integer part longer than
        # one digit: every legitimate leading-zero read in the corpus is "0.50"
        # (635) or "0.2" (13), i.e. split_at == 1 in 648 of 648. So "050" is a
        # "0.50" whose dot was lost, and "05.50" is a "95.50" whose leading digit
        # was occluded into a '0' -- measured: the real chip sprite over the
        # bottom rows of the '9' of a 95.50 BB stack re-labels it '0' at 0.570
        # and, with the real decimal still located at position 2, shipped a
        # confident 5.5. Testing the separator's POSITION rather than merely its
        # existence costs 0 of the 18,006 development crops.
        if n > 1 and digits[0] == "0" and split_at != 1:
            return AmountRead(None, digits, score, "leading_zero_no_dot", n)

        raw = f"{digits[:split_at]}.{digits[split_at:]}" if split_at is not None else digits
        try:
            value = float(raw)
        except ValueError:                                   # pragma: no cover
            return AmountRead(None, raw, score, "no_digit_run", n)
        return AmountRead(value, raw, score,
                          "dot" if split_at is not None else "integer", n)

    def read_number(self, crop_bgr: np.ndarray, min_score: float = 0.55) -> tuple[float | None, str]:
        """Return (value, debug_string). Thin wrapper over read_number_detail.

        Loses the refusal CODE, so it is a calibration/debug convenience only. The
        production path goes through read_amount_detail_from_image, which keeps it:
        a caller that needs to distinguish "the reader refused" from "no read was
        attempted" cannot get that from this signature."""
        r = self.read_number_detail(crop_bgr, min_score)
        return r.value, r.raw

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
    h, s, _v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
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


# Small ClubWPT windows (e.g. ~1050x730 captures) render HUD digits under the
# calibrated run-height floor (12px). Lowering that floor re-admits confident
# wrong values on native-small noise; instead the production entrypoint upscales
# into the band and re-reads. Job-4 stacks need denser scales, green-chip
# masking, digit-compatible trailing-fraction matching, and a guarded P2-waive
# path -- exact native digit equality alone recovered ~20% of below-floor stacks.
#
# Hard rules learned from adversarial review:
#   * never recover from ``no_digit_run`` (sprite fragments become 0.0)
#   * never invent a decimal from digit count alone on an *unrelated* suffix
#     refuse (the old 50.0 / 350.0 bug); suffix digits may be parsed only when
#     they are related to the native under-floor run and still pass consensus
#   * free consensus is allowed only with stricter factor/margin gates than
#     digit-compatible consensus
#   * count distinct upscale factors, not (factor, skip_ink) pairs
_SMALL_RENDER_UPSCALE_FACTORS = (
    1.3, 1.45, 1.6, 1.75, 1.8, 1.9, 2.0, 2.1, 2.12, 2.16, 2.2, 2.3, 2.32, 2.4, 2.5, 2.6, 2.7, 2.85,
)
_SMALL_RENDER_MIN_AGREEING_SCALES = 2
_SMALL_RENDER_FREE_MIN_AGREEING = 3
_SMALL_RENDER_FREE_MARGIN = 2
_SMALL_RENDER_BOX_EXPAND_PX = (0, 6, 10)
_SMALL_RENDER_RECOVERABLE = frozenset(
    {
        "below_calibrated_render_size",
        "ambiguous_longest_run",
    }
)


def _digits_only(raw: str) -> str:
    return "".join(ch for ch in raw if ch.isdigit())


def _ambiguous_digit_pieces(raw: str) -> list[str]:
    """Digit runs from an ``ambiguous_longest_run`` raw like ``19|20``."""
    if "|" not in (raw or ""):
        return []
    pieces = ["".join(ch for ch in part if ch.isdigit()) for part in raw.split("|")]
    return [p for p in pieces if p]


def _pieces_in_order(pieces: list[str], retried_digits: str) -> bool:
    """True when each ambiguous piece appears in order inside the recovered run.

    Caps inserted digits between/around pieces at 2. Single-digit pieces are
    allowed only when the recovered run is at least 3 digits (blocks ``1|1``
    matching a lone ``11`` fragment while still recovering ``181``).
    """
    if len(pieces) < 2 or not retried_digits:
        return False
    if any(len(p) < 2 for p in pieces) and len(retried_digits) < 3:
        return False
    pos = 0
    for piece in pieces:
        idx = retried_digits.find(piece, pos)
        if idx < 0:
            return False
        pos = idx + len(piece)
    if len(retried_digits) > sum(len(p) for p in pieces) + 2:
        return False
    return True


def _integer_digits(raw: str) -> str:
    if "." in raw:
        return _digits_only(raw.split(".", 1)[0])
    return _digits_only(raw)


def _near_digit_match(native_digits: str, retried_digits: str) -> bool:
    """Same-length hamming==1, allowing one trailing frac-zero on the recovery.

    Exact match after stripping a trailing zero is rejected — that is the forged
    ``2122`` -> ``2122.0`` path, not a real glyph confusion.
    """
    if len(native_digits) < 4 or not retried_digits:
        return False
    candidates = [retried_digits]
    if (
        retried_digits.endswith("0")
        and len(retried_digits) == len(native_digits) + 1
    ):
        candidates.append(retried_digits[:-1])
    for cand in candidates:
        if len(cand) != len(native_digits):
            continue
        if sum(a != b for a, b in zip(cand, native_digits, strict=True)) == 1:
            return True
    return False


def _parse_related_suffix_digits(
    native_digits: str, native_raw: str, refused: AmountRead
) -> AmountRead | None:
    """Turn a related ``suffix_not_bb`` digit run into a decimal AmountRead.

    Digits were proven; only the BB terminator failed. Refuse unless the digit
    string is related to the native under-floor evidence (blocks inventing
    50.0 / 350.0 from unrelated suffix refuses).
    """
    if refused.decimal_source != "suffix_not_bb":
        return None
    digits = _digits_only(refused.raw)
    if len(digits) < 3:
        return None
    for frac_len in (2, 1):
        if len(digits) <= frac_len:
            continue
        int_part = digits[:-frac_len]
        frac = digits[-frac_len:]
        # Strict: recovered integer digits must equal the native run. This is
        # the 528 -> 528.80 case. Looser splits (218 -> 2.18) are the measured
        # wrong-value regressions from inventing a decimal point.
        if int_part != native_digits:
            continue
        candidate = AmountRead(
            float(f"{int_part}.{frac}"),
            f"{int_part}.{frac}",
            refused.score,
            "dot",
            len(digits),
        )
        return candidate
    return None


def _is_forged_trailing_zero_extension(
    native_digits: str, retried: AmountRead | str
) -> bool:
    """True for spurious decimals glued onto an unchanged native integer run.

    Blocks ``2122`` -> ``2122.0`` and ``350`` -> ``350.03``. Allows ``528`` ->
    ``528.80`` (non-zero-leading frac) and ``2122`` -> ``212.20`` (integer part
    shorter than native).
    """
    if isinstance(retried, str):
        retried_digits = _digits_only(retried)
        int_d = retried_digits
        raw = retried
        frac = ""
    else:
        retried_digits = _digits_only(retried.raw)
        int_d = _integer_digits(retried.raw)
        raw = retried.raw or ""
        frac = (
            "".join(ch for ch in raw.split(".", 1)[1] if ch.isdigit())
            if "." in raw
            else ""
        )
    if not native_digits or int_d != native_digits or "." not in raw:
        return False
    if len(retried_digits) <= len(native_digits):
        return False
    if not frac or set(frac) <= {"0"}:
        return True
    # Leading-zero fractional noise on an unchanged integer (``350.03``).
    if frac.startswith("0"):
        return True
    # Single fractional digit glued onto the native integer (``350.8``) is
    # almost always hostile-interpolation noise; real BB recoveries use two
    # fractional places (``528.80``).
    if len(frac) == 1:
        return True
    return False


def _mask_green_chip(crop_bgr: np.ndarray) -> np.ndarray:
    """Blank saturated green pixels (HUD chip icon) that trip P2 after upscale."""
    out = crop_bgr.copy()
    if out.size == 0:
        return out
    b, g, r = cv2.split(out)
    green = (
        (g.astype(np.int16) > 70)
        & (g.astype(np.int16) > r.astype(np.int16) + 20)
        & (g.astype(np.int16) > b.astype(np.int16) + 20)
    )
    if np.any(green):
        out[green] = np.median(out.reshape(-1, 3), axis=0).astype(np.uint8)
    return out


def _mask_left_fraction(crop_bgr: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """Blank the leftmost slice. Safe as a *variant* only when gated: compat
    path uses `_digit_runs_compatible`; free path uses `_soft_digit_related`
    (both reject shorter runs that drop a native leading digit)."""
    out = crop_bgr.copy()
    if out.size == 0:
        return out
    cut = max(1, int(out.shape[1] * frac))
    out[:, :cut] = np.median(out.reshape(-1, 3), axis=0).astype(np.uint8)
    return out


def _soft_digit_related(
    native_digits: str, retried: AmountRead, *, native_raw: str = ""
) -> bool:
    """Weaker relatedness for free-consensus fallback.

    Never promotes a strictly shorter digit run than the native evidence (blocks
    191→19 and left-mask 0.50→50). Empty native defers to consensus gates alone.
    Ambiguous natives (``19|20``) match when recovered contains the pieces in order.
    """
    retried_digits = _digits_only(retried.raw)
    if not retried_digits or retried_digits in {"0", "00"}:
        return False
    pieces = _ambiguous_digit_pieces(native_raw)
    if pieces:
        # Ambiguous natives are gated solely by piece order — joined-digit
        # equality would re-admit ``1|1`` -> ``11``.
        return _pieces_in_order(pieces, retried_digits)
    if not native_digits:
        return True
    # Single-digit natives are fragments (clipped ``.60`` -> native ``6`` -> 60).
    if len(native_digits) <= 1:
        return False
    if len(retried_digits) < len(native_digits):
        return False
    if retried_digits == native_digits:
        # Single-digit equality re-admits sprite/clip fragments.
        if len(native_digits) <= 1:
            return False
        # Two-digit integer equality recovers true short stacks (``50`` BB).
        # Truncated longer stacks at hostile downscales (``191`` -> ``19``) are
        # an accepted residual: the pixels no longer contain the leading digit.
        if len(native_digits) == 2 and (
            "." in (retried.raw or "") or retried.decimal_source == "dot"
        ):
            return False
        if "." in (retried.raw or ""):
            if len(_integer_digits(retried.raw)) * 2 < len(native_digits):
                return False
        return True
    if (
        len(native_digits) == len(retried_digits)
        and len(native_digits) >= 4
        and sum(a != b for a, b in zip(native_digits, retried_digits, strict=True)) <= 1
    ):
        return True
    if _near_digit_match(native_digits, retried_digits):
        return True
    # Longer recovery that extends or embeds the native run. Allow +3 so a
    # truncated under-floor native like ``19`` can grow into ``198.50``.
    # Short natives may grow by +1 when the recovery is an integer (``21`` ->
    # ``218``). Decimal *prefix* growth needs +3 (blocks ``22`` -> ``22.42``);
    # decimal *suffix* growth allows +2 (``30`` -> ``43.30`` when native only
    # saw the trailing digits).
    is_decimal = "." in (retried.raw or "") or retried.decimal_source == "dot"
    extra = len(retried_digits) - len(native_digits)
    if _is_forged_trailing_zero_extension(native_digits, retried):
        return False
    if retried_digits.startswith(native_digits):
        min_extra = 3 if (len(native_digits) <= 2 and is_decimal) else 1
        if min_extra <= extra <= 3:
            return True
    if retried_digits.endswith(native_digits):
        min_extra = 2 if (len(native_digits) <= 2 and is_decimal) else 1
        if min_extra <= extra <= 3:
            return True
    return False


def _digit_runs_compatible(
    native_digits: str, retried: AmountRead, *, native_raw: str = ""
) -> bool:
    """Native under-floor runs often miss a trailing fractional zero or a leading digit."""
    retried_digits = _digits_only(retried.raw)
    if not native_digits or not retried_digits:
        return False
    pieces = _ambiguous_digit_pieces(native_raw)
    if pieces:
        return _pieces_in_order(pieces, retried_digits)
    # Single-digit natives are fragments (clipped ``.60`` -> native ``6`` -> 60).
    if len(native_digits) <= 1:
        return False
    # Short native runs must expand into a longer proven numeral for *fragments*
    # (native ``3`` -> 3.0). Two-digit integer equality is allowed below for
    # true short stacks (``50`` BB).
    if len(native_digits) <= 1 and len(retried_digits) <= len(native_digits):
        return False
    if retried_digits == native_digits:
        if len(native_digits) <= 1:
            return False
        if len(native_digits) == 2 and (
            "." in (retried.raw or "") or retried.decimal_source == "dot"
        ):
            return False
        if "." in (retried.raw or ""):
            if len(_integer_digits(retried.raw)) * 2 < len(native_digits):
                return False
        return True
    if retried.decimal_source == "dot" and "." in retried.raw:
        frac = "".join(ch for ch in retried.raw.split(".", 1)[1] if ch.isdigit())
        # ``native + frac`` is only safe when native already looks like a full
        # integer stack (>=3 digits). Short natives + one frac digit invent
        # ``21`` -> ``21.8`` while the screen shows ``218``.
        if len(native_digits) >= 3 and retried_digits == native_digits + frac:
            # Forged trailing ``.0`` / leading-zero frac noise on unchanged integer.
            if not _is_forged_trailing_zero_extension(native_digits, retried):
                return True
        if native_digits in retried_digits and abs(
            len(retried_digits) - len(native_digits)
        ) <= 3:
            if len(native_digits) > 2 or len(retried_digits) >= len(native_digits) + 3:
                if not _is_forged_trailing_zero_extension(native_digits, retried):
                    return True
    if (
        retried_digits.endswith(native_digits)
        and 1 <= len(retried_digits) - len(native_digits) <= 3
        and len(native_digits) >= 3
        and not _is_forged_trailing_zero_extension(native_digits, retried)
    ):
        return True
    # Truncated native prefix of a longer recovered run (e.g. "19" -> "19850").
    # Short natives may grow by +1 for integers (``21`` -> ``218``) but need
    # +3 when the recovery is a decimal *prefix* (blocks ``22`` -> ``22.42``).
    is_decimal = "." in (retried.raw or "") or retried.decimal_source == "dot"
    min_extra = 3 if (len(native_digits) <= 2 and is_decimal) else 1
    if (
        len(native_digits) >= 2
        and len(retried_digits) >= len(native_digits) + min_extra
        and len(retried_digits) - len(native_digits) <= 3
        and retried_digits.startswith(native_digits)
        and not _is_forged_trailing_zero_extension(native_digits, retried)
    ):
        return True
    # Trailing-digit native under a longer decimal (``30`` -> ``43.30``).
    if (
        len(native_digits) >= 2
        and is_decimal
        and retried_digits.endswith(native_digits)
        and 2 <= len(retried_digits) - len(native_digits) <= 3
        and not _is_forged_trailing_zero_extension(native_digits, retried)
    ):
        return True
    # Same-length single-glyph confusion (e.g. native "21520" vs recovered "21820").
    if (
        len(native_digits) == len(retried_digits)
        and len(native_digits) >= 4
        and sum(a != b for a, b in zip(native_digits, retried_digits, strict=True)) == 1
    ):
        return True
    if _near_digit_match(native_digits, retried_digits):
        return True
    return False


def _reinterpret_integer_as_bb_decimal(
    native_digits: str, read: AmountRead
) -> AmountRead | None:
    """When consensus returns an integer whose digits equal the native run,
    insert the ClubWPT decimal (4 digits -> 1 frac place, 5+ -> 2).

    Fixes ``2122`` shipping as ``2122.0`` instead of ``212.2``.
    """
    if not native_digits or _digits_only(read.raw) != native_digits:
        return None
    if "." in (read.raw or "") or read.decimal_source == "dot":
        return None
    n = len(native_digits)
    frac_len = 2 if n >= 5 else 1 if n == 4 else None
    if frac_len is None:
        return None
    int_part = native_digits[:-frac_len]
    frac = native_digits[-frac_len:]
    if len(int_part) < 2:
        return None
    return AmountRead(
        float(f"{int_part}.{frac}"),
        f"{int_part}.{frac}",
        read.score,
        "dot",
        n,
    )


def _pick_amount_consensus(
    votes: dict[float, dict[float, AmountRead]],
    *,
    min_agree: int,
    margin: int = 1,
) -> AmountRead | None:
    """``votes`` maps value -> {upscale_factor: AmountRead}; factors must be distinct."""
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda item: -len(item[1]))
    _value, by_factor = ranked[0]
    if len(by_factor) < min_agree:
        return None
    if len(ranked) > 1 and len(by_factor) < len(ranked[1][1]) + margin:
        return None
    reads = list(by_factor.values())
    return reads[len(reads) // 2]


def _upscale_crop(crop_bgr: np.ndarray, factor: float) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    return cv2.resize(
        crop_bgr,
        (max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
        interpolation=cv2.INTER_CUBIC,
    )


def _small_render_variants(crop_bgr: np.ndarray) -> list[np.ndarray]:
    # Green-chip blanking only. Left-fraction masking was removed after it
    # blanked the leading ``0`` in ``0.50`` and let free consensus ship ``50.0``.
    # Skip the duplicate when blanking changed nothing -- same vote set, less work.
    green = _mask_green_chip(crop_bgr)
    if green is crop_bgr or (
        green.shape == crop_bgr.shape and bool(np.array_equal(green, crop_bgr))
    ):
        return [crop_bgr]
    return [crop_bgr, green]


def _unbeatable_amount_consensus(
    votes: dict[float, dict[float, AmountRead]],
    *,
    min_agree: int,
    margin: int,
    remaining_factors: int,
) -> AmountRead | None:
    """Consensus that remaining factors cannot overturn, even if they all dissent."""
    got = _pick_amount_consensus(votes, min_agree=min_agree, margin=margin)
    if got is None:
        return None
    ranked = sorted(votes.items(), key=lambda item: -len(item[1]))
    best = len(ranked[0][1])
    second = len(ranked[1][1]) if len(ranked) > 1 else 0
    if best >= second + margin + remaining_factors:
        return got
    return None


def _accepted_recovered_amount(
    bank: TemplateOCR,
    crop_bgr: np.ndarray,
    digits_for_votes: str,
    got: AmountRead | None,
) -> AmountRead | None:
    if got is None:
        return None
    if _short_equality_is_clipped_fragment(bank, crop_bgr, digits_for_votes, got):
        return None
    reinterpreted = _reinterpret_integer_as_bb_decimal(digits_for_votes, got)
    return reinterpreted if reinterpreted is not None else got


def _short_equality_is_clipped_fragment(
    bank: TemplateOCR,
    crop_bgr: np.ndarray,
    native_digits: str,
    got: AmountRead,
) -> bool:
    """Reject 2-digit equality when left-expanding the crop reveals more digits.

    Catches clipped ``.60`` -> native ``60`` -> confident 60.0 while a wider
    crop still shows leading ink the tight YOLO box dropped.
    """
    if len(native_digits) != 2 or _digits_only(got.raw) != native_digits:
        return False
    if got.decimal_source == "dot" or "." in (got.raw or ""):
        return False
    h, w = crop_bgr.shape[:2]
    if w < 4:
        return False
    pad = max(4, w // 5)
    padded = cv2.copyMakeBorder(
        crop_bgr, 0, 0, pad, 0, cv2.BORDER_REPLICATE
    )
    wider = bank.read_number_detail(padded)
    wider_digits = _digits_only(wider.raw)
    if wider_digits and len(wider_digits) > len(native_digits):
        return True
    # Also try a mild upscale of the left-padded crop.
    up = _upscale_crop(padded, 2.0)
    wider_up = bank.read_number_detail(
        up, skip_unexplained_ink=True, allow_b8_suffix=True
    )
    wider_up_digits = _digits_only(wider_up.raw)
    return bool(wider_up_digits and len(wider_up_digits) > len(native_digits))


def _recover_small_render_amount(
    bank: TemplateOCR, img_bgr: np.ndarray, xyxy: Sequence[float], native: AmountRead
) -> AmountRead:
    """Recover a below-floor / ambiguous HUD amount via multi-scale consensus."""
    native_raw = native.raw or ""
    native_digits = _digits_only(native_raw)
    # Lone under-floor "0" / all-zero ambiguous runs ("0|0") are the measured
    # sprite-fragment and lost-leading-decimal failure modes: every upscale can
    # invent a confident non-zero (0.0 or 50.0) while the screen is unrelated.
    # Keep them unknown. True 0 BB stacks at calibrated sizes still read natively.
    if not native_digits or set(native_digits) <= {"0"}:
        return native
    # Ambiguous longest-run raw looks like "19|20"; joining the digit pieces is
    # the under-floor evidence for digit-compatibility checks.
    if "|" in native_raw:
        joined = "".join(ch for ch in native_raw if ch.isdigit())
        if len(joined) > len(native_digits):
            native_digits = joined
        if set(native_digits) <= {"0"}:
            return native

    h_img, w_img = img_bgr.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    for expand in _SMALL_RENDER_BOX_EXPAND_PX:
        xa = max(0, x1 - expand)
        ya = max(0, y1 - expand // 2)
        xb = min(w_img, x2 + expand)
        yb = min(h_img, y2 + expand // 2)
        base = img_bgr[ya:yb, xa:xb]
        if base.size == 0:
            continue
        digits_for_votes = native_digits
        raw_for_votes = native_raw
        if expand:
            expanded_native = bank.read_number_detail(base)
            if expanded_native.value is not None:
                return expanded_native
            expanded_digits = _digits_only(expanded_native.raw)
            if not expanded_digits or expanded_digits == "0":
                continue
            digits_for_votes = expanded_digits
            if expanded_native.raw:
                raw_for_votes = expanded_native.raw

        # value -> {factor: AmountRead}. skip_ink / b8 retries share the factor
        # key so they cannot alone satisfy min_agree. Factor-outer order lets us
        # stop once remaining scales cannot overturn consensus; that is the same
        # accept/reject set as scanning every scale, just less work on easy wins.
        compat_votes: dict[float, dict[float, AmountRead]] = {}
        free_votes: dict[float, dict[float, AmountRead]] = {}
        variants = _small_render_variants(base)
        factors = _SMALL_RENDER_UPSCALE_FACTORS
        for factor_index, factor in enumerate(factors):
            for variant in variants:
                up = _upscale_crop(variant, factor)
                for skip_ink, allow_b8 in (
                    (False, False),
                    (True, False),
                    (True, True),
                ):
                    retried = bank.read_number_detail(
                        up,
                        skip_unexplained_ink=skip_ink,
                        allow_b8_suffix=allow_b8,
                    )
                    if retried.value is None:
                        parsed = _parse_related_suffix_digits(
                            digits_for_votes, raw_for_votes, retried
                        )
                        if parsed is None:
                            continue
                        retried = parsed
                    # Never promote under-floor lone-zero fragments via free votes.
                    if retried.value == 0.0 and _digits_only(retried.raw) in {"0", "00"}:
                        continue
                    if _soft_digit_related(
                        digits_for_votes, retried, native_raw=raw_for_votes
                    ):
                        free_votes.setdefault(retried.value, {}).setdefault(
                            factor, retried
                        )
                    if _digit_runs_compatible(
                        digits_for_votes, retried, native_raw=raw_for_votes
                    ):
                        compat_votes.setdefault(retried.value, {}).setdefault(
                            factor, retried
                        )
            remaining = len(factors) - factor_index - 1
            accepted = _accepted_recovered_amount(
                bank,
                base,
                digits_for_votes,
                _unbeatable_amount_consensus(
                    compat_votes,
                    min_agree=_SMALL_RENDER_MIN_AGREEING_SCALES,
                    margin=1,
                    remaining_factors=remaining,
                ),
            )
            if accepted is not None:
                return accepted
            # Stricter free consensus for truncated/confused native digits: more
            # independent factors and a clear margin. Zero wrong values on the
            # frozen adversarial fixture scale sweep with these gates.
            accepted = _accepted_recovered_amount(
                bank,
                base,
                digits_for_votes,
                _unbeatable_amount_consensus(
                    free_votes,
                    min_agree=_SMALL_RENDER_FREE_MIN_AGREEING,
                    margin=_SMALL_RENDER_FREE_MARGIN,
                    remaining_factors=remaining,
                ),
            )
            if accepted is not None:
                return accepted
        # Exact digit-complete decimal: one upscale proved a decimal whose digit
        # string equals the under-floor native run (``46280`` -> ``462.80``).
        # Requires a 3+ digit integer part so ``080`` cannot become ``0.80``.
        for votes in (compat_votes, free_votes):
            if len(votes) != 1:
                continue
            _value, by_factor = next(iter(votes.items()))
            # Need at least 2 distinct factors even for digit-complete decimals —
            # a lone factor is the adversarial single-scale invention path.
            if len(by_factor) < 2:
                continue
            read = next(iter(by_factor.values()))
            if (
                read.decimal_source == "dot"
                and _digits_only(read.raw) == digits_for_votes
                and len(_integer_digits(read.raw)) >= 3
                and not _is_forged_trailing_zero_extension(digits_for_votes, read)
            ):
                return read
    return native


def read_amount_detail_from_image(img_bgr: np.ndarray, xyxy: Sequence[float]) -> AmountRead | None:
    """Full read detail, or None when no template bank is calibrated at all.

    THREE distinct outcomes, and the caller must keep them apart:
      * `AmountRead(value=<float>, ...)`  -- a proven read
      * `AmountRead(value=None, decimal_source=<REFUSAL_CODES>)` -- the reader ran
        and REFUSED; the amount is unknown, which is not zero
      * `None` (this function's return) -- no bank exists, so no read was attempted.
        The Detection layer records this as `attr_source="reader_unavailable"`.

    The old `read_amount_from_image` flattened all three into one `float | None`
    and is deleted; it had no caller outside this module."""
    bank = _bank()
    if bank is None:
        return None
    crop = _crop(img_bgr, xyxy)
    detail = bank.read_number_detail(crop)
    if detail.value is not None or crop.size == 0:
        return detail
    if detail.decimal_source not in _SMALL_RENDER_RECOVERABLE:
        return detail
    return _recover_small_render_amount(bank, img_bgr, xyxy, detail)


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
        # green = CALL or BET. There is no safe default between them: picking one
        # asserts a fact the pixels do not carry (see region_detections
        # read_pill_action / PILL_BET_OR_CALL).
        return PILL_BET_OR_CALL
    return "check" if dealt_in else "fold"
