"""Property and fuzz tests for range notation, normalization and weights (PLAN Phase 14).

Three layers get generated here.

* The **notation grammar** (`poker_tracker.solver.ranges.parse_range`), which is
  the only parser in the project that reads weights. Every token form is
  generated -- pairs, suited/offsuit/both, `+` runs, `-` spans, explicit combos,
  eval7 `%()` weights and TexasSolver `HAND:weight` weights -- and checked
  against hold'em combinatorics worked out independently of the parser. The
  invariants: the expanded combos are exactly what the notation denotes, no
  combo appears twice, weights land in (0, 1], `range_percent` is the weighted
  share of 1326, re-parsing the parser's own output changes nothing, token order
  does not matter, and an unparseable token aborts the range instead of
  quietly yielding the tokens around it.

* The **study labels** (`poker_tracker.math.ranges`), which the equity engine
  turns into a villain range. They are a ladder -- premium, tight, standard,
  loose, very_loose -- and the ladder is the contract: each label must contain
  every combo of the one below it and be wider than it. `loose` used to be
  narrower than `standard`, so tagging a villain looser handed the equity
  engine a tighter range.

* The **positional charts** (`poker_tracker.math.preflop_ranges`), whose
  descriptions declare a width the notation has to actually have.

Label and position normalization are also generated against hostile text,
because both are called on free-form operator input and neither may raise.
"""

from __future__ import annotations

import re

import eval7
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from poker_tracker.math.equity import _resolve_range
from poker_tracker.math.preflop_ranges import (
    TOTAL_COMBOS,
    available_ranges,
    normalize_position,
    range_percent,
    resolve_preflop_range,
)
from poker_tracker.math.ranges import (
    RANGE_LABELS,
    RANGES,
    estimate_villain_range_label,
    normalize_range_label,
    range_notation,
)
from poker_tracker.solver.ranges import normalize_weighted_notation, parse_range
from poker_tracker.solver.texassolver import TEXASSOLVER_MIN_WEIGHT

RANK_ORDER = "23456789TJQKA"
SUIT_CHARS = "cdhs"

# The label ladder, narrowest first. Each entry must contain the previous one.
LABEL_LADDER = ["premium", "tight", "standard", "loose", "very_loose"]

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
)

# Weights that survive a round trip through both spellings exactly: above the
# TexasSolver minimum, and expressible in the six decimals the parser prints.
SUPPORTED_WEIGHTS = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 1.0]


# --- hold'em combinatorics, worked out without the parser -----------------------------


def _combo_key(first: str, second: str) -> str:
    return "|".join(sorted((first, second)))


def _combos_of(high: str, low: str, kind: str) -> set[str]:
    """Every combo of one rank pairing: 6 for a pair, 4 suited, 12 offsuit, 16 both."""
    out: set[str] = set()
    for first_suit in SUIT_CHARS:
        for second_suit in SUIT_CHARS:
            first, second = f"{high}{first_suit}", f"{low}{second_suit}"
            if first == second:
                continue
            suited = first_suit == second_suit
            if high == low or kind == "" or (kind == "s") == suited:
                out.add(_combo_key(first, second))
    return out


def _combos_of_run(high: str, low: str, kind: str) -> set[str]:
    """`+` notation: hold the high rank, walk the low rank up to it."""
    out: set[str] = set()
    if high == low:
        for rank in RANK_ORDER[RANK_ORDER.index(high) :]:
            out |= _combos_of(rank, rank, kind)
    else:
        for rank in RANK_ORDER[RANK_ORDER.index(low) : RANK_ORDER.index(high)]:
            out |= _combos_of(high, rank, kind)
    return out


def _combos_of_span(high: str, first_low: str, second_low: str, kind: str) -> set[str]:
    """`-` notation: an inclusive span, in either order."""
    out: set[str] = set()
    lower, upper = sorted((RANK_ORDER.index(first_low), RANK_ORDER.index(second_low)))
    for index in range(lower, upper + 1):
        rank = RANK_ORDER[index]
        out |= _combos_of(rank if high == first_low else high, rank, kind)
    return out


# --- notation strategies --------------------------------------------------------------


@st.composite
def _rank_pairing(draw: st.DrawFn) -> tuple[str, str, str]:
    high_index = draw(st.integers(min_value=0, max_value=12))
    low_index = draw(st.integers(min_value=0, max_value=high_index))
    kind = "" if high_index == low_index else draw(st.sampled_from(["", "s", "o"]))
    return RANK_ORDER[high_index], RANK_ORDER[low_index], kind


@st.composite
def unweighted_token(draw: st.DrawFn) -> tuple[str, set[str]]:
    """One notation token plus the combos it denotes, derived independently."""
    form = draw(st.sampled_from(["exact", "run", "span", "combo"]))
    if form == "combo":
        first = draw(st.sampled_from([f"{r}{s}" for r in RANK_ORDER for s in SUIT_CHARS]))
        second = draw(st.sampled_from([f"{r}{s}" for r in RANK_ORDER for s in SUIT_CHARS]))
        assume(first != second)
        # eval7 wants the higher rank first in an explicit combo.
        if RANK_ORDER.index(first[0]) < RANK_ORDER.index(second[0]):
            first, second = second, first
        return f"{first}{second}", {_combo_key(first, second)}

    high, low, kind = draw(_rank_pairing())
    if form == "exact":
        return f"{high}{low}{kind}", _combos_of(high, low, kind)
    if form == "run":
        return f"{high}{low}{kind}+", _combos_of_run(high, low, kind)

    if high == low:
        other = RANK_ORDER[draw(st.integers(min_value=0, max_value=RANK_ORDER.index(high)))]
        return f"{high}{high}-{other}{other}", _combos_of_span(high, high, other, kind)
    # A span has to stay inside one suitedness class, so the far end of an
    # unpaired span must stay below the fixed high rank.
    other = RANK_ORDER[draw(st.integers(min_value=0, max_value=RANK_ORDER.index(high) - 1))]
    return (
        f"{high}{low}{kind}-{high}{other}{kind}",
        _combos_of_span(high, low, other, kind),
    )


@st.composite
def weighted_token(draw: st.DrawFn) -> tuple[str, set[str], float]:
    """A token, its combos, and the weight every one of those combos carries."""
    token, combos = draw(unweighted_token())
    style = draw(st.sampled_from(["bare", "percent", "colon"]))
    if style == "bare":
        return token, combos, 1.0
    weight = draw(st.sampled_from(SUPPORTED_WEIGHTS))
    if style == "percent":
        percent = f"{weight * 100:.6f}".rstrip("0").rstrip(".")
        return f"{percent}%({token})", combos, weight
    return f"{token}:{weight:.6f}".rstrip("0").rstrip("."), combos, weight


@st.composite
def notation(draw: st.DrawFn) -> tuple[str, dict[str, float]]:
    """A whole comma-separated range plus the combo -> weight map it denotes.

    Overlapping tokens are allowed and expected: the parser resolves a combo
    claimed twice to the highest weight claimed for it.
    """
    tokens = draw(st.lists(weighted_token(), min_size=1, max_size=4))
    expected: dict[str, float] = {}
    for _, combos, weight in tokens:
        for combo in combos:
            expected[combo] = max(expected.get(combo, 0.0), weight)
    return ",".join(token for token, _, _ in tokens), expected


def _weights_of(parsed) -> dict[str, float]:
    """Read the weight the parser assigned to each combo out of its own output."""
    weights: dict[str, float] = {}
    for token in parsed.solver_notation.split(","):
        combo, _, weight_text = token.partition(":")
        weights[_combo_key(combo[:2], combo[2:])] = float(weight_text) if weight_text else 1.0
    return weights


# --- the notation grammar -------------------------------------------------------------


@SETTINGS
@given(notation())
def test_the_expanded_combos_are_exactly_what_the_notation_denotes(
    generated: tuple[str, dict[str, float]],
) -> None:
    """A range is never silently truncated and never silently widened."""
    text, expected = generated
    parsed = parse_range(text)

    assert set(parsed.combo_keys) == set(expected), text
    assert parsed.combo_count == len(expected), text
    # combo_keys is a set, so a duplicated combo would show up as a count that
    # outruns it. They must agree.
    assert len(parsed.combo_keys) == parsed.combo_count, text


@SETTINGS
@given(notation())
def test_every_combo_carries_the_weight_the_notation_gave_it(
    generated: tuple[str, dict[str, float]],
) -> None:
    text, expected = generated
    weights = _weights_of(parse_range(text))

    assert set(weights) == set(expected), text
    for combo, weight in weights.items():
        assert weight == pytest.approx(expected[combo], abs=1e-6), (text, combo)
        assert 0 < weight <= 1.0, (text, combo, weight)


@SETTINGS
@given(notation())
def test_range_percent_is_the_weighted_share_of_all_1326_combos(
    generated: tuple[str, dict[str, float]],
) -> None:
    text, expected = generated
    parsed = parse_range(text)

    assert parsed.range_percent == pytest.approx(sum(expected.values()) / TOTAL_COMBOS, abs=1e-6), (
        text
    )
    assert 0 < parsed.range_percent <= 1.0, text


@SETTINGS
@given(notation())
def test_re_parsing_the_parsers_own_output_changes_nothing(
    generated: tuple[str, dict[str, float]],
) -> None:
    """`solver_notation` is what reaches the solver, so it has to mean the same range."""
    text, _ = generated
    parsed = parse_range(text)
    reparsed = parse_range(parsed.solver_notation)

    assert reparsed.combo_keys == parsed.combo_keys, text
    assert _weights_of(reparsed) == pytest.approx(_weights_of(parsed), abs=1e-6), text
    assert reparsed.range_percent == pytest.approx(parsed.range_percent, abs=1e-6), text
    assert reparsed.solver_notation == parsed.solver_notation, text


@SETTINGS
@given(notation())
def test_normalizing_weighted_notation_is_idempotent(
    generated: tuple[str, dict[str, float]],
) -> None:
    text, _ = generated
    once = normalize_weighted_notation(text)
    assert normalize_weighted_notation(once) == once, text
    assert parse_range(once).combo_keys == parse_range(text).combo_keys, text


@SETTINGS
@given(notation(), st.randoms(use_true_random=False))
def test_token_order_does_not_change_the_parsed_range(
    generated: tuple[str, dict[str, float]], rng
) -> None:
    text, _ = generated
    tokens = text.split(",")
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    assume(shuffled != tokens)

    original, reordered = parse_range(text), parse_range(",".join(shuffled))
    assert reordered.combo_keys == original.combo_keys, text
    assert _weights_of(reordered) == pytest.approx(_weights_of(original), abs=1e-6), text
    assert reordered.range_percent == pytest.approx(original.range_percent, abs=1e-6), text


UNPARSEABLE_TOKENS = [
    "XX",
    "1h",
    "AAs",
    "Ah",
    "A",
    "9",
    "s",
    "AKx",
    "AA++",
    "AA:0.5:0.5",
    "AA:2.0",
    "AA:-0.5",
    "((AA)",
    "AA)",
    "%AA",
    "-AA",
    "AA-",
    "10h10s",
    "\U0001f0a1",
]


@pytest.mark.parametrize("junk", UNPARSEABLE_TOKENS)
def test_each_junk_token_really_is_unparseable_on_its_own(junk: str) -> None:
    """Guards the fuzz below: a token that quietly parsed would prove nothing."""
    with pytest.raises(ValueError):
        parse_range(junk)


@SETTINGS
@given(notation(), st.sampled_from(UNPARSEABLE_TOKENS), st.integers(min_value=0, max_value=8))
def test_one_unparseable_token_aborts_the_whole_range(
    generated: tuple[str, dict[str, float]], junk: str, position: int
) -> None:
    """The tokens around a bad one are never returned as a smaller range.

    Silently dropping the token the operator could not spell would hand the
    solver a range they never asked for, and nothing downstream would know.
    """
    text, _ = generated
    tokens = text.split(",")
    index = position % (len(tokens) + 1)
    polluted = ",".join([*tokens[:index], junk, *tokens[index:]])

    with pytest.raises(ValueError):
        parse_range(polluted)


@pytest.mark.parametrize("text", ["", " ", ",", ",,", "  ,  ,  "])
def test_an_empty_range_is_refused(text: str) -> None:
    with pytest.raises(ValueError):
        parse_range(text)


@SETTINGS
@given(st.sampled_from([0.000001, 0.0001, 0.001, 0.004, TEXASSOLVER_MIN_WEIGHT]))
def test_a_weight_texassolver_would_drop_is_refused(weight: float) -> None:
    """TexasSolver silently discards combos at or below its minimum weight.

    The parser refuses them rather than emit a command file whose range is
    quietly smaller than the one the operator wrote.
    """
    with pytest.raises(ValueError, match="silently drops weights"):
        parse_range(f"AA:{weight:.10f}".rstrip("0"))


def test_a_weight_below_the_printing_precision_is_also_refused() -> None:
    """The instance the guard used to miss, kept as its own named case.

    The guard was spelled ``float(f"{weight:.6f}") > 0``, which asked whether the
    weight PRINTED as something other than zero rather than whether it WAS
    something the solver would keep. A weight of 1e-7 printed as `0.000000`,
    tested equal to zero, and was accepted as six live combos and serialized into
    an `AdAc:0` token -- notation the parser itself then rejects, which is how the
    round-trip property found it.
    """
    with pytest.raises(ValueError, match="silently drops weights"):
        parse_range("AA:0.0000001")


@st.composite
def _weight_text(draw: st.DrawFn) -> str:
    """A weight spelled as ``0.<digits>``, drawn across every magnitude.

    The count of leading zeros is drawn rather than the value, so the band below
    the printing precision -- which a uniform float would land in about once in
    two million draws -- is sampled as often as the ordinary one.
    """
    leading_zeros = draw(st.integers(min_value=0, max_value=13))
    digits = draw(st.integers(min_value=1, max_value=999999))
    return f"0.{'0' * leading_zeros}{digits}"


# One step of the six-decimal precision a weight is submitted at. A value within
# a step of the cutoff may fall on either side of it once it has been through
# eval7, and which side is not the point of the assertions below.
_WEIGHT_STEP = 10**-6


@SETTINGS
@given(_weight_text())
def test_a_weight_is_judged_on_its_value_not_on_how_it_prints(text: str) -> None:
    """The family invariant, stated over numbers rather than over their spelling.

    Every weight is either refused for being one TexasSolver would drop, or it
    reaches the solver as a weight above that cutoff. Nothing in between: no
    weight may be accepted and then emitted as something the solver discards,
    which is what a guard that reads the printed form permits whenever the
    printed form loses the value.
    """
    weight = float(text)
    try:
        parsed = parse_range(f"AA:{text}")
    except ValueError as exc:
        assert "silently drops weights" in str(exc), (text, str(exc))
        assert weight <= TEXASSOLVER_MIN_WEIGHT + _WEIGHT_STEP, text
        return

    assert weight > TEXASSOLVER_MIN_WEIGHT - _WEIGHT_STEP, text
    for emitted in _weights_of(parsed).values():
        assert emitted > TEXASSOLVER_MIN_WEIGHT, (text, parsed.solver_notation)
    assert parsed.range_percent > 0, text
    # The emitted notation is the only carrier of these weights, so it has to
    # survive being read back as the same range.
    assert _weights_of(parse_range(parsed.solver_notation)) == pytest.approx(
        _weights_of(parsed), abs=1e-9
    ), text


@SETTINGS
@given(notation())
def test_no_emitted_token_ever_carries_a_droppable_weight(
    generated: tuple[str, dict[str, float]],
) -> None:
    """Said over whole generated ranges as well, not just single-token ones."""
    text, _ = generated
    parsed = parse_range(text)

    for token in parsed.solver_notation.split(","):
        _, separator, weight_text = token.partition(":")
        assert weight_text != "0", (text, token)
        if separator:
            assert float(weight_text) > TEXASSOLVER_MIN_WEIGHT, (text, token)


@pytest.mark.parametrize(
    "text",
    ["AA:0.0000001", "AA:0.00000000001", "0.00001%(AA)", "AA:0.5,KK:0.0000001", "AA:0"],
)
def test_a_sub_precision_weight_never_reaches_the_solver_as_zero(text: str) -> None:
    """Both spellings of a weight, and both places the value could be lost.

    The colon form is rejected while it is still a number the operator wrote; the
    eval7 percent form survives normalization and is rejected on the weight that
    would actually be submitted. Either way the refusal names the same cause,
    rather than blaming the operator's spelling for a value the parser rounded
    away on its way to eval7.
    """
    with pytest.raises(ValueError, match="silently drops weights"):
        parse_range(text)


# --- the study-label ladder -----------------------------------------------------------


def test_every_label_has_a_definition_and_every_definition_has_a_label() -> None:
    assert RANGE_LABELS == set(RANGES)
    for label, description in RANGES.items():
        assert description.label == label
        assert normalize_range_label(label) == label


def test_only_the_unknown_label_has_no_notation() -> None:
    for label in RANGE_LABELS:
        assert (range_notation(label) is None) is (label == "unknown")


@pytest.mark.parametrize("label", LABEL_LADDER)
def test_each_label_expands_to_distinct_full_weight_combos(label: str) -> None:
    notation_text = range_notation(label)
    assert notation_text is not None
    parsed = parse_range(notation_text)
    assert parsed.combo_count == len(parsed.combo_keys)
    assert set(_weights_of(parsed).values()) == {1.0}


def test_the_label_ladder_widens_at_every_step() -> None:
    """A looser label must be a wider range. `loose` used to be narrower than
    `standard` (15.7% against 21.6%), so an operator who tagged a villain
    loose/passive/station got an equity number computed against a tighter range
    than the default -- an under-estimate of Hero's equity presented as an
    improvement on it.
    """
    widths = [range_percent(range_notation(label)) for label in LABEL_LADDER]

    assert widths == sorted(widths), dict(zip(LABEL_LADDER, widths, strict=True))
    assert len(set(widths)) == len(widths), dict(zip(LABEL_LADDER, widths, strict=True))


def test_each_label_contains_every_combo_of_the_label_below_it() -> None:
    """Width alone is not enough: the ladder has to nest, or loosening a read
    would drop hands the tighter range already held."""
    previous_label, previous = LABEL_LADDER[0], parse_range(range_notation(LABEL_LADDER[0]))
    for label in LABEL_LADDER[1:]:
        current = parse_range(range_notation(label))
        missing = set(previous.combo_keys) - set(current.combo_keys)
        assert not missing, f"{label} drops {len(missing)} combos held by {previous_label}"
        previous_label, previous = label, current


def test_a_label_that_declares_a_width_actually_has_it() -> None:
    """The descriptions are shown to the operator, so a stale one is a wrong read."""
    declared_any = False
    for label, description in RANGES.items():
        declared = re.search(r"~(\d+)%", description.description)
        if declared is None:
            continue
        declared_any = True
        assert range_percent(description.notation) == pytest.approx(
            int(declared.group(1)) / 100, abs=0.01
        ), label
    assert declared_any


@SETTINGS
@given(st.text(max_size=24))
def test_label_normalization_is_total_and_idempotent(text: str) -> None:
    normalized = normalize_range_label(text)
    assert normalized in RANGE_LABELS, text
    assert normalize_range_label(normalized) == normalized, text


@SETTINGS
@given(st.lists(st.text(max_size=12), max_size=4), st.text(max_size=40))
def test_estimating_a_label_only_ever_returns_a_defined_label(tags: list[str], notes: str) -> None:
    assert estimate_villain_range_label(tags, notes) in RANGE_LABELS


# --- the positional charts ------------------------------------------------------------


def test_every_chart_is_as_wide_as_its_description_claims() -> None:
    """The descriptions carry a declared width; the notation has to have it."""
    for chart in available_ranges():
        declared = re.search(r"~(\d+)%", chart.description)
        assert declared is not None, chart
        assert range_percent(chart.notation) == pytest.approx(
            int(declared.group(1)) / 100, abs=0.01
        ), chart


def test_every_chart_expands_to_distinct_full_weight_combos() -> None:
    for chart in available_ranges():
        parsed = parse_range(chart.notation)
        assert parsed.combo_count == len(parsed.combo_keys), chart
        assert set(_weights_of(parsed).values()) == {1.0}, chart
        assert 0 < parsed.range_percent < 1, chart


@SETTINGS
@given(st.text(max_size=20))
def test_position_normalization_is_total_and_idempotent(text: str) -> None:
    normalized = normalize_position(text)
    if normalized is None:
        return
    assert normalize_position(normalized) == normalized, text
    assert resolve_preflop_range(normalized, "rfi") == resolve_preflop_range(text, "rfi")


@SETTINGS
@given(st.text(max_size=20), st.text(max_size=20))
def test_resolving_a_chart_never_raises_on_free_form_input(position: str, scenario: str) -> None:
    chart = resolve_preflop_range(position, scenario)
    if chart is not None:
        assert normalize_position(position) == chart.position


# --- villain-range resolution in the equity engine -------------------------------------


# What an operator can put in the villain-range box: a study label in any
# casing, raw eval7 notation, or whatever else they typed.
VILLAIN_RANGE_INPUT = st.one_of(
    st.sampled_from(sorted(RANGE_LABELS)),
    st.sampled_from(sorted(RANGE_LABELS)).map(str.upper),
    st.sampled_from([chart.notation for chart in available_ranges()]),
    unweighted_token().map(lambda token: token[0]),
    st.text(max_size=24),
)


@SETTINGS
@given(VILLAIN_RANGE_INPUT)
def test_resolved_villain_notation_always_parses_to_a_real_range(text: str) -> None:
    """`_resolve_range` accepts raw notation, so it must never hand back a
    notation the equity engine cannot expand."""
    label, resolved = _resolve_range(text)
    assert isinstance(label, str)
    if resolved is None:
        return
    assert eval7.HandRange(resolved).hands


@SETTINGS
@given(VILLAIN_RANGE_INPUT)
def test_resolving_a_resolved_range_returns_the_same_range(text: str) -> None:
    """Resolution is a fixed point: feeding a resolved notation back in is a no-op.

    Whatever the equity engine records as the villain range has to resolve to
    itself, or a re-run of the same review would silently score a different range.
    """
    _, resolved = _resolve_range(text)
    assume(resolved is not None)
    relabel, reresolved = _resolve_range(resolved)
    assert reresolved is not None
    assert eval7.HandRange(reresolved).hands == eval7.HandRange(resolved).hands, text
    assert (relabel, reresolved) == _resolve_range(reresolved)


@pytest.mark.parametrize("label", sorted(RANGE_LABELS))
def test_a_known_label_resolves_to_its_own_definition(label: str) -> None:
    for spelling in (label, label.upper(), f"  {label}  "):
        assert _resolve_range(spelling) == (label, range_notation(label))
