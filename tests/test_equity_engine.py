"""Real equity engine (eval7-backed) tests.

Asserts textbook equities within tolerance, exact-vs-Monte-Carlo method
selection, reproducibility of the seeded preflop path, card removal, and the
`unknown`-range honesty (equity=None). eval7 is a declared dependency, so these
run unconditionally; the import is guarded only to skip cleanly if it is absent.
"""
import pytest

from poker_tracker.equity import (
    Eval7EquityCalculator,
    EquityResult,
    PlaceholderEquityCalculator,
    _HAS_EVAL7,
    get_equity_calculator,
)

pytestmark = pytest.mark.skipif(not _HAS_EVAL7, reason="eval7 not installed")


def _eq(hero, board, villain_range, **kwargs):
    return Eval7EquityCalculator(**kwargs).calculate_equity(hero, board, villain_range)


def test_factory_returns_real_engine_when_eval7_present():
    assert isinstance(get_equity_calculator(), Eval7EquityCalculator)


def test_preflop_uses_seeded_monte_carlo_and_is_reproducible():
    a = _eq("As Ah", "", "premium")
    b = _eq("As Ah", "", "premium")
    assert a.method == "monte_carlo"
    assert a.equity == b.equity  # deterministic across runs
    # AA vs {AA,KK,QQ,AKs,AKo}: AA blocked (Hero has both aces), so ~KK/QQ/AK -> strong.
    assert 0.80 <= a.equity <= 0.90
    assert a.confidence == pytest.approx(0.85)


def test_postflop_uses_exact_enumeration():
    result = _eq("As Ah", "Kd 7c 2h", "premium")
    assert result.method == "enumeration"
    assert result.confidence == pytest.approx(0.95)
    assert result.equity is not None


def test_river_equity_is_exact_win_or_loss():
    # Hero has the nut flush; villain premium range cannot beat it here -> 100%.
    win = _eq("Ah Kh", "Qh Jh Th 2c 3d", "premium")
    assert win.method == "enumeration"
    assert win.equity == pytest.approx(1.0)


def test_unknown_range_returns_none_equity_honestly():
    result = _eq("As Ks", "", "unknown")
    assert result.equity is None
    assert result.method == "range_unavailable"
    assert result.confidence == 0.0


def test_card_removal_blocks_conflicting_villain_combos():
    # Hero holds AsKd; villain "premium" combos using those exact cards are removed,
    # but the range still has valid combos, so equity is a real number.
    result = _eq("As Kd", "Qh Jc 2d", "premium")
    assert result.equity is not None
    assert result.method == "enumeration"


def test_result_shape_and_normalization():
    result = _eq("ah qs", "qd 7s 2c", "loose")
    assert isinstance(result, EquityResult)
    assert result.hero_hand == "AhQs"
    assert result.board == "Qd 7s 2c"
    assert result.villain_range_label == "loose"
    assert 0.0 <= result.equity <= 1.0


def test_dominated_hand_has_low_equity_vs_premium():
    # 72o vs a premium range on a dry board should be poor.
    result = _eq("7d 2c", "Ah Ks Qh", "premium")
    assert result.equity < 0.15


def test_placeholder_still_available_as_labeled_fallback():
    placeholder = PlaceholderEquityCalculator().calculate_equity("As Ah", "", "premium")
    assert placeholder.method == "placeholder"
    assert placeholder.confidence == pytest.approx(0.2)
