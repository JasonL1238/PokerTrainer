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


def test_custom_range_notation_accepted():
    result = _eq("As Ah", "", "KK")
    assert result.method == "monte_carlo"
    assert result.equity is not None


def test_invalid_custom_notation_falls_back_to_unavailable():
    result = _eq("As Ah", "", "not a range !!!")
    assert result.equity is None
    assert result.method == "range_unavailable"


def test_golden_value_aa_vs_kk():
    # Textbook: AA vs KK preflop ~= 81.9%. Seeded MC at 100k iters, tight band.
    result = _eq("As Ah", "", "KK")
    assert result.equity == pytest.approx(0.819, abs=0.01)


def test_golden_value_ak_suited_vs_pair_below():
    # Classic coin flip: AKs vs 22 preflop ~= 49-50%.
    result = _eq("Ah Kh", "", "22")
    assert result.equity == pytest.approx(0.50, abs=0.015)


def test_board_plays_ties_split_exactly():
    # Royal flush on board: every runout is a tie -> equity exactly 0.5.
    result = _eq("2c 2d", "As Ks Qs Js Ts", "premium")
    assert result.method == "enumeration"
    assert result.equity == pytest.approx(0.5)


def test_call_ev_consistent_with_required_equity():
    # Cross-module consistency: at the required calling equity, EV(call) == 0.
    from poker_tracker.ev import call_ev
    from poker_tracker.pot_odds import required_equity_to_call

    equity = required_equity_to_call(30, 90)
    assert call_ev(equity, 90, 30) == pytest.approx(0.0)


def test_monte_carlo_reports_standard_error():
    result = _eq("Ah Kh", "", "standard", iterations=20_000)
    assert result.method == "monte_carlo"
    assert result.std_error is not None
    assert 0 < result.std_error < 0.01


def test_enumeration_has_no_standard_error():
    result = _eq("Ah Kh", "Qh Jh Th", "premium")
    assert result.method == "enumeration"
    assert result.std_error is None


def test_multiway_equity_pot_share():
    calc = Eval7EquityCalculator(iterations=20_000)
    result = calc.calculate_equity_multiway("As Ad", "Ac 7h 2d", ["KK", "QQ"])
    assert result.method == "monte_carlo_multiway"
    # Top set vs two dominated pairs should hold nearly the whole pot.
    assert result.equity == pytest.approx(0.997, abs=0.005)
    assert result.std_error is not None


def test_multiway_equity_is_reproducible_and_below_heads_up():
    calc = Eval7EquityCalculator(iterations=20_000)
    heads_up = calc.calculate_equity("Ah Kh", "", "standard").equity
    first = calc.calculate_equity_multiway("Ah Kh", "", ["standard", "loose"])
    second = calc.calculate_equity_multiway("Ah Kh", "", ["standard", "loose"])
    assert first.equity == second.equity  # seeded → deterministic
    assert first.equity < heads_up  # a second live range always costs pot share


def test_multiway_requires_two_ranges_and_honest_unknown():
    calc = Eval7EquityCalculator(iterations=1_000)
    with pytest.raises(ValueError):
        calc.calculate_equity_multiway("Ah Kh", "", ["standard"])
    result = calc.calculate_equity_multiway("Ah Kh", "", ["standard", "unknown"])
    assert result.equity is None
    assert result.method == "range_unavailable"
