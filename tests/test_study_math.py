from __future__ import annotations

import math

import pytest

from poker_tracker.study_math import (
    REALIZATION_FACTOR_GUIDE,
    bluff_to_value_ratio,
    mean_confidence_interval,
    monte_carlo_confidence_interval,
    monte_carlo_standard_error,
    optimal_bluff_fraction,
    outs_to_equity_exact,
    outs_to_equity_rule,
    realized_equity,
)


class TestOutsToEquityRule:
    def test_nine_outs_one_street(self):
        assert outs_to_equity_rule(9, 1) == pytest.approx(0.18)

    def test_nine_outs_two_streets(self):
        assert outs_to_equity_rule(9, 2) == pytest.approx(0.36)

    def test_fifteen_outs_two_streets(self):
        assert outs_to_equity_rule(15, 2) == pytest.approx(0.60)

    def test_zero_outs(self):
        assert outs_to_equity_rule(0, 1) == pytest.approx(0.0)

    def test_caps_at_one(self):
        assert outs_to_equity_rule(20, 2) <= 1.0

    @pytest.mark.parametrize("outs", [-1, 21])
    def test_invalid_outs_raise(self, outs):
        with pytest.raises(ValueError):
            outs_to_equity_rule(outs, 1)

    @pytest.mark.parametrize("streets", [0, 3])
    def test_invalid_streets_raise(self, streets):
        with pytest.raises(ValueError):
            outs_to_equity_rule(9, streets)


class TestOutsToEquityExact:
    def test_nine_outs_flop_one_street(self):
        assert outs_to_equity_exact(9, 47, 1) == pytest.approx(9 / 47)
        assert outs_to_equity_exact(9, 47, 1) == pytest.approx(0.19149, abs=1e-5)

    def test_nine_outs_flop_two_streets(self):
        expected = 1 - math.comb(38, 2) / math.comb(47, 2)
        assert expected == pytest.approx(1 - 703 / 1081)
        assert outs_to_equity_exact(9, 47, 2) == pytest.approx(expected)
        assert outs_to_equity_exact(9, 47, 2) == pytest.approx(0.34968, abs=1e-5)

    def test_eight_outs_turn_one_street(self):
        assert outs_to_equity_exact(8, 46, 1) == pytest.approx(0.17391, abs=1e-5)

    def test_zero_outs(self):
        assert outs_to_equity_exact(0, 47, 2) == pytest.approx(0.0)

    def test_all_outs_is_certainty(self):
        assert outs_to_equity_exact(47, 47, 1) == pytest.approx(1.0)
        assert outs_to_equity_exact(47, 47, 2) == pytest.approx(1.0)

    def test_negative_outs_raise(self):
        with pytest.raises(ValueError):
            outs_to_equity_exact(-1, 47, 1)

    def test_outs_exceeding_unseen_raise(self):
        with pytest.raises(ValueError):
            outs_to_equity_exact(48, 47, 1)

    def test_unseen_not_exceeding_streets_raise(self):
        with pytest.raises(ValueError):
            outs_to_equity_exact(1, 2, 2)

    @pytest.mark.parametrize("streets", [0, 3])
    def test_invalid_streets_raise(self, streets):
        with pytest.raises(ValueError):
            outs_to_equity_exact(9, 47, streets)


class TestOptimalBluffFraction:
    def test_pot_sized_bet_is_one_third(self):
        assert optimal_bluff_fraction(100, 100) == pytest.approx(1 / 3)

    def test_half_pot_bet_is_quarter(self):
        assert optimal_bluff_fraction(50, 100) == pytest.approx(0.25)

    @pytest.mark.parametrize("bet,pot", [(0, 100), (-1, 100), (100, 0), (100, -5)])
    def test_non_positive_values_raise(self, bet, pot):
        with pytest.raises(ValueError):
            optimal_bluff_fraction(bet, pot)


class TestBluffToValueRatio:
    def test_pot_sized_bet_is_half(self):
        assert bluff_to_value_ratio(100, 100) == pytest.approx(0.5)

    def test_consistent_with_bluff_fraction(self):
        frac = optimal_bluff_fraction(75, 100)
        assert bluff_to_value_ratio(75, 100) == pytest.approx(frac / (1 - frac))

    @pytest.mark.parametrize("bet,pot", [(0, 100), (100, 0)])
    def test_non_positive_values_raise(self, bet, pot):
        with pytest.raises(ValueError):
            bluff_to_value_ratio(bet, pot)


class TestRealizedEquity:
    def test_basic_multiplication(self):
        assert realized_equity(0.5, 0.85) == pytest.approx(0.425)

    def test_caps_at_one(self):
        assert realized_equity(0.99, 1.05) == pytest.approx(1.0)

    def test_guide_values_are_usable(self):
        for factor in REALIZATION_FACTOR_GUIDE.values():
            assert 0 < factor <= 2
            assert 0.0 <= realized_equity(0.5, factor) <= 1.0

    @pytest.mark.parametrize("raw", [-0.1, 1.1])
    def test_invalid_equity_raises(self, raw):
        with pytest.raises(ValueError):
            realized_equity(raw, 1.0)

    @pytest.mark.parametrize("factor", [0, -0.5, 2.01])
    def test_invalid_factor_raises(self, factor):
        with pytest.raises(ValueError):
            realized_equity(0.5, factor)


class TestMonteCarloStandardError:
    def test_half_equity_ten_thousand_iterations(self):
        assert monte_carlo_standard_error(0.5, 10000) == pytest.approx(0.005)

    def test_degenerate_equity_has_zero_error(self):
        assert monte_carlo_standard_error(0.0, 1000) == pytest.approx(0.0)
        assert monte_carlo_standard_error(1.0, 1000) == pytest.approx(0.0)

    @pytest.mark.parametrize("equity", [-0.1, 1.1])
    def test_invalid_equity_raises(self, equity):
        with pytest.raises(ValueError):
            monte_carlo_standard_error(equity, 1000)

    @pytest.mark.parametrize("iterations", [0, -100])
    def test_invalid_iterations_raise(self, iterations):
        with pytest.raises(ValueError):
            monte_carlo_standard_error(0.5, iterations)


class TestMonteCarloConfidenceInterval:
    def test_half_equity_ten_thousand_iterations(self):
        low, high = monte_carlo_confidence_interval(0.5, 10000)
        assert low == pytest.approx(0.4902)
        assert high == pytest.approx(0.5098)

    def test_clamped_to_unit_interval(self):
        low, high = monte_carlo_confidence_interval(0.01, 10)
        assert low == 0.0
        low, high = monte_carlo_confidence_interval(0.99, 10)
        assert high == 1.0

    def test_invalid_z_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_confidence_interval(0.5, 10000, z=0)

    def test_invalid_equity_raises(self):
        with pytest.raises(ValueError):
            monte_carlo_confidence_interval(1.5, 10000)


class TestMeanConfidenceInterval:
    def test_one_two_three(self):
        mean, low, high = mean_confidence_interval([1.0, 2.0, 3.0])
        half_width = 1.96 * 1.0 / math.sqrt(3)
        assert mean == pytest.approx(2.0)
        assert half_width == pytest.approx(1.1316, abs=1e-4)
        assert low == pytest.approx(0.8684, abs=1e-4)
        assert high == pytest.approx(3.1316, abs=1e-4)

    def test_identical_values_collapse_interval(self):
        mean, low, high = mean_confidence_interval([2.0, 2.0, 2.0])
        assert (mean, low, high) == pytest.approx((2.0, 2.0, 2.0))

    @pytest.mark.parametrize("values", [[], [1.0]])
    def test_too_few_values_raise(self, values):
        with pytest.raises(ValueError):
            mean_confidence_interval(values)

    def test_invalid_z_raises(self):
        with pytest.raises(ValueError):
            mean_confidence_interval([1.0, 2.0], z=-1.0)
