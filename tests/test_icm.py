from __future__ import annotations

import pytest

from poker_tracker.icm import icm_equities, icm_risk_premium


def test_heads_up_golden_value() -> None:
    # P1st = 0.75, P2nd = 0.25 -> 0.6 * 0.75 + 0.4 * 0.25 = 0.55
    equities = icm_equities([75, 25], [0.6, 0.4])
    assert equities[0] == pytest.approx(0.55)
    assert equities[1] == pytest.approx(0.45)


def test_three_handed_golden_value() -> None:
    # P1st = 0.5, P2nd = 0.3 * (50/70) + 0.2 * (50/80) = 0.3392857,
    # P3rd = 0.1607143 -> equity = 0.25 + 0.10178571 + 0.03214286
    equities = icm_equities([50, 30, 20], [0.5, 0.3, 0.2])
    assert equities[0] == pytest.approx(0.3839286, abs=1e-6)


def test_equal_stacks_winner_take_all() -> None:
    equities = icm_equities([100, 100, 100, 100], [1.0])
    for equity in equities:
        assert equity == pytest.approx(0.25)


def test_equities_sum_to_total_payouts() -> None:
    stacks = [120.0, 74.0, 51.0, 23.0, 9.0]
    payouts = [0.45, 0.27, 0.18, 0.1]
    equities = icm_equities(stacks, payouts)
    assert sum(equities) == pytest.approx(sum(payouts))


def test_larger_stack_never_has_less_equity() -> None:
    equities = icm_equities([40, 30, 20, 10], [0.5, 0.3, 0.2])
    assert equities == sorted(equities, reverse=True)


def test_ten_players_is_fast_and_consistent() -> None:
    stacks = [float(i) for i in range(1, 11)]
    payouts = [50.0, 30.0, 20.0, 10.0, 5.0]
    equities = icm_equities(stacks, payouts)
    assert sum(equities) == pytest.approx(sum(payouts))


def test_single_player_raises() -> None:
    with pytest.raises(ValueError):
        icm_equities([100], [1.0])


def test_zero_stack_raises() -> None:
    with pytest.raises(ValueError):
        icm_equities([100, 0], [0.6, 0.4])


def test_ascending_payouts_raise() -> None:
    with pytest.raises(ValueError):
        icm_equities([50, 30, 20], [0.2, 0.3, 0.5])


def test_payouts_longer_than_stacks_raise() -> None:
    with pytest.raises(ValueError):
        icm_equities([50, 50], [0.5, 0.3, 0.2])


def test_empty_payouts_raise() -> None:
    with pytest.raises(ValueError):
        icm_equities([50, 50], [])


def test_more_than_ten_players_raise() -> None:
    with pytest.raises(ValueError):
        icm_equities([10.0] * 11, [1.0])


def test_risk_premium_positive_on_bubble() -> None:
    premium = icm_risk_premium([40, 30, 20, 10], [0.5, 0.3, 0.2], 0, 10)
    assert premium > 0


def test_risk_premium_invalid_hero_index_raises() -> None:
    with pytest.raises(ValueError):
        icm_risk_premium([40, 30], [0.6, 0.4], 2, 10)


def test_risk_premium_invalid_risk_amount_raises() -> None:
    with pytest.raises(ValueError):
        icm_risk_premium([40, 30], [0.6, 0.4], 0, 0)
    with pytest.raises(ValueError):
        icm_risk_premium([40, 30], [0.6, 0.4], 0, 40)
