from __future__ import annotations

from itertools import permutations

import pytest

from poker_tracker.math.icm import (
    icm_equities,
    icm_risk_premium,
    icm_risk_premium_by_opponent,
    icm_risk_premium_range,
)


def _exhaustive_icm(stacks: list[float], payouts: list[float]) -> list[float]:
    """Reference ICM by enumerating every finishing order.

    Independent of the module's memoized recursion: each finishing order gets
    the product of its chip-proportional step probabilities, and every player
    collects the payout for the seat they finish in. Factorial, so tests keep
    the field small.
    """
    equities = [0.0] * len(stacks)
    for order in permutations(range(len(stacks))):
        probability = 1.0
        remaining = set(order)
        for player in order:
            probability *= stacks[player] / sum(stacks[i] for i in remaining)
            remaining.discard(player)
        for place, player in enumerate(order):
            if place < len(payouts):
                equities[player] += probability * payouts[place]
    return equities


def _exhaustive_premium(
    stacks: list[float],
    payouts: list[float],
    hero_index: int,
    risk_amount: float,
    winner_index: int,
) -> float:
    moved = list(stacks)
    moved[hero_index] -= risk_amount
    moved[winner_index] += risk_amount
    return _exhaustive_icm(stacks, payouts)[hero_index] - (
        _exhaustive_icm(moved, payouts)[hero_index]
    )


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


def test_equities_match_exhaustive_enumeration() -> None:
    for stacks, payouts in [
        ([75.0, 25.0], [0.6, 0.4]),
        ([50.0, 30.0, 20.0], [0.5, 0.3, 0.2]),
        ([40.0, 30.0, 20.0, 10.0], [0.5, 0.3, 0.2]),
        ([5000.0, 3000.0, 2000.0], [50.0, 30.0, 20.0]),
        ([120.0, 74.0, 51.0, 23.0, 9.0], [0.45, 0.27, 0.18, 0.1]),
        ([100.0, 100.0, 100.0, 100.0], [1.0]),
    ]:
        assert icm_equities(stacks, payouts) == pytest.approx(
            _exhaustive_icm(stacks, payouts)
        )


def test_risk_premium_matches_exhaustive_truth_for_each_winner() -> None:
    # The premium depends on who wins the chips, so every opponent's entry is
    # checked against the permutation reference rather than against the module.
    for stacks, payouts, hero, risk in [
        ([5000.0, 3000.0, 2000.0], [50.0, 30.0, 20.0], 0, 1000.0),
        ([40.0, 30.0, 20.0, 10.0], [0.5, 0.3, 0.2], 0, 10.0),
        ([14.0, 51.0, 8.0], [80.0, 60.0, 10.0], 1, 35.0),
        ([100.0, 100.0, 100.0, 100.0], [50.0, 30.0, 20.0], 3, 50.0),
    ]:
        by_opponent = icm_risk_premium_by_opponent(stacks, payouts, hero, risk)
        assert set(by_opponent) == {i for i in range(len(stacks)) if i != hero}
        for winner, premium in by_opponent.items():
            expected = _exhaustive_premium(stacks, payouts, hero, risk, winner)
            assert premium == pytest.approx(expected, abs=1e-9)
            assert icm_risk_premium(
                stacks, payouts, hero, risk, winner_index=winner
            ) == pytest.approx(expected, abs=1e-9)


def test_risk_premium_is_an_outcome_the_tournament_can_produce() -> None:
    # Removing the risked chips from play instead of paying them to someone
    # shrinks the chip total and inflates Hero's post-loss share, which lands
    # the premium below every real outcome. The reported value must be one of
    # the real outcomes.
    for stacks, payouts, hero, risk in [
        ([5000.0, 3000.0, 2000.0], [50.0, 30.0, 20.0], 0, 1000.0),
        ([40.0, 30.0, 20.0, 10.0], [0.5, 0.3, 0.2], 0, 10.0),
        ([100.0, 100.0, 100.0, 100.0], [50.0, 30.0, 20.0], 0, 50.0),
        ([9000.0, 500.0, 300.0, 200.0], [65.0, 35.0], 0, 200.0),
    ]:
        outcomes = [
            _exhaustive_premium(stacks, payouts, hero, risk, winner)
            for winner in range(len(stacks))
            if winner != hero
        ]
        premium = icm_risk_premium(stacks, payouts, hero, risk)
        assert min(outcomes) - 1e-9 <= premium <= max(outcomes) + 1e-9
        assert min(abs(premium - outcome) for outcome in outcomes) < 1e-9
        # The chips-vanish formulation is what this replaces; it must not come back.
        vanished = list(stacks)
        vanished[hero] -= risk
        deleted_premium = (
            icm_equities(stacks, payouts)[hero] - icm_equities(vanished, payouts)[hero]
        )
        assert deleted_premium < min(outcomes) - 1e-9


def test_risk_premium_default_is_the_worst_single_winner() -> None:
    stacks, payouts, hero, risk = [5000.0, 3000.0, 2000.0], [50.0, 30.0, 20.0], 0, 1000.0
    span = icm_risk_premium_range(stacks, payouts, hero, risk)
    assert icm_risk_premium(stacks, payouts, hero, risk) == pytest.approx(span.high)
    assert span.low == pytest.approx(min(span.by_opponent.values()))
    assert span.high == pytest.approx(max(span.by_opponent.values()))
    # The app's default ICM screen. Exhaustively true span: 2.7262..2.9643.
    assert span.low == pytest.approx(2.7262, abs=1e-4)
    assert span.high == pytest.approx(2.9643, abs=1e-4)


def test_docstring_numbers_for_the_default_screen_are_the_real_ones() -> None:
    # icm_risk_premium's docstring quotes this case; keep the quoted figures
    # honest rather than leaving them as prose nobody checks.
    stacks, payouts, hero, risk = [5000.0, 3000.0, 2000.0], [50.0, 30.0, 20.0], 0, 1000.0
    vanished = list(stacks)
    vanished[hero] -= risk
    deleted_premium = (
        icm_equities(stacks, payouts)[hero] - icm_equities(vanished, payouts)[hero]
    )
    assert deleted_premium == pytest.approx(1.57, abs=5e-3)
    span = icm_risk_premium_range(stacks, payouts, hero, risk)
    assert (span.low, span.high) == pytest.approx((2.73, 2.96), abs=5e-3)


def test_risk_premium_conserves_chips_in_every_scenario() -> None:
    stacks, payouts, hero, risk = [40.0, 30.0, 20.0, 10.0], [0.5, 0.3, 0.2], 0, 10.0
    # Hero pays a premium for chips that stay on the table, so the field's
    # equity gain must exactly offset Hero's loss.
    before = icm_equities(stacks, payouts)
    for winner in icm_risk_premium_by_opponent(stacks, payouts, hero, risk):
        moved = list(stacks)
        moved[hero] -= risk
        moved[winner] += risk
        assert sum(moved) == pytest.approx(sum(stacks))
        after = icm_equities(moved, payouts)
        assert sum(after) == pytest.approx(sum(before))


def test_split_chips_can_cost_more_than_any_single_winner() -> None:
    # Pins the caveat in icm_risk_premium's docstring: the single-winner span is
    # not a bound when Hero's chips are split across a multiway all-in.
    stacks, payouts, hero, risk = [14.0, 51.0, 8.0], [80.0, 60.0, 10.0], 1, 35.0
    span = icm_risk_premium_range(stacks, payouts, hero, risk)
    split = list(stacks)
    split[hero] -= risk
    split[0] += risk / 2
    split[2] += risk / 2
    current = icm_equities(stacks, payouts)[hero]
    split_premium = current - icm_equities(split, payouts)[hero]
    assert split_premium > span.high


def test_risk_premium_rejects_hero_as_the_winner() -> None:
    with pytest.raises(ValueError):
        icm_risk_premium([40, 30, 20], [0.6, 0.4], 0, 10, winner_index=0)
    with pytest.raises(ValueError):
        icm_risk_premium([40, 30, 20], [0.6, 0.4], 0, 10, winner_index=3)


def test_risk_premium_invalid_hero_index_raises() -> None:
    with pytest.raises(ValueError):
        icm_risk_premium([40, 30], [0.6, 0.4], 2, 10)


def test_risk_premium_invalid_risk_amount_raises() -> None:
    with pytest.raises(ValueError):
        icm_risk_premium([40, 30], [0.6, 0.4], 0, 0)
    with pytest.raises(ValueError):
        icm_risk_premium([40, 30], [0.6, 0.4], 0, 40)
