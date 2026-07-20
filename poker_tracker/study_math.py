from __future__ import annotations

import math
import statistics

# Rough study heuristics for equity realization, NOT solver output.
# Realization factor multiplies raw (all-in) equity to estimate the share of
# the pot a hand actually captures given position and range dynamics.
# Values are defensible baselines for post-session study discussion only.
REALIZATION_FACTOR_GUIDE: dict[str, float] = {
    "in_position_strong": 1.05,
    "in_position": 1.0,
    "out_of_position": 0.85,
    "out_of_position_capped": 0.75,
    "multiway_oop": 0.7,
}


def outs_to_equity_rule(outs: int, streets_to_come: int) -> float:
    """Estimate hit equity with the Rule of 2 and 4.

    Formula: outs * 0.02 with one street to come, outs * 0.04 with two
    streets to come. This is the standard at-the-table approximation of the
    exact combinatorial probability; the result is capped at 1.0.
    Requires 0 <= outs <= 20 and streets_to_come in (1, 2).
    """
    if outs < 0 or outs > 20:
        raise ValueError("outs must be between 0 and 20.")
    _require_streets(streets_to_come)
    return min(outs * 0.02 * streets_to_come, 1.0)


def outs_to_equity_exact(outs: int, unseen_cards: int, streets_to_come: int) -> float:
    """Exact probability of hitting at least one out.

    One street: outs / unseen_cards.
    Two streets: 1 - C(unseen_cards - outs, 2) / C(unseen_cards, 2),
    i.e. one minus the probability that both remaining cards miss.
    Typical usage: unseen_cards=47 on the flop (52 - 2 hero - 3 board),
    46 on the turn. Assumes villain cards are unseen (part of the deck).
    Requires 0 <= outs <= unseen_cards and unseen_cards > streets_to_come.
    """
    _require_streets(streets_to_come)
    if outs < 0:
        raise ValueError("outs must not be negative.")
    if outs > unseen_cards:
        raise ValueError("outs must not exceed unseen_cards.")
    if unseen_cards <= streets_to_come:
        raise ValueError("unseen_cards must exceed streets_to_come.")
    if streets_to_come == 1:
        return outs / unseen_cards
    return 1 - math.comb(unseen_cards - outs, 2) / math.comb(unseen_cards, 2)


def optimal_bluff_fraction(bet_size: float, pot_size: float) -> float:
    """River-optimal fraction of a polarized betting range that is bluffs.

    Formula: bet_size / (pot_size + 2 * bet_size). At this bluff fraction the
    caller is indifferent: pot odds offered equal the bettor's bluff density.
    Assumes a polarized river range (value hands always win when called,
    bluffs always lose). Check: pot-sized bet -> 1/3.
    """
    _require_positive(bet_size, "bet_size")
    _require_positive(pot_size, "pot_size")
    return bet_size / (pot_size + 2 * bet_size)


def bluff_to_value_ratio(bet_size: float, pot_size: float) -> float:
    """Optimal number of bluff combos per value combo on the river.

    Formula: optimal_bluff_fraction / (1 - optimal_bluff_fraction)
           = bet_size / (pot_size + bet_size).
    Same polarized-range assumptions as `optimal_bluff_fraction`.
    Check: pot-sized bet -> 0.5, i.e. 1 bluff per 2 value bets.
    """
    _require_positive(bet_size, "bet_size")
    _require_positive(pot_size, "pot_size")
    return bet_size / (pot_size + bet_size)


def realized_equity(raw_equity: float, realization_factor: float) -> float:
    """Estimate realized equity: raw_equity * realization_factor.

    The realization factor discounts (or boosts) raw all-in equity for
    positional and playability effects; see REALIZATION_FACTOR_GUIDE for
    rough study baselines (heuristics, not solver output). The result is
    capped to [0, 1]. Requires raw_equity in [0, 1] and
    realization_factor in (0, 2].
    """
    _require_probability(raw_equity, "raw_equity")
    if realization_factor <= 0 or realization_factor > 2:
        raise ValueError("realization_factor must be in (0, 2].")
    return min(max(raw_equity * realization_factor, 0.0), 1.0)


def monte_carlo_standard_error(equity: float, iterations: int) -> float:
    """Standard error of a Monte Carlo equity estimate.

    Formula (binomial SE): sqrt(equity * (1 - equity) / iterations).
    Assumes each iteration is an independent Bernoulli trial.
    """
    _require_probability(equity, "equity")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    return math.sqrt(equity * (1 - equity) / iterations)


def monte_carlo_confidence_interval(
    equity: float,
    iterations: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Normal-approximation confidence interval for a Monte Carlo equity.

    Formula: (equity - z * SE, equity + z * SE) with
    SE = sqrt(equity * (1 - equity) / iterations), clamped to [0, 1].
    Default z=1.96 gives a ~95% interval.
    """
    _require_positive(z, "z")
    se = monte_carlo_standard_error(equity, iterations)
    return (max(equity - z * se, 0.0), min(equity + z * se, 1.0))


def mean_confidence_interval(
    values: list[float],
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Confidence interval for the mean of a sample.

    Returns (mean, low, high) where low/high = mean -/+ z * s / sqrt(n),
    with s the sample standard deviation (n-1 denominator). Uses the normal
    approximation (z, not t), which is fine for study-sized samples.
    Requires at least 2 values and z > 0.
    """
    _require_positive(z, "z")
    if len(values) < 2:
        raise ValueError("values must contain at least 2 items.")
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    return (mean, mean - z * se, mean + z * se)


def _require_streets(streets_to_come: int) -> None:
    if streets_to_come not in (1, 2):
        raise ValueError("streets_to_come must be 1 or 2.")


def _require_probability(value: float, name: str) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
