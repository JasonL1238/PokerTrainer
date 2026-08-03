from __future__ import annotations

from dataclasses import dataclass

# The Malmuth-Harville recursion memoizes on the set of surviving players, so it
# costs O(2^n * n); cap the field size to keep that tractable.
_MAX_PLAYERS = 10


def icm_equities(stacks: list[float], payouts: list[float]) -> list[float]:
    """Return each player's ICM equity under the Malmuth-Harville model.

    Model: P(player i finishes 1st) = stack_i / total_chips. Conditional on a
    winner, the same rule is applied recursively to the remaining players for
    2nd place, and so on through the paid places. Each player's equity is the
    sum over places of P(finish in place) * payout for that place, so results
    are in the same units as `payouts` and returned in `stacks` order.
    """
    _validate_stacks_and_payouts(stacks, payouts)

    n = len(stacks)
    memo: dict[frozenset[int], dict[int, float]] = {}

    def solve(remaining: frozenset[int]) -> dict[int, float]:
        cached = memo.get(remaining)
        if cached is not None:
            return cached
        # Place index is implied by how many players have already finished.
        place = n - len(remaining)
        equities = {i: 0.0 for i in remaining}
        total = sum(stacks[i] for i in remaining)
        for winner in remaining:
            p_win = stacks[winner] / total
            equities[winner] += p_win * payouts[place]
            if place + 1 < len(payouts) and len(remaining) > 1:
                for player, equity in solve(remaining - {winner}).items():
                    equities[player] += p_win * equity
        memo[remaining] = equities
        return equities

    result = solve(frozenset(range(n)))
    return [result.get(i, 0.0) for i in range(n)]


@dataclass(frozen=True)
class IcmRiskPremiumRange:
    """Hero's risk premium across the opponents who could win the risked chips.

    `by_opponent` maps each opponent's index in `stacks` to the premium Hero
    pays if that opponent wins the chips; `low` and `high` are the smallest and
    largest of those. The span is over single winners only - see
    `icm_risk_premium` for what that does and does not cover.
    """

    low: float
    high: float
    by_opponent: dict[int, float]


def icm_risk_premium_by_opponent(
    stacks: list[float],
    payouts: list[float],
    hero_index: int,
    risk_amount: float,
) -> dict[int, float]:
    """Return the premium Hero pays for each opponent who could win the chips."""
    _validate_risk(stacks, payouts, hero_index, risk_amount)
    current = icm_equities(stacks, payouts)[hero_index]
    return {
        opponent: current - _equity_after_transfer(
            stacks, payouts, hero_index, risk_amount, opponent
        )
        for opponent in range(len(stacks))
        if opponent != hero_index
    }


def icm_risk_premium_range(
    stacks: list[float],
    payouts: list[float],
    hero_index: int,
    risk_amount: float,
) -> IcmRiskPremiumRange:
    """Return the low/high risk premium over the opponent who wins the chips.

    The answer genuinely depends on who Hero pays, so this is the shape callers
    should display when they can show more than one number.
    """
    by_opponent = icm_risk_premium_by_opponent(stacks, payouts, hero_index, risk_amount)
    return IcmRiskPremiumRange(
        low=min(by_opponent.values()),
        high=max(by_opponent.values()),
        by_opponent=by_opponent,
    )


def icm_risk_premium(
    stacks: list[float],
    payouts: list[float],
    hero_index: int,
    risk_amount: float,
    *,
    winner_index: int | None = None,
) -> float:
    """Return the $EV cost to Hero of losing `risk_amount` chips.

    Chips are conserved: what Hero loses is added to the opponent who wins it.
    ICM equity is a function of stack *shares*, so deleting the risked chips
    from the table instead shrinks the denominator and inflates Hero's
    post-loss share. That formulation reports a premium below every outcome the
    tournament can produce - on [5000, 3000, 2000] paying [50, 30, 20] it
    returns 1.57 when the real answer is between 2.73 and 2.96 - so "you cannot
    know which opponent won the chips" is the reason it is wrong, not a defence
    of it.

    Which opponent wins the chips does change the answer, so with
    `winner_index` this is exact for that opponent, and without it the largest
    single-winner premium is returned. That default is deliberate: it is
    conservative for a risk decision, and it is a premium some real outcome
    produces, which an average over opponents would not be. Callers that can
    render a span should use `icm_risk_premium_range`.

    Two things this does not model, both of which can move the true premium
    outside the single-winner span. Hero's chips may be split between several
    opponents in a multiway all-in, and a split can cost Hero more than any
    single winner would. An opponent shorter than `risk_amount` cannot win all
    of it in one pot, so their entry describes a run of pots rather than one
    confrontation.
    """
    by_opponent = icm_risk_premium_by_opponent(stacks, payouts, hero_index, risk_amount)
    if winner_index is None:
        return max(by_opponent.values())
    if winner_index not in by_opponent:
        raise ValueError("winner_index must be an opponent's index into stacks.")
    return by_opponent[winner_index]


def _equity_after_transfer(
    stacks: list[float],
    payouts: list[float],
    hero_index: int,
    risk_amount: float,
    winner_index: int,
) -> float:
    moved = list(stacks)
    moved[hero_index] -= risk_amount
    moved[winner_index] += risk_amount
    return icm_equities(moved, payouts)[hero_index]


def _validate_risk(
    stacks: list[float],
    payouts: list[float],
    hero_index: int,
    risk_amount: float,
) -> None:
    _validate_stacks_and_payouts(stacks, payouts)
    if hero_index < 0 or hero_index >= len(stacks):
        raise ValueError("hero_index must be a valid index into stacks.")
    if risk_amount <= 0:
        raise ValueError("risk_amount must be positive.")
    if risk_amount >= stacks[hero_index]:
        raise ValueError("risk_amount must be less than the hero stack.")


def _validate_stacks_and_payouts(stacks: list[float], payouts: list[float]) -> None:
    if len(stacks) < 2:
        raise ValueError("at least 2 players are required.")
    if len(stacks) > _MAX_PLAYERS:
        raise ValueError(f"at most {_MAX_PLAYERS} players are supported.")
    if any(stack <= 0 for stack in stacks):
        raise ValueError("stacks must be positive.")
    if not payouts:
        raise ValueError("payouts must be non-empty.")
    if len(payouts) > len(stacks):
        raise ValueError("payouts must not exceed the number of players.")
    if any(payout < 0 for payout in payouts):
        raise ValueError("payouts must be non-negative.")
    if any(later > earlier for earlier, later in zip(payouts, payouts[1:], strict=False)):
        raise ValueError("payouts must be non-increasing.")
