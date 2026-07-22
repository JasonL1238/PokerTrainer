from __future__ import annotations

# Malmuth-Harville is factorial in paid places; cap field size to keep the
# recursion tractable.
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


def icm_risk_premium(
    stacks: list[float],
    payouts: list[float],
    hero_index: int,
    risk_amount: float,
) -> float:
    """Return the $EV cost of Hero losing `risk_amount` chips.

    Formula: icm_equities(current)[hero] minus Hero's equity after reducing
    Hero's stack by `risk_amount` (the lost chips are removed from play, e.g.
    shipped to an opponent covered elsewhere in the analysis).
    """
    if hero_index < 0 or hero_index >= len(stacks):
        raise ValueError("hero_index must be a valid index into stacks.")
    if risk_amount <= 0:
        raise ValueError("risk_amount must be positive.")
    if risk_amount >= stacks[hero_index]:
        raise ValueError("risk_amount must be less than the hero stack.")
    current = icm_equities(stacks, payouts)[hero_index]
    reduced_stacks = list(stacks)
    reduced_stacks[hero_index] -= risk_amount
    reduced = icm_equities(reduced_stacks, payouts)[hero_index]
    return current - reduced


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
    if any(later > earlier for earlier, later in zip(payouts, payouts[1:])):
        raise ValueError("payouts must be non-increasing.")
