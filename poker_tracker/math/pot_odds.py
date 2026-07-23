from __future__ import annotations


def required_equity_to_call(call_amount: float, pot_before_call: float) -> float:
    """Return break-even equity required to call.

    Formula: call_amount / (pot_before_call + call_amount).
    """
    _require_positive(call_amount, "call_amount")
    _require_positive(pot_before_call, "pot_before_call")
    return call_amount / (pot_before_call + call_amount)


def rake_amount(
    gross_pot: float,
    rake_rate: float = 0.0,
    rake_cap: float | None = None,
) -> float:
    """Return rake removed from a completed pot.

    ``rake_rate`` is a decimal (``0.05`` means 5%). ``rake_cap`` uses the
    same units as the pot. Room-specific rules such as no-flop-no-drop remain
    an explicit user assumption rather than an invisible guess.
    """
    if gross_pot < 0:
        raise ValueError("gross_pot must not be negative.")
    if rake_rate < 0 or rake_rate > 1:
        raise ValueError("rake_rate must be between 0 and 1.")
    if rake_cap is not None and rake_cap < 0:
        raise ValueError("rake_cap must not be negative.")
    uncapped = gross_pot * rake_rate
    return uncapped if rake_cap is None else min(uncapped, rake_cap)


def required_equity_to_call_after_rake(
    call_amount: float,
    pot_before_call: float,
    rake_rate: float = 0.0,
    rake_cap: float | None = None,
) -> float:
    """Return break-even call equity after modeled rake."""
    _require_positive(call_amount, "call_amount")
    _require_positive(pot_before_call, "pot_before_call")
    gross_final_pot = pot_before_call + call_amount
    net_final_pot = gross_final_pot - rake_amount(gross_final_pot, rake_rate, rake_cap)
    if net_final_pot <= 0:
        raise ValueError("Rake leaves no distributable pot.")
    return call_amount / net_final_pot


def break_even_bluff_frequency(bet_size: float, pot_size: float) -> float:
    """Return how often a pure bluff must work before considering equity when called."""
    _require_positive(bet_size, "bet_size")
    _require_positive(pot_size, "pot_size")
    return bet_size / (pot_size + bet_size)


def minimum_defense_frequency(bet_size: float, pot_size: float) -> float:
    """Return the MDF: how often the defender must continue vs a bet.

    Formula: pot_size / (pot_size + bet_size). This is the complement of the
    break-even bluff frequency (alpha). Defending less often than the MDF makes
    any-two-cards bluffs immediately profitable for the bettor.
    """
    _require_positive(bet_size, "bet_size")
    _require_positive(pot_size, "pot_size")
    return pot_size / (pot_size + bet_size)


def pot_odds_offered_by_bet(bet_size: float, pot_size: float) -> float:
    """Return the caller's break-even equity against a heads-up bet.

    A bet ``B`` into pot ``P`` creates a ``P + 2B`` final pot if called, so
    the price is ``B / (P + 2B)``.
    """
    _require_positive(bet_size, "bet_size")
    _require_positive(pot_size, "pot_size")
    return bet_size / (pot_size + 2 * bet_size)


def stack_to_pot_ratio(effective_stack: float, pot_size: float) -> float:
    """Return effective stack divided by the current pot (SPR)."""
    if effective_stack < 0:
        raise ValueError("effective_stack must not be negative.")
    _require_positive(pot_size, "pot_size")
    return effective_stack / pot_size


def format_percentage(value: float) -> str:
    """Format a decimal probability as a one-decimal percentage."""
    if value < 0 or value > 1:
        raise ValueError("Percentage value must be between 0 and 1.")
    return f"{value * 100:.1f}%"


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
