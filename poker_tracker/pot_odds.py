from __future__ import annotations


def required_equity_to_call(call_amount: float, pot_before_call: float) -> float:
    """Return break-even equity required to call.

    Formula: call_amount / (pot_before_call + call_amount).
    """
    _require_positive(call_amount, "call_amount")
    _require_positive(pot_before_call, "pot_before_call")
    return call_amount / (pot_before_call + call_amount)


def break_even_bluff_frequency(bet_size: float, pot_size: float) -> float:
    """Return how often a pure bluff must work before considering equity when called."""
    _require_positive(bet_size, "bet_size")
    _require_positive(pot_size, "pot_size")
    return bet_size / (pot_size + bet_size)


def value_bet_call_threshold(bet_size: float, pot_size: float) -> float:
    """Return a simple call-frequency threshold for comparing value/bluff pressure.

    This uses the same risk/reward ratio as a break-even bluff. It is a review aid,
    not a complete value-betting model.
    """
    return break_even_bluff_frequency(bet_size, pot_size)


def format_percentage(value: float) -> str:
    """Format a decimal probability as a one-decimal percentage."""
    if value < 0 or value > 1:
        raise ValueError("Percentage value must be between 0 and 1.")
    return f"{value * 100:.1f}%"


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
