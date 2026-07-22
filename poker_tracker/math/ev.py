from __future__ import annotations


def call_ev(equity: float, pot_before_call: float, call_amount: float) -> float:
    """Approximate EV of calling.

    Assumes no future betting. `pot_before_call` is the pot Hero is facing,
    including the bet to be called. Winning nets the pot; losing costs the call.
    Formula: equity * pot_before_call - (1 - equity) * call_amount.
    This is consistent with `required_equity_to_call`: EV is exactly 0 at the
    break-even equity call/(pot+call).
    """
    _require_probability(equity, "equity")
    _require_positive(pot_before_call, "pot_before_call")
    _require_positive(call_amount, "call_amount")
    return equity * pot_before_call - (1 - equity) * call_amount


def bluff_ev(fold_frequency: float, pot_size: float, bet_size: float) -> float:
    """Approximate EV of a pure bluff.

    Assumes Hero has zero equity when called.
    Formula: fold_frequency * pot_size - (1 - fold_frequency) * bet_size.
    """
    _require_probability(fold_frequency, "fold_frequency")
    _require_positive(pot_size, "pot_size")
    _require_positive(bet_size, "bet_size")
    return fold_frequency * pot_size - (1 - fold_frequency) * bet_size


def semi_bluff_ev(
    fold_frequency: float,
    equity_when_called: float,
    pot_size: float,
    bet_size: float,
) -> float:
    """Approximate EV of a semi-bluff.

    Assumes folds win the current pot immediately. When called, Hero wins
    `pot_size + bet_size` (the pot plus villain's call) with probability
    `equity_when_called` and loses the bet otherwise.
    """
    _require_probability(fold_frequency, "fold_frequency")
    _require_probability(equity_when_called, "equity_when_called")
    _require_positive(pot_size, "pot_size")
    _require_positive(bet_size, "bet_size")
    called_ev = equity_when_called * (pot_size + bet_size) - (1 - equity_when_called) * bet_size
    return fold_frequency * pot_size + (1 - fold_frequency) * called_ev


def _require_probability(value: float, name: str) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
