from __future__ import annotations


def call_ev(equity: float, pot_before_call: float, call_amount: float) -> float:
    """Approximate EV of calling.

    Assumes no future betting and that `equity` is Hero's chance to win the final
    pot after calling. Formula:
    equity * (pot_before_call + call_amount) - (1 - equity) * call_amount.
    """
    _require_probability(equity, "equity")
    _require_positive(pot_before_call, "pot_before_call")
    _require_positive(call_amount, "call_amount")
    return equity * (pot_before_call + call_amount) - (1 - equity) * call_amount


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

    Assumes folds win the current pot immediately, and calls realize the supplied
    equity against a final pot of `pot_size + bet_size`.
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
