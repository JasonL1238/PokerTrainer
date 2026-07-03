from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from poker_tracker.cards import compact_cards, parse_board_cards, parse_hero_cards, spaced_cards
from poker_tracker.ranges import RangeDescription, get_range_description, normalize_range_label


class EquityResult(BaseModel):
    hero_hand: str
    board: str
    villain_range_label: str
    equity: float | None = Field(default=None, ge=0, le=1)
    method: str
    confidence: float = Field(ge=0, le=1)
    notes: str


class EquityCalculator(Protocol):
    def calculate_equity(
        self,
        hero_cards: str,
        board_cards: str,
        villain_range: str,
    ) -> EquityResult:
        """Estimate Hero equity for post-session review."""


class PlaceholderEquityCalculator:
    """Deterministic low-confidence equity estimator.

    This is not a real poker equity engine. It exists so the UI, prompts, and
    review logic can be wired to an interface before a real calculator is added.
    """

    def calculate_equity(
        self,
        hero_cards: str,
        board_cards: str,
        villain_range: str,
    ) -> EquityResult:
        hero = parse_hero_cards(hero_cards)
        board = parse_board_cards(board_cards)
        label = normalize_range_label(villain_range)
        range_description = get_range_description(label)
        equity = _rough_placeholder_equity(hero, board, range_description)
        return EquityResult(
            hero_hand=compact_cards(hero),
            board=spaced_cards(board),
            villain_range_label=label,
            equity=equity,
            method="placeholder",
            confidence=0.2,
            notes="Not a real equity calculation yet. Use only as a rough placeholder.",
        )


def _rough_placeholder_equity(hero: list, board: list, villain_range: RangeDescription) -> float:
    base_by_range = {
        "premium": 0.34,
        "tight": 0.40,
        "standard": 0.47,
        "loose": 0.53,
        "very_loose": 0.58,
        "unknown": 0.50,
    }
    equity = base_by_range[villain_range.label]
    ranks = [card.rank for card in hero]
    board_ranks = [card.rank for card in board]
    if ranks[0] == ranks[1]:
        equity += 0.04
    if any(rank in board_ranks for rank in ranks):
        equity += 0.05
    if len(board) >= 4 and ranks[0] != ranks[1]:
        equity -= 0.02
    return max(0.05, min(0.95, round(equity, 3)))


# TODO: Add a real calculator implementation behind this interface later.
# TODO: Solver outputs should be stored/labeled separately from estimates.
