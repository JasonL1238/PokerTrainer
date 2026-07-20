from __future__ import annotations

import random
from typing import Protocol

from pydantic import BaseModel, Field

from poker_tracker.cards import (
    Card,
    compact_cards,
    parse_board_cards,
    parse_hero_cards,
    parse_visible_cards,
    spaced_cards,
)
from poker_tracker.ranges import (
    RangeDescription,
    get_range_description,
    normalize_range_label,
    range_notation,
)

try:  # eval7 is the equity engine's evaluator + range parser (optional at import time).
    import eval7

    _HAS_EVAL7 = True
except ImportError:  # pragma: no cover - exercised only when the dependency is absent.
    eval7 = None
    _HAS_EVAL7 = False


# Preflop equity is estimated with a *seeded* Monte-Carlo so results are exact-reproducible
# across runs (deterministic, per the CV-lab discipline). Postflop is exact enumeration.
MONTE_CARLO_ITERATIONS = 100_000
MONTE_CARLO_SEED = 1_234_567
EXACT_CONFIDENCE = 0.95
MONTE_CARLO_CONFIDENCE = 0.85


class EquityResult(BaseModel):
    hero_hand: str
    board: str
    villain_range_label: str
    equity: float | None = Field(default=None, ge=0, le=1)
    method: str
    confidence: float = Field(ge=0, le=1)
    notes: str
    # Monte-Carlo sampling error (one standard error of the equity estimate);
    # None for exact enumeration and unavailable results.
    std_error: float | None = Field(default=None, ge=0)


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


class Eval7EquityCalculator:
    """Real equity engine backed by eval7 (hand evaluator + range parser).

    Postflop (a known flop/turn/river) uses exact enumeration; preflop uses a
    seeded Monte-Carlo so the result is reproducible run to run. Card removal is
    applied so villain combos never reuse Hero or board cards. This computes
    Hero-vs-range hot/cold equity, not solver strategy, and reports the method
    and confidence it actually used.
    """

    def __init__(
        self,
        *,
        iterations: int = MONTE_CARLO_ITERATIONS,
        seed: int = MONTE_CARLO_SEED,
    ) -> None:
        if not _HAS_EVAL7:  # pragma: no cover - guarded by get_equity_calculator().
            raise RuntimeError("eval7 is not installed; cannot build a real equity engine.")
        self.iterations = iterations
        self.seed = seed

    def calculate_equity(
        self,
        hero_cards: str,
        board_cards: str,
        villain_range: str,
    ) -> EquityResult:
        hero = parse_hero_cards(hero_cards)
        board = parse_board_cards(board_cards)
        parse_visible_cards(hero_cards, board_cards)  # reject hero/board card overlap
        label, notation = _resolve_range(villain_range)

        if notation is None:
            return self._result(
                hero,
                board,
                label,
                equity=None,
                method="range_unavailable",
                confidence=0.0,
                notes=(
                    "No range definition for an unknown villain; add tags/notes to "
                    "estimate a villain range before computing equity."
                ),
            )

        dead = {str(card) for card in hero} | {str(card) for card in board}
        combos = [
            (hand, weight)
            for hand, weight in eval7.HandRange(notation).hands
            if str(hand[0]) not in dead and str(hand[1]) not in dead
        ]
        if not combos:
            return self._result(
                hero,
                board,
                label,
                equity=None,
                method="no_valid_combos",
                confidence=0.0,
                notes="Every hand in the villain range is blocked by Hero/board cards.",
            )

        if len(board) >= 3:
            equity = self._exact(hero, board, combos)
            method, confidence, std_error = "enumeration", EXACT_CONFIDENCE, None
            notes = f"Exact enumeration vs a {len(combos)}-combo {label} range."
        else:
            equity, std_error = self._monte_carlo(hero, board, combos)
            method, confidence = "monte_carlo", MONTE_CARLO_CONFIDENCE
            notes = (
                f"Seeded Monte-Carlo ({self.iterations:,} iterations) vs a "
                f"{len(combos)}-combo {label} range."
            )

        return self._result(
            hero,
            board,
            label,
            equity=round(equity, 4),
            method=method,
            confidence=confidence,
            notes=notes,
            std_error=std_error,
        )

    def calculate_equity_multiway(
        self,
        hero_cards: str,
        board_cards: str,
        villain_ranges: list[str],
    ) -> EquityResult:
        """Hero pot-share equity vs two or more villain ranges.

        Uses a seeded Monte-Carlo (villain combos sampled with rejection so no
        card is dealt twice); each showdown awards Hero their split-adjusted
        share of the pot. Reported equity is expected pot share, the multiway
        analogue of heads-up hot/cold equity.
        """
        if len(villain_ranges) < 2:
            raise ValueError("calculate_equity_multiway needs at least two villain ranges.")
        hero = parse_hero_cards(hero_cards)
        board = parse_board_cards(board_cards)
        parse_visible_cards(hero_cards, board_cards)

        labels: list[str] = []
        notations: list[str] = []
        for villain_range in villain_ranges:
            label, notation = _resolve_range(villain_range)
            if notation is None:
                return self._result(
                    hero,
                    board,
                    " | ".join([*labels, label]),
                    equity=None,
                    method="range_unavailable",
                    confidence=0.0,
                    notes=f"No range definition for villain range {label!r}.",
                )
            labels.append(label)
            notations.append(notation)

        dead = {str(card) for card in hero} | {str(card) for card in board}
        villain_combos = []
        for notation in notations:
            combos = [
                (hand, weight)
                for hand, weight in eval7.HandRange(notation).hands
                if str(hand[0]) not in dead and str(hand[1]) not in dead
            ]
            if not combos:
                return self._result(
                    hero,
                    board,
                    " | ".join(labels),
                    equity=None,
                    method="no_valid_combos",
                    confidence=0.0,
                    notes="A villain range is fully blocked by Hero/board cards.",
                )
            villain_combos.append(combos)

        equity, std_error = self._monte_carlo_multiway(hero, board, villain_combos)
        return self._result(
            hero,
            board,
            " | ".join(labels),
            equity=round(equity, 4),
            method="monte_carlo_multiway",
            confidence=MONTE_CARLO_CONFIDENCE,
            notes=(
                f"Seeded Monte-Carlo ({self.iterations:,} iterations) pot-share equity vs "
                f"{len(villain_combos)} villain ranges."
            ),
            std_error=std_error,
        )

    def _exact(self, hero: list[Card], board: list[Card], combos: list) -> float:
        hero_e = [eval7.Card(str(card)) for card in hero]
        board_e = [eval7.Card(str(card)) for card in board]
        full = [eval7.Card(f"{rank}{suit}") for rank in "23456789TJQKA" for suit in "cdhs"]
        dead = {str(card) for card in hero + board}
        wins = ties = total = 0.0
        for hand, weight in combos:
            villain = list(hand)
            used = dead | {str(villain[0]), str(villain[1])}
            deck = [card for card in full if str(card) not in used]
            need = 5 - len(board_e)
            for completion in _combinations(deck, need):
                hero_score = eval7.evaluate(hero_e + board_e + list(completion))
                villain_score = eval7.evaluate(villain + board_e + list(completion))
                if hero_score > villain_score:
                    wins += weight
                elif hero_score == villain_score:
                    ties += weight
                total += weight
        return (wins + ties / 2) / total

    def _monte_carlo(
        self, hero: list[Card], board: list[Card], combos: list
    ) -> tuple[float, float]:
        rng = random.Random(self.seed)
        hero_e = [eval7.Card(str(card)) for card in hero]
        board_e = [eval7.Card(str(card)) for card in board]
        full = [eval7.Card(f"{rank}{suit}") for rank in "23456789TJQKA" for suit in "cdhs"]
        dead = {str(card) for card in hero + board}
        need = 5 - len(board_e)
        stats = _RunningEquity()
        for _ in range(self.iterations):
            hand, weight = combos[rng.randrange(len(combos))]
            villain = list(hand)
            used = dead | {str(villain[0]), str(villain[1])}
            deck = [card for card in full if str(card) not in used]
            completion = rng.sample(deck, need)
            hero_score = eval7.evaluate(hero_e + board_e + completion)
            villain_score = eval7.evaluate(villain + board_e + completion)
            if hero_score > villain_score:
                share = 1.0
            elif hero_score == villain_score:
                share = 0.5
            else:
                share = 0.0
            stats.add(share, weight)
        return stats.mean(), stats.standard_error()

    def _monte_carlo_multiway(
        self, hero: list[Card], board: list[Card], villain_combos: list[list]
    ) -> tuple[float, float]:
        rng = random.Random(self.seed)
        hero_e = [eval7.Card(str(card)) for card in hero]
        board_e = [eval7.Card(str(card)) for card in board]
        full = [eval7.Card(f"{rank}{suit}") for rank in "23456789TJQKA" for suit in "cdhs"]
        dead = {str(card) for card in hero + board}
        need = 5 - len(board_e)
        stats = _RunningEquity()
        iterations = 0
        while iterations < self.iterations:
            # Rejection sampling: draw one combo per villain, retry the whole
            # deal if two villains claim the same card.
            used = set(dead)
            villains = []
            weight = 1.0
            clash = False
            for combos in villain_combos:
                hand, hand_weight = combos[rng.randrange(len(combos))]
                cards = {str(hand[0]), str(hand[1])}
                if cards & used:
                    clash = True
                    break
                used |= cards
                villains.append(list(hand))
                weight *= hand_weight
            if clash:
                continue
            iterations += 1
            deck = [card for card in full if str(card) not in used]
            completion = rng.sample(deck, need)
            hero_score = eval7.evaluate(hero_e + board_e + completion)
            villain_scores = [
                eval7.evaluate(villain + board_e + completion) for villain in villains
            ]
            best_villain = max(villain_scores)
            if hero_score > best_villain:
                share = 1.0
            elif hero_score == best_villain:
                share = 1.0 / (1 + villain_scores.count(best_villain))
            else:
                share = 0.0
            stats.add(share, weight)
        return stats.mean(), stats.standard_error()

    @staticmethod
    def _result(
        hero: list[Card],
        board: list[Card],
        label: str,
        *,
        equity: float | None,
        method: str,
        confidence: float,
        notes: str,
        std_error: float | None = None,
    ) -> EquityResult:
        return EquityResult(
            hero_hand=compact_cards(hero),
            board=spaced_cards(board),
            villain_range_label=label,
            equity=equity,
            method=method,
            confidence=confidence,
            notes=notes,
            std_error=std_error,
        )


class _RunningEquity:
    """Weighted running mean and standard error for Monte-Carlo pot shares.

    Uses the effective sample size (sum(w))^2 / sum(w^2) so weighted ranges do
    not overstate the estimate's precision.
    """

    def __init__(self) -> None:
        self._sum_w = 0.0
        self._sum_wx = 0.0
        self._sum_wx2 = 0.0
        self._sum_w2 = 0.0

    def add(self, share: float, weight: float) -> None:
        self._sum_w += weight
        self._sum_wx += weight * share
        self._sum_wx2 += weight * share * share
        self._sum_w2 += weight * weight

    def mean(self) -> float:
        return self._sum_wx / self._sum_w

    def standard_error(self) -> float:
        mean = self.mean()
        variance = max(0.0, self._sum_wx2 / self._sum_w - mean * mean)
        effective_n = (self._sum_w * self._sum_w) / self._sum_w2
        return (variance / effective_n) ** 0.5


def _resolve_range(villain_range: str) -> tuple[str, str | None]:
    """Resolve a villain range input to (label, eval7 notation).

    Known labels map to their predefined notation; anything else is accepted as
    raw eval7 range notation (e.g. "KK" or "22+,ATs+") if it parses, otherwise
    the notation is None and callers report range_unavailable.
    """
    label = normalize_range_label(villain_range)
    notation = range_notation(label)
    if notation is None:
        custom = (villain_range or "").strip()
        if custom and custom.lower() != "unknown":
            try:
                if eval7.HandRange(custom).hands:
                    return custom, custom
            except Exception:  # eval7 raises its own RangeStringError on bad syntax
                pass
    return label, notation


def _combinations(items: list, choose: int):
    """Yield combinations without importing itertools at call sites (readability)."""
    from itertools import combinations

    return combinations(items, choose)


def get_equity_calculator() -> EquityCalculator:
    """Return the best available equity calculator.

    Prefers the real eval7-backed engine; falls back to the labeled placeholder
    only if eval7 is not installed, so callers never have to branch on it.
    """
    if _HAS_EVAL7:
        return Eval7EquityCalculator()
    return PlaceholderEquityCalculator()


# TODO: Solver outputs (strategy frequencies/EV) should be stored/labeled separately
# from these hot/cold equity estimates when a solver integration is added.
