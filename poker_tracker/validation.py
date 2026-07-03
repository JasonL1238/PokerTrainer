from __future__ import annotations

from typing import Iterable

from poker_tracker.cards import CardParseError, parse_cards


RANKS = "23456789TJQKA"
SUITS = "cdhs"


class CardValidationError(ValueError):
    """Raised when a card string cannot be parsed as standard rank/suit cards."""


def normalize_cards(value: str, *, expected_counts: Iterable[int] | None = None) -> str:
    """Normalize card text like `AhQs` or `Ah Qs` into `Ah Qs`."""
    try:
        cards = parse_cards(value)
        allowed_counts = set(expected_counts or [])
        if allowed_counts and len(cards) not in allowed_counts:
            expected = ", ".join(str(count) for count in sorted(allowed_counts))
            raise CardValidationError(f"Expected {expected} cards, got {len(cards)}.")
        return " ".join(str(card) for card in cards)
    except CardParseError as exc:
        raise CardValidationError(str(exc)) from exc


def validate_tags(tags: list[str], allowed_tags: set[str]) -> list[str]:
    """Return unique uppercase tags after checking each tag is supported."""
    cleaned: list[str] = []
    for tag in tags:
        normalized = tag.strip().upper()
        if not normalized:
            continue
        if normalized not in allowed_tags:
            raise ValueError(f"Unsupported hand tag: {tag}")
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _normalize_card(token: str) -> str:
    if len(token) != 2:
        raise CardValidationError(f"Invalid card token: {token}")

    rank = token[0].upper()
    suit = token[1].lower()
    if rank not in RANKS or suit not in SUITS:
        raise CardValidationError(f"Invalid card token: {token}")
    return f"{rank}{suit}"
