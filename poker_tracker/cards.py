from __future__ import annotations

from dataclasses import dataclass


RANKS = "23456789TJQKA"
SUITS = "hdcs"


class CardParseError(ValueError):
    """Raised when card text is malformed or contains impossible duplicates."""


@dataclass(frozen=True, order=True)
class Card:
    """A normalized playing card such as `Ah` or `Td`."""

    rank: str
    suit: str

    def __post_init__(self) -> None:
        rank = self.rank.upper()
        suit = self.suit.lower()
        if rank not in RANKS:
            raise CardParseError(f"Invalid card rank: {self.rank}")
        if suit not in SUITS:
            raise CardParseError(f"Invalid card suit: {self.suit}")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "suit", suit)

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


def parse_card(value: str) -> Card:
    """Parse one card token like `Ah`, `Qs`, `Td`, or `9c`."""
    token = value.strip()
    if len(token) != 2:
        raise CardParseError(f"Invalid card token: {value}")
    return Card(rank=token[0], suit=token[1])


def parse_cards(value: str | list[str] | tuple[str, ...]) -> list[Card]:
    """Parse flexible card input into normalized cards while rejecting duplicates."""
    if isinstance(value, str):
        tokens = _tokenize_cards(value)
    else:
        tokens = list(value)

    cards = [parse_card(token) for token in tokens]
    _check_duplicates(cards)
    return cards


def parse_hero_cards(value: str | list[str] | tuple[str, ...]) -> list[Card]:
    """Parse exactly two hero hole cards."""
    cards = parse_cards(value)
    if len(cards) != 2:
        raise CardParseError(f"Hero cards must contain exactly 2 cards, got {len(cards)}.")
    return cards


def parse_board_cards(value: str | list[str] | tuple[str, ...]) -> list[Card]:
    """Parse zero to five board cards."""
    cards = parse_cards(value)
    if len(cards) > 5:
        raise CardParseError(f"Board can contain at most 5 cards, got {len(cards)}.")
    return cards


def parse_visible_cards(
    hero_cards: str | list[str] | tuple[str, ...],
    board_cards: str | list[str] | tuple[str, ...] = "",
) -> list[Card]:
    """Parse hero and board cards together and reject duplicates across both groups."""
    hero = parse_hero_cards(hero_cards)
    board = parse_board_cards(board_cards)
    combined = [*hero, *board]
    _check_duplicates(combined)
    return combined


def normalize_card_list(cards: list[Card]) -> list[str]:
    """Return cards as normalized strings, e.g. `["Ah", "Qs"]`."""
    return [str(card) for card in cards]


def compact_cards(cards: list[Card]) -> str:
    """Return cards in compact form, e.g. `AhQs`."""
    return "".join(normalize_card_list(cards))


def spaced_cards(cards: list[Card]) -> str:
    """Return cards in readable spaced form, e.g. `Ah Qs`."""
    return " ".join(normalize_card_list(cards))


def _tokenize_cards(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    cleaned = text.replace(",", " ").replace("-", " ").replace("/", " ")
    # Accept the common "10" spelling for tens ("10h" -> "Th"). "10" can only
    # ever appear as a ten rank, so a plain substring replace is safe.
    cleaned = cleaned.replace("10", "T")
    tokens = cleaned.split()
    if len(tokens) == 1 and len(tokens[0]) > 2:
        compact = tokens[0]
        if len(compact) % 2 != 0:
            raise CardParseError(f"Invalid compact card string: {value}")
        return [compact[index : index + 2] for index in range(0, len(compact), 2)]
    return tokens


def _check_duplicates(cards: list[Card]) -> None:
    labels = normalize_card_list(cards)
    if len(labels) != len(set(labels)):
        raise CardParseError("Duplicate cards are not allowed.")
