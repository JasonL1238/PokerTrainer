"""Property and fuzz tests for card uniqueness and normalization (PLAN Phase 14).

The example-based suite in tests/test_cards_math.py checks the card spellings
somebody thought of. These generate the ones nobody did: every casing of every
card, every separator, the "10" spelling for tens, arbitrary unicode, and
arbitrary multisets drawn from the deck.

Four invariants hold for every input:

* Parsing is total. `parse_cards` either returns cards or raises
  `CardParseError`. It never returns a card that is not one of the 52, and it
  never coerces malformed text into a card that the text did not denote.
* Normalization is idempotent. Re-parsing what the module printed gives the
  same cards, and `normalize_cards` reaches a fixed point in one step.
* Normalization is injective. Two inputs land on the same card exactly when
  they denote the same rank and suit; nothing else collides.
* Uniqueness is joint. A card may appear once across hero and board together,
  not once in each.

The card constructor is generated against directly because it is the module's
public value type and the only place the rank/suit alphabet is enforced.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from poker_tracker.math.cards import (
    RANKS,
    SUITS,
    Card,
    CardParseError,
    compact_cards,
    normalize_card_list,
    parse_board_cards,
    parse_card,
    parse_cards,
    parse_hero_cards,
    parse_visible_cards,
    spaced_cards,
)
from poker_tracker.persistence.validation import CardValidationError, normalize_cards

DECK = [f"{rank}{suit}" for rank in RANKS for suit in SUITS]
SEPARATORS = (" ", "  ", ",", ", ", "-", "/", "\t", "\n")

SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)

# Text drawn from the symbols a real caller mixes up -- rank/suit letters in
# both cases, the digits of the "10" spelling, and every separator the
# tokenizer honours. Arbitrary unicode is generated separately; this alphabet is
# what actually reaches the interesting branches.
POKERISH_TEXT = st.text(alphabet="23456789TJQKAhdcstjqka10 ,-/", max_size=14)
ARBITRARY_TEXT = st.text(alphabet=st.characters(codec="utf-8"), max_size=12)
HOSTILE_TEXT = st.one_of(POKERISH_TEXT, ARBITRARY_TEXT)


def _is_one_of_the_fifty_two(text: str) -> bool:
    return len(text) == 2 and text[0] in RANKS and text[1] in SUITS


@st.composite
def spelled_card(draw: st.DrawFn) -> tuple[str, str]:
    """A single card plus one arbitrary spelling of it (casing, "10" for T)."""
    label = draw(st.sampled_from(DECK))
    rank, suit = label[0], label[1]
    spelled_rank = draw(st.sampled_from([rank, rank.lower()]))
    if rank == "T":
        spelled_rank = draw(st.sampled_from([rank, rank.lower(), "10"]))
    spelled_suit = draw(st.sampled_from([suit, suit.upper()]))
    return label, f"{spelled_rank}{spelled_suit}"


@st.composite
def deck_subset(draw: st.DrawFn, *, max_size: int = 7) -> list[str]:
    return draw(st.lists(st.sampled_from(DECK), max_size=max_size, unique=True))


# --- totality ------------------------------------------------------------------------


@SETTINGS
@given(HOSTILE_TEXT)
def test_parsing_is_total_and_never_invents_a_card(text: str) -> None:
    """Either cards or a CardParseError -- never a silently wrong card.

    Anything that comes back must be one of the 52, and must re-parse to
    itself, so no input can produce a value the module cannot read back.
    """
    try:
        cards = parse_cards(text)
    except CardParseError:
        return

    for card in cards:
        label = str(card)
        assert _is_one_of_the_fifty_two(label), (text, label)
        assert str(parse_card(label)) == label, (text, label)


@SETTINGS
@given(HOSTILE_TEXT)
def test_a_parsed_group_never_contains_a_duplicate(text: str) -> None:
    try:
        cards = parse_cards(text)
    except CardParseError:
        return
    labels = normalize_card_list(cards)
    assert len(set(labels)) == len(labels), (text, labels)


@SETTINGS
@given(HOSTILE_TEXT)
def test_hero_and_board_helpers_enforce_their_own_counts(text: str) -> None:
    """The count rules are absolute: no input reaches a caller with the wrong shape."""
    try:
        hero = parse_hero_cards(text)
    except CardParseError:
        pass
    else:
        assert len(hero) == 2, (text, hero)

    try:
        board = parse_board_cards(text)
    except CardParseError:
        pass
    else:
        assert len(board) <= 5, (text, board)


# --- normalization -------------------------------------------------------------------


@SETTINGS
@given(HOSTILE_TEXT)
def test_normalization_reaches_a_fixed_point_in_one_step(text: str) -> None:
    """normalize_cards is idempotent: normalizing its own output changes nothing."""
    try:
        once = normalize_cards(text)
    except CardValidationError:
        return
    assert normalize_cards(once) == once, (text, once)


@SETTINGS
@given(HOSTILE_TEXT)
def test_every_rendering_of_a_parsed_group_re_reads_as_that_group(text: str) -> None:
    """compact and spaced renderings are both readable and mean the same thing."""
    try:
        cards = parse_cards(text)
    except CardParseError:
        return
    labels = normalize_card_list(cards)
    assert normalize_card_list(parse_cards(spaced_cards(cards))) == labels, (text,)
    assert normalize_card_list(parse_cards(compact_cards(cards))) == labels, (text,)


@SETTINGS
@given(spelled_card())
def test_every_spelling_of_a_card_normalizes_to_that_card(spelling: tuple[str, str]) -> None:
    label, text = spelling
    assert normalize_card_list(parse_cards(text)) == [label]
    assert normalize_cards(text) == label


def test_the_ten_spelling_is_a_group_level_convenience_only() -> None:
    """`parse_card` reads one token and does not know the "10" spelling.

    `_tokenize_cards` rewrites "10" to "T" before it reaches `parse_card`, so
    the flexible spelling exists on the group entrypoints only. Recorded here
    because it is an asymmetry a caller can hit: a list element is not
    tokenized, so `["10h", "10s"]` is refused while `"10h 10s"` is accepted.
    """
    assert normalize_card_list(parse_cards("10h 10s")) == ["Th", "Ts"]
    with pytest.raises(CardParseError):
        parse_card("10h")
    with pytest.raises(CardParseError):
        parse_cards(["10h", "10s"])


@SETTINGS
@given(deck_subset(), st.sampled_from(SEPARATORS), st.booleans())
def test_separator_case_and_ten_spelling_all_denote_the_same_cards(
    cards: list[str], separator: str, upper: bool
) -> None:
    """Every way of writing the same hand reads back as the same hand."""
    written = separator.join(cards)
    if upper:
        written = written.upper()
    assert normalize_card_list(parse_cards(written)) == cards

    spelled_tens = separator.join(card.replace("T", "10") for card in cards)
    assert normalize_card_list(parse_cards(spelled_tens)) == cards

    compact = "".join(cards)
    assert normalize_card_list(parse_cards(compact)) == cards


# --- injectivity ---------------------------------------------------------------------


@SETTINGS
@given(
    st.text(alphabet="23456789TJQKAhdcstjqka", min_size=2, max_size=2),
    st.text(alphabet="23456789TJQKAhdcstjqka", min_size=2, max_size=2),
)
def test_two_tokens_collide_only_when_they_denote_the_same_card(first: str, second: str) -> None:
    """Nothing but case folding may merge two distinct card tokens."""
    try:
        left, right = parse_card(first), parse_card(second)
    except CardParseError:
        return
    denote_the_same = (first[0].upper(), first[1].lower()) == (
        second[0].upper(),
        second[1].lower(),
    )
    assert (str(left) == str(right)) is denote_the_same, (first, second)


def test_the_whole_deck_normalizes_to_fifty_two_distinct_cards() -> None:
    parsed = parse_cards(DECK)
    assert len(parsed) == 52
    assert len({str(card) for card in parsed}) == 52
    assert sorted(normalize_card_list(parsed)) == sorted(DECK)


# --- uniqueness ----------------------------------------------------------------------


@SETTINGS
@given(deck_subset(max_size=7))
def test_a_dealt_subset_of_the_deck_never_trips_the_duplicate_guard(
    cards: list[str],
) -> None:
    assert normalize_card_list(parse_cards(cards)) == cards


@SETTINGS
@given(deck_subset(max_size=6), st.data())
def test_a_repeat_anywhere_in_the_group_is_rejected(cards: list[str], data: st.DrawFn) -> None:
    """A duplicate is caught wherever it sits and however it is spelled."""
    assume(cards)
    repeated = data.draw(st.sampled_from(cards))
    position = data.draw(st.integers(min_value=0, max_value=len(cards)))
    spelling = data.draw(st.sampled_from([repeated, repeated.lower(), repeated.upper()]))
    polluted = [*cards[:position], spelling, *cards[position:]]
    separator = data.draw(st.sampled_from(SEPARATORS))
    with pytest.raises(CardParseError):
        parse_cards(separator.join(polluted))


@SETTINGS
@given(deck_subset(max_size=7))
def test_hero_and_board_uniqueness_is_checked_jointly(cards: list[str]) -> None:
    """Two cards that are each fine on their own side still clash across sides."""
    assume(len(cards) >= 3)
    hero, board = cards[:2], cards[2:]
    assert len(parse_visible_cards(" ".join(hero), " ".join(board))) == len(cards)

    # Move one board card into Hero's hand: both groups stay internally clean,
    # and only the joint check can see the clash.
    with pytest.raises(CardParseError):
        parse_visible_cards(f"{hero[0]} {board[0]}", " ".join(board))


# --- rejection rather than coercion --------------------------------------------------


@pytest.mark.parametrize(
    ("rank", "suit"),
    [
        ("", ""),  # `"" in RANKS` is True: the empty card was built and printed as ""
        ("", "h"),
        ("A", ""),
        ("TJ", "h"),  # `"TJ" in RANKS` is True: a two-rank card printed as "TJh"
        ("QK", "s"),
        ("A", "hd"),  # `"hd" in SUITS` is True: a two-suit card printed as "Ahd"
        ("A", "dc"),
        ("23456789TJQKA", "h"),
        ("A", "hdcs"),
    ],
)
def test_card_rejects_a_rank_or_suit_that_is_not_exactly_one_symbol(rank: str, suit: str) -> None:
    """Rank/suit validation is membership, not substring containment.

    `RANKS` and `SUITS` are strings, so `x in RANKS` is a substring test. Every
    case here is a substring of the alphabet without being a symbol of it, and
    each one used to build a Card that no parser could read back.
    """
    with pytest.raises(CardParseError):
        Card(rank=rank, suit=suit)


@SETTINGS
@given(
    st.text(alphabet="23456789TJQKA", max_size=4),
    st.text(alphabet="hdcs", max_size=4),
)
def test_only_single_symbol_rank_and_suit_build_a_card(rank: str, suit: str) -> None:
    """Generated over the alphabet itself, so every rejection is a length rejection."""
    try:
        card = Card(rank=rank, suit=suit)
    except CardParseError:
        assert len(rank) != 1 or len(suit) != 1, (rank, suit)
        return
    assert len(rank) == 1 and len(suit) == 1, (rank, suit)
    assert _is_one_of_the_fifty_two(str(card))


@SETTINGS
@given(st.text(alphabet="23456789TJQKAhdcs", max_size=11))
def test_a_compact_string_of_odd_length_is_refused_not_truncated(text: str) -> None:
    """An odd-length compact run is malformed; dropping the tail would be a wrong read."""
    assume(len(text) > 2 and len(text) % 2 == 1)
    with pytest.raises(CardParseError):
        parse_cards(text)


@pytest.mark.parametrize(
    "text",
    [
        "A",
        "Ahh",
        "A h",
        "Ah Kd Qs Jc Td 9h 8c",  # seven cards is more than any board
        "Ah\x00",
        "\x00h",
        "AhKdQ",
        "1h 0s",
        "T0h",
        "100h",
        "Ah10",
    ],
)
def test_hostile_tokens_are_refused_rather_than_coerced(text: str) -> None:
    with pytest.raises(CardParseError):
        parse_board_cards(text)
