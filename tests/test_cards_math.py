import pytest

from poker_tracker.cards import (
    CardParseError,
    compact_cards,
    normalize_card_list,
    parse_board_cards,
    parse_card,
    parse_hero_cards,
    parse_visible_cards,
    spaced_cards,
)
from poker_tracker.ev import bluff_ev, call_ev, semi_bluff_ev
from poker_tracker.pot_odds import (
    break_even_bluff_frequency,
    format_percentage,
    required_equity_to_call,
    value_bet_call_threshold,
)


def test_valid_card_parsing_and_normalization() -> None:
    card = parse_card("ah")
    hero = parse_hero_cards("AhQs")

    assert str(card) == "Ah"
    assert normalize_card_list(hero) == ["Ah", "Qs"]
    assert compact_cards(hero) == "AhQs"
    assert spaced_cards(hero) == "Ah Qs"


def test_invalid_card_parsing() -> None:
    with pytest.raises(CardParseError):
        parse_card("Ax")
    with pytest.raises(CardParseError):
        parse_hero_cards("Ah")


def test_duplicate_card_detection() -> None:
    with pytest.raises(CardParseError):
        parse_hero_cards("Ah Ah")
    with pytest.raises(CardParseError):
        parse_visible_cards("Ah Qs", "Ah 7d 2c")


def test_board_length_validation() -> None:
    assert parse_board_cards("") == []
    assert len(parse_board_cards("Ah Kd Qs Jc Td")) == 5
    with pytest.raises(CardParseError):
        parse_board_cards("Ah Kd Qs Jc Td 9h")


def test_pot_odds_calculations() -> None:
    assert required_equity_to_call(25, 75) == pytest.approx(0.25)
    assert break_even_bluff_frequency(50, 100) == pytest.approx(1 / 3)
    assert value_bet_call_threshold(50, 100) == pytest.approx(1 / 3)
    assert format_percentage(0.3333) == "33.3%"


def test_pot_odds_edge_cases() -> None:
    with pytest.raises(ValueError):
        required_equity_to_call(0, 100)
    with pytest.raises(ValueError):
        break_even_bluff_frequency(10, 0)
    with pytest.raises(ValueError):
        format_percentage(1.2)


def test_ev_helpers() -> None:
    assert call_ev(0.25, 75, 25) == pytest.approx(6.25)
    assert bluff_ev(0.5, 100, 50) == pytest.approx(25)
    assert semi_bluff_ev(0.4, 0.3, 100, 50) == pytest.approx(46)


def test_ev_edge_cases() -> None:
    with pytest.raises(ValueError):
        call_ev(1.2, 100, 10)
    with pytest.raises(ValueError):
        bluff_ev(0.5, -1, 10)
