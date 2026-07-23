import pytest

from poker_tracker.math.cards import (
    CardParseError,
    compact_cards,
    normalize_card_list,
    parse_board_cards,
    parse_card,
    parse_hero_cards,
    parse_visible_cards,
    spaced_cards,
)
from poker_tracker.math.ev import (
    bluff_ev,
    call_ev,
    semi_bluff_break_even_fold_frequency,
    semi_bluff_ev,
)
from poker_tracker.math.pot_odds import (
    break_even_bluff_frequency,
    format_percentage,
    minimum_defense_frequency,
    pot_odds_offered_by_bet,
    rake_amount,
    required_equity_to_call,
    required_equity_to_call_after_rake,
    stack_to_pot_ratio,
)


def test_valid_card_parsing_and_normalization() -> None:
    card = parse_card("ah")
    hero = parse_hero_cards("AhQs")

    assert str(card) == "Ah"
    assert normalize_card_list(hero) == ["Ah", "Qs"]
    assert compact_cards(hero) == "AhQs"
    assert spaced_cards(hero) == "Ah Qs"
    assert normalize_card_list(parse_hero_cards(["td", "9C"])) == ["Td", "9c"]
    assert normalize_card_list(parse_board_cards("Ah,Kd/Qs-Jc Td")) == ["Ah", "Kd", "Qs", "Jc", "Td"]


def test_invalid_card_parsing() -> None:
    with pytest.raises(CardParseError):
        parse_card("Ax")
    with pytest.raises(CardParseError):
        parse_hero_cards("Ah")
    with pytest.raises(CardParseError):
        parse_board_cards("AhKdQ")


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
    assert minimum_defense_frequency(50, 100) == pytest.approx(2 / 3)
    assert pot_odds_offered_by_bet(50, 100) == pytest.approx(0.25)
    assert stack_to_pot_ratio(200, 50) == pytest.approx(4.0)
    # MDF and the break-even bluff frequency (alpha) are complements.
    assert minimum_defense_frequency(75, 100) + break_even_bluff_frequency(75, 100) == pytest.approx(1.0)
    assert format_percentage(0.3333) == "33.3%"


def test_pot_odds_edge_cases() -> None:
    with pytest.raises(ValueError):
        required_equity_to_call(0, 100)
    with pytest.raises(ValueError):
        required_equity_to_call(10, -1)
    with pytest.raises(ValueError):
        break_even_bluff_frequency(10, 0)
    with pytest.raises(ValueError):
        format_percentage(1.2)
    assert format_percentage(0) == "0.0%"
    assert format_percentage(1) == "100.0%"


def test_ev_helpers() -> None:
    # At exactly the required equity (call/(pot+call)), calling is break-even.
    assert call_ev(required_equity_to_call(25, 75), 75, 25) == pytest.approx(0.0)
    assert call_ev(0.4, 75, 25) == pytest.approx(0.4 * 75 - 0.6 * 25)  # +15
    assert call_ev(0.1, 75, 25) == pytest.approx(0.1 * 75 - 0.9 * 25)  # -15
    # At exactly the break-even fold frequency, a pure bluff is break-even.
    assert bluff_ev(break_even_bluff_frequency(50, 100), 100, 50) == pytest.approx(0.0)
    assert bluff_ev(0.5, 100, 50) == pytest.approx(25)
    assert semi_bluff_ev(0.4, 0.3, 100, 50) == pytest.approx(46)
    # With zero equity when called, a semi-bluff degenerates to a pure bluff.
    assert semi_bluff_ev(0.5, 0.0, 100, 50) == pytest.approx(bluff_ev(0.5, 100, 50))
    assert semi_bluff_break_even_fold_frequency(0.0, 100, 50) == pytest.approx(1 / 3)
    needed = semi_bluff_break_even_fold_frequency(0.2, 100, 50)
    assert semi_bluff_ev(needed, 0.2, 100, 50) == pytest.approx(0.0)


def test_rake_adjusted_call_math_is_internally_consistent() -> None:
    # Calling 25 into 75 creates a 100 BB pot. Five percent rake raises the
    # break-even threshold from 25% to 25/95.
    assert rake_amount(100, 0.05, 10) == pytest.approx(5.0)
    threshold = required_equity_to_call_after_rake(25, 75, 0.05, 10)
    assert threshold == pytest.approx(25 / 95)
    assert call_ev(threshold, 75, 25, rake_rate=0.05, rake_cap=10) == pytest.approx(0.0)


def test_rake_cap_and_validation() -> None:
    assert rake_amount(200, 0.05, 3) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        rake_amount(100, 1.01)
    with pytest.raises(ValueError):
        rake_amount(100, 0.05, -1)


def test_ten_rank_normalization() -> None:
    assert str(parse_card("Th")) == "Th"
    assert normalize_card_list(parse_hero_cards("10h 10s")) == ["Th", "Ts"]
    assert normalize_card_list(parse_board_cards("10h9c8d")) == ["Th", "9c", "8d"]


def test_ev_edge_cases() -> None:
    with pytest.raises(ValueError):
        call_ev(1.2, 100, 10)
    with pytest.raises(ValueError):
        call_ev(0.5, 0, 10)
    with pytest.raises(ValueError):
        bluff_ev(0.5, -1, 10)
    with pytest.raises(ValueError):
        semi_bluff_ev(-0.1, 0.2, 100, 10)
    with pytest.raises(ValueError):
        semi_bluff_ev(0.1, 1.1, 100, 10)
