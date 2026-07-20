import eval7
import pytest

from poker_tracker.preflop_ranges import (
    POSITIONS,
    SCENARIOS,
    available_ranges,
    get_preflop_range,
    range_percent,
)
from poker_tracker.ranges import range_notation


def test_every_defined_range_parses_in_eval7_with_combos() -> None:
    for chart in available_ranges():
        combos = eval7.HandRange(chart.notation).hands
        assert len(combos) > 0, chart


def test_rfi_percentages_are_in_expected_windows() -> None:
    assert 0.13 <= range_percent(get_preflop_range("UTG", "rfi").notation) <= 0.19
    assert 0.35 <= range_percent(get_preflop_range("BTN", "rfi").notation) <= 0.50


def test_rfi_width_is_monotonic_from_utg_to_btn() -> None:
    widths = [
        range_percent(get_preflop_range(position, "rfi").notation)
        for position in ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN"]
    ]
    assert widths == sorted(widths)
    assert widths[-1] > widths[0]


def test_bb_defend_is_wider_than_bb_3bet() -> None:
    call = range_percent(get_preflop_range("BB", "vs_open_call").notation)
    three_bet = range_percent(get_preflop_range("BB", "vs_open_3bet").notation)
    assert call > three_bet


def test_unknown_position_scenario_and_undefined_combo_raise() -> None:
    with pytest.raises(ValueError):
        get_preflop_range("MP9", "rfi")
    with pytest.raises(ValueError):
        get_preflop_range("BTN", "limp")
    with pytest.raises(ValueError):
        get_preflop_range("BB", "rfi")  # the big blind never raises first in


def test_positions_and_scenarios_constants_cover_all_charts() -> None:
    for chart in available_ranges():
        assert chart.position in POSITIONS
        assert chart.scenario in SCENARIOS


def test_standard_label_is_a_genuine_default_width() -> None:
    notation = range_notation("standard")
    assert notation is not None
    assert 0.15 <= range_percent(notation) <= 0.25
