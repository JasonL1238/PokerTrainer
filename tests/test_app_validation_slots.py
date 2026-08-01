"""Slot-contention guard for Import validation action rows."""
from __future__ import annotations

from app import contested_timeline_slots
from poker_tracker.persistence.models import Action


def _action(street: str, index: int | None, name: str = "Seat2") -> Action:
    return Action(
        hand_id=1,
        player_key=f"seat:{name}",
        street=street,
        action_index=index,
        player_name=name,
        position="UTG+1",
        action_type="check",
    )


def test_moving_a_row_to_another_street_marks_both_slots_contested() -> None:
    """B6 round 6 F2: DB action indexes are per-street, so a street correction
    lands the row on a real slot there and it inherits that line's frame and
    stack figure — which the frame then appears to confirm."""
    actions = [
        _action("turn", 1, "Seat2"),   # moved here from the flop
        _action("turn", 1, "Seat4"),   # the turn's real first action
        _action("river", 1, "Seat2"),
    ]
    assert contested_timeline_slots(actions) == {("turn", 1)}


def test_clean_ordering_contests_nothing() -> None:
    actions = [
        _action("preflop", 1),
        _action("preflop", 2),
        _action("flop", 1),
        _action("flop", 2),
    ]
    assert contested_timeline_slots(actions) == set()


def test_rows_without_an_index_are_ignored() -> None:
    assert contested_timeline_slots([_action("flop", None), _action("flop", None)]) == set()
