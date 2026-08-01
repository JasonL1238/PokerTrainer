"""Which reconstructed line a saved row is explained by.

These pin the app-side attribution that round 7 introduced. Round 8 proved it
had no test at all: four separate mutations — including reverting the whole
repair by ignoring stored provenance — left the entire suite green.
"""
from __future__ import annotations

import pytest

from app import _cv_issues_for_db_action
from poker_tracker.persistence.models import Action
from poker_tracker.ui.reconstruction_review import ValidationFrameContext

FRAME_A = "/frames/f0.jpg"
FRAME_B = "/frames/f1.jpg"


def _states() -> list[dict]:
    return [
        {
            "state_index": 0,
            "time_s": 0.0,
            "image": FRAME_A,
            "board_cards": [],
            "dealt_in": [1, 2],
            "stacks": {"1": 200.0, "2": 300.0},
            "bets": {},
            "bets_unknown": {},
            "stacks_unknown": {},
            "unmeasured_transitions": [],
            "coverage_gap": False,
        },
        {
            "state_index": 1,
            "time_s": 5.0,
            "image": FRAME_B,
            "board_cards": [],
            "dealt_in": [1, 2],
            "stacks": {"1": 180.0, "2": 280.0},
            "bets": {},
            "bets_unknown": {},
            "stacks_unknown": {},
            "unmeasured_transitions": [],
            "coverage_gap": False,
        },
    ]


def _hand() -> dict:
    return {
        "hand_number": 1,
        "t_start": 0.0,
        "warnings": [],
        "players": [
            {"seat": 1, "player_name": "Seat1", "position": "UTG", "starting_stack": 200.0},
            {"seat": 2, "player_name": "Seat2", "position": "BB", "starting_stack": 300.0},
        ],
        "actions": [
            # Two lines on ONE frame, by different seats — the shape that makes
            # an actor correction dangerous.
            {
                "street": "preflop", "action_index": 1, "seat": 1,
                "player_name": "Seat1", "position": "UTG", "action_type": "call",
                "amount": 10.0, "source_image": FRAME_A, "derivation": "stack_delta",
            },
            {
                "street": "preflop", "action_index": 2, "seat": 2,
                "player_name": "Seat2", "position": "BB", "action_type": "raise",
                "amount": 3.0, "source_image": FRAME_A, "derivation": "stack_delta",
            },
            {
                "street": "flop", "action_index": 1, "seat": 1,
                "player_name": "Seat1", "position": "UTG", "action_type": "call",
                "amount": 20.0, "source_image": FRAME_B, "derivation": "stack_delta",
            },
        ],
    }


def _context() -> ValidationFrameContext:
    return ValidationFrameContext(
        job_id=1,
        hand_number=1,
        timeline_hand=_hand(),
        states=_states(),
        reviews_by_image={},
        cursor_key="c",
        pending_hand_key="p",
        recording_start_s=0.0,
    )


def _row(**overrides) -> Action:
    base = dict(
        hand_id=1,
        street="preflop",
        action_index=1,
        player_name="Seat1",
        position="UTG",
        action_type="call",
        amount=None,
        source_image=FRAME_A,
    )
    base.update(overrides)
    return Action(**base)


def _texts(issues) -> str:
    return " ".join(issue.detail for issue in issues)


def test_stored_provenance_is_used_instead_of_the_slot() -> None:
    """Reverting the repair (ignoring source_image) must fail a test."""
    issues = _cv_issues_for_db_action(_row(), _context())
    assert "10 BB" in _texts(issues)


def test_a_moved_row_does_not_borrow_the_new_slots_figures() -> None:
    """Round 7's headline case: a street correction lands on another line."""
    moved = _row(street="flop", action_index=1)
    issues = _cv_issues_for_db_action(moved, _context())
    texts = _texts(issues)
    assert "20 BB" not in texts, "quoted the flop line's amount"
    assert issues, "moving the row silently cleared its warning"


def test_an_actor_correction_does_not_borrow_the_other_seats_figures() -> None:
    """Round 8: the seat key follows the row's CURRENT actor, so on a frame
    carrying two seats' lines it landed on the wrong one."""
    reassigned = _row(player_name="Seat2", position="BB")
    texts = _texts(_cv_issues_for_db_action(reassigned, _context()))
    assert "3 BB" not in texts, "quoted the other seat's raise"


def test_a_slot_match_disagreeing_with_stored_provenance_is_rejected() -> None:
    """Inverting the disagreement guard must fail a test."""
    # Row claims the flop slot but records frame A as its origin.
    conflicted = _row(street="flop", action_index=1, source_image=FRAME_A)
    texts = _texts(_cv_issues_for_db_action(conflicted, _context()))
    assert "20 BB" not in texts


@pytest.mark.parametrize(
    "overrides",
    [
        {"street": "flop", "action_index": 1},
        {"action_type": "bet"},
        {"player_name": "Seat2", "position": "BB"},
    ],
)
def test_an_edited_row_is_labelled_rather_than_silently_trusted(overrides) -> None:
    """Removing the 'Edited line' note must fail a test."""
    issues = _cv_issues_for_db_action(_row(**overrides), _context())
    assert issues, overrides
    # Either the row is still attributed and flagged as edited, or it fell
    # back to the unattributable path, which says so in the message itself.
    assert any(
        issue.kind == "Edited line" or "edited or added" in issue.detail
        for issue in issues
    ), overrides


def test_an_edit_never_silently_clears_a_live_warning() -> None:
    """The governing principle: a worse row must not read as resolved."""
    before = _cv_issues_for_db_action(_row(), _context())
    assert before
    for overrides in (
        {"street": "turn", "action_index": 1},
        {"action_index": 9},
        {"action_type": "bet"},
    ):
        after = _cv_issues_for_db_action(_row(**overrides), _context())
        assert after, f"warnings vanished after {overrides}"


def test_a_row_without_provenance_still_falls_back_to_its_slot() -> None:
    legacy = _row(source_image=None)
    assert _cv_issues_for_db_action(legacy, _context())


def test_provenance_beats_the_slot_when_they_disagree() -> None:
    """The decisive case: the row's slot and its recorded frame point at
    different lines. Ignoring provenance entirely must fail here — round 8
    showed that mutation survived the whole suite."""
    # Slot says preflop#1 (the 10 BB call); provenance says frame B, which
    # carries exactly one line (the 20 BB flop call).
    row = _row(street="preflop", action_index=1, source_image=FRAME_B)
    texts = _texts(_cv_issues_for_db_action(row, _context()))
    assert "20 BB" in texts, "did not follow the recorded source frame"
    assert "10 BB" not in texts, "followed the slot instead of provenance"
