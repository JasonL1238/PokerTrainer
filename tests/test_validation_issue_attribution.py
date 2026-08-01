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


def test_the_saved_type_reaches_the_money_gates_through_the_app() -> None:
    """A9 round 9 F4: the type/street threading was pinned only by calling
    cv_issues_for_timeline_action directly with the kwargs — dropping them at
    the single call site left 537/537 retyped folds silent with a green suite."""
    # A reconstructed call retyped to a fold must stop demanding an amount.
    as_fold = _row(action_type="fold")
    assert not [
        issue
        for issue in _cv_issues_for_db_action(as_fold, _context())
        if issue.kind == "Amount unknown"
    ]
    # A reconstructed fold retyped to a bet, amount empty, must be flagged.
    fold_hand = _hand()
    fold_hand["actions"][0]["action_type"] = "fold"
    fold_hand["actions"][0]["amount"] = None
    context = ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=fold_hand, states=_states(),
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    assert [
        issue
        for issue in _cv_issues_for_db_action(_row(action_type="bet"), context)
        if issue.kind == "Amount unknown"
    ]


def test_the_saved_street_reaches_the_hedge_through_the_app() -> None:
    """Same for db_street: dropping it at the call site must fail here. The
    hedge is the only thing telling the operator the frame below describes a
    different street's line."""
    moved = _row(street="turn", action_index=1)
    issues = _cv_issues_for_db_action(moved, _context())
    hedge = next(
        (issue for issue in issues if issue.kind == "Moved off its source street"),
        None,
    )
    assert hedge is not None, "a moved row was not hedged"
    assert "reconstructed on the preflop" in hedge.detail
    assert "now saved on the turn" in hedge.detail


def test_a_detached_row_never_borrows_a_neighbours_derivation() -> None:
    """A9 round 9 F3: 49 frames carry lines with mixed derivations. After an
    actor correction the frame+seat key finds the NEW seat's line, so
    borrowing its derivation accused an observed row of having been inferred
    and told the operator to delete it."""
    hand = _hand()
    # Seat 1's line on frame A was inferred; seat 2's was observed.
    hand["actions"][0]["derivation"] = "inferred_round_complete"
    hand["actions"][1]["derivation"] = "action_pill"
    states = _states()
    states[0]["dealt_in"] = [2]        # seat 1 holds no cards on this frame
    context = ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=hand, states=states,
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    # Seat 2's OBSERVED row, actor corrected to Seat1 — whose line on the same
    # frame is inferred. It must not inherit that and be accused.
    detached = Action(
        hand_id=1,
        street="preflop",
        action_index=2,
        player_name="Seat1",
        position="UTG",
        action_type="raise",
        amount=None,
        source_image=FRAME_A,
    )
    issues = _cv_issues_for_db_action(detached, context)
    # The row may still be questioned on FRAME evidence — the frame really
    # does show its seat holding nothing — but never with a borrowed claim
    # about how the line was derived.
    assert not any(
        "it was inferred because the betting round completed" in issue.detail
        for issue in issues
    ), "an observed row was described as inferred using a neighbour's derivation"
    assert not any("was not observed" in issue.detail for issue in issues)


def test_the_backfill_only_runs_for_the_job_the_hand_came_from() -> None:
    """A9 round 9 F1: several jobs can share a video and resolve to the same
    DB hand, so opening another job's validation stamped ITS frames."""
    from poker_tracker.ui.reconstruction_review import job_id_from_hand_notes

    notes = "CV draft from YOLO card timeline. timeline=/x/job_1_timeline.json"
    assert job_id_from_hand_notes(notes) == 1
    assert job_id_from_hand_notes("manual hand") is None


def test_an_order_edit_does_not_erase_a_phantom_accusation() -> None:
    """B10 round 10 F1: one keystroke in Order cleared the accusation and the
    row went completely silent. The evidence is (frame, seat), and neither
    changes when the order does — the fourth consecutive round to find a
    silent-clear in this family."""
    hand = _hand()
    hand["actions"][0]["derivation"] = "inferred_round_complete"
    states = _states()
    states[0]["dealt_in"] = [2]        # seat 1 holds no cards on its own frame
    context = ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=hand, states=states,
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    before = _cv_issues_for_db_action(_row(action_type="call"), context)
    assert any(
        issue.kind == "Action may not belong to this hand" for issue in before
    )
    for overrides in (
        {"action_type": "call", "action_index": 9},                   # order
        {"action_type": "call", "street": "turn", "action_index": 1},  # street
        {"action_type": "bet"},                                        # type
    ):
        after = _cv_issues_for_db_action(_row(**overrides), context)
        assert any(
            issue.kind == "Action may not belong to this hand" for issue in after
        ), f"accusation vanished after {overrides}"


def test_a_fold_is_never_accused_of_not_belonging_after_an_edit() -> None:
    """A folding seat loses its cards on its own frame, so absence there is the
    expected observation, not evidence the action never happened."""
    hand = _hand()
    hand["actions"][0]["derivation"] = "inferred_round_complete"
    states = _states()
    states[0]["dealt_in"] = [2]
    context = ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=hand, states=states,
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    issues = _cv_issues_for_db_action(
        _row(action_type="fold", action_index=9), context
    )
    assert not any(
        issue.kind == "Action may not belong to this hand" for issue in issues
    )


def test_a_foreign_jobs_frames_explain_nothing() -> None:
    """B10 round 10 F8: guarding only the backfill left the RENDER able to
    describe a hand's rows using another job's frames, producing claims true
    of neither. The panel must derive nothing in that situation."""
    from poker_tracker.ui.reconstruction_review import job_id_from_hand_notes

    notes = "CV draft from YOLO card timeline. timeline=/x/job_1_timeline.json"
    assert job_id_from_hand_notes(notes) == 1
    # A context from a different job must not be used to explain this hand.
    foreign = ValidationFrameContext(
        job_id=3, hand_number=1, timeline_hand=_hand(), states=_states(),
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    assert foreign.job_id != job_id_from_hand_notes(notes)
    # With no context at all, nothing frame-derived is emitted — which is what
    # the guard reduces the foreign case to.
    assert _cv_issues_for_db_action(_row(), None) == []


def test_the_attributed_path_threads_the_saved_amount_and_stack() -> None:
    """A10 round 10 F5: forcing db_amount/db_stack_before to None at the call
    sites changed 138 and 468 real rows respectively, with a green suite — the
    pure function was covered, the wiring was not."""
    # Amount filled in: the row must NOT be told its amount is unknown.
    filled = _row(amount=10.0)
    assert not [
        issue
        for issue in _cv_issues_for_db_action(filled, _context())
        if issue.kind == "Amount unknown"
    ], "a filled amount was still reported unknown"
    # Amount empty: it must be.
    assert [
        issue
        for issue in _cv_issues_for_db_action(_row(), _context())
        if issue.kind == "Amount unknown"
    ]
    # A saved stack equal to the action-frame read must be questioned; forcing
    # db_stack_before to None would lose that entirely.
    hand = _hand()
    hand["actions"][0]["amount"] = 10.0
    states = _states()
    states[0]["bets"] = {"1": 10.0}
    states[0]["stacks"] = {"1": 190.0}
    context = ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=hand, states=states,
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    assert [
        issue
        for issue in _cv_issues_for_db_action(
            _row(amount=10.0, stack_before=190.0), context
        )
        if issue.kind == "Stack before looks post-action"
    ], "a post-action stack was not questioned"


def test_an_attributed_row_on_the_wrong_street_is_hedged() -> None:
    """A10 round 10 F4: the hedge was only pinned through the detached stub,
    leaving db_street at the main call site free to be dropped."""
    # Provenance points at frame B (a single-line frame, the flop call) while
    # the row claims preflop — attributed, but on the wrong street.
    row = _row(street="preflop", action_index=1, source_image=FRAME_B)
    issues = _cv_issues_for_db_action(row, _context())
    hedge = next(
        (issue for issue in issues if issue.kind == "Moved off its source street"),
        None,
    )
    assert hedge is not None, "an attributed row on the wrong street was not hedged"
    assert "reconstructed on the flop" in hedge.detail
    assert "now saved on the preflop" in hedge.detail
