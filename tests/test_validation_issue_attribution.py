"""Which reconstructed line a saved row is explained by.

These pin the app-side attribution that round 7 introduced. Round 8 proved it
had no test at all: four separate mutations — including reverting the whole
repair by ignoring stored provenance — left the entire suite green.
"""
from __future__ import annotations

import pytest

from app import _cv_issues_for_db_action
from poker_tracker.persistence.models import Action
from poker_tracker.ui.reconstruction_review import ActionCvIssue, ValidationFrameContext

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


def test_an_order_edit_keeps_the_stack_warning_too() -> None:
    """B11 round 11 F5: round 10 hardened the phantom accusation against an
    Order keystroke but left its sibling on the same derivation branch, so the
    stack warning still vanished and the row went silent."""
    hand = _hand()
    hand["actions"][0]["derivation"] = "inferred_round_complete"
    hand["actions"][0]["stack_before"] = None
    context = ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=hand, states=_states(),
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    before = _cv_issues_for_db_action(_row(amount=10.0), context)
    assert any(issue.kind == "Stack before unknown" for issue in before)
    after = _cv_issues_for_db_action(_row(amount=10.0, action_index=9), context)
    assert any(
        issue.kind == "Stack before unknown" for issue in after
    ), "an order edit silenced the stack warning"


def test_a_delete_instruction_never_asserts_refused_provenance() -> None:
    """B11 round 11 F3: the message that instructs a DELETE claimed "the frame
    this line came from" even for rows whose provenance the backfill had
    explicitly refused to record."""
    from app import _stranded_seat_issue

    states = _states()
    states[0]["dealt_in"] = [2]        # seat 1 holds no cards here
    states[1]["dealt_in"] = [1, 2]     # ... but did earlier? no: index 1 is later
    context = ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=_hand(), states=states,
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    # Without recorded provenance the frame follows the row's CURRENT street
    # and order, so it moves with the edit. Recommending a delete on evidence
    # the edit itself produced would destroy real rows: say nothing.
    assert (
        _stranded_seat_issue(
            _row(action_type="bet", source_image=None), context, FRAME_A
        )
        is None
    )

    recorded = _stranded_seat_issue(
        _row(action_type="bet"), context, FRAME_A
    )
    assert recorded is not None
    assert "The frame this line came from" in recorded.detail

    # A fold legitimately loses its cards on its own frame.
    assert _stranded_seat_issue(
        _row(action_type="fold"), context, FRAME_A
    ) is None


def test_a_seat_never_seen_holding_cards_is_described_as_such() -> None:
    """B11 round 11 F6: 'it had already left the hand' was asserted for a seat
    that never entered it — disprovable by looking at any earlier frame."""
    from app import _stranded_seat_issue

    # A third frame, so frame B is not the terminal one (absence on a hand's
    # last retained frame proves nothing and is guarded separately).
    states = [*_states(), {**_states()[1], "image": "/frames/f2.jpg"}]
    states[0]["dealt_in"] = [2]
    states[1]["dealt_in"] = [2]
    context = ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=_hand(), states=states,
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    # Frame B is later, and seat 1 never held cards on any earlier frame.
    issue = _stranded_seat_issue(_row(action_type="bet"), context, FRAME_B)
    assert issue is not None
    assert "may never have been in this hand" in issue.detail
    assert "already left the hand" not in issue.detail

    # With an earlier frame showing cards, the stronger claim is correct.
    states[0]["dealt_in"] = [1, 2]
    issue = _stranded_seat_issue(_row(action_type="bet"), context, FRAME_B)
    assert issue is not None
    assert "already left the hand" in issue.detail


def test_the_bet_box_ownership_guard_reaches_the_detached_path() -> None:
    """A11 round 11 F2: the stub carried no action_index, so the guard that
    stops a blind-poster being told its stack is post-action returned False
    unconditionally — 62 of 62 calls from this path."""
    hand = _hand()
    # Seat 1 posts a blind first, then acts again on the same street.
    hand["actions"] = [
        {
            "street": "preflop", "action_index": 1, "seat": 1,
            "player_name": "Seat1", "position": "UTG",
            "action_type": "post_blind", "amount": 1.0,
            "source_image": FRAME_A, "derivation": "action_pill",
        },
        {
            "street": "preflop", "action_index": 3, "seat": 1,
            "player_name": "Seat1", "position": "UTG", "action_type": "call",
            "amount": 10.0, "stack_before": 200.0,
            "source_image": FRAME_A, "derivation": "stack_delta",
        },
    ]
    states = _states()
    states[0]["bets"] = {"1": 1.0}          # the blind, not this action
    states[0]["stacks"] = {"1": 200.0}
    context = ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=hand, states=states,
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )
    # A type edit detaches the row, so the stub path runs.
    detached = _row(
        action_index=3, action_type="bet", amount=10.0, stack_before=200.0
    )
    assert not [
        issue
        for issue in _cv_issues_for_db_action(detached, context)
        if issue.kind == "Stack before looks post-action"
    ], "a blind-poster was told its stack was post-action"


def _no_cards_context() -> ValidationFrameContext:
    states = _states()
    states[0]["dealt_in"] = [2]          # seat 1 holds nothing on frame A
    states[0]["stacks"] = {"1": 181.0}   # ... its post-fold stack
    return ValidationFrameContext(
        job_id=1, hand_number=1, timeline_hand=_hand(), states=states,
        reviews_by_image={}, cursor_key="c", pending_hand_key="p",
        recording_start_s=0.0,
    )


def test_an_added_row_on_a_frame_without_its_seat_is_never_handed_a_value() -> None:
    """B12 round 12 F2: a row the operator adds has no provenance, so the
    delete-instruction path cannot fire — and nothing else questioned it. The
    panel offered a post-fold stack as the stack BEFORE a later action, with
    the field opened to invite it."""
    added = _row(action_type="check", amount=None, source_image=None)
    issues = _cv_issues_for_db_action(added, _no_cards_context())
    assert any(
        issue.kind == "Seat not in the hand on this frame" for issue in issues
    ), "nothing questioned a row on a frame that shows no such seat"
    assert all(
        not issue.offers_a_value for issue in issues
    ), "a value was offered for a seat the frame does not show"


def test_no_bet_box_is_never_denied_when_the_reader_read_one() -> None:
    """B12 round 12 F1: round 11's repair landed in one of two sibling paths."""
    context = _no_cards_context()
    context.states[0]["bets"] = {"1": 13.0}
    issue = next(
        issue
        for issue in _cv_issues_for_db_action(
            _row(action_type="bet", source_image=None), context
        )
        if issue.kind == "Amount unknown"
    )
    assert "read 13 BB in its bet box" in issue.detail
    assert "there was no bet box to read" not in issue.detail


def test_the_unattributable_path_never_overclaims_provenance() -> None:
    """The recorded/guessed split must apply to every branch, not just two."""
    context = _no_cards_context()
    guessed = _cv_issues_for_db_action(
        _row(action_type="bet", source_image=None), context
    )
    assert any(
        "closest frame the reconstruction can attribute" in issue.detail
        for issue in guessed
    )
    assert not any(
        "the frame this line came from" in issue.detail.lower()
        for issue in guessed
    )


def test_a_stranded_row_says_its_frame_checks_no_longer_apply() -> None:
    """B12 round 12 F3: an edit on a row with no recorded provenance dropped
    severe warnings and the caption turned reassuring, with nothing saying the
    checks could no longer be made."""
    context = _context()
    stranded = _row(street="turn", action_index=7, source_image=None)
    issues = _cv_issues_for_db_action(stranded, context)
    assert any(
        issue.kind == "Frame checks no longer apply" for issue in issues
    )


def test_the_badge_shows_the_most_severe_kinds_first() -> None:
    """B12 round 12 F5: badge order followed emission, so a claim that a saved
    number is wrong could be truncated behind a positional hedge."""
    from app import _issue_badge_rank

    ranks = [
        _issue_badge_rank(ActionCvIssue(kind=kind, detail=""))
        for kind in (
            "Action may not belong to this hand",
            "Stack before looks post-action",
            "Coverage gap",
            "Moved off its source street",
        )
    ]
    assert ranks == sorted(ranks), "severity order is not monotonic"


def test_the_caption_gate_reads_the_flag_not_the_prose() -> None:
    """A12 round 12 F3: the flag was pinned at the producer and unpinned at
    its only consumer — ignoring it flips 1111 of 1378 captions."""
    from app import _issue_requests_a_stack_value

    offered = ActionCvIssue(
        kind="Stack before unknown",
        detail="The reconstruction read 182.2 BB for this seat on frame 12.",
        offers_a_value=True,
    )
    withheld = ActionCvIssue(
        kind="Stack before unknown",
        detail="The reconstruction read 182.2 BB for this seat on frame 12.",
        offers_a_value=False,
    )
    assert _issue_requests_a_stack_value(offered) is True
    assert _issue_requests_a_stack_value(withheld) is False, (
        "the caption gate ignored the flag and read the wording instead"
    )
    # A non-stack kind never opens the field, however it is worded.
    assert (
        _issue_requests_a_stack_value(
            ActionCvIssue(kind="Coverage gap", detail="x", offers_a_value=True)
        )
        is False
    )


def test_an_unedited_row_is_never_announced_as_edited() -> None:
    """A12 round 12 F7: 'detached' was pinned in one direction only, so always
    announcing it passed — putting 'you edited this' on 707 untouched rows."""
    issues = _cv_issues_for_db_action(_row(), _context())
    assert issues
    assert not any(issue.kind == "Edited line" for issue in issues)


def test_the_seat_absence_notice_is_never_duplicated() -> None:
    """A12 round 12 F13."""
    issues = _cv_issues_for_db_action(
        _row(action_type="check", amount=None, source_image=None),
        _no_cards_context(),
    )
    kinds = [issue.kind for issue in issues]
    assert kinds.count("Seat not in the hand on this frame") <= 1
