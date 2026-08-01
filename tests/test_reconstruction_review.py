import json
from pathlib import Path

from poker_tracker.persistence.completion import parse_completion_evidence
from poker_tracker.ui.reconstruction_review import (
    ACTION_MAY_NOT_BELONG,
    STACK_VALUE_KINDS,
    UNKNOWN_AMOUNT_CODE_TEXT,
    cv_issues_for_timeline_action,
    frame_issue_targets,
    history_impacts,
    key_frames_from_completion_evidence,
    match_db_action_to_frame_target,
    match_db_action_to_timeline_action,
    observed_facts,
    resolve_study_approve_key_frames,
    select_key_frames_for_review,
    states_for_hand,
    timeline_action_by_frame_and_seat,
    timeline_source_image_for_slot,
)


def _fixture():
    states = [
        {
            "state_index": 0,
            "time_s": 0.0,
            "image": "/tmp/a.jpg",
            "stage": "preflop",
            "hero_cards": ["As", "Kd"],
            "board_cards": [],
            "pot": 1.5,
            "dealt_in": [0, 4],
            "stacks": {0: 100, 4: 100},
            "bets": {},
            "pills": {},
            "active_seat": 4,
        },
        {
            "state_index": 1,
            "time_s": 1.0,
            "image": "/tmp/b.jpg",
            "stage": "preflop",
            "hero_cards": ["As", "Kd"],
            "board_cards": [],
            "pot": 3.0,
            "dealt_in": [0, 4],
            "stacks": {0: 100, 4: 97},
            "bets": {4: 3},
            "pills": {4: "raise"},
            "active_seat": 0,
        },
    ]
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 1.0,
        "hero": ["As", "Kd"],
        "source_images": ["/tmp/a.jpg", "/tmp/b.jpg"],
        "players": [
            {"seat": 0, "player_name": "Hero", "position": "SB"},
            {"seat": 4, "player_name": "Seat4", "position": "BTN"},
        ],
        "actions": [
            {
                "street": "preflop",
                "seat": 4,
                "position": "BTN",
                "player_name": "Seat4",
                "action_type": "raise",
                "amount": 3.0,
                "source_image": "/tmp/b.jpg",
                "derivation": "stack_delta",
            }
        ],
        "result": "Hero folds",
        "pot": 3.0,
        "reconciled": True,
    }
    return {"states": states, "hands": [hand]}, hand


def test_frame_review_pairs_observations_with_history_impacts() -> None:
    timeline, hand = _fixture()
    states = states_for_hand(timeline, hand)

    assert [state["image"] for state in states] == ["/tmp/a.jpg", "/tmp/b.jpg"]
    assert ("Hero cards", "As Kd") in observed_facts(states[0])
    impacts = history_impacts(hand, states, 1)
    assert any("BTN Seat4 raise 3 BB" in impact["text"] for impact in impacts)
    assert any(impact["source"] == "stack delta" for impact in impacts)
    assert any(impact["kind"] == "Settlement" for impact in impacts)


def test_frame_issue_targets_link_flagged_frames_to_actions() -> None:
    timeline, hand = _fixture()
    states = states_for_hand(timeline, hand)
    reviews = {
        "/tmp/a.jpg": {"status": "correct", "issue_types": [], "notes": ""},
        "/tmp/b.jpg": {
            "status": "incorrect",
            "issue_types": ["Action / player"],
            "notes": "Should be a call",
        },
    }

    targets = frame_issue_targets(hand, states, reviews)
    assert len(targets) == 1
    assert targets[0].frame_index == 1
    assert targets[0].issue_types == ("Action / player",)
    assert targets[0].action_labels() == (
        "Preflop · BTN Seat4 · Raise 3 BB",
    )
    matched = match_db_action_to_frame_target(
        street="preflop",
        action_type="raise",
        player_name="Seat4",
        position="BTN",
        amount=3.0,
        targets=targets,
    )
    assert matched is not None
    assert matched.source_image == "/tmp/b.jpg"


def test_match_db_action_requires_actor_identity() -> None:
    targets = frame_issue_targets(
        {
            "actions": [
                {
                    "street": "preflop",
                    "seat": 4,
                    "position": "BTN",
                    "player_name": "Seat4",
                    "action_type": "fold",
                    "amount": None,
                    "source_image": "/tmp/b.jpg",
                }
            ]
        },
        [
            {"image": "/tmp/b.jpg", "time_s": 1.0},
        ],
        {
            "/tmp/b.jpg": {
                "status": "incorrect",
                "issue_types": ["Action / player"],
                "notes": "",
            }
        },
    )
    assert (
        match_db_action_to_frame_target(
            street="preflop",
            action_type="fold",
            player_name="Hero",
            position="SB",
            amount=None,
            targets=targets,
        )
        is None
    )
    assert (
        match_db_action_to_frame_target(
            street="preflop",
            action_type="raise",
            player_name="Seat4",
            position="BTN",
            amount=3.0,
            targets=targets,
        )
        is None
    )
    assert (
        match_db_action_to_frame_target(
            street="preflop",
            action_type="fold",
            player_name="Seat4",
            position="BTN",
            amount=None,
            targets=targets,
        )
        is not None
    )


def test_match_db_action_rejects_amount_mismatch() -> None:
    targets = frame_issue_targets(
        {
            "actions": [
                {
                    "street": "preflop",
                    "position": "BTN",
                    "player_name": "Seat4",
                    "action_type": "raise",
                    "amount": 3.0,
                    "source_image": "/tmp/b.jpg",
                }
            ]
        },
        [{"image": "/tmp/b.jpg", "time_s": 1.0}],
        {
            "/tmp/b.jpg": {
                "status": "incorrect",
                "issue_types": ["Amount / stack"],
                "notes": "",
            }
        },
    )
    assert (
        match_db_action_to_frame_target(
            street="preflop",
            action_type="raise",
            player_name="Seat4",
            position="BTN",
            amount=7.0,
            targets=targets,
        )
        is None
    )


def test_select_key_frames_picks_hero_streets_and_terminal() -> None:
    states = [
        {
            "time_s": 0.0,
            "image": "/tmp/hero.jpg",
            "hero_cards": ["As", "Kd"],
            "board_cards": [],
        },
        {
            "time_s": 1.0,
            "image": "/tmp/flop.jpg",
            "hero_cards": ["As", "Kd"],
            "board_cards": ["2c", "3d", "4h"],
        },
        {
            "time_s": 2.0,
            "image": "/tmp/turn.jpg",
            "hero_cards": ["As", "Kd"],
            "board_cards": ["2c", "3d", "4h", "5s"],
        },
        {
            "time_s": 3.0,
            "image": "/tmp/river.jpg",
            "hero_cards": ["As", "Kd"],
            "board_cards": ["2c", "3d", "4h", "5s", "6c"],
        },
        {
            "time_s": 4.0,
            "image": "/tmp/end.jpg",
            "hero_cards": ["As", "Kd"],
            "board_cards": ["2c", "3d", "4h", "5s", "6c"],
        },
    ]

    frames = select_key_frames_for_review(states)
    assert [(frame.label, frame.image_path) for frame in frames] == [
        ("Hero cards", "/tmp/hero.jpg"),
        ("Flop", "/tmp/flop.jpg"),
        ("Turn", "/tmp/turn.jpg"),
        ("River", "/tmp/river.jpg"),
        ("Terminal", "/tmp/end.jpg"),
    ]


def test_select_key_frames_dedupes_when_terminal_reuses_street_image() -> None:
    states = [
        {
            "time_s": 0.0,
            "image": "/tmp/hero.jpg",
            "hero_cards": ["As", "Kd"],
            "board_cards": [],
        },
        {
            "time_s": 1.0,
            "image": "/tmp/flop.jpg",
            "hero_cards": ["As", "Kd"],
            "board_cards": ["2c", "3d", "4h"],
        },
    ]

    frames = select_key_frames_for_review(states)
    assert [frame.label for frame in frames] == ["Hero cards", "Flop"]
    assert [frame.image_path for frame in frames] == ["/tmp/hero.jpg", "/tmp/flop.jpg"]


def test_key_frames_from_completion_evidence_uses_boundaries() -> None:
    evidence = parse_completion_evidence(
        {
            "evidence_version": 1,
            "preceding_boundary": {
                "kind": "hand_start",
                "frame_ref": "frames/start.png",
                "timestamp_s": 1.0,
            },
            "following_boundary": {
                "kind": "hand_end",
                "frame_ref": "frames/end.png",
                "timestamp_s": 9.0,
            },
            "source_frames": [
                "frames/start.png",
                "frames/mid_a.png",
                "frames/mid_b.png",
                "frames/end.png",
            ],
        }
    )

    frames = key_frames_from_completion_evidence(evidence)
    assert [frame.label for frame in frames] == [
        "Hand start",
        "Source 1",
        "Source 2",
        "Terminal",
    ]
    assert frames[0].image_path == "frames/start.png"
    assert frames[-1].image_path == "frames/end.png"


def test_resolve_study_approve_key_frames_prefers_timeline(tmp_path: Path) -> None:
    timeline = {
        "hands": [
            {
                "hand_number": 3,
                "t_start": 0.0,
                "t_end": 2.0,
                "source_images": ["/tmp/a.jpg", "/tmp/b.jpg"],
                "hero": ["Ah", "Kh"],
            }
        ],
        "states": [
            {
                "time_s": 0.0,
                "image": "/tmp/a.jpg",
                "hero_cards": ["Ah", "Kh"],
                "board_cards": [],
                "state_index": 0,
            },
            {
                "time_s": 2.0,
                "image": "/tmp/b.jpg",
                "hero_cards": ["Ah", "Kh"],
                "board_cards": ["2c", "3d", "4h"],
                "state_index": 1,
            },
        ],
    }
    (tmp_path / "job_42_timeline.json").write_text(
        json.dumps(timeline), encoding="utf-8"
    )
    evidence = parse_completion_evidence(
        {
            "evidence_version": 1,
            "preceding_boundary": {"frame_ref": "frames/ignored.png"},
            "following_boundary": {"frame_ref": "frames/ignored_end.png"},
            "source_frames": ["frames/ignored.png"],
        }
    )

    frames = resolve_study_approve_key_frames(
        job_id=42,
        hand_number=3,
        evidence=evidence,
        timeline_dir=tmp_path,
    )
    assert [(frame.label, frame.image_path) for frame in frames] == [
        ("Hero cards", "/tmp/a.jpg"),
        ("Flop", "/tmp/b.jpg"),
    ]


def _cv_issue_fixture():
    """Hand-1-like timeline slice: partial start, refused bet reads, a gap."""
    states = [
        {
            "state_index": 0,
            "time_s": 0.0,
            "image": "/tmp/f0.jpg",
            "stage": "preflop",
            "bets": {},
            "bets_unknown": {"1": "below_calibrated_render_size"},
            "stacks_unknown": {},
            "unmeasured_transitions": [7],
            "coverage_gap": False,
            "prior_gap_s": 0.0,
        },
        {
            "state_index": 1,
            "time_s": 9.0,
            "image": "/tmp/f1.jpg",
            "stage": "preflop",
            "bets": {"7": 12.0},
            "bets_unknown": {},
            "stacks_unknown": {},
            "unmeasured_transitions": [],
            "coverage_gap": True,
            "prior_gap_s": 9.0,
        },
        # A terminal frame, so "seat holds no cards" on the LAST frame (where
        # the table has already cleared) is distinguishable from mid-hand.
        {
            "state_index": 2,
            "time_s": 12.0,
            "image": "/tmp/f2.jpg",
            "stage": "preflop",
            "bets": {},
            "bets_unknown": {},
            "stacks_unknown": {},
            "unmeasured_transitions": [],
            "coverage_gap": False,
            "prior_gap_s": 3.0,
        },
    ]
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 9.0,
        "warnings": ["starting_stack_unknown", "amounts_unknown_in_ledger"],
        "players": [
            {"seat": 1, "player_name": "Seat1", "position": "UTG+1", "starting_stack": None},
            {"seat": 7, "player_name": "Seat7", "position": "BB", "starting_stack": 224.2},
        ],
        "actions": [
            {
                "street": "preflop",
                "action_index": 1,
                "seat": 1,
                "player_name": "Seat1",
                "position": "UTG+1",
                "action_type": "raise",
                "amount": None,
                "source_image": "/tmp/f0.jpg",
                "derivation": "action_pill",
            },
            {
                "street": "preflop",
                "action_index": 2,
                "seat": 7,
                "player_name": "Seat7",
                "position": "BB",
                "action_type": "raise",
                "amount": 12.0,
                "source_image": "/tmp/f1.jpg",
                "derivation": "stack_delta",
            },
        ],
    }
    return hand, states


def test_match_db_action_to_timeline_action_uses_street_and_index() -> None:
    hand, _states = _cv_issue_fixture()
    matched = match_db_action_to_timeline_action(
        hand,
        street="Preflop",
        action_index=2,
        action_type="raise",
        position="BB",
        player_name="Seat7",
    )
    assert matched is not None
    assert matched["seat"] == 7


def test_match_db_action_to_timeline_action_rejects_mismatches() -> None:
    hand, _states = _cv_issue_fixture()
    # Same slot index but the operator reassigned the actor: do not borrow issues.
    assert (
        match_db_action_to_timeline_action(
            hand,
            street="preflop",
            action_index=1,
            action_type="raise",
            position="BB",
            player_name="Seat7",
        )
        is None
    )
    # Operator corrected the action type (raise -> check): the slot no longer
    # matches, so the check cannot inherit the raise's amount issues.
    assert (
        match_db_action_to_timeline_action(
            hand,
            street="preflop",
            action_index=1,
            action_type="check",
            position="UTG+1",
            player_name="Seat1",
        )
        is None
    )
    assert (
        match_db_action_to_timeline_action(
            hand,
            street="preflop",
            action_index=None,
            action_type="raise",
            position="UTG+1",
            player_name="Seat1",
        )
        is None
    )


def test_cv_issues_report_refused_amount_with_code_and_remedy() -> None:
    hand, states = _cv_issue_fixture()
    issues = cv_issues_for_timeline_action(
        hand["actions"][0],
        hand,
        states,
        db_amount=None,
        db_stack_before=None,
    )
    kinds = [issue.kind for issue in issues]
    assert "Amount unknown" in kinds
    amount_issue = next(issue for issue in issues if issue.kind == "Amount unknown")
    assert "refused to guess" in amount_issue.detail
    assert "enter the chips this seat added" in amount_issue.detail
    assert "1272×896" in amount_issue.detail
    assert amount_issue.frame_index == 0
    # Seat 1 committed before the recording started (t_start == 0):
    # its starting stack is unknown.
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "recording starts mid-hand" in stack_issue.detail


def test_cv_issues_clear_once_operator_fills_in_values() -> None:
    hand, states = _cv_issue_fixture()
    issues = cv_issues_for_timeline_action(
        hand["actions"][0],
        hand,
        states,
        db_amount=6.0,
        db_stack_before=212.2,
    )
    assert all(
        issue.kind not in {"Amount unknown", "Stack before unknown"}
        for issue in issues
    )


def test_cv_issues_flag_coverage_gap_and_unmeasured_transition() -> None:
    hand, states = _cv_issue_fixture()
    gap_issues = cv_issues_for_timeline_action(
        hand["actions"][1],
        hand,
        states,
        db_amount=12.0,
        db_stack_before=224.2,
    )
    kinds = [issue.kind for issue in gap_issues]
    assert kinds == ["Coverage gap"]
    # Wording must not claim the video went unsampled; only that no new
    # distinct readable state was retained.
    assert "No new readable table state was retained for 9s" in gap_issues[0].detail
    assert "sampled frames" not in gap_issues[0].detail

    transition_issues = cv_issues_for_timeline_action(
        {
            "street": "preflop",
            "action_index": 3,
            "seat": 7,
            "player_name": "Seat7",
            "position": "BB",
            "action_type": "check",
            "amount": None,
            "source_image": "/tmp/f0.jpg",
        },
        hand,
        states,
        db_amount=None,
        db_stack_before=224.2,
    )
    assert [issue.kind for issue in transition_issues] == ["Unmeasured transition"]
    assert "could not be tracked continuously" in transition_issues[0].detail


def test_no_mid_hand_recording_claim_for_later_hands() -> None:
    """A hand starting at t=48 was fully on camera; starting_stack_unknown on
    it means a refused read, never "committed before the recording started"."""
    hand, states = _cv_issue_fixture()
    hand = dict(hand, t_start=48.0, t_end=57.0)
    issues = cv_issues_for_timeline_action(
        hand["actions"][0],
        hand,
        states,
        db_amount=None,
        db_stack_before=None,
    )
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "recording" not in stack_issue.detail
    assert "never established cleanly" in stack_issue.detail


def test_stack_before_prefers_true_mechanisms_over_stories() -> None:
    hand, states = _cv_issue_fixture()
    # An inferred round-completion line carries no frame reading by design.
    inferred = {
        "street": "turn",
        "action_index": 2,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "check",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "inferred_round_complete",
    }
    issues = cv_issues_for_timeline_action(
        inferred, hand, states, db_amount=None, db_stack_before=None
    )
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "inferred from the betting round completing" in stack_issue.detail
    assert "recording" not in stack_issue.detail
    # Seat 7's starting stack IS known and nothing explains the hole:
    # emit no story at all rather than a fabricated one.
    plain = dict(inferred, derivation="action_pill")
    issues = cv_issues_for_timeline_action(
        plain, hand, states, db_amount=None, db_stack_before=None
    )
    assert all(issue.kind != "Stack before unknown" for issue in issues)


def test_amount_issue_reports_timeline_read_when_db_amount_cleared() -> None:
    """If the reconstruction read an amount but the saved row lost it, say that
    — never claim no frame showed it."""
    hand, states = _cv_issue_fixture()
    issues = cv_issues_for_timeline_action(
        hand["actions"][1],
        hand,
        states,
        db_amount=None,
        db_stack_before=224.2,
    )
    amount_issue = next(issue for issue in issues if issue.kind == "Amount unknown")
    assert "The reconstruction read 12 BB" in amount_issue.detail
    assert "No readable frame" not in amount_issue.detail


def test_amount_issue_uses_readable_bet_before_generic_fallback() -> None:
    """A2 round 2: never claim 'no readable frame showed this amount' when the
    source frame's bet box WAS read."""
    hand, states = _cv_issue_fixture()
    states[0]["bets"] = {"1": 12.8}
    states[0]["bets_unknown"] = {}
    issues = cv_issues_for_timeline_action(
        dict(hand["actions"][0], derivation="amount_unknown"),
        hand,
        states,
        db_amount=None,
        db_stack_before=212.2,
    )
    amount_issue = next(issue for issue in issues if issue.kind == "Amount unknown")
    assert "12.8 BB total in this seat's bet box" in amount_issue.detail
    assert "could not be isolated" in amount_issue.detail
    assert "chips this seat added here" in amount_issue.detail
    assert "No readable frame" not in amount_issue.detail


def test_amount_issue_names_inference_for_inferred_lines() -> None:
    hand, states = _cv_issue_fixture()
    inferred_call = {
        "street": "preflop",
        "action_index": 3,
        "seat": 1,
        "player_name": "Seat1",
        "position": "UTG+1",
        "action_type": "call",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "inferred_still_in",
    }
    issues = cv_issues_for_timeline_action(
        inferred_call, hand, states, db_amount=None, db_stack_before=212.2
    )
    amount_issue = next(issue for issue in issues if issue.kind == "Amount unknown")
    assert "inferred from the seat still being in the hand" in amount_issue.detail
    # Not an OCR failure: no refusal-code story.
    assert "refused" not in amount_issue.detail


def test_stack_issue_reports_timeline_value_when_field_cleared() -> None:
    """A2 round 2: a cleared stack field must report the computed value, not a
    fabricated mechanism story."""
    hand, states = _cv_issue_fixture()
    issues = cv_issues_for_timeline_action(
        dict(hand["actions"][0], stack_before=212.2),
        hand,
        states,
        db_amount=6.0,
        db_stack_before=None,
    )
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "The reconstruction computed 212.2 BB" in stack_issue.detail
    assert "recording" not in stack_issue.detail


def test_stack_issue_scans_backward_for_the_latest_refusal() -> None:
    """A2 round 2: stack-before comes from EARLIER frames; a refusal there must
    be found instead of emitting nothing."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks_unknown"] = {"7": "no_digit_run"}
    later_action = {
        "street": "preflop",
        "action_index": 3,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "bet",
        "amount": 4.0,
        "source_image": "/tmp/f1.jpg",
        "derivation": "bet_text",
    }
    issues = cv_issues_for_timeline_action(
        later_action, hand, states, db_amount=4.0, db_stack_before=None
    )
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "On frame 1" in stack_issue.detail
    assert "no digits were found" in stack_issue.detail
    # Jump goes to the frame with the refusal, not the action's source frame.
    assert stack_issue.frame_index == 0
    # A readable stack nearer the action (still strictly before it) stops the
    # backward scan, so no refusal story is told.
    states[1]["stacks"] = {"7": 224.2}
    on_third = dict(later_action, source_image=states[2]["image"])
    issues = cv_issues_for_timeline_action(
        on_third, hand, states, db_amount=4.0, db_stack_before=None
    )
    assert all(issue.kind != "Stack before unknown" for issue in issues)


def test_inferred_stack_issue_points_at_visible_stack() -> None:
    """B2 round 2: when the jump-target frame shows the seat's stack, say so
    and tell the operator to enter it."""
    hand, states = _cv_issue_fixture()
    # The value must come from a frame BEFORE the action: the action's own
    # frame shows the stack after the chips moved.
    states[0]["stacks"] = {"7": 269.1}
    inferred = {
        "street": "turn",
        "action_index": 2,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "check",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "inferred_round_complete",
    }
    issues = cv_issues_for_timeline_action(
        inferred, hand, states, db_amount=None, db_stack_before=None
    )
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "read 269.1 BB for this seat on frame 1" in stack_issue.detail
    assert "More fields \u2192 Stack before (BB)" in stack_issue.detail
    assert stack_issue.frame_index == 0


def test_recording_start_offset_keeps_mid_hand_claim_honest() -> None:
    """A2 round 2 (speculation made real): a recording that opens on a lobby
    gives its first hand t_start > 0; the mid-hand claim must follow the
    recording's own first sampled second."""
    hand, states = _cv_issue_fixture()
    hand = dict(hand, t_start=30.0)
    issues = cv_issues_for_timeline_action(
        hand["actions"][0],
        hand,
        states,
        db_amount=6.0,
        db_stack_before=None,
        recording_start_s=30.0,
    )
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "recording starts mid-hand" in stack_issue.detail
    # And the same hand with a recording that started earlier: no such claim.
    issues = cv_issues_for_timeline_action(
        hand["actions"][0],
        hand,
        states,
        db_amount=6.0,
        db_stack_before=None,
        recording_start_s=0.0,
    )
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "recording" not in stack_issue.detail


def test_identity_only_match_keeps_frame_flag_after_type_correction() -> None:
    """B2 round 2: frame flags are frame-level; correcting an action's type
    must not detach the row from its flagged source frame."""
    timeline, hand = _fixture()
    states = states_for_hand(timeline, hand)
    reviews = {
        "/tmp/b.jpg": {
            "status": "incorrect",
            "issue_types": ["Cards / board"],
            "notes": "",
        }
    }
    targets = frame_issue_targets(hand, states, reviews)
    # Operator corrected raise -> bet: the strict match fails...
    strict = match_db_action_to_frame_target(
        street="preflop",
        action_type="bet",
        player_name="Seat4",
        position="BTN",
        amount=3.0,
        targets=targets,
    )
    assert strict is None
    # ...but the identity-only fallback still finds the flagged frame.
    relaxed = match_db_action_to_frame_target(
        street="preflop",
        action_type="bet",
        player_name="Seat4",
        position="BTN",
        amount=3.0,
        targets=targets,
        identity_only=True,
    )
    assert relaxed is not None
    assert relaxed.status == "incorrect"
    assert relaxed.source_image == "/tmp/b.jpg"


def test_stack_issue_names_unknown_amount_as_the_blocker() -> None:
    """A money action whose own amount is unknown cannot have its stack-before
    back-computed; say that instead of emitting nothing. (Seat 7's starting
    stack IS known, so no other rung explains the hole.)"""
    hand, states = _cv_issue_fixture()
    unresolved = {
        "street": "preflop",
        "action_index": 3,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "bet",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "amount_unknown",
    }
    issues = cv_issues_for_timeline_action(
        unresolved,
        hand,
        states,
        db_amount=None,
        db_stack_before=None,
    )
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "this line's own amount is unknown" in stack_issue.detail
    assert "More fields \u2192 Stack before (BB)" in stack_issue.detail


def test_stack_issue_names_the_real_starting_stack_mechanism() -> None:
    """A3 round 3: committed_at_start_unknown means the stack READ FINE but the
    chips already on the felt could not be sized. Saying 'never established
    cleanly' there is disprovable by looking at the frame."""
    hand, states = _cv_issue_fixture()
    hand = dict(
        hand,
        t_start=48.0,
        players=[
            {
                "seat": 1,
                "player_name": "Seat1",
                "position": "UTG+1",
                "starting_stack": None,
                "starting_stack_unknown": "committed_at_start_unknown",
            }
        ],
    )
    states[0]["stacks"] = {"1": 406.1}
    later = dict(hand["actions"][0], source_image="/tmp/f1.jpg")
    issues = cv_issues_for_timeline_action(
        later, hand, states, db_amount=6.0, db_stack_before=None
    )
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "stack read fine" in stack_issue.detail
    assert "could not be sized" in stack_issue.detail
    assert "never established cleanly" not in stack_issue.detail
    # And it points at a frame that actually carries the value.
    assert "read 406.1 BB for this seat on frame 1" in stack_issue.detail


def test_inferred_line_on_a_frame_without_cards_offers_deletion() -> None:
    """B2 round 3: three of hand 2's inferred checks cited a frame where the
    seat holds no cards. That frame is evidence the action did not happen, so
    the next step must not be 'enter a number'."""
    hand, states = _cv_issue_fixture()
    states[1]["dealt_in"] = [0, 2, 4]
    states[1]["stacks"] = {"7": 269.1}
    inferred = {
        "street": "turn",
        "action_index": 2,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "check",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "inferred_round_complete",
    }
    issues = cv_issues_for_timeline_action(
        inferred, hand, states, db_amount=None, db_stack_before=None
    )
    issue = next(
        issue for issue in issues if issue.kind == ACTION_MAY_NOT_BELONG
    )
    # The evidence must be what the frames establish — the seat was dealt in
    # and later had no cards, i.e. it folded — not the falsifiable claim that
    # it was never in the hand.
    assert "held cards through frame 1" in issue.detail
    assert "had already folded" in issue.detail
    assert "add it under 'Add a missing action'" in issue.detail
    assert "269.1" not in issue.detail
    # It must NOT ask for a stack: that field is what legitimizes a fake row.
    assert all(issue.kind != "Stack before unknown" for issue in issues)


def test_stack_blocker_tracks_the_saved_amount_not_the_timeline() -> None:
    """B2 round 3: the 'own amount is unknown' branch keyed off the timeline
    amount, so it persisted verbatim after the operator filled the amount in."""
    hand, states = _cv_issue_fixture()
    unresolved = {
        "street": "preflop",
        "action_index": 3,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "bet",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "amount_unknown",
    }
    before = cv_issues_for_timeline_action(
        unresolved, hand, states, db_amount=None, db_stack_before=None
    )
    assert any(
        "this line's own amount is unknown" in issue.detail for issue in before
    )
    after = cv_issues_for_timeline_action(
        unresolved, hand, states, db_amount=3.0, db_stack_before=None
    )
    assert not any(
        "own amount is unknown" in issue.detail for issue in after
    ), "stale dependency claim survived the operator resolving the amount"


def test_stack_hints_name_a_frame_that_carries_the_value() -> None:
    """B2 round 3: 'read it off the nearest legible frame' while jumping to the
    illegible one — and sometimes no legible frame exists at all."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks"] = {"1": 224.2}
    states[1]["stacks_unknown"] = {"1": "no_digit_run"}
    action = {
        "street": "preflop",
        "action_index": 3,
        "seat": 1,
        "player_name": "Seat1",
        "position": "UTG+1",
        "action_type": "check",
        "amount": None,
        "source_image": states[2]["image"],
        "derivation": "action_pill",
    }
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=None, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert "no digits were found" in issue.detail
    assert "read 224.2 BB for this seat on frame 1" in issue.detail
    # The jump must land on the legible frame, not the refused one.
    assert issue.frame_index == 0

    # No frame BEFORE the action shows it: say so rather than send them hunting.
    states[0].pop("stacks")
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=4.0, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert "No frame before this action shows this seat's stack" in issue.detail
    assert "leave the field empty" in issue.detail


def test_every_stack_message_names_the_control_that_holds_the_field() -> None:
    """B2 round 3 S1: 'enter it below' pointed at a field hidden behind an
    unchecked More-fields box."""
    hand, states = _cv_issue_fixture()
    variants = [
        dict(hand["actions"][0], stack_before=212.2),
        dict(hand["actions"][0], derivation="inferred_round_complete"),
        dict(hand["actions"][0], derivation="amount_unknown"),
    ]
    for timeline_action in variants:
        issues = cv_issues_for_timeline_action(
            timeline_action, hand, states, db_amount=None, db_stack_before=None
        )
        stack_issue = next(
            (issue for issue in issues if issue.kind == "Stack before unknown"),
            None,
        )
        if stack_issue is None:
            continue
        assert "More fields → Stack before (BB)" in stack_issue.detail


def test_generic_amount_fallback_requires_an_all_frames_scan() -> None:
    """A3 round 3: 'No readable frame showed this amount' was asserted after
    checking exactly one frame."""
    hand, states = _cv_issue_fixture()
    states[0]["bets_unknown"] = {}
    states[1]["bets"] = {"1": 9.5}
    action = dict(hand["actions"][0], derivation="amount_unknown")
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=None, db_stack_before=212.2
        )
        if issue.kind == "Amount unknown"
    )
    assert "read 9.5 BB there on frame 2" in issue.detail
    assert "No frame in this hand" not in issue.detail

    # Genuinely absent everywhere: the absolute claim is now licensed.
    states[1].pop("bets")
    later_hand = dict(hand, t_start=48.0)
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, later_hand, states, db_amount=None, db_stack_before=212.2
        )
        if issue.kind == "Amount unknown"
    )
    assert "No frame in this hand shows this seat's bet box" in issue.detail

    # On the mid-hand opener the same absence gets the stronger explanation,
    # but only after the scan confirms no frame shows it.
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=None, db_stack_before=212.2
        )
        if issue.kind == "Amount unknown"
    )
    assert "recording starts mid-hand" in issue.detail
    assert "no frame shows this seat's bet box" in issue.detail


def test_ambiguity_guard_inspects_every_top_scoring_candidate() -> None:
    """A3 round 3: with two candidates on one frame and a third elsewhere, the
    top-2 comparison passed and silently wired the row to the wrong frame."""
    hand_payload = {
        "actions": [
            {
                "street": "preflop",
                "position": "BTN",
                "player_name": "Seat4",
                "action_type": "call",
                "amount": None,
                "source_image": "/tmp/x.jpg",
            },
            {
                "street": "preflop",
                "position": "BTN",
                "player_name": "Seat4",
                "action_type": "call",
                "amount": None,
                "source_image": "/tmp/x.jpg",
            },
            {
                "street": "preflop",
                "position": "BTN",
                "player_name": "Seat4",
                "action_type": "call",
                "amount": None,
                "source_image": "/tmp/y.jpg",
            },
        ]
    }
    states = [
        {"image": "/tmp/x.jpg", "time_s": 1.0},
        {"image": "/tmp/y.jpg", "time_s": 2.0},
    ]
    reviews = {
        image: {"status": "incorrect", "issue_types": ["Pot"], "notes": ""}
        for image in ("/tmp/x.jpg", "/tmp/y.jpg")
    }
    targets = frame_issue_targets(hand_payload, states, reviews)
    assert (
        match_db_action_to_frame_target(
            street="preflop",
            action_type="call",
            player_name="Seat4",
            position="BTN",
            amount=None,
            targets=targets,
        )
        is None
    )


def test_source_image_lookup_ignores_type_but_not_identity() -> None:
    """The frame a row came from must survive a type correction, without
    letting a different actor borrow it."""
    hand, _states = _cv_issue_fixture()
    assert (
        timeline_source_image_for_slot(
            hand,
            street="preflop",
            action_index=1,
            position="UTG+1",
            player_name="Seat1",
        )
        == "/tmp/f0.jpg"
    )
    # Type is irrelevant here (no type argument at all), but a different actor
    # in the slot must not resolve.
    assert (
        timeline_source_image_for_slot(
            hand,
            street="preflop",
            action_index=1,
            position="BB",
            player_name="Seat7",
        )
        is None
    )
    assert (
        timeline_source_image_for_slot(
            hand,
            street="preflop",
            action_index=None,
            position="UTG+1",
            player_name="Seat1",
        )
        is None
    )


def test_refusal_codes_all_have_human_text() -> None:
    """A3/B2 round 3: three codes present in real timelines fell through to
    machine-speak."""
    for code in (
        "below_calibrated_render_size",
        "no_digit_run",
        "bet_boxes_disagree",
        "stack_boxes_disagree",
        "ambiguous_longest_run",
    ):
        assert code in UNKNOWN_AMOUNT_CODE_TEXT
        assert "_" not in UNKNOWN_AMOUNT_CODE_TEXT[code]


def test_showdown_cards_are_not_evidence_a_seat_was_out() -> None:
    """A4 round 4 [critical]: dealt_in counts card BACKS. At showdown villain
    cards flip face-up into villain_cards, so absence from dealt_in accused six
    real actions of never happening — and told the operator to delete them."""
    hand, states = _cv_issue_fixture()
    states[1]["dealt_in"] = [0, 2, 4]
    states[1]["villain_cards"] = {"7": ["Kd", "7d"]}
    inferred = {
        "street": "river",
        "action_index": 2,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "check",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "inferred_round_complete",
    }
    issues = cv_issues_for_timeline_action(
        inferred, hand, states, db_amount=None, db_stack_before=None
    )
    assert all(issue.kind != ACTION_MAY_NOT_BELONG for issue in issues)
    assert not any("delete" in issue.detail.lower() for issue in issues)


def test_terminal_frame_absence_is_not_evidence_a_seat_was_out() -> None:
    """A4 round 4 [critical]: a hand's last retained frame has already cleared
    the table, so nobody holds cards there."""
    hand, states = _cv_issue_fixture()
    states[-1]["dealt_in"] = [0]
    inferred = {
        "street": "river",
        "action_index": 2,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "check",
        "amount": None,
        "source_image": states[-1]["image"],
        "derivation": "inferred_round_complete",
    }
    issues = cv_issues_for_timeline_action(
        inferred, hand, states, db_amount=None, db_stack_before=None
    )
    assert all(issue.kind != ACTION_MAY_NOT_BELONG for issue in issues)


def test_deletion_offer_never_asks_for_a_stack_value() -> None:
    """B4 round 4 H1: the delete branch reused the stack kind, so the editor
    pre-opened the stack field and captioned it as requested — recreating the
    phantom row one control below."""
    hand, states = _cv_issue_fixture()
    states[1]["dealt_in"] = [0, 2, 4]
    states[1]["stacks"] = {"7": 269.1}
    inferred = {
        "street": "turn",
        "action_index": 2,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "check",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "inferred_round_complete",
    }
    issues = cv_issues_for_timeline_action(
        inferred, hand, states, db_amount=None, db_stack_before=None
    )
    assert any(issue.kind == ACTION_MAY_NOT_BELONG for issue in issues)
    assert all(issue.kind not in STACK_VALUE_KINDS for issue in issues)


def test_stack_hint_never_offers_a_post_action_reading() -> None:
    """A4 round 4 F2: the scan started at distance 0, so 30 of 32 hints named
    the action's own frame — which shows the stack AFTER the chips moved."""
    hand, states = _cv_issue_fixture()
    # Seat 1's starting stack is unknown in the fixture, so a hint is offered.
    states[0]["stacks"] = {"1": 200.0}   # before
    states[1]["stacks"] = {"1": 188.0}   # after this action's chips moved
    action = {
        "street": "preflop",
        "action_index": 3,
        "seat": 1,
        "player_name": "Seat1",
        "position": "UTG+1",
        "action_type": "bet",
        "amount": 12.0,
        "source_image": "/tmp/f1.jpg",
        "derivation": "bet_text",
    }
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert "200 BB" in issue.detail
    assert "188" not in issue.detail
    assert issue.frame_index == 0


def test_stack_hints_attribute_numbers_to_the_reader_not_the_image() -> None:
    """A4 round 4 F3: OCR misreads exist in this corpus (a 10x decimal error),
    so 'the frame shows X' overstates what is known."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks"] = {"1": 142.8}
    action = dict(
        hand["actions"][0], source_image="/tmp/f1.jpg", action_index=9
    )
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert "The reconstruction read 142.8 BB" in issue.detail
    assert "frame shows" not in issue.detail.lower()


def test_computed_stack_message_jumps_to_a_frame_carrying_that_value() -> None:
    """B4 round 4 H3: 194 of 468 'confirm it' messages jumped to a frame
    showing a different number."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks"] = {"7": 224.2}
    states[1]["stacks"] = {"7": 212.2}
    action = dict(hand["actions"][1], stack_before=224.2, source_image="/tmp/f1.jpg")
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert "computed 224.2 BB" in issue.detail
    assert "cannot confirm itself" in issue.detail
    assert "reader's own value for frame 1" in issue.detail
    assert issue.frame_index == 0


def test_inferred_reasons_distinguish_the_two_mechanisms() -> None:
    """B4 round 4 M4: 'inferred from the betting round completing' was asserted
    on inferred_still_in lines, collapsing two distinct mechanisms."""
    hand, states = _cv_issue_fixture()
    still_in = dict(
        hand["actions"][0], action_type="call", derivation="inferred_still_in"
    )
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            still_in, hand, states, db_amount=None, db_stack_before=212.2
        )
        if issue.kind == "Amount unknown"
    )
    assert "seat still being in the hand" in issue.detail
    assert "betting round completing" not in issue.detail


def test_every_amount_message_warns_that_bet_boxes_show_street_totals() -> None:
    """B4 round 4 M1: the caveat that prevents entering a street total instead
    of the increment appeared in only 2 of 6 branches."""
    hand, states = _cv_issue_fixture()
    variants = [
        dict(hand["actions"][0]),
        dict(hand["actions"][0], derivation="inferred_still_in"),
        dict(hand["actions"][1], amount=None, source_image="/tmp/f0.jpg"),
    ]
    for timeline_action in variants:
        issues = cv_issues_for_timeline_action(
            timeline_action, hand, states, db_amount=None, db_stack_before=212.2
        )
        amount_issue = next(
            (issue for issue in issues if issue.kind == "Amount unknown"), None
        )
        if amount_issue is None:
            continue
        assert (
            "chips this seat added" in amount_issue.detail
            or "total for the street" in amount_issue.detail
        ), amount_issue.detail


def test_computed_stack_never_confirms_itself() -> None:
    """B5 round 5 F1: matching a computed stack against the same OCR read it
    came from has no diagnostic power and launders a misread as verified."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks"] = {"7": 142.8}
    action = dict(hand["actions"][1], stack_before=142.8)
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert "cannot confirm itself" in issue.detail
    assert "read the stack off that frame yourself" in issue.detail
    # Never assert the image shows it — this is an OCR read.
    assert "shows that value" not in issue.detail


def test_incoherent_ledger_is_surfaced_on_computed_stacks() -> None:
    hand, states = _cv_issue_fixture()
    hand = dict(hand, warnings=[*hand["warnings"], "stack_ledger_incoherent"])
    states[0]["stacks"] = {"7": 142.8}
    action = dict(hand["actions"][1], stack_before=142.8)
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert "chip ledger does not balance" in issue.detail


def test_carrier_frame_is_never_after_the_action() -> None:
    """A5 round 5 F1: an unbounded scan offered the terminal settlement frame
    as evidence for a preflop stack on 277 of 468 rows."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks"] = {"7": 224.2}
    states[2]["stacks"] = {"7": 224.2}   # same value returns later in the hand
    action = dict(hand["actions"][1], stack_before=224.2)
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert issue.frame_index == 0
    assert "frame 1" in issue.detail
    assert "frame 3" not in issue.detail


def test_amount_branch_never_instructs_a_delete() -> None:
    """A5/B5 round 5 F2: the second delete-offer site kept the round-3 wording,
    with no terminal guard and under an 'Amount unknown' heading."""
    hand, states = _cv_issue_fixture()
    states[-1]["dealt_in"] = [0]
    on_terminal = {
        "street": "river",
        "action_index": 2,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "call",
        "amount": None,
        "source_image": states[-1]["image"],
        "derivation": "inferred_round_complete",
    }
    issues = cv_issues_for_timeline_action(
        on_terminal, hand, states, db_amount=None, db_stack_before=None
    )
    assert not any("delete" in issue.detail.lower() for issue in issues)
    # And a mid-hand absence explains the missing read without accusing.
    states[1]["dealt_in"] = [0, 2, 4]
    mid_hand = dict(on_terminal, source_image="/tmp/f1.jpg")
    issues = cv_issues_for_timeline_action(
        mid_hand, hand, states, db_amount=None, db_stack_before=212.2
    )
    amount_issue = next(
        issue for issue in issues if issue.kind == "Amount unknown"
    )
    assert "no cards for this seat" in amount_issue.detail
    assert "delete" not in amount_issue.detail.lower()


def test_stack_hint_flags_intervening_commitments_as_stale() -> None:
    """B5 round 5 F4: a reading from before this seat's earlier bet is not the
    stack before THIS action, but read as a number to type in."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks"] = {"1": 1450.8}
    # Seat 1 commits chips on frame 2, between the reading and the action.
    hand = dict(
        hand,
        actions=[
            *hand["actions"],
            {
                "street": "preflop",
                "action_index": 8,
                "seat": 1,
                "player_name": "Seat1",
                "position": "UTG+1",
                "action_type": "raise",
                "amount": 10.0,
                "source_image": "/tmp/f1.jpg",
                "derivation": "stack_delta",
            },
        ],
    )
    later = {
        "street": "flop",
        "action_index": 1,
        "seat": 1,
        "player_name": "Seat1",
        "position": "UTG+1",
        "action_type": "bet",
        "amount": 12.8,
        "source_image": states[2]["image"],
        "derivation": "bet_text",
    }
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            later, hand, states, db_amount=12.8, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert "put chips in since then" in issue.detail
    assert "not the stack before this action" in issue.detail


def test_bet_box_clause_only_claims_a_box_that_was_read() -> None:
    """B5 round 5 F3: the parenthetical asserted a bet box on frames where the
    client had already swept the bets."""
    hand, states = _cv_issue_fixture()
    swept = dict(hand["actions"][1], source_image=states[2]["image"])
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            swept, hand, states, db_amount=None, db_stack_before=224.2
        )
        if issue.kind == "Amount unknown"
    )
    assert "recorded no bet box for this seat on frame 3" in issue.detail

    # When the reader DID see a box and refuse it, say that instead of
    # claiming nothing was there — its own record contradicts the claim.
    states[2]["bets_unknown"] = {"7": "below_calibrated_render_size"}
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            swept, hand, states, db_amount=None, db_stack_before=224.2
        )
        if issue.kind == "Amount unknown"
    )
    assert "On frame 3, the on-screen text rendered below" in issue.detail
    assert "recorded no bet box" not in issue.detail

    with_box = dict(hand["actions"][1])
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            with_box, hand, states, db_amount=None, db_stack_before=224.2
        )
        if issue.kind == "Amount unknown"
    )
    assert "bet box shows the seat's total for the street" in issue.detail


def test_single_card_is_not_proof_a_seat_is_live() -> None:
    """A5 round 5 F4: deal animations put in-flight board cards in
    villain_cards, attributed to whichever seat they pass."""
    hand, states = _cv_issue_fixture()
    states[1]["dealt_in"] = [0, 2, 4]
    states[1]["villain_cards"] = {"7": ["7h"]}   # one in-flight card
    inferred = {
        "street": "turn",
        "action_index": 2,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "check",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "inferred_round_complete",
    }
    issues = cv_issues_for_timeline_action(
        inferred, hand, states, db_amount=None, db_stack_before=None
    )
    assert any(issue.kind == ACTION_MAY_NOT_BELONG for issue in issues)


def test_malformed_timeline_values_do_not_crash_the_panel() -> None:
    """A5 round 5 F5: unguarded casts took down Import validation rather than
    degrading on a partially-written timeline."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks"] = {"1": "n/a"}
    states[0]["dealt_in"] = ["x", 1]
    states[0]["unmeasured_transitions"] = ["y"]
    action = dict(hand["actions"][0], seat="1")
    issues = cv_issues_for_timeline_action(
        action, hand, states, db_amount=None, db_stack_before=None
    )
    assert isinstance(issues, list)


def test_carrier_frame_is_strictly_before_the_action() -> None:
    """B6 round 6 F5: the bound was inclusive, so 274 of 468 rows named the
    action's own post-action frame."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks"] = {"7": 224.2}
    states[1]["stacks"] = {"7": 224.2}   # also on the action's own frame
    action = dict(hand["actions"][1], stack_before=224.2)
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert issue.frame_index == 0
    assert "frame 1" in issue.detail


def test_phantom_evidence_is_what_the_frames_establish() -> None:
    """B6 round 6 F1: 'may not have been in the hand at all' is disproved by
    frame 1, where the seat is dealt in. The true claim is that it folded."""
    hand, states = _cv_issue_fixture()
    states[0]["dealt_in"] = [0, 2, 4, 7]
    states[1]["dealt_in"] = [0, 2, 4]
    inferred = {
        "street": "turn",
        "action_index": 2,
        "seat": 7,
        "player_name": "Seat7",
        "position": "BB",
        "action_type": "check",
        "amount": None,
        "source_image": "/tmp/f1.jpg",
        "derivation": "inferred_round_complete",
    }
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            inferred, hand, states, db_amount=None, db_stack_before=None
        )
        if issue.kind == ACTION_MAY_NOT_BELONG
    )
    assert "held cards through frame 1" in issue.detail
    assert "had already folded" in issue.detail
    assert "may not have been in the hand at all" not in issue.detail

    # With no card evidence anywhere, the weaker claim is the honest one.
    states[0]["dealt_in"] = [0, 2, 4]
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            inferred, hand, states, db_amount=None, db_stack_before=None
        )
        if issue.kind == ACTION_MAY_NOT_BELONG
    )
    assert "may not have been in the hand at all" in issue.detail


def test_saved_stack_taken_from_the_action_frame_is_flagged() -> None:
    """B7 round 7 H3: the rule that an action's own frame reads the stack
    AFTER the chips moved governed only frames the panel cites. A saved value
    copied from that frame was never checked."""
    hand, states = _cv_issue_fixture()
    states[1]["stacks"] = {"7": 212.2}
    action = dict(hand["actions"][1], stack_before=None)
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=212.2
        )
        if issue.kind == "Stack before looks post-action"
    )
    assert "after this seat's chips moved" in issue.detail
    assert "not before it" in issue.detail
    assert issue.frame_index == 1

    # A pre-action figure is fine and must stay silent.
    assert not [
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=224.2
        )
        if issue.kind == "Stack before looks post-action"
    ]


def test_row_moved_off_its_source_street_is_hedged() -> None:
    """A8/B8 round 8: the hedge compared the frame's board size to the
    TIMELINE's street, so it never fired for an operator street correction
    (1940 of 2004 missed) and every one of its 16 firings was false — a
    closing action is routinely only observable on the next street's first
    frame, which is exactly why the reconstruction read it there."""
    hand, states = _cv_issue_fixture()
    # Unedited: the row sits where it was reconstructed, so no hedge.
    assert not [
        issue
        for issue in cv_issues_for_timeline_action(
            hand["actions"][1], hand, states, db_amount=None, db_stack_before=224.2
        )
        if issue.kind == "Moved off its source street"
    ]
    # Operator moved it to the river: hedge, without claiming the frame
    # cannot show the original action.
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            hand["actions"][1],
            hand,
            states,
            db_amount=None,
            db_stack_before=224.2,
            db_street="river",
        )
        if issue.kind == "Moved off its source street"
    )
    assert "reconstructed on the preflop" in issue.detail
    assert "now saved on the river" in issue.detail
    assert "cannot show this action" not in issue.detail


def test_money_gates_read_the_saved_type_not_the_reconstructed_one() -> None:
    """A8 round 8: 144 rows retyped to a fold were still asked for an amount,
    and 1455 folds retyped to a money type went completely silent."""
    hand, states = _cv_issue_fixture()
    money_row = hand["actions"][0]        # a raise with no amount
    # Retyped to a check: stop demanding an amount.
    assert not [
        issue
        for issue in cv_issues_for_timeline_action(
            money_row,
            hand,
            states,
            db_amount=None,
            db_stack_before=212.2,
            db_action_type="check",
        )
        if issue.kind == "Amount unknown"
    ]
    # A reconstructed check retyped to a bet with an empty amount must flag.
    check_row = dict(money_row, action_type="check", amount=None)
    assert [
        issue
        for issue in cv_issues_for_timeline_action(
            check_row,
            hand,
            states,
            db_amount=None,
            db_stack_before=212.2,
            db_action_type="bet",
        )
        if issue.kind == "Amount unknown"
    ]


def test_timeline_action_by_frame_and_seat_is_edit_proof_but_refuses_ties() -> None:
    """The key that survives every correction, and declines to guess."""
    hand, _states = _cv_issue_fixture()
    found = timeline_action_by_frame_and_seat(hand, "/tmp/f0.jpg", 1)
    assert found is not None
    assert found["action_index"] == 1
    assert timeline_action_by_frame_and_seat(hand, "/tmp/f0.jpg", 99) is None
    assert timeline_action_by_frame_and_seat(hand, None, 1) is None

    # Two lines by one seat on one frame is ambiguous: refuse rather than guess.
    ambiguous = dict(
        hand,
        actions=[*hand["actions"], dict(hand["actions"][0], action_index=9)],
    )
    assert timeline_action_by_frame_and_seat(ambiguous, "/tmp/f0.jpg", 1) is None


def test_malformed_amount_and_stack_degrade_instead_of_raising() -> None:
    """A7 round 7 H2/H3: the previous 'guarded cast' repair turned a
    ValueError into a TypeError, and left one raw float() ahead of the guard."""
    hand, states = _cv_issue_fixture()
    for field, value in (
        ("amount", "unknown"),
        ("amount", []),
        ("stack_before", "n/a"),
        ("stack_before", {}),
        ("stack_before", "1,5"),
    ):
        action = dict(hand["actions"][0], **{field: value})
        issues = cv_issues_for_timeline_action(
            action, hand, states, db_amount=None, db_stack_before=None
        )
        assert isinstance(issues, list)


def test_clearing_a_post_action_stack_does_not_re_offer_it() -> None:
    """B8 round 8: clearing the field produced 'the reconstruction computed
    212.2 ... re-enter it', and re-entering produced 'that is the stack AFTER
    this action'. The two messages looped, and the correct value was never
    named."""
    hand, states = _cv_issue_fixture()
    states[0]["stacks"] = {"1": 212.2}
    action = dict(hand["actions"][0], stack_before=212.2, amount=3.0)
    issue = next(
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=3.0, db_stack_before=None
        )
        if issue.kind == "Stack before unknown"
    )
    assert "AFTER this seat's chips moved" in issue.detail
    assert "not the stack before this action" in issue.detail
    assert "add this action's 3 BB back to it" in issue.detail
    assert "No earlier frame reads that value" not in issue.detail


def test_post_action_claims_require_the_frame_to_show_chips() -> None:
    """A9 round 9 F2: both places that make this claim must check the frame
    shows the seat committing something — that IS the premise. A fold moves no
    chips, so condemning its stack is condemning a correct value."""
    hand, states = _cv_issue_fixture()
    states[1]["stacks"] = {"7": 212.2}
    states[1]["bets"] = {}
    states[1]["bets_unknown"] = {}
    action = dict(hand["actions"][1], stack_before=212.2)
    # Saved-value check.
    assert not [
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=212.2
        )
        if issue.kind == "Stack before looks post-action"
    ]
    # Cleared-field twin inside the stack ladder.
    assert not [
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=None
        )
        if "AFTER this seat's chips moved" in issue.detail
    ]
    # With a bet box present, both fire again.
    states[1]["bets"] = {"7": 12.0}
    assert [
        issue
        for issue in cv_issues_for_timeline_action(
            action, hand, states, db_amount=12.0, db_stack_before=212.2
        )
        if issue.kind == "Stack before looks post-action"
    ]


def test_a_fold_is_never_told_its_stack_is_post_action() -> None:
    """A fold moves no chips even on a frame where the seat has money out from
    an earlier action, so BOTH the saved-value check and the cleared-field
    twin must refuse it."""
    hand, states = _cv_issue_fixture()
    states[1]["stacks"] = {"7": 212.2}
    states[1]["bets"] = {"7": 12.0}
    action = dict(hand["actions"][1], stack_before=212.2)
    assert not [
        issue
        for issue in cv_issues_for_timeline_action(
            action,
            hand,
            states,
            db_amount=None,
            db_stack_before=212.2,
            db_action_type="fold",
        )
        if issue.kind == "Stack before looks post-action"
    ]
    # The cleared-field twin inside the stack ladder.
    assert not [
        issue
        for issue in cv_issues_for_timeline_action(
            action,
            hand,
            states,
            db_amount=None,
            db_stack_before=None,
            db_action_type="fold",
        )
        if "AFTER this seat's chips moved" in issue.detail
    ]


def test_post_action_claim_requires_this_action_to_own_the_box() -> None:
    """B10 round 10 F3: chips in the box prove a commitment, not WHICH one. A
    seat that posted a blind has money out before it acts again, so the
    frame's stack read is not 'after this action'."""
    hand, states = _cv_issue_fixture()
    states[1]["stacks"] = {"7": 212.2}
    states[1]["bets"] = {"7": 1.0}          # the blind post, not this action
    hand = dict(
        hand,
        actions=[
            {
                "street": "preflop", "action_index": 1, "seat": 7,
                "player_name": "Seat7", "position": "BB",
                "action_type": "post_blind", "amount": 1.0,
                "source_image": "/tmp/f1.jpg", "derivation": "action_pill",
            },
            *hand["actions"],
        ],
    )
    later = dict(hand["actions"][2], action_index=5, seat=7, stack_before=212.2)
    assert not [
        issue
        for issue in cv_issues_for_timeline_action(
            later, hand, states, db_amount=12.0, db_stack_before=212.2
        )
        if issue.kind == "Stack before looks post-action"
    ]
