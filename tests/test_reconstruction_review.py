import json
from pathlib import Path

from poker_tracker.persistence.completion import parse_completion_evidence
from poker_tracker.ui.reconstruction_review import (
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
        "Preflop · BTN Seat4 · raise 3 BB",
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
    assert "enter it in the Amount field" in amount_issue.detail
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
    assert "12.8 BB in this seat's bet box" in amount_issue.detail
    assert "could not be isolated" in amount_issue.detail
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
    assert "inferred from how the betting round completed" in amount_issue.detail
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
    assert "no digit run" in stack_issue.detail
    # Jump goes to the frame with the refusal, not the action's source frame.
    assert stack_issue.frame_index == 0
    # A readable stack closer to the action stops the scan: no story then.
    states[1]["stacks"] = {"7": 224.2}
    issues = cv_issues_for_timeline_action(
        later_action, hand, states, db_amount=4.0, db_stack_before=None
    )
    assert all(issue.kind != "Stack before unknown" for issue in issues)


def test_inferred_stack_issue_points_at_visible_stack() -> None:
    """B2 round 2: when the jump-target frame shows the seat's stack, say so
    and tell the operator to enter it."""
    hand, states = _cv_issue_fixture()
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
    stack_issue = next(
        issue for issue in issues if issue.kind == "Stack before unknown"
    )
    assert "shows this seat at 269.1 BB" in stack_issue.detail
    assert "enter it below" in stack_issue.detail


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
    assert "because the action's own amount is unknown" in stack_issue.detail
    assert "Resolve the amount first" in stack_issue.detail
