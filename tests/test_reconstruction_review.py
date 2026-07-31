import json
from pathlib import Path

from poker_tracker.persistence.completion import parse_completion_evidence
from poker_tracker.ui.reconstruction_review import (
    history_impacts,
    key_frames_from_completion_evidence,
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
