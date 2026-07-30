"""Partial frame-validation progress helpers stay honest under resume."""

from __future__ import annotations

from types import SimpleNamespace

from poker_tracker.ui.reconstruction_review import (
    first_unreviewed_frame_index,
    hand_frame_progress,
    hand_validation_label,
    job_id_from_hand_notes,
)


def test_hand_frame_progress_counts_only_saved_verdicts() -> None:
    hand = {"source_images": ["a.jpg", "b.jpg", "c.jpg"]}
    reviews = {
        "a.jpg": SimpleNamespace(status="correct"),
        "c.jpg": SimpleNamespace(status="incorrect"),
    }
    progress = hand_frame_progress(hand, reviews)
    assert progress == {
        "total": 3,
        "reviewed": 2,
        "remaining": 1,
        "flagged": 1,
    }


def test_hand_frame_progress_prefers_navigable_image_list() -> None:
    hand = {"source_images": ["a.jpg", "ghost.jpg", "c.jpg"]}
    reviews = {"a.jpg": SimpleNamespace(status="correct")}
    progress = hand_frame_progress(
        hand, reviews, countable_images=["a.jpg", "c.jpg"]
    )
    assert progress == {
        "total": 2,
        "reviewed": 1,
        "remaining": 1,
        "flagged": 0,
    }


def test_hand_validation_label_marks_partial_progress() -> None:
    hand = {"hand_number": 2, "hero": ["Ah", "Kd"], "source_images": ["a.jpg", "b.jpg"]}
    label = hand_validation_label(
        hand, {"a.jpg": SimpleNamespace(status="correct")}
    )
    assert "Hand #2" in label
    assert "1/2 validated" in label
    assert "in progress" in label


def test_first_unreviewed_frame_index_resumes_after_saved_prefix() -> None:
    states = [
        {"image": "a.jpg"},
        {"image": "b.jpg"},
        {"image": "c.jpg"},
    ]
    reviews = {"a.jpg": SimpleNamespace(status="correct")}
    assert first_unreviewed_frame_index(states, reviews) == 1


def test_first_unreviewed_frame_index_treats_non_verdict_as_open() -> None:
    states = [{"image": "a.jpg"}, {"image": "b.jpg"}]
    reviews = {"a.jpg": SimpleNamespace(status="unreviewed")}
    assert first_unreviewed_frame_index(states, reviews) == 0


def test_first_unreviewed_frame_index_stays_on_last_when_complete() -> None:
    states = [{"image": "a.jpg"}, {"image": "b.jpg"}]
    reviews = {
        "a.jpg": SimpleNamespace(status="correct"),
        "b.jpg": SimpleNamespace(status="incorrect"),
    }
    assert first_unreviewed_frame_index(states, reviews) == 1


def test_job_id_from_hand_notes_parses_timeline_path() -> None:
    assert (
        job_id_from_hand_notes(
            "CV draft from YOLO card timeline. timeline=/tmp/job_12_timeline.json; t=0..1"
        )
        == 12
    )
    assert job_id_from_hand_notes("manual hand") is None
