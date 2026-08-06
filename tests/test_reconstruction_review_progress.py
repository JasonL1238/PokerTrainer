"""Partial frame-validation progress helpers stay honest under resume."""

from __future__ import annotations

from types import SimpleNamespace

from poker_tracker.ui.reconstruction_review import (
    empty_hands_review_message,
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


def test_empty_hands_review_message_explains_all_nontable_frames() -> None:
    message = empty_hands_review_message(
        {
            "metadata": {"layout_profile": "1052x732-unsupported"},
            "summary": {
                "frames": 1912,
                "table_frames": 0,
                "nontable_frames": 1912,
                "hands": 0,
            },
            "hands": [],
            "states": [],
        }
    )
    assert "non-table" in message
    assert "1052x732-unsupported" in message


def test_empty_hands_review_message_names_the_measured_rejection() -> None:
    """The screen classifier's own tally, not a guess about the window size.

    This message used to tell the operator their client was "below the
    calibrated ClubWPT window size" and to re-record at 1272x896 or larger. It
    said that to a 1344x836 recording -- wider than the size it demanded -- whose
    frames were fine, and it said it purely because the layout carried an
    "-unsupported" suffix. No recommendation the run did not measure belongs here.
    """
    message = empty_hands_review_message(
        {
            "metadata": {
                "layout_profile": "1344x836-unsupported",
                "nontable_reasons": {
                    "no_coin_constellation": 523,
                    "scale_outside_band": 498,
                    "seat_count_out_of_range": 245,
                },
            },
            "summary": {
                "frames": 1021,
                "table_frames": 0,
                "nontable_frames": 1021,
                "hands": 0,
            },
            "hands": [],
            "states": [],
        }
    )
    assert "523" in message, "name the count the classifier actually recorded"
    assert "chip-coin constellation" in message
    assert "1272" not in message and "896" not in message, (
        "a window-size recommendation this run never measured"
    )


def test_empty_hands_review_message_survives_an_unknown_rejection_code() -> None:
    """A code the UI has no phrasing for is still reported, not swallowed."""
    message = empty_hands_review_message(
        {
            "metadata": {"nontable_reasons": {"some_future_check": 7}},
            "summary": {
                "frames": 7,
                "table_frames": 0,
                "nontable_frames": 7,
                "hands": 0,
            },
            "hands": [],
            "states": [],
        }
    )
    assert "some_future_check" in message
    assert "7 of them" in message


def test_empty_hands_review_message_mentions_unsupported_layout() -> None:
    message = empty_hands_review_message(
        {
            "metadata": {"layout_profile": "640x448-unsupported"},
            "summary": {"frames": 10, "hands": 0},
            "hands": [],
            "states": [],
        }
    )
    assert "640x448-unsupported" in message
    assert "no hands" in message.lower()
