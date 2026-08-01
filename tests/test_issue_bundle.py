"""Phase 6: a saved issue must stay reproducible after the models move on.

The frozen evidence snapshot records the symptom. The bundle records what you
would have to put back to see it again — which recording, which frames, which
model weights, which configuration. Without those, "the turn card was wrong" is
unreproducible once the detector is retrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandIssue,
    Session,
    VideoRecord,
)
from poker_tracker.services.issue_bundle import (
    BUNDLE_SCHEMA_VERSION,
    build_issue_bundle,
    identify_artifact,
    serialize_issue_bundle,
)


@pytest.fixture
def seeded(tmp_path: Path):
    db = PokerDatabase(":memory:")
    db.init_db()
    session = db.create_session(Session(name="Bundle"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=4))

    video_path = tmp_path / "session.mov"
    video_path.write_bytes(b"recording-bytes")
    db.create_video(
        VideoRecord(
            session_id=session.id,
            original_filename="session.mov",
            stored_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
            content_sha256="",
            duration_seconds=120.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=3600,
        )
    )

    frame = tmp_path / "t000042.00.jpg"
    frame.write_bytes(b"frame-bytes")
    db.create_action(
        Action(
            hand_id=hand.id,
            street="flop",
            action_index=1,
            action_type="bet",
            player_name="Hero",
            position="BTN",
            amount=5.0,
            source_image=str(frame),
        )
    )

    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["cards"],
            description="Turn read as 3, it was a 9.",
        )
    )
    yield db, issue.id, video_path, frame
    db.close()


def test_bundle_identifies_the_source_recording_without_copying_it(seeded):
    db, issue_id, video_path, _frame = seeded
    bundle = build_issue_bundle(db, issue_id)

    assert bundle["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    recording = bundle["source_recordings"][0]
    assert recording["file"]["path"] == str(video_path)
    assert recording["file"]["present"] is True
    assert len(recording["file"]["sha256"]) == 64
    assert recording["duration_seconds"] == 120.0

    # The bytes themselves are never in the bundle.
    serialized = serialize_issue_bundle(bundle)
    assert "recording-bytes" not in serialized


def test_bundle_records_the_frames_the_actions_came_from(seeded):
    db, issue_id, _video, frame = seeded
    bundle = build_issue_bundle(db, issue_id)
    paths = [entry["path"] for entry in bundle["source_frames"]]
    assert str(frame) in paths


def test_a_missing_artifact_is_reported_as_missing_not_omitted(seeded):
    """Absent and never-recorded must not look the same six months later."""
    db, issue_id, video_path, _frame = seeded
    video_path.unlink()
    bundle = build_issue_bundle(db, issue_id)
    recording = bundle["source_recordings"][0]
    assert recording["file"]["present"] is False
    assert recording["file"]["sha256"] is None
    # The database's own record of what it was survives the file.
    assert recording["original_filename"] == "session.mov"


def test_bundle_carries_model_hashes_and_environment(seeded):
    db, issue_id, _video, _frame = seeded
    bundle = build_issue_bundle(db, issue_id)
    assert set(bundle["models"]) == {"region_detector", "card_classifier"}
    assert "dependencies" in bundle["environment"]
    assert "git" in bundle["environment"]


def test_bundle_redacts_a_credential_pasted_into_free_text(tmp_path: Path):
    """An operator pastes a key into a description; the bundle is shareable."""
    db = PokerDatabase(":memory:")
    db.init_db()
    session = db.create_session(Session(name="Leak"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["coaching"],
            description="Coach failed with api_key=sk-ant-super-secret-value",
        )
    )
    serialized = serialize_issue_bundle(build_issue_bundle(db, issue.id))
    assert "sk-ant-super-secret-value" not in serialized
    assert "<redacted>" in serialized
    db.close()


def test_bundle_is_valid_json(seeded):
    db, issue_id, _video, _frame = seeded
    parsed = json.loads(serialize_issue_bundle(build_issue_bundle(db, issue_id)))
    assert parsed["issue"]["description"].startswith("Turn read as 3")


def test_unknown_issue_is_rejected(seeded):
    db, _issue_id, _video, _frame = seeded
    with pytest.raises(ValueError, match="was not found"):
        build_issue_bundle(db, 9999)


# --- Artifact identity ------------------------------------------------------


def test_large_files_record_size_without_hashing(tmp_path: Path):
    """Rehashing a 12 GB recording for every bundle is not worth the minutes."""
    big = tmp_path / "big.mov"
    big.write_bytes(b"x" * 4096)
    identity = identify_artifact(big, hash_limit_bytes=1024)
    assert identity.present is True
    assert identity.bytes == 4096
    # Unhashed, but the size still catches replacement and truncation.
    assert identity.sha256 is None


def test_small_files_are_hashed(tmp_path: Path):
    small = tmp_path / "small.json"
    small.write_bytes(b"{}")
    identity = identify_artifact(small, hash_limit_bytes=1024)
    assert len(identity.sha256) == 64


@pytest.mark.parametrize("value", [None, "", "   "])
def test_blank_paths_identify_nothing(value):
    assert identify_artifact(value, hash_limit_bytes=1024) is None
