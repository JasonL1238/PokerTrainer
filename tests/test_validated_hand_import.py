"""Validate-then-import: auto full hands vs explicit incomplete drafts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    ProcessingJob,
    ReconstructionFrameReview,
    Session,
    VideoRecord,
)
from poker_tracker.services.validated_hand_import import (
    autonomous_import_blockers,
    ensure_hand_imported,
    hand_frames_validated,
    hand_passes_autonomous_import_gate,
)
from poker_tracker.ui import reconstruction_review


def _spine_hand(**overrides):
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 8.0,
        "n_states": 20,
        "hero": ["As", "Kd"],
        "board": ["2c", "7d", "9h", "Ts", "Jc"],
        "complete_cards": True,
        "warnings": [],
        "players": [
            {
                "seat": 0,
                "position": "SB",
                "player_name": "Hero",
                "starting_stack": 100.0,
                "is_hero": True,
            },
            {
                "seat": 4,
                "position": "BTN",
                "player_name": "Seat4",
                "starting_stack": 100.0,
                "is_hero": False,
            },
        ],
        "actions": [
            {
                "street": "preflop",
                "action_index": 1,
                "seat": 4,
                "position": "BTN",
                "player_name": "Seat4",
                "action_type": "raise",
                "amount": 3.0,
                "pot_before": 0.0,
                "stack_before": 100.0,
            },
            {
                "street": "flop",
                "action_index": 1,
                "seat": 0,
                "position": "SB",
                "player_name": "Hero",
                "action_type": "bet",
                "amount": 7.0,
                "pot_before": 6.0,
                "stack_before": 97.0,
            },
        ],
        "streets": [{"street": s} for s in ("preflop", "flop", "turn", "river")],
        "pot": 20.0,
        "side_pot": None,
        "winner_seat": 0,
        "result": "Hero wins",
        "hero_bb_won": 10.0,
        "hero_folded": False,
        "reconciled": True,
        "amounts_unknown": 0,
        "amounts_rejected": 0,
        "anchor_missing_states": 0,
        "hero_seat_confirmed": True,
        "terminal_event": "showdown",
        "source_images": ["f.jpg"],
    }
    hand.update(overrides)
    return hand


def _make_db(tmp_path: Path) -> PokerDatabase:
    db = PokerDatabase(tmp_path / "tracker.sqlite3")
    db.init_db()
    return db


def _seed_job(
    db: PokerDatabase,
    tmp_path: Path,
    timeline: dict,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, int, Path]:
    session = db.create_session(Session(name="Dest", platform="ClubWPT Gold"))
    video = db.create_video(
        VideoRecord(
            original_filename="clip.mp4",
            stored_path=str(tmp_path / "clip.mp4"),
            file_size_bytes=0,
            session_id=session.id,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id,
            job_type="cv_reconstruction",
            status="completed",
        )
    )
    timelines = tmp_path / "cv_timelines"
    timelines.mkdir(exist_ok=True)
    path = timelines / f"job_{job.id}_timeline.json"
    path.write_text(json.dumps(timeline), encoding="utf-8")
    monkeypatch.setattr(reconstruction_review, "CV_TIMELINES_DIR", timelines)
    monkeypatch.setattr(
        "poker_tracker.services.validated_hand_import.CV_TIMELINES_DIR", timelines
    )
    monkeypatch.setattr(
        "poker_tracker.services.validated_hand_import.DATA_DIR", tmp_path
    )
    return job.id, session.id, timelines


def test_hand_frames_validated_requires_all_correct() -> None:
    hand = {"source_images": ["a.jpg", "b.jpg"]}
    assert (
        hand_frames_validated(
            hand, {"a.jpg": SimpleNamespace(status="correct")}
        )
        is False
    )
    assert (
        hand_frames_validated(
            hand,
            {
                "a.jpg": SimpleNamespace(status="correct"),
                "b.jpg": SimpleNamespace(status="incorrect"),
            },
        )
        is False
    )
    assert (
        hand_frames_validated(
            hand,
            {
                "a.jpg": SimpleNamespace(status="correct"),
                "b.jpg": SimpleNamespace(status="correct"),
            },
        )
        is True
    )


def test_autonomous_gate_blocks_mid_start_and_incomplete(tmp_path: Path) -> None:
    timeline_path = tmp_path / "timeline.json"
    opener = _spine_hand(
        hand_number=1,
        source_images=["a.jpg"],
        terminal_event="showdown",
    )
    incomplete = _spine_hand(
        hand_number=2,
        complete_cards=False,
        hero=["As"],
        board=["2c", "7d"],
        source_images=["b.jpg"],
        terminal_event="showdown",
    )
    full = _spine_hand(
        hand_number=3,
        source_images=["c.jpg"],
        terminal_event="showdown",
    )
    timeline = {
        "states": [
            {"image": "a.jpg", "time_s": 1.0},
            {"image": "b.jpg", "time_s": 2.0},
            {"image": "c.jpg", "time_s": 3.0},
        ],
        "hands": [opener, incomplete, full],
    }
    reviews = {"c.jpg": SimpleNamespace(status="correct")}

    opener_gate = autonomous_import_blockers(
        timeline,
        opener,
        timeline_path=timeline_path,
        reviews_by_image={"a.jpg": SimpleNamespace(status="correct")},
    )
    assert opener_gate.ok is False
    assert any("partial_start" in reason for reason in opener_gate.reasons)

    incomplete_gate = autonomous_import_blockers(
        timeline,
        incomplete,
        timeline_path=timeline_path,
        reviews_by_image={"b.jpg": SimpleNamespace(status="correct")},
    )
    assert incomplete_gate.ok is False

    # Last hand with an observed terminal is still a full hand for the auto gate.
    assert (
        hand_passes_autonomous_import_gate(
            timeline,
            full,
            timeline_path=timeline_path,
            reviews_by_image=reviews,
        )
        is True
    )

    # Interior full hand with observed terminal and correct frames.
    middle_full = _spine_hand(
        hand_number=2,
        source_images=["b.jpg"],
        terminal_event="showdown",
    )
    three = {
        "states": [
            {
                "image": "a.jpg",
                "time_s": 1.0,
                "board_cards": [],
                "hero_cards": ["As", "Kd"],
            },
            {
                "image": "b.jpg",
                "time_s": 2.0,
                "board_cards": ["2c", "7d", "9h"],
                "hero_cards": ["As", "Kd"],
            },
            {
                "image": "c.jpg",
                "time_s": 3.0,
                "board_cards": ["2c", "7d", "9h", "Ts", "Jc"],
                "hero_cards": ["As", "Kd"],
            },
        ],
        "hands": [
            _spine_hand(
                hand_number=1, source_images=["a.jpg"], terminal_event="showdown"
            ),
            middle_full,
            _spine_hand(
                hand_number=3, source_images=["c.jpg"], terminal_event="showdown"
            ),
        ],
    }
    assert (
        hand_passes_autonomous_import_gate(
            three,
            middle_full,
            timeline_path=timeline_path,
            reviews_by_image={"b.jpg": SimpleNamespace(status="correct")},
        )
        is True
    )


def test_ensure_auto_imports_full_hand_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    middle = _spine_hand(
        hand_number=2,
        source_images=["b.jpg"],
        terminal_event="showdown",
        t_start=2.0,
        t_end=5.0,
    )
    timeline = {
        "states": [
            {
                "image": "a.jpg",
                "time_s": 1.0,
                "board_cards": [],
                "hero_cards": ["As", "Kd"],
            },
            {
                "image": "b.jpg",
                "time_s": 3.0,
                "board_cards": ["2c", "7d", "9h", "Ts", "Jc"],
                "hero_cards": ["As", "Kd"],
            },
            {
                "image": "c.jpg",
                "time_s": 6.0,
                "board_cards": ["2c", "7d", "9h", "Ts", "Jc"],
                "hero_cards": ["As", "Kd"],
            },
        ],
        "hands": [
            _spine_hand(
                hand_number=1,
                source_images=["a.jpg"],
                terminal_event="showdown",
                t_start=0.0,
                t_end=1.5,
            ),
            middle,
            _spine_hand(
                hand_number=3,
                source_images=["c.jpg"],
                terminal_event="showdown",
                t_start=5.5,
                t_end=8.0,
            ),
        ],
    }
    db = _make_db(tmp_path)
    job_id, session_id, _ = _seed_job(db, tmp_path, timeline, monkeypatch=monkeypatch)
    db.upsert_reconstruction_frame_review(
        ReconstructionFrameReview(
            job_id=job_id,
            hand_number=2,
            source_image="b.jpg",
            timestamp_seconds=3.0,
            status="correct",
        )
    )

    first = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)
    assert first.status == "imported"
    assert first.hand_id is not None
    hands = db.fetch_hands_by_session(session_id)
    assert len(hands) == 1
    assert "timeline_hand_number=2" in hands[0].notes

    second = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)
    assert second.status == "already_present"
    assert len(db.fetch_hands_by_session(session_id)) == 1
    db.close()


def test_ensure_auto_blocks_incomplete_but_draft_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = _spine_hand(
        hand_number=2,
        complete_cards=False,
        hero=["As"],
        board=["2c"],
        source_images=["b.jpg"],
        terminal_event="unobserved",
        t_start=2.0,
        t_end=4.0,
    )
    timeline = {
        "states": [
            {"image": "a.jpg", "time_s": 1.0, "board_cards": [], "hero_cards": ["As", "Kd"]},
            {"image": "b.jpg", "time_s": 3.0, "board_cards": ["2c"], "hero_cards": ["As"]},
            {"image": "c.jpg", "time_s": 5.0, "board_cards": [], "hero_cards": ["As", "Kd"]},
        ],
        "hands": [
            _spine_hand(hand_number=1, source_images=["a.jpg"], terminal_event="showdown"),
            incomplete,
            _spine_hand(hand_number=3, source_images=["c.jpg"], terminal_event="showdown"),
        ],
    }
    db = _make_db(tmp_path)
    job_id, session_id, _ = _seed_job(db, tmp_path, timeline, monkeypatch=monkeypatch)
    db.upsert_reconstruction_frame_review(
        ReconstructionFrameReview(
            job_id=job_id,
            hand_number=2,
            source_image="b.jpg",
            timestamp_seconds=3.0,
            status="correct",
        )
    )

    blocked = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)
    assert blocked.status == "blocked"
    assert db.fetch_hands_by_session(session_id) == []

    drafted = ensure_hand_imported(db, job_id, 2, mode="draft", data_dir=tmp_path)
    assert drafted.status == "imported"
    hands = db.fetch_hands_by_session(session_id)
    assert len(hands) == 1
    assert hands[0].completion_status in {"partial", "uncertain"}
    assert hands[0].review_status == "needs_correction"
    db.close()


def test_ensure_auto_blocks_when_frame_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    middle = _spine_hand(
        hand_number=2,
        source_images=["b1.jpg", "b2.jpg"],
        terminal_event="showdown",
    )
    timeline = {
        "states": [
            {"image": "a.jpg", "time_s": 1.0},
            {"image": "b1.jpg", "time_s": 2.0},
            {"image": "b2.jpg", "time_s": 3.0},
            {"image": "c.jpg", "time_s": 4.0},
        ],
        "hands": [
            _spine_hand(hand_number=1, source_images=["a.jpg"], terminal_event="showdown"),
            middle,
            _spine_hand(hand_number=3, source_images=["c.jpg"], terminal_event="showdown"),
        ],
    }
    db = _make_db(tmp_path)
    job_id, session_id, _ = _seed_job(db, tmp_path, timeline, monkeypatch=monkeypatch)
    for image, status in (("b1.jpg", "correct"), ("b2.jpg", "incorrect")):
        db.upsert_reconstruction_frame_review(
            ReconstructionFrameReview(
                job_id=job_id,
                hand_number=2,
                source_image=image,
                timestamp_seconds=2.0,
                status=status,
            )
        )

    result = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)
    assert result.status == "blocked"
    assert any("flagged" in reason for reason in result.reasons)
    assert db.fetch_hands_by_session(session_id) == []
    db.close()


def test_unknown_mode_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = {
        "states": [{"image": "a.jpg", "time_s": 1.0}],
        "hands": [
            _spine_hand(hand_number=1, source_images=["a.jpg"], terminal_event="showdown")
        ],
    }
    db = _make_db(tmp_path)
    job_id, session_id, _ = _seed_job(db, tmp_path, timeline, monkeypatch=monkeypatch)
    result = ensure_hand_imported(
        db, job_id, 1, mode="automatic", data_dir=tmp_path  # type: ignore[arg-type]
    )
    assert result.status == "blocked"
    assert any("unknown import mode" in reason for reason in result.reasons)
    assert db.fetch_hands_by_session(session_id) == []
    db.close()


def test_draft_import_is_not_study_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from poker_tracker.services.study_readiness import evaluate_study_readiness

    incomplete = _spine_hand(
        hand_number=2,
        complete_cards=False,
        hero=["As"],
        board=["2c"],
        source_images=["b.jpg"],
        terminal_event="unobserved",
        t_start=2.0,
        t_end=4.0,
    )
    timeline = {
        "states": [
            {"image": "a.jpg", "time_s": 1.0},
            {"image": "b.jpg", "time_s": 3.0},
            {"image": "c.jpg", "time_s": 5.0},
        ],
        "hands": [
            _spine_hand(hand_number=1, source_images=["a.jpg"], terminal_event="showdown"),
            incomplete,
            _spine_hand(hand_number=3, source_images=["c.jpg"], terminal_event="showdown"),
        ],
    }
    db = _make_db(tmp_path)
    job_id, session_id, _ = _seed_job(db, tmp_path, timeline, monkeypatch=monkeypatch)
    drafted = ensure_hand_imported(db, job_id, 2, mode="draft", data_dir=tmp_path)
    assert drafted.status == "imported"
    hand = db.fetch_hand(drafted.hand_id)
    assert hand is not None
    assert hand.study_inclusion == "auto"
    assert hand.review_status == "needs_correction"
    readiness = evaluate_study_readiness(hand, accounting=None)
    assert readiness.is_ready is False
    db.close()


def test_identity_survives_notes_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    middle = _spine_hand(
        hand_number=2,
        source_images=["b.jpg"],
        terminal_event="showdown",
        t_start=2.0,
        t_end=5.0,
    )
    timeline = {
        "states": [
            {"image": "a.jpg", "time_s": 1.0},
            {"image": "b.jpg", "time_s": 3.0},
            {"image": "c.jpg", "time_s": 6.0},
        ],
        "hands": [
            _spine_hand(hand_number=1, source_images=["a.jpg"], terminal_event="showdown"),
            middle,
            _spine_hand(hand_number=3, source_images=["c.jpg"], terminal_event="showdown"),
        ],
    }
    db = _make_db(tmp_path)
    job_id, session_id, _ = _seed_job(db, tmp_path, timeline, monkeypatch=monkeypatch)
    db.upsert_reconstruction_frame_review(
        ReconstructionFrameReview(
            job_id=job_id,
            hand_number=2,
            source_image="b.jpg",
            timestamp_seconds=3.0,
            status="correct",
        )
    )
    first = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)
    assert first.status == "imported"
    hand = db.fetch_hand(first.hand_id)
    assert hand is not None
    db.update_hand_facts(
        hand.model_copy(update={"notes": "Operator wiped provenance notes."}),
        correction_notes="notes only",
    )
    second = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)
    assert second.status == "already_present"
    assert len(db.fetch_hands_by_session(session_id)) == 1
    db.close()
