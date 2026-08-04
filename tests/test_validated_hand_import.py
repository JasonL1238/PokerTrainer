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
    find_existing_imported_hand,
    find_imported_hand_in_any_session,
    hand_frames_validated,
    hand_passes_autonomous_import_gate,
    import_all_autonomous_eligible,
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


# ---------------------------------------------------------------------------
# The rollback point has to prove itself on every import, not only on the first
# ---------------------------------------------------------------------------


def _preimport_files(backups: Path, job_id: int) -> list[Path]:
    from poker_tracker.persistence.backup import find_snapshots

    return find_snapshots(backups, purpose="preimport", scope=f"job{job_id}")


def test_a_retained_snapshot_is_reverified_before_every_import(tmp_path: Path) -> None:
    """Existence was standing in for verification after the first call.

    ``ensure_preimport_snapshot`` short-circuited on ``find_snapshots(...)``
    before ``verify_snapshot`` was ever reached, so exactly one hand per job got
    the guarantee its docstring states. Truncate the retained file in between --
    a full mount, a truncating copy, an interrupted rsync -- and the second call
    still returned "proceed" while the only artifact backing that decision was a
    file the product's own verifier grades ``fail``.

    Written against the long-standing ``ensure_preimport_snapshot`` signature on
    purpose: the assertion is about the decision, not about any new API.
    """
    from poker_tracker.maintenance.data_health import verify_snapshot
    from poker_tracker.services.validated_hand_import import ensure_preimport_snapshot

    db = _make_db(tmp_path)
    backups = tmp_path / "backups"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    assert (
        ensure_preimport_snapshot(db, job_id=7, backups=backups, data_dir=data_dir)
        is None
    )
    retained = _preimport_files(backups, 7)
    assert len(retained) == 1
    retained[0].write_bytes(b"")
    assert (
        verify_snapshot(
            retained[0], live_database=Path(db.db_path), data_dir=data_dir
        ).status
        == "fail"
    )

    refusal = ensure_preimport_snapshot(
        db, job_id=7, backups=backups, data_dir=data_dir
    )

    if refusal is None:
        # Proceeding is only honest if SOMETHING retained for this job restores.
        surviving = [
            path
            for path in _preimport_files(backups, 7)
            if verify_snapshot(
                path, live_database=Path(db.db_path), data_dir=data_dir
            ).status
            != "fail"
        ]
        assert surviving, (
            "the import was allowed to proceed against a snapshot the product's "
            "own verifier grades 'fail'"
        )
    db.close()


def test_no_verified_rollback_point_means_no_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot that stops restoring must stop the import, not just the first one.

    The imported hand here is the SECOND of the job, which is precisely the call
    the old existence short-circuit skipped verification on.
    """
    from poker_tracker.maintenance.data_health import CheckResult

    timeline = _four_hand_timeline()
    db = _make_db(tmp_path)
    job_id, session_id, _ = _seed_job(db, tmp_path, timeline, monkeypatch=monkeypatch)
    _mark_frames_correct(db, job_id, (2, "b"), (3, "c"))

    first = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)
    assert first.status == "imported"

    # From here on nothing restores: the mount is gone, the copies are corrupt.
    monkeypatch.setattr(
        "poker_tracker.services.validated_hand_import.verify_snapshot",
        lambda *args, **kwargs: CheckResult(
            "backup_verification", "fail", "did not survive an isolated restore.", ()
        ),
    )
    second = ensure_hand_imported(db, job_id, 3, mode="auto", data_dir=tmp_path)

    assert second.status == "blocked", second
    assert "pre-import snapshot unavailable" in second.reasons
    assert len(db.fetch_hands_by_session(session_id)) == 1
    db.close()


def test_a_snapshot_that_still_restores_is_reused_rather_than_recopied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-verifying must not turn back into one full copy of the database per hand.

    That regression is what the per-job snapshot exists to prevent: eight hands
    meant eight copies, and the rollback point was evicted by the imports it was
    taken to protect.
    """
    db = _make_db(tmp_path)
    job_id, _, _ = _seed_job(
        db, tmp_path, _four_hand_timeline(), monkeypatch=monkeypatch
    )
    _mark_frames_correct(db, job_id, (2, "b"), (3, "c"))

    first = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)
    second = ensure_hand_imported(db, job_id, 3, mode="auto", data_dir=tmp_path)

    assert (first.status, second.status) == ("imported", "imported")
    assert len(_preimport_files(tmp_path / "backups", job_id)) == 1
    db.close()


def test_the_result_names_the_rollback_point_it_was_allowed_to_proceed_against(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller could not tell "verified" from "did not look".

    ``ensure_preimport_snapshot`` returned ``None`` for both, and
    ``HandImportResult`` carried nothing about the snapshot at all, so an import
    with no usable rollback point rendered as an ordinary success.
    """
    db = _make_db(tmp_path)
    job_id, _, _ = _seed_job(
        db, tmp_path, _four_hand_timeline(), monkeypatch=monkeypatch
    )
    _mark_frames_correct(db, job_id, (2, "b"))

    result = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)

    assert result.status == "imported"
    assert result.rollback_point == _preimport_files(tmp_path / "backups", job_id)[0].name
    db.close()


def _four_hand_timeline() -> dict:
    return {
        "states": [
            {"image": f"{letter}.jpg", "time_s": index, "hero_cards": ["As", "Kd"]}
            for index, letter in enumerate("abcd", start=1)
        ],
        "hands": [
            _spine_hand(
                hand_number=number,
                source_images=[f"{letter}.jpg"],
                terminal_event="showdown",
                t_start=float(number),
                t_end=float(number) + 0.5,
            )
            for number, letter in enumerate("abcd", start=1)
        ],
    }


def _mark_frames_correct(
    db: PokerDatabase, job_id: int, *frames: tuple[int, str]
) -> None:
    for number, letter in frames:
        db.upsert_reconstruction_frame_review(
            ReconstructionFrameReview(
                job_id=job_id,
                hand_number=number,
                source_image=f"{letter}.jpg",
                timestamp_seconds=float(number),
                status="correct",
            )
        )


@pytest.mark.parametrize("status", ["queued", "running", "cancelling", "cancelled", "failed"])
def test_only_a_completed_job_can_have_its_timeline_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """The job-status gate belongs here, not only on the review screen.

    Cancelling a reconstruction signals the worker, so its cleanup can be
    skipped and a timeline written moments before the cancel can survive under
    a row that never completed. app.py filters the review surface to completed
    jobs, but this module is reachable without it -- the recovery scans and the
    draft path call in directly -- so an unfinished run's partial reading of the
    recording could land in the study database with nothing saying so.
    """

    db = _make_db(tmp_path)
    job_id, session_id, _ = _seed_job(
        db, tmp_path, _four_hand_timeline(), monkeypatch=monkeypatch
    )
    _mark_frames_correct(db, job_id, (2, "b"))
    db.update_processing_job(job_id, status=status)

    for mode in ("auto", "draft"):
        result = ensure_hand_imported(db, job_id, 2, mode=mode, data_dir=tmp_path)
        assert result.status == "blocked", mode
        assert status in result.message
        assert db.fetch_hands_by_session(session_id) == []

    db.update_processing_job(job_id, status="completed")
    assert ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path).status == (
        "imported"
    )
    db.close()


# ---------------------------------------------------------------------------
# Already-imported is not a per-session fact
# ---------------------------------------------------------------------------


def _move_video_to_new_session(db: PokerDatabase, job_id: int, name: str) -> int:
    """Attach this job's recording to a fresh session, as the attach list does."""
    job = db.fetch_processing_job(job_id)
    assert job is not None
    moved_to = db.create_session(Session(name=name, platform="ClubWPT Gold"))
    assert moved_to.id is not None
    db.update_video_session(job.video_id, moved_to.id)
    return moved_to.id


def test_moving_a_recording_to_another_session_does_not_reimport_its_hands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One real hand, played once, must not become two rows in two sessions.

    The destination is read off ``video.session_id`` at import time, so attaching a
    recording to a second session moves it. The de-duplication probe used to scan
    that destination alone, which reported every hand already imported from the
    recording as absent -- and the import then happily added a second copy of each.
    Both copies count toward their session's results, so the two sessions disagree
    about a hand that happened once and the portfolio totals count it twice.
    """
    db = _make_db(tmp_path)
    job_id, first_session, _ = _seed_job(
        db, tmp_path, _four_hand_timeline(), monkeypatch=monkeypatch
    )
    _mark_frames_correct(db, job_id, (2, "b"))
    assert ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path).status == (
        "imported"
    )
    original = db.fetch_hands_by_session(first_session)
    assert len(original) == 1

    second_session = _move_video_to_new_session(db, job_id, "Attached elsewhere")

    result = ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path)

    assert result.status == "already_present"
    assert db.fetch_hands_by_session(second_session) == [], "the hand was imported twice"
    assert [hand.id for hand in db.fetch_hands_by_session(first_session)] == [
        original[0].id
    ], "the original copy was disturbed"
    # The result names where the hand actually is, not where the import aimed:
    # ensure_draft_for_review treats already_present as success and renders this id.
    assert result.session_id == first_session
    assert result.hand_id == original[0].id
    assert f"session #{first_session}" in result.message
    db.close()


def test_the_render_time_recovery_scan_does_not_duplicate_after_a_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The duplication path needed no button, which is what made it dangerous.

    ``import_all_autonomous_eligible`` runs from the evidence-review page on render.
    Frame verdicts are keyed by JOB, so they survive a recording being attached
    elsewhere and every already-validated hand still passes the autonomous gate --
    the scan therefore re-imported the whole timeline into the new session by itself.
    """
    db = _make_db(tmp_path)
    job_id, first_session, _ = _seed_job(
        db, tmp_path, _four_hand_timeline(), monkeypatch=monkeypatch
    )
    _mark_frames_correct(db, job_id, (1, "a"), (2, "b"), (3, "c"), (4, "d"))
    first_scan = import_all_autonomous_eligible(db, job_id, data_dir=tmp_path)
    imported_first = [item for item in first_scan if item.status == "imported"]
    assert imported_first, "nothing was imported, so the test proves nothing"
    landed = {hand.id for hand in db.fetch_hands_by_session(first_session)}
    assert len(landed) == len(imported_first)

    second_session = _move_video_to_new_session(db, job_id, "Moved mid-review")
    second_scan = import_all_autonomous_eligible(db, job_id, data_dir=tmp_path)

    # The timeline's first and last hands stay blocked in both scans (partial_start /
    # partial_end), so the claim is about what the scan ADDS, not about every status.
    assert "imported" not in [item.status for item in second_scan]
    assert {item.status for item in second_scan} == {"already_present", "blocked"}
    assert db.fetch_hands_by_session(second_session) == []
    assert {hand.id for hand in db.fetch_hands_by_session(first_session)} == landed
    db.close()


def test_the_two_finders_answer_different_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scoped finder must stay scoped; only the dedupe question crosses sessions.

    Four call sites ask "is this hand in the session on screen" to decide what to
    draw -- an in-session badge, an onboarding step, the row beside the frames. If
    the scoped finder started answering yes off another session's copy, each would
    claim a hand this session does not hold. The bug was one question answered with
    the other's scope, so the fix has to keep both available and distinct.
    """
    db = _make_db(tmp_path)
    job_id, first_session, _ = _seed_job(
        db, tmp_path, _four_hand_timeline(), monkeypatch=monkeypatch
    )
    _mark_frames_correct(db, job_id, (2, "b"))
    assert ensure_hand_imported(db, job_id, 2, mode="auto", data_dir=tmp_path).status == (
        "imported"
    )
    elsewhere = _move_video_to_new_session(db, job_id, "Somewhere else")

    scoped = find_existing_imported_hand(
        db, session_id=elsewhere, job_id=job_id, timeline_hand_number=2
    )
    assert scoped is None, "the scoped finder reached outside its session"

    anywhere = find_imported_hand_in_any_session(
        db, job_id=job_id, timeline_hand_number=2
    )
    assert anywhere is not None
    assert anywhere.session_id == first_session

    # A timeline hand that was never imported is absent under both questions.
    assert (
        find_imported_hand_in_any_session(db, job_id=job_id, timeline_hand_number=4)
        is None
    )
    db.close()
