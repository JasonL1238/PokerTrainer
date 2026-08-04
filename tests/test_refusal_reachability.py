"""Every refusal's clearing action has to be reachable from the refusal.

Round 6 found the per-hand repair workspace -- cards, players, actions, the
blind and ante declarations, settlement assumptions, source warnings and
debugging issues -- hosted in exactly one place: inside a completed
reconstruction job's frame review. A manually entered hand, or a reconstructed
one whose recording was later deleted, therefore read blockers naming actions
no screen offered, while the delete that produced that state described itself as
leaving hands unaffected.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Hand,
    HandPlayer,
    ProcessingJob,
    ReconstructionFrameReview,
    Session,
    VideoRecord,
)
from poker_tracker.ui.navigation import Page

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _seed_hand_without_a_recording(path: Path, *, review_status: str) -> int:
    """One session, one hand, no video: the state with no reconstruction review."""

    db = PokerDatabase(str(path))
    db.init_db()
    session = db.create_session(Session(name="Orphaned"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="No-limit Hold'em",
            table_size=6,
            hero_position="BTN",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=20,
            review_status=review_status,
            source_type="cv_import",
            completion_status="uncertain",
        )
    )
    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            seat_index=0,
            player_name="Hero",
            position="BTN",
            starting_stack=100,
            is_hero=True,
        )
    )
    hand_id = hand.id
    db.close()
    return hand_id


def _configure(path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()


def _button_labels(app: AppTest) -> list[str]:
    return [str(item.label) for item in app.button]


def test_study_offers_the_repair_route_when_the_session_has_no_recording(
    tmp_path: Path, monkeypatch
) -> None:
    """Study refuses an unapproved hand and names Import. Import has to answer.

    Without a recording there is no reconstruction review, and the review was
    the sole host of every editor a blocker names -- so this screen refused the
    hand, told the operator where to fix it, and offered nothing that went
    there.
    """

    path = tmp_path / "study.db"
    hand_id = _seed_hand_without_a_recording(path, review_status="needs_correction")
    _configure(path, monkeypatch)

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["study_hand_id"] = hand_id
    app.run()
    next(item for item in app.radio if "Study" in list(item.options)).set_value(Page.STUDY)
    app.run()
    assert not list(app.exception), list(app.exception)

    assert any("is not approved for study yet" in str(item.value) for item in app.warning)
    repair = [label for label in _button_labels(app) if label.startswith("Fix hand #1")]
    assert repair, (
        "the refusal named Import validation but offered no control that reaches "
        f"it; buttons were {_button_labels(app)}"
    )


def test_import_hosts_the_pinned_hand_repair_workspace(tmp_path: Path, monkeypatch) -> None:
    """The route the button takes must land on the editors, not on an upload form."""

    path = tmp_path / "import.db"
    hand_id = _seed_hand_without_a_recording(path, review_status="needs_correction")
    _configure(path, monkeypatch)

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["import_repair_hand_id"] = hand_id
    app.run()
    next(item for item in app.radio if "Import" in list(item.options)).set_value(Page.IMPORT)
    app.run()
    assert not list(app.exception), list(app.exception)

    markdown = "\n".join(str(item.value) for item in app.markdown)
    assert "Repairing hand #1" in markdown
    assert "Fix this hand" in markdown
    # The blocker for an undeclared ante mode or blind structure names this
    # tool; if it is not offered here the route leads nowhere useful.
    tools = [option for item in app.selectbox for option in list(item.options)]
    assert "Chip stacks / accounting" in tools, tools
    assert "Back to recordings" in _button_labels(app)


def test_the_pinned_repair_releases_itself_when_the_hand_is_gone(
    tmp_path: Path, monkeypatch
) -> None:
    """A pin to a deleted hand must not strand Import the way Study once was."""

    path = tmp_path / "gone.db"
    _seed_hand_without_a_recording(path, review_status="needs_correction")
    _configure(path, monkeypatch)

    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["import_repair_hand_id"] = 9999
    app.run()
    next(item for item in app.radio if "Import" in list(item.options)).set_value(Page.IMPORT)
    app.run()
    assert not list(app.exception), list(app.exception)

    assert any(
        "no longer exists" in str(item.value) for item in app.warning
    ), [str(item.value) for item in app.warning]
    assert "import_repair_hand_id" not in app.session_state


def _seed_video_with_saved_frame_verdicts(path: Path) -> tuple[int, int, int]:
    db = PokerDatabase(str(path))
    db.init_db()
    session = db.create_session(Session(name="Recorded"))
    video = db.create_video(
        VideoRecord(
            original_filename="clip.mp4",
            stored_path=str(path.parent / "clip.mp4"),
            file_size_bytes=0,
            session_id=session.id,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(video_id=video.id, job_type="cv_reconstruction", status="completed")
    )
    db.upsert_reconstruction_frame_review(
        ReconstructionFrameReview(
            job_id=job.id,
            hand_number=1,
            source_image="a.jpg",
            timestamp_seconds=1.0,
            status="correct",
        )
    )
    ids = (video.id, job.id, session.id)
    db.close()
    return ids


def test_deleting_a_video_destroys_the_saved_frame_verdicts(tmp_path: Path) -> None:
    """The fact the warning has to state, asserted against the database."""

    path = tmp_path / "cascade.db"
    video_id, job_id, _ = _seed_video_with_saved_frame_verdicts(path)
    db = PokerDatabase(str(path))
    assert db.fetch_reconstruction_frame_reviews(job_id)
    db.delete_video(video_id)
    assert db.fetch_processing_job(job_id) is None
    assert db.fetch_reconstruction_frame_reviews(job_id) == []
    db.close()


def test_the_video_delete_warning_states_what_it_actually_removes(
    tmp_path: Path, monkeypatch
) -> None:
    """It used to say hands and sessions were unaffected and stop there.

    True of the rows, false of everything the operator would notice: the jobs
    cascade, the frame verdicts saved against them go with them, and the
    reconstruction review that hosted every per-hand editor disappears.
    """

    path = tmp_path / "danger.db"
    video_id, _, _ = _seed_video_with_saved_frame_verdicts(path)
    _configure(path, monkeypatch)

    script = tmp_path / "_danger_zone.py"
    script.write_text(
        "\n".join(
            [
                "from poker_tracker.persistence.db import PokerDatabase",
                "import app as app_module",
                f"db = PokerDatabase(r'{path}')",
                "db.init_db()",
                f"video = db.fetch_video({video_id})",
                "app_module.show_video_jobs_and_frames(db, video)",
            ]
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not list(app.exception), list(app.exception)

    warnings = "\n".join(str(item.value) for item in app.warning)
    assert "frame verdicts" in warnings, warnings
    assert "reconstruction jobs" in warnings, warnings
    assert "Hands and sessions are unaffected." not in warnings, warnings
    # This delete now snapshots, like its siblings, so "No rollback snapshot is
    # written" would be the false sentence. What it must still not do is inherit
    # their promise: a snapshot copies the DATABASE, and this deletion's whole
    # payload is files. Restoring it returns the rows and none of the recording,
    # so the warning has to separate the two rather than say "a snapshot is
    # written first" and stop, which is what the siblings can afford to say.
    assert "No rollback snapshot" not in warnings, warnings
    assert "snapshot is written first" in warnings, warnings
    assert "rows can be brought back" in warnings, warnings
    assert "the recording and its frames cannot" in warnings, warnings
    # The hands now go with the recording, so the warning that once promised they
    # were kept has to say the opposite -- and has to say which hands are spared,
    # because "every hand" would be false of a manually entered one.
    assert "hand reconstructed from this recording is deleted too" in warnings, warnings
    assert "entered by hand are not touched" in warnings, warnings


def _child_worker_script(
    *, repo: Path, db_path: Path, data_dir: Path, job_id: int, ready: Path
) -> str:
    """A worker that produces this run's artifacts and then waits to be signalled."""

    return "\n".join(
        [
            "import os, sys, time",
            f"os.environ['POKER_DB_PATH'] = r'{db_path}'",
            f"os.environ['POKER_DATA_DIR'] = r'{data_dir}'",
            f"sys.path.insert(0, r'{repo}')",
            "from pathlib import Path",
            "from poker_tracker.ui import run_cv_job",
            "run_cv_job.resolve_stored_video_path = lambda p, **k: Path(p)",
            "run_cv_job.assert_stored_video_matches_record = lambda *a, **k: None",
            "run_cv_job.require_playable_video = lambda *a, **k: None",
            "def fake(command, db, job_id, deadline, progress_path, limits=None):",
            "    frame_dir = Path(command[command.index('--frame-dir') + 1])",
            "    out = Path(command[command.index('--out') + 1])",
            "    frame_dir.mkdir(parents=True, exist_ok=True)",
            "    (frame_dir / 't000000.00.jpg').write_bytes(b'x')",
            "    out.write_text('{}', encoding='utf-8')",
            f"    Path(r'{ready}').write_text('ready', encoding='utf-8')",
            "    time.sleep(120)",
            "run_cv_job._run_pipeline = fake",
            "raise SystemExit(run_cv_job.run_job(",
            f"    job_id={job_id},",
            "    video_path=Path(r'%s')," % (data_dir / "clip.mp4"),
            "    session_name='Cancelled',",
            f"    db_path=Path(r'{db_path}'),",
            "))",
        ]
    )


def test_a_signalled_worker_still_discards_its_partial_artifacts(tmp_path: Path) -> None:
    """Cancelling sends SIGTERM, and SIGTERM used to skip the cleanup entirely.

    ``_discard_partial_artifacts`` names the cancel path in its own docstring,
    but CPython's default disposition for SIGTERM exits without unwinding, so
    the ``finally`` that calls it never ran: a cancelled reconstruction left its
    partial timeline and one JPEG per sampled second behind, and the timeline is
    the artifact a later reader can mistake for a finished one.
    """

    import os
    import signal
    import subprocess
    import sys
    import time

    repo = Path(__file__).resolve().parent.parent
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "clip.mp4").write_bytes(b"not really a video")
    db_path = tmp_path / "worker.db"

    db = PokerDatabase(str(db_path))
    db.init_db()
    session = db.create_session(Session(name="Cancelled"))
    video = db.create_video(
        VideoRecord(
            original_filename="clip.mp4",
            stored_path=str(data_dir / "clip.mp4"),
            file_size_bytes=18,
            session_id=session.id,
            duration_seconds=5.0,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(video_id=video.id, job_type="cv_reconstruction", status="running")
    )
    job_id = job.id
    db.close()

    ready = tmp_path / "ready.txt"
    script = tmp_path / "_worker.py"
    script.write_text(
        _child_worker_script(
            repo=repo, db_path=db_path, data_dir=data_dir, job_id=job_id, ready=ready
        ),
        encoding="utf-8",
    )
    child = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not ready.is_file():
            time.sleep(0.05)
        assert ready.is_file(), "the stub pipeline never produced its artifacts"
        frame_dir = data_dir / "frames" / f"cv_job_{job_id}"
        timeline = data_dir / "cv_timelines" / f"job_{job_id}_timeline.json"
        assert frame_dir.is_dir() and timeline.is_file()

        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)

    assert not timeline.exists(), "a cancelled run left a timeline a later reader can load"
    assert not frame_dir.exists(), "a cancelled run left its sampled frames on disk"
    assert not (data_dir / "cv_timelines" / f"job_{job_id}_progress.json").exists()


def _seed_reviewed_hand(path: Path, *, ready: bool) -> int:
    """A hand marked reviewed, with or without a trust blocker under it."""

    from poker_tracker.persistence.models import (
        Action,
        CoachingResponse,
        HandSettlement,
        SettlementEntry,
    )
    from poker_tracker.services.hand_accounting import persist_reconciliation

    db = PokerDatabase(str(path))
    db.init_db()
    session = db.create_session(Session(name="Reviewed"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="No-limit Hold'em",
            table_size=6,
            hero_position="BTN",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=20,
            hero_bb_won=10,
            review_status="reviewed",
            source_type="manual",
            completion_status="not_applicable",
        )
    )
    hero = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            seat_index=0,
            player_name="Hero",
            position="BTN",
            starting_stack=100,
            is_hero=True,
        )
    )
    villain = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="villain",
            seat_index=1,
            player_name="Villain",
            position="BB",
            starting_stack=100,
        )
    )
    for actor, action_type in ((hero, "bet"), (villain, "call")):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=actor.player_key,
                player_name=actor.player_name,
                position=actor.position,
                street="river",
                action_type=action_type,
                amount=10,
                amount_semantics="incremental",
            )
        )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled", rake_rate=0.0))
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key=hero.player_key,
                player_name=hero.player_name,
                amount=20,
                entry_order=1,
            )
        ],
    )
    persist_reconciliation(db, hand.id)
    # Promoted while it reconciled, exactly as the product does it. The store's
    # own floor cannot see accounting, so the blocked variant below is produced
    # the way an operator produces it: by editing the hand afterwards.
    db.update_hand_status(hand.id, "reviewed")
    if not ready:
        # A stale coaching row blocks study and does not demote the hand, which
        # is exactly how "reviewed with blockers" arises in the product: the
        # v20 migration stales retained coaching in place.
        db.create_coaching_response(
            CoachingResponse(
                provider_name="test",
                model_name="test",
                raw_prompt="p",
                raw_response="r",
                review_type="hand",
                hand_id=hand.id,
                session_id=session.id,
                is_stale=True,
                stale_reason="Underlying evidence changed.",
            )
        )
    hand_id = hand.id
    db.close()
    return hand_id


def test_the_draft_header_does_not_approve_a_hand_the_panel_below_refuses(
    tmp_path: Path,
) -> None:
    """One screen, two statements about the same hand, one rule.

    The header read ``review_status`` alone, so a hand marked reviewed that
    later picked up a blocker was captioned "Approved for Study." above the fix
    panel listing what was blocking it.
    """

    import app as app_module

    blocked_path = tmp_path / "blocked.db"
    blocked_id = _seed_reviewed_hand(blocked_path, ready=False)
    db = PokerDatabase(str(blocked_path))
    caption = app_module.draft_review_caption(db, db.fetch_hand(blocked_id))
    db.close()
    assert "Approved for Study." not in caption, caption
    assert "Marked reviewed" in caption and "trust check(s)" in caption, caption

    ready_path = tmp_path / "ready.db"
    ready_id = _seed_reviewed_hand(ready_path, ready=True)
    db = PokerDatabase(str(ready_path))
    ready_hand = db.fetch_hand(ready_id)
    ready_caption = app_module.draft_review_caption(db, ready_hand)
    db.close()
    # The positive control: the header must still be able to say it, or the
    # repair would have replaced a false claim with a useless one.
    assert ready_caption == "Approved for Study.", ready_caption


def test_the_coaching_empty_state_names_the_cause_that_was_recorded(tmp_path: Path) -> None:
    """A rejected answer is not a changed hand, and the row records which it was.

    ``is_stale`` carries two causes. This line named a hand change
    unconditionally, so a review its own grounding check rejected sent the
    operator looking for a correction to undo, contradicting the readiness
    blocker printed on the same page.
    """

    from datetime import UTC, datetime
    from types import SimpleNamespace

    import app as app_module
    from poker_tracker.coaching.grounding import UNGROUNDED_STALE_PREFIX

    def review(reason: str, minute: int):
        return SimpleNamespace(
            stale_reason=reason,
            created_at=datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
        )

    assert app_module._no_current_coaching_detail([]) == (
        "Generate a post-session review below."
    )

    ungrounded = app_module._no_current_coaching_detail(
        [review(UNGROUNDED_STALE_PREFIX + "invented Kd.", 1)]
    )
    assert "asserted facts its own prompt does not support" in ungrounded, ungrounded
    assert "changed" not in ungrounded, ungrounded

    changed = app_module._no_current_coaching_detail(
        [review("Underlying evidence changed.", 1)]
    )
    assert "changed" in changed, changed

    # The governing row is the newest stale one, the same rule the readiness
    # blocker uses, so a later rejection is not described by an older cause.
    mixed = app_module._no_current_coaching_detail(
        [
            review("Underlying evidence changed.", 1),
            review(UNGROUNDED_STALE_PREFIX + "invented Kd.", 5),
        ]
    )
    assert "asserted facts its own prompt does not support" in mixed, mixed

    unrecorded = app_module._no_current_coaching_detail([review("", 1)])
    assert "does not record what made it stale" in unrecorded, unrecorded


def test_the_completed_job_says_what_window_it_asked_the_sampler_for(
    tmp_path: Path,
) -> None:
    """A run that read part of the recording used to read like one that read all of it.

    Neither the timeline summary, the export summary nor the job message
    carried the sampling window, and the window is derived from a duration
    probe the codebase itself distrusts. When ingest measured no duration at
    all the bound is a 24-hour ceiling, and the row has to say so rather than
    let a reader take it for the recording's length.
    """

    from pathlib import Path as _Path

    import poker_tracker.ui.run_cv_job as worker

    data_dir = tmp_path / "data"
    (data_dir / "cv_timelines").mkdir(parents=True)
    (data_dir / "exports").mkdir(parents=True)
    (data_dir / "backups").mkdir(parents=True)
    (data_dir / "frames").mkdir(parents=True)
    (data_dir / "clip.mp4").write_bytes(b"not really a video")
    db_path = tmp_path / "window.db"

    db = PokerDatabase(str(db_path))
    db.init_db()
    session = db.create_session(Session(name="Window"))

    def _job_for(duration: float | None) -> int:
        video = db.create_video(
            VideoRecord(
                original_filename="clip.mp4",
                stored_path=str(data_dir / "clip.mp4"),
                file_size_bytes=18,
                session_id=session.id,
                duration_seconds=duration,
            )
        )
        return db.create_processing_job(
            ProcessingJob(
                video_id=video.id, job_type="cv_reconstruction", status="running"
            )
        ).id

    measured_job = _job_for(1911.0)
    unmeasured_job = _job_for(None)
    db.close()

    def _run(monkeypatched_job: int) -> str:
        original = {
            name: getattr(worker, name)
            for name in (
                "resolve_stored_video_path",
                "assert_stored_video_matches_record",
                "require_playable_video",
                "_run_pipeline",
                "export_timeline",
                "ensure_data_directories",
                "_verified_backup_note",
                "backup_database",
            )
        }
        worker.resolve_stored_video_path = lambda p, **k: _Path(p)
        worker.assert_stored_video_matches_record = lambda *a, **k: None
        worker.require_playable_video = lambda *a, **k: None
        worker.ensure_data_directories = lambda *a, **k: {
            "data": str(data_dir),
            "cv_timelines": str(data_dir / "cv_timelines"),
            "exports": str(data_dir / "exports"),
            "backups": str(data_dir / "backups"),
            "frames": str(data_dir / "frames"),
        }
        worker._run_pipeline = lambda *a, **k: None
        worker.export_timeline = lambda *a, **k: {
            "cv_import_summary": {"exported_hands": 1}
        }
        worker.backup_database = lambda *a, **k: data_dir / "backups" / "snap.db"
        worker._verified_backup_note = lambda *a, **k: "backup verified"
        try:
            assert (
                worker.run_job(
                    job_id=monkeypatched_job,
                    video_path=data_dir / "clip.mp4",
                    session_name="Window",
                    db_path=db_path,
                )
                == 0
            )
        finally:
            for name, value in original.items():
                setattr(worker, name, value)
        reopened = PokerDatabase(str(db_path))
        message = reopened.fetch_processing_job(monkeypatched_job).message
        reopened.close()
        return message

    measured = _run(measured_job)
    assert "sampling window requested 0–1911s at 1s" in measured, measured

    unmeasured = _run(unmeasured_job)
    assert "duration was never measured" in unmeasured, unmeasured
    assert "ceiling rather than the recording's length" in unmeasured, unmeasured


def _misattribution_context(seat_by_player_key: dict[str, int]):
    """Two seats, one frame: seat 3 is live on it, seat 4 has already folded."""

    from poker_tracker.ui.reconstruction_review import ValidationFrameContext

    frame = "/frames/cv_job_1/t000005.00.jpg"
    timeline_hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "warnings": [],
        "players": [
            {"seat": 3, "player_name": "Seat3", "position": "UTG", "starting_stack": 100.0},
            {"seat": 4, "player_name": "Seat4", "position": "MP", "starting_stack": 100.0},
        ],
        "actions": [
            {
                "street": "flop",
                "action_index": 1,
                "seat": 3,
                "player_name": "Seat3",
                "position": "UTG",
                "action_type": "bet",
                "amount": 7.0,
                "source_image": frame,
                "derivation": "action_pill",
            }
        ],
    }
    states = [
        {
            "state_index": 0,
            "time_s": 5.0,
            "image": frame,
            "board_cards": ["2c", "7d", "9h"],
            # Seat 4 is not dealt in on this frame; seat 3 is.
            "dealt_in": [3],
            "stacks": {"3": 93.0},
            "bets": {"3": 7.0},
            "bets_unknown": {},
            "stacks_unknown": {},
            "unmeasured_transitions": [],
        },
        {
            # A later frame, so the row's own frame is not the hand's last one:
            # absence on a terminal frame is not evidence and is guarded already.
            "state_index": 1,
            "time_s": 6.0,
            "image": frame.replace("t000005", "t000006"),
            "board_cards": ["2c", "7d", "9h"],
            "dealt_in": [3],
            "stacks": {"3": 93.0},
            "bets": {"3": 7.0},
            "bets_unknown": {},
            "stacks_unknown": {},
            "unmeasured_transitions": [],
        },
    ]
    return ValidationFrameContext(
        job_id=1,
        hand_number=1,
        timeline_hand=timeline_hand,
        states=states,
        reviews_by_image={},
        cursor_key="c",
        pending_hand_key="p",
        recording_start_s=0.0,
        seat_by_player_key=seat_by_player_key,
    ), frame


def test_a_renamed_reseated_row_is_not_explained_with_another_seats_frame(
    tmp_path: Path,
) -> None:
    """One player correction used to move a row's frame evidence to another seat.

    ``update_hand_player`` writes the new name AND the new position onto every
    one of that seat's action rows in the same submission. The seat behind every
    frame-level claim was then resolved by matching those rewritten fields back
    into the frozen timeline roster: a row renamed off the roster and re-seated
    onto another timeline player's position resolved to that player's seat, and
    the row was explained -- and told to delete itself -- from a seat that is
    not the one that acted.
    """

    import app as app_module
    from poker_tracker.persistence.models import Action
    from poker_tracker.ui.reconstruction_review import ACTION_MAY_NOT_BELONG

    context, frame = _misattribution_context({"seat:3": 3})
    row = Action(
        hand_id=1,
        player_key="seat:3",
        street="flop",
        action_index=1,
        # Both fields as `update_hand_player` leaves them after one correction:
        # a name that is on no roster entry, and a position that is on another.
        player_name="Bob",
        position="MP",
        action_type="bet",
        amount=7.0,
        stack_before=100.0,
        source_image=frame,
    )

    assert app_module._seat_index_for_action(row, context) == 3
    issues = app_module._cv_issues_for_db_action(row, context)
    kinds = {issue.kind for issue in issues}
    # Seat 4 folded before this frame and seat 3 is holding cards on it, so
    # resolving to seat 4 produced the module's single most destructive
    # message about a line that is entirely sound.
    assert ACTION_MAY_NOT_BELONG not in kinds, [
        (issue.kind, issue.detail) for issue in issues
    ]
    assert "Seat not in the hand on this frame" not in kinds, kinds

    # The stored seat is what makes it right; without one the resolver has only
    # the rewritten fields, and it must refuse rather than name seat 4.
    unstored, _ = _misattribution_context({})
    assert app_module._seat_index_for_action(row, unstored) is None
    assert ACTION_MAY_NOT_BELONG not in {
        issue.kind for issue in app_module._cv_issues_for_db_action(row, unstored)
    }
