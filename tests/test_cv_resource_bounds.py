"""What bounds a CV reconstruction, and what it says when a bound is hit.

The rule these tests exist to hold is the same one the solver side already
holds: a limit the operator asked for and did not get must never be reported as
applied, and a run that ends on a limit has to say which limit, where that limit
comes from, and what happened to the work in progress. A job row that stops at
"82%" with "timed out" is a progress reading, not an outcome.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Hand,
    ProcessingJob,
    Session,
    SolverRun,
    VideoRecord,
)
from poker_tracker.runtime import limits as runtime_limits
from poker_tracker.ui import frame_extraction, run_cv_job, video_ingest


@pytest.fixture(autouse=True)
def clear_cv_limit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test inherits a bound from the machine it happens to run on."""
    for variable in (
        run_cv_job.CV_TIMEOUT_ENV_VAR,
        run_cv_job.CV_MEMORY_ENV_VAR,
        frame_extraction.FRAME_CEILING_ENV_VAR,
    ):
        monkeypatch.delenv(variable, raising=False)


def _make_db(path: Path) -> PokerDatabase:
    db = PokerDatabase(str(path))
    db.init_db()
    return db


def _queued_job(db: PokerDatabase, tmp_path: Path) -> ProcessingJob:
    recording = tmp_path / "session.mp4"
    recording.write_bytes(b"completed-session-video")
    video = db.create_video(
        VideoRecord(
            original_filename=recording.name,
            stored_path=str(recording),
            file_size_bytes=recording.stat().st_size,
        )
    )
    return db.create_processing_job(
        ProcessingJob(
            video_id=video.id,
            job_type="cv_reconstruction",
            status="running",
        )
    )


def _worker_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "cv_timelines": tmp_path / "timelines",
        "exports": tmp_path / "exports",
        "backups": tmp_path / "backups",
        "frames": tmp_path / "frames",
        "data": tmp_path,
    }
    for path in paths.values():
        path.mkdir(exist_ok=True)
    return paths


def _stub_worker_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Path]:
    """Everything past the pipeline, so a test can end at the bound it is about."""
    paths = _worker_paths(tmp_path)
    monkeypatch.setattr(run_cv_job, "ensure_data_directories", lambda: paths)
    monkeypatch.setattr(
        run_cv_job, "assert_stored_video_matches_record", lambda *a, **k: None
    )
    monkeypatch.setattr(run_cv_job, "require_playable_video", lambda *a, **k: None)
    monkeypatch.setattr(
        run_cv_job, "resolve_stored_video_path", lambda path, **kwargs: Path(path)
    )
    monkeypatch.setattr(
        run_cv_job,
        "export_timeline",
        lambda *a, **k: {"cv_import_summary": {"exported_hands": 1}},
    )
    return paths


# --- The two documented variables -------------------------------------------


def test_timeout_variable_sets_the_deadline_the_worker_actually_runs_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(run_cv_job.CV_TIMEOUT_ENV_VAR, "900")
    db_path = tmp_path / "tracker.sqlite3"
    db = _make_db(db_path)
    job = _queued_job(db, tmp_path)
    db.close()
    _stub_worker_dependencies(monkeypatch, tmp_path)
    seen: dict[str, object] = {}

    def capture(command, db_arg, job_id, deadline, progress_path, limits):
        seen["deadline"] = deadline
        seen["limits"] = limits

    monkeypatch.setattr(run_cv_job, "_run_pipeline", capture)

    before = time.monotonic()
    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(tmp_path / "session.mp4"),
        session_name="Bounded",
        db_path=db_path,
    )

    assert exit_code == 0
    assert seen["limits"].timeout_seconds == 900
    assert before + 899 <= seen["deadline"] <= time.monotonic() + 900


def test_an_explicit_cli_timeout_still_wins_over_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(run_cv_job.CV_TIMEOUT_ENV_VAR, "900")
    assert run_cv_job.resolve_limits(300).timeout_seconds == 300
    assert run_cv_job.resolve_limits().timeout_seconds == 900
    monkeypatch.delenv(run_cv_job.CV_TIMEOUT_ENV_VAR)
    assert (
        run_cv_job.resolve_limits().timeout_seconds
        == run_cv_job.DEFAULT_TIMEOUT_SECONDS
    )


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("POKERTRAINER_CV_TIMEOUT_SECONDS", "one hour"),
        ("POKERTRAINER_CV_TIMEOUT_SECONDS", "30"),
        ("POKERTRAINER_CV_TIMEOUT_SECONDS", "9999999"),
        ("POKERTRAINER_CV_MEMORY_GB", "8GB"),
        ("POKERTRAINER_CV_MEMORY_GB", "0"),
        ("POKERTRAINER_CV_MEMORY_GB", "-4"),
    ],
)
def test_a_misconfigured_bound_fails_the_job_naming_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    """The bound is validated before the recording is opened, not after.

    Silently falling back to the default would leave the operator believing the
    value they set is in force, which is the same defect as a cap that could not
    be installed.
    """
    monkeypatch.setenv(variable, value)
    db_path = tmp_path / "tracker.sqlite3"
    db = _make_db(db_path)
    job = _queued_job(db, tmp_path)
    db.close()
    _stub_worker_dependencies(monkeypatch, tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("The pipeline must not start under a bad bound.")

    monkeypatch.setattr(run_cv_job, "_run_pipeline", refuse)

    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(tmp_path / "session.mp4"),
        session_name="Bounded",
        db_path=db_path,
    )

    checked = PokerDatabase(str(db_path))
    saved = checked.fetch_processing_job(job.id)
    checked.close()
    assert exit_code == 1
    assert saved.status == "failed"
    assert variable in (saved.error_message or "")


def test_the_memory_variable_puts_a_cap_on_the_pipeline_and_nothing_else_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict] = []

    def fake_popen(command, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(pid=4321, poll=lambda: 0, returncode=0, wait=lambda **k: 0)

    monkeypatch.setattr(run_cv_job.subprocess, "Popen", fake_popen)
    progress_path = tmp_path / "progress.json"

    run_cv_job._run_pipeline(
        ["true"], None, 1, time.monotonic() + 60, progress_path, run_cv_job.resolve_limits()
    )
    assert captured[-1]["preexec_fn"] is None

    monkeypatch.setenv(run_cv_job.CV_MEMORY_ENV_VAR, "6")
    run_cv_job._run_pipeline(
        ["true"], None, 1, time.monotonic() + 60, progress_path, run_cv_job.resolve_limits()
    )
    assert captured[-1]["preexec_fn"] is not None


def test_a_cap_that_cannot_be_installed_stops_the_job_instead_of_running_uncapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS refuses RLIMIT_AS outright, and that must not become a silent bypass."""
    monkeypatch.setenv(run_cv_job.CV_MEMORY_ENV_VAR, "6")

    def refusing_popen(command, **kwargs):
        raise subprocess.SubprocessError("Exception occurred in preexec_fn.")

    monkeypatch.setattr(run_cv_job.subprocess, "Popen", refusing_popen)

    with pytest.raises(RuntimeError) as failure:
        run_cv_job._run_pipeline(
            ["true"],
            None,
            1,
            time.monotonic() + 60,
            tmp_path / "progress.json",
            run_cv_job.resolve_limits(),
        )
    message = str(failure.value)
    assert run_cv_job.CV_MEMORY_ENV_VAR in message
    assert "6 GB" in message
    assert "no reconstruction was started" in message


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Darwin refuses setrlimit(RLIMIT_AS)."
)
def test_the_limiter_hook_really_caps_address_space_where_the_platform_allows_it(
    tmp_path: Path,
) -> None:
    cap = 3 * 1024**3
    completed = subprocess.run(
        [sys.executable, "-c", "import resource;print(resource.getrlimit(resource.RLIMIT_AS)[0])"],
        preexec_fn=runtime_limits.memory_limiter(cap, variable="POKERTRAINER_CV_MEMORY_GB"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert int(completed.stdout.strip()) == cap


# --- What a bound says when it is hit ---------------------------------------


def test_the_timeout_message_states_the_outcome_the_limit_and_the_variable() -> None:
    with pytest.raises(TimeoutError) as expired:
        run_cv_job._check_deadline(time.monotonic() - 1, 900)
    message = str(expired.value)
    assert "900-second" in message
    assert run_cv_job.CV_TIMEOUT_ENV_VAR in message
    assert "No hands were exported" in message
    assert "discarded" in message
    # A terminal state states its outcome, never its last progress reading.
    assert "%" not in message


def test_a_timed_out_job_discards_its_partial_timeline_and_review_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(run_cv_job.CV_TIMEOUT_ENV_VAR, "120")
    db_path = tmp_path / "tracker.sqlite3"
    db = _make_db(db_path)
    job = _queued_job(db, tmp_path)
    db.close()
    paths = _stub_worker_dependencies(monkeypatch, tmp_path)
    frame_dir = paths["frames"] / f"cv_job_{job.id}"
    timeline_path = paths["cv_timelines"] / f"job_{job.id}_timeline.json"

    def expire(command, db_arg, job_id, deadline, progress_path, limits):
        frame_dir.mkdir(parents=True, exist_ok=True)
        (frame_dir / "frame_000001.jpg").write_bytes(b"jpeg")
        timeline_path.write_text("{}", encoding="utf-8")
        run_cv_job._check_deadline(time.monotonic() - 1, limits.timeout_seconds)

    monkeypatch.setattr(run_cv_job, "_run_pipeline", expire)

    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(tmp_path / "session.mp4"),
        session_name="Bounded",
        db_path=db_path,
    )

    checked = PokerDatabase(str(db_path))
    saved = checked.fetch_processing_job(job.id)
    checked.close()
    assert exit_code == 1
    assert saved.status == "failed"
    assert "120-second" in (saved.error_message or "")
    assert run_cv_job.CV_TIMEOUT_ENV_VAR in (saved.error_message or "")
    assert not frame_dir.exists()
    assert not timeline_path.exists()


def test_a_completed_job_keeps_the_review_frames_its_hands_point_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracker.sqlite3"
    db = _make_db(db_path)
    job = _queued_job(db, tmp_path)
    db.close()
    paths = _stub_worker_dependencies(monkeypatch, tmp_path)
    frame_dir = paths["frames"] / f"cv_job_{job.id}"

    def succeed(command, db_arg, job_id, deadline, progress_path, limits):
        frame_dir.mkdir(parents=True, exist_ok=True)
        (frame_dir / "frame_000001.jpg").write_bytes(b"jpeg")

    monkeypatch.setattr(run_cv_job, "_run_pipeline", succeed)

    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(tmp_path / "session.mp4"),
        session_name="Bounded",
        db_path=db_path,
    )

    assert exit_code == 0
    assert (frame_dir / "frame_000001.jpg").exists()


def test_a_nonzero_exit_under_a_cap_names_the_cap_without_blaming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(run_cv_job.CV_MEMORY_ENV_VAR, "6")

    def fake_popen(command, **kwargs):
        return SimpleNamespace(pid=4321, poll=lambda: 1, returncode=1, wait=lambda **k: 1)

    monkeypatch.setattr(run_cv_job.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError) as failure:
        run_cv_job._run_pipeline(
            ["true"],
            None,
            1,
            time.monotonic() + 60,
            tmp_path / "progress.json",
            run_cv_job.resolve_limits(),
        )
    message = str(failure.value)
    assert "exited with code 1" in message
    assert run_cv_job.CV_MEMORY_ENV_VAR in message
    assert "6 GB" in message
    # The cap is where to look, not a diagnosis: RLIMIT_AS is indistinguishable
    # from any other allocation failure from outside the child.
    assert "check the job log" in message


# --- One heavy job at a time, from every direction ---------------------------


def test_the_worker_entrypoint_refuses_to_run_while_a_solver_run_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracker.sqlite3"
    db = _make_db(db_path)
    job = _queued_job(db, tmp_path)
    session = db.create_session(Session(name="Study"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            status="running",
            backend_name="texassolver",
            backend_version="test",
            input_hash="a" * 64,
            # This process: the admission check asks whether a worker is really
            # there, not whether a row says so.
            pid=os.getpid(),
        )
    )
    db.close()
    _stub_worker_dependencies(monkeypatch, tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("A second heavy job must not reach the pipeline.")

    monkeypatch.setattr(run_cv_job, "_run_pipeline", refuse)

    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(tmp_path / "session.mp4"),
        session_name="Bounded",
        db_path=db_path,
    )

    checked = PokerDatabase(str(db_path))
    saved = checked.fetch_processing_job(job.id)
    checked.close()
    assert exit_code == 1
    assert saved.status == "failed"
    assert "solver run" in (saved.error_message or "")
    assert "one heavy" in (saved.error_message or "").lower()


def test_the_worker_entrypoint_refuses_to_run_while_another_cv_job_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "tracker.sqlite3"
    db = _make_db(db_path)
    first = _queued_job(db, tmp_path)
    db.update_processing_job(first.id, pid=os.getpid())
    second = db.create_processing_job(
        ProcessingJob(
            video_id=first.video_id,
            job_type="cv_reconstruction",
            status="running",
        )
    )
    db.close()
    _stub_worker_dependencies(monkeypatch, tmp_path)

    def refuse(*args, **kwargs):
        raise AssertionError("A second heavy job must not reach the pipeline.")

    monkeypatch.setattr(run_cv_job, "_run_pipeline", refuse)

    exit_code = run_cv_job.run_job(
        job_id=second.id,
        video_path=Path(tmp_path / "session.mp4"),
        session_name="Bounded",
        db_path=db_path,
    )

    checked = PokerDatabase(str(db_path))
    saved = checked.fetch_processing_job(second.id)
    checked.close()
    assert exit_code == 1
    assert saved.status == "failed"
    assert f"job #{first.id}" in (saved.error_message or "")


def test_a_solver_launch_is_refused_while_a_cv_reconstruction_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direction the suite never covered: CV holding the machine, solver asking.

    The mirror case (CV refused while a solve runs) is pinned in
    tests/test_cv_jobs.py; two concurrent solves are pinned in tests/test_solver.py.
    Without this one the guard in solver/jobs.py could be deleted and the suite
    would stay green.
    """
    from poker_tracker.solver import jobs as solver_jobs
    from poker_tracker.solver.models import ResolvedRange, SolverPlayer, SolverSpot

    binary = tmp_path / "console_solver"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(solver_jobs, "configured_binary", lambda: binary)
    monkeypatch.setattr(solver_jobs, "configured_resource_dir", lambda _binary: tmp_path)

    db_path = tmp_path / "tracker.sqlite3"
    db = _make_db(db_path)
    job = _queued_job(db, tmp_path)
    session = db.create_session(Session(name="Study"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))

    spot = SolverSpot(
        hand_id=hand.id,
        table_size=6,
        street="flop",
        board="AhKd7c",
        pot=10.0,
        effective_stack=100.0,
        pot_type="single_raised",
        preflop_aggressor_key="hero",
        oop=SolverPlayer(player_key="hero", player_name="Hero", role="oop", is_hero=True),
        ip=SolverPlayer(player_key="villain", player_name="Villain", role="ip"),
        hero_cards="AsKs",
    )
    range_oop = ResolvedRange(
        player_key="hero",
        player_name="Hero",
        role="oop",
        source="user",
        profile_name="Hero range",
        notation="AA",
        solver_notation="AA",
        combo_count=6,
        range_percent=0.005,
    )
    range_ip = range_oop.model_copy(
        update={"player_key": "villain", "player_name": "Villain", "role": "ip"}
    )

    with pytest.raises(solver_jobs.SolverJobAlreadyRunningError) as refused:
        solver_jobs.start_solver_job(db, spot, range_ip, range_oop)

    assert f"job #{job.id}" in str(refused.value)
    assert db.fetch_active_solver_runs() == []
    db.close()


# --- Bounded retained artifacts and bounded temporary files ------------------


def test_the_frame_ceiling_bounds_an_extraction_that_asked_for_no_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(frame_extraction.FRAME_CEILING_ENV_VAR, "4")
    db = PokerDatabase(":memory:")
    db.init_db()
    video_path = _synthetic_video(tmp_path / "synthetic.avi")
    video = db.create_video(
        VideoRecord(
            original_filename=video_path.name,
            stored_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
        )
    )

    summary = frame_extraction.extract_frames_for_video(
        db,
        video.id,
        frames_per_second=10,
        max_frames=None,
        frames_dir=tmp_path / "frames",
    )

    assert summary.frames_extracted == 4
    assert summary.frame_limit == 4
    # A truncated frame set that does not say so is a partial reading presented
    # as the whole recording.
    assert summary.truncated_at_limit is True
    job = db.fetch_jobs_by_video(video.id)[0]
    assert frame_extraction.FRAME_CEILING_ENV_VAR in job.message
    assert "the recording has more" in job.message
    db.close()


def test_an_extraction_that_fits_under_the_ceiling_never_claims_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(frame_extraction.FRAME_CEILING_ENV_VAR, "500")
    db = PokerDatabase(":memory:")
    db.init_db()
    video_path = _synthetic_video(tmp_path / "synthetic.avi")
    video = db.create_video(
        VideoRecord(
            original_filename=video_path.name,
            stored_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
        )
    )

    summary = frame_extraction.extract_frames_for_video(
        db, video.id, frames_per_second=2, frames_dir=tmp_path / "frames"
    )

    assert summary.truncated_at_limit is False
    assert frame_extraction.FRAME_CEILING_ENV_VAR not in db.fetch_jobs_by_video(video.id)[0].message
    db.close()


def test_a_video_with_no_readable_duration_is_bounded_without_claiming_it_has_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound still applies; the claim about what was left behind does not.

    With no fps and no frame count there is no denominator, so "the recording
    has more" would be a number reported without the thing that makes it mean
    anything.
    """
    import numpy as np

    class UnmeasurableCapture:
        def __init__(self, _path: str) -> None:
            self.position = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            import cv2

            if prop == cv2.CAP_PROP_POS_FRAMES:
                return float(self.position)
            return 0.0

        def set(self, prop: int, value: float) -> bool:
            return True

        def read(self):
            self.position += 1
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    monkeypatch.setenv(frame_extraction.FRAME_CEILING_ENV_VAR, "3")
    monkeypatch.setattr(frame_extraction.cv2, "VideoCapture", UnmeasurableCapture)
    db = PokerDatabase(":memory:")
    db.init_db()
    unreadable = tmp_path / "unmeasurable.avi"
    unreadable.write_bytes(b"header only")
    video = db.create_video(
        VideoRecord(
            original_filename=unreadable.name,
            stored_path=str(unreadable),
            file_size_bytes=unreadable.stat().st_size,
        )
    )

    summary = frame_extraction.extract_frames_for_video(
        db, video.id, frames_per_second=1, frames_dir=tmp_path / "frames"
    )

    assert summary.frames_extracted == 3
    assert summary.frame_limit == 3
    assert summary.truncated_at_limit is False
    assert (
        frame_extraction.FRAME_CEILING_ENV_VAR
        not in db.fetch_jobs_by_video(video.id)[0].message
    )
    db.close()


def test_a_misconfigured_frame_ceiling_is_refused_before_a_job_row_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(frame_extraction.FRAME_CEILING_ENV_VAR, "lots")
    db = PokerDatabase(":memory:")
    db.init_db()
    video_path = _synthetic_video(tmp_path / "synthetic.avi")
    video = db.create_video(
        VideoRecord(
            original_filename=video_path.name,
            stored_path=str(video_path),
            file_size_bytes=video_path.stat().st_size,
        )
    )

    with pytest.raises(ValueError, match=frame_extraction.FRAME_CEILING_ENV_VAR):
        frame_extraction.extract_frames_for_video(
            db, video.id, max_frames=2, frames_dir=tmp_path / "frames"
        )

    assert db.fetch_jobs_by_video(video.id) == []
    db.close()


def test_abandoned_upload_partials_are_swept_and_live_ones_are_left_alone(
    tmp_path: Path,
) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    abandoned = videos_dir / f"{video_ingest.UPLOAD_PARTIAL_PREFIX}dead.mp4"
    abandoned.write_bytes(b"half an upload")
    in_flight = videos_dir / f"{video_ingest.UPLOAD_PARTIAL_PREFIX}live.mp4"
    in_flight.write_bytes(b"still uploading")
    recording = videos_dir / "session.mp4"
    recording.write_bytes(b"a real recording")
    day = video_ingest.UPLOAD_PARTIAL_MAX_AGE_SECONDS
    os.utime(abandoned, (time.time() - day - 60, time.time() - day - 60))

    removed = video_ingest.sweep_stale_upload_partials(videos_dir)

    assert removed == 1
    assert not abandoned.exists()
    assert in_flight.exists()
    assert recording.exists()


def test_ingesting_a_recording_sweeps_the_partials_nothing_else_ever_looks_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep no code path calls is not a bound.

    Retention manages generated artifacts only, and the video vault is
    deliberately never expired by age, so ingest is the one moment anything
    visits this directory with a reason to tidy it.
    """
    import io

    videos_dir = tmp_path / "data" / "videos"
    videos_dir.mkdir(parents=True)
    abandoned = videos_dir / f"{video_ingest.UPLOAD_PARTIAL_PREFIX}dead.mp4"
    abandoned.write_bytes(b"half an upload")
    stale = time.time() - video_ingest.UPLOAD_PARTIAL_MAX_AGE_SECONDS - 60
    os.utime(abandoned, (stale, stale))
    monkeypatch.setattr(
        video_ingest,
        "require_playable_video",
        lambda path, **kwargs: SimpleNamespace(error=None),
    )

    ingested = video_ingest.ingest_uploaded_video(
        io.BytesIO(b"a real recording"), "session.mp4", videos_dir
    )

    assert ingested.path.exists()
    assert not abandoned.exists()


def test_the_sweep_never_removes_a_directory_wearing_the_partial_prefix(
    tmp_path: Path,
) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    impostor = videos_dir / f"{video_ingest.UPLOAD_PARTIAL_PREFIX}dir"
    impostor.mkdir()

    assert video_ingest.sweep_stale_upload_partials(videos_dir, now=time.time() + 10**7) == 0
    assert impostor.is_dir()


def _synthetic_video(path: Path) -> Path:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(20):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :] = (index * 10 % 255, index * 5 % 255, index * 20 % 255)
        writer.write(frame)
    writer.release()
    return path
