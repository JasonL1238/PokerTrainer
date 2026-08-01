"""Phase 4 failure injection: every failure must have a safe terminal state.

PLAN.md lists fifteen failures a reconstruction job must survive. Nine of them
had no coverage; this file adds those. The invariant under test is not that the
job succeeds — it is that a failed job leaves NO partially imported session, NO
job stuck in a running state, and the prior database intact.

The other six (process kill, app restart, progress-file corruption, backup
failure, corrupt recording, simultaneous CV/solver) are covered in
tests/test_cv_jobs.py and tests/test_video_ingest.py.
"""

from __future__ import annotations

import errno
import sqlite3
import subprocess
from pathlib import Path

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import ProcessingJob, Session
from poker_tracker.ui import run_cv_job
from poker_tracker.ui.video_ingest import (
    VideoIngestLimits,
    ingest_uploaded_video,
    require_playable_video,
)
from tests.test_cv_jobs import add_video, create_synthetic_video, make_db, videos_dir  # noqa: F401


def _queued_job(db: PokerDatabase, video_id: int) -> ProcessingJob:
    return db.create_processing_job(
        ProcessingJob(video_id=video_id, job_type="cv_reconstruction")
    )


def _run_and_expect_failure(db_path: Path, job_id: int, video: Path) -> int:
    return run_cv_job.run_job(
        job_id=job_id,
        video_path=video,
        session_name="Injected failure",
        db_path=db_path,
        timeout_seconds=30,
    )


def _assert_safe_terminal_state(db: PokerDatabase, job_id: int) -> None:
    """A failed job is terminal, carries a reason, and imported nothing."""
    job = db.fetch_processing_job(job_id)
    assert job is not None
    assert job.status in {"failed", "cancelled"}, f"job left in {job.status!r}"
    assert job.pid is None, "a terminal job must not still claim a worker pid"
    assert (job.error_message or "").strip(), "a failure must say why"
    assert db._execute("SELECT COUNT(*) FROM hands").fetchone()[0] == 0


@pytest.fixture
def job_workspace(tmp_path: Path, videos_dir: Path):  # noqa: F811
    db_path = tmp_path / "study.db"
    db = make_db(str(db_path))
    video_path = videos_dir / "session.mov"
    record = add_video(db, video_path, playable=True, with_hash=True)
    assert record.id is not None
    job = _queued_job(db, record.id)
    assert job.id is not None
    yield db, db_path, video_path, job.id
    db.close()


# --- Pipeline-side failures -------------------------------------------------


@pytest.mark.parametrize(
    ("label", "returncode", "stderr"),
    [
        ("unreadable model", 1, "PermissionError: [Errno 13] weights not readable"),
        ("model exception", 1, "RuntimeError: shape mismatch in detector head"),
        ("missing ffmpeg", 127, "ffmpeg: command not found"),
        ("unsupported codec", 1, "Unsupported codec 'prores'"),
    ],
)
def test_pipeline_failure_ends_in_a_safe_terminal_state(
    job_workspace, monkeypatch, label, returncode, stderr
):
    """Whatever the subprocess dies of, the job must not stay running."""
    db, db_path, video_path, job_id = job_workspace

    class DeadProcess:
        pid = 4242

        def __init__(self) -> None:
            self._polls = 0
            self.returncode = returncode

        def poll(self):
            self._polls += 1
            return returncode

        def wait(self, timeout=None):
            return returncode

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: DeadProcess())
    code = _run_and_expect_failure(db_path, job_id, video_path)

    assert code == 1, label
    _assert_safe_terminal_state(db, job_id)


def test_timeout_terminates_the_worker_and_fails_the_job(job_workspace, monkeypatch):
    db, db_path, video_path, job_id = job_workspace

    terminated: list[int] = []

    class HangingProcess:
        pid = 5150
        returncode = -15

        def poll(self):
            return None  # never exits

        def wait(self, timeout=None):
            return -15

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: HangingProcess())
    monkeypatch.setattr(
        run_cv_job, "_terminate_process_group", lambda pid: terminated.append(pid)
    )

    code = run_cv_job.run_job(
        job_id=job_id,
        video_path=video_path,
        session_name="Timeout",
        db_path=db_path,
        timeout_seconds=0,  # already past the deadline on the first check
    )
    assert code == 1
    assert terminated == [5150], "the hung process group must be signalled"
    _assert_safe_terminal_state(db, job_id)
    assert "timeout" in (db.fetch_processing_job(job_id).error_message or "").lower()


# --- Storage-side failures --------------------------------------------------


def test_disk_full_during_ingest_leaves_no_partial_file(tmp_path: Path):
    """A short write must not leave a truncated recording behind."""
    videos = tmp_path / "videos"
    videos.mkdir()

    class ExplodingSource:
        def __init__(self) -> None:
            self._sent = False

        def read(self, size):
            if self._sent:
                raise OSError(errno.ENOSPC, "No space left on device")
            self._sent = True
            return b"\x00" * 1024

    with pytest.raises(OSError) as excinfo:
        ingest_uploaded_video(ExplodingSource(), "session.mov", videos)
    assert excinfo.value.errno == errno.ENOSPC
    # Neither the destination nor the temp file may survive.
    assert list(videos.iterdir()) == []


def test_permission_denied_on_the_storage_root_is_rejected_before_writing(
    tmp_path: Path
):
    videos = tmp_path / "videos"
    videos.mkdir()
    videos.chmod(0o500)  # readable and traversable, not writable
    try:
        with pytest.raises(ValueError, match="not writable"):
            ingest_uploaded_video(_FakeUpload(b"data"), "session.mov", videos)
    finally:
        videos.chmod(0o700)


def test_unreadable_video_file_is_rejected(tmp_path: Path):
    video = create_synthetic_video(tmp_path / "clip.mov")
    video.chmod(0o000)
    try:
        with pytest.raises(ValueError):
            require_playable_video(video)
    finally:
        video.chmod(0o600)


class _FakeUpload:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._done = False

    def read(self, size):
        if self._done:
            return b""
        self._done = True
        return self._payload


# --- Database-side failures -------------------------------------------------


def test_database_locked_surfaces_rather_than_corrupting(tmp_path: Path):
    """A busy database must raise, not silently skip the write."""
    db_path = tmp_path / "locked.db"
    db = make_db(str(db_path))
    try:
        blocker = sqlite3.connect(db_path, timeout=0.1)
        blocker.execute("BEGIN EXCLUSIVE")
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
                db.create_session(Session(name="blocked", platform="ClubWPT Gold"))
        finally:
            blocker.rollback()
            blocker.close()
    finally:
        db.close()


def test_import_validation_failure_leaves_the_database_unchanged(
    job_workspace, monkeypatch
):
    """A rejected export must not half-import, and must keep the prior state."""
    db, db_path, video_path, job_id = job_workspace
    sessions_before = db._execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    class FinishedProcess:
        pid = 9001
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FinishedProcess())

    def _reject(*_args, **_kwargs):
        raise ValueError("Reconstructed timeline failed import validation.")

    monkeypatch.setattr(run_cv_job, "export_timeline", _reject)

    code = _run_and_expect_failure(db_path, job_id, video_path)
    assert code == 1
    _assert_safe_terminal_state(db, job_id)
    assert db._execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == sessions_before


# --- The bound on what a failure may reveal ---------------------------------


def test_failure_messages_are_bounded_and_single_line(job_workspace, monkeypatch):
    """An unbounded traceback in a job row is both a UI and a leak problem."""
    db, db_path, video_path, job_id = job_workspace

    class FinishedProcess:
        pid = 9002
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FinishedProcess())
    secret = "sk-ant-super-secret-key"
    noise = ("x" * 200 + "\n") * 20

    def _explode(*_args, **_kwargs):
        # The secret leads. Truncation cannot save us here — only redaction can.
        raise ValueError(f"calling provider with api_key={secret}\n{noise}")

    monkeypatch.setattr(run_cv_job, "export_timeline", _explode)
    _run_and_expect_failure(db_path, job_id, video_path)

    message = db.fetch_processing_job(job_id).error_message or ""
    assert len(message) <= 500, "job error messages must stay bounded"
    assert "\n" not in message, "newlines break the job list rendering"
    assert secret not in message
    assert "<redacted>" in message, "the key must be scrubbed, not merely cut off"


@pytest.mark.parametrize(
    "limits",
    [
        VideoIngestLimits(max_duration_seconds=0.5),
        VideoIngestLimits(max_width=32),
        VideoIngestLimits(min_frame_count=10_000),
    ],
)
def test_out_of_bounds_recordings_are_rejected(tmp_path: Path, limits):
    video = create_synthetic_video(tmp_path / "clip.mov")
    with pytest.raises(ValueError):
        require_playable_video(video, limits=limits)
