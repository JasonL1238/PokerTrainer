"""The full ingest chain, driven end to end against one synthetic recording.

    video metadata -> stored video -> job -> timeline -> export -> backup -> import

Every other test in the suite crosses part of this: tests/test_cv_jobs.py runs the
worker but never imports a hand, and tests/test_yolo_e2e_coaching_seam.py imports a
hand that never came from a video or a job. Nothing proved that what the decoder
read off the file is what the job told the pipeline, what the exporter wrote, what
the snapshot preserved, and what the imported hand is finally attributed to.

Only the CV pipeline itself is stubbed, and only because running two YOLO models is
not a unit-test operation. The stub is handed the real argv the worker built and
writes a real timeline to the real path that argv names, so the job lifecycle, the
exporter, the snapshot, the snapshot verification and the validated-hand import are
all the production code paths.

The assertions are about DATA, not about calls returning. Each seam re-reads what
arrived and compares it to what was sent, with particular attention to the three
things that change representation on the way: filesystem paths, timestamps, and
anything that round-trips through JSON.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence import backup_inventory
from poker_tracker.persistence.backup import find_snapshots
from poker_tracker.persistence.completion import parse_completion_evidence
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import VideoRecord
from poker_tracker.services import validated_hand_import
from poker_tracker.ui import cv_jobs, run_cv_job, video_ingest, video_storage
from poker_tracker.ui.video_metadata import extract_video_metadata

# What the synthetic recording is authored to contain. Every later assertion about
# duration, geometry and frame count traces back to these four numbers, so a seam
# that quietly substitutes a default instead of carrying the measured value has
# nowhere to hide.
SOURCE_FPS = 10.0
SOURCE_WIDTH = 64
SOURCE_HEIGHT = 48
SOURCE_FRAMES = 20
SOURCE_DURATION_SECONDS = SOURCE_FRAMES / SOURCE_FPS

TIMELINE_HAND_NUMBER = 42
SESSION_NAME = "Chain session 07-01"

# Ground truth for the one reconstructed hand, in the app's card spelling. The
# timeline below states the cards in raw detector labels ("AS", "10H") so the
# normalization in the exporter is exercised rather than bypassed.
GROUND_TRUTH_HERO = "As Th"
GROUND_TRUTH_BOARD = "Qd 7s 2c 9h Kc"
GROUND_TRUTH_POT = 42.5
GROUND_TRUTH_HERO_BB_WON = 21.0
GROUND_TRUTH_T_START = 3.5
GROUND_TRUTH_T_END = 61.25
GROUND_TRUTH_FRAME_REFS = (
    "images/val/frame_000035.jpg",
    "images/val/frame_000240.jpg",
    "images/val/frame_000480.jpg",
    "images/val/frame_000612.jpg",
)
GROUND_TRUTH_LAYOUT_PROFILE = "clubwpt_gold_9max"
GROUND_TRUTH_PIPELINE_VERSION = "two_model_spine_v7"
GROUND_TRUTH_MODEL_VERSIONS = {
    "detector": "yolo11_v7",
    "card_classifier": "cards_v3",
}


def _synthetic_recording(path: Path) -> Path:
    """Write a small, genuinely decodable MJPEG recording with known geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        SOURCE_FPS,
        (SOURCE_WIDTH, SOURCE_HEIGHT),
    )
    assert writer.isOpened(), "OpenCV could not open an MJPEG writer for the fixture."
    for index in range(SOURCE_FRAMES):
        frame = np.zeros((SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (index * 10 % 255, index * 5 % 255, index * 20 % 255)
        writer.write(frame)
    writer.release()
    return path


def _spine_timeline() -> dict[str, Any]:
    """One reconstruction-spine hand, with the metadata the exporter attributes to."""
    return {
        "metadata": {
            "layout_profile": GROUND_TRUTH_LAYOUT_PROFILE,
            "source": GROUND_TRUTH_PIPELINE_VERSION,
            "model_versions": dict(GROUND_TRUTH_MODEL_VERSIONS),
        },
        "states": [
            {
                "image": GROUND_TRUTH_FRAME_REFS[0],
                "time_s": GROUND_TRUTH_T_START,
                "hero_cards": ["AS", "10H"],
                "board_cards": [],
            },
            {
                "image": GROUND_TRUTH_FRAME_REFS[1],
                "time_s": 24.0,
                "hero_cards": ["AS", "10H"],
                "board_cards": ["QD", "7S", "2C"],
            },
            {
                "image": GROUND_TRUTH_FRAME_REFS[2],
                "time_s": 48.0,
                "hero_cards": ["AS", "10H"],
                "board_cards": ["QD", "7S", "2C", "9H"],
            },
            {
                "image": GROUND_TRUTH_FRAME_REFS[3],
                "time_s": GROUND_TRUTH_T_END,
                "hero_cards": ["AS", "10H"],
                "board_cards": ["QD", "7S", "2C", "9H", "KC"],
            },
        ],
        "hands": [
            {
                "hand_number": TIMELINE_HAND_NUMBER,
                "t_start": GROUND_TRUTH_T_START,
                "t_end": GROUND_TRUTH_T_END,
                "hero": ["AS", "10H"],
                "board": ["QD", "7S", "2C", "9H", "KC"],
                "complete_cards": True,
                "warnings": [],
                "hero_seat_confirmed": True,
                "terminal_event": "showdown",
                "players": [
                    {
                        "seat": 0,
                        "position": "SB",
                        "player_name": "Hero",
                        "starting_stack": 100.0,
                        "is_hero": True,
                    },
                    {
                        "seat": 5,
                        "position": "BTN",
                        "player_name": "Villain",
                        "starting_stack": 100.0,
                        "is_hero": False,
                    },
                ],
                "actions": [
                    {
                        "street": "preflop",
                        "action_index": 1,
                        "seat": 5,
                        "position": "BTN",
                        "player_name": "Villain",
                        "action_type": "raise",
                        "amount": 3.0,
                        "pot_before": 1.5,
                        "stack_before": 100.0,
                    },
                    {
                        "street": "preflop",
                        "action_index": 2,
                        "seat": 0,
                        "position": "SB",
                        "player_name": "Hero",
                        "action_type": "call",
                        "amount": 2.5,
                        "pot_before": 4.5,
                        "stack_before": 99.0,
                    },
                    {
                        "street": "flop",
                        "action_index": 1,
                        "seat": 0,
                        "position": "SB",
                        "player_name": "Hero",
                        "action_type": "check",
                        "amount": None,
                        "pot_before": 6.5,
                        "stack_before": 96.5,
                    },
                    {
                        "street": "flop",
                        "action_index": 2,
                        "seat": 5,
                        "position": "BTN",
                        "player_name": "Villain",
                        "action_type": "bet",
                        "amount": 4.0,
                        "pot_before": 6.5,
                        "stack_before": 96.5,
                    },
                    {
                        "street": "flop",
                        "action_index": 3,
                        "seat": 0,
                        "position": "SB",
                        "player_name": "Hero",
                        "action_type": "call",
                        "amount": 4.0,
                        "pot_before": 10.5,
                        "stack_before": 96.5,
                    },
                    {
                        "street": "turn",
                        "action_index": 1,
                        "seat": 0,
                        "position": "SB",
                        "player_name": "Hero",
                        "action_type": "check",
                        "amount": None,
                        "pot_before": 14.5,
                        "stack_before": 92.5,
                    },
                    {
                        "street": "turn",
                        "action_index": 2,
                        "seat": 5,
                        "position": "BTN",
                        "player_name": "Villain",
                        "action_type": "check",
                        "amount": None,
                        "pot_before": 14.5,
                        "stack_before": 92.5,
                    },
                    {
                        "street": "river",
                        "action_index": 1,
                        "seat": 0,
                        "position": "SB",
                        "player_name": "Hero",
                        "action_type": "bet",
                        "amount": 12.0,
                        "pot_before": 14.5,
                        "stack_before": 92.5,
                    },
                    {
                        "street": "river",
                        "action_index": 2,
                        "seat": 5,
                        "position": "BTN",
                        "player_name": "Villain",
                        "action_type": "call",
                        "amount": 12.0,
                        "pot_before": 26.5,
                        "stack_before": 92.5,
                    },
                ],
                "pot": GROUND_TRUTH_POT,
                "winner_seat": 0,
                "result": "Hero wins",
                "hero_bb_won": GROUND_TRUTH_HERO_BB_WON,
                "reconciled": True,
                "source_images": list(GROUND_TRUTH_FRAME_REFS),
            }
        ],
    }


class _ChainEnv:
    """One isolated data root plus the recording, so no test touches the real one."""

    def __init__(self, root: Path, paths: dict[str, Path], source: Path) -> None:
        self.root = root
        self.paths = paths
        self.source = source
        self.db_path = root / "poker_tracker.db"
        self.pipeline_commands: list[list[str]] = []


@pytest.fixture
def chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _ChainEnv:
    root = tmp_path / "data"
    paths = video_storage.ensure_data_directories(root)

    # Every module that resolves a storage root from an import-time constant. The
    # worker and the launcher each hold their own binding of VIDEOS_DIR, so a
    # single patch on video_storage would leave one of them pointing at the
    # operator's real data directory.
    monkeypatch.setattr(video_storage, "DATA_DIR", root)
    monkeypatch.setattr(video_storage, "VIDEOS_DIR", paths["videos"])
    monkeypatch.setattr(video_storage, "FRAMES_DIR", paths["frames"])
    monkeypatch.setattr(video_storage, "EXPORTS_DIR", paths["exports"])
    monkeypatch.setattr(video_storage, "CV_TIMELINES_DIR", paths["cv_timelines"])
    monkeypatch.setattr(video_storage, "JOB_LOGS_DIR", paths["job_logs"])
    monkeypatch.setattr(cv_jobs, "VIDEOS_DIR", paths["videos"])
    monkeypatch.setattr(cv_jobs, "CV_TIMELINES_DIR", paths["cv_timelines"])
    monkeypatch.setattr(cv_jobs, "ensure_data_directories", lambda *a, **k: paths)
    monkeypatch.setattr(run_cv_job, "VIDEOS_DIR", paths["videos"])
    monkeypatch.setattr(run_cv_job, "ensure_data_directories", lambda *a, **k: paths)
    monkeypatch.setattr(video_ingest, "VIDEOS_DIR", paths["videos"])
    monkeypatch.setattr(backup_module, "DATA_DIR", root)
    monkeypatch.setattr(backup_module, "BACKUPS_DIR", paths["backups"])
    monkeypatch.setattr(backup_module, "LIVE_DB_PATH", root / "poker_tracker.db")
    monkeypatch.setattr(backup_inventory, "DATA_DIR", root)

    source = _synthetic_recording(tmp_path / "incoming" / "Session 07-01 5.00 PM.avi")
    return _ChainEnv(root, paths, source)


def _stub_pipeline(chain: _ChainEnv, timeline: dict[str, Any]):
    """Stand in for the two-model pipeline: record argv, write the timeline it names.

    The stub deliberately reads ``--out`` out of the command rather than being told
    where to write. If the worker ever stops passing the path it later reads back,
    the timeline lands somewhere else and the export fails, instead of the test
    quietly agreeing with a broken argument.
    """

    def _run(command, db, job_id, deadline, progress_path, limits=None) -> None:  # noqa: ANN001
        chain.pipeline_commands.append(list(command))
        out_index = command.index("--out")
        destination = Path(command[out_index + 1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(timeline), encoding="utf-8")

    return _run


def _argument(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _open_snapshot_read_only(snapshot: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{snapshot.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def test_recording_survives_ingest_job_timeline_export_backup_and_import(
    chain: _ChainEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeline = _spine_timeline()
    db = PokerDatabase(str(chain.db_path))
    db.init_db()

    # ---------------------------------------------------------------- seam 1
    # Video metadata -> stored video. What the decoder measured off the file has
    # to be what the row says, byte for byte, after a trip through SQLite.
    read_metadata = extract_video_metadata(chain.source)
    assert read_metadata.error == ""
    assert read_metadata.fps == SOURCE_FPS
    assert read_metadata.width == SOURCE_WIDTH
    assert read_metadata.height == SOURCE_HEIGHT
    assert read_metadata.frame_count == SOURCE_FRAMES
    assert read_metadata.duration_seconds == SOURCE_DURATION_SECONDS

    with chain.source.open("rb") as handle:
        ingested = video_ingest.ingest_uploaded_video(
            handle, chain.source.name, chain.paths["videos"]
        )
    assert ingested.metadata == read_metadata
    assert ingested.file_size_bytes == chain.source.stat().st_size
    assert ingested.path.parent == chain.paths["videos"]

    uploaded_at = datetime(2026, 7, 1, 17, 5, 30, tzinfo=UTC)
    stored = db.create_video(
        VideoRecord(
            original_filename=chain.source.name,
            stored_path=str(ingested.path),
            file_size_bytes=ingested.file_size_bytes,
            content_sha256=ingested.content_sha256,
            duration_seconds=ingested.metadata.duration_seconds,
            fps=ingested.metadata.fps,
            width=ingested.metadata.width,
            height=ingested.metadata.height,
            frame_count=ingested.metadata.frame_count,
            uploaded_at=uploaded_at,
            notes="Chain fixture recording",
        )
    )
    assert stored.id is not None

    persisted = db.fetch_video(stored.id)
    assert persisted is not None
    assert persisted.stored_path == str(ingested.path)
    assert Path(persisted.stored_path).is_file()
    assert persisted.original_filename == chain.source.name
    assert persisted.file_size_bytes == ingested.file_size_bytes
    assert persisted.content_sha256 == ingested.content_sha256
    assert persisted.duration_seconds == SOURCE_DURATION_SECONDS
    assert persisted.fps == SOURCE_FPS
    assert persisted.width == SOURCE_WIDTH
    assert persisted.height == SOURCE_HEIGHT
    assert persisted.frame_count == SOURCE_FRAMES
    # Timestamps are stored as text and rebuilt; the instant must be preserved,
    # including its offset, not merely the calendar date.
    assert persisted.uploaded_at == uploaded_at

    # ---------------------------------------------------------------- seam 2
    # Stored video -> job. The launcher must address the row it was handed and
    # hand the worker the recording that row names.
    monkeypatch.setattr(
        cv_jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=424242)
    )
    job = cv_jobs.start_cv_job(db, stored.id, persisted.stored_path, SESSION_NAME)
    assert job.id is not None
    assert job.video_id == stored.id
    assert job.job_type == "cv_reconstruction"
    assert job.status == "running"
    assert job.pid == 424242

    # ---------------------------------------------------------------- seam 3
    # Job -> timeline. The worker builds the pipeline argv from the stored row, so
    # the recording path and the measured duration have to arrive there intact.
    monkeypatch.setattr(run_cv_job, "_run_pipeline", _stub_pipeline(chain, timeline))
    db.close()

    exit_code = run_cv_job.run_job(
        job_id=job.id,
        video_path=Path(persisted.stored_path),
        session_name=SESSION_NAME,
        db_path=chain.db_path,
    )
    assert exit_code == 0

    assert len(chain.pipeline_commands) == 1
    command = chain.pipeline_commands[0]
    assert _argument(command, "--video") == persisted.stored_path
    assert float(_argument(command, "--end")) == SOURCE_DURATION_SECONDS
    timeline_path = Path(_argument(command, "--out"))
    assert timeline_path == chain.paths["cv_timelines"] / f"job_{job.id}_timeline.json"
    assert json.loads(timeline_path.read_text(encoding="utf-8")) == timeline

    db = PokerDatabase(str(chain.db_path))
    finished = db.fetch_processing_job(job.id)
    assert finished is not None
    assert finished.status == "completed"
    assert finished.progress_percent == 100
    assert finished.pid is None
    assert "1 hands exported" in finished.message
    # The worker verifies the snapshot it just wrote and reports the outcome on the
    # job row. A snapshot reported as merely "written" is a recovery point on paper.
    assert "verified" in finished.message
    assert "FAILED VERIFICATION" not in finished.message

    linked = db.fetch_video(stored.id)
    assert linked is not None
    assert linked.session_id is not None
    # The row the job touched must still be the recording that was ingested.
    assert linked.stored_path == str(ingested.path)
    assert linked.content_sha256 == ingested.content_sha256
    assert linked.duration_seconds == SOURCE_DURATION_SECONDS

    destination_session = db.fetch_session(linked.session_id)
    assert destination_session is not None
    assert destination_session.name == SESSION_NAME

    # ---------------------------------------------------------------- seam 4
    # Timeline -> export. The exporter's own JSON file on disk is what the later
    # import reads, so it is asserted from disk rather than from the return value.
    export_path = chain.paths["exports"] / f"job_{job.id}_session.json"
    assert export_path.is_file()
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["cv_import_summary"]["timeline"] == str(timeline_path)
    assert exported["cv_import_summary"]["timeline_hands"] == 1
    assert exported["cv_import_summary"]["exported_hands"] == 1
    assert exported["cv_import_summary"]["skipped_hands"] == 0
    exported_hand = exported["hands"][0]["hand"]
    assert exported_hand["hero_cards"] == GROUND_TRUTH_HERO
    assert exported_hand["board_cards"] == GROUND_TRUTH_BOARD
    assert exported_hand["pot_size"] == GROUND_TRUTH_POT
    assert exported_hand["hero_bb_won"] == GROUND_TRUTH_HERO_BB_WON
    assert exported_hand["source_type"] == "cv_import"
    exported_evidence = exported_hand["completion_evidence"]
    assert exported_evidence["first_source_timestamp_s"] == GROUND_TRUTH_T_START
    assert exported_evidence["last_source_timestamp_s"] == GROUND_TRUTH_T_END
    assert tuple(exported_evidence["source_frames"]) == GROUND_TRUTH_FRAME_REFS
    assert exported_evidence["layout_profile"] == GROUND_TRUTH_LAYOUT_PROFILE
    assert exported_evidence["pipeline_version"] == GROUND_TRUTH_PIPELINE_VERSION
    assert exported_evidence["model_versions"] == GROUND_TRUTH_MODEL_VERSIONS

    # ---------------------------------------------------------------- seam 5
    # Export -> backup. The snapshot the worker took is opened as a database and
    # read back: the recording row has to be there, unchanged, with its artifact
    # inventory naming the file it points at.
    snapshots = find_snapshots(chain.paths["backups"], purpose="routine")
    assert len(snapshots) == 1
    snapshot = snapshots[0]

    with _open_snapshot_read_only(snapshot) as connection:
        rows = connection.execute("SELECT * FROM videos").fetchall()
    assert len(rows) == 1
    snapshot_video = dict(rows[0])
    assert snapshot_video["stored_path"] == str(ingested.path)
    assert snapshot_video["content_sha256"] == ingested.content_sha256
    assert snapshot_video["file_size_bytes"] == ingested.file_size_bytes
    assert snapshot_video["duration_seconds"] == SOURCE_DURATION_SECONDS

    inventory = backup_inventory.load_inventory(snapshot)
    assert inventory is not None
    assert inventory["error"] is None
    recorded = {
        entry["path"]: entry
        for entry in inventory["artifacts"]
        if entry["source"] == "videos.stored_path"
    }
    assert str(ingested.path) in recorded
    video_entry = recorded[str(ingested.path)]
    assert video_entry["present"] is True
    assert video_entry["bytes"] == ingested.file_size_bytes
    assert video_entry["sha256"] == ingested.content_sha256

    # A restored snapshot has to be a working database, not just a readable file.
    restored_path = chain.root / "restored.sqlite3"
    shutil.copy2(snapshot, restored_path)
    restored = PokerDatabase(str(restored_path))
    restored_video = restored.fetch_video(stored.id)
    assert restored_video is not None
    assert restored_video.stored_path == str(ingested.path)
    assert restored_video.uploaded_at == uploaded_at
    assert restored_video.duration_seconds == SOURCE_DURATION_SECONDS
    restored.close()

    # ---------------------------------------------------------------- seam 6
    # Timeline -> import. The product's only path from a reconstructed timeline
    # into the study database, reading the timeline the job wrote and landing the
    # hand in the session the job linked the recording to.
    result = validated_hand_import.ensure_hand_imported(
        db,
        job.id,
        TIMELINE_HAND_NUMBER,
        mode="draft",
        timeline_dir=chain.paths["cv_timelines"],
        data_dir=chain.paths["data"],
    )
    assert result.status == "imported", result.message
    assert result.session_id == linked.session_id
    assert result.hand_id is not None

    imported = db.fetch_hand(result.hand_id)
    assert imported is not None
    assert imported.session_id == linked.session_id
    assert imported.hero_cards == GROUND_TRUTH_HERO
    assert imported.board_cards == GROUND_TRUTH_BOARD
    assert imported.pot_size == GROUND_TRUTH_POT
    assert imported.hero_bb_won == GROUND_TRUTH_HERO_BB_WON
    assert imported.result == "Hero wins"
    assert imported.source_type == "cv_import"
    assert imported.hero_position == "SB"
    # An import may never land a hand as the operator's own attestation.
    assert imported.review_status == "needs_correction"

    players = db.fetch_players_by_hand(result.hand_id)
    actions = db.fetch_actions_by_hand(result.hand_id)
    assert {player.player_name for player in players} == {"Hero", "Villain"}
    assert sum(1 for player in players if player.is_hero) == 1
    assert len(actions) == 9
    assert {action.street for action in actions} == {
        "preflop",
        "flop",
        "turn",
        "river",
    }
    assert [action.amount for action in actions if action.street == "river"] == [
        12.0,
        12.0,
    ]

    # The evidence that says WHICH recording, WHICH frames and WHICH models this
    # hand came from is the whole provenance chain. It crossed the exporter, a JSON
    # file, the import validator and SQLite; it has to arrive unchanged.
    evidence = parse_completion_evidence(imported.completion_evidence)
    assert evidence.first_source_timestamp_s == GROUND_TRUTH_T_START
    assert evidence.last_source_timestamp_s == GROUND_TRUTH_T_END
    assert evidence.source_frames == GROUND_TRUTH_FRAME_REFS
    assert evidence.layout_profile == GROUND_TRUTH_LAYOUT_PROFILE
    assert evidence.pipeline_version == GROUND_TRUTH_PIPELINE_VERSION
    assert evidence.model_versions == GROUND_TRUTH_MODEL_VERSIONS
    identity = evidence.extra.get(validated_hand_import.CV_TIMELINE_IDENTITY_KEY)
    assert identity == {
        "job_id": job.id,
        "timeline_hand_number": TIMELINE_HAND_NUMBER,
    }

    # The import took its own rollback point, scoped to the job, and it holds the
    # state from BEFORE the hand landed.
    preimport = find_snapshots(
        chain.paths["backups"], purpose="preimport", scope=f"job{job.id}"
    )
    assert len(preimport) == 1
    with _open_snapshot_read_only(preimport[0]) as connection:
        before = connection.execute("SELECT COUNT(*) FROM hands").fetchone()[0]
    assert before == 0
    assert len(db.fetch_hands_by_session(linked.session_id)) == 1

    db.close()


def _ingest_and_queue(
    chain: _ChainEnv, monkeypatch: pytest.MonkeyPatch, timeline: dict[str, Any]
) -> tuple[int, int, Path]:
    """Drive the chain up to a running job, ready for the worker. Returns ids."""
    db = PokerDatabase(str(chain.db_path))
    db.init_db()
    with chain.source.open("rb") as handle:
        ingested = video_ingest.ingest_uploaded_video(
            handle, chain.source.name, chain.paths["videos"]
        )
    stored = db.create_video(
        VideoRecord(
            original_filename=chain.source.name,
            stored_path=str(ingested.path),
            file_size_bytes=ingested.file_size_bytes,
            content_sha256=ingested.content_sha256,
            duration_seconds=ingested.metadata.duration_seconds,
            fps=ingested.metadata.fps,
            width=ingested.metadata.width,
            height=ingested.metadata.height,
            frame_count=ingested.metadata.frame_count,
        )
    )
    monkeypatch.setattr(
        cv_jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=424242)
    )
    job = cv_jobs.start_cv_job(db, stored.id, str(ingested.path), SESSION_NAME)
    monkeypatch.setattr(run_cv_job, "_run_pipeline", _stub_pipeline(chain, timeline))
    db.close()
    return job.id, stored.id, ingested.path


def test_rerunning_a_finished_job_is_refused_before_any_artifact_is_touched(
    chain: _ChainEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second worker on a completed job id must not erase what the first recorded.

    The worker is a documented recovery entrypoint that takes a job id, and its
    terminal write carries an optimistic status guard that ``update_processing_job``
    answers by returning the row unchanged rather than raising. Re-running a
    completed id therefore re-ran the whole reconstruction, rewrote the timeline,
    took a second backup, overwrote the finished row's progress and message with
    this run's heartbeats, silently lost its own completion to the guard, and
    returned 0.
    """
    timeline = _spine_timeline()
    job_id, _video_id, video_path = _ingest_and_queue(chain, monkeypatch, timeline)
    assert (
        run_cv_job.run_job(
            job_id=job_id,
            video_path=video_path,
            session_name=SESSION_NAME,
            db_path=chain.db_path,
        )
        == 0
    )

    db = PokerDatabase(str(chain.db_path))
    finished = db.fetch_processing_job(job_id)
    assert finished is not None
    assert finished.status == "completed"
    db.close()

    timeline_path = chain.paths["cv_timelines"] / f"job_{job_id}_timeline.json"
    timeline_bytes = timeline_path.read_bytes()
    backups_before = len(find_snapshots(chain.paths["backups"], purpose="routine"))
    runs_before = len(chain.pipeline_commands)

    exit_code = run_cv_job.run_job(
        job_id=job_id,
        video_path=video_path,
        session_name=SESSION_NAME,
        db_path=chain.db_path,
    )

    db = PokerDatabase(str(chain.db_path))
    after = db.fetch_processing_job(job_id)
    assert after is not None
    # The finished row still says what the run that actually finished recorded.
    assert after.status == "completed"
    assert after.progress_percent == 100
    assert after.message == finished.message
    assert "hands exported" in after.message
    assert "verified" in after.message
    db.close()

    # Nothing was read from the recording and nothing was written.
    assert len(chain.pipeline_commands) == runs_before
    assert timeline_path.read_bytes() == timeline_bytes
    assert len(find_snapshots(chain.paths["backups"], purpose="routine")) == backups_before
    assert exit_code == 1


def test_worker_does_not_report_success_when_the_row_refuses_its_completion(
    chain: _ChainEnv, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A reconstruction the reconciler failed underneath must not exit 0 in silence.

    ``reconcile_stuck_jobs`` runs on every app rerun and fails a job whose heartbeat
    has expired; the worker heartbeats around, but not during, the export, the
    backup and the isolated restore that verifies it. When the row is failed by the
    time the worker writes its terminal state, the optimistic guard drops that
    write and ``update_processing_job`` returns the unchanged row, so the run's
    outcome is recorded nowhere -- and the worker used to answer 0 anyway.
    """
    timeline = _spine_timeline()
    job_id, video_id, video_path = _ingest_and_queue(chain, monkeypatch, timeline)
    monkeypatch.setattr(cv_jobs, "_pid_is_alive", lambda pid: False)

    inner = run_cv_job._run_pipeline

    def reconcile_midway(command, db, job, deadline, progress, limits=None):  # noqa: ANN001
        inner(command, db, job, deadline, progress, limits)
        # The app's own reconciler, on its own connection, exactly as a Streamlit
        # rerun would call it while this worker is still exporting.
        watcher = PokerDatabase(str(chain.db_path))
        assert cv_jobs.reconcile_stuck_jobs(
            watcher, stale_after=timedelta(seconds=0)
        ) == [job]
        watcher.close()

    monkeypatch.setattr(run_cv_job, "_run_pipeline", reconcile_midway)

    exit_code = run_cv_job.run_job(
        job_id=job_id,
        video_path=video_path,
        session_name=SESSION_NAME,
        db_path=chain.db_path,
    )

    assert exit_code == 1
    assert f"job #{job_id}" in capsys.readouterr().err

    db = PokerDatabase(str(chain.db_path))
    after = db.fetch_processing_job(job_id)
    assert after is not None
    assert after.status == "failed"
    # The reconstruction really did finish, so its artifacts are kept and the
    # destination link stands; only the row could not be told.
    linked = db.fetch_video(video_id)
    assert linked is not None
    assert linked.session_id is not None
    db.close()
    assert (chain.paths["cv_timelines"] / f"job_{job_id}_timeline.json").is_file()
    assert (chain.paths["exports"] / f"job_{job_id}_session.json").is_file()
    assert find_snapshots(chain.paths["backups"], purpose="routine")
