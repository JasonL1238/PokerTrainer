"""A snapshot is only a recovery point if it survives, is inventoried, and restores.

Three properties are pinned here, each of which was previously assumed rather
than checked: a snapshot taken before a destructive operation survives routine
rotation and is taken once per operation rather than once per row; a snapshot
records the external artifacts its rows point at, since those are never copied
with it; and a retained snapshot is restored in isolation and read back deeply
enough that an empty or hollowed-out one cannot be reported as good.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from poker_tracker.maintenance.data_health import (
    CheckResult,
    audit_data_health,
    verify_snapshot,
)
from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence.backup import (
    BACKUP_KEEP_COUNT,
    LIVE_DB_PATH,
    PINNED_KEEP_COUNT,
    PREIMPORT_KEEP_COUNT,
    backup_database,
    backups_dir_for,
    find_snapshots,
)
from poker_tracker.persistence.backup_inventory import inventory_path, load_inventory
from poker_tracker.persistence.db import DEFAULT_DB_PATH, PokerDatabase
from poker_tracker.persistence.models import (
    Hand,
    ProcessingJob,
    ReconstructionFrameReview,
    Session,
    VideoRecord,
)
from poker_tracker.services import validated_hand_import
from poker_tracker.ui import reconstruction_review


def _spine_hand(**overrides: Any) -> dict[str, Any]:
    """One complete reconstructed hand, in the shape the timeline exporter emits."""
    hand: dict[str, Any] = {
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


def _importable_timeline(count: int) -> dict[str, Any]:
    """A timeline whose middle hands all pass the autonomous import gate.

    The first and last hands are boundary padding: a hand at either end of the
    recording is refused as partial, so a batch of N importable hands needs N + 2.
    """
    images = [f"{index}.jpg" for index in range(count + 2)]
    return {
        "states": [
            {
                "image": image,
                "time_s": float(index),
                "board_cards": ["2c", "7d", "9h", "Ts", "Jc"],
                "hero_cards": ["As", "Kd"],
            }
            for index, image in enumerate(images)
        ],
        "hands": [
            _spine_hand(
                hand_number=index + 1,
                source_images=[image],
                terminal_event="showdown",
                t_start=float(index) - 0.25,
                t_end=float(index) + 0.25,
            )
            for index, image in enumerate(images)
        ],
    }


def _seed_cv_job(
    db: PokerDatabase,
    root: Path,
    timeline: dict[str, Any],
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, int]:
    session = db.create_session(Session(name="Dest", platform="ClubWPT Gold"))
    video = db.create_video(
        VideoRecord(
            original_filename="clip.mp4",
            stored_path=str(root / "clip.mp4"),
            file_size_bytes=0,
            session_id=session.id,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id, job_type="cv_reconstruction", status="completed"
        )
    )
    timelines = root / "cv_timelines"
    timelines.mkdir(exist_ok=True)
    (timelines / f"job_{job.id}_timeline.json").write_text(
        json.dumps(timeline), encoding="utf-8"
    )
    monkeypatch.setattr(reconstruction_review, "CV_TIMELINES_DIR", timelines)
    monkeypatch.setattr(validated_hand_import, "CV_TIMELINES_DIR", timelines)
    monkeypatch.setattr(validated_hand_import, "DATA_DIR", root)
    for hand in timeline["hands"]:
        for image in hand["source_images"]:
            db.upsert_reconstruction_frame_review(
                ReconstructionFrameReview(
                    job_id=job.id,
                    hand_number=hand["hand_number"],
                    source_image=image,
                    timestamp_seconds=0.0,
                    status="correct",
                )
            )
    return job.id, session.id


def _seeded_study_database(path: Path) -> PokerDatabase:
    """One session with one completed hand, its players, actions and settlement."""
    from poker_tracker.persistence.models import (
        Action,
        HandPlayer,
        HandSettlement,
    )

    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Study", date_played=date(2026, 1, 1)))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="As Kd",
            board_cards="2c 7d 9h",
            completion_status="complete",
        )
    )
    db.create_hand_player(
        HandPlayer(hand_id=hand.id, player_key="hero", player_name="Hero", is_hero=True)
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            street="preflop",
            action_index=1,
            action_type="bet",
            player_name="Hero",
            amount=3.0,
        )
    )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    return db


def test_a_batch_import_takes_one_pinned_snapshot_not_one_per_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eight hands used to mean eight copies, and the rollback point rotated away.

    The snapshot competed for the five-slot routine rotation, so the pre-import
    state was evicted by the imports it was taken to protect.
    """
    db = PokerDatabase(tmp_path / "tracker.sqlite3")
    db.init_db()
    job_id, session_id = _seed_cv_job(
        db, tmp_path, _importable_timeline(6), monkeypatch=monkeypatch
    )
    backups = tmp_path / "backups"
    job_snapshot = backup_database(Path(db.db_path), backups)

    results = validated_hand_import.import_all_autonomous_eligible(
        db, job_id, data_dir=tmp_path
    )

    imported = [item for item in results if item.status == "imported"]
    # More hands than the rotation has slots: with one snapshot per hand the
    # earliest copies, and the CV job's own, were evicted by the batch itself.
    assert len(imported) > BACKUP_KEEP_COUNT, results
    assert len(db.fetch_hands_by_session(session_id)) == len(imported)
    preimport = find_snapshots(backups, purpose="preimport", scope=f"job{job_id}")
    assert len(preimport) == 1
    # The rollback point still holds the state that preceded the whole batch.
    restored = PokerDatabase(preimport[0])
    assert restored.fetch_hands_by_session(session_id) == []
    restored.close()
    # And the snapshot the CV job itself wrote was not pushed out by the batch.
    assert job_snapshot.is_file()
    db.close()


def test_a_second_import_from_the_same_job_reuses_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interactive path adds hands one click at a time, in separate reruns.

    The retained snapshot is its own marker, so "already snapshotted for this
    job" survives a restart with no marker file and no schema column.
    """
    db = PokerDatabase(tmp_path / "tracker.sqlite3")
    db.init_db()
    job_id, _ = _seed_cv_job(
        db, tmp_path, _importable_timeline(3), monkeypatch=monkeypatch
    )
    backups = tmp_path / "backups"

    first = validated_hand_import.ensure_hand_imported(
        db, job_id, 2, mode="auto", data_dir=tmp_path
    )
    second = validated_hand_import.ensure_hand_imported(
        db, job_id, 3, mode="auto", data_dir=tmp_path
    )

    assert (first.status, second.status) == ("imported", "imported")
    assert len(find_snapshots(backups, purpose="preimport", scope=f"job{job_id}")) == 1
    db.close()


def test_an_unverifiable_snapshot_blocks_the_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An import with no proven rollback point must not proceed."""
    db = PokerDatabase(tmp_path / "tracker.sqlite3")
    db.init_db()
    job_id, session_id = _seed_cv_job(
        db, tmp_path, _importable_timeline(1), monkeypatch=monkeypatch
    )
    monkeypatch.setattr(
        validated_hand_import,
        "verify_snapshot",
        lambda *args, **kwargs: CheckResult(
            "backup_verification", "fail", "restored copy is empty", ("no sessions",)
        ),
    )

    result = validated_hand_import.ensure_hand_imported(
        db, job_id, 2, mode="auto", data_dir=tmp_path
    )

    assert result.status == "blocked"
    assert "no rollback point" in result.message
    assert db.fetch_hands_by_session(session_id) == []
    db.close()


def test_each_snapshot_purpose_retains_its_own_pool(tmp_path: Path) -> None:
    """Retention follows from what a snapshot is for, not from when it arrived."""
    database = tmp_path / "live.sqlite3"
    sqlite3.connect(database).close()
    backups = tmp_path / "backups"

    premigration = backup_database(database, backups, pinned=True)
    for index in range(PREIMPORT_KEEP_COUNT + 2):
        backup_database(database, backups, purpose="preimport", scope=f"job{index}")
    for _ in range(BACKUP_KEEP_COUNT + 3):
        backup_database(database, backups)

    assert premigration.is_file()
    assert len(find_snapshots(backups, purpose="premigration")) == 1
    assert len(find_snapshots(backups, purpose="preimport")) == PREIMPORT_KEEP_COUNT
    assert len(find_snapshots(backups, purpose="routine")) == BACKUP_KEEP_COUNT
    assert PINNED_KEEP_COUNT >= 1


def test_the_inventory_records_every_artifact_the_snapshot_references(
    tmp_path: Path,
) -> None:
    """A snapshot restores rows; the files those rows point at are never copied.

    Without the inventory a restored database is a set of paths whose current
    state nobody can enumerate, let alone check.
    """
    data_dir = tmp_path / "data"
    videos = data_dir / "videos"
    timelines = data_dir / "cv_timelines"
    for directory in (videos, timelines):
        directory.mkdir(parents=True)
    recording = videos / "clip.mp4"
    recording.write_bytes(b"recording-bytes")
    database = tmp_path / "poker_tracker.db"
    db = _seeded_study_database(database)
    session = db.fetch_sessions()[0]
    video = db.create_video(
        VideoRecord(
            original_filename="clip.mp4",
            stored_path=str(recording),
            file_size_bytes=recording.stat().st_size,
            session_id=session.id,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id, job_type="cv_reconstruction", status="completed"
        )
    )
    timeline = timelines / f"job_{job.id}_timeline.json"
    timeline.write_text("{}", encoding="utf-8")
    db.close()

    snapshot = backup_database(
        database,
        data_dir / "backups",
        purpose="preimport",
        scope="job1",
        data_dir=data_dir,
    )

    inventory = load_inventory(snapshot)
    assert inventory is not None
    by_source = {
        (entry["source"], entry["path"]): entry for entry in inventory["artifacts"]
    }
    assert ("videos.stored_path", str(recording)) in by_source
    assert ("processing_jobs.timeline", str(timeline)) in by_source
    recorded = by_source[("videos.stored_path", str(recording))]
    assert recorded["present"] is True
    assert recorded["bytes"] == recording.stat().st_size
    assert recorded["sha256"]
    assert inventory["unreadable_sources"] == []
    assert inventory["error"] is None


def test_rotation_removes_a_snapshot_and_its_inventory_together(
    tmp_path: Path,
) -> None:
    """An inventory outliving its snapshot describes a restore point that is gone."""
    database = tmp_path / "live.sqlite3"
    db = PokerDatabase(database)
    db.init_db()
    db.close()
    backups = tmp_path / "backups"

    first = backup_database(database, backups)
    assert inventory_path(first).is_file()
    for _ in range(BACKUP_KEEP_COUNT):
        backup_database(database, backups)

    assert not first.exists()
    assert not inventory_path(first).exists()
    survivors = find_snapshots(backups, purpose="routine")
    assert len(survivors) == BACKUP_KEEP_COUNT
    assert all(inventory_path(path).is_file() for path in survivors)


def test_an_artifact_deleted_after_the_snapshot_is_named_in_the_audit(
    tmp_path: Path,
) -> None:
    """Retention sweeping a recording away must not be discoverable only on restore."""
    data_dir = tmp_path / "data"
    videos = data_dir / "videos"
    videos.mkdir(parents=True)
    recording = videos / "clip.mp4"
    recording.write_bytes(b"recording-bytes")
    database = tmp_path / "poker_tracker.db"
    db = _seeded_study_database(database)
    db.create_video(
        VideoRecord(
            original_filename="clip.mp4",
            stored_path=str(recording),
            file_size_bytes=recording.stat().st_size,
            session_id=db.fetch_sessions()[0].id,
        )
    )
    db.close()
    backups = data_dir / "backups"
    backup_database(database, backups)
    recording.unlink()

    report = audit_data_health(database, data_dir=data_dir, backup_dir=backups)

    check = next(item for item in report.checks if item.name == "backups")
    assert check.status == "warning"
    assert any(
        "artifact missing since the snapshot" in detail and "clip.mp4" in detail
        for detail in check.details
    )


def test_a_backup_holding_no_hands_is_not_reported_as_a_clean_restore(
    tmp_path: Path,
) -> None:
    """quick_check passes just as happily on an empty database.

    A snapshot taken from a truncated file used to be reported as having passed a
    restore drill, which is the only verified recovery point saying it is good
    while holding nothing.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = tmp_path / "poker_tracker.db"
    db = _seeded_study_database(database)
    db.close()
    backups = data_dir / "backups"
    backups.mkdir()
    empty = PokerDatabase(tmp_path / "empty.sqlite3")
    empty.init_db()
    empty.close()
    shutil.copy2(
        tmp_path / "empty.sqlite3",
        backups / "poker_tracker_20260101T000000000000Z.sqlite3",
    )

    report = audit_data_health(database, data_dir=data_dir, backup_dir=backups)

    check = next(item for item in report.checks if item.name == "backups")
    assert check.status == "warning"
    assert any("cannot restore any study history" in detail for detail in check.details)
    assert any("live database: 1 session(s), 1 hand(s)" in detail for detail in check.details)


def test_a_restored_backup_reads_one_completed_hand_end_to_end(
    tmp_path: Path,
) -> None:
    """The drill has to open the history, not just the file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = tmp_path / "poker_tracker.db"
    db = _seeded_study_database(database)
    db.close()
    snapshot = backup_database(database, data_dir / "backups")

    healthy = verify_snapshot(snapshot, live_database=database, data_dir=data_dir)
    assert healthy.status == "pass"
    assert "1 session(s), 1 hand(s), 1 completed" in healthy.message

    with sqlite3.connect(snapshot) as connection:
        connection.execute("DELETE FROM hand_players")
    hollow = verify_snapshot(snapshot, live_database=database, data_dir=data_dir)
    assert hollow.status == "warning"
    assert any("no hand_players row" in detail for detail in hollow.details)

    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE hands SET completion_evidence = 'not json'")
    corrupt = verify_snapshot(snapshot, live_database=database, data_dir=data_dir)
    assert corrupt.status == "fail"
    assert any("completion_evidence" in detail for detail in corrupt.details)


def test_a_snapshot_verifies_without_being_told_where_the_live_database_is(
    tmp_path: Path,
) -> None:
    """A caller that only has the snapshot must not have it compared with itself.

    The same-file and hard-link rejections exist so a "backup" that is really the
    live database cannot pass; with no live database named they have nothing to
    compare against and must not fire.
    """
    database = tmp_path / "poker_tracker.db"
    db = _seeded_study_database(database)
    db.close()
    snapshot = backup_database(database, tmp_path / "backups")

    result = verify_snapshot(snapshot)

    assert result.status == "pass", result.details
    assert "1 hand(s)" in result.message


def test_every_evidence_purpose_is_audited(tmp_path: Path) -> None:
    """A snapshot outside the routine glob must not be invisible to the audit."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = tmp_path / "poker_tracker.db"
    db = _seeded_study_database(database)
    db.close()
    backups = data_dir / "backups"
    backup_database(database, backups, pinned=True)
    backup_database(database, backups, purpose="preimport", scope="job7")

    report = audit_data_health(database, data_dir=data_dir, backup_dir=backups)

    check = next(item for item in report.checks if item.name == "backups")
    assert check.status == "pass"
    assert check.message.startswith("All 2 retained backup(s)")
    assert any("preimport-job7" in detail for detail in check.details)


def test_issue_evidence_is_verified_on_the_restored_copy(tmp_path: Path) -> None:
    """The frozen snapshot is the only record of what a flagged hand looked like."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = tmp_path / "poker_tracker.db"
    db = _seeded_study_database(database)
    hand_id = db.fetch_hands_by_session(db.fetch_sessions()[0].id)[0].id
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO hand_issues ("
            "  hand_id, description, evidence_snapshot, created_at, updated_at"
            ") VALUES (?, 'wrong flop', '{\"hand\": {}}', ?, ?)",
            (hand_id, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        )
    db.close()
    snapshot = backup_database(database, data_dir / "backups")
    assert (
        verify_snapshot(snapshot, live_database=database, data_dir=data_dir).status
        == "pass"
    )

    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE hand_issues SET evidence_snapshot = '{}'")
    blanked = verify_snapshot(snapshot, live_database=database, data_dir=data_dir)
    assert blanked.status == "warning"
    assert any("evidence_snapshot is empty" in detail for detail in blanked.details)

    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE hand_issues SET evidence_snapshot = 'broken'")
    broken = verify_snapshot(snapshot, live_database=database, data_dir=data_dir)
    assert broken.status == "fail"
    assert any("not readable JSON" in detail for detail in broken.details)


def _run_cv_job_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db_path: Path, job_id: int
) -> str:
    """Run the worker with the pipeline stubbed out, and return its final message."""
    from poker_tracker.ui import run_cv_job

    paths = {
        name: tmp_path / name
        for name in ("cv_timelines", "exports", "backups", "frames")
    }
    for path in paths.values():
        path.mkdir(exist_ok=True)
    paths["data"] = tmp_path
    monkeypatch.setattr(run_cv_job, "ensure_data_directories", lambda: paths)
    monkeypatch.setattr(run_cv_job, "_run_pipeline", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_cv_job,
        "export_timeline",
        lambda *args, **kwargs: {"cv_import_summary": {"exported_hands": 1}},
    )
    monkeypatch.setattr(run_cv_job, "require_playable_video", lambda *args: None)
    monkeypatch.setattr(
        run_cv_job, "assert_stored_video_matches_record", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        run_cv_job, "resolve_stored_video_path", lambda path, **kwargs: path
    )
    assert (
        run_cv_job.run_job(
            job_id=job_id,
            video_path=tmp_path / "clip.mp4",
            session_name="Reconstructed",
            db_path=db_path,
        )
        == 0
    )
    checked = PokerDatabase(db_path)
    message = checked.fetch_processing_job(job_id).message
    checked.close()
    return message


def test_the_cv_job_verifies_the_backup_it_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification that waits for a maintenance flag verifies nothing for months.

    The worker is the one place that both writes a snapshot and has an
    operator-visible surface to report on, so the drill runs there.
    """
    from poker_tracker.ui import run_cv_job

    db_path = tmp_path / "tracker.sqlite3"
    db = _seeded_study_database(db_path)
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"clip")
    video = db.create_video(
        VideoRecord(
            original_filename="clip.mp4",
            stored_path=str(video_file),
            file_size_bytes=video_file.stat().st_size,
        )
    )
    jobs = [
        db.create_processing_job(
            ProcessingJob(
                video_id=video.id, job_type="cv_reconstruction", status="running"
            )
        ).id
        for _ in range(2)
    ]
    db.close()

    verified = _run_cv_job_for(tmp_path, monkeypatch, db_path, jobs[0])
    assert "backup" in verified and "verified" in verified

    monkeypatch.setattr(
        run_cv_job,
        "verify_snapshot",
        lambda *args, **kwargs: CheckResult(
            "backup_verification", "fail", "corrupt", ("quick_check: page 3 broken",)
        ),
    )
    failed = _run_cv_job_for(tmp_path, monkeypatch, db_path, jobs[1])
    assert "FAILED VERIFICATION" in failed
    assert "page 3 broken" in failed


def test_the_drill_changes_neither_the_live_database_nor_the_backup_directory(
    tmp_path: Path,
) -> None:
    """The isolated restore is the one operation allowed nowhere near live data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = tmp_path / "poker_tracker.db"
    db = _seeded_study_database(database)
    db.close()
    backups = data_dir / "backups"
    backup_database(database, backups)
    backup_database(database, backups, pinned=True)
    live_before = database.read_bytes()
    before = sorted(
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in backups.iterdir()
    )

    report = audit_data_health(database, data_dir=data_dir, backup_dir=backups)

    assert report.healthy
    assert database.read_bytes() == live_before
    assert (
        sorted(
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in backups.iterdir()
        )
        == before
    )


# --- A snapshot belongs with the database it protects -----------------------


def test_a_database_that_is_not_the_live_one_snapshots_beside_itself(tmp_path: Path):
    """Migrating a copy used to evict the operator's real rollback points.

    Anything that opens a database runs the migration chain, and the chain takes
    a pre-migration snapshot. A temporary fixture, a restored copy, or a backup
    being audited therefore wrote into the live data/backups and competed for
    the pinned slots that protect the actual study history.
    """
    elsewhere = tmp_path / "restored"
    elsewhere.mkdir()
    database = elsewhere / "copy.sqlite3"
    _seeded_study_database(database).close()

    destination = backup_database(database, pinned=True)

    assert destination.parent == elsewhere / "backups"
    assert backup_module.BACKUPS_DIR not in destination.parents


def test_the_live_database_still_snapshots_where_operators_look(tmp_path: Path):
    """The runbooks, the restore drill and the audit all expect BACKUPS_DIR."""
    assert backups_dir_for(Path(DEFAULT_DB_PATH)) == backup_module.BACKUPS_DIR
    # Resolved, not compared as text: a symlinked or non-normalized spelling of
    # the live database is still the live database.
    spelled_differently = Path(DEFAULT_DB_PATH).parent / "." / Path(DEFAULT_DB_PATH).name
    assert backups_dir_for(spelled_differently) == backup_module.BACKUPS_DIR


def test_the_duplicated_live_database_constant_cannot_drift():
    """backup.py cannot import db.py, so the two spellings are pinned together."""
    assert Path(DEFAULT_DB_PATH).resolve() == LIVE_DB_PATH.resolve()
