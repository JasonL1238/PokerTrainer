from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

from poker_tracker.maintenance.data_health import audit_data_health, main
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase


def _create_database(
    path: Path,
    *,
    video_path: Path | None = None,
    frame_path: Path | None = None,
    review_image_path: Path | None = None,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                date_played TEXT NOT NULL
            );
            CREATE TABLE hands (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL,
                hand_number INTEGER NOT NULL,
                review_status TEXT NOT NULL,
                source_type TEXT NOT NULL,
                completion_status TEXT NOT NULL DEFAULT 'not_applicable',
                completion_evidence TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE TABLE hand_players (
                id INTEGER PRIMARY KEY,
                hand_id INTEGER NOT NULL,
                player_key TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id)
            );
            CREATE TABLE actions (
                id INTEGER PRIMARY KEY,
                hand_id INTEGER NOT NULL,
                street TEXT NOT NULL,
                action_index INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id)
            );
            CREATE TABLE hand_reviews (
                id INTEGER PRIMARY KEY,
                hand_id INTEGER NOT NULL,
                FOREIGN KEY (hand_id) REFERENCES hands(id)
            );
            CREATE TABLE coaching_reviews (
                id INTEGER PRIMARY KEY,
                review_type TEXT NOT NULL
            );
            CREATE TABLE videos (
                id INTEGER PRIMARY KEY,
                stored_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE processing_jobs (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                video_id INTEGER NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id)
            );
            CREATE TABLE extracted_frames (
                id INTEGER PRIMARY KEY,
                video_id INTEGER NOT NULL,
                job_id INTEGER NOT NULL,
                image_path TEXT NOT NULL
            );
            CREATE TABLE reconstruction_frame_reviews (
                id INTEGER PRIMARY KEY,
                source_image TEXT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO schema_metadata (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.execute(
            "INSERT INTO sessions (id, name, date_played) VALUES (1, 'Test', '2026-07-28')"
        )
        connection.execute(
            """
            INSERT INTO hands (
                id, session_id, hand_number, review_status, source_type
            ) VALUES (1, 1, 1, 'unreviewed', 'manual')
            """
        )
        if video_path is not None:
            connection.execute(
                """
                INSERT INTO videos (id, stored_path, file_size_bytes)
                VALUES (1, ?, ?)
                """,
                (str(video_path), video_path.stat().st_size),
            )
            connection.execute(
                """
                INSERT INTO processing_jobs (id, status, video_id)
                VALUES (1, 'completed', 1)
                """
            )
        if frame_path is not None:
            connection.execute(
                """
                INSERT INTO extracted_frames (id, video_id, job_id, image_path)
                VALUES (1, 1, 1, ?)
                """,
                (str(frame_path),),
            )
        if review_image_path is not None:
            connection.execute(
                """
                INSERT INTO reconstruction_frame_reviews (id, source_image)
                VALUES (1, ?)
                """,
                (str(review_image_path),),
            )


def _checks_by_name(report) -> dict[str, object]:
    return {check.name: check for check in report.checks}


def test_audit_passes_for_valid_database_artifacts_and_backup_restore(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    backup_dir = data_dir / "backups"
    backup_dir.mkdir(parents=True)
    video = data_dir / "videos" / "session.mp4"
    frame = data_dir / "frames" / "frame.jpg"
    review = data_dir / "frames" / "review.jpg"
    for path, content in (
        (video, b"video"),
        (frame, b"frame"),
        (review, b"review"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    database = tmp_path / "poker_tracker.db"
    _create_database(
        database,
        video_path=video,
        frame_path=frame,
        review_image_path=review,
    )
    shutil.copy2(database, backup_dir / "poker_tracker_20260728T000000Z.sqlite3")

    report = audit_data_health(
        database,
        data_dir=data_dir,
        expected_schema_version=SCHEMA_VERSION,
        restore_backups=True,
    )

    checks = _checks_by_name(report)
    assert report.healthy
    assert checks["sqlite_quick_check"].status == "pass"
    assert checks["foreign_key_check"].status == "pass"
    assert checks["artifact_files"].message.startswith("All 3")
    assert "restore drill" in checks["backups"].message


def test_audit_accepts_the_current_application_schema(tmp_path: Path) -> None:
    database = tmp_path / "poker_tracker.db"
    app_database = PokerDatabase(database)
    app_database.init_db()
    app_database.close()

    report = audit_data_health(
        database,
        data_dir=tmp_path / "data",
        expected_schema_version=SCHEMA_VERSION,
    )

    assert report.healthy
    assert _checks_by_name(report)["schema_contract"].status == "pass"


def test_audit_reports_missing_artifacts_and_changed_video_size(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    video = data_dir / "videos" / "session.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original")
    missing_frame = data_dir / "frames" / "missing.jpg"
    database = tmp_path / "poker_tracker.db"
    _create_database(database, video_path=video, frame_path=missing_frame)
    video.write_bytes(b"changed-size")

    report = audit_data_health(database, data_dir=data_dir)

    artifact_check = _checks_by_name(report)["artifact_files"]
    assert not report.healthy
    assert artifact_check.status == "fail"
    assert "1 missing artifact(s)" in artifact_check.message
    assert "1 video size mismatch(es)" in artifact_check.message


def test_audit_reports_broken_foreign_keys(tmp_path: Path) -> None:
    database = tmp_path / "poker_tracker.db"
    _create_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO hands (
                id, session_id, hand_number, review_status, source_type
            ) VALUES (2, 999, 2, 'unreviewed', 'manual')
            """
        )

    report = audit_data_health(database, data_dir=tmp_path / "data")

    check = _checks_by_name(report)["foreign_key_check"]
    assert not report.healthy
    assert check.status == "fail"
    assert "hands row 2" in check.details[0]


def test_audit_does_not_create_a_missing_database(tmp_path: Path) -> None:
    database = tmp_path / "missing.db"

    report = audit_data_health(database, data_dir=tmp_path / "data")

    assert not report.healthy
    assert not database.exists()
    assert _checks_by_name(report)["database_file"].status == "fail"


def test_audit_reports_corrupt_retained_backup(tmp_path: Path) -> None:
    database = tmp_path / "poker_tracker.db"
    _create_database(database)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    (backup_dir / "poker_tracker_corrupt.sqlite3").write_bytes(b"not sqlite")

    report = audit_data_health(
        database,
        data_dir=tmp_path / "data",
        backup_dir=backup_dir,
        restore_backups=True,
    )

    backup_check = _checks_by_name(report)["backups"]
    assert not report.healthy
    assert backup_check.status == "fail"
    assert "poker_tracker_corrupt.sqlite3" in backup_check.details[0]


def test_audit_reports_backup_with_invalid_schema_metadata(tmp_path: Path) -> None:
    database = tmp_path / "poker_tracker.db"
    _create_database(database)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    backup = backup_dir / "poker_tracker_invalid.sqlite3"
    shutil.copy2(database, backup)
    with sqlite3.connect(backup) as connection:
        connection.execute(
            "UPDATE schema_metadata SET value = 'invalid' WHERE key = 'schema_version'"
        )

    report = audit_data_health(
        database,
        data_dir=tmp_path / "data",
        backup_dir=backup_dir,
    )

    backup_check = _checks_by_name(report)["backups"]
    assert not report.healthy
    assert backup_check.status == "fail"
    assert "schema version is not an integer" in backup_check.details[0]


def test_audit_rejects_metadata_only_database(tmp_path: Path) -> None:
    database = tmp_path / "metadata-only.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_metadata VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    report = audit_data_health(
        database,
        data_dir=tmp_path / "data",
        expected_schema_version=SCHEMA_VERSION,
    )

    schema_check = _checks_by_name(report)["schema_contract"]
    assert not report.healthy
    assert schema_check.status == "fail"
    assert any("missing table: sessions" == detail for detail in schema_check.details)


def test_audit_rejects_missing_required_core_column(tmp_path: Path) -> None:
    database = tmp_path / "poker_tracker.db"
    _create_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE videos RENAME TO old_videos")
        connection.execute(
            "CREATE TABLE videos (id INTEGER PRIMARY KEY, stored_path TEXT NOT NULL)"
        )

    report = audit_data_health(database, data_dir=tmp_path / "data")

    schema_check = _checks_by_name(report)["schema_contract"]
    assert not report.healthy
    assert schema_check.status == "fail"
    assert "videos: missing column(s) content_sha256, file_size_bytes" in schema_check.details


def test_audit_rejects_backup_symlink_and_hard_link_to_live_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "poker_tracker.db"
    _create_database(database)

    symlink_dir = tmp_path / "symlink_backups"
    symlink_dir.mkdir()
    (symlink_dir / "poker_tracker_symlink.sqlite3").symlink_to(database)
    symlink_report = audit_data_health(
        database,
        data_dir=tmp_path / "data",
        backup_dir=symlink_dir,
        restore_backups=True,
    )
    assert not symlink_report.healthy
    assert "symlinks are not independent" in _checks_by_name(symlink_report)[
        "backups"
    ].details[0]

    hardlink_dir = tmp_path / "hardlink_backups"
    hardlink_dir.mkdir()
    os.link(database, hardlink_dir / "poker_tracker_hardlink.sqlite3")
    hardlink_report = audit_data_health(
        database,
        data_dir=tmp_path / "data",
        backup_dir=hardlink_dir,
        restore_backups=True,
    )
    assert not hardlink_report.healthy
    assert "hard-linked files are not independent" in _checks_by_name(
        hardlink_report
    )["backups"].details[0]


def test_audit_rejects_newer_backup_and_warns_for_older_backup(
    tmp_path: Path,
) -> None:
    database = tmp_path / "poker_tracker.db"
    _create_database(database)

    newer_dir = tmp_path / "newer"
    newer_dir.mkdir()
    newer = newer_dir / "poker_tracker_newer.sqlite3"
    shutil.copy2(database, newer)
    with sqlite3.connect(newer) as connection:
        connection.execute(
            "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION + 1),),
        )
    newer_report = audit_data_health(
        database,
        data_dir=tmp_path / "data",
        backup_dir=newer_dir,
        expected_schema_version=SCHEMA_VERSION,
    )
    assert not newer_report.healthy
    assert "newer than this PokerTrainer build" in _checks_by_name(newer_report)[
        "backups"
    ].details[0]

    older_dir = tmp_path / "older"
    older_dir.mkdir()
    older = older_dir / "poker_tracker_older.sqlite3"
    shutil.copy2(database, older)
    with sqlite3.connect(older) as connection:
        connection.execute(
            "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION - 1),),
        )
    older_report = audit_data_health(
        database,
        data_dir=tmp_path / "data",
        backup_dir=older_dir,
        expected_schema_version=SCHEMA_VERSION,
    )
    backup_check = _checks_by_name(older_report)["backups"]
    assert older_report.healthy
    assert backup_check.status == "warning"
    assert "older than this build" in backup_check.details[0]


def test_relative_artifact_uses_valid_second_root(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    valid_frame = data_dir / "frames" / "item.jpg"
    valid_frame.parent.mkdir(parents=True)
    valid_frame.write_bytes(b"frame")
    video = data_dir / "videos" / "session.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    database = tmp_path / "poker_tracker.db"
    _create_database(database, video_path=video)
    (tmp_path / "frames" / "item.jpg").mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO extracted_frames (id, video_id, job_id, image_path)
            VALUES (1, 1, 1, 'frames/item.jpg')
            """
        )

    report = audit_data_health(database, data_dir=data_dir)

    assert _checks_by_name(report)["artifact_files"].status == "pass"


def test_artifact_failure_details_are_bounded_but_count_is_exact(
    tmp_path: Path,
) -> None:
    database = tmp_path / "poker_tracker.db"
    _create_database(database)
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO videos (id, stored_path, file_size_bytes)
            VALUES (?, ?, 1)
            """,
            [
                (index, str(tmp_path / f"missing-{index}.mp4"))
                for index in range(1, 26)
            ],
        )

    report = audit_data_health(database, data_dir=tmp_path / "data")

    artifact_check = _checks_by_name(report)["artifact_files"]
    assert "Found 25 missing artifact(s)" in artifact_check.message
    assert len(artifact_check.details) == 21
    assert artifact_check.details[-1] == "... and 5 more"


def test_cli_json_is_machine_readable_and_warnings_do_not_fail(
    tmp_path: Path,
    capsys,
) -> None:
    database = tmp_path / "poker_tracker.db"
    _create_database(database)

    exit_code = main(
        [
            "--db",
            str(database),
            "--data-dir",
            str(tmp_path / "data"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["healthy"] is True
    assert payload["has_warnings"] is True
    assert payload["checks"][-1]["name"] == "backups"


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="POSIX permissions require a non-root test process.",
)
def test_cli_json_fails_for_inaccessible_explicit_backup_directory(
    tmp_path: Path,
    capsys,
) -> None:
    database = tmp_path / "poker_tracker.db"
    _create_database(database)
    backup_dir = tmp_path / "locked-backups"
    backup_dir.mkdir()
    backup_dir.chmod(0)
    try:
        exit_code = main(
            [
                "--db",
                str(database),
                "--data-dir",
                str(tmp_path / "data"),
                "--backup-dir",
                str(backup_dir),
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
    finally:
        backup_dir.chmod(0o700)

    backup_check = payload["checks"][-1]
    assert exit_code == 1
    assert payload["healthy"] is False
    assert backup_check["name"] == "backups"
    assert backup_check["status"] == "fail"
    assert backup_check["message"] == "Backup directory cannot be read."


def _current_schema_database(path: Path) -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def test_every_recorded_path_column_is_audited(tmp_path: Path) -> None:
    """Solver outputs, per-action frames and regression fixtures were unaudited.

    _ARTIFACT_REFERENCES listed three of the nine columns retention protects, so
    a report could say every recorded artifact reference was present while six
    kinds of file dangled.
    """
    from poker_tracker.persistence.models import Hand, Session

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database = tmp_path / "poker_tracker.db"
    db = _current_schema_database(database)
    session = db.create_session(Session(name="Artifacts"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    files = {}
    for name in ("solver_result.json", "action_frame.jpg", "fixture.json"):
        path = data_dir / name
        path.write_bytes(b"present")
        files[name] = path
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO actions ("
            "  hand_id, street, action_index, action_type, player_name, source_image"
            ") VALUES (?, 'preflop', 1, 'bet', 'Hero', ?)",
            (hand.id, str(files["action_frame.jpg"])),
        )
        connection.execute(
            "INSERT INTO solver_runs (hand_id, input_hash, result_path, created_at)"
            " VALUES (?, 'hash', ?, '2026-01-01T00:00:00+00:00')",
            (hand.id, str(files["solver_result.json"])),
        )
        connection.execute(
            "INSERT INTO hand_issues (hand_id, description, created_at, updated_at)"
            " VALUES (?, 'issue', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
        , (hand.id,))
        connection.execute(
            "INSERT INTO regression_cases (issue_id, kind, fixture_path, created_at, updated_at)"
            " VALUES (1, 'timeline', ?, '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')",
            (str(files["fixture.json"]),),
        )
    db.close()

    healthy = audit_data_health(
        database, data_dir=data_dir, restore_backups=False
    )
    assert _checks_by_name(healthy)["artifact_files"].status == "pass"
    assert _checks_by_name(healthy)["artifact_files"].message.startswith("All 3")

    for path in files.values():
        path.unlink()
    report = audit_data_health(database, data_dir=data_dir, restore_backups=False)

    check = _checks_by_name(report)["artifact_files"]
    assert check.status == "fail"
    assert "Found 3 missing artifact(s)" in check.message
    joined = " ".join(check.details)
    assert "actions row" in joined
    assert "solver_runs row" in joined
    assert "regression_cases row" in joined


def test_audited_path_columns_are_exactly_the_retention_list() -> None:
    """The audit and retention must never disagree about what a path column is."""
    from poker_tracker.maintenance.data_health import _ARTIFACT_REFERENCES

    assert {f"{table}.{column}" for table, column in _ARTIFACT_REFERENCES} == {
        label for label, _ in PokerDatabase.ARTIFACT_PATH_COLUMNS
    }


def test_a_path_column_whose_query_disagrees_with_its_label_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derivation only helps if the label and the SQL describe the same rows."""
    from poker_tracker.maintenance.data_health import _artifact_reference_columns

    monkeypatch.setattr(
        PokerDatabase,
        "ARTIFACT_PATH_COLUMNS",
        (("videos.stored_path", "SELECT stored_path FROM extracted_frames"),),
    )
    with pytest.raises(RuntimeError, match="cannot be audited"):
        _artifact_reference_columns()


def test_blank_path_columns_are_not_reported_as_missing_files(tmp_path: Path) -> None:
    """Most path columns default to '', which means "no file", not "a lost file"."""
    from poker_tracker.persistence.models import Hand, Session

    database = tmp_path / "poker_tracker.db"
    db = _current_schema_database(database)
    session = db.create_session(Session(name="Blank"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO solver_runs (hand_id, input_hash, created_at)"
            " VALUES (?, 'hash', '2026-01-01T00:00:00+00:00')",
            (hand.id,),
        )
    db.close()

    report = audit_data_health(
        database, data_dir=tmp_path / "data", restore_backups=False
    )

    check = _checks_by_name(report)["artifact_files"]
    assert check.status == "pass"
    assert check.message.startswith("All 0")


def test_missing_reconstruction_timeline_is_reported(tmp_path: Path) -> None:
    """A timeline has no column anywhere, and a missing one blocks every import."""
    from poker_tracker.persistence.models import ProcessingJob, Session, VideoRecord

    data_dir = tmp_path / "data"
    (data_dir / "cv_timelines").mkdir(parents=True)
    video_file = data_dir / "clip.mp4"
    video_file.write_bytes(b"clip")
    database = tmp_path / "poker_tracker.db"
    db = _current_schema_database(database)
    session = db.create_session(Session(name="Timelines"))
    video = db.create_video(
        VideoRecord(
            original_filename="clip.mp4",
            stored_path=str(video_file),
            file_size_bytes=video_file.stat().st_size,
            session_id=session.id,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id, job_type="cv_reconstruction", status="completed"
        )
    )
    timeline = data_dir / "cv_timelines" / f"job_{job.id}_timeline.json"
    timeline.write_text("{}", encoding="utf-8")
    db.close()

    present = audit_data_health(database, data_dir=data_dir, restore_backups=False)
    assert _checks_by_name(present)["timeline_files"].status == "pass"

    timeline.unlink()
    report = audit_data_health(database, data_dir=data_dir, restore_backups=False)

    check = _checks_by_name(report)["timeline_files"]
    assert check.status == "warning"
    assert f"job_{job.id}_timeline.json" in check.details[0]


def test_minimum_schema_does_not_require_completion_columns() -> None:
    """A retained pre-v13 backup must stay healthy: a missing contract column is a
    hard fail that _connection_issues escalates to a backup failure."""
    from poker_tracker.maintenance.data_health import _MINIMUM_SCHEMA

    assert "completion_status" not in _MINIMUM_SCHEMA["hands"]
    assert "completion_evidence" not in _MINIMUM_SCHEMA["hands"]


def test_schema_contract_requires_completion_columns_once_v13_is_claimed(
    tmp_path: Path,
) -> None:
    """Regression: a database stamped 13 without the Phase 1 columns used to pass
    both schema_version and schema_contract, so the audit gave no signal at all."""
    database = tmp_path / "stamped13.sqlite3"
    _create_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE hands DROP COLUMN completion_status")
        connection.execute("ALTER TABLE hands DROP COLUMN completion_evidence")

    report = audit_data_health(
        database,
        data_dir=tmp_path,
        backup_dir=tmp_path / "backups",
        expected_schema_version=SCHEMA_VERSION,
    )
    contract = _checks_by_name(report)["schema_contract"]

    assert contract.status == "fail"
    assert any("completion_status" in detail for detail in contract.details)


def test_schema_contract_still_passes_for_a_pre_v13_database(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite3"
    _create_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE hands DROP COLUMN completion_status")
        connection.execute("ALTER TABLE hands DROP COLUMN completion_evidence")
        connection.execute(
            "UPDATE schema_metadata SET value = '12' WHERE key = 'schema_version'"
        )

    report = audit_data_health(
        database,
        data_dir=tmp_path,
        backup_dir=tmp_path / "backups",
        expected_schema_version=SCHEMA_VERSION,
    )

    assert _checks_by_name(report)["schema_contract"].status == "pass"


def test_a_lost_index_is_named_by_the_schema_check(tmp_path: Path) -> None:
    """The audit judged a database by a hand-written table-and-column list.

    ``idx_actions_hand_street_order`` is the unique index migration 8 exists to
    create, and it is the only thing stopping two actions claiming one slot in a
    street's order. Dropped, the audit reported "Core PokerTrainer tables and
    columns are present." and HEALTHY while duplicate ordering was written --
    the audit's own subject matter, invisible to the audit.
    """
    database = tmp_path / "poker_tracker.db"
    _current_schema_database(database).close()
    with sqlite3.connect(database) as connection:
        connection.execute("DROP INDEX idx_actions_hand_street_order")

    report = audit_data_health(
        database, data_dir=tmp_path / "data", restore_backups=False
    )

    check = _checks_by_name(report)["schema_contract"]
    assert check.status == "warning"
    assert any("idx_actions_hand_street_order" in detail for detail in check.details)


def test_a_column_added_after_the_hand_written_list_is_still_required(
    tmp_path: Path,
) -> None:
    """The versioned contract stopped at 14 while the schema reached 18.

    Anything added later was outside what the audit knew to ask for, so the list
    protected exactly as much as whoever last remembered to extend it. The
    requirement is derived from the schema this build creates instead, which is
    why this test names a column nobody had to add here.
    """
    database = tmp_path / "poker_tracker.db"
    _current_schema_database(database).close()
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE solver_runs DROP COLUMN run_parameters")

    report = audit_data_health(
        database, data_dir=tmp_path / "data", restore_backups=False
    )

    check = _checks_by_name(report)["schema_contract"]
    assert check.status == "warning"
    assert any("solver_runs.run_parameters" in detail for detail in check.details)


def test_an_older_database_is_not_judged_against_todays_schema(tmp_path: Path) -> None:
    """A retained file from an older build legitimately lacks later additions.

    The derived check is only meaningful at the current version; applied to a
    v13 file it would report every object added since as damage, which is how a
    check earns its way into being ignored.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from legacy_schema_fixtures import write_legacy_database

    database = tmp_path / "poker_tracker.db"
    write_legacy_database(database, stored_version=13)

    report = audit_data_health(
        database, data_dir=tmp_path / "data", restore_backups=False
    )

    assert _checks_by_name(report)["schema_contract"].status == "pass"


def test_the_attestation_check_states_how_many_attestations_it_examined(
    tmp_path: Path,
) -> None:
    """A pass with no denominator is the shape a whole round of findings shared.

    Every other check on this report says what it looked at, and this one used to
    say only that all of them were fine -- a sentence an empty database answers
    exactly as confidently as a full one. The number is what tells an operator
    whether the pass is evidence.
    """
    from poker_tracker.persistence.models import Hand, Session

    database = tmp_path / "poker_tracker.db"
    db = _current_schema_database(database)
    session = db.create_session(Session(name="Attestations"))
    db.close()

    empty = audit_data_health(
        database, data_dir=tmp_path / "data", restore_backups=False
    )
    check = _checks_by_name(empty)["settlement_attestations"]
    assert check.status == "pass"
    assert "All 0 " in check.message

    db = PokerDatabase(database)
    db.init_db()
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            completion_evidence={"confirmed_assumption_codes": ["code-a", "code-b"]},
        )
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO hand_corrections (hand_id, correction_type, notes, "
            "created_at) VALUES (?, 'settlement', ?, '2026-01-01T00:00:00')",
            (hand.id, "attested code-a and code-b"),
        )
    db.close()

    populated = audit_data_health(
        database, data_dir=tmp_path / "data", restore_backups=False
    )
    check = _checks_by_name(populated)["settlement_attestations"]
    assert check.status == "pass"
    assert "All 2 " in check.message
