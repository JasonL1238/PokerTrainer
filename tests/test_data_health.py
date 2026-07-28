from __future__ import annotations

import json
import os
import shutil
import sqlite3
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
                file_size_bytes INTEGER NOT NULL
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
    assert "videos: missing column(s) file_size_bytes" in schema_check.details


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
