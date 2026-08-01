"""The Phase 11 exit gate, executed: seed, back up, destroy, recover, compare.

Every test here restores a real snapshot into a throwaway root and asks whether
the study history came back. The drill's value is entirely in the failing cases:
a backup holding nothing, a data directory whose recordings are gone, a snapshot
with no inventory. Those are the outcomes that used to be indistinguishable from
a good recovery.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from poker_tracker.maintenance import data_health
from poker_tracker.maintenance.recovery import (
    RecoveryDrillReport,
    run_recovery_drill,
)
from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence.backup import backup_database
from poker_tracker.persistence.backup_inventory import inventory_path
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    ExtractedFrame,
    Hand,
    HandIssue,
    HandPlayer,
    HandSettlement,
    ProcessingJob,
    Session,
    SettlementEntry,
    SolverRun,
    VideoRecord,
)
from poker_tracker.services.hand_accounting import persist_reconciliation
from poker_tracker.ui.video_storage import ensure_data_directories
from tests.conftest import attest_declared_assumptions


@pytest.fixture
def machine(tmp_path: Path) -> dict[str, Path]:
    """A source machine: a database, a persistent data root, and a backup directory."""
    root = tmp_path / "source"
    data_root = root / "data"
    ensure_data_directories(data_root)
    return {
        "root": root,
        "database": root / "poker_tracker.db",
        "data_root": data_root,
        "backups": tmp_path / "offsite_backups",
        "target": tmp_path / "drill",
    }


def _seed_study_history(machine: dict[str, Path]) -> dict[str, Any]:
    """A session, a completed hand with settlement, a flagged hand, and real artifacts.

    The issue lives on a SECOND hand on purpose: filing one demotes its hand's
    completion status to ``uncertain`` (``_flag_hand_for_debugging``), so a
    fixture that flagged the completed hand would leave the history with no
    completed hand to read back and would be testing the wrong thing.
    """
    data_root = machine["data_root"]
    video_file = data_root / "videos" / "session_one.mp4"
    video_file.write_bytes(b"not really a video, but a real file")
    frame_file = data_root / "frames" / "frame_0001.png"
    frame_file.write_bytes(b"frame")
    solver_file = data_root / "solver" / "run_1.json"
    solver_file.parent.mkdir(parents=True, exist_ok=True)
    solver_file.write_text("{}", encoding="utf-8")

    db = PokerDatabase(machine["database"])
    db.init_db()
    session = db.create_session(Session(name="Tuesday night", date_played="2026-07-01"))
    video = db.create_video(
        VideoRecord(
            session_id=session.id,
            original_filename="session_one.mp4",
            stored_path=str(video_file),
            file_size_bytes=video_file.stat().st_size,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(
            job_type="cv_reconstruction",
            status="completed",
            video_id=int(video.id or 0),
        )
    )
    db.create_extracted_frame(
        ExtractedFrame(
            video_id=int(video.id or 0),
            job_id=int(job.id or 0),
            timestamp_seconds=1.0,
            frame_index=1,
            image_path=str(frame_file),
        )
    )
    timeline_file = data_root / "cv_timelines" / f"job_{job.id}_timeline.json"
    timeline_file.write_text(json.dumps({"states": [], "hands": []}), encoding="utf-8")

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
            review_status="unreviewed",
            source_type="manual",
            completion_status="complete",
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
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="settled", rake_rate=0.0)
    )
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
    persist_reconciliation(db, int(hand.id or 0))
    attest_declared_assumptions(db, int(hand.id or 0), only="declared_pot_awards")
    db.create_solver_run(
        SolverRun(
            hand_id=int(hand.id or 0),
            status="completed",
            input_hash="deadbeef",
            result_path=str(solver_file),
        )
    )
    flagged = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=2,
            game_type="No-limit Hold'em",
            table_size=6,
            hero_position="CO",
            review_status="unreviewed",
            source_type="manual",
        )
    )
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=int(flagged.id or 0),
            issue_types=["pot_or_result"],
            description="Pot award looked wrong on the river.",
        )
    )
    expected = {
        "hand_id": int(hand.id or 0),
        "flagged_hand_id": int(flagged.id or 0),
        "session_name": session.name,
        "issue_id": int(issue.id or 0),
        "job_id": int(job.id or 0),
        "video_file": video_file,
        "frame_file": frame_file,
        "solver_file": solver_file,
        "timeline_file": timeline_file,
        "counts": {
            "sessions": 1,
            "hands": 2,
            "completed_hands": 1,
            "hand_issues": 1,
        },
        "players": [player.player_key for player in db.fetch_players_by_hand(hand.id)],
        "actions": [
            action.action_type for action in db.fetch_actions_by_hand(hand.id)
        ],
        "settlement_entries": len(db.fetch_settlement_entries(hand.id)),
    }
    db.close()
    return expected


def _take_backup(machine: dict[str, Path], expected: dict[str, Any]) -> Path:
    """Snapshot the database; ``backup_database`` writes the inventory beside it."""
    return backup_database(machine["database"], machine["backups"])


def _inventory_of(snapshot: Path) -> dict[str, Any]:
    payload = json.loads(
        inventory_path(snapshot).read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return payload


def _record_counts_in_inventory(snapshot: Path, counts: dict[str, int]) -> None:
    """Add the count block the drill compares against.

    ``backup_inventory`` records artifacts but no row counts, so the drill's
    count comparison has nothing to read on a real snapshot today and says so.
    Writing the block here pins the shape the drill consumes, so that a writer
    adding counts later has something to conform to.
    """
    path = inventory_path(snapshot)
    payload = _inventory_of(snapshot)
    payload["counts"] = counts
    path.write_text(json.dumps(payload), encoding="utf-8")


def _destroy(machine: dict[str, Path]) -> None:
    """Lose the live database, as a dead disk would."""
    for suffix in ("", "-wal", "-shm"):
        Path(f"{machine['database']}{suffix}").unlink(missing_ok=True)


def _run(machine: dict[str, Path], snapshot: Path, **kwargs: Any) -> RecoveryDrillReport:
    return run_recovery_drill(
        backup_path=snapshot,
        data_root=machine["data_root"],
        target_root=machine["target"],
        **kwargs,
    )


def _check(report: RecoveryDrillReport, name: str) -> Any:
    matches = [check for check in report.checks if check.name == name]
    assert matches, f"no check named {name} in {[c.name for c in report.checks]}"
    return matches[0]


def test_seed_back_up_destroy_recover_reproduces_the_whole_study_history(
    machine: dict[str, Path],
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    _record_counts_in_inventory(snapshot, expected["counts"])
    _destroy(machine)

    report = _run(machine, snapshot)

    assert report.outcome == "recovered", [
        (c.name, c.status, c.message, c.details) for c in report.checks
    ]
    assert report.exit_code == 0
    assert report.counts == expected["counts"]
    assert report.missing_artifacts == ()
    for name in (
        "restored_open",
        "study_history_counts",
        "backup_inventory",
        "issue_evidence",
        "completed_hand_readback",
        "recovered_artifacts",
        "foreign_key_check",
        "schema_version",
    ):
        assert _check(report, name).status == "pass"
    assert "matches the inventory" in _check(report, "study_history_counts").message

    # The destroyed machine stays destroyed: recovery happens in the drill root.
    assert not machine["database"].exists()
    restored = machine["target"] / "poker_tracker.db"
    assert restored.is_file()

    recovered = PokerDatabase(restored)
    recovered.init_db()
    try:
        assert recovered.schema_version() == SCHEMA_VERSION
        sessions = recovered.fetch_sessions()
        assert [session.name for session in sessions] == [expected["session_name"]]
        hands = recovered.fetch_all_hands()
        assert sorted(int(hand.id or 0) for hand in hands) == [
            expected["hand_id"],
            expected["flagged_hand_id"],
        ]
        hand_id = expected["hand_id"]
        assert [
            player.player_key for player in recovered.fetch_players_by_hand(hand_id)
        ] == expected["players"]
        assert [
            action.action_type for action in recovered.fetch_actions_by_hand(hand_id)
        ] == expected["actions"]
        settlement = recovered.fetch_hand_settlement(hand_id)
        assert settlement is not None and settlement.status == "reconciled"
        assert len(recovered.fetch_settlement_entries(hand_id)) == (
            expected["settlement_entries"]
        )
        issues = recovered.fetch_hand_issues()
        assert len(issues) == 1
        evidence = issues[0].evidence_snapshot["hand"]
        assert isinstance(evidence, dict)
        assert evidence["id"] == expected["flagged_hand_id"]
    finally:
        recovered.close()


def test_backup_with_no_study_history_is_not_a_recovery(
    machine: dict[str, Path],
) -> None:
    """The check that a clean-opening empty snapshot used to pass."""
    db = PokerDatabase(machine["database"])
    db.init_db()
    db.close()
    snapshot = backup_database(machine["database"], machine["backups"])
    _record_counts_in_inventory(snapshot, {"sessions": 0, "hands": 0})
    _destroy(machine)

    report = _run(machine, snapshot)

    assert report.outcome == "partial"
    assert report.exit_code == 1
    counts = _check(report, "study_history_counts")
    assert counts.status == "fail"
    assert "no study history" in counts.message
    assert _check(report, "completed_hand_readback").status == "fail"


def test_missing_recordings_make_the_recovery_partial(
    machine: dict[str, Path],
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    _destroy(machine)
    expected["video_file"].unlink()
    expected["solver_file"].unlink()

    report = _run(machine, snapshot)

    assert report.outcome == "partial"
    assert report.exit_code == 1
    artifacts = _check(report, "recovered_artifacts")
    assert artifacts.status == "fail"
    assert "PARTIAL RECOVERY" in artifacts.message
    assert set(report.missing_artifacts) == {
        str(expected["video_file"]),
        str(expected["solver_file"]),
    }


def test_missing_timeline_is_reported_even_though_no_column_names_it(
    machine: dict[str, Path],
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    _destroy(machine)
    expected["timeline_file"].unlink()

    report = _run(machine, snapshot)

    assert report.outcome == "partial"
    assert str(expected["timeline_file"]) in report.missing_artifacts
    assert any(
        str(expected["timeline_file"]) in detail
        for detail in _check(report, "recovered_artifacts").details
    )


def test_snapshot_without_an_inventory_cannot_be_called_complete(
    machine: dict[str, Path],
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    inventory_path(snapshot).unlink()
    _destroy(machine)

    report = _run(machine, snapshot)

    assert report.outcome == "unverified"
    assert report.exit_code == 1
    inventory = _check(report, "backup_inventory")
    assert inventory.status == "fail"
    assert "cannot be shown to be complete" in inventory.message
    # Everything else still restored; the verdict is about proof, not damage.
    assert _check(report, "study_history_counts").status == "pass"
    assert _check(report, "completed_hand_readback").status == "pass"


def test_inventory_disagreeing_with_the_restored_counts_fails(
    machine: dict[str, Path],
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    _record_counts_in_inventory(snapshot, {**expected["counts"], "hands": 9})
    _destroy(machine)

    report = _run(machine, snapshot)

    # Damage, not doubt: the inventory itself is fine, the history is short.
    assert report.outcome == "partial"
    assert _check(report, "backup_inventory").status == "pass"
    counts = _check(report, "study_history_counts")
    assert counts.status == "fail"
    assert any("inventory=9" in detail for detail in counts.details)


def test_a_product_written_inventory_reports_that_it_carries_no_counts(
    machine: dict[str, Path],
) -> None:
    """The one gap the drill cannot close on its own, stated in every run.

    ``backup_inventory`` records artifacts and no row counts, so a real snapshot
    verifies its artifact set and self-reports its totals. That must be visible
    rather than implied, or an operator reads a green drill as proof that the
    same number of hands came back.
    """
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    assert "counts" not in _inventory_of(snapshot)
    _destroy(machine)

    report = _run(machine, snapshot)

    assert report.outcome == "recovered"
    inventory = _check(report, "backup_inventory")
    assert inventory.status == "warning"
    assert "no session or hand counts" in inventory.message
    assert "matches the inventory" not in (
        _check(report, "study_history_counts").message
    )


def test_a_recording_replaced_since_the_snapshot_is_not_a_clean_recovery(
    machine: dict[str, Path],
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    _destroy(machine)
    expected["video_file"].write_bytes(b"a different recording entirely, truncated")

    report = _run(machine, snapshot)

    assert report.outcome == "partial"
    artifacts = _check(report, "recovered_artifacts")
    assert artifacts.status == "fail"
    assert report.missing_artifacts == ()
    assert any(
        "replaced since the snapshot" in detail and str(expected["video_file"]) in detail
        for detail in artifacts.details
    )


def test_unreadable_inventory_is_louder_than_an_absent_one(
    machine: dict[str, Path],
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    inventory_path(snapshot).write_text("{not json", encoding="utf-8")
    _destroy(machine)

    report = _run(machine, snapshot)

    inventory = _check(report, "backup_inventory")
    assert inventory.status == "fail"
    assert "could not be read" in inventory.message


def test_blanked_issue_evidence_is_reported(machine: dict[str, Path]) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    _destroy(machine)
    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE hand_issues SET evidence_snapshot = '{}'")

    report = _run(machine, snapshot)

    assert report.outcome == "partial"
    evidence = _check(report, "issue_evidence")
    assert evidence.status == "fail"
    assert "lost their evidence" in evidence.message


def test_a_hand_the_application_cannot_read_fails_the_drill(
    machine: dict[str, Path],
) -> None:
    """A row SQLite is happy with is not a hand the product can open."""
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    _destroy(machine)
    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE hand_players SET starting_stack = -5")

    report = _run(machine, snapshot)

    assert report.outcome == "partial"
    readback = _check(report, "completed_hand_readback")
    assert readback.status == "fail"
    assert "could not be read back" in readback.message
    # The point of the check: nothing structural noticed.
    assert _check(report, "sqlite_quick_check").status == "pass"
    assert _check(report, "foreign_key_check").status == "pass"
    assert _check(report, "study_history_counts").status == "pass"


def test_drill_refuses_a_target_inside_the_live_data_directory(
    machine: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    monkeypatch.setattr(data_health, "DEFAULT_DATA_DIR", machine["data_root"])
    monkeypatch.setattr(data_health, "DEFAULT_DATABASE_PATH", machine["database"])
    target = machine["data_root"] / "drill"

    report = run_recovery_drill(
        backup_path=snapshot,
        data_root=machine["data_root"],
        target_root=target,
    )

    assert report.outcome == "not_performed"
    assert report.exit_code == 2
    assert not target.exists()
    assert machine["database"].is_file()
    reasons = _check(report, "drill_preconditions").details
    assert any("overlaps the live data directory" in reason for reason in reasons)


def test_drill_refuses_a_target_holding_the_live_database(
    machine: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    monkeypatch.setattr(data_health, "DEFAULT_DATABASE_PATH", machine["database"])
    monkeypatch.setattr(data_health, "DEFAULT_DATA_DIR", machine["root"] / "elsewhere")

    report = run_recovery_drill(
        backup_path=snapshot,
        data_root=machine["data_root"],
        target_root=machine["root"],
    )

    assert report.outcome == "not_performed"
    assert report.exit_code == 2
    assert machine["database"].is_file()
    reasons = _check(report, "drill_preconditions").details
    assert any("contains the live database" in reason for reason in reasons)


def test_drill_refuses_to_restore_over_an_existing_database(
    machine: dict[str, Path],
) -> None:
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    machine["target"].mkdir(parents=True)
    occupied = machine["target"] / "poker_tracker.db"
    occupied.write_bytes(b"someone else's file")

    report = _run(machine, snapshot)

    assert report.outcome == "not_performed"
    assert report.exit_code == 2
    assert occupied.read_bytes() == b"someone else's file"


def test_drill_refuses_the_live_database_as_its_own_backup(
    machine: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_study_history(machine)
    monkeypatch.setattr(data_health, "DEFAULT_DATABASE_PATH", machine["database"])

    report = _run(machine, machine["database"])

    assert report.outcome == "not_performed"
    reasons = _check(report, "drill_preconditions").details
    assert any("not a backup of itself" in reason for reason in reasons)


def test_migrating_the_restored_copy_never_writes_to_the_operators_backups(
    machine: dict[str, Path],
) -> None:
    """The drill's own pre-migration snapshot must not evict a real restore point."""
    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    _destroy(machine)
    # An older snapshot: opening it applies the remaining migrations, which is
    # what makes init_db take a pinned pre-migration copy.
    with sqlite3.connect(snapshot) as connection:
        connection.execute(
            "UPDATE schema_metadata SET value = ? WHERE key = 'schema_version'",
            (str(SCHEMA_VERSION - 1),),
        )
    before = sorted(path.name for path in machine["backups"].iterdir())
    operator_backups = backup_module.BACKUPS_DIR

    report = _run(machine, snapshot)

    assert report.outcome == "recovered", [
        (c.name, c.status, c.message, c.details) for c in report.checks
    ]
    assert sorted(path.name for path in machine["backups"].iterdir()) == before
    assert not list(Path(operator_backups).glob("*.sqlite3"))
    assert backup_module.BACKUPS_DIR == operator_backups
    assert list((machine["target"] / "backups").glob("*.sqlite3"))


def test_cli_reports_the_outcome_and_returns_the_contract_exit_code(
    machine: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from poker_tracker.maintenance.recovery import main

    expected = _seed_study_history(machine)
    snapshot = _take_backup(machine, expected)
    _destroy(machine)

    code = main(
        [
            "--backup",
            str(snapshot),
            "--data-dir",
            str(machine["data_root"]),
            "--target",
            str(machine["target"]),
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "recovered"
    assert payload["exit_code"] == 0
    assert payload["counts"] == expected["counts"]
    assert payload["missing_artifacts"] == []
