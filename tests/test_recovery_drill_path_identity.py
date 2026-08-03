"""The recovery drill's isolation guard must not be defeated by a spelling.

The drill refuses to run when its target overlaps the live data root, because a
drill that restored a three-month-old snapshot over the operator's own database
would destroy exactly the history it was run to protect. That refusal was
comparing path TEXT, so ``DATA/drill`` and ``data/drill`` -- one directory on the
case-insensitive filesystem macOS ships -- read as unrelated and the guard
failed open. A guard whose only job is to refuse must never fail in the
permissive direction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poker_tracker.maintenance import data_health
from poker_tracker.maintenance.recovery import run_recovery_drill
from poker_tracker.persistence.db import PokerDatabase


@pytest.fixture
def live(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """A live root the drill must refuse to touch, with a snapshot beside it."""
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    database = data_root / "poker_tracker.db"
    database.write_bytes(b"live study history")
    backup = tmp_path / "offsite" / "snapshot.sqlite3"
    backup.parent.mkdir(parents=True)
    # A real snapshot, so a run the guard allows gets far enough to prove the
    # guard is the only thing that stopped the refused ones.
    source = PokerDatabase(tmp_path / "source.db")
    source.init_db()
    source.close()
    backup.write_bytes((tmp_path / "source.db").read_bytes())
    monkeypatch.setattr(data_health, "DEFAULT_DATA_DIR", data_root)
    monkeypatch.setattr(data_health, "DEFAULT_DATABASE_PATH", database)
    return {"data_root": data_root, "database": database, "backup": backup}


def _filesystem_ignores_case(directory: Path) -> bool:
    probe = directory / "CaseProbe"
    probe.write_bytes(b"probe")
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.unlink()


def test_a_target_inside_the_live_root_is_refused_however_it_is_spelled(live):
    """``DATA/drill`` is inside ``data`` on this filesystem, so the drill refuses.

    The comparison case-folds unconditionally, so a case-sensitive filesystem --
    where the two spellings really could be two directories -- refuses as well.
    That over-matches by design: the cost is a drill an operator has to point
    somewhere else, and the cost of the other error is the study history.
    """
    target = live["data_root"].parent / "DATA" / "drill"

    report = run_recovery_drill(
        backup_path=live["backup"],
        data_root=live["data_root"],
        target_root=target,
    )

    assert report.outcome == "not_performed"
    assert report.exit_code == 2
    refusal = report.checks[0]
    assert refusal.name == "drill_preconditions"
    assert any("overlaps the live data directory" in item for item in refusal.details)


def test_a_target_containing_the_live_database_is_refused_however_it_is_spelled(
    tmp_path: Path, monkeypatch, live
):
    """The live database sitting under the target is caught by the same primitive."""
    live_database = live["data_root"] / "nested" / "poker_tracker.db"
    live_database.parent.mkdir(parents=True)
    live_database.write_bytes(b"live study history")
    monkeypatch.setattr(data_health, "DEFAULT_DATABASE_PATH", live_database)
    # A target that contains the live database but does NOT overlap the data
    # root, so only the containment check can produce this refusal.
    monkeypatch.setattr(data_health, "DEFAULT_DATA_DIR", tmp_path / "elsewhere")
    (tmp_path / "elsewhere").mkdir()

    report = run_recovery_drill(
        backup_path=live["backup"],
        data_root=tmp_path / "elsewhere",
        target_root=live["data_root"].parent / "DATA" / "NESTED",
    )

    assert report.outcome == "not_performed"
    assert any(
        "contains the live database" in item for item in report.checks[0].details
    )


def test_the_live_database_is_not_accepted_as_a_backup_under_another_spelling(live):
    """A snapshot argument that is really the live file is refused by identity.

    ``(st_dev, st_ino)`` settles this outright where the file exists, which is
    what a string comparison of two spellings could not.
    """
    if not _filesystem_ignores_case(live["data_root"]):
        pytest.skip("case-sensitive filesystem: the two spellings are two files")

    report = run_recovery_drill(
        backup_path=live["data_root"] / "POKER_TRACKER.DB",
        data_root=live["data_root"],
        target_root=live["data_root"].parent / "drill",
    )

    assert report.outcome == "not_performed"
    assert any(
        "not a backup of itself" in item for item in report.checks[0].details
    )


def test_a_target_genuinely_outside_the_live_root_is_still_allowed(live):
    """The guard refuses overlap, not every drill: an unrelated target proceeds.

    Without this the fix could be "refuse everything", which passes every other
    test here and makes the drill unusable.
    """
    report = run_recovery_drill(
        backup_path=live["backup"],
        data_root=live["data_root"],
        target_root=live["data_root"].parent / "drill",
    )

    assert report.outcome != "not_performed"
    assert [check.name for check in report.checks][0] != "drill_preconditions"
