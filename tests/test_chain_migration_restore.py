"""Migration, application reads, backup and isolated restore, joined as one chain.

Each of the four steps already had a test. ``test_migration_matrix`` proves every
supported version migrates without losing a row; ``test_persistence_integrity``
proves the pre-migration snapshot is written before the chain runs;
``test_recovery_drill`` proves a snapshot restores into a throwaway root and is
verified there. What none of them does is run one operator's study history
through all four in order, so the seams between them were never observed: a
migration that produced rows the application cannot compose, a backup taken of
the migrated file that carried a different history than the one just read, a
restore that opened cleanly and held something else.

The mechanisms here are the existing ones. The legacy databases come from
``tests.legacy_schema_fixtures`` -- genuinely old physical schemas, not a
current-shape file with a rewritten stamp -- and the restore is
``poker_tracker.maintenance.recovery.run_recovery_drill``, the drill an operator
actually runs. Nothing here builds a third migration harness or a second restore.

What is new is the comparison. ``_study_history`` composes the whole history
through the readers the Study surfaces use, and the chain asserts that the same
structure comes out of the restored copy as went into the backup, value for
value. A chain that only asserted "no exception was raised" would pass while a
seam silently dropped every settlement entry, so the last test damages a snapshot
on purpose and requires the comparison to say what changed.

The pre-migration snapshot is followed separately, because it is the one artifact
that answers a different question: not "did the history survive the migration"
but "can the operator get back the state the irreversible migration overwrote".
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from poker_tracker.maintenance.recovery import RecoveryDrillReport, run_recovery_drill
from poker_tracker.math.accounting import LedgerError
from poker_tracker.persistence.backup import backup_database, find_snapshots
from poker_tracker.persistence.backup_inventory import inventory_path
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.models import HandSettlement, SettlementEntry
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import evaluate_study_readiness
from poker_tracker.ui.video_storage import ensure_data_directories
from tests.conftest import attest_declared_assumptions
from tests.legacy_schema_fixtures import write_legacy_database

# The schema the seeded fixture was written at for the value assertions below.
# Chosen because 13 is the one irreversible migration in the chain -- it rewrites
# review_status on every hand that is not provably manual -- so the pre-migration
# snapshot has something to be the only copy of.
LEGACY_VERSION = 12

# Every file the legacy fixture's rows point at, relative to the data root, plus
# the timeline job 1 implies by convention. Written as real files because the
# drill reports a missing artifact as a partial recovery, and a chain that ran
# with none of them present would be asserting against that failure instead of
# against the history.
FIXTURE_ARTIFACTS: tuple[str, ...] = (
    "videos/session.mp4",
    "frames/job_1/frame_000375.png",
    "solver/run_1/result.json",
    "solver/run_1/run.log",
    "solver/run_1/command.txt",
    "cv_timelines/job_1_timeline.json",
    # Referenced only by the schema-17 regression_cases rows, and written for
    # every version anyway: an artifact that arrives with a later schema must not
    # make the chain's earlier cases depend on which files happen to exist.
    "tests/fixtures/boundary.json",
    "reports/boundary.md",
)


@pytest.fixture
def machine(tmp_path: Path) -> dict[str, Path]:
    """One operator's machine: a database, its data root, and somewhere to restore."""
    root = tmp_path / "machine"
    data_root = root / "data"
    ensure_data_directories(data_root)
    for relative in FIXTURE_ARTIFACTS:
        artifact = data_root / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"recorded evidence")
    return {
        "root": root,
        "database": root / "poker_tracker.db",
        "data_root": data_root,
        # The migration writes its pinned snapshot beside the database it is
        # migrating, which is what `backups_dir_for` does for any database that
        # is not the live one.
        "premigration_backups": root / "backups",
        "backups": tmp_path / "offsite_backups",
        "target": tmp_path / "drill",
    }


def _seed_legacy(machine: dict[str, Path], stored_version: int | None) -> None:
    write_legacy_database(machine["database"], stored_version=stored_version)


def _open_migrated(machine: dict[str, Path]) -> PokerDatabase:
    db = PokerDatabase(machine["database"])
    db.init_db()
    return db


def _destroy(machine: dict[str, Path]) -> None:
    """Lose the live database and its sidecars, as a failed disk would."""
    for suffix in ("", "-wal", "-shm"):
        Path(f"{machine['database']}{suffix}").unlink(missing_ok=True)


def _run_drill(
    machine: dict[str, Path], snapshot: Path, *, target: Path | None = None
) -> RecoveryDrillReport:
    return run_recovery_drill(
        backup_path=snapshot,
        data_root=machine["data_root"],
        target_root=machine["target"] if target is None else target,
    )


def _check(report: RecoveryDrillReport, name: str) -> Any:
    matches = [check for check in report.checks if check.name == name]
    assert matches, f"no check named {name} in {[c.name for c in report.checks]}"
    return matches[0]


def _describe(report: RecoveryDrillReport) -> list[str]:
    """Only what went wrong, so a failure message names the check that disagreed."""
    return [
        f"[{check.status}] {check.name}: {check.message} {check.details}"
        for check in report.checks
        if check.status != "pass"
    ]


# --------------------------------------------------------------------------
# The study history, composed the way the application composes it
# --------------------------------------------------------------------------


def _study_history(db: PokerDatabase) -> dict[str, Any]:
    """The whole history as the Study surfaces read it, as comparable plain data.

    Read through the model layer rather than with SELECT, because the claim under
    test is that the APPLICATION can read what the migration produced. A raw
    query would pass on rows Pydantic refuses, a settlement the reconciliation
    cannot cross-check, or evidence the readiness rules cannot parse -- exactly
    the damage a migration introduces and a row count cannot see.
    """
    return {
        "sessions": [
            {
                "name": session.name,
                "date_played": session.date_played.isoformat(),
                "platform": session.platform,
                "stakes": session.stakes,
                "hands": [
                    _hand_facts(db, hand) for hand in db.fetch_hands_by_session(session.id)
                ],
                "coaching_reviews": [
                    review.review_type
                    for review in db.fetch_coaching_reviews_by_session(session.id)
                ],
            }
            for session in db.fetch_sessions()
        ],
        "videos": [
            {
                "original_filename": video.original_filename,
                "stored_path": video.stored_path,
                "file_size_bytes": video.file_size_bytes,
                "duration_seconds": video.duration_seconds,
                "frame_count": video.frame_count,
            }
            for video in db.fetch_videos()
        ],
        "issues": [
            {
                "hand_id": issue.hand_id,
                "status": issue.status,
                "issue_types": list(issue.issue_types),
                "description": issue.description,
                "evidence_snapshot": issue.evidence_snapshot,
            }
            for issue in db.fetch_hand_issues()
        ],
    }


def _hand_facts(db: PokerDatabase, hand: Any) -> dict[str, Any]:
    hand_id = int(hand.id or 0)
    accounting, accounting_error = _reconcile(db, hand_id)
    settlement = db.fetch_hand_settlement(hand_id)
    readiness = evaluate_study_readiness(
        hand,
        accounting=accounting,
        accounting_error=accounting_error,
        hand_issues=db.fetch_hand_issues(hand_id=hand_id),
        coaching_reviews=db.fetch_coaching_reviews_by_hand(hand_id),
        hand_reviews=db.fetch_reviews_by_hand(hand_id),
        solver_runs=db.fetch_solver_runs_by_hand(hand_id),
    )
    return {
        "hand_number": hand.hand_number,
        "hero_position": hand.hero_position,
        "hero_cards": hand.hero_cards,
        "board_cards": hand.board_cards,
        "pot_size": hand.pot_size,
        "hero_bb_won": hand.hero_bb_won,
        "review_status": hand.review_status,
        "completion_status": hand.completion_status,
        "source_type": hand.source_type,
        "study_inclusion": hand.study_inclusion,
        "tags": list(hand.tags),
        "notes": hand.notes,
        "players": [
            {
                "player_key": player.player_key,
                "seat_index": player.seat_index,
                "player_name": player.player_name,
                "position": player.position,
                "starting_stack": player.starting_stack,
                "is_hero": player.is_hero,
            }
            for player in db.fetch_players_by_hand(hand_id)
        ],
        "actions": [
            {
                "street": action.street,
                "action_index": action.action_index,
                "player_key": action.player_key,
                "player_name": action.player_name,
                "action_type": action.action_type,
                "amount": action.amount,
                "amount_semantics": action.amount_semantics,
                "source_image": action.source_image,
            }
            for action in db.fetch_actions_by_hand(hand_id)
        ],
        "settlement": None
        if settlement is None
        else {
            "status": settlement.status,
            "dead_money": settlement.dead_money,
            "rake_rate": settlement.rake_rate,
            "gross_pot": settlement.gross_pot,
            "rake_amount": settlement.rake_amount,
            "net_pot": settlement.net_pot,
            "is_balanced": settlement.is_balanced,
        },
        "settlement_entries": [
            {
                "entry_type": entry.entry_type,
                "pot_index": entry.pot_index,
                "player_key": entry.player_key,
                "player_name": entry.player_name,
                "amount": entry.amount,
            }
            for entry in db.fetch_settlement_entries(hand_id)
        ],
        "corrections": [
            {
                "correction_type": correction.correction_type,
                "before_state": correction.before_state,
                "after_state": correction.after_state,
                "notes": correction.notes,
            }
            for correction in db.fetch_hand_corrections(hand_id)
        ],
        "hand_reviews": [
            review.study_lesson for review in db.fetch_reviews_by_hand(hand_id)
        ],
        "coaching_reviews": [
            review.raw_response for review in db.fetch_coaching_reviews_by_hand(hand_id)
        ],
        "solver_runs": [
            {
                "status": run.status,
                "input_hash": run.input_hash,
                "result_path": run.result_path,
                "exploitability_pct": run.exploitability_pct,
            }
            for run in db.fetch_solver_runs_by_hand(hand_id)
        ],
        "accounting_authoritative": None
        if accounting is None
        else accounting.is_authoritative,
        "accounting_issues": [] if accounting is None else list(accounting.issues),
        "accounting_error": accounting_error,
        "study_ready": readiness.is_ready,
        "blockers": [blocker.code for blocker in readiness.blockers],
    }


def _reconcile(db: PokerDatabase, hand_id: int) -> tuple[Any, str | None]:
    """Reconcile the way ``analytics.resolve_hand_evidence`` does, refusal included.

    An incomplete legacy draft -- a hand with no seats recorded -- makes the
    ledger refuse to build, and the product treats that as a hand it can still
    show with a blocker rather than as a hand it cannot read. Composing it any
    other way here would make the chain assert something the application does
    not do.
    """
    try:
        return reconcile_persisted_hand(db, hand_id), None
    except LedgerError as exc:
        return None, str(exc)


def _differences(before: Any, after: Any, path: str = "") -> list[str]:
    """Where two composed histories disagree, named by the field that differs.

    An equality assertion on two nested structures reports the whole structure
    twice and leaves the reader to find the seam. This walks them together so a
    failure names the hand and the column.
    """
    if isinstance(before, dict) and isinstance(after, dict):
        found: list[str] = []
        for key in sorted(set(before) | set(after)):
            if key not in before:
                found.append(f"{path}.{key}: absent before, {after[key]!r} after")
            elif key not in after:
                found.append(f"{path}.{key}: {before[key]!r} before, absent after")
            else:
                found.extend(_differences(before[key], after[key], f"{path}.{key}"))
        return found
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [f"{path}: {len(before)} item(s) before, {len(after)} after"]
        found = []
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            found.extend(_differences(left, right, f"{path}[{index}]"))
        return found
    if before != after:
        return [f"{path}: {before!r} before, {after!r} after"]
    return []


def _raw_facts(database: Path) -> dict[str, Any]:
    """Read a database without this build's model layer.

    The pre-migration snapshot is at a schema this build refuses to open without
    migrating it, and migrating it is what destroys the state it exists to hold.
    So the rollback point is read as bytes, which is also how an operator
    inspects one before deciding to put it back.
    """
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        stamp = connection.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        return {
            "schema_version": None if stamp is None else int(stamp[0]),
            "hands": {
                int(row["id"]): {
                    "review_status": row["review_status"],
                    "source_type": row["source_type"],
                    "hero_cards": row["hero_cards"],
                    "pot_size": row["pot_size"],
                    "notes": row["notes"],
                }
                for row in connection.execute(
                    "SELECT id, review_status, source_type, hero_cards, pot_size, notes "
                    "FROM hands ORDER BY id"
                )
            },
            "settlement_entries": [
                (row["entry_type"], row["player_name"], row["amount"])
                for row in connection.execute(
                    "SELECT entry_type, player_name, amount FROM settlement_entries "
                    "ORDER BY id"
                )
            ],
            "tables": {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            },
        }
    finally:
        connection.close()


# --------------------------------------------------------------------------
# 1. Migration -> application reads
# --------------------------------------------------------------------------


def test_the_application_reads_the_operators_facts_out_of_the_migrated_database(
    machine: dict[str, Path],
) -> None:
    """The first seam, asserted by value rather than by row count.

    Every fact below was written into a genuine schema-12 file. A migration that
    renumbered, defaulted or blanked any of them would keep the row counts the
    matrix already checks and would change these.
    """
    _seed_legacy(machine, LEGACY_VERSION)

    db = _open_migrated(machine)
    try:
        assert db.schema_version() == SCHEMA_VERSION
        history = _study_history(db)
    finally:
        db.close()

    assert [session["name"] for session in history["sessions"]] == ["Legacy session"]
    hands = history["sessions"][0]["hands"]
    assert [hand["hand_number"] for hand in hands] == [1, 2, 3, 4]

    reconstructed = hands[1]
    assert reconstructed["hero_cards"] == "Kh Kd"
    assert reconstructed["board_cards"] == "2s 5h 9c"
    assert reconstructed["pot_size"] == 30.0
    assert reconstructed["hero_bb_won"] == -12.0
    assert reconstructed["source_type"] == "cv_import"
    assert reconstructed["notes"] == "reconstructed hand"
    assert [player["player_name"] for player in reconstructed["players"]] == [
        "Hero",
        "Villain",
    ]
    assert [action["action_type"] for action in reconstructed["actions"]] == [
        "bet",
        "call",
    ]
    assert reconstructed["settlement"]["gross_pot"] == 30.0
    assert reconstructed["settlement"]["rake_amount"] == 1.5
    assert reconstructed["settlement_entries"][0] == {
        "entry_type": "award",
        "pot_index": 0,
        "player_key": "villain",
        "player_name": "Villain",
        "amount": 28.5,
    }
    assert reconstructed["corrections"] == [
        {
            "correction_type": "hand_facts",
            "before_state": {"pot_size": "28.0"},
            "after_state": {"pot_size": "30.0"},
            "notes": "Frame shows the river bet.",
        }
    ]
    # The retained ANALYSIS TEXT survives the chain intact. Its FRESHNESS does
    # not, and deliberately: this hand's settlement declares dead_money, so
    # _migrate_to_v20 re-derives its pot layers under ruling 5's cap and marks
    # everything written against the old layering as no longer current. The
    # lesson and the response are still here, word for word -- what changed is
    # that the product no longer presents them as describing today's figures.
    assert reconstructed["hand_reviews"] == ["lesson"]
    assert reconstructed["coaching_reviews"] == ["response"]
    assert reconstructed["solver_runs"] == [
        {
            # 'completed' before the chain ran; the solver input was built from
            # the ledger this migration re-derived. See _migrate_to_v20.
            "status": "stale",
            "input_hash": "legacy-hash",
            "result_path": "solver/run_1/result.json",
            "exploitability_pct": 0.4,
        }
    ]
    assert history["videos"] == [
        {
            "original_filename": "session.mp4",
            "stored_path": "videos/session.mp4",
            "file_size_bytes": 1024,
            "duration_seconds": 600.0,
            "frame_count": 18000,
        }
    ]
    assert history["issues"] == [
        {
            "hand_id": 3,
            "status": "open",
            "issue_types": ["hand_boundary"],
            "description": "Boundary is unclear.",
            "evidence_snapshot": {"hand_number": 3},
        }
    ]
    # The one column migration 13 is documented to rewrite, observed doing it:
    # the operator's own 'reviewed' on a reconstructed hand is not preserved.
    assert reconstructed["review_status"] == "needs_correction"
    assert hands[0]["review_status"] == "reviewed"

    # The legacy file also holds a settlement row this build's model refuses --
    # an entry_type the current schema has no name for. It reads back marked
    # unreadable rather than as a plausible refund of nothing, and the marker
    # reaches the surface that decides whether the hand may be studied. A
    # migration that let a row like this through as study-ready would be the
    # failure this whole chain exists to catch.
    assert reconstructed["settlement_entries"][1]["player_name"] == "(unreadable)"
    assert reconstructed["settlement_entries"][1]["amount"] is None
    assert reconstructed["accounting_authoritative"] is None
    assert "unreadable" in (reconstructed["accounting_error"] or "")
    assert reconstructed["study_ready"] is False
    assert "ACCOUNTING_NOT_AUTHORITATIVE" in reconstructed["blockers"]


# --------------------------------------------------------------------------
# 2. Migration -> application reads -> backup -> isolated restore
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stored_version", [None, 7, 12])
def test_the_history_the_application_read_is_the_history_that_comes_back(
    machine: dict[str, Path], stored_version: int | None
) -> None:
    """The whole chain, with the same comparison at the far end as at the near one.

    The restore is the drill an operator runs, not a file copy, so the assertion
    covers the drill's own verdict as well as the data: a drill that called this
    recovery incomplete would be as much of a failure as a restore that came back
    short.

    The parameters stop below schema 13. From 13 on the fixture's ``hands`` row 2
    keeps ``completion_status = 'complete'`` while its ``settlement_entries``
    still hold a legacy row this build's model refuses, so the drill reports a
    completed hand it cannot reconcile -- a true statement about the fixture, on
    both sides of the backup, and nothing to do with what survived the restore.
    The test below carries a hand that DOES reconcile through the same chain.
    """
    _seed_legacy(machine, stored_version)
    db = _open_migrated(machine)
    try:
        migrated = _study_history(db)
    finally:
        db.close()

    snapshot = backup_database(
        machine["database"], machine["backups"], data_dir=machine["data_root"]
    )
    _destroy(machine)

    report = _run_drill(machine, snapshot)

    assert report.outcome == "recovered", _describe(report)
    assert report.exit_code == 0
    assert report.missing_artifacts == ()
    assert not machine["database"].exists()

    restored = PokerDatabase(machine["target"] / "poker_tracker.db")
    restored.init_db()
    try:
        recovered = _study_history(restored)
    finally:
        restored.close()

    assert _differences(migrated, recovered) == []
    # Not vacuous: two empty histories also compare equal, so the structure that
    # compared equal has to be shown to hold the operator's work. Only facts
    # every seeded version carries are named -- hand_issues arrived at schema 12,
    # settlements at 7 -- so the case that is really being run is the one the
    # parameter says.
    assert report.counts["sessions"] == 1
    assert report.counts["hands"] == 4
    assert report.counts["hand_issues"] == len(migrated["issues"])
    hands = migrated["sessions"][0]["hands"]
    assert [hand["hand_number"] for hand in hands] == [1, 2, 3, 4]
    assert [player["player_name"] for player in hands[1]["players"]] == [
        "Hero",
        "Villain",
    ]
    assert [action["action_type"] for action in hands[1]["actions"]] == ["bet", "call"]
    assert hands[1]["hero_cards"] == "Kh Kd"
    assert migrated["videos"][0]["stored_path"] == "videos/session.mp4"


def test_a_reconciled_hand_keeps_its_accounting_across_the_whole_chain(
    machine: dict[str, Path],
) -> None:
    """The seam that matters most: an authoritative reconciliation is not re-derived.

    A migrated hand whose settlement the operator declared and attested is the
    only kind of hand the product will let into Study. Its authority rests on
    rows in four tables and on an attestation recorded as a correction, and a
    restore that brought back three of the four would produce a hand that still
    lists, still opens, and is quietly no longer study-ready.
    """
    _seed_legacy(machine, LEGACY_VERSION)
    db = _open_migrated(machine)
    try:
        # Declared through the product's own writers, replacing the legacy row
        # this build's model refuses, so the hand reconciles the way an operator
        # would have made it reconcile: the recorded pot and hero result are
        # corrected to what the actions actually add up to, then the award is
        # declared and attested.
        stored = db.fetch_hand(2)
        assert stored is not None
        db.update_hand_facts(
            stored.model_copy(update={"pot_size": 20.0, "hero_bb_won": -10.0}),
            correction_notes="Frame shows a 20 chip pot.",
        )
        db.upsert_hand_settlement(
            HandSettlement(hand_id=2, status="unsettled", dead_money=0, rake_rate=0)
        )
        db.replace_settlement_entries(
            2,
            [
                SettlementEntry(
                    hand_id=2,
                    entry_type="award",
                    pot_index=0,
                    player_key="villain",
                    player_name="Villain",
                    amount=20,
                    entry_order=1,
                )
            ],
        )
        persist_reconciliation(db, 2)
        assert attest_declared_assumptions(db, 2, only="declared_pot_awards")
        assert reconcile_persisted_hand(db, 2).is_authoritative
        migrated = _study_history(db)
    finally:
        db.close()
    snapshot = backup_database(
        machine["database"], machine["backups"], data_dir=machine["data_root"]
    )
    _destroy(machine)

    report = _run_drill(machine, snapshot)

    assert report.outcome == "recovered", _describe(report)
    restored = PokerDatabase(machine["target"] / "poker_tracker.db")
    restored.init_db()
    try:
        recovered = _study_history(restored)
    finally:
        restored.close()

    assert _differences(migrated, recovered) == []
    hand = recovered["sessions"][0]["hands"][1]
    assert hand["accounting_authoritative"] is True
    assert hand["accounting_error"] is None
    assert hand["settlement"]["status"] == "reconciled"
    assert [
        (entry["entry_type"], entry["player_name"], entry["amount"])
        for entry in hand["settlement_entries"]
    ] == [("award", "Villain", 20.0)]
    assert "ACCOUNTING_NOT_AUTHORITATIVE" not in hand["blockers"]


def test_the_comparison_reports_a_seam_that_dropped_the_settlement(
    machine: dict[str, Path],
) -> None:
    """A guard that cannot fail proves nothing, so make the restore lose something.

    Deleting the award entry from the snapshot is what a backup that copied the
    hands but not their settlement would produce. Every structural check still
    passes on the result, the row counts the drill reports are unchanged, and the
    comparison above is the only thing that notices.
    """
    _seed_legacy(machine, LEGACY_VERSION)
    db = _open_migrated(machine)
    try:
        migrated = _study_history(db)
    finally:
        db.close()
    snapshot = backup_database(
        machine["database"], machine["backups"], data_dir=machine["data_root"]
    )
    _destroy(machine)
    with sqlite3.connect(snapshot) as connection:
        connection.execute("DELETE FROM settlement_entries WHERE entry_type = 'award'")

    report = _run_drill(machine, snapshot)
    restored = PokerDatabase(machine["target"] / "poker_tracker.db")
    restored.init_db()
    try:
        recovered = _study_history(restored)
    finally:
        restored.close()

    assert report.counts["hands"] == 4
    assert _check(report, "sqlite_quick_check").status == "pass"
    assert _check(report, "foreign_key_check").status == "pass"
    assert _check(report, "study_history_counts").status == "pass"
    differences = _differences(migrated, recovered)
    assert any("settlement_entries" in difference for difference in differences), (
        differences
    )


# --------------------------------------------------------------------------
# 3. The pre-migration snapshot: the rollback point, followed separately
# --------------------------------------------------------------------------


def test_the_pre_migration_snapshot_holds_what_the_migration_overwrote(
    machine: dict[str, Path],
) -> None:
    """Restore the rollback point in isolation and check the OPERATOR'S data came back.

    ``migration 13`` forces ``review_status`` back to ``needs_correction`` on
    every hand that is not provably manual, discarding a confirmation the
    operator made by hand. The snapshot ``init_db`` takes immediately before is
    the only artifact anywhere that still holds it, and until now nothing checked
    that it does -- only that the file opens.
    """
    _seed_legacy(machine, LEGACY_VERSION)
    before = _raw_facts(machine["database"])
    assert before["hands"][2]["review_status"] == "reviewed"

    db = _open_migrated(machine)
    try:
        after = {
            int(row["id"]): row["review_status"]
            for row in db._execute("SELECT id, review_status FROM hands").fetchall()
        }
    finally:
        db.close()
    assert after[2] == "needs_correction"

    snapshots = find_snapshots(machine["premigration_backups"], purpose="premigration")
    assert len(snapshots) == 1, sorted(
        path.name for path in machine["premigration_backups"].iterdir()
    )

    # Restored somewhere else entirely, and read there: the live file is not
    # touched, which is the whole procedure the runbook describes.
    rollback_root = machine["root"].parent / "rollback"
    rollback_root.mkdir()
    rollback = rollback_root / "restored.sqlite3"
    shutil.copyfile(snapshots[0], rollback)
    recovered = _raw_facts(rollback)

    assert recovered["schema_version"] == LEGACY_VERSION
    assert recovered["hands"] == before["hands"]
    assert recovered["settlement_entries"] == before["settlement_entries"]
    # The snapshot is genuinely the older shape, not the migrated file renamed:
    # a table this build creates is absent from it.
    assert "regression_cases" not in recovered["tables"]
    assert machine["database"].is_file()


def test_the_drill_can_verify_the_products_own_pre_migration_snapshot(
    machine: dict[str, Path],
) -> None:
    """The rollback point has to pass the drill the runbook gates a restore on.

    "Only once the drill exits 0: put the database in place" is the documented
    procedure, and the newest snapshot in the backup directory after an upgrade
    is the pre-migration one -- so this is the snapshot an operator drills first.

    Its inventory necessarily records the artifact sources the older schema did
    not have: ``ARTIFACT_PATH_COLUMNS`` names ``actions.source_image`` and the
    ``regression_cases`` paths, and a schema-12 file has neither. Those are the
    snapshot's shape, not a failure to inventory it, and reading them as an
    incomplete inventory told an operator holding an intact rollback point that
    it could not be shown to be complete.
    """
    _seed_legacy(machine, LEGACY_VERSION)
    _open_migrated(machine).close()
    snapshot = find_snapshots(
        machine["premigration_backups"], purpose="premigration"
    )[0]
    inventory = json.loads(inventory_path(snapshot).read_text(encoding="utf-8"))
    assert inventory["error"] is None
    assert inventory["unreadable_sources"] == [
        "actions.source_image",
        "regression_cases.fixture_path",
        "regression_cases.report_path",
    ]
    _destroy(machine)

    report = _run_drill(machine, snapshot)

    check = _check(report, "backup_inventory")
    assert check.status != "fail", (check.message, check.details)
    # Reported, not hidden: the older snapshot's inventory really does cover
    # fewer sources, and the operator is told which.
    assert any("actions.source_image" in detail for detail in check.details), check
    assert report.outcome == "recovered", _describe(report)
    assert report.exit_code == 0

    # Pinned because it is the easiest thing here to misread: the drill MIGRATES
    # the copy it restores, so a green drill on a pre-migration snapshot says the
    # history is recoverable, NOT that putting this file in place returns the
    # pre-migration state. The snapshot itself is left alone -- the test above is
    # what checks the rollback content, and it reads the file directly.
    drilled = _raw_facts(machine["target"] / "poker_tracker.db")
    assert drilled["schema_version"] == SCHEMA_VERSION
    assert drilled["hands"][2]["review_status"] == "needs_correction"
    assert _raw_facts(snapshot)["schema_version"] == LEGACY_VERSION
    assert _raw_facts(snapshot)["hands"][2]["review_status"] == "reviewed"


def test_an_inventory_that_could_not_read_a_source_the_snapshot_has_still_fails(
    machine: dict[str, Path],
) -> None:
    """The other direction of the same judgement, so the fix cannot swallow damage.

    A source the snapshot's schema DOES declare, recorded as unreadable, means
    the inventory ran against a database it could not query. Nothing about the
    snapshot's age explains that, and the completeness it claims is unfounded.
    """
    _seed_legacy(machine, LEGACY_VERSION)
    _open_migrated(machine).close()
    snapshot = find_snapshots(
        machine["premigration_backups"], purpose="premigration"
    )[0]
    path = inventory_path(snapshot)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unreadable_sources"] = [*payload["unreadable_sources"], "videos.stored_path"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    _destroy(machine)

    report = _run_drill(machine, snapshot)

    check = _check(report, "backup_inventory")
    assert check.status == "fail"
    assert any("videos.stored_path" in detail for detail in check.details), check
    assert report.outcome == "unverified"
    assert report.exit_code == 1


# --------------------------------------------------------------------------
# 4. Reading a recovered history the application tolerates
# --------------------------------------------------------------------------


def test_an_incomplete_draft_is_not_reported_as_a_history_that_did_not_come_back(
    machine: dict[str, Path],
) -> None:
    """A legacy hand with no seats recorded is readable; the drill said otherwise.

    ``build_ledger_from_records`` refuses a hand with no players, and the product
    handles that in both places it can happen: ``poker_visuals`` renders the
    hand's recorded observations without a derived ledger ("incomplete legacy/CV
    drafts can still be reviewed"), and ``resolve_hand_evidence`` carries the
    refusal into ``accounting_error`` so readiness reports it as a blocker.

    The drill read the same hand through a bare ``reconcile_persisted_hand`` and
    called the whole recovery PARTIAL -- "part of the study history did not come
    back" and exit 1 -- for a history in which nothing was missing. After schema
    13, a legacy database holds no completed hand at all, so the fallback subject
    is exactly one of these drafts and this is the ordinary outcome, not an edge.
    """
    _seed_legacy(machine, LEGACY_VERSION)
    db = _open_migrated(machine)
    try:
        assert not [
            hand for hand in db.fetch_all_hands() if hand.completion_status == "complete"
        ]
        drafts = [
            int(hand.id or 0)
            for hand in db.fetch_all_hands()
            if not db.fetch_players_by_hand(int(hand.id or 0))
        ]
    finally:
        db.close()
    assert drafts, "the fixture is meant to hold a hand with no seats recorded"
    snapshot = backup_database(
        machine["database"], machine["backups"], data_dir=machine["data_root"]
    )
    _destroy(machine)

    report = _run_drill(machine, snapshot)

    readback = _check(report, "completed_hand_readback")
    assert readback.status == "warning", (readback.message, readback.details)
    assert "no completed hand" in readback.message.lower()
    # The refusal is stated rather than hidden behind a pass.
    assert any("At least one player is required" in detail for detail in readback.details)
    assert report.outcome == "recovered", _describe(report)
    assert report.exit_code == 0


def test_a_completed_hand_the_ledger_refuses_is_still_a_failed_recovery(
    machine: dict[str, Path],
) -> None:
    """The guarantee the tolerance above must not swallow.

    A hand the history records as COMPLETE, whose ledger will not build, is
    damage: the reconciliation that made it complete no longer reproduces. Only
    the fallback draft is tolerated, and only because the product tolerates it.
    """
    _seed_legacy(machine, LEGACY_VERSION)
    db = _open_migrated(machine)
    try:
        db._execute("UPDATE hands SET completion_status = 'complete' WHERE id = 2")
        db._commit()
    finally:
        db.close()
    snapshot = backup_database(
        machine["database"], machine["backups"], data_dir=machine["data_root"]
    )
    _destroy(machine)
    with sqlite3.connect(snapshot) as connection:
        connection.execute("UPDATE hand_players SET starting_stack = -5")

    report = _run_drill(machine, snapshot)

    readback = _check(report, "completed_hand_readback")
    assert readback.status == "fail"
    assert "could not be read back" in readback.message
    assert report.outcome == "partial"
    assert report.exit_code == 1
    # Nothing structural noticed, which is why this check exists at all.
    assert _check(report, "sqlite_quick_check").status == "pass"
    assert _check(report, "study_history_counts").status == "pass"
