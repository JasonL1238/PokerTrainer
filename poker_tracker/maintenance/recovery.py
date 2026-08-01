"""Executable fresh-machine recovery drill.

Phase 11's exit gate is "a fresh machine with the repository, environment
configuration, persistent data directory, and verified backup can recover the
complete study history". The runbooks described the pieces in prose and nothing
executed the sequence, so a restore that produced an empty database and a
restore that produced the operator's whole history were indistinguishable
outcomes. This module performs the sequence and states which of the two
happened.

Two properties matter more than breadth of checking. The drill never touches the
live database -- it refuses to run at all when its target would land on the
operator's own root, rather than trusting the caller to have picked a safe one.
And a recovery whose external artifacts are gone is reported as PARTIAL rather
than as a passing restore with a warning attached: a database whose recordings
and timelines no longer exist has not recovered the study history, however
cleanly SQLite opens it.

Completeness is judged against the inventory ``persistence.backup_inventory``
writes beside each snapshot; this module consumes it and never builds a second
one. That inventory records artifacts but no row counts, so the session and hand
totals are currently self-reported by the restored database, and the drill says
so in its own report rather than letting a green run imply otherwise.

Exit contract, matching ``poker_tracker.release_gate.runner``: 0 the history was
recovered and verified, 1 recovery is incomplete or unverifiable, 2 the drill
could not be performed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import stat
import sys
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from poker_tracker.maintenance import data_health
from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence import backup_inventory
from poker_tracker.persistence.backup import resolve_artifact_path
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.models import Hand
from poker_tracker.services.hand_accounting import reconcile_persisted_hand
from poker_tracker.services.study_readiness import evaluate_study_readiness
from poker_tracker.ui.video_storage import ensure_data_directories

# The repository's one definition of the exit-code contract, imported rather than
# restated so a drill run from a script cannot disagree with the release gate
# about what "2" means.
from poker_tracker.validation.corpus import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_SETUP_INVALID,
)

CheckStatus = Literal["pass", "warning", "fail"]
Outcome = Literal["recovered", "unverified", "partial", "not_performed"]

RESTORED_DATABASE_NAME = "poker_tracker.db"
# Re-exported, never restated: the inventory beside a snapshot is written by
# poker_tracker.persistence.backup_inventory, and a second spelling of its name
# here would be a drill that looked for a file the product does not write.
INVENTORY_SUFFIX = backup_inventory.INVENTORY_SUFFIX
DETAIL_LIMIT = 20


@dataclass(frozen=True)
class RecoveryCheck:
    """One independently actionable verification performed on the restored copy."""

    name: str
    status: CheckStatus
    message: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackupInventory:
    """The drill's view of the inventory ``backup_inventory`` writes beside a snapshot.

    A ``.sqlite3`` snapshot records rows that point at videos, frames, timelines
    and solver outputs living outside SQLite. Nothing inside the snapshot says
    which of those existed when it was taken, so "the restored database
    references a file that is not here" cannot be told apart from "that file was
    never there". Both readings fit the same bytes and only one is a recovery
    failure, which is why the inventory is what makes completeness checkable at
    all.

    ``payload`` is kept whole and handed back to
    ``backup_inventory.inventory_findings`` rather than re-interpreted here: that
    module owns the artifact format, and a second reader of it would drift.

    DEPENDENCY, stated because the drill reports it every run: the inventory
    records artifacts but no row counts, so "the restored history has the same
    number of sessions and hands as the source" cannot be established from it.
    ``counts`` reads a ``counts`` block if one is ever added, under either the
    nested or the flat spelling; until then the drill says so instead of implying
    the totals were checked.
    """

    inventory_path: Path
    payload: dict[str, Any]
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveryDrillReport:
    """The verdict, the evidence behind it, and the numbers to compare by hand."""

    outcome: Outcome
    backup_path: str
    data_root: str
    target_root: str
    restored_database: str
    checked_at: str
    counts: dict[str, int]
    missing_artifacts: tuple[str, ...]
    checks: tuple[RecoveryCheck, ...]

    @property
    def exit_code(self) -> int:
        if self.outcome == "recovered":
            return EXIT_OK
        if self.outcome == "not_performed":
            return EXIT_SETUP_INVALID
        return EXIT_GATE_FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "exit_code": self.exit_code,
            "backup_path": self.backup_path,
            "data_root": self.data_root,
            "target_root": self.target_root,
            "restored_database": self.restored_database,
            "checked_at": self.checked_at,
            "counts": dict(self.counts),
            "missing_artifacts": list(self.missing_artifacts),
            "checks": [asdict(check) for check in self.checks],
        }


def run_recovery_drill(
    *,
    backup_path: str | Path,
    data_root: str | Path,
    target_root: str | Path,
    inventory_path: str | Path | None = None,
    expected_schema_version: int = SCHEMA_VERSION,
) -> RecoveryDrillReport:
    """Restore ``backup_path`` under ``target_root`` and verify the history came back.

    ``data_root`` is the persistent data directory the operator brought with
    them; it is only ever read, and it is where the restored rows' videos,
    frames, timelines and solver outputs are looked for. ``target_root`` is the
    throwaway location the restored database is written into. Nothing is written
    anywhere else, and the drill refuses outright rather than writing into a root
    that could be the live one.
    """

    backup = _absolute(backup_path)
    data = _absolute(data_root)
    target = _absolute(target_root)
    restored = target / RESTORED_DATABASE_NAME
    started_at = datetime.now(UTC).isoformat()

    def report(
        outcome: Outcome,
        checks: Sequence[RecoveryCheck],
        *,
        counts: dict[str, int] | None = None,
        missing: Sequence[str] = (),
    ) -> RecoveryDrillReport:
        return RecoveryDrillReport(
            outcome=outcome,
            backup_path=str(backup),
            data_root=str(data),
            target_root=str(target),
            restored_database=str(restored),
            checked_at=started_at,
            counts=counts or {},
            missing_artifacts=tuple(missing),
            checks=tuple(checks),
        )

    refusals = _refusals(backup=backup, data_root=data, target_root=target,
                         restored=restored)
    if refusals:
        return report(
            "not_performed",
            [
                RecoveryCheck(
                    "drill_preconditions",
                    "fail",
                    "The drill refused to run; nothing was restored or verified.",
                    _limited(refusals),
                )
            ],
        )

    try:
        ensure_data_directories(target)
        shutil.copyfile(backup, restored)
        # A snapshot written by an older build can still be in WAL mode with live
        # sidecars. Copying them keeps the restored file from silently losing the
        # committed frames they hold.
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{backup}{suffix}")
            if sidecar.is_file():
                shutil.copyfile(sidecar, Path(f"{restored}{suffix}"))
    except OSError as exc:
        return report(
            "not_performed",
            [
                RecoveryCheck(
                    "restored_database",
                    "fail",
                    "The backup could not be copied into the drill location.",
                    (f"{type(exc).__name__}: {exc}",),
                )
            ],
        )

    checks: list[RecoveryCheck] = [
        RecoveryCheck(
            "restored_database",
            "pass",
            f"Backup restored to {restored} ({restored.stat().st_size} bytes).",
        )
    ]

    inventory, inventory_error = _load_inventory(backup, inventory_path)

    with _snapshots_redirected_to(target / "backups"):
        db = PokerDatabase(restored)
        try:
            try:
                db.init_db()
            except (RuntimeError, sqlite3.Error) as exc:
                checks.append(
                    RecoveryCheck(
                        "restored_open",
                        "fail",
                        "This build cannot open the restored database.",
                        (f"{type(exc).__name__}: {exc}",),
                    )
                )
                return report(_outcome(checks), checks)
            checks.append(
                RecoveryCheck(
                    "restored_open",
                    "pass",
                    "The restored database opened and migrated to schema "
                    f"version {db.schema_version()}.",
                )
            )
            checks.extend(
                _structural_checks(
                    restored,
                    data_root=data,
                    expected_schema_version=expected_schema_version,
                )
            )
            checks.append(_inventory_check(inventory, inventory_error))
            counts, count_checks = _history_checks(db, inventory)
            checks.extend(count_checks)
            checks.append(_issue_evidence_check(db))
            checks.append(_completed_hand_readback_check(db))
            artifact_check, missing = _artifact_checks(db, restored, data, inventory)
            checks.append(artifact_check)
        finally:
            db.close()

    return report(_outcome(checks), checks, counts=counts, missing=missing)


def _refusals(
    *, backup: Path, data_root: Path, target_root: Path, restored: Path
) -> list[str]:
    """Every reason this drill must not start, stated before anything is written.

    The isolation rules are checked here rather than trusted to the caller
    because the failure they prevent is unrecoverable: a drill that restored a
    three-month-old snapshot over ``poker_tracker.db`` would destroy exactly the
    history it was run to protect. ``data_health``'s constants are the single
    definition of where the live roots are, and they are read through the module
    so that redirecting them redirects this too.
    """
    live_database = _absolute(data_health.DEFAULT_DATABASE_PATH)
    live_data = _absolute(data_health.DEFAULT_DATA_DIR)
    reasons: list[str] = []

    if restored.resolve() == live_database.resolve():
        reasons.append(
            f"the restored database would be written to the live database "
            f"{live_database}"
        )
    if _is_within(live_database, target_root):
        reasons.append(
            f"the drill location {target_root} contains the live database "
            f"{live_database}"
        )
    if _is_within(live_data, target_root) or _is_within(target_root, live_data):
        reasons.append(
            f"the drill location {target_root} overlaps the live data directory "
            f"{live_data}"
        )
    if restored.exists():
        # Overwriting whatever is already there would make the drill's own result
        # depend on what a previous run left behind.
        reasons.append(f"a database already exists at {restored}")

    try:
        backup_stat = backup.stat()
    except OSError as exc:
        reasons.append(f"the backup cannot be read: {type(exc).__name__}: {exc}")
    else:
        if not stat.S_ISREG(backup_stat.st_mode):
            reasons.append(f"the backup {backup} is not a regular file")
        elif backup.resolve() == live_database.resolve():
            reasons.append(
                "the live database is not a backup of itself; supply a snapshot"
            )

    if not data_root.is_dir():
        reasons.append(
            f"the persistent data directory {data_root} does not exist, so no "
            "artifact reference can be resolved"
        )
    return reasons


@contextmanager
def _snapshots_redirected_to(directory: Path) -> Iterator[None]:
    """Keep any snapshot the restored copy triggers inside the drill's own tree.

    ``init_db`` snapshots a real database before migrating it, and
    ``backup_database`` resolves its destination from
    ``persistence.backup.BACKUPS_DIR`` at call time. Restoring an older snapshot
    would therefore write a pinned pre-migration copy into the OPERATOR'S
    retained set, so running the drill would consume a retention slot holding a
    real rollback point. That constant is the single definition of the backup
    directory, which is why the redirect happens there and not on a copy.
    """
    directory.mkdir(parents=True, exist_ok=True)
    previous = backup_module.BACKUPS_DIR
    backup_module.BACKUPS_DIR = directory
    try:
        yield
    finally:
        backup_module.BACKUPS_DIR = previous


# Health checks the drill deliberately does not republish as its own verdict.
# `artifact_files` asks the drill's question about a different subject: it
# resolves paths against the live layout and counts a NULL or blank column as a
# missing file, so a manual hand's unset `actions.source_image` and a solver run
# with no log reads as lost data. `recovered_artifacts` below answers the same
# question against the recovered data root, over the same exhaustive column list
# plus the timelines that have no column, and skips references that were never
# set. Two answers to one question, one of them wrong, is worse than one.
# `backups` audits the operator's whole retained set, which is
# `python -m poker_tracker.maintenance --restore-backups`, not this drill.
_UNFOLDED_HEALTH_CHECKS = frozenset({"artifact_files", "backups"})


def _structural_checks(
    restored: Path,
    *,
    data_root: Path,
    expected_schema_version: int,
) -> list[RecoveryCheck]:
    """Integrity, foreign keys, schema version and contract, from the existing audit.

    ``audit_data_health`` already answers these questions read-only, so the drill
    consumes it rather than re-implementing a second auditor that could disagree
    with the one operators run by hand.
    """
    report = data_health.audit_data_health(
        restored,
        data_dir=data_root,
        # The drill's own tree, not the operator's retained set: the `backups`
        # check is discarded below, and pointing it at a real backup directory
        # would open every snapshot there for a result nothing reads.
        backup_dir=restored.parent / "backups",
        expected_schema_version=expected_schema_version,
        restore_backups=False,
    )
    return [
        RecoveryCheck(check.name, check.status, check.message, tuple(check.details))
        for check in report.checks
        if check.name not in _UNFOLDED_HEALTH_CHECKS
    ]


def _history_checks(
    db: PokerDatabase, inventory: BackupInventory | None
) -> tuple[dict[str, int], list[RecoveryCheck]]:
    """Count the study history through the surfaces the application reads it with.

    The comparison against the inventory lives here rather than with the
    inventory check because a count that disagrees with the source is damage,
    while a missing inventory is only unproven, and the verdict has to be able to
    tell an operator which of the two happened.
    """
    sessions = db.fetch_sessions()
    hands = db.fetch_all_hands()
    completed = [hand for hand in hands if hand.completion_status == "complete"]
    issues = db.fetch_hand_issues()
    counts = {
        "sessions": len(sessions),
        "hands": len(hands),
        "completed_hands": len(completed),
        "hand_issues": len(issues),
    }
    summary = ", ".join(f"{name}={value}" for name, value in counts.items())
    if not sessions or not hands:
        # A restore that opens cleanly and holds nothing is the most dangerous
        # result available here: every structural check passes on it.
        return counts, [
            RecoveryCheck(
                "study_history_counts",
                "fail",
                "The restored database contains no study history.",
                (summary,),
            )
        ]
    stated = inventory.counts if inventory is not None else {}
    mismatches = [
        f"{name}: inventory={expected}, restored={counts[name]}"
        for name, expected in sorted(stated.items())
        if name in counts and counts[name] != expected
    ]
    if mismatches:
        return counts, [
            RecoveryCheck(
                "study_history_counts",
                "fail",
                f"The restored history is short of what the inventory records "
                f"in {len(mismatches)} count(s).",
                _limited([summary, *mismatches]),
            )
        ]
    verified = " and matches the inventory" if stated else ""
    return counts, [
        RecoveryCheck(
            "study_history_counts",
            "pass",
            f"The restored history reads back as {summary}{verified}.",
        )
    ]


def _inventory_check(
    inventory: BackupInventory | None, inventory_error: str | None
) -> RecoveryCheck:
    """Is there an independent record of what this snapshot was supposed to hold?

    Comparing a restored database against itself proves nothing, so completeness
    is only establishable when the snapshot carries a record written beside it.
    Its absence is reported as a failure rather than a warning: the gate asks
    whether the COMPLETE history came back, and "unknown" is not a yes.
    """
    if inventory_error is not None:
        return RecoveryCheck(
            "backup_inventory",
            "fail",
            "The inventory beside this backup could not be read.",
            (inventory_error,),
        )
    if inventory is None:
        return RecoveryCheck(
            "backup_inventory",
            "fail",
            "No inventory accompanies this backup, so the restored history "
            "cannot be shown to be complete.",
            (
                f"expected a JSON file named <snapshot>{INVENTORY_SUFFIX}",
                "counts and artifact references below were derived from the "
                "restored database alone",
            ),
        )
    # The inventory's own record of what it could not see. An inventory that
    # failed halfway describes a smaller history than the snapshot holds, and
    # every completeness answer drawn from it would be optimistic.
    incomplete: list[str] = []
    if inventory.payload.get("error"):
        incomplete.append(f"inventory failed: {inventory.payload['error']}")
    unreadable = inventory.payload.get("unreadable_sources") or []
    if isinstance(unreadable, list):
        incomplete.extend(f"inventory could not read {source}" for source in unreadable)
    if incomplete:
        return RecoveryCheck(
            "backup_inventory",
            "fail",
            "The inventory beside this backup is itself incomplete.",
            _limited(incomplete),
        )
    if not inventory.counts:
        return RecoveryCheck(
            "backup_inventory",
            "warning",
            "The inventory records artifacts but no session or hand counts, so "
            "the restored totals could not be compared with the source.",
            (
                str(inventory.inventory_path),
                "artifact references are verified; totals are self-reported",
            ),
        )
    stated = ", ".join(
        f"{name}={value}" for name, value in sorted(inventory.counts.items())
    )
    return RecoveryCheck(
        "backup_inventory",
        "pass",
        f"The backup carries an inventory recording {stated}.",
    )


def _issue_evidence_check(db: PokerDatabase) -> RecoveryCheck:
    """Frozen issue evidence has to survive a restore or the issue is unusable.

    A ``hand_issues`` row whose ``evidence_snapshot`` came back empty still lists
    in the queue and still blocks its hand, but the frozen record of what the
    hand looked like when it was flagged -- the entire reason the row exists -- is
    gone, and nothing else in the database would report it.
    """
    issues = db.fetch_hand_issues()
    if not issues:
        return RecoveryCheck(
            "issue_evidence",
            "pass",
            "No hand issues were recorded, so no issue evidence had to survive.",
        )
    problems: list[str] = []
    for issue in issues:
        snapshot = issue.evidence_snapshot
        if not isinstance(snapshot, dict) or not snapshot:
            problems.append(f"issue {issue.id}: evidence snapshot is empty")
            continue
        recorded_hand = snapshot.get("hand")
        if isinstance(recorded_hand, dict):
            recorded_id = recorded_hand.get("id")
            if recorded_id is not None and recorded_id != issue.hand_id:
                problems.append(
                    f"issue {issue.id}: evidence names hand {recorded_id}, "
                    f"row names hand {issue.hand_id}"
                )
    if problems:
        return RecoveryCheck(
            "issue_evidence",
            "fail",
            f"{len(problems)} of {len(issues)} issue(s) lost their evidence.",
            _limited(problems),
        )
    return RecoveryCheck(
        "issue_evidence",
        "pass",
        f"All {len(issues)} recorded issue(s) kept their evidence snapshot.",
    )


def _completed_hand_readback_check(db: PokerDatabase) -> RecoveryCheck:
    """Read one hand the way the application does, not the way SQLite does.

    ``PRAGMA quick_check`` and a row count both pass on a database the product
    cannot actually open a hand from -- a column the models refuse, a settlement
    the reconciliation cannot cross-check, evidence the readiness rules cannot
    parse. The claim being tested is that the APPLICATION can read the recovered
    history, so this composes the same evidence a Study page does and fails on
    anything that raises on the way.
    """
    hands = db.fetch_all_hands()
    if not hands:
        return RecoveryCheck(
            "completed_hand_readback",
            "fail",
            "The restored database holds no hand to read back.",
        )
    completed = [hand for hand in hands if hand.completion_status == "complete"]
    subject = completed[0] if completed else hands[0]
    if subject.id is None:
        return RecoveryCheck(
            "completed_hand_readback",
            "fail",
            "The restored hand has no identifier and cannot be read back.",
        )
    try:
        details = _read_hand_end_to_end(db, subject)
    except Exception as exc:  # noqa: BLE001 - any failure to read is the finding
        return RecoveryCheck(
            "completed_hand_readback",
            "fail",
            f"Hand {subject.id} could not be read back through the application.",
            (f"{type(exc).__name__}: {exc}",),
        )
    if not completed:
        return RecoveryCheck(
            "completed_hand_readback",
            "warning",
            f"No completed hand was restored; hand {subject.id} "
            f"({subject.completion_status}) reads back end to end instead.",
            details,
        )
    return RecoveryCheck(
        "completed_hand_readback",
        "pass",
        f"Completed hand {subject.id} reads back end to end.",
        details,
    )


def _read_hand_end_to_end(db: PokerDatabase, hand: Hand) -> tuple[str, ...]:
    """Compose one hand from durable records exactly as a study surface does."""
    hand_id = int(hand.id or 0)
    stored = db.fetch_hand(hand_id)
    if stored is None:
        raise LookupError(f"hand {hand_id} disappeared between listing and reading")
    session = db.fetch_session(stored.session_id)
    if session is None:
        raise LookupError(f"hand {hand_id} references missing session "
                          f"{stored.session_id}")
    players = db.fetch_players_by_hand(hand_id)
    actions = db.fetch_actions_by_hand(hand_id)
    settlement = db.fetch_hand_settlement(hand_id)
    entries = db.fetch_settlement_entries(hand_id)
    accounting = reconcile_persisted_hand(db, hand_id)
    readiness = evaluate_study_readiness(
        stored,
        accounting=accounting,
        hand_issues=db.fetch_hand_issues(hand_id=hand_id),
        coaching_reviews=db.fetch_coaching_reviews_by_hand(hand_id),
        hand_reviews=db.fetch_reviews_by_hand(hand_id),
        solver_runs=db.fetch_solver_runs_by_hand(hand_id),
    )
    blockers = ", ".join(blocker.code for blocker in readiness.blockers) or "none"
    return (
        f"session: {session.name}",
        f"players: {len(players)}",
        f"actions: {len(actions)}",
        f"settlement: {'absent' if settlement is None else settlement.status}",
        f"settlement entries: {len(entries)}",
        f"accounting authoritative: {accounting.is_authoritative}",
        f"study ready: {readiness.is_ready} (blockers: {blockers})",
    )


def _artifact_checks(
    db: PokerDatabase,
    restored: Path,
    data_root: Path,
    inventory: BackupInventory | None,
) -> tuple[RecoveryCheck, tuple[str, ...]]:
    """Which files the recovered history needs and which of them are not here.

    The reference set comes from ``PokerDatabase.ARTIFACT_PATH_COLUMNS``, which
    retention already treats as exhaustive, plus the CV timelines that have no
    column at all and are addressed by convention. A missing artifact is a
    failure and not a warning: a recovered database whose recordings are gone is
    a partial recovery, and calling it anything softer is how a partial recovery
    gets filed as a good one.
    """
    referenced, unreadable = db.referenced_artifact_paths()
    timelines, timeline_error = _referenced_timelines(restored, data_root)
    if timeline_error is not None:
        unreadable = [*unreadable, timeline_error]
    # An inventory entry the restored rows no longer name is still part of the
    # history this snapshot described; leaving it out would let a lost row hide a
    # lost file.
    wanted = sorted(
        {*referenced, *timelines, *_inventoried_paths(inventory)}
    )
    missing = [
        stored
        for stored in wanted
        if not resolve_artifact_path(stored, restored, data_root).is_file()
    ]
    absolute_missing = [stored for stored in missing if Path(stored).is_absolute()]
    replaced = _replaced_since_snapshot(inventory, restored, data_root)
    if unreadable:
        return (
            RecoveryCheck(
                "recovered_artifacts",
                "fail",
                "The recovered database could not be asked what files it needs.",
                _limited([f"unreadable reference source: {item}"
                          for item in unreadable]),
            ),
            tuple(missing),
        )
    if missing:
        details = [f"missing: {stored}" for stored in missing[:DETAIL_LIMIT]]
        if len(missing) > DETAIL_LIMIT:
            details.append(f"... and {len(missing) - DETAIL_LIMIT} more")
        if absolute_missing:
            # Worth naming explicitly: this is the ordinary way a technically
            # complete data directory still fails the drill.
            details.append(
                f"{len(absolute_missing)} of these are absolute paths recorded on "
                "the machine that wrote them; mount the data directory at the "
                "same path it had there, or the references cannot resolve"
            )
        details.extend(replaced)
        return (
            RecoveryCheck(
                "recovered_artifacts",
                "fail",
                f"PARTIAL RECOVERY: {len(missing)} of {len(wanted)} referenced "
                f"artifact(s) are not present under {data_root}.",
                _limited(details),
            ),
            tuple(missing),
        )
    if replaced:
        return (
            RecoveryCheck(
                "recovered_artifacts",
                "fail",
                f"PARTIAL RECOVERY: all {len(wanted)} referenced artifact(s) are "
                f"present, but {len(replaced)} no longer match the snapshot.",
                _limited(replaced),
            ),
            (),
        )
    return (
        RecoveryCheck(
            "recovered_artifacts",
            "pass",
            f"All {len(wanted)} artifact(s) the recovered history references are "
            "present.",
        ),
        (),
    )


def _replaced_since_snapshot(
    inventory: BackupInventory | None, restored: Path, data_root: Path
) -> list[str]:
    """Artifacts that are still there but are no longer the file the snapshot saw.

    A recording that was replaced or truncated after the snapshot restores as a
    present path pointing at different bytes, which every existence check calls
    healthy. The inventory recorded each file's size, so the drill can say the
    recovered history's evidence is not the evidence it was taken with.

    The size comparison is done here rather than through
    ``backup_inventory.inventory_findings`` because that function returns missing
    and changed files in one flat list of sentences, and the drill has to
    attribute the two to different verdicts.
    """
    if inventory is None:
        return []
    entries = inventory.payload.get("artifacts")
    if not isinstance(entries, list):
        return []
    findings: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("present"):
            continue
        stored = entry.get("path")
        recorded = _optional_int(entry.get("bytes"))
        if not isinstance(stored, str) or recorded is None:
            continue
        resolved = resolve_artifact_path(stored.strip(), restored, data_root)
        try:
            if not resolved.is_file():
                continue
            actual = resolved.stat().st_size
        except OSError:
            continue
        if actual != recorded:
            findings.append(
                f"replaced since the snapshot: {stored} was {recorded} bytes, "
                f"is now {actual}"
            )
    return findings


def _referenced_timelines(
    restored: Path, data_root: Path
) -> tuple[list[str], str | None]:
    """Timeline paths a completed reconstruction job implies, on THIS machine.

    Timelines are the one artifact class with no database column, and a missing
    one hard-blocks every later validated-hand import, so a recovery that lost
    them has lost work even though no row dangles.
    ``backup_inventory.timeline_paths`` owns the naming rule; the drill supplies
    the recovered machine's timeline directory so a data root mounted somewhere
    new still resolves.
    """
    try:
        # A separate read-only connection because no public reader enumerates
        # every job -- fetch_recent_jobs is capped, and an exhaustive question
        # must not be answered from a truncated list.
        with closing(
            sqlite3.connect(
                f"{restored.resolve().as_uri()}?mode=ro", uri=True, timeout=5
            )
        ) as connection:
            connection.execute("PRAGMA query_only = ON")
            return backup_inventory.timeline_paths(
                connection, data_root / "cv_timelines"
            ), None
    except sqlite3.Error as exc:
        return [], f"{backup_inventory.TIMELINE_SOURCE}: {type(exc).__name__}: {exc}"


def _load_inventory(
    backup: Path, explicit: str | Path | None
) -> tuple[BackupInventory | None, str | None]:
    """The inventory beside ``backup``, or the reason it could not be used.

    ``backup_inventory.load_inventory`` returns None both for "no inventory" and
    for "an inventory that will not parse". Those are different findings -- the
    second means the operator has a file they believe describes this snapshot and
    it does not -- so the file's existence is checked separately here.
    """
    path = _absolute(explicit) if explicit is not None else (
        backup_inventory.inventory_path(backup)
    )
    if not path.is_file():
        if explicit is not None:
            return None, f"{path} does not exist"
        return None, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{path}: {type(exc).__name__}: {exc}"
    if not isinstance(raw, dict):
        return None, f"{path}: inventory is not a JSON object"
    return (
        BackupInventory(
            inventory_path=path,
            payload=raw,
            counts=_inventory_counts(raw),
        ),
        None,
    )


# The count names the drill compares, each with the spellings an inventory may
# reasonably use for it. Accepting both keeps the reader from rejecting a valid
# inventory over a plural, which would read to an operator as a lost backup.
_COUNT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sessions", ("sessions", "session_count")),
    ("hands", ("hands", "hand_count")),
    ("completed_hands", ("completed_hands", "completed_hand_count")),
    ("hand_issues", ("hand_issues", "hand_issue_count", "issues", "issue_count")),
)


def _inventory_counts(raw: dict[str, Any]) -> dict[str, int]:
    """Counts under either the nested or the flat spelling, ignoring the rest."""
    nested = raw.get("counts")
    source: dict[str, Any] = nested if isinstance(nested, dict) else raw
    counts: dict[str, int] = {}
    for name, aliases in _COUNT_ALIASES:
        for key in aliases:
            value = _optional_int(source.get(key))
            if value is not None:
                counts[name] = value
                break
    return counts


def _inventoried_paths(inventory: BackupInventory | None) -> set[str]:
    """Paths the inventory says existed when the snapshot was taken."""
    if inventory is None:
        return set()
    entries = inventory.payload.get("artifacts")
    if not isinstance(entries, list):
        return set()
    return {
        stored.strip()
        for entry in entries
        if isinstance(entry, dict) and entry.get("present")
        for stored in (entry.get("path"),)
        if isinstance(stored, str) and stored.strip()
    }


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _outcome(checks: Sequence[RecoveryCheck]) -> Outcome:
    """Separate "something is provably gone" from "completeness is unknown".

    Both exit 1, because neither is a verified recovery, but an operator whose
    only failure is a missing inventory has a different problem from one whose
    recordings are gone, and one word in the verdict is what tells them apart.
    """
    failed = {check.name for check in checks if check.status == "fail"}
    if not failed:
        return "recovered"
    if failed == {"backup_inventory"}:
        return "unverified"
    return "partial"


def _absolute(value: str | Path) -> Path:
    path = Path(value)
    try:
        path = path.expanduser()
    except RuntimeError:
        pass
    return path.absolute()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _limited(details: Sequence[str]) -> tuple[str, ...]:
    values = list(details)
    if len(values) <= DETAIL_LIMIT:
        return tuple(values)
    return (*values[:DETAIL_LIMIT], f"... and {len(values) - DETAIL_LIMIT} more")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m poker_tracker.maintenance.recovery",
        description=(
            "Restore a PokerTrainer backup into an isolated location and verify "
            "that the study history came back."
        ),
    )
    parser.add_argument(
        "--backup", type=Path, required=True, help="Snapshot to restore."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=data_health.DEFAULT_DATA_DIR,
        help="Persistent data directory holding videos, frames and timelines.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Empty throwaway directory the restored database is written into.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help=f"Inventory to verify against (default: <backup>{INVENTORY_SUFFIX}).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit a machine-readable JSON report."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_recovery_drill(
        backup_path=args.backup,
        data_root=args.data_dir,
        target_root=args.target,
        inventory_path=args.inventory,
    )
    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_report(report)
    return report.exit_code


_OUTCOME_HEADLINE: dict[str, str] = {
    "recovered": "RECOVERED: the study history came back and was verified.",
    "unverified": "UNVERIFIED: the history restored, but completeness is unproven.",
    "partial": "PARTIAL: part of the study history did not come back.",
    "not_performed": "NOT PERFORMED: the drill could not run.",
}


def _print_report(report: RecoveryDrillReport) -> None:
    print(_OUTCOME_HEADLINE[report.outcome])
    print(f"Backup: {report.backup_path}")
    print(f"Data directory: {report.data_root}")
    print(f"Restored to: {report.restored_database}")
    if report.counts:
        print(
            "Counts: "
            + ", ".join(f"{name}={value}" for name, value in report.counts.items())
        )
    for check in report.checks:
        print(f"[{check.status.upper()}] {check.name}: {check.message}")
        for detail in check.details:
            print(f"  - {detail}")


if __name__ == "__main__":
    raise SystemExit(main())
