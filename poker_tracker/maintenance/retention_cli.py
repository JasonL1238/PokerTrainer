"""Operator CLI for the storage audit and retention sweep.

    python -m poker_tracker.maintenance.retention_cli            # audit only
    python -m poker_tracker.maintenance.retention_cli --apply    # delete what it listed

The audit is the default and the apply path re-runs it, so the operator always
sees the plan before anything is removed. There is no flag that deletes without
printing what it is about to delete.

The exit code is decided once, from the outcome, before either renderer runs.
``--json`` changes how the same outcome is written down and can never change
what it was: a script that branches on the exit code has to get the same answer
as the operator reading the text.

    0  audit only, or an apply that removed everything it offered
    1  an apply attempted a deletion and it failed
    2  the configuration was invalid; nothing was examined
    3  retention refused to act — a reference source was unreadable, or the
       plan went stale and a file it offered is referenced again
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from poker_tracker.maintenance.data_health import DEFAULT_DATA_DIR, DEFAULT_DATABASE_PATH
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.services.retention import (
    RETENTION_ENV_VARS,
    RetentionOutcome,
    RetentionPolicy,
    StorageAudit,
    apply_retention,
    audit_storage,
)
from poker_tracker.ui.video_storage import ensure_data_directories

# Outcome class -> exit code. The single place either output mode gets a status
# from; a renderer that computed its own would be free to disagree with the one
# a caller scripts against, which is exactly how --json and text drifted apart.
EXIT_CODES: dict[str, int] = {
    "nothing-to-do": 0,
    "would-delete": 0,
    "deleted": 0,
    "deletion-failed": 1,
    "error": 2,
    "refused": 3,
}


def exit_code_for(outcome_class: str) -> int:
    return EXIT_CODES[outcome_class]


def _human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit generated artifacts against the retention policy. "
            "Files the database still references are never offered for deletion."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete the files listed by the audit. Without this, nothing is removed.",
    )
    parser.add_argument(
        "--include-orphan-videos",
        action="store_true",
        help=(
            "Also offer source recordings no database row points at. "
            "Off by default: a recording is the one artifact nothing can rebuild."
        ),
    )
    parser.add_argument(
        "--purge-now",
        action="store_true",
        help=(
            "Ignore every retention window and offer all unreferenced managed "
            "artifacts regardless of age. This is the only way to get a zero-day "
            "window; setting one in the environment is refused. Still a dry run "
            "without --apply, and files the database references are still kept."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    return parser


@dataclass(frozen=True)
class _Report:
    """One run's result, classified before anything is printed."""

    audit: StorageAudit
    policy: RetentionPolicy
    data_dir: Path
    applied: bool
    outcome: RetentionOutcome | None

    @property
    def outcome_class(self) -> str:
        if self.outcome is not None and self.outcome.failures:
            return "deletion-failed"
        if self.audit.unreadable_references or (
            self.outcome is not None and self.outcome.skipped
        ):
            return "refused"
        if self.outcome is not None and self.outcome.removed:
            return "deleted"
        if self.audit.deletable:
            return "would-delete"
        return "nothing-to-do"

    @property
    def exit_code(self) -> int:
        return exit_code_for(self.outcome_class)


def _render_text(report: _Report) -> None:
    audit = report.audit
    policy = report.policy
    print(f"PokerTrainer storage audit: {report.data_dir}")
    if policy.purge_immediately:
        print(
            "PURGE NOW: retention windows ignored. Every unreferenced managed "
            "artifact is eligible regardless of age."
        )
    print("Retention windows (days):")
    for category, variable in sorted(RETENTION_ENV_VARS.items()):
        print(f"  {category:<14} {policy.window_days(category):>5}   [{variable}]")
    print()

    summary = audit.by_category()
    if not summary:
        print("No managed artifacts found.")
    for category in sorted(summary):
        stats = summary[category]
        print(
            f"{category:<14} {stats['files']:>6} files  {_human(stats['bytes']):>10}  "
            f"referenced={stats['referenced']:<6} "
            f"deletable={stats['deletable']} ({_human(stats['deletable_bytes'])})"
        )

    if audit.unreadable_references:
        print()
        print(
            "HELD BACK: could not read "
            f"{', '.join(audit.unreadable_references)}. Nothing will be deleted "
            "until every reference source can be checked."
        )
    if audit.errors:
        print()
        print("Errors while scanning:")
        for error in audit.errors[:20]:
            print(f"  - {error}")

    print()
    deletable = audit.deletable
    if deletable:
        print(
            f"Eligible for deletion: {len(deletable)} files, "
            f"{_human(audit.reclaimable_bytes)}"
        )
        for entry in deletable[:50]:
            print(f"  {entry.path}  ({entry.reason})")
        if len(deletable) > 50:
            print(f"  ... and {len(deletable) - 50} more")
    else:
        print("Nothing is eligible for deletion.")

    outcome = report.outcome
    if outcome is None:
        if deletable:
            print()
            print("Dry run. Re-run with --apply to delete the files listed above.")
        return

    print()
    print(
        f"Removed {len(outcome.removed)} files, "
        f"reclaimed {_human(outcome.reclaimed_bytes)}."
    )
    for skipped in outcome.skipped:
        print(f"  KEPT {skipped}")
    for failure in outcome.failures:
        print(f"  FAILED {failure}")


def _render_json(report: _Report) -> None:
    outcome = report.outcome
    payload = report.audit.to_dict()
    payload["applied"] = report.applied
    payload["removed"] = [str(p) for p in outcome.removed] if outcome else []
    payload["reclaimed_bytes"] = outcome.reclaimed_bytes if outcome else 0
    payload["failures"] = list(outcome.failures) if outcome else []
    payload["skipped"] = list(outcome.skipped) if outcome else []
    payload["purge_now"] = report.policy.purge_immediately
    payload["outcome"] = report.outcome_class
    payload["exit_code"] = report.exit_code
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _render_error(message: str, *, as_json: bool) -> int:
    """Report a run that never got as far as an audit, in the requested format."""
    print(message, file=sys.stderr)
    code = exit_code_for("error")
    if as_json:
        json.dump(
            {"outcome": "error", "error": message, "exit_code": code},
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    return code


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        policy = RetentionPolicy.from_env(
            include_orphan_videos=args.include_orphan_videos,
            purge_immediately=args.purge_now,
        )
    except ValueError as exc:
        return _render_error(
            f"Invalid retention configuration: {exc}", as_json=args.json
        )

    paths = ensure_data_directories(args.data_dir)
    db = PokerDatabase(args.db)
    try:
        db.init_db()
        audit = audit_storage(db, paths, policy)
        # The re-check inside apply_retention needs the connection open, so the
        # sweep happens here rather than after the database is closed.
        outcome = apply_retention(audit, confirm=True) if args.apply else None
    finally:
        db.close()

    report = _Report(
        audit=audit,
        policy=policy,
        data_dir=args.data_dir,
        applied=args.apply,
        outcome=outcome,
    )
    if args.json:
        _render_json(report)
    else:
        _render_text(report)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
