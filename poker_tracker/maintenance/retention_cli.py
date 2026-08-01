"""Operator CLI for the storage audit and retention sweep.

    python -m poker_tracker.maintenance.retention            # audit only
    python -m poker_tracker.maintenance.retention --apply    # delete what it listed

The audit is the default and the apply path re-runs it, so the operator always
sees the plan before anything is removed. There is no flag that deletes without
printing what it is about to delete.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from poker_tracker.maintenance.data_health import DEFAULT_DATA_DIR, DEFAULT_DATABASE_PATH
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.services.retention import (
    RETENTION_ENV_VARS,
    RetentionPolicy,
    StorageAudit,
    apply_retention,
    audit_storage,
)
from poker_tracker.ui.video_storage import ensure_data_directories


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
    parser.add_argument("--json", action="store_true", help="Emit a JSON report.")
    return parser


def _print_audit(audit: StorageAudit, policy: RetentionPolicy, *, data_dir: Path) -> None:
    print(f"PokerTrainer storage audit: {data_dir}")
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
    if not deletable:
        print("Nothing is eligible for deletion.")
        return
    print(f"Eligible for deletion: {len(deletable)} files, {_human(audit.reclaimable_bytes)}")
    for entry in deletable[:50]:
        print(f"  {entry.path}  ({entry.reason})")
    if len(deletable) > 50:
        print(f"  ... and {len(deletable) - 50} more")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        policy = RetentionPolicy.from_env(
            include_orphan_videos=args.include_orphan_videos
        )
    except ValueError as exc:
        print(f"Invalid retention configuration: {exc}", file=sys.stderr)
        return 2

    paths = ensure_data_directories(args.data_dir)
    db = PokerDatabase(args.db)
    try:
        db.init_db()
        audit = audit_storage(db, paths, policy)
    finally:
        db.close()

    if args.json:
        payload = audit.to_dict()
        payload["applied"] = False
        if args.apply:
            outcome = apply_retention(audit, confirm=True)
            payload["applied"] = True
            payload["removed"] = [str(p) for p in outcome.removed]
            payload["reclaimed_bytes"] = outcome.reclaimed_bytes
            payload["failures"] = outcome.failures
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    _print_audit(audit, policy, data_dir=args.data_dir)
    if not args.apply:
        if audit.deletable:
            print()
            print("Dry run. Re-run with --apply to delete the files listed above.")
        return 0

    outcome = apply_retention(audit, confirm=True)
    print()
    print(f"Removed {len(outcome.removed)} files, reclaimed {_human(outcome.reclaimed_bytes)}.")
    for failure in outcome.failures:
        print(f"  FAILED {failure}")
    return 1 if outcome.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
