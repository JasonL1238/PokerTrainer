"""CLI: ``python -m poker_tracker.release_gate``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from poker_tracker.release_gate.runner import run_release_gate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executable private-beta release gate (Phase 3).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("validation/clubwpt_v1.json"),
        help="Corpus manifest path",
    )
    parser.add_argument(
        "--mode",
        choices=("fixture", "full", "container"),
        default="fixture",
        help="fixture=CI schemas+optional predictions; full/container fail closed until wired",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("data/release_reports"),
        help="Directory for the deterministic JSON report",
    )
    parser.add_argument(
        "--require-recordings",
        action="store_true",
        help="Verify raw videos under POKER_VALIDATION_ROOT (implied for full/container)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_release_gate(
        manifest_path=args.manifest,
        mode=args.mode,
        report_dir=args.report_dir,
        require_recordings=args.require_recordings,
    )
    print(json.dumps(result.report, indent=2, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
