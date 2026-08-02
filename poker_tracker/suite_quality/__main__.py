"""CLI: ``python -m poker_tracker.suite_quality``.

    skips      audit every skip declaration in a test tree
    flake      run the suite N times, shuffling order, and name what disagreed
    coverage   summarise a `coverage json` payload per core module

Exit codes are gate-usable: 0 clean, 2 the check found something, 1 misuse.
``coverage`` never exits 2 -- it reports, and a percentage is not a verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poker_tracker.suite_quality import coverage_report, flake, skip_policy

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FOUND = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m poker_tracker.suite_quality",
        description="Suite-quality checks for the PLAN Phase 14 exit gate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    skips = sub.add_parser("skips", help="Audit skip declarations for an evaluable reason")
    skips.add_argument("roots", nargs="*", type=Path, default=[Path("tests")])

    hunt = sub.add_parser("flake", help="Repeat the suite under shuffled order and compare")
    hunt.add_argument("--passes", type=int, default=5)
    hunt.add_argument("--seeds", default="", help="comma-separated seeds for the shuffled passes")
    hunt.add_argument("--repo", type=Path, default=Path("."))
    hunt.add_argument("--report-dir", type=Path, default=Path("data/suite_quality"))
    hunt.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="extra argument passed through to each pytest pass (repeatable)",
    )
    hunt.add_argument(
        "--plugin",
        default=flake.PLUGIN,
        help="import path of the shuffling plugin as seen from --repo",
    )

    cov = sub.add_parser("coverage", help="Per-core-module summary of a coverage JSON payload")
    cov.add_argument("payload", type=Path)
    cov.add_argument("--threshold", type=float, default=60.0)
    cov.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="repository root, used to find core files the payload never mentions",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "skips":
        verdicts = [verdict for root in args.roots for verdict in skip_policy.audit(root)]
        print(skip_policy.format_report(verdicts))
        stale = skip_policy.stale_registrations(verdicts)
        if stale:
            print("\nReviewed skips no test claims any more (delete the registration):")
            for reason in stale:
                print(f"  {reason!r}")
        found = skip_policy.violations(verdicts)
        return EXIT_FOUND if found or stale else EXIT_OK

    if args.command == "flake":
        if args.passes < 2:
            print("flake needs at least 2 passes to compare anything", file=sys.stderr)
            return EXIT_USAGE
        seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
        report = flake.hunt(
            repo=args.repo.resolve(),
            report_dir=args.report_dir.resolve(),
            passes=args.passes,
            seeds=seeds or None,
            pytest_args=args.pytest_arg,
            plugin=args.plugin,
        )
        flake.write_report(report, args.report_dir.resolve() / "flake_report.json")
        print(flake.format_report(report))
        return EXIT_OK if report.stable else EXIT_FOUND

    if args.command == "coverage":
        measurements = coverage_report.load_measurements(coverage_report.read_json(args.payload))
        groups = coverage_report.group_by_module(measurements)
        print(coverage_report.format_report(groups, threshold=args.threshold))
        print()
        print(
            coverage_report.format_undiscovered(
                coverage_report.undiscovered_files(
                    measurements, repo_root=args.repo.resolve()
                )
            )
        )
        return EXIT_OK

    return EXIT_USAGE  # pragma: no cover - argparse requires a subcommand


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
