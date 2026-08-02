"""CLI: ``python -m poker_tracker.perf``.

Three subcommands, one artifact each.

    run            take the measurements and write a report
    new-baseline   write an empty baseline that has measured nothing
    compare        judge a report against a baseline

Exit codes are meant to be usable from a gate: 0 success, 2 a regression against
the baseline, 4 the representative-session requirement not satisfied. A run in
which measurements simply could not be taken exits 0 -- not measuring is not a
product failure, and the report says which numbers are missing and why.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from poker_tracker.perf.compare import DEFAULT_TOLERANCE, compare_reports
from poker_tracker.perf.harness import (
    HarnessOptions,
    default_workspace,
    empty_baseline,
    repo_root,
    run_harness,
    summarize,
    write_json,
)
from poker_tracker.perf.probes import PROBE_GROUPS

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_REGRESSION = 2
EXIT_SESSION_CHECK = 4

DEFAULT_REPORT = Path("data/perf_reports/perf_report.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m poker_tracker.perf",
        description="Local performance and resource measurement harness (Phase 13).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Take measurements and write a report")
    run.add_argument(
        "--groups",
        default=",".join(PROBE_GROUPS),
        help=f"comma-separated subset of: {', '.join(PROBE_GROUPS)} (default: all)",
    )
    run.add_argument("--skip", default="", help="comma-separated groups to leave out")
    run.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    run.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="scratch directory the harness may write to (default: a new temp dir)",
    )
    run.add_argument(
        "--clean-workspace",
        action="store_true",
        help="remove the workspace afterwards, discarding the probe logs with it",
    )
    run.add_argument("--manifest", type=Path, default=Path("validation/clubwpt_v1.json"))
    run.add_argument(
        "--video",
        type=Path,
        default=None,
        help="recording to use for the representative-session measurement",
    )
    run.add_argument("--upload-bytes", type=int, default=32 * 1024 * 1024)
    run.add_argument("--baseline", type=Path, default=None)
    run.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    run.add_argument("--fail-on-regression", action="store_true")
    run.add_argument(
        "--require-session-check",
        action="store_true",
        help="exit non-zero unless a representative session was measured within the limit",
    )
    run.add_argument("--quiet", action="store_true")

    baseline = sub.add_parser(
        "new-baseline", help="Write an empty baseline: every metric never measured"
    )
    baseline.add_argument("--out", type=Path, required=True)

    compare = sub.add_parser("compare", help="Judge a report against a baseline")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--report", type=Path, required=True)
    compare.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    compare.add_argument("--out", type=Path, default=None)
    compare.add_argument("--fail-on-regression", action="store_true")
    return parser


def _selected_groups(requested: str, skipped: str) -> tuple[str, ...]:
    wanted = {g.strip() for g in requested.split(",") if g.strip()}
    unwanted = {g.strip() for g in skipped.split(",") if g.strip()}
    unknown = sorted((wanted | unwanted) - set(PROBE_GROUPS))
    if unknown:
        raise ValueError(
            f"unknown probe group(s): {', '.join(unknown)}; "
            f"known groups are {', '.join(PROBE_GROUPS)}"
        )
    return tuple(g for g in PROBE_GROUPS if g in wanted and g not in unwanted)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = repo_root()

    if args.command == "new-baseline":
        write_json(empty_baseline(), args.out)
        print(f"empty baseline -> {args.out}")
        return EXIT_OK

    if args.command == "compare":
        comparison = compare_reports(
            _load(args.baseline), _load(args.report), tolerance=args.tolerance
        )
        if args.out:
            write_json(comparison, args.out)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        if args.fail_on_regression and comparison["regressions"]:
            return EXIT_REGRESSION
        return EXIT_OK

    try:
        groups = _selected_groups(args.groups, args.skip)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    workspace = args.workspace or default_workspace()
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        report = run_harness(
            HarnessOptions(
                workspace=workspace,
                repo_root=root,
                groups=groups,
                manifest_path=args.manifest,
                video=args.video,
                upload_bytes=args.upload_bytes,
            )
        )
        payload = report.to_dict()
        out = args.out if args.out.is_absolute() else root / args.out
        write_json(payload, out)
        if not args.quiet:
            print(summarize(payload))
            print(f"report -> {out}")
            print(f"workspace -> {workspace}")
        exit_code = EXIT_OK
        if args.baseline is not None:
            comparison = compare_reports(
                _load(args.baseline), payload, tolerance=args.tolerance
            )
            write_json(comparison, out.with_name(out.stem + "_comparison.json"))
            if not args.quiet:
                print(
                    f"comparison: {comparison['compared']} judged, "
                    f"{len(comparison['regressions'])} regressed"
                    + ("" if comparison["comparable"] else " (different host: not judged)")
                )
            if args.fail_on_regression and comparison["regressions"]:
                exit_code = EXIT_REGRESSION
        if args.require_session_check:
            check = payload["checks"][0]
            if not check["certifies_release_gate"]:
                print(
                    f"representative-session check does not certify: {check['status']} "
                    f"({check['reason']})",
                    file=sys.stderr,
                )
                exit_code = EXIT_SESSION_CHECK
        return exit_code
    finally:
        if args.clean_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
