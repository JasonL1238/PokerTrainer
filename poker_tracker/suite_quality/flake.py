"""Run the suite more than once and name what did not agree with itself.

A verifier saw one failure in twelve full-suite runs and could neither
reproduce it nor say which test it was. That is the worst state available: the
suite has proven it is nondeterministic and has told nobody where. A single
green run says nothing about it, and neither does a single red one.

The harness takes N passes. Pass 1 runs in collection order, so the ordinary
run is one of the samples; the rest run under ``random_order`` with distinct
seeds, because inter-test state leakage shows up as an order dependency far
more often than as a coin flip. Each pass writes a JUnit XML report -- pytest's
own, so no third-party plugin has to be installed to record outcomes -- and the
report is the per-test outcome record the aggregate is built from.

What comes back names three separate diseases, which is the point:

``unstable``
    the same test id both passed and failed across passes. This is the flake.

``consistently_failing``
    failed in every pass it ran in. Not a flake -- a broken test, and reporting
    it as flakiness would be how a real failure gets waved through.

``order_dependent``
    ran in some passes and not others, or errored in setup only under a
    shuffle. Collection or fixture state changed with order.

Reruns are deliberately not automatic. Rerunning until green is how a flake
becomes permanent; this reports and stops.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

PASSED = "passed"
FAILED = "failed"
ERROR = "error"
SKIPPED = "skipped"

# Outcomes that mean the test did not do its job in that pass.
BAD_OUTCOMES = frozenset({FAILED, ERROR})

# The top-level shim, not this package's module. pytest loads a `-p` plugin
# before any conftest, so naming anything under `poker_tracker` here imports the
# application before tests/conftest.py can redirect the operator's database and
# data directory out of the repository.
PLUGIN = "sq_random_order"


@dataclass(frozen=True)
class PassResult:
    """One whole-suite run."""

    label: str
    seed: int | None
    exit_code: int
    duration_s: float
    outcomes: Mapping[str, str]

    def outcome_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.outcomes.values():
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts


@dataclass
class FlakeReport:
    passes: list[PassResult] = field(default_factory=list)
    unstable: list[dict[str, object]] = field(default_factory=list)
    consistently_failing: list[dict[str, object]] = field(default_factory=list)
    order_dependent: list[dict[str, object]] = field(default_factory=list)

    @property
    def stable(self) -> bool:
        return not (self.unstable or self.order_dependent)

    def to_json(self) -> dict[str, object]:
        return {
            "passes": [
                {
                    "label": item.label,
                    "seed": item.seed,
                    "exit_code": item.exit_code,
                    "duration_s": round(item.duration_s, 2),
                    "counts": item.outcome_counts(),
                }
                for item in self.passes
            ],
            "unstable": self.unstable,
            "consistently_failing": self.consistently_failing,
            "order_dependent": self.order_dependent,
            "stable": self.stable,
        }


def parse_junit(path: Path) -> dict[str, str]:
    """Per-test outcome from a pytest ``--junit-xml`` report.

    Keyed by ``file::test`` rather than the raw classname, so the same test is
    the same key in every pass regardless of the order it ran in.
    """
    root = ET.parse(path).getroot()
    outcomes: dict[str, str] = {}
    for case in root.iter("testcase"):
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        node_id = f"{classname.replace('.', '/')}.py::{name}" if classname else name
        outcome = PASSED
        for child in case:
            if child.tag == "failure":
                outcome = FAILED
            elif child.tag == "error":
                outcome = ERROR
            elif child.tag == "skipped":
                outcome = SKIPPED
        # A test id can appear twice when a fixture errors after the call
        # phase; the worse outcome is the honest one.
        previous = outcomes.get(node_id)
        if previous is None or (previous == PASSED and outcome != PASSED):
            outcomes[node_id] = outcome
    return outcomes


def build_report(passes: Sequence[PassResult]) -> FlakeReport:
    """Compare the passes test by test."""
    report = FlakeReport(passes=list(passes))
    if not passes:
        return report

    every_id: set[str] = set()
    for item in passes:
        every_id.update(item.outcomes)

    for node_id in sorted(every_id):
        seen = {item.label: item.outcomes.get(node_id) for item in passes}
        present = {label: value for label, value in seen.items() if value is not None}
        missing = sorted(label for label, value in seen.items() if value is None)
        distinct = set(present.values())

        if missing:
            report.order_dependent.append(
                {
                    "test": node_id,
                    "absent_from": missing,
                    "outcomes": present,
                    "detail": "collected in some passes and not others",
                }
            )
            continue

        if distinct & BAD_OUTCOMES and distinct - BAD_OUTCOMES:
            report.unstable.append({"test": node_id, "outcomes": present})
        elif distinct <= BAD_OUTCOMES and distinct:
            report.consistently_failing.append({"test": node_id, "outcomes": present})
        elif distinct == {PASSED, SKIPPED}:
            # Not a flake in the "wrong answer" sense, but a test that runs only
            # sometimes is a coverage hole that moves, so it is still named.
            report.order_dependent.append(
                {
                    "test": node_id,
                    "absent_from": [],
                    "outcomes": present,
                    "detail": "skipped in some passes and executed in others",
                }
            )
    return report


def run_pass(
    *,
    label: str,
    seed: int | None,
    repo: Path,
    report_dir: Path,
    pytest_args: Iterable[str] = (),
    plugin: str = PLUGIN,
) -> PassResult:
    """Execute one whole-suite pass in a subprocess and read its JUnit report.

    ``plugin`` is settable because the tree being measured is often not the tree
    this module lives in: hunting a flake in a pristine checkout means running
    pytest with that checkout as the working directory, where an import path
    into this package does not resolve.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    junit = report_dir / f"junit-{label}.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junit-xml={junit}",
        *pytest_args,
    ]
    if seed is not None:
        command += ["-p", plugin, "--sq-seed", str(seed)]

    started = time.monotonic()
    completed = subprocess.run(command, cwd=repo, capture_output=True, text=True)
    duration = time.monotonic() - started
    (report_dir / f"stdout-{label}.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    outcomes = parse_junit(junit) if junit.exists() else {}
    return PassResult(
        label=label,
        seed=seed,
        exit_code=completed.returncode,
        duration_s=duration,
        outcomes=outcomes,
    )


def hunt(
    *,
    repo: Path,
    report_dir: Path,
    passes: int,
    seeds: Sequence[int] | None = None,
    pytest_args: Iterable[str] = (),
    plugin: str = PLUGIN,
) -> FlakeReport:
    """Pass 1 in collection order, the rest shuffled, then compare."""
    chosen = list(seeds) if seeds else [1000 + index for index in range(passes - 1)]
    results = [
        run_pass(
            label="ordered",
            seed=None,
            repo=repo,
            report_dir=report_dir,
            pytest_args=pytest_args,
            plugin=plugin,
        )
    ]
    for index in range(passes - 1):
        seed = chosen[index % len(chosen)] if chosen else index
        results.append(
            run_pass(
                label=f"seed{seed}",
                seed=seed,
                repo=repo,
                report_dir=report_dir,
                pytest_args=pytest_args,
                plugin=plugin,
            )
        )
    return build_report(results)


def format_report(report: FlakeReport) -> str:
    lines = [f"{len(report.passes)} passes"]
    for item in report.passes:
        order = "collection order" if item.seed is None else f"seed {item.seed}"
        counts = item.outcome_counts()
        lines.append(
            f"  {item.label:<10} {order:<18} exit={item.exit_code} "
            f"{item.duration_s:6.1f}s  {counts}"
        )
    if report.unstable:
        lines.append("\nUNSTABLE -- passed in one pass and failed in another:")
        for entry in report.unstable:
            lines.append(f"  {entry['test']}")
            lines.append(f"      {entry['outcomes']}")
    if report.order_dependent:
        lines.append("\nORDER DEPENDENT -- did not run the same way in every pass:")
        for entry in report.order_dependent:
            lines.append(f"  {entry['test']}: {entry['detail']}")
            lines.append(f"      {entry['outcomes']}")
    if report.consistently_failing:
        lines.append("\nCONSISTENTLY FAILING -- broken, not flaky:")
        for entry in report.consistently_failing:
            lines.append(f"  {entry['test']}")
    if report.stable and not report.consistently_failing:
        lines.append("\nEvery test produced the same outcome in every pass.")
    return "\n".join(lines)


def write_report(report: FlakeReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json(), indent=2, sort_keys=True), encoding="utf-8")
