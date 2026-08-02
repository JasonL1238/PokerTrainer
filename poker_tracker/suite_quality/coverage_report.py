"""Say which core modules the suite actually executes, per module, honestly.

Coverage was not measured at all before this. The useful output is not a
percentage -- it is the name of important code that nothing runs, which is why
this module reports and refuses to gate. A ``fail_under`` number invites two
bad habits: writing tests that execute lines without asserting anything, and
lowering the number when it is inconvenient. Neither finds a bug.

Measurement is deliberately ``coverage run -m pytest`` rather than
``pytest --cov``. ``pytest-cov`` registers itself as a pytest plugin the moment
it is installed, which changes every other run on the machine; ``coverage`` is
a library that only does something when it is invoked. See
``docs/RUNBOOKS.md`` for the command.

The grouping mirrors the product's own module boundaries because that is the
unit a reader can act on: "``poker_tracker/coaching`` is at 41%% and
``solver_grounding.py`` is the untouched half of it" is actionable, and "the
repository is at 78%%" is not.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# The modules PLAN Phase 14 calls core: everything that decides what the
# operator is shown or what is written down. cv_lab/ is research tooling and
# tests/ measure themselves, so both stay out.
CORE_PACKAGES: tuple[str, ...] = (
    "poker_tracker/coaching",
    "poker_tracker/maintenance",
    "poker_tracker/math",
    "poker_tracker/perf",
    "poker_tracker/persistence",
    "poker_tracker/release_gate",
    "poker_tracker/runtime",
    "poker_tracker/safety",
    "poker_tracker/services",
    "poker_tracker/solver",
    "poker_tracker/ui",
    "poker_tracker/validation",
)

# app.py is not a package but is 9500 lines of the product, so it is its own group.
SINGLE_FILE_GROUPS: tuple[str, ...] = ("app.py",)


@dataclass(frozen=True)
class FileCoverage:
    path: str
    statements: int
    covered: int

    @property
    def missing(self) -> int:
        return self.statements - self.covered

    @property
    def percent(self) -> float:
        if self.statements == 0:
            return 100.0
        return 100.0 * self.covered / self.statements


@dataclass(frozen=True)
class GroupCoverage:
    name: str
    files: tuple[FileCoverage, ...]

    @property
    def statements(self) -> int:
        return sum(item.statements for item in self.files)

    @property
    def covered(self) -> int:
        return sum(item.covered for item in self.files)

    @property
    def missing(self) -> int:
        return self.statements - self.covered

    @property
    def percent(self) -> float:
        if self.statements == 0:
            return 100.0
        return 100.0 * self.covered / self.statements


def load_measurements(payload: Mapping[str, object]) -> list[FileCoverage]:
    """Read ``coverage json`` output into one record per measured file."""
    files = payload.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("coverage payload has no 'files' object; was `coverage json` run?")
    measurements: list[FileCoverage] = []
    for path, entry in files.items():
        summary = entry.get("summary") if isinstance(entry, Mapping) else None
        if not isinstance(summary, Mapping):
            continue
        statements = int(summary.get("num_statements", 0))
        covered = int(summary.get("covered_lines", 0))
        measurements.append(
            FileCoverage(path=_normalise(path), statements=statements, covered=covered)
        )
    return sorted(measurements, key=lambda item: item.path)


def _normalise(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def group_by_module(
    measurements: Iterable[FileCoverage],
    *,
    packages: Sequence[str] = CORE_PACKAGES,
    single_files: Sequence[str] = SINGLE_FILE_GROUPS,
) -> list[GroupCoverage]:
    """Bucket files into the product's own module boundaries.

    A file that matches no core package is dropped rather than pooled into an
    "other" bucket: a number covering cv_lab research scripts and the core
    persistence layer at once is the kind of average this module exists to
    avoid reporting.
    """
    buckets: dict[str, list[FileCoverage]] = {name: [] for name in [*packages, *single_files]}
    for item in measurements:
        if item.path in single_files:
            buckets[item.path].append(item)
            continue
        for package in packages:
            if item.path.startswith(package + "/"):
                buckets[package].append(item)
                break
    return [
        GroupCoverage(name=name, files=tuple(sorted(files, key=lambda f: f.path)))
        for name, files in buckets.items()
        if files
    ]


def under_covered(
    groups: Iterable[GroupCoverage], *, threshold: float = 60.0, min_statements: int = 20
) -> list[FileCoverage]:
    """Files below ``threshold``, worst first by unexecuted statement count.

    Ordered by how much code is unexecuted rather than by percentage, because
    a 0%%-covered 400-statement module is the finding and a 55%%-covered
    25-statement helper is noise.
    """
    flagged = [
        item
        for group in groups
        for item in group.files
        if item.statements >= min_statements and item.percent < threshold
    ]
    return sorted(flagged, key=lambda item: (-item.missing, item.path))


def format_report(groups: Sequence[GroupCoverage], *, threshold: float = 60.0) -> str:
    lines = [f"{'module':<34}{'stmts':>8}{'run':>8}{'missing':>9}{'percent':>9}"]
    lines.append("-" * 68)
    for group in sorted(groups, key=lambda item: item.percent):
        lines.append(
            f"{group.name:<34}{group.statements:>8}{group.covered:>8}"
            f"{group.missing:>9}{group.percent:>8.1f}%"
        )
    total_statements = sum(group.statements for group in groups)
    total_covered = sum(group.covered for group in groups)
    overall = 100.0 * total_covered / total_statements if total_statements else 100.0
    lines.append("-" * 68)
    lines.append(
        f"{'core total':<34}{total_statements:>8}{total_covered:>8}"
        f"{total_statements - total_covered:>9}{overall:>8.1f}%"
    )

    flagged = under_covered(groups, threshold=threshold)
    if flagged:
        lines.append(f"\nUnder {threshold:.0f}%, worst first by unexecuted statements:")
        for item in flagged:
            lines.append(
                f"  {item.path:<58}{item.percent:>6.1f}%  {item.missing} of "
                f"{item.statements} never executed"
            )
    else:
        lines.append(f"\nNo core file is under {threshold:.0f}%.")
    return "\n".join(lines)


def undiscovered_files(
    measurements: Iterable[FileCoverage],
    *,
    repo_root: Path,
    packages: Sequence[str] = CORE_PACKAGES,
    single_files: Sequence[str] = SINGLE_FILE_GROUPS,
) -> list[tuple[str, int]]:
    """Core files on disk that the coverage payload does not mention at all.

    Coverage reports a file it never executed only if its walk of the source
    tree reaches it, and that walk descends a directory only when the directory
    holds an ``__init__.py``. Four core packages here -- coaching, math,
    persistence and ui -- have none, so a module in them that no test imports is
    absent from the report rather than sitting in it at 0%. The zero that is
    missing is exactly the zero worth reading, so it is reconstructed here from
    the filesystem instead of being trusted to the payload.

    Line counts, not statement counts: the file was never parsed by coverage,
    and inventing a statement count with a different parser would put a number
    in the report that does not mean what the other numbers mean.
    """
    measured = {item.path for item in measurements}
    findings: list[tuple[str, int]] = []
    candidates: list[Path] = []
    for package in packages:
        directory = repo_root / package
        if directory.is_dir():
            candidates.extend(sorted(directory.rglob("*.py")))
    candidates.extend(repo_root / name for name in single_files)

    for path in candidates:
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(repo_root).as_posix()
        if relative in measured:
            continue
        lines = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        findings.append((relative, lines))
    return sorted(findings, key=lambda item: (-item[1], item[0]))


def format_undiscovered(entries: Sequence[tuple[str, int]]) -> str:
    if not entries:
        return "Every core file appears in the coverage payload."
    lines = [
        "Never imported by the suite, so absent from the coverage payload "
        "(largest first, by source lines):"
    ]
    lines.extend(f"  {path:<58}{count:>6} lines" for path, count in entries)
    return "\n".join(lines)


def read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} is not a coverage JSON object")
    return payload
