"""Baseline comparison: what turns a number into a detected regression.

Two rules do the work here.

A missing number on either side is never a verdict. If the baseline never
measured a metric, or this run could not, the entry says which side is empty --
it does not read the gap as an improvement or as a regression.

Numbers from different machines are not compared at all. A reconstruction that
takes twice as long on a two-core cloud host than on a laptop is not a
regression, and calling it one teaches a reader to ignore the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poker_tracker.perf.measurement import MEASURED

# How far a metric may move before it counts as a change. Wall-clock work on a
# shared desktop moves several percent between identical runs; a quarter is wide
# enough that a flagged entry is worth reading.
DEFAULT_TOLERANCE = 0.25

REGRESSED = "regressed"
IMPROVED = "improved"
UNCHANGED = "unchanged"
MISSING_BASELINE = "missing_baseline"
MISSING_CURRENT = "missing_current"
INCOMPARABLE_HOST = "incomparable_host"


@dataclass(frozen=True)
class Comparison:
    name: str
    status: str
    unit: str
    baseline_value: float | None
    current_value: float | None
    delta: float | None
    ratio: float | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "unit": self.unit,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "delta": self.delta,
            "ratio": self.ratio,
            "note": self.note,
        }


def _by_name(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("name")): entry
        for entry in report.get("measurements") or []
        if isinstance(entry, dict) and entry.get("name")
    }


def _value(entry: dict[str, Any] | None) -> float | None:
    if entry is None or entry.get("status") != MEASURED:
        return None
    value = entry.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def hosts_match(baseline: dict[str, Any], current: dict[str, Any]) -> bool:
    return (baseline.get("host_fingerprint") or {}) == (
        current.get("host_fingerprint") or {}
    )


def compare_reports(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Compare a current report against a baseline, metric by metric."""
    same_host = hosts_match(baseline, current)
    baseline_metrics = _by_name(baseline)
    current_metrics = _by_name(current)
    entries: list[Comparison] = []
    for name in sorted(set(baseline_metrics) | set(current_metrics)):
        base_entry = baseline_metrics.get(name)
        current_entry = current_metrics.get(name)
        unit = str((current_entry or base_entry or {}).get("unit") or "")
        base_value = _value(base_entry)
        current_value = _value(current_entry)
        if current_value is None:
            entries.append(
                Comparison(
                    name=name,
                    status=MISSING_CURRENT,
                    unit=unit,
                    baseline_value=base_value,
                    current_value=None,
                    delta=None,
                    ratio=None,
                    note=str(
                        (current_entry or {}).get("not_taken_reason")
                        or "this run recorded no value"
                    ),
                )
            )
            continue
        if base_value is None:
            entries.append(
                Comparison(
                    name=name,
                    status=MISSING_BASELINE,
                    unit=unit,
                    baseline_value=None,
                    current_value=current_value,
                    delta=None,
                    ratio=None,
                    note=str(
                        (base_entry or {}).get("not_taken_reason")
                        or "the baseline recorded no value"
                    ),
                )
            )
            continue
        if not same_host:
            entries.append(
                Comparison(
                    name=name,
                    status=INCOMPARABLE_HOST,
                    unit=unit,
                    baseline_value=base_value,
                    current_value=current_value,
                    delta=round(current_value - base_value, 4),
                    ratio=None,
                    note="baseline and current run came from different machines",
                )
            )
            continue
        lower_is_better = bool(
            (current_entry or base_entry or {}).get("lower_is_better", True)
        )
        entries.append(
            _verdict(
                name=name,
                unit=unit,
                base_value=base_value,
                current_value=current_value,
                lower_is_better=lower_is_better,
                tolerance=tolerance,
            )
        )
    regressions = [e.name for e in entries if e.status == REGRESSED]
    return {
        "kind": "pokertrainer_perf_comparison",
        "tolerance": tolerance,
        "comparable": same_host,
        "baseline_host_fingerprint": baseline.get("host_fingerprint"),
        "current_host_fingerprint": current.get("host_fingerprint"),
        "incomparable_reason": (
            None
            if same_host
            else "host fingerprints differ; measured values are reported but not judged"
        ),
        "entries": [entry.to_dict() for entry in entries],
        "regressions": regressions,
        "compared": sum(
            1
            for e in entries
            if e.status in {REGRESSED, IMPROVED, UNCHANGED}
        ),
    }


def _verdict(
    *,
    name: str,
    unit: str,
    base_value: float,
    current_value: float,
    lower_is_better: bool,
    tolerance: float,
) -> Comparison:
    delta = current_value - base_value
    ratio = (current_value / base_value) if base_value else None
    if base_value == 0:
        # A baseline of zero has no meaningful ratio. Movement away from zero is
        # reported as a change without inventing a percentage for it.
        status = UNCHANGED if delta == 0 else (
            REGRESSED if (delta > 0) == lower_is_better else IMPROVED
        )
        note = "baseline was zero; judged by direction only"
        return Comparison(
            name=name,
            status=status,
            unit=unit,
            baseline_value=base_value,
            current_value=current_value,
            delta=round(delta, 4),
            ratio=None,
            note=note,
        )
    worse = delta > 0 if lower_is_better else delta < 0
    beyond = abs(delta) > abs(base_value) * tolerance
    if not beyond:
        status = UNCHANGED
        note = f"within the {tolerance:.0%} tolerance"
    elif worse:
        status = REGRESSED
        note = f"{'slower' if lower_is_better else 'lower'} than baseline by more than {tolerance:.0%}"
    else:
        status = IMPROVED
        note = f"{'faster' if lower_is_better else 'higher'} than baseline by more than {tolerance:.0%}"
    return Comparison(
        name=name,
        status=status,
        unit=unit,
        baseline_value=base_value,
        current_value=current_value,
        delta=round(delta, 4),
        ratio=round(ratio, 4) if ratio is not None else None,
        note=note,
    )
