"""Typed measurement records for the local performance harness.

Two lessons this project already paid for shape this module.

The release gate once reported zero errors for a run that scored nothing, so a
quantity that was never obtained is ``None`` here and carries the reason it is
missing -- never ``0`` -- exactly as
``poker_tracker.release_gate.runner._aggregate_metrics`` withholds its counts.

And a runtime figure without the machine that produced it is not evidence, so a
report states its host once and every individual measurement states the
conditions it was taken under.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poker_tracker.release_gate.environment import collect_environment

SCHEMA_VERSION = 1

# Status of one measurement in a report.
MEASURED = "measured"
NOT_TAKEN = "not_taken"
# Baseline-only: this metric has a slot in the baseline and no run has ever
# filled it. Kept distinct from NOT_TAKEN, which describes a specific run that
# tried and could not.
NEVER_MEASURED = "never_measured"

UNIT_SECONDS = "seconds"
UNIT_BYTES = "bytes"
UNIT_FPS = "frames_per_second"
UNIT_COUNT = "count"
UNIT_MB_PER_SECOND = "megabytes_per_second"

# The label of the machine designated as the supported local reference. No
# machine has been designated yet; docs/PERFORMANCE.md says how to designate
# one, and until then no run can certify a release gate.
REFERENCE_HOST_ENV = "POKERTRAINER_PERF_REFERENCE_HOST"
HOST_LABEL_ENV = "POKERTRAINER_PERF_HOST_LABEL"


@dataclass(frozen=True)
class MeasurementSpec:
    """The declaration of a metric, independent of whether it was obtained.

    Specs exist so a report's key set is fixed: a probe that cannot run still
    emits its metric as ``not_taken``. A metric that silently disappeared from a
    report would read as "nothing to see", which is the failure this harness
    exists to prevent.
    """

    name: str
    unit: str
    group: str
    description: str
    # Direction of improvement, used by baseline comparison. Throughput and free
    # disk get better as they rise; everything else gets better as it falls.
    lower_is_better: bool = True


@dataclass(frozen=True)
class Measurement:
    spec: MeasurementSpec
    status: str
    value: float | int | None
    not_taken_reason: str | None
    conditions: dict[str, Any]

    @property
    def taken(self) -> bool:
        return self.status == MEASURED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "unit": self.spec.unit,
            "group": self.spec.group,
            "description": self.spec.description,
            "lower_is_better": self.spec.lower_is_better,
            "status": self.status,
            "value": self.value,
            "not_taken_reason": self.not_taken_reason,
            "conditions": dict(self.conditions),
        }


def measured(
    spec: MeasurementSpec,
    *,
    value: float | int,
    probe: str,
    conditions: dict[str, Any] | None = None,
) -> Measurement:
    """Record a number that was actually obtained.

    ``value`` may legitimately be ``0`` -- a run that leaked no temporary files
    measured zero of them -- so zero is accepted here and ``None`` is refused.
    A probe that returns nothing must say so through ``not_taken``.
    """
    if value is None:
        raise ValueError(
            f"{spec.name}: a measured value cannot be None; use not_taken(reason=...)"
        )
    if isinstance(value, bool):
        raise ValueError(f"{spec.name}: a measurement is a number, not a flag")
    return Measurement(
        spec=spec,
        status=MEASURED,
        value=value,
        not_taken_reason=None,
        conditions={"probe": probe, **(conditions or {})},
    )


def not_taken(
    spec: MeasurementSpec,
    *,
    reason: str,
    probe: str,
    conditions: dict[str, Any] | None = None,
) -> Measurement:
    """Record that this run did not obtain the number, and why.

    The reason is mandatory. "Not measured" without a cause is indistinguishable
    from a harness bug, and a reader cannot tell whether the gap is fixable.
    """
    if not reason or not reason.strip():
        raise ValueError(f"{spec.name}: a not-taken measurement needs a reason")
    return Measurement(
        spec=spec,
        status=NOT_TAKEN,
        value=None,
        not_taken_reason=reason.strip(),
        conditions={"probe": probe, **(conditions or {})},
    )


def never_measured(spec: MeasurementSpec) -> dict[str, Any]:
    """The baseline entry for a metric no run has ever produced."""
    return {
        "name": spec.name,
        "unit": spec.unit,
        "group": spec.group,
        "description": spec.description,
        "lower_is_better": spec.lower_is_better,
        "status": NEVER_MEASURED,
        "value": None,
        "not_taken_reason": "no run has ever recorded this metric",
        "conditions": {},
    }


def _load_average() -> list[float] | None:
    try:
        return [round(v, 3) for v in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def describe_host(repo_root: Path) -> dict[str, Any]:
    """Everything needed to decide whether two runs are comparable.

    Built on ``release_gate.environment.collect_environment`` so the host block
    of a performance report and of a release report describe the machine the
    same way, including its secret redaction.
    """
    environment = collect_environment(repo_root)
    label = (os.environ.get(HOST_LABEL_ENV) or "").strip() or None
    designated = (os.environ.get(REFERENCE_HOST_ENV) or "").strip() or None
    return {
        "label": label,
        "designated_reference_label": designated,
        # A run only certifies the phase's one-hour requirement when it happened
        # on the machine the project designated. Nothing designates one today.
        "is_designated_reference": bool(label and designated and label == designated),
        "system": environment.get("system"),
        "machine": environment.get("machine"),
        "platform": environment.get("platform"),
        "processor": environment.get("processor"),
        "cpu_count": environment.get("cpu_count"),
        "memory_bytes": environment.get("memory_bytes"),
        "python": environment.get("python"),
        "ffmpeg": environment.get("ffmpeg"),
        "dependencies": environment.get("dependencies"),
        "git": environment.get("git"),
        "env": environment.get("env"),
        "load_average_at_start": _load_average(),
    }


def unknown_host() -> dict[str, Any]:
    """The host block for a baseline that was never produced by a run."""
    return {
        "label": None,
        "designated_reference_label": None,
        "is_designated_reference": False,
        "system": None,
        "machine": None,
        "platform": None,
        "processor": None,
        "cpu_count": None,
        "memory_bytes": None,
        "python": None,
        "ffmpeg": None,
        "dependencies": {},
        "git": {},
        "env": {},
        "load_average_at_start": None,
    }


# The fields that decide whether two runs' numbers may be compared at all.
FINGERPRINT_FIELDS = ("system", "machine", "cpu_count", "python", "label")


def host_fingerprint(host: dict[str, Any]) -> dict[str, Any]:
    return {key: host.get(key) for key in FINGERPRINT_FIELDS}


@dataclass
class PerfReport:
    host: dict[str, Any]
    measurements: list[Measurement] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    groups_requested: list[str] = field(default_factory=list)
    started_at: str = ""
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        # Measurements are emitted in a fixed name order so two reports of the
        # same harness diff line-for-line rather than by accident of probe
        # scheduling.
        ordered = sorted(self.measurements, key=lambda m: m.spec.name)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "pokertrainer_perf_report",
            "started_at": self.started_at,
            "elapsed_s": round(self.elapsed_s, 3),
            "groups_requested": sorted(self.groups_requested),
            "host": self.host,
            "host_fingerprint": host_fingerprint(self.host),
            "measurements": [m.to_dict() for m in ordered],
            "checks": self.checks,
            "notes": self.notes,
            "summary": {
                "measurements_total": len(ordered),
                "measurements_taken": sum(1 for m in ordered if m.taken),
                "measurements_not_taken": sum(1 for m in ordered if not m.taken),
            },
        }


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
