"""Orchestration for the local performance and resource harness.

The harness runs the probes that were asked for, brackets the whole run with a
resource envelope, and writes one machine-readable report. Three properties are
load-bearing:

* the report always carries every declared metric, so a probe that could not run
  appears as a withheld number with a reason rather than as an absent key;
* every report states the host and the conditions, because a runtime figure
  without its machine is not evidence;
* the output is stable enough that a later run diffs against it, which is what
  turns a regression from something noticed into something detected.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poker_tracker.perf import probes as probe_module
from poker_tracker.perf.measurement import (
    NOT_TAKEN,
    Measurement,
    PerfReport,
    describe_host,
    measured,
    never_measured,
    not_taken,
    unknown_host,
    utc_now_iso,
)
from poker_tracker.perf.probes import (
    ALL_SPECS,
    DISK_DATA_ROOT,
    DISK_FREE,
    DISK_WORKSPACE_GROWTH,
    HARNESS_PEAK_RSS,
    LOG_GROWTH,
    PROBE_GROUPS,
    PROBES,
    RECONSTRUCTION_SECONDS,
    SESSION_LIMIT_SECONDS,
    TEMP_LEAKED_BYTES,
    TEMP_LEAKED_COUNT,
    UPLOAD_SECONDS,
    ProbeContext,
    group_not_taken,
)
from poker_tracker.release_gate.resources import (
    directory_bytes,
    disk_usage,
    peak_memory_bytes,
)

SESSION_CHECK_NAME = "representative_session_completes_within_one_hour"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class HarnessOptions:
    workspace: Path
    repo_root: Path = field(default_factory=repo_root)
    groups: tuple[str, ...] = PROBE_GROUPS
    manifest_path: Path = Path("validation/clubwpt_v1.json")
    video: Path | None = None
    upload_bytes: int = 32 * 1024 * 1024
    timeouts: dict[str, float] = field(default_factory=dict)
    db_path: Path | None = None
    data_root: Path | None = None


def _configured_db_path() -> Path:
    from poker_tracker.persistence.db import DEFAULT_DB_PATH

    return Path(DEFAULT_DB_PATH)


def _configured_data_root() -> Path:
    from poker_tracker.ui.video_storage import DATA_DIR

    return Path(DATA_DIR)


@dataclass
class _Envelope:
    """Before/after snapshots that turn resource use into a signed delta.

    A point-in-time total answers "how big is the data directory", which is not
    the question. "How much did this workload add" is, and only a bracketed pair
    can answer it.
    """

    workspace_bytes: int
    log_bytes: int
    temp_entries: dict[str, int]


def _log_bytes(context: ProbeContext) -> int:
    return directory_bytes(context.log_dir)


def _temp_entries(context: ProbeContext) -> dict[str, int]:
    root = context.tmp_dir
    if not root.is_dir():
        return {}
    entries: dict[str, int] = {}
    for child in root.iterdir():
        try:
            entries[child.name] = (
                directory_bytes(child) if child.is_dir() else child.lstat().st_size
            )
        except OSError:
            entries[child.name] = 0
    return entries


def _snapshot(context: ProbeContext) -> _Envelope:
    return _Envelope(
        workspace_bytes=directory_bytes(context.workspace),
        log_bytes=_log_bytes(context),
        temp_entries=_temp_entries(context),
    )


def _resource_measurements(
    context: ProbeContext, before: _Envelope, after: _Envelope, groups: tuple[str, ...]
) -> list[Measurement]:
    leaked = {
        name: size
        for name, size in after.temp_entries.items()
        if name not in before.temp_entries
    }
    conditions = {
        "workload": sorted(groups) or ["none"],
        "workspace": str(context.workspace),
        "bracketed": "snapshot taken before the first probe and after the last",
    }
    temp_conditions = {
        **conditions,
        "tmpdir": str(context.tmp_dir),
        "scope": "children run with TMPDIR redirected here; the parent's own temp use is excluded",
        "leaked_entries": sorted(leaked),
    }
    results = [
        measured(
            HARNESS_PEAK_RSS,
            value=peak_memory_bytes(),
            probe="resource_envelope",
            conditions={
                **conditions,
                "scope": "harness process and every child it waited on",
            },
        ),
        measured(
            DISK_WORKSPACE_GROWTH,
            value=after.workspace_bytes - before.workspace_bytes,
            probe="resource_envelope",
            conditions={
                **conditions,
                "before_bytes": before.workspace_bytes,
                "after_bytes": after.workspace_bytes,
                "signed": True,
            },
        ),
        measured(
            DISK_DATA_ROOT,
            value=directory_bytes(context.data_root),
            probe="resource_envelope",
            conditions={
                **conditions,
                "path": str(context.data_root),
                "note": "snapshot of the operator's data root, read only",
            },
        ),
        measured(
            TEMP_LEAKED_COUNT,
            value=len(leaked),
            probe="resource_envelope",
            conditions=temp_conditions,
        ),
        measured(
            TEMP_LEAKED_BYTES,
            value=sum(leaked.values()),
            probe="resource_envelope",
            conditions=temp_conditions,
        ),
        measured(
            LOG_GROWTH,
            value=after.log_bytes - before.log_bytes,
            probe="resource_envelope",
            conditions={
                **conditions,
                "path": str(context.log_dir),
                "before_bytes": before.log_bytes,
                "after_bytes": after.log_bytes,
                "signed": True,
            },
        ),
    ]
    usage = disk_usage(context.workspace)
    if usage is None:
        results.append(
            not_taken(
                DISK_FREE,
                reason="the filesystem holding the workspace reported no usage",
                probe="resource_envelope",
                conditions=conditions,
            )
        )
    else:
        results.append(
            measured(
                DISK_FREE,
                value=usage.free_bytes,
                probe="resource_envelope",
                conditions={**conditions, "total_bytes": usage.total_bytes},
            )
        )
    return results


def evaluate_session_check(
    measurements: list[Measurement], host: dict[str, Any]
) -> dict[str, Any]:
    """Phase 13's one-hour requirement, reported honestly when never run.

    The observed quantity is the machine work a session costs -- storing the
    recording and reconstructing it. Human review time is not in it and is not
    claimed to be. When no reconstruction was measured the check is ``never_run``
    and names what is missing; it never resolves to a pass by default.
    """
    by_name = {m.spec.name: m for m in measurements}
    reconstruction = by_name.get(RECONSTRUCTION_SECONDS.name)
    upload = by_name.get(UPLOAD_SECONDS.name)
    components: dict[str, Any] = {
        RECONSTRUCTION_SECONDS.name: (
            reconstruction.value if reconstruction and reconstruction.taken else None
        ),
        UPLOAD_SECONDS.name: upload.value if upload and upload.taken else None,
    }
    base = {
        "name": SESSION_CHECK_NAME,
        "limit_seconds": SESSION_LIMIT_SECONDS,
        "components": components,
        "covers": "storing one recording and reconstructing it end to end",
        "excludes": "operator review time, solver runs and any second recording",
        "reference_machine": host.get("designated_reference_label"),
        "host_label": host.get("label"),
    }
    if reconstruction is None or not reconstruction.taken:
        reason = (
            reconstruction.not_taken_reason
            if reconstruction is not None
            else "the reconstruction metric was not produced"
        )
        return {
            **base,
            "status": "never_run",
            "observed_seconds": None,
            "certifies_release_gate": False,
            "reason": f"no representative session has been measured: {reason}",
        }
    observed = float(reconstruction.value or 0.0)
    if upload is not None and upload.taken:
        observed += float(upload.value or 0.0)
    within = observed <= SESSION_LIMIT_SECONDS
    certifies = bool(host.get("is_designated_reference")) and within
    if host.get("is_designated_reference"):
        note = "measured on the designated reference machine"
    else:
        note = (
            "measured on a host that is not the designated reference machine, so it "
            "is informational and certifies no release gate"
        )
    return {
        **base,
        "status": "within_limit" if within else "exceeded_limit",
        "observed_seconds": round(observed, 3),
        "certifies_release_gate": certifies,
        "reason": note,
    }


def run_harness(options: HarnessOptions) -> PerfReport:
    """Take every requested measurement and return one report."""
    context = ProbeContext(
        repo_root=options.repo_root,
        workspace=options.workspace,
        db_path=options.db_path or _configured_db_path(),
        data_root=options.data_root or _configured_data_root(),
        manifest_path=(
            options.manifest_path
            if options.manifest_path.is_absolute()
            else options.repo_root / options.manifest_path
        ),
        video=options.video,
        upload_bytes=options.upload_bytes,
        timeouts={**probe_module.DEFAULT_TIMEOUTS, **(options.timeouts or {})},
    )
    context.prepare()
    host = describe_host(options.repo_root)
    started_at = utc_now_iso()
    started = time.perf_counter()
    before = _snapshot(context)
    measurements: list[Measurement] = []
    requested = tuple(g for g in PROBE_GROUPS if g in set(options.groups))
    for group in PROBE_GROUPS:
        if group not in requested:
            measurements.extend(
                group_not_taken(
                    group,
                    f"group '{group}' was not requested for this run",
                    probe="harness",
                )
            )
            continue
        try:
            measurements.extend(PROBES[group](context))
        except Exception as exc:  # a broken probe must not lose the whole run
            measurements.extend(
                group_not_taken(
                    group,
                    f"probe raised {type(exc).__name__}: {exc}",
                    probe="harness",
                )
            )
    after = _snapshot(context)
    measurements.extend(_resource_measurements(context, before, after, requested))
    _assert_every_metric_present(measurements)
    return PerfReport(
        host=host,
        measurements=measurements,
        checks=[evaluate_session_check(measurements, host)],
        groups_requested=list(requested),
        started_at=started_at,
        elapsed_s=time.perf_counter() - started,
        notes=[
            "Numbers describe this host under these conditions; compare only against "
            "a baseline from the same fingerprint.",
            f"Probe logs and artifacts: {context.log_dir}",
        ],
    )


def _assert_every_metric_present(measurements: list[Measurement]) -> None:
    """A report that silently dropped a metric would read as nothing to see."""
    seen = {m.spec.name for m in measurements}
    missing = sorted(spec.name for spec in ALL_SPECS if spec.name not in seen)
    if missing:
        raise RuntimeError(f"harness produced no record for: {', '.join(missing)}")
    duplicates = sorted(
        name for name in seen if sum(1 for m in measurements if m.spec.name == name) > 1
    )
    if duplicates:
        raise RuntimeError(f"harness produced duplicate records for: {', '.join(duplicates)}")


def empty_baseline() -> dict[str, Any]:
    """A baseline that has measured nothing, and says so for every metric.

    Shipped in place of invented numbers. A fabricated baseline would make the
    first real run look like a regression or a pass depending on which way the
    fiction leaned, and neither verdict would mean anything.
    """
    return {
        "schema_version": 1,
        "kind": "pokertrainer_perf_report",
        "started_at": None,
        "elapsed_s": None,
        "groups_requested": [],
        "host": unknown_host(),
        "host_fingerprint": {
            "system": None,
            "machine": None,
            "cpu_count": None,
            "python": None,
            "label": None,
        },
        "measurements": [never_measured(spec) for spec in sorted(ALL_SPECS, key=lambda s: s.name)],
        "checks": [
            {
                "name": SESSION_CHECK_NAME,
                "limit_seconds": SESSION_LIMIT_SECONDS,
                "status": "never_run",
                "observed_seconds": None,
                "certifies_release_gate": False,
                "reference_machine": None,
                "host_label": None,
                "components": {},
                "covers": "storing one recording and reconstructing it end to end",
                "excludes": "operator review time, solver runs and any second recording",
                "reason": "this baseline has never been produced by a run",
            }
        ],
        "notes": [
            "Empty baseline. No measurement in this file was ever taken.",
            "Replace it with a real run's report once a reference machine is designated; "
            "see docs/PERFORMANCE.md.",
        ],
        "summary": {
            "measurements_total": len(ALL_SPECS),
            "measurements_taken": 0,
            "measurements_not_taken": len(ALL_SPECS),
        },
    }


def write_json(payload: dict[str, Any], path: Path) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def summarize(report: dict[str, Any]) -> str:
    """One human-readable block; the JSON stays the machine-readable artifact."""
    lines = [
        f"host: {report['host'].get('label') or 'unlabelled'} "
        f"({report['host'].get('machine')} {report['host'].get('system')}, "
        f"{report['host'].get('cpu_count')} cpu)",
        f"taken: {report['summary']['measurements_taken']}/"
        f"{report['summary']['measurements_total']}",
    ]
    for entry in report["measurements"]:
        if entry["status"] == NOT_TAKEN:
            lines.append(f"  NOT TAKEN  {entry['name']}: {entry['not_taken_reason']}")
        else:
            lines.append(f"  {entry['value']:>16}  {entry['name']} ({entry['unit']})")
    for check in report["checks"]:
        lines.append(
            f"check {check['name']}: {check['status']} "
            f"(certifies release gate: {check['certifies_release_gate']})"
        )
    return "\n".join(lines)


def default_workspace(parent: Path | None = None) -> Path:
    import tempfile

    base = parent or Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="pokertrainer-perf-", dir=str(base)))
