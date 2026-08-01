"""Deterministic release-report serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Keys whose values legitimately differ between two runs of an identical
# verdict. They are quarantined out of the stable body so ``diff`` of two
# reports shows verdict changes only -- the workflow PLAN.md asks for when it
# says to retain reports for comparison. Measured resource use belongs here:
# peak RSS and free disk move every run, and leaving them inline makes every
# diff non-empty, which trains a reader to ignore diffs entirely.
NON_DETERMINISTIC_KEYS = frozenset({"elapsed_s", "timing", "report_path", "resources"})


def write_report(report: dict[str, Any], report_dir: Path) -> Path:
    """Write a deterministic JSON report.

    Measurements that vary run to run are kept under ``measurements``, separate
    from the verdict body, so two identical outcomes compare byte-equal.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "release_gate_report.json"
    stable = {
        key: value
        for key, value in report.items()
        if key not in NON_DETERMINISTIC_KEYS
    }
    # Drop per-stage wall clocks from the durable artifact.
    stages = []
    for stage in stable.get("stages") or []:
        if isinstance(stage, dict):
            stages.append({k: v for k, v in stage.items() if k != "elapsed_s"})
        else:
            stages.append(stage)
    stable["stages"] = stages
    # Keep the varying measurements alongside, clearly separated.
    durable = {
        **stable,
        "measurements": {
            "elapsed_s": report.get("elapsed_s"),
            "resources": report.get("resources"),
            "stages": [
                {"name": s.get("name"), "elapsed_s": s.get("elapsed_s")}
                for s in (report.get("stages") or [])
                if isinstance(s, dict)
            ],
        },
    }
    payload = json.dumps(durable, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8")
    return path
