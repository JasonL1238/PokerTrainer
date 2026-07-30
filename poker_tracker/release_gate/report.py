"""Deterministic release-report serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(report: dict[str, Any], report_dir: Path) -> Path:
    """Write a deterministic JSON report.

    Wall-clock timings are stored under ``timing`` and are excluded from the
    stable fingerprint body so identical gate outcomes compare equal.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "release_gate_report.json"
    stable = {
        key: value
        for key, value in report.items()
        if key not in {"elapsed_s", "timing", "report_path"}
    }
    # Drop per-stage wall clocks from the durable artifact.
    stages = []
    for stage in stable.get("stages") or []:
        if isinstance(stage, dict):
            stages.append({k: v for k, v in stage.items() if k != "elapsed_s"})
        else:
            stages.append(stage)
    stable["stages"] = stages
    # Keep non-deterministic timing alongside, clearly separated.
    durable = {
        **stable,
        "timing": {
            "elapsed_s": report.get("elapsed_s"),
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
