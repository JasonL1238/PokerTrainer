"""Release-gate orchestration across fixture / full / container modes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from poker_tracker.release_gate.environment import collect_environment
from poker_tracker.release_gate.evaluate import evaluate_answer_key_against_timeline
from poker_tracker.release_gate.report import write_report
from poker_tracker.validation.corpus import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_SETUP_INVALID,
    check_corpus,
)
from poker_tracker.validation.hashing import sha256_file
from poker_tracker.validation.schemas import load_json, require_mapping

Mode = Literal["fixture", "full", "container"]


@dataclass
class ReleaseGateResult:
    ok: bool
    exit_code: int
    report: dict[str, Any] = field(default_factory=dict)
    report_path: Path | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stage(
    name: str,
    *,
    ok: bool,
    detail: dict[str, Any] | None = None,
    skipped: str | None = None,
    elapsed_s: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "skipped": skipped,
        "elapsed_s": round(elapsed_s, 3),
        "detail": detail or {},
    }


def run_release_gate(
    *,
    manifest_path: Path,
    mode: Mode,
    report_dir: Path,
    require_recordings: bool = False,
) -> ReleaseGateResult:
    started = time.perf_counter()
    repo_root = _repo_root()
    stages: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    exit_code = EXIT_OK

    def fail(code: int, path: str, message: str) -> None:
        nonlocal exit_code
        issues.append({"path": path, "message": message})
        exit_code = max(exit_code, code)

    if mode not in {"fixture", "full", "container"}:
        fail(EXIT_SETUP_INVALID, "mode", f"unsupported mode {mode!r}")
        report = {
            "ok": False,
            "exit_code": EXIT_SETUP_INVALID,
            "mode": mode,
            "issues": issues,
            "stages": stages,
            "environment": collect_environment(repo_root),
        }
        path = write_report(report, report_dir)
        return ReleaseGateResult(ok=False, exit_code=EXIT_SETUP_INVALID, report=report, report_path=path)

    # --- Stage: corpus integrity (always) ---
    t0 = time.perf_counter()
    corpus = check_corpus(
        manifest_path,
        require_release_minimums=True,
        require_recording_files=require_recordings or mode in {"full", "container"},
    )
    stages.append(
        _stage(
            "corpus",
            ok=corpus.ok,
            elapsed_s=time.perf_counter() - t0,
            detail={
                "stats": corpus.stats,
                "issues": [{"path": i.path, "message": i.message} for i in corpus.issues],
                "warnings": [{"path": w.path, "message": w.message} for w in corpus.warnings],
            },
        )
    )
    if not corpus.ok:
        for issue in corpus.issues:
            fail(corpus.exit_code, f"corpus.{issue.path}", issue.message)

    # --- Stage: mode-specific execution ---
    t1 = time.perf_counter()
    if mode == "fixture":
        fixture_detail = _run_fixture_predictions(manifest_path)
        fixture_ok = fixture_detail.get("ok", False)
        if corpus.ok and not fixture_ok:
            # Accuracy/product failure only once corpus setup is valid.
            reason = fixture_detail.get("fail_closed") or "fixture_gate_failed"
            fail(EXIT_GATE_FAILED, "fixture", str(reason))
        stages.append(
            _stage(
                "fixture_eval",
                ok=fixture_ok if corpus.ok else False,
                elapsed_s=time.perf_counter() - t1,
                detail=fixture_detail,
                skipped=None if corpus.ok else "corpus_setup_invalid",
            )
        )
    elif mode == "full":
        stages.append(
            _stage(
                "full_video",
                ok=False,
                elapsed_s=time.perf_counter() - t1,
                detail={
                    "implemented": False,
                    "message": (
                        "full mode requires POKER_VALIDATION_ROOT recordings, "
                        "pinned models, and the CV reconstruction path; "
                        "not yet wired. Failing closed."
                    ),
                },
            )
        )
        if corpus.ok:
            fail(EXIT_SETUP_INVALID, "full", "full mode is not implemented yet")
        else:
            # Keep corpus exit code dominant when setup already failed.
            pass
    else:  # container
        stages.append(
            _stage(
                "container",
                ok=False,
                elapsed_s=time.perf_counter() - t1,
                detail={
                    "implemented": False,
                    "message": (
                        "container mode must re-run the acceptance path inside the "
                        "pinned Docker image; not yet wired. Failing closed."
                    ),
                },
            )
        )
        if corpus.ok:
            fail(EXIT_SETUP_INVALID, "container", "container mode is not implemented yet")

    report = {
        "ok": exit_code == EXIT_OK,
        "exit_code": exit_code,
        "mode": mode,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "issues": issues,
        "stages": stages,
        "environment": collect_environment(repo_root),
        "adversarial_rounds": [],
    }
    path = write_report(report, report_dir)
    report["report_path"] = str(path)
    return ReleaseGateResult(
        ok=exit_code == EXIT_OK,
        exit_code=exit_code,
        report=report,
        report_path=path,
    )


def _run_fixture_predictions(manifest_path: Path) -> dict[str, Any]:
    """Score optional per-case prediction timelines for fixture/runtime cases.

    Predictions live next to the truth file as ``*.prediction.json`` when present.
    Missing predictions for scored cases fail closed once the corpus itself is
    release-complete; while the corpus is incomplete the corpus stage already
    owns the exit code.
    """
    try:
        document = require_mapping(load_json(manifest_path), label="manifest")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "fail_closed": f"manifest_unreadable:{exc}", "cases": []}

    cases = document.get("cases") or []
    case_reports: list[dict[str, Any]] = []
    scored = 0
    failed = 0
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        runtime_class = case.get("runtime_class")
        counts = case.get("counts_toward_release")
        if runtime_class == "fixture" or counts is False:
            # Plumbing fixtures are not accuracy-scored by the release gate.
            continue
        truth_relpath = case.get("truth_relpath")
        if not isinstance(truth_relpath, str):
            continue
        truth_path = (manifest_path.parent / truth_relpath).resolve()
        prediction_path = truth_path.with_suffix(truth_path.suffix + ".prediction.json")
        if not prediction_path.is_file():
            # Alternate: sibling ``<stem>.prediction.json``
            prediction_path = truth_path.with_name(truth_path.stem + ".prediction.json")
        if not prediction_path.is_file():
            case_reports.append(
                {
                    "case_id": case.get("case_id"),
                    "ok": False,
                    "fail_closed": "missing_prediction_timeline",
                }
            )
            failed += 1
            continue
        try:
            truth = require_mapping(load_json(truth_path), label="truth")
            timeline = require_mapping(load_json(prediction_path), label="timeline")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            case_reports.append(
                {
                    "case_id": case.get("case_id"),
                    "ok": False,
                    "fail_closed": f"unreadable_artifacts:{exc}",
                }
            )
            failed += 1
            continue
        evaluation = evaluate_answer_key_against_timeline(truth, timeline)
        scored += 1
        if not evaluation.get("ok"):
            failed += 1
        case_reports.append(
            {
                "case_id": case.get("case_id"),
                "prediction_path": str(prediction_path),
                **evaluation,
            }
        )

    if scored == 0 and failed == 0:
        return {
            "ok": False,
            "fail_closed": "zero_scored_hands",
            "cases": case_reports,
            "note": "no scored cases with prediction timelines were available",
        }
    return {
        "ok": failed == 0 and scored > 0,
        "fail_closed": None if failed == 0 and scored > 0 else "fixture_failures",
        "cases_scored": scored,
        "cases_failed": failed,
        "cases": case_reports,
    }
