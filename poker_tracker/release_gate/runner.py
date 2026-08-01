"""Release-gate orchestration across fixture / full / container modes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from poker_tracker.release_gate.environment import collect_environment
from poker_tracker.release_gate.evaluate import evaluate_answer_key_against_timeline
from poker_tracker.release_gate.models import (
    allowlist_violations,
    resolve_models,
    unpinned_cases,
)
from poker_tracker.release_gate.report import write_report
from poker_tracker.release_gate.resources import collect_resources
from poker_tracker.validation.corpus import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_SETUP_INVALID,
    check_corpus,
)
from poker_tracker.validation.hashing import sha256_file
from poker_tracker.validation.schemas import load_json, require_mapping

Mode = Literal["fixture", "full", "container"]

PIPELINE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "cv_lab"
    / "scripts"
    / "pipeline"
    / "run_two_model_pipeline.py"
)
# A single recording may legitimately take many minutes; beyond this the run is
# treated as hung so one bad case cannot stall an unattended release.
FULL_CASE_TIMEOUT_SECONDS = 3 * 60 * 60
CONTAINER_TIMEOUT_SECONDS = 60 * 60
DEFAULT_CONTAINER_IMAGE = "pokertrainer:release-gate"


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


def _scored_cases(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Manifest cases the gate scores for accuracy.

    Plumbing fixtures exist to exercise schemas and are excluded by the same
    rule the corpus minimums use, so a corpus of fixtures can never look like a
    passing release.
    """
    cases = document.get("cases") or []
    return [
        case
        for case in cases
        if isinstance(case, dict)
        and case.get("runtime_class") != "fixture"
        and case.get("counts_toward_release") is not False
    ]


def _load_manifest(manifest_path: Path) -> dict[str, Any] | str:
    try:
        return require_mapping(load_json(manifest_path), label="manifest")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return f"manifest_unreadable:{exc}"


def run_release_gate(
    *,
    manifest_path: Path,
    mode: Mode,
    report_dir: Path,
    require_recordings: bool = False,
    container_image: str | None = None,
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
        return ReleaseGateResult(
            ok=False, exit_code=EXIT_SETUP_INVALID, report=report, report_path=path
        )

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

    # --- Stage: model identity and allowlists ---
    t_models = time.perf_counter()
    resolved_models = resolve_models(repo_root)
    document = _load_manifest(manifest_path)
    model_detail: dict[str, Any] = {"models": resolved_models}
    models_ok = True
    if isinstance(document, str):
        model_detail["error"] = document
        models_ok = False
    else:
        scored = _scored_cases(document)
        violations = {
            str(case.get("case_id")): allowlist_violations(case, resolved_models)
            for case in scored
        }
        violations = {k: v for k, v in violations.items() if v}
        unpinned = unpinned_cases(scored)
        model_detail["allowlist_violations"] = violations
        model_detail["unpinned_cases"] = unpinned
        if violations:
            models_ok = False
            for case_id, reasons in violations.items():
                fail(EXIT_SETUP_INVALID, f"models.{case_id}", "; ".join(reasons))
        # An unpinned case is unreproducible. Fixture mode reports it; a real
        # video run refuses to certify it.
        if unpinned and mode in {"full", "container"}:
            models_ok = False
            fail(
                EXIT_SETUP_INVALID,
                "models.unpinned",
                f"cases pin no model weights: {', '.join(sorted(unpinned))}",
            )
    stages.append(
        _stage(
            "models",
            ok=models_ok,
            elapsed_s=time.perf_counter() - t_models,
            detail=model_detail,
        )
    )

    # --- Stage: mode-specific execution ---
    t1 = time.perf_counter()
    if mode == "fixture":
        detail = _run_fixture_predictions(manifest_path)
        stage_name = "fixture_eval"
    elif mode == "full":
        detail = _run_full_reconstruction(
            manifest_path, report_dir=report_dir, resolved_models=resolved_models
        )
        stage_name = "full_video"
    else:
        detail = _run_container_acceptance(
            manifest_path,
            repo_root=repo_root,
            report_dir=report_dir,
            image=container_image or os.environ.get(
                "POKER_RELEASE_GATE_IMAGE", DEFAULT_CONTAINER_IMAGE
            ),
        )
        stage_name = "container"

    stage_ok = bool(detail.get("ok")) if corpus.ok else False
    if corpus.ok and not stage_ok:
        reason = str(detail.get("fail_closed") or f"{stage_name}_failed")
        # Setup problems (absent vault, absent models, absent Docker) are exit 2;
        # a real accuracy miss is exit 1.
        code = EXIT_SETUP_INVALID if detail.get("setup_invalid") else EXIT_GATE_FAILED
        fail(code, stage_name, reason)
    stages.append(
        _stage(
            stage_name,
            ok=stage_ok,
            elapsed_s=time.perf_counter() - t1,
            detail=detail,
            skipped=None if corpus.ok else "corpus_setup_invalid",
        )
    )

    aggregate = _aggregate_metrics(detail)
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
        "aggregate": aggregate,
        "models": resolved_models,
        "resources": collect_resources(
            artifact_dirs={
                "report_dir": report_dir,
                "release_artifacts": report_dir / "artifacts",
            }
        ),
        "artifacts": sorted(detail.get("artifacts") or []),
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


def _aggregate_metrics(detail: dict[str, Any]) -> dict[str, Any]:
    """Roll per-case evaluations into the report's headline numbers.

    Completion precision and recall are recomputed from summed confusion counts
    rather than averaged across cases: averaging rates would let a case with one
    hand outvote a case with forty.
    """
    cases = [c for c in (detail.get("cases") or []) if isinstance(c, dict)]
    hands = sum(int(c.get("hands_scored") or 0) for c in cases)
    critical = sum(int(c.get("critical_errors") or 0) for c in cases)
    total_errors = sum(int(c.get("total_errors") or 0) for c in cases)
    budget = sum(int(c.get("noncritical_budget_violations") or 0) for c in cases)
    spurious = sum(len(c.get("spurious_predicted_hands") or []) for c in cases)
    excluded = sorted({f for c in cases for f in (c.get("excluded_facts") or [])})
    tp = fp = fn = tn = 0
    for case in cases:
        completion = case.get("completion")
        if not isinstance(completion, dict):
            continue
        tp += int(completion.get("true_positive") or 0)
        fp += int(completion.get("false_positive") or 0)
        fn += int(completion.get("false_negative") or 0)
        tn += int(completion.get("true_negative") or 0)
    return {
        "cases_scored": int(detail.get("cases_scored") or 0),
        "cases_failed": int(detail.get("cases_failed") or 0),
        "hands_scored": hands,
        "total_errors": total_errors,
        "critical_errors": critical,
        "noncritical_budget_violations": budget,
        "spurious_predicted_hands": spurious,
        "excluded_checks": excluded,
        "completion": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "recall": (tp / (tp + fn)) if (tp + fn) else None,
        },
    }


def _case_prediction_path(manifest_path: Path, case: dict[str, Any]) -> Path | None:
    truth_relpath = case.get("truth_relpath")
    if not isinstance(truth_relpath, str):
        return None
    truth_path = (manifest_path.parent / truth_relpath).resolve()
    candidate = truth_path.with_suffix(truth_path.suffix + ".prediction.json")
    if candidate.is_file():
        return candidate
    sibling = truth_path.with_name(truth_path.stem + ".prediction.json")
    return sibling if sibling.is_file() else None


def _score_case(
    manifest_path: Path,
    case: dict[str, Any],
    prediction_path: Path,
) -> dict[str, Any]:
    truth_relpath = str(case.get("truth_relpath"))
    truth_path = (manifest_path.parent / truth_relpath).resolve()
    try:
        truth = require_mapping(load_json(truth_path), label="truth")
        timeline = require_mapping(load_json(prediction_path), label="timeline")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "case_id": case.get("case_id"),
            "ok": False,
            "fail_closed": f"unreadable_artifacts:{exc}",
        }
    evaluation = evaluate_answer_key_against_timeline(truth, timeline)
    return {
        "case_id": case.get("case_id"),
        "prediction_path": str(prediction_path),
        **evaluation,
    }


def _run_fixture_predictions(manifest_path: Path) -> dict[str, Any]:
    """Score retained per-case prediction timelines without decoding video.

    Predictions live next to the truth file as ``*.prediction.json``. Missing
    predictions for scored cases fail closed; while the corpus itself is
    incomplete the corpus stage already owns the exit code.
    """
    document = _load_manifest(manifest_path)
    if isinstance(document, str):
        return {"ok": False, "fail_closed": document, "setup_invalid": True, "cases": []}

    case_reports: list[dict[str, Any]] = []
    scored = 0
    failed = 0
    for case in _scored_cases(document):
        prediction_path = _case_prediction_path(manifest_path, case)
        if prediction_path is None:
            case_reports.append(
                {
                    "case_id": case.get("case_id"),
                    "ok": False,
                    "fail_closed": "missing_prediction_timeline",
                }
            )
            failed += 1
            continue
        report = _score_case(manifest_path, case, prediction_path)
        if str(report.get("fail_closed") or "").startswith("unreadable_artifacts"):
            failed += 1
            case_reports.append(report)
            continue
        scored += 1
        if not report.get("ok"):
            failed += 1
        case_reports.append(report)

    if scored == 0 and failed == 0:
        return {
            "ok": False,
            "fail_closed": "zero_scored_hands",
            "setup_invalid": True,
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


def _validation_root() -> Path | None:
    raw = os.environ.get("POKER_VALIDATION_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _run_full_reconstruction(
    manifest_path: Path,
    *,
    report_dir: Path,
    resolved_models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Decode each corpus recording with the pinned models and score the result.

    This is the only mode whose verdict covers decoding, sampling, anchoring and
    boundary detection, because it is the only one that runs them. Every
    precondition it cannot satisfy — no vault, no weights, no recording — is a
    setup failure rather than an accuracy result, so a missing model can never
    be mistaken for a model that scored badly.
    """
    document = _load_manifest(manifest_path)
    if isinstance(document, str):
        return {"ok": False, "fail_closed": document, "setup_invalid": True, "cases": []}

    root = _validation_root()
    if root is None or not root.is_dir():
        return {
            "ok": False,
            "fail_closed": "POKER_VALIDATION_ROOT is unset or not a directory",
            "setup_invalid": True,
            "cases": [],
        }
    missing_models = [
        role for role, entry in resolved_models.items() if not entry.get("present")
    ]
    if missing_models:
        return {
            "ok": False,
            "fail_closed": f"pinned weights not installed: {', '.join(missing_models)}",
            "setup_invalid": True,
            "cases": [],
        }
    if not PIPELINE_SCRIPT.is_file():
        return {
            "ok": False,
            "fail_closed": f"reconstruction pipeline not found at {PIPELINE_SCRIPT}",
            "setup_invalid": True,
            "cases": [],
        }

    cases = _scored_cases(document)
    if not cases:
        return {
            "ok": False,
            "fail_closed": "zero_scored_hands",
            "setup_invalid": True,
            "cases": [],
            "note": "manifest contains no release-scored cases",
        }

    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    case_reports: list[dict[str, Any]] = []
    artifacts: list[str] = []
    scored = 0
    failed = 0
    setup_invalid = False
    for case in cases:
        case_id = str(case.get("case_id"))
        recording = case.get("recording")
        logical_name = (
            recording.get("logical_name") if isinstance(recording, dict) else None
        )
        if not isinstance(logical_name, str) or not logical_name:
            case_reports.append(
                {"case_id": case_id, "ok": False, "fail_closed": "case has no recording"}
            )
            failed += 1
            setup_invalid = True
            continue
        video_path = (root / logical_name).resolve()
        if not video_path.is_file():
            case_reports.append(
                {
                    "case_id": case_id,
                    "ok": False,
                    "fail_closed": f"recording not found under the vault: {logical_name}",
                }
            )
            failed += 1
            setup_invalid = True
            continue

        timeline_path = artifact_dir / f"{case_id}.timeline.json"
        started = time.perf_counter()
        run = _run_pipeline_for_case(
            video_path=video_path,
            timeline_path=timeline_path,
            duration_s=(
                recording.get("duration_s") if isinstance(recording, dict) else None
            ),
            detector=resolved_models["region_detector"]["path"],
            classifier=resolved_models["card_classifier"]["path"],
        )
        elapsed = round(time.perf_counter() - started, 3)
        if not run["ok"]:
            case_reports.append(
                {
                    "case_id": case_id,
                    "ok": False,
                    "fail_closed": run["error"],
                    "recording": logical_name,
                    "elapsed_s": elapsed,
                }
            )
            failed += 1
            continue
        artifacts.append(str(timeline_path))
        report = _score_case(manifest_path, case, timeline_path)
        report["recording"] = logical_name
        report["elapsed_s"] = elapsed
        if str(report.get("fail_closed") or "").startswith("unreadable_artifacts"):
            failed += 1
            case_reports.append(report)
            continue
        scored += 1
        if not report.get("ok"):
            failed += 1
        case_reports.append(report)

    return {
        "ok": failed == 0 and scored > 0,
        "fail_closed": None if failed == 0 and scored > 0 else "full_video_failures",
        "setup_invalid": setup_invalid and scored == 0,
        "cases_scored": scored,
        "cases_failed": failed,
        "cases": case_reports,
        "artifacts": artifacts,
    }


def _run_pipeline_for_case(
    *,
    video_path: Path,
    timeline_path: Path,
    duration_s: Any,
    detector: str | None,
    classifier: str | None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    end = float(duration_s) if isinstance(duration_s, (int, float)) else 86_400.0
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--video",
        str(video_path),
        "--start",
        "0",
        "--end",
        str(end),
        "--interval",
        "1",
        "--out",
        str(timeline_path),
    ]
    if detector:
        command += ["--model1", str(repo_root / detector)]
    if classifier:
        command += ["--model2", str(repo_root / classifier)]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=FULL_CASE_TIMEOUT_SECONDS,
            cwd=repo_root,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "reconstruction timed out"}
    except OSError as exc:
        return {"ok": False, "error": f"reconstruction could not start: {exc}"}
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-5:]
        return {
            "ok": False,
            "error": f"reconstruction exited {completed.returncode}: {' | '.join(tail)}",
        }
    if not timeline_path.is_file():
        return {"ok": False, "error": "reconstruction produced no timeline"}
    return {"ok": True, "error": None}


def _run_container_acceptance(
    manifest_path: Path,
    *,
    repo_root: Path,
    report_dir: Path,
    image: str,
) -> dict[str, Any]:
    """Re-run the fixture acceptance path inside the pinned image.

    The point is that the container and the host reach the same verdict, so this
    executes the gate itself in the image and compares exit codes rather than
    reimplementing any checks.
    """
    docker = _docker_available()
    if docker is not None:
        return {"ok": False, "fail_closed": docker, "setup_invalid": True, "cases": []}
    if not _image_present(image):
        return {
            "ok": False,
            "fail_closed": (
                f"container image {image!r} is not built; "
                "build it or set POKER_RELEASE_GATE_IMAGE"
            ),
            "setup_invalid": True,
            "cases": [],
        }

    container_reports = report_dir / "container"
    container_reports.mkdir(parents=True, exist_ok=True)
    try:
        relative_manifest = manifest_path.resolve().relative_to(repo_root)
    except ValueError:
        return {
            "ok": False,
            "fail_closed": "manifest must live inside the repository for container mode",
            "setup_invalid": True,
            "cases": [],
        }
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{repo_root}:/repo:ro",
        "-v",
        f"{container_reports}:/reports",
        "-w",
        "/repo",
        image,
        "python",
        "-m",
        "poker_tracker.release_gate",
        "--manifest",
        f"/repo/{relative_manifest.as_posix()}",
        "--mode",
        "fixture",
        "--report-dir",
        "/reports",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=CONTAINER_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "fail_closed": "container acceptance timed out",
            "cases": [],
        }
    except OSError as exc:
        return {
            "ok": False,
            "fail_closed": f"docker run could not start: {exc}",
            "setup_invalid": True,
            "cases": [],
        }

    inner_path = container_reports / "release_gate_report.json"
    inner: dict[str, Any] | None = None
    if inner_path.is_file():
        try:
            inner = require_mapping(load_json(inner_path), label="container report")
        except (OSError, ValueError, json.JSONDecodeError):
            inner = None
    detail: dict[str, Any] = {
        "image": image,
        "exit_code": completed.returncode,
        "report_path": str(inner_path) if inner_path.is_file() else None,
        "artifacts": [str(inner_path)] if inner_path.is_file() else [],
        "cases": [],
    }
    if inner is None:
        detail.update(
            {
                "ok": False,
                "fail_closed": "container run produced no readable report",
                "stderr_tail": (completed.stderr or "").strip().splitlines()[-5:],
            }
        )
        return detail
    detail["cases"] = _container_case_reports(inner)
    detail["cases_scored"] = int(
        (inner.get("aggregate") or {}).get("cases_scored") or 0
    )
    detail["cases_failed"] = int((inner.get("aggregate") or {}).get("cases_failed") or 0)
    detail["ok"] = bool(inner.get("ok")) and completed.returncode == 0
    detail["fail_closed"] = (
        None if detail["ok"] else f"container gate exited {completed.returncode}"
    )
    return detail


def _container_case_reports(inner: dict[str, Any]) -> list[dict[str, Any]]:
    for stage in inner.get("stages") or []:
        if isinstance(stage, dict) and stage.get("name") == "fixture_eval":
            cases = (stage.get("detail") or {}).get("cases")
            if isinstance(cases, list):
                return [case for case in cases if isinstance(case, dict)]
    return []


def _docker_available() -> str | None:
    """None when Docker can run, otherwise the reason it cannot."""
    try:
        completed = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "docker is not installed or not on PATH"
    if completed.returncode != 0:
        return "docker daemon is not reachable"
    return None


def _image_present(image: str) -> bool:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0
