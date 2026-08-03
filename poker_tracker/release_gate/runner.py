"""Release-gate orchestration across fixture / full / container modes."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from poker_tracker.release_gate import certification as cert
from poker_tracker.release_gate.environment import (
    collect_environment,
    ffmpeg_version,
    video_duration_seconds,
)
from poker_tracker.release_gate.evaluate import evaluate_answer_key_against_timeline
from poker_tracker.release_gate.models import (
    MODEL_CANDIDATES,
    allowlist_violations,
    partially_pinned_cases,
    resolve_models,
    unpinned_cases,
)
from poker_tracker.release_gate.report import write_report
from poker_tracker.release_gate.resources import collect_resources
from poker_tracker.safety.redaction import redact_text, safe_error_message
from poker_tracker.validation.corpus import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_SETUP_INVALID,
    check_corpus,
)
from poker_tracker.validation.hashing import sha256_bytes, sha256_file
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
# Every model role a reconstruction runs. Pinning is complete only when all of
# them are pinned, so the roles come from the same table the resolver uses.
MODEL_ROLES = frozenset(MODEL_CANDIDATES)
# The stage that performs the run, per mode.
EXECUTION_STAGE_NAMES: dict[str, str] = {
    "fixture": "fixture_eval",
    "full": "full_video",
    "container": "container",
}


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
            # Every report carries a certification block, including this one. A
            # consumer that reads coverage off each report must find "covered
            # nothing" here rather than a missing key it has to interpret.
            "certification": cert.certification(
                mode=mode, executed=[], measured=False
            ),
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
    # Pinning completeness is a REPRODUCIBILITY requirement, and only a mode
    # that actually runs the weights can be unreproducible for lack of them.
    # Fixture mode loads no model, so incomplete pinning is reported there and
    # gates nothing -- otherwise the stage reads "failed" inside a verdict that
    # says the gate passed, which is worse than either answer alone.
    pinning_is_required = mode in {"full", "container"}
    model_skipped: str | None = (
        None if pinning_is_required else "fixture mode loads no model weights"
    )
    if isinstance(document, str):
        model_detail["error"] = document
        models_ok = False
        model_skipped = None
    else:
        scored = _scored_cases(document)
        violations = {
            str(case.get("case_id")): allowlist_violations(case, resolved_models)
            for case in scored
        }
        violations = {k: v for k, v in violations.items() if v}
        unpinned = unpinned_cases(scored)
        # A case that pins one role and leaves another free is not pinned: the
        # unpinned role runs whatever happens to be on disk.
        partial = partially_pinned_cases(scored, resolved_models)
        model_detail["allowlist_violations"] = violations
        model_detail["unpinned_cases"] = unpinned
        model_detail["partially_pinned_cases"] = partial
        model_detail["scored_cases"] = len(scored)
        model_detail["pinning_required"] = pinning_is_required
        # A hash the manifest claims that disagrees with the installed weights is
        # a manifest/environment contradiction in ANY mode, not a reproducibility
        # preference, so it always fails.
        if violations:
            models_ok = False
            model_skipped = None
            for case_id, reasons in violations.items():
                fail(EXIT_SETUP_INVALID, f"models.{case_id}", "; ".join(reasons))
        if pinning_is_required:
            if unpinned:
                models_ok = False
                fail(
                    EXIT_SETUP_INVALID,
                    "models.unpinned",
                    f"cases pin no model weights: {', '.join(sorted(unpinned))}",
                )
            if partial:
                models_ok = False
                fail(
                    EXIT_SETUP_INVALID,
                    "models.partially_pinned",
                    "; ".join(
                        f"{case}: {', '.join(roles)}" for case, roles in partial.items()
                    ),
                )
            if not scored:
                # Nothing was examined, so "no violations" is not a result.
                models_ok = False
                fail(
                    EXIT_SETUP_INVALID,
                    "models.no_cases",
                    "no release-scored cases; nothing was checked",
                )
        elif not scored:
            model_detail["note"] = "no release-scored cases; nothing was checked"
    stages.append(
        _stage(
            "models",
            ok=models_ok,
            skipped=model_skipped,
            elapsed_s=time.perf_counter() - t_models,
            detail=model_detail,
        )
    )

    # --- Stage: mode-specific execution ---
    t1 = time.perf_counter()
    stage_name = EXECUTION_STAGE_NAMES[mode]
    try:
        if mode == "fixture":
            detail = _run_fixture_predictions(manifest_path)
        elif mode == "full":
            detail = _run_full_reconstruction(
                manifest_path, report_dir=report_dir, resolved_models=resolved_models
            )
        else:
            detail = _run_container_acceptance(
                manifest_path,
                repo_root=repo_root,
                report_dir=report_dir,
                image=container_image or os.environ.get(
                    "POKER_RELEASE_GATE_IMAGE", DEFAULT_CONTAINER_IMAGE
                ),
            )
    except Exception as exc:  # noqa: BLE001 - a crash must still produce evidence
        # An unhandled error used to propagate out of the CLI, exiting 1 with no
        # report written at all. That is indistinguishable from a genuine
        # accuracy failure and leaves nothing to diagnose.
        detail = {
            "ok": False,
            "fail_closed": f"gate crashed: {safe_error_message(exc)}",
            "setup_invalid": True,
            "cases": [],
        }

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
    # A report may not say the gate passed while carrying a stage that failed.
    # The exit code and the stage list are computed separately, and any drift
    # between them is a reporting bug that must not resolve in favour of "pass".
    failed_stages = [
        stage["name"] for stage in stages if not stage["ok"] and not stage["skipped"]
    ]
    if failed_stages and exit_code == EXIT_OK:
        fail(
            EXIT_SETUP_INVALID,
            "stages",
            f"stage(s) failed without setting an exit code: {', '.join(failed_stages)}",
        )
    report = {
        "ok": exit_code == EXIT_OK,
        "exit_code": exit_code,
        "mode": mode,
        # Present only when this run was launched by an outer container-mode run,
        # which uses it to prove the report it reads is the one it just caused.
        "run_token": os.environ.get("POKER_RELEASE_GATE_RUN_TOKEN") or None,
        # Derived from what the mode-specific stage reported doing, never from
        # the mode that was requested.
        "certification": cert.certification(
            mode=mode,
            executed=detail.get("executed"),
            measured=bool(aggregate.get("measured")),
        ),
        "manifest_path": _portable_path(manifest_path, repo_root),
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
    # "Zero errors" and "zero measurements" produce identical counts. Every
    # count below is meaningless unless something was actually scored, so the
    # report says which of the two it is instead of leaving a reader to infer
    # quality from a row of zeros.
    measured = hands > 0
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
        "measured": measured,
        "unmeasured_reason": (
            None if measured else "no hand was scored; every count below is vacuous"
        ),
        "cases_scored": int(detail.get("cases_scored") or 0),
        "cases_failed": int(detail.get("cases_failed") or 0),
        "hands_scored": hands,
        "total_errors": total_errors if measured else None,
        "critical_errors": critical if measured else None,
        "noncritical_budget_violations": budget if measured else None,
        "spurious_predicted_hands": spurious if measured else None,
        "excluded_checks": excluded,
        "completion": {
            "measured": measured,
            "true_positive": tp if measured else None,
            "false_positive": fp if measured else None,
            "false_negative": fn if measured else None,
            "true_negative": tn if measured else None,
            "precision": (tp / (tp + fp)) if (measured and (tp + fp)) else None,
            "recall": (tp / (tp + fn)) if (measured and (tp + fn)) else None,
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
        # Paths are recorded relative to the manifest so an archived report does
        # not carry the operator's home directory and account name.
        "prediction_path": _portable_path(prediction_path, manifest_path.parent),
        # Identify exactly which answer key and which timeline produced this
        # score. Without these an edited key yields an indistinguishable report.
        "truth_sha256": sha256_file(truth_path) if truth_path.is_file() else None,
        "prediction_sha256": (
            sha256_file(prediction_path) if prediction_path.is_file() else None
        ),
        **evaluation,
    }


def _portable_path(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` when possible, else its basename.

    Release reports are archived as evidence, so an absolute path leaks the
    operator's username and home directory into a document that outlives the run.
    """
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


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
        # Scoring is claimed only when a case was actually scored; a run that
        # found no readable prediction performed no act at all.
        "executed": [cert.SCORE_TIMELINES] if scored else [],
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
    if ffmpeg_version() is None:
        # Decoding needs it. Discovering that per-case at decode time turns one
        # setup problem into N accuracy-shaped failures.
        return {
            "ok": False,
            "fail_closed": "FFmpeg is not installed; full mode cannot decode video",
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
    # Acts are accumulated from cases that actually got that far. A full-mode run
    # that fails every case before decoding claims no decoding.
    executed: set[str] = set()
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

        duration_s, duration_source = _resolve_case_duration(recording, video_path)
        if duration_s is None:
            case_reports.append(
                {
                    "case_id": case_id,
                    "ok": False,
                    "fail_closed": (
                        "recording duration is unknown: the manifest declares no "
                        f"duration_s and {logical_name} could not be probed"
                    ),
                    "recording": logical_name,
                    "duration_source": duration_source,
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
            duration_s=duration_s,
            detector=resolved_models["region_detector"]["path"],
            classifier=resolved_models["card_classifier"]["path"],
        )
        elapsed = round(time.perf_counter() - started, 3)
        executed.update(cert.normalize_acts(run.get("executed")))
        if not run["ok"]:
            case_reports.append(
                {
                    "case_id": case_id,
                    "ok": False,
                    "fail_closed": run["error"],
                    "recording": logical_name,
                    "duration_s": duration_s,
                    "duration_source": duration_source,
                    "elapsed_s": elapsed,
                }
            )
            failed += 1
            continue
        artifacts.append(str(timeline_path))
        report = _score_case(manifest_path, case, timeline_path)
        report["recording"] = logical_name
        # Which weights this case's timeline came out of, alongside the hashes of
        # the timeline itself, so the score can be tied to what produced it.
        report["models_used"] = {
            role: {
                "path": path,
                "sha256": (resolved_models.get(role) or {}).get("sha256"),
            }
            for role, path in sorted((run.get("weights") or {}).items())
        }
        # The window that was actually reconstructed, and where its bound came
        # from. A score covers the stretch of recording that was sampled, and a
        # reader cannot tell that from the hand counts alone.
        report["duration_s"] = duration_s
        report["duration_source"] = duration_source
        report["elapsed_s"] = elapsed
        if str(report.get("fail_closed") or "").startswith("unreadable_artifacts"):
            failed += 1
            case_reports.append(report)
            continue
        scored += 1
        executed.add(cert.SCORE_TIMELINES)
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
        "executed": sorted(executed),
    }


def _resolve_case_duration(
    recording: Any, video_path: Path
) -> tuple[float | None, str]:
    """How long the recording is, and where that number came from.

    A missing ``duration_s`` is missing information, and the gate used to turn it
    into 86 400 seconds. That is the same failure shape as reporting an unmeasured
    run as zero errors: the number is not a measurement, and everything computed
    from it inherits that. Sampling a day out of a short recording ends either as
    a timeout reported as a reconstruction failure, or as a timeline built from
    repeats of the final frame -- a wrong result that nothing rejects.

    The manifest's own figure wins, because that is the corpus's pinned
    description of the recording. Otherwise the file itself is probed. When
    neither answers, the case is unmeasurable and the caller fails it rather than
    choosing a bound nothing supports.
    """
    declared = recording.get("duration_s") if isinstance(recording, dict) else None
    if isinstance(declared, (int, float)) and not isinstance(declared, bool):
        if declared > 0 and math.isfinite(declared):
            return float(declared), "manifest"
    probed = video_duration_seconds(video_path)
    if probed is not None and probed > 0 and math.isfinite(probed):
        return float(probed), "probed_from_recording"
    return None, "unmeasurable"


def _run_pipeline_for_case(
    *,
    video_path: Path,
    timeline_path: Path,
    duration_s: Any,
    detector: str | None,
    classifier: str | None,
) -> dict[str, Any]:
    repo_root = _repo_root()
    if (
        not isinstance(duration_s, (int, float))
        or isinstance(duration_s, bool)
        or not math.isfinite(duration_s)
        or duration_s <= 0
    ):
        # The caller owns resolving this; a fallback here would reintroduce the
        # unbounded window one layer down, where no case report can describe it.
        return {
            "ok": False,
            "error": "recording duration is unknown; refusing to sample an unbounded window",
        }
    end = float(duration_s)
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
    # What actually goes on the command line, so the record of weights used is
    # taken from the invocation rather than from what the caller resolved.
    weights: dict[str, str] = {}
    if detector:
        command += ["--model1", str(repo_root / detector)]
        weights["region_detector"] = str(detector)
    if classifier:
        command += ["--model2", str(repo_root / classifier)]
        weights["card_classifier"] = str(classifier)
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
    # A timeline out of a pipeline that exited cleanly is the evidence that this
    # recording was decoded and reconstructed. Pinned-weight execution is claimed
    # only when both roles were pinned on the command line: a role left off runs
    # whatever default the script picks, which is not the pinned weights.
    executed = [cert.DECODE_VIDEO, cert.RECONSTRUCT]
    if len(weights) == len(MODEL_ROLES):
        executed.append(cert.LOAD_WEIGHTS)
    return {"ok": True, "error": None, "executed": executed, "weights": weights}


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

    What runs inside is the FIXTURE gate: the image is built without the corpus
    vault and is given the manifest directory alone, so it decodes no video and
    loads no weights. The acts this returns therefore come from the inner
    report, and a container run does not certify a release on its own -- it
    certifies that the image reproduces the host's fixture verdict.
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

    # A fresh directory per run. Reusing one lets a leftover report from an
    # earlier run be read as this run's verdict when the container exits 0
    # without writing -- which publishes hand counts nothing measured.
    container_reports = report_dir / "container"
    if container_reports.exists():
        shutil.rmtree(container_reports)
    container_reports.mkdir(parents=True)

    corpus_dir = manifest_path.resolve().parent
    manifest_name = manifest_path.name
    # Proof of authorship. The token is generated here, handed to the container,
    # and must come back inside the report; a stale or foreign file cannot carry
    # it. Derived from the run's own inputs so scripts stay reproducible.
    token = sha256_bytes(
        f"{image}|{manifest_path.resolve()}|{time.time_ns()}".encode()
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        # The corpus is data, mounted read-only somewhere that is NOT on
        # sys.path. Mounting the repository at the working directory would put
        # the host's poker_tracker ahead of the image's on sys.path, so the gate
        # under test would be the host's code and the image would contribute
        # only an interpreter.
        "-v",
        f"{corpus_dir}:/corpus:ro",
        "-v",
        f"{container_reports}:/reports",
        "-e",
        f"POKER_RELEASE_GATE_RUN_TOKEN={token}",
        "-w",
        "/app",
        image,
        "python",
        "-m",
        "poker_tracker.release_gate",
        "--manifest",
        f"/corpus/{manifest_name}",
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
            # A hung container is an environment problem, not a measurement of
            # product quality.
            "setup_invalid": True,
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
        "report_path": _portable_path(inner_path, report_dir)
        if inner_path.is_file()
        else None,
        "artifacts": [str(inner_path)] if inner_path.is_file() else [],
        "cases": [],
    }
    if inner is None:
        detail.update(
            {
                "ok": False,
                "fail_closed": "container run produced no readable report",
                "setup_invalid": True,
                "stderr_tail": [
                    redact_text(line)
                    for line in (completed.stderr or "").strip().splitlines()[-5:]
                ],
            }
        )
        return detail
    if inner.get("run_token") != token:
        detail.update(
            {
                "ok": False,
                "fail_closed": (
                    "container report does not carry this run's token; "
                    "it was not produced by this run"
                ),
                "setup_invalid": True,
            }
        )
        return detail
    detail["cases"] = _container_case_reports(inner)
    detail["inner_mode"] = inner.get("mode")
    # The acts this run performed are the acts the inner gate performed, plus the
    # fact that the image ran the gate at all. They are read from the inner
    # report -- which the token above proves this run caused -- rather than
    # assumed from the mode this function asked for, because what the outer run
    # requests and what the image executes are two different things.
    inner_certification = inner.get("certification")
    inner_acts = set(
        cert.normalize_acts(
            inner_certification.get("executed")
            if isinstance(inner_certification, dict)
            else None
        )
    )
    if any(int(case.get("hands_scored") or 0) > 0 for case in detail["cases"]):
        # Scored hands in the inner case reports are evidence of scoring in their
        # own right, and an image built from an older gate carries no act record
        # at all. Reading it from the reports keeps that image's coverage honest
        # in both directions.
        inner_acts.add(cert.SCORE_TIMELINES)
    detail["executed"] = [cert.RUN_IN_IMAGE, *sorted(inner_acts)]
    detail["cases_scored"] = int(
        (inner.get("aggregate") or {}).get("cases_scored") or 0
    )
    detail["cases_failed"] = int((inner.get("aggregate") or {}).get("cases_failed") or 0)
    detail["ok"] = bool(inner.get("ok")) and completed.returncode == 0
    # The inner gate's own exit code carries its meaning: 2 is a setup problem
    # inside the image, not the product scoring badly.
    detail["setup_invalid"] = completed.returncode == EXIT_SETUP_INVALID
    detail["fail_closed"] = (
        None if detail["ok"] else f"container gate exited {completed.returncode}"
    )
    return detail


def _container_case_reports(inner: dict[str, Any]) -> list[dict[str, Any]]:
    """The inner run's per-case reports, found by the mode the inner run says it ran.

    Looking only under ``fixture_eval`` would silently return nothing if the gate
    inside the image ever ran a different mode, and no cases reads downstream as
    a run with nothing to score rather than as a report this code failed to read.
    """
    wanted = EXECUTION_STAGE_NAMES.get(str(inner.get("mode")))
    names = {wanted} if wanted else set(EXECUTION_STAGE_NAMES.values())
    for stage in inner.get("stages") or []:
        if not isinstance(stage, dict) or stage.get("name") not in names:
            continue
        detail = stage.get("detail")
        cases = detail.get("cases") if isinstance(detail, dict) else None
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
