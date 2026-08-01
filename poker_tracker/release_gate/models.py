"""Model weight discovery and allowlist enforcement for release runs.

A release verdict is only reproducible if the weights that produced it are
identified. A case may pin the exact weights it was scored against through its
``model_allowlist``; running different weights against that case invalidates the
verdict rather than quietly producing a new one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from poker_tracker.validation.hashing import sha256_file

# Logical model role -> repository-relative candidate paths, best first.
MODEL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "region_detector": (
        "cv_lab/models/region_spine_v1.pt",
        "cv_lab/runs/yolo_cards/region_spine_v1/weights/best.pt",
    ),
    "card_classifier": (
        "cv_lab/models/card_cls_v1.pt",
        "cv_lab/runs/yolo_cards/card_cls_v1/weights/best.pt",
    ),
}


def resolve_models(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Hash whichever pinned weights are present, and say so when absent."""
    resolved: dict[str, dict[str, Any]] = {}
    for role, candidates in MODEL_CANDIDATES.items():
        entry: dict[str, Any] = {"path": None, "sha256": None, "present": False}
        for relative in candidates:
            path = repo_root / relative
            if path.is_file():
                entry = {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "present": True,
                    "bytes": path.stat().st_size,
                }
                break
        resolved[role] = entry
    return resolved


def allowlist_violations(
    case: dict[str, Any],
    resolved: dict[str, dict[str, Any]],
) -> list[str]:
    """Roles whose running weights disagree with what the case pinned.

    An empty or missing allowlist pins nothing and is reported by the caller as
    an unpinned case, not treated here as agreement.
    """
    allowlist = case.get("model_allowlist")
    if not isinstance(allowlist, dict):
        return []
    violations: list[str] = []
    for role, expected in allowlist.items():
        entry = resolved.get(role)
        if entry is None:
            violations.append(f"{role}: case pins a model this build does not define")
            continue
        if not entry.get("present"):
            violations.append(f"{role}: pinned weights are not installed")
            continue
        expected_hash = expected.get("sha256") if isinstance(expected, dict) else expected
        if not isinstance(expected_hash, str) or not expected_hash.strip():
            violations.append(f"{role}: allowlist entry carries no sha256")
            continue
        if expected_hash != entry.get("sha256"):
            violations.append(
                f"{role}: running {entry.get('sha256')} but case pins {expected_hash}"
            )
    return violations


def partially_pinned_cases(
    cases: list[Any],
    resolved: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Scored cases that pin some model roles but not all of them.

    ``allowlist_violations`` enumerates what the case *claims*, so a case that
    pins the detector and says nothing about the classifier passes it cleanly --
    while the classifier runs whatever happens to be installed. Enforcement has
    to enumerate the obligation, which is every role the run actually uses.
    """
    partial: dict[str, list[str]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        if case.get("counts_toward_release") is False:
            continue
        allowlist = case.get("model_allowlist")
        if not isinstance(allowlist, dict) or not allowlist:
            # Fully unpinned; ``unpinned_cases`` owns that report.
            continue
        missing = sorted(role for role in resolved if role not in allowlist)
        if missing:
            partial[str(case.get("case_id"))] = missing
    return partial


def unpinned_cases(cases: list[Any]) -> list[str]:
    """Scored cases that pin no model at all.

    Reported so an empty ``model_allowlist`` reads as "nobody pinned this",
    which is what it means, rather than as a passing check.
    """
    unpinned: list[str] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        if case.get("counts_toward_release") is False:
            continue
        allowlist = case.get("model_allowlist")
        if not isinstance(allowlist, dict) or not allowlist:
            unpinned.append(str(case.get("case_id")))
    return unpinned
