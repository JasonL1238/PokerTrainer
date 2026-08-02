"""The redacted diagnostics bundle the Settings page offers for download.

A diagnostics bundle is the one artifact in this product that is built to leave
the machine. Everything an operator would attach to a bug report is here at
once -- resolved configuration, dependency versions, model identity, store
counts and the health audit -- which is exactly the combination that leaks a
provider key if any single field is passed through unexamined.

Two rules hold the whole module together:

* Nothing is serialized until it has been through ``redact_structure``. Redacting
  the finished JSON does not work, because ``json.dumps`` escapes the quotes
  inside every string field and an escaped key no longer matches the assignment
  pattern -- see that function's own docstring. ``serialize_diagnostics`` is the
  only writer here and it applies the scrub itself, so a caller cannot get the
  order wrong.
* No absolute path and no operator content. Paths are reported as
  ``parent/name``: a home directory carries the operator's account name and a
  data root often carries a client's. Hand notes, coaching text, video filenames
  and session names are not collected at all -- the bundle answers "what is this
  install", never "what did they play".

Composed here rather than in ``app.py`` so the payload can be built and asserted
on without a Streamlit runtime.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poker_tracker.maintenance.data_health import HealthReport
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.release_gate.environment import collect_environment, redact_mapping
from poker_tracker.release_gate.models import resolve_models
from poker_tracker.safety.redaction import redact_structure

# Runtime settings this build reads. Reported by NAME and set/unset only: the
# value of POKER_DB_PATH is an absolute path through the operator's home
# directory, and the value of an auth variable is a password. An operator
# debugging a misconfiguration needs to know which variables this build consults
# and which of them are set, which is the whole of what this list gives them.
DOCUMENTED_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("POKER_DB_PATH", "SQLite database file. Put it on a persistent mount."),
    ("POKER_DATA_DIR", "Root for videos, frames, timelines, exports and backups."),
    ("POKER_DB_BUSY_TIMEOUT_MS", "How long a write waits for a competing writer."),
    ("POKERTRAINER_REQUIRE_AUTH", "Force the password gate on even without APP_PASSWORD."),
    ("APP_PASSWORD", "Password for the gate. Never displayed by this product."),
    ("POKER_CV_DEVICE", "Force cpu / mps / cuda for reconstruction."),
    ("POKERTRAINER_CV_TIMEOUT_SECONDS", "Ceiling on one reconstruction job."),
    ("POKERTRAINER_CV_MEMORY_GB", "Address-space ceiling for the CV worker."),
    ("TEXAS_SOLVER_PATH", "Absolute path to the pinned console_solver build."),
    ("TEXAS_SOLVER_RESOURCE_DIR", "Solver resources, when not beside the binary."),
    ("POKERTRAINER_SOLVER_THREADS", "Thread count for a solver run."),
    ("POKER_TRACKER_LLM_PROVIDER", "Coaching provider name."),
    ("POKER_TRACKER_LLM_MODEL", "Coaching model name."),
    ("ANTHROPIC_API_KEY", "Coaching credential. Never displayed by this product."),
    ("OPENAI_API_KEY", "Coaching credential. Never displayed by this product."),
)

# The calibrated floor the reconstruction spine stamps into every timeline's
# layout_profile. Read from the pipeline rather than respelled, because a
# duplicated constant that drifted would let Settings advertise support for a
# geometry the reader is not calibrated for.
_LAYOUT_FALLBACK = "Not resolvable in this build"


def supported_layout_profiles() -> dict[str, Any]:
    """What table geometry this build's readers are calibrated for.

    Imported lazily. The constant lives in ``cv_lab`` with the code that applies
    it, and this module is loaded by the Settings page on every render while the
    pipeline package is only needed when someone asks this question.
    """
    try:
        from cv_lab.scripts.pipeline.build_yolo_hand_timeline import (
            _MIN_CALIBRATED_HEIGHT,
            _MIN_CALIBRATED_WIDTH,
        )
    except Exception:  # a diagnostics readout must never take the page down
        return {
            "minimum_width": None,
            "minimum_height": None,
            "statement": _LAYOUT_FALLBACK,
        }
    return {
        "minimum_width": _MIN_CALIBRATED_WIDTH,
        "minimum_height": _MIN_CALIBRATED_HEIGHT,
        "statement": (
            f"Client windows at least {_MIN_CALIBRATED_WIDTH}x{_MIN_CALIBRATED_HEIGHT}. "
            "A recording below that is stamped '-unsupported' on the hand's layout "
            "profile and the seat anchors and OCR templates are extrapolating."
        ),
    }


def observed_layout_profiles(db: PokerDatabase) -> list[dict[str, Any]]:
    """Every layout profile the stored hands were actually reconstructed at.

    The calibrated floor says what this build claims; this says what it was
    given. An operator whose recordings all land on an unsupported geometry is
    looking at the explanation for every downstream misread, and until now the
    fact was recorded per hand and displayed nowhere.
    """
    counts: dict[tuple[str, bool], int] = {}
    for hand in db.fetch_all_hands():
        evidence = hand.completion_evidence or {}
        if not isinstance(evidence, Mapping):
            continue
        profile = str(evidence.get("layout_profile") or "").strip()
        if not profile:
            continue
        key = (profile, bool(evidence.get("layout_supported")))
        counts[key] = counts.get(key, 0) + 1
    return [
        {"layout_profile": profile, "supported": supported, "hands": count}
        for (profile, supported), count in sorted(counts.items())
    ]


def environment_variable_report() -> list[dict[str, Any]]:
    """Which documented variables are set. Never what they are set to.

    ``configured`` is a boolean on purpose. A truncated or masked value is still
    a value, and the shapes this product's variables carry -- an absolute path,
    a password, an API key -- are all things that must not be rendered at all.
    """
    return [
        {
            "name": name,
            "purpose": purpose,
            "configured": bool((os.environ.get(name) or "").strip()),
        }
        for name, purpose in DOCUMENTED_ENV_VARS
    ]


def short_path(value: str | Path | None) -> str | None:
    """A path as ``parent/name``, which identifies the file without the operator."""
    if value is None:
        return None
    path = Path(str(value))
    parent = path.parent.name
    return f"{parent}/{path.name}" if parent else path.name


def store_counts(db: PokerDatabase) -> dict[str, int]:
    """Row counts only -- how much is here, never what any of it says."""
    hands = db.fetch_all_hands()
    by_source: dict[str, int] = {}
    by_review: dict[str, int] = {}
    by_completion: dict[str, int] = {}
    for hand in hands:
        by_source[hand.source_type] = by_source.get(hand.source_type, 0) + 1
        by_review[hand.review_status] = by_review.get(hand.review_status, 0) + 1
        by_completion[hand.completion_status] = by_completion.get(hand.completion_status, 0) + 1
    counts = {
        "sessions": len(db.fetch_sessions()),
        "hands": len(hands),
        "videos": len(db.fetch_videos()),
        "processing_jobs": len(db.fetch_all_jobs()),
        "open_hand_issues": len(db.fetch_hand_issues(status="open")),
        "hands_with_stale_analysis": len(db.fetch_stale_review_hand_ids()),
        "reconciled_settlements": len(db.fetch_reconciled_settlement_hand_ids()),
    }
    counts.update({f"hands_source_{name}": count for name, count in sorted(by_source.items())})
    counts.update({f"hands_review_{name}": count for name, count in sorted(by_review.items())})
    counts.update(
        {f"hands_completion_{name}": count for name, count in sorted(by_completion.items())}
    )
    return counts


def build_diagnostics_payload(
    db: PokerDatabase,
    *,
    repo_root: Path,
    database_path: str | Path,
    health: HealthReport | None = None,
) -> dict[str, Any]:
    """Assemble the bundle. Still unredacted -- ``serialize_diagnostics`` is the writer.

    ``health`` is optional and reported as ``null`` when absent, rather than being
    run here. The audit opens the database read-only, walks every recorded
    artifact and restores each retained snapshot; a download button must not
    silently pay that, and a bundle that claimed a clean store it never checked
    would be worse than one that says it did not look.
    """
    environment = redact_mapping(collect_environment(repo_root))
    # collect_environment carries a resolved interpreter path and an env subset
    # whose values are absolute paths. Both identify the operator's filesystem,
    # and neither answers a question the bundle exists to answer.
    environment.pop("env", None)
    environment["executable"] = short_path(environment.get("executable"))
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "schema": {
            "expected_version": SCHEMA_VERSION,
            "database": short_path(database_path),
        },
        "environment": environment,
        "environment_variables": environment_variable_report(),
        "models": resolve_models(repo_root),
        "layout_support": {
            "calibrated": supported_layout_profiles(),
            "observed": observed_layout_profiles(db),
        },
        "store": store_counts(db),
        "health": None if health is None else _health_summary(health),
    }


def _health_summary(report: HealthReport) -> dict[str, Any]:
    """The audit as data, with its paths shortened and its details bounded."""
    return {
        "checked_at": report.checked_at,
        "healthy": report.healthy,
        "has_warnings": report.has_warnings,
        "database_path": short_path(report.database_path),
        "data_dir": short_path(report.data_dir),
        "backup_dir": short_path(report.backup_dir),
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "message": check.message,
                "details": list(check.details[:20]),
            }
            for check in report.checks
        ],
    }


def serialize_diagnostics(payload: Mapping[str, Any]) -> bytes:
    """Scrub every string in the structure, THEN encode it.

    The only serializer in this module, so the order cannot be got wrong by a
    caller. ``default=str`` keeps a stray datetime from raising after the scrub
    has already run -- a bundle that fails to encode at that point would be a
    download button that works until the day it matters.
    """
    import json

    return json.dumps(
        redact_structure(dict(payload)), indent=2, sort_keys=True, default=str
    ).encode("utf-8")
