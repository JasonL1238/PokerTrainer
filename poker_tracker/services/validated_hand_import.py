"""Add reconstructed timeline hands to a session after validation or draft choice.

CV jobs no longer bulk-import. This module is the only path that lands a
timeline hand into the study database: ``auto`` after every frame is Correct and
the hand is full / not mid-start, or ``draft`` when the operator explicitly asks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
    _boundary_flags,
    hand_to_import_payload,
    timeline_hand_number_from_notes,
)
from poker_tracker.persistence.backup import backup_database
from poker_tracker.persistence.completion import (
    OBSERVED_TERMINAL_EVENTS,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import import_hands_into_session
from poker_tracker.persistence.models import Hand
from poker_tracker.ui.reconstruction_review import (
    hand_frame_progress,
    job_id_from_hand_notes,
    load_timeline_for_job,
    states_for_hand,
    timeline_path_for_job,
)
from poker_tracker.ui.video_storage import CV_TIMELINES_DIR, DATA_DIR, ensure_data_directories

ImportMode = Literal["auto", "draft"]

# Survives notes/tag edits: lives in completion_evidence.extra via dump/parse.
CV_TIMELINE_IDENTITY_KEY = "cv_timeline_identity"


@dataclass(frozen=True)
class ImportGateResult:
    """Why a timeline hand may or may not be auto-imported."""

    ok: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class HandImportResult:
    """Outcome of attempting to land one timeline hand in the destination session."""

    status: Literal["imported", "already_present", "blocked", "skipped"]
    hand_id: int | None = None
    session_id: int | None = None
    reasons: tuple[str, ...] = ()
    message: str = ""


def hand_frames_validated(
    hand: dict[str, Any],
    reviews_by_image: dict[str, Any],
    *,
    countable_images: list[str] | None = None,
) -> bool:
    """True when every navigable frame is Correct (none remaining, none flagged)."""
    progress = hand_frame_progress(
        hand, reviews_by_image, countable_images=countable_images
    )
    return (
        progress["total"] > 0
        and progress["remaining"] == 0
        and progress["flagged"] == 0
    )


def autonomous_import_blockers(
    timeline: dict[str, Any],
    hand: dict[str, Any],
    *,
    timeline_path: Path,
    reviews_by_image: dict[str, Any],
) -> ImportGateResult:
    """Return whether a hand may be auto-added, with human-readable refusal reasons."""
    reasons: list[str] = []
    states = states_for_hand(timeline, hand)
    images = [str(state.get("image") or "") for state in states if state.get("image")]
    if not hand_frames_validated(hand, reviews_by_image, countable_images=images):
        progress = hand_frame_progress(
            hand, reviews_by_image, countable_images=images
        )
        if progress["total"] == 0:
            reasons.append("no retained frames to validate")
        elif progress["flagged"]:
            reasons.append(f"{progress['flagged']} frame(s) flagged incorrect")
        else:
            reasons.append(
                f"{progress['remaining']} frame(s) still need a Correct verdict"
            )

    timeline_hands = list(timeline.get("hands") or [])
    try:
        position = next(
            index
            for index, item in enumerate(timeline_hands)
            if int(item.get("hand_number", -1)) == int(hand.get("hand_number", -2))
        )
    except StopIteration:
        return ImportGateResult(ok=False, reasons=("timeline hand not found",))

    preceded_by_hand = position > 0
    followed_by_hand = position < len(timeline_hands) - 1
    partial_start, partial_end, terminal_event = _boundary_flags(
        hand, preceded_by_hand=preceded_by_hand, followed_by_hand=followed_by_hand
    )
    if partial_start:
        reasons.append("hand started mid-recording (partial_start)")
    if partial_end:
        reasons.append("hand end was truncated (partial_end)")
    if terminal_event not in OBSERVED_TERMINAL_EVENTS:
        reasons.append(f"terminal event not observed ({terminal_event or 'missing'})")

    payload = None
    try:
        payload = hand_to_import_payload(
            hand,
            output_hand_number=max(1, int(hand.get("hand_number") or 1)),
            timeline_path=timeline_path,
            include_incomplete=False,
            preceded_by_hand=preceded_by_hand,
            followed_by_hand=followed_by_hand,
            metadata=timeline.get("metadata") or {},
        )
    except Exception as exc:  # noqa: BLE001 - malformed spine rows must block, not crash
        reasons.append(f"full-hand export failed ({type(exc).__name__}: {exc})")
    if payload is None and not any(
        reason.startswith("full-hand export failed") for reason in reasons
    ):
        reasons.append("incomplete or invalid cards for a full-hand import")

    return ImportGateResult(ok=not reasons, reasons=tuple(reasons))


def hand_passes_autonomous_import_gate(
    timeline: dict[str, Any],
    hand: dict[str, Any],
    *,
    timeline_path: Path,
    reviews_by_image: dict[str, Any],
) -> bool:
    return autonomous_import_blockers(
        timeline,
        hand,
        timeline_path=timeline_path,
        reviews_by_image=reviews_by_image,
    ).ok


def find_existing_imported_hand(
    db: PokerDatabase,
    *,
    session_id: int,
    job_id: int,
    timeline_hand_number: int,
    related_job_ids: set[int] | None = None,
) -> Hand | None:
    """Return the session hand already imported from this job/timeline hand, if any.

    ``related_job_ids`` lets a re-run of reconstruction on the same video recognize
    hands already added from an earlier job for the same timeline hand number.
    """
    accepted_jobs = set(related_job_ids or ())
    accepted_jobs.add(job_id)
    for hand in db.fetch_hands_by_session(session_id):
        evidence = parse_completion_evidence(hand.completion_evidence)
        identity = evidence.extra.get(CV_TIMELINE_IDENTITY_KEY)
        if isinstance(identity, dict):
            try:
                if (
                    int(identity.get("job_id")) in accepted_jobs
                    and int(identity.get("timeline_hand_number")) == timeline_hand_number
                ):
                    return hand
            except (TypeError, ValueError):
                pass
        notes_job = job_id_from_hand_notes(hand.notes)
        if notes_job is None or notes_job not in accepted_jobs:
            continue
        if timeline_hand_number_from_notes(hand.notes) == timeline_hand_number:
            return hand
    return None


def related_cv_job_ids(db: PokerDatabase, video_id: int) -> set[int]:
    return {
        job.id
        for job in db.fetch_jobs_by_video(video_id)
        if job.job_type == "cv_reconstruction" and job.id is not None
    }

def ensure_hand_imported(
    db: PokerDatabase,
    job_id: int,
    timeline_hand_number: int,
    *,
    mode: ImportMode,
    timeline_dir: Path | None = None,
    data_dir: Path | None = None,
    reviews_by_image: dict[str, Any] | None = None,
) -> HandImportResult:
    """Import one timeline hand into the video's destination session.

    ``mode="auto"`` refuses unless frames are all Correct and the hand is full /
    not mid-start. ``mode="draft"`` is operator-initiated and allows incompletes.
    """
    if mode not in {"auto", "draft"}:
        return HandImportResult(
            status="blocked",
            reasons=(f"unknown import mode {mode!r}",),
            message=f"Unsupported import mode: {mode!r}.",
        )
    timeline_dir = timeline_dir if timeline_dir is not None else CV_TIMELINES_DIR
    data_dir = data_dir if data_dir is not None else DATA_DIR
    job = db.fetch_processing_job(job_id)
    if job is None:
        return HandImportResult(
            status="blocked", reasons=("processing job not found",), message="Job not found."
        )
    video = db.fetch_video(job.video_id)
    if video is None or video.session_id is None:
        return HandImportResult(
            status="blocked",
            reasons=("destination session not linked to video",),
            message="Link a destination session before importing hands.",
        )
    session_id = video.session_id
    related_jobs = related_cv_job_ids(db, job.video_id)
    existing = find_existing_imported_hand(
        db,
        session_id=session_id,
        job_id=job_id,
        timeline_hand_number=timeline_hand_number,
        related_job_ids=related_jobs,
    )
    if existing is not None:
        return HandImportResult(
            status="already_present",
            hand_id=existing.id,
            session_id=session_id,
            message=f"Hand already in session #{session_id}.",
        )

    timeline = load_timeline_for_job(job_id, timeline_dir)
    if timeline is None:
        return HandImportResult(
            status="blocked",
            reasons=("timeline missing",),
            message=f"Timeline for job #{job_id} was not found.",
        )
    timeline_path = timeline_path_for_job(job_id, timeline_dir)
    hand = _timeline_hand(timeline, timeline_hand_number)
    if hand is None:
        return HandImportResult(
            status="blocked",
            reasons=("timeline hand not found",),
            message=f"Timeline hand #{timeline_hand_number} was not found.",
        )

    if reviews_by_image is None:
        reviews_by_image = {
            review.source_image: review
            for review in db.fetch_reconstruction_frame_reviews(
                job_id, hand_number=timeline_hand_number
            )
        }

    if mode == "auto":
        gate = autonomous_import_blockers(
            timeline,
            hand,
            timeline_path=timeline_path,
            reviews_by_image=reviews_by_image,
        )
        if not gate.ok:
            return HandImportResult(
                status="blocked",
                session_id=session_id,
                reasons=gate.reasons,
                message="; ".join(gate.reasons),
            )
        include_incomplete = False
    else:
        include_incomplete = True

    session = db.fetch_session(session_id)
    one_hand_payload = _one_hand_session_payload(
        timeline,
        timeline_path=timeline_path,
        timeline_hand_number=timeline_hand_number,
        session_name=session.name if session is not None else "CV",
        include_incomplete=include_incomplete,
        job_id=job_id,
    )
    if one_hand_payload is None or not one_hand_payload.get("hands"):
        return HandImportResult(
            status="blocked",
            session_id=session_id,
            reasons=("export produced no importable hand",),
            message="Could not build an import payload for this hand.",
        )

    paths = ensure_data_directories(data_dir)
    backup_database(Path(db.db_path), Path(paths["backups"]))
    with db.transaction(immediate=True):
        existing = find_existing_imported_hand(
            db,
            session_id=session_id,
            job_id=job_id,
            timeline_hand_number=timeline_hand_number,
            related_job_ids=related_jobs,
        )
        if existing is not None:
            return HandImportResult(
                status="already_present",
                hand_id=existing.id,
                session_id=session_id,
                message=f"Hand already in session #{session_id}.",
            )
        import_hands_into_session(db, one_hand_payload, session_id)
        saved = find_existing_imported_hand(
            db,
            session_id=session_id,
            job_id=job_id,
            timeline_hand_number=timeline_hand_number,
            related_job_ids=related_jobs,
        )
    if saved is None:
        return HandImportResult(
            status="blocked",
            session_id=session_id,
            reasons=("import did not persist the hand",),
            message="Import finished without a matching hand row.",
        )
    label = "draft" if mode == "draft" else "validated hand"
    return HandImportResult(
        status="imported",
        hand_id=saved.id,
        session_id=session_id,
        message=f"Added {label} #{timeline_hand_number} to session #{session_id}.",
    )


def import_all_autonomous_eligible(
    db: PokerDatabase,
    job_id: int,
    *,
    timeline_dir: Path | None = None,
    data_dir: Path | None = None,
) -> list[HandImportResult]:
    """Page-scan recovery: auto-import every eligible timeline hand not yet present."""
    timeline_dir = timeline_dir if timeline_dir is not None else CV_TIMELINES_DIR
    data_dir = data_dir if data_dir is not None else DATA_DIR
    timeline = load_timeline_for_job(job_id, timeline_dir)
    if timeline is None:
        return []
    reviews = db.fetch_reconstruction_frame_reviews(job_id)
    reviews_by_hand: dict[int, dict[str, Any]] = {}
    for review in reviews:
        reviews_by_hand.setdefault(review.hand_number, {})[review.source_image] = review
    results: list[HandImportResult] = []
    for hand in timeline.get("hands") or []:
        try:
            number = int(hand.get("hand_number") or 0)
            if number < 1:
                continue
            result = ensure_hand_imported(
                db,
                job_id,
                number,
                mode="auto",
                timeline_dir=timeline_dir,
                data_dir=data_dir,
                reviews_by_image=reviews_by_hand.get(number, {}),
            )
        except Exception as exc:  # noqa: BLE001 - one bad hand must not abort the scan
            result = HandImportResult(
                status="blocked",
                reasons=(f"import crashed ({type(exc).__name__})",),
                message=str(exc)[:300],
            )
        if result.status in {"imported", "already_present", "blocked"}:
            results.append(result)
    return results


def _timeline_hand(timeline: dict[str, Any], timeline_hand_number: int) -> dict[str, Any] | None:
    for hand in timeline.get("hands") or []:
        if int(hand.get("hand_number") or 0) == timeline_hand_number:
            return hand
    return None


def _stamp_identity(
    hand_payload: dict[str, Any], *, job_id: int, timeline_hand_number: int
) -> None:
    hand = hand_payload.setdefault("hand", {})
    evidence = dict(hand.get("completion_evidence") or {})
    evidence[CV_TIMELINE_IDENTITY_KEY] = {
        "job_id": job_id,
        "timeline_hand_number": timeline_hand_number,
    }
    hand["completion_evidence"] = evidence


def _one_hand_session_payload(
    timeline: dict[str, Any],
    *,
    timeline_path: Path,
    timeline_hand_number: int,
    session_name: str,
    include_incomplete: bool,
    job_id: int,
) -> dict[str, Any] | None:
    full_hands = list(timeline.get("hands") or [])
    try:
        position = next(
            index
            for index, hand in enumerate(full_hands)
            if int(hand.get("hand_number") or 0) == timeline_hand_number
        )
    except StopIteration:
        return None
    hand = full_hands[position]
    preceded = position > 0
    followed = position < len(full_hands) - 1
    from cv_lab.scripts.eval.validate_yolo_card_timeline import validate_timeline
    from poker_tracker.persistence.models import Session
    from datetime import date

    codes: list[str] = []
    if isinstance(timeline.get("states"), list):
        validation_report = validate_timeline(timeline)
        for report in validation_report.get("hands", []):
            if report.get("hand_number") == hand.get("hand_number"):
                codes = sorted(
                    {
                        warning.get("code")
                        for warning in report.get("warnings", [])
                        if warning.get("code")
                    }
                )
                break
    try:
        hand_payload = hand_to_import_payload(
            hand,
            output_hand_number=timeline_hand_number,
            timeline_path=timeline_path,
            include_incomplete=include_incomplete,
            validation_codes=codes,
            preceded_by_hand=preceded,
            followed_by_hand=followed,
            metadata=timeline.get("metadata") or {},
        )
    except Exception:
        return None
    if hand_payload is None:
        return None
    _stamp_identity(
        hand_payload, job_id=job_id, timeline_hand_number=timeline_hand_number
    )
    session = Session(
        name=session_name,
        date_played=date.today(),
        platform="ClubWPT Gold",
        notes="Imported from offline YOLO card timeline after operator validation.",
    )
    return {
        "export_version": 5,
        "session": session.model_dump(mode="json"),
        "hands": [hand_payload],
        "cv_import_summary": {
            "timeline": str(timeline_path),
            "timeline_hands": 1,
            "exported_hands": 1,
            "skipped_hands": 0,
            "skipped": [],
        },
    }
