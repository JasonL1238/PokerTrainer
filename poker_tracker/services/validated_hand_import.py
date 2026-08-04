"""Add reconstructed timeline hands to a session after validation or draft choice.

CV jobs no longer bulk-import. This module is the only path that lands a
timeline hand into the study database: ``auto`` after every frame is Correct and
the hand is full / not mid-start, or ``draft`` when the operator explicitly asks.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
    _boundary_flags,
    hand_to_import_payload,
    timeline_hand_number_from_notes,
)
from poker_tracker.maintenance.data_health import verify_snapshot
from poker_tracker.persistence.backup import backup_database, find_snapshots
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

# The only job status whose timeline is a finished reading of the recording.
# The gate used to live entirely in app.py, which filters the review surface to
# completed jobs; this module is reachable without it -- the recovery scans and
# the draft path call in directly -- and a cancelled worker's artifacts can
# survive on disk, because the process is signalled rather than asked to stop.
COMPLETED_JOB_STATUS = "completed"


@dataclass(frozen=True)
class ImportGateResult:
    """Why a timeline hand may or may not be auto-imported."""

    ok: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class HandImportResult:
    """Outcome of attempting to land one timeline hand in the destination session.

    ``rollback_point`` is additive and names the snapshot file this import was
    allowed to proceed against, AFTER that file was proved to survive an isolated
    restore. It exists because the caller previously could not tell a verified
    rollback point from an unverified one: ``ensure_preimport_snapshot`` returned
    ``None`` both when it had just verified a snapshot and when it had merely
    seen one in the directory listing, and nothing on this result carried the
    difference. Every existing consumer reads ``status``/``hand_id``/``message``
    unchanged.
    """

    status: Literal["imported", "already_present", "blocked", "skipped"]
    hand_id: int | None = None
    session_id: int | None = None
    reasons: tuple[str, ...] = ()
    message: str = ""
    rollback_point: str | None = None


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


def _first_import_of(
    candidates: Iterable[Hand],
    *,
    job_id: int,
    timeline_hand_number: int,
    related_job_ids: set[int] | None,
) -> Hand | None:
    """The first candidate row that is this timeline hand's import, or None.

    The identity test lives here and nowhere else, because TWO different questions
    are asked with it -- "is this hand in the destination session" and "does it
    exist in any session at all" -- and a rule copied into both drifts. Evidence
    first, notes second: the notes fallback carries legacy CV drafts whose identity
    was never stamped into completion evidence.
    """
    accepted_jobs = set(related_job_ids or ())
    accepted_jobs.add(job_id)
    for hand in candidates:
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


def find_existing_imported_hand(
    db: PokerDatabase,
    *,
    session_id: int,
    job_id: int,
    timeline_hand_number: int,
    related_job_ids: set[int] | None = None,
) -> Hand | None:
    """Return the hand already imported into THIS session from this timeline hand.

    ``related_job_ids`` lets a re-run of reconstruction on the same video recognize
    hands already added from an earlier job for the same timeline hand number.

    Scoped deliberately. The UI asks this to decide what to draw for one session --
    an "in session" badge, an onboarding step, the row to render beside the frames --
    and every one of those answers is about the session on screen. The de-duplication
    question is a different one; see ``find_imported_hand_in_any_session``.
    """
    return _first_import_of(
        db.fetch_hands_by_session(session_id),
        job_id=job_id,
        timeline_hand_number=timeline_hand_number,
        related_job_ids=related_job_ids,
    )


def find_imported_hand_in_any_session(
    db: PokerDatabase,
    *,
    job_id: int,
    timeline_hand_number: int,
    related_job_ids: set[int] | None = None,
) -> Hand | None:
    """The same import, looked for across EVERY session. The de-duplication question.

    A hand is a record of one real hand that was played once, so "already imported"
    cannot be a per-session fact -- and asking it per session is how one real hand
    came to exist twice. The import destination is read off ``video.session_id`` at
    import time (see ``ensure_hand_imported``), so attaching a recording to a second
    session moved the destination while leaving the earlier hands where they were.
    A destination-scoped probe then reported every one of them absent, and the
    evidence-review page's own recovery scan -- which runs on render, needs no
    button, and passes the gate because frame verdicts are keyed by JOB and survive
    the move -- re-imported the entire timeline into the new session. Both copies
    counted toward session results and analytics.

    Unindexed by necessity: a hand records its originating job inside its completion
    evidence and no column indexes it, which is the same constraint
    ``hands_reconstructed_from_video`` documents. That makes this a full scan, so it
    belongs on an import action and must not be called from anything that renders.
    """
    return _first_import_of(
        db.fetch_all_hands(),
        job_id=job_id,
        timeline_hand_number=timeline_hand_number,
        related_job_ids=related_job_ids,
    )


def related_cv_job_ids(db: PokerDatabase, video_id: int) -> set[int]:
    return {
        job.id
        for job in db.fetch_jobs_by_video(video_id)
        if job.job_type == "cv_reconstruction" and job.id is not None
    }

def ensure_draft_for_review(
    db: PokerDatabase,
    job_id: int,
    timeline_hand_number: int,
    *,
    timeline_dir: Path | None = None,
    data_dir: Path | None = None,
    reviews_by_image: dict[str, Any] | None = None,
) -> HandImportResult:
    """Land a draft session hand so Import can edit while reviewing frames.

    Treats ``imported`` and ``already_present`` as success. Does not promote the
    hand to study-ready; that happens only after validation finishes cleanly.
    """
    return ensure_hand_imported(
        db,
        job_id,
        timeline_hand_number,
        mode="draft",
        timeline_dir=timeline_dir,
        data_dir=data_dir,
        reviews_by_image=reviews_by_image,
    )


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
    if job.status != COMPLETED_JOB_STATUS:
        return HandImportResult(
            status="blocked",
            reasons=(f"reconstruction job is {job.status}, not completed",),
            message=(
                f"Reconstruction job #{job_id} is {job.status}. Only a completed "
                "run's timeline may be imported: a run that was cancelled, "
                "failed or is still writing can leave a timeline on disk that "
                "covers part of the recording and reads exactly like a whole "
                "one. Run the reconstruction again and import from that job."
            ),
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
    rollback = verified_preimport_snapshot(
        db, job_id=job_id, backups=Path(paths["backups"]), data_dir=Path(paths["data"])
    )
    if rollback.refusal is not None or rollback.path is None:
        return HandImportResult(
            status="blocked",
            session_id=session_id,
            reasons=("pre-import snapshot unavailable",),
            message=rollback.refusal or "No verified pre-import rollback point.",
        )
    rollback_point = rollback.path.name
    with db.transaction(immediate=True):
        existing = find_imported_hand_in_any_session(
            db,
            job_id=job_id,
            timeline_hand_number=timeline_hand_number,
            related_job_ids=related_jobs,
        )
        if existing is not None:
            holder = existing.session_id
            if holder == session_id:
                message = f"Hand already in session #{session_id}."
            else:
                message = (
                    f"Hand #{timeline_hand_number} from this recording was already "
                    f"imported into session #{holder}, so nothing was added to "
                    f"session #{session_id} — a second copy would be one real hand "
                    "counted twice in both sessions' results. Move the existing "
                    "hand between sessions instead of importing it again."
                )
            return HandImportResult(
                status="already_present",
                hand_id=existing.id,
                # The session that actually HOLDS the hand, not the one that asked.
                # Reporting the destination is what let a caller believe the hand
                # had landed where it requested: ensure_draft_for_review treats
                # already_present as success and renders this id.
                session_id=holder,
                message=message,
                rollback_point=rollback_point,
            )
        import_hands_into_session(db, one_hand_payload, session_id)
        # Deliberately the session-scoped finder: this asks "did the row land HERE",
        # which is a claim about the destination. The cross-session form would answer
        # yes off a copy in another session and report an import that never happened.
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
            rollback_point=rollback_point,
        )
    label = "draft" if mode == "draft" else "validated hand"
    return HandImportResult(
        status="imported",
        hand_id=saved.id,
        session_id=session_id,
        message=f"Added {label} #{timeline_hand_number} to session #{session_id}.",
        rollback_point=rollback_point,
    )


@dataclass(frozen=True)
class PreimportSnapshot:
    """The verified rollback point an import may proceed against, or the refusal.

    Two states, and they are no longer spelled the same way. ``refusal`` set means
    this import must not proceed; ``path`` set means a snapshot was proved -- in
    this call, not in some earlier one -- to survive an isolated restore.
    """

    path: Path | None = None
    refusal: str | None = None


def ensure_preimport_snapshot(
    db: PokerDatabase,
    *,
    job_id: int,
    backups: Path,
    data_dir: Path,
) -> str | None:
    """Backwards-compatible spelling: the refusal string, or None to proceed."""
    return verified_preimport_snapshot(
        db, job_id=job_id, backups=backups, data_dir=data_dir
    ).refusal


def verified_preimport_snapshot(
    db: PokerDatabase,
    *,
    job_id: int,
    backups: Path,
    data_dir: Path,
) -> PreimportSnapshot:
    """Take one verified rollback point per CV job. Returns a refusal, or None.

    The snapshot used to be taken per imported HAND, and unpinned: importing
    eight validated hands wrote eight full copies of the database and pushed the
    pre-import state out of the five-slot rotation, so the oldest surviving copy
    was one taken after three hands had already landed. The rollback point the
    snapshot exists to provide was destroyed by the act of taking it repeatedly.

    What is being protected is the state before this job's hands land, and that
    state stops existing after the first import, so one snapshot per job is not
    an economy -- it is the correct number. The retained snapshot is its own
    marker: its filename carries the job, so the interactive one-hand-at-a-time
    path and the batch scan both find it and neither takes a second.

    It is verified before the import proceeds and refuses the import if it does
    not survive an isolated restore, on the same reasoning as the pre-migration
    snapshot: an unverified rollback point is not a rollback point, and the
    moment before the write is the last moment at which refusing still helps.

    EVERY call verifies, and that is the round-16 repair. Existence used to stand
    in for verification -- ``if find_snapshots(...): return None`` short-circuited
    before ``verify_snapshot`` was ever reached -- so exactly one hand per job
    ever got the guarantee this docstring states. The interactive path it
    describes adds hands one click at a time, so hands 2..N of a job proceeded
    against a file nobody had looked at since hand 1. If the backup mount filled,
    or a truncating copy or an interrupted rsync left the snapshot zero bytes in
    between, the listing still contained the name, the function still returned
    "proceed", and the remaining hands landed in the study database with no
    usable rollback point and nothing anywhere saying so -- a snapshot the
    product's own verifier grades ``fail``, reported as a verified one.

    A retained artifact cannot be its own proof-of-check. It is a fine MARKER --
    the filename still answers "has this job been snapshotted" from the directory
    listing, with no marker file and no schema column -- but the question the
    import actually depends on is "does that file still restore", and only
    restoring it answers that. The cost is one isolated restore per hand rather
    than one per job; a rollback point that is not checked is not cheaper than
    one that is, it is absent.

    When no retained snapshot for this job restores any more, this takes a fresh
    one rather than dead-ending the operator on a corrupt file they would have to
    find and delete by hand. That replacement is honest about less than the first
    snapshot was: it postdates whatever already landed, so it is a rollback point
    for the hands still to come. The same is true after rotation evicts this
    job's snapshot (the preimport pool is per-CLASS, so five other jobs push this
    one out), and it is the most any snapshot taken at that moment could be.

    KNOWN AND NOT FIXED HERE: the replacement is written under the same
    ``preimport-job<N>-<stamp>`` name as the original, and that name asserts it
    precedes job N's imports when it does not. Correcting the assertion means
    changing the snapshot filename grammar in ``poker_tracker.persistence.backup``
    -- a persisted artifact shape that rotation, the health audit and the restore
    runbook all match on -- which is not this module's to change.
    """
    scope = f"job{job_id}"
    live = Path(db.db_path)
    stale: list[str] = []
    for snapshot in find_snapshots(backups, purpose="preimport", scope=scope):
        verification = verify_snapshot(
            snapshot, live_database=live, data_dir=data_dir
        )
        if verification.status != "fail":
            return PreimportSnapshot(path=snapshot)
        stale.append(
            f"{snapshot.name} ({'; '.join(verification.details) or verification.message})"
        )
    context = (
        ""
        if not stale
        else (
            " The retained snapshot(s) for this job no longer restore: "
            f"{'; '.join(stale)}."
        )
    )
    try:
        snapshot = backup_database(
            live,
            backups,
            purpose="preimport",
            scope=scope,
            data_dir=data_dir,
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        return PreimportSnapshot(
            refusal=(
                f"Could not write a pre-import snapshot to {backups}: "
                f"{type(exc).__name__}: {exc}.{context}"
            )
        )
    verification = verify_snapshot(snapshot, live_database=live, data_dir=data_dir)
    if verification.status == "fail":
        detail = "; ".join(verification.details) or verification.message
        return PreimportSnapshot(
            refusal=(
                f"The pre-import snapshot {snapshot.name} did not survive an isolated "
                f"restore, so this import has no rollback point: {detail}.{context}"
            )
        )
    return PreimportSnapshot(path=snapshot)


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
    from datetime import date

    from cv_lab.scripts.eval.validate_yolo_card_timeline import validate_timeline
    from poker_tracker.persistence.models import Session

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
