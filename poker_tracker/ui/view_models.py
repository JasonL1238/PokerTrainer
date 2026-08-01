"""Pure display transformations used by the product UI."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from poker_tracker.persistence.completion import (
    BoundaryEvidence,
    CompletionEvidence,
    has_operator_manual_completion,
)
from poker_tracker.persistence.models import Hand, ProcessingJob, Session, VideoRecord
from poker_tracker.safety.redaction import redact_text

# A progress reading only describes a job that is still doing work. Every other
# status is terminal, and the two sets below are the only place that distinction
# is made, so a status added later cannot quietly acquire a progress bar: it
# falls through to "stopped without succeeding", which is the safe side.
LIVE_JOB_STATUSES = frozenset({"queued", "running", "cancelling"})
SUCCEEDED_JOB_STATUSES = frozenset({"completed"})

# What one job of each type actually commits, and the verb the product uses for
# it. "imported" is deliberate for reconstruction: the promise the operator is
# owed is that no failed or partial job ever claims a hand was imported.
_COMMITTED_UNITS: dict[str, tuple[str, str, str]] = {
    "cv_reconstruction": ("hand", "hands", "imported"),
    "frame_extraction": ("frame", "frames", "extracted"),
}
_UNKNOWN_UNIT = ("record", "records", "written")


@dataclass(frozen=True)
class PortfolioSummary:
    session_count: int
    hand_count: int
    reviewed_count: int
    review_percent: float
    net_bb: float


@dataclass(frozen=True)
class SessionRow:
    session_id: int
    name: str
    date_played: str
    platform: str
    stakes: str
    hand_count: int
    reviewed_count: int
    net_bb: float


@dataclass(frozen=True)
class HandRow:
    hand_id: int
    session_id: int
    session_name: str
    hand_number: int
    hero_cards: str
    board_cards: str
    position: str
    result_bb: float | None
    review_status: str
    confidence_label: str
    source_label: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class JobOutcome:
    """What a job is doing, or what it left behind once it stopped.

    A percentage is a reading of work in flight. The moment a job stops without
    succeeding it stops being a reading of anything, and "82%" beside a failed
    import is read as "most of my hands got in" -- the precise claim this
    product is not allowed to make. So ``progress_percent`` is ``None`` for
    every terminal status and ``progress_label`` degrades to a dash, and their
    place is taken by ``statement``: a count that was queried from the database
    rather than inferred from how far the worker happened to get.

    ``committed_count is None`` means the caller did not look. That is said
    plainly instead of being rendered as zero, because "nothing was imported"
    is itself a claim about the database and has to be earned by a query.
    """

    job_id: int
    job_type: str
    status: str
    is_live: bool
    succeeded: bool
    headline: str
    statement: str
    progress_percent: float | None
    progress_label: str
    committed_count: int | None
    destination: str
    error_message: str
    log_path: str
    log_tail: tuple[str, ...]


@dataclass(frozen=True)
class JobRow:
    job_id: int
    video_id: int
    filename: str
    job_type: str
    status: str
    # ``None`` once the job has stopped: a stale reading is not an outcome, and
    # a row that cannot carry one cannot be formatted into a misleading percent.
    progress_percent: float | None
    message: str
    age_label: str
    outcome: JobOutcome

    @property
    def progress_label(self) -> str:
        return self.outcome.progress_label

    @property
    def outcome_statement(self) -> str:
        return self.outcome.statement


def build_portfolio_summary(hands: Iterable[Hand], session_count: int) -> PortfolioSummary:
    items = list(hands)
    reviewed = sum(hand.review_status == "reviewed" for hand in items)
    results = [hand.hero_bb_won for hand in items if hand.hero_bb_won is not None]
    return PortfolioSummary(
        session_count=session_count,
        hand_count=len(items),
        reviewed_count=reviewed,
        review_percent=(100 * reviewed / len(items)) if items else 0,
        net_bb=sum(results),
    )


def build_session_rows(
    sessions: Iterable[Session],
    hands_by_session: Mapping[int, Iterable[Hand]],
) -> list[SessionRow]:
    rows: list[SessionRow] = []
    for session in sessions:
        if session.id is None:
            continue
        hands = list(hands_by_session.get(session.id, []))
        rows.append(
            SessionRow(
                session_id=session.id,
                name=session.name,
                date_played=session.date_played.isoformat(),
                platform=session.platform or "Manual",
                stakes=session.stakes or "—",
                hand_count=len(hands),
                reviewed_count=sum(hand.review_status == "reviewed" for hand in hands),
                net_bb=sum(hand.hero_bb_won for hand in hands if hand.hero_bb_won is not None),
            )
        )
    return rows


def build_hand_rows(
    sessions: Iterable[Session],
    hands: Iterable[Hand],
) -> list[HandRow]:
    session_names = {session.id: session.name for session in sessions if session.id is not None}
    rows: list[HandRow] = []
    for hand in hands:
        if hand.id is None:
            continue
        rows.append(
            HandRow(
                hand_id=hand.id,
                session_id=hand.session_id,
                session_name=session_names.get(hand.session_id, "Unknown session"),
                hand_number=hand.hand_number,
                hero_cards=hand.hero_cards or "Unknown",
                board_cards=hand.board_cards or "—",
                position=hand.hero_position or "—",
                result_bb=hand.hero_bb_won,
                review_status=hand.review_status,
                confidence_label=confidence_label(hand.confidence_score),
                source_label=hand.source_type.replace("_", " ").title(),
                tags=tuple(hand.tags),
            )
        )
    return rows


def job_is_live(status: str) -> bool:
    """Whether the job is still working, and its progress reading still means something."""
    return (status or "").strip().lower() in LIVE_JOB_STATUSES


def job_succeeded(status: str) -> bool:
    return (status or "").strip().lower() in SUCCEEDED_JOB_STATUSES


def job_stopped_without_success(status: str) -> bool:
    """Failed, cancelled, timed out, worker died -- and anything added later.

    Deliberately defined by exclusion. A status this build does not recognize is
    treated as a terminal failure rather than as live work, so a new terminal
    state cannot inherit a progress bar by being forgotten here.
    """
    return not job_is_live(status) and not job_succeeded(status)


def build_job_outcome(
    job: ProcessingJob,
    *,
    committed_count: int | None = None,
    destination: str = "",
    log_path: str = "",
    log_tail: Iterable[str] = (),
) -> JobOutcome:
    """Phrase one job's state as an outcome the operator can act on.

    Every caller that has queried what the job actually committed passes
    ``committed_count``; the wording below never guesses it from the progress
    figure, which is the whole point. Free text reaching the screen -- the
    worker's message, the failure detail, the log tail -- is scrubbed here so
    that no surface can display an unredacted credential by choosing to render
    a field directly.
    """
    status = (job.status or "").strip().lower()
    live = job_is_live(status)
    succeeded = job_succeeded(status)
    singular, plural, verb = _COMMITTED_UNITS.get(job.job_type, _UNKNOWN_UNIT)
    title = status.replace("_", " ").title() or "Unknown"
    message = redact_text(job.message or "").strip()
    error_message = redact_text(job.error_message or "").strip()
    where = f" in {destination}" if destination else ""

    if live:
        progress_percent: float | None = job.progress_percent
        progress_label = f"{job.progress_percent:.0f}%"
        headline = title
        statement = message or "Working."
    else:
        progress_percent = None
        progress_label = "—"
        count_phrase = (
            ""
            if committed_count is None
            else (f"1 {singular}" if committed_count == 1 else f"{committed_count} {plural}")
        )
        if succeeded:
            headline = title
            if committed_count is None:
                statement = message or "Finished."
            elif committed_count == 0:
                statement = (
                    f"{message + '. ' if message else ''}"
                    f"No {plural} from this job are in the study database yet."
                )
            else:
                statement = (
                    f"{message + '. ' if message else ''}"
                    f"{count_phrase} from this job {'is' if committed_count == 1 else 'are'} "
                    f"in the study database{where}."
                )
        elif committed_count is None:
            headline = title
            statement = (
                f"{title}. This view did not check what the job committed; open the "
                "video's reconstruction panel for the outcome."
            )
        elif committed_count == 0:
            headline = f"{title} · nothing {verb}"
            statement = (
                f"No {plural} were {verb}. Nothing from this job reached the study database."
            )
        else:
            headline = f"{title} · {count_phrase} {verb}"
            statement = (
                f"{count_phrase} {'was' if committed_count == 1 else 'were'} {verb} "
                f"before the job stopped, and {'it is' if committed_count == 1 else 'they are'} "
                f"{('in ' + destination) if destination else 'in the study database'}. "
                f"Nothing else from this job reached the study database."
            )

    return JobOutcome(
        job_id=job.id or 0,
        job_type=job.job_type,
        status=status,
        is_live=live,
        succeeded=succeeded,
        headline=headline,
        statement=statement,
        progress_percent=progress_percent,
        progress_label=progress_label,
        committed_count=committed_count,
        destination=destination,
        error_message=error_message,
        log_path=log_path,
        log_tail=tuple(redact_text(line) for line in log_tail),
    )


def build_job_rows(
    jobs: Iterable[ProcessingJob],
    videos: Iterable[VideoRecord],
    *,
    outcomes: Mapping[int, JobOutcome] | None = None,
    now: datetime | None = None,
) -> list[JobRow]:
    """Rows for a job list, each carrying the outcome it is allowed to display.

    ``outcomes`` comes from the resolver that queried the database. A caller
    that supplies none still gets rows, but the outcome they carry says the
    committed work was not checked rather than inventing a zero.
    """
    video_names = {video.id: video.original_filename for video in videos if video.id is not None}
    resolved = outcomes or {}
    current = now or datetime.now(UTC)
    rows: list[JobRow] = []
    for job in jobs:
        if job.id is None:
            continue
        outcome = resolved.get(job.id) or build_job_outcome(job)
        rows.append(
            JobRow(
                job_id=job.id,
                video_id=job.video_id,
                filename=video_names.get(job.video_id, "Unknown video"),
                job_type=job.job_type.replace("_", " ").title(),
                status=job.status,
                progress_percent=outcome.progress_percent,
                message=redact_text(job.message or job.error_message),
                age_label=format_age(job.created_at, current),
                outcome=outcome,
            )
        )
    return rows


def completion_evidence_rows(
    evidence: CompletionEvidence,
) -> tuple[tuple[str, str], ...]:
    """The reconstruction evidence, as label/value pairs a promotion gate can show.

    The confirmation checkbox reads "I have read the evidence above and confirm
    this hand is correct", and until this existed there was no evidence above it:
    the pipeline wrote the boundaries, the terminal event, the boundary
    confidence, the source timestamps and frames, the layout profile and the
    pipeline/model versions into ``completion_evidence``, the store persisted
    them, the exporter round-tripped them and ``study_confirmation_key`` digested
    them -- and not one of them was ever rendered. The only thing drawn above the
    box on an otherwise-clean hand was the sentence saying the box was unticked,
    so Phase 1's final control asked the operator to attest to something the
    product had never shown them.

    Pure and total: every field degrades to a printable string, because the
    parser never raises and an unreadable blob must still render as the little it
    could recover rather than break the page that gates the promotion.
    """

    def boundary(label: str, value: BoundaryEvidence) -> tuple[str, str] | None:
        if not value.kind and value.timestamp_s is None and not value.frame_ref:
            return None
        parts = [value.kind or "unknown"]
        if value.timestamp_s is not None:
            parts.append(f"at {value.timestamp_s:g}s")
        if value.confidence is not None:
            parts.append(f"confidence {value.confidence:g}")
        if value.frame_ref:
            parts.append(value.frame_ref)
        if value.codes:
            parts.append(", ".join(value.codes))
        return (label, " · ".join(parts))

    def flag(value: bool | None) -> str:
        return "Unknown" if value is None else ("Yes" if value else "No")

    rows: list[tuple[str, str]] = [
        (
            "Evidence version",
            f"{evidence.evidence_version}"
            + ("" if evidence.is_known else " (not readable by this build)"),
        ),
        ("Starts mid-hand", flag(evidence.partial_start)),
        ("Ends mid-hand", flag(evidence.partial_end)),
        ("Terminal event", evidence.terminal_event or "Not recorded"),
        (
            "Boundary confidence",
            "Not recorded"
            if evidence.boundary_confidence is None
            else f"{evidence.boundary_confidence:g}",
        ),
    ]
    if has_operator_manual_completion(evidence):
        op_terminal = evidence.extra.get("operator_terminal_event") or "Not recorded"
        rows.append(("Operator-finalized", f"Yes · terminal {op_terminal}"))
    span = [
        f"{value:g}s"
        for value in (
            evidence.first_source_timestamp_s,
            evidence.last_source_timestamp_s,
        )
        if value is not None
    ]
    rows.append(("Source timestamps", " → ".join(span) if span else "Not recorded"))
    for row in (
        boundary("Preceding boundary", evidence.preceding_boundary),
        boundary("Following boundary", evidence.following_boundary),
    ):
        if row is not None:
            rows.append(row)
    rows.append(
        (
            "Source frames",
            f"{len(evidence.source_frames)} · {', '.join(evidence.source_frames)}"
            if evidence.source_frames
            else "None recorded",
        )
    )
    rows.append(
        (
            "Table layout",
            f"{evidence.layout_profile or 'Profile not recorded'} · "
            f"{'supported' if evidence.layout_supported else 'not confirmed'} · "
            f"{evidence.table_size if evidence.table_size is not None else '?'} seats",
        )
    )
    rows.append(("Pipeline version", evidence.pipeline_version or "Not recorded"))
    rows.append(
        (
            "Model versions",
            ", ".join(f"{name}={value}" for name, value in sorted(evidence.model_versions.items()))
            or "Not recorded",
        )
    )
    if evidence.warning_codes:
        rows.append(("Warning codes", ", ".join(evidence.warning_codes)))
    if evidence.rejection_codes:
        rows.append(("Rejection codes", ", ".join(evidence.rejection_codes)))
    if evidence.acknowledged_codes:
        rows.append(("Acknowledged codes", ", ".join(evidence.acknowledged_codes)))
    if evidence.declared_settlement_codes:
        # A third kind of statement again, and the reason it is drawn on its own
        # row: this is what the OPERATOR declared, where the warning codes above
        # are what the PIPELINE could not prove. Filing it with the warnings made
        # the evidence panel present a rake the operator typed as a finding of the
        # reconstruction.
        rows.append(
            (
                "Declared settlement inputs",
                ", ".join(evidence.declared_settlement_codes),
            )
        )
    if evidence.confirmed_assumption_codes:
        # Rendered separately because it is a different kind of statement: an
        # acknowledged code says "I accept this pipeline note", and this says
        # "I assert these specific unobserved chips were taken or added".
        rows.append(
            (
                "Confirmed settlement assumptions",
                ", ".join(evidence.confirmed_assumption_codes),
            )
        )
    return tuple(rows)


def confidence_label(score: float | None) -> str:
    if score is None:
        return "Not scored"
    if score >= 0.85:
        return "High"
    if score >= 0.6:
        return "Medium"
    return "Low"


def format_age(value: datetime, now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    seconds = max(0, int((current - value).total_seconds()))
    if seconds < 60:
        return "Just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86_400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86_400}d ago"
