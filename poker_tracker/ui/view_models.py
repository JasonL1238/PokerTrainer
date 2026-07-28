"""Pure display transformations used by the product UI."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from poker_tracker.persistence.models import Hand, ProcessingJob, Session, VideoRecord


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
class JobRow:
    job_id: int
    video_id: int
    filename: str
    job_type: str
    status: str
    progress_percent: float
    message: str
    age_label: str


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


def build_job_rows(
    jobs: Iterable[ProcessingJob],
    videos: Iterable[VideoRecord],
    *,
    now: datetime | None = None,
) -> list[JobRow]:
    video_names = {video.id: video.original_filename for video in videos if video.id is not None}
    current = now or datetime.now(UTC)
    return [
        JobRow(
            job_id=job.id or 0,
            video_id=job.video_id,
            filename=video_names.get(job.video_id, "Unknown video"),
            job_type=job.job_type.replace("_", " ").title(),
            status=job.status,
            progress_percent=job.progress_percent,
            message=job.message or job.error_message,
            age_label=format_age(job.created_at, current),
        )
        for job in jobs
        if job.id is not None
    ]


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
