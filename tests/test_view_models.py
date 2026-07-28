from datetime import UTC, datetime, timedelta

from poker_tracker.persistence.models import Hand, ProcessingJob, Session, VideoRecord
from poker_tracker.ui.view_models import (
    build_hand_rows,
    build_job_rows,
    build_portfolio_summary,
    build_session_rows,
    confidence_label,
    format_age,
)


def test_portfolio_and_session_rows_preserve_sample_counts() -> None:
    session = Session(id=4, name="Sunday review", stakes="1/2")
    hands = [
        Hand(id=1, session_id=4, hand_number=1, hero_bb_won=12, review_status="reviewed"),
        Hand(id=2, session_id=4, hand_number=2, hero_bb_won=-5),
        Hand(id=3, session_id=4, hand_number=3),
    ]

    summary = build_portfolio_summary(hands, 1)
    rows = build_session_rows([session], {4: hands})

    assert summary.hand_count == 3
    assert summary.reviewed_count == 1
    assert summary.review_percent == 100 / 3
    assert summary.net_bb == 7
    assert rows[0].hand_count == 3
    assert rows[0].net_bb == 7


def test_hand_rows_label_source_confidence_and_unknown_cards() -> None:
    session = Session(id=7, name="CV import")
    hands = [
        Hand(
            id=8,
            session_id=7,
            hand_number=2,
            confidence_score=0.59,
            source_type="cv_import",
            tags=["LOW_CONFIDENCE"],
        )
    ]

    row = build_hand_rows([session], hands)[0]

    assert row.session_name == "CV import"
    assert row.hero_cards == "Unknown"
    assert row.confidence_label == "Low"
    assert row.source_label == "Cv Import"
    assert row.tags == ("LOW_CONFIDENCE",)


def test_job_rows_include_video_and_relative_age() -> None:
    now = datetime.now(UTC)
    video = VideoRecord(id=2, original_filename="session.mov", stored_path="/tmp/x", file_size_bytes=1)
    job = ProcessingJob(
        id=3,
        job_type="cv_reconstruction",
        status="running",
        video_id=2,
        progress_percent=40,
        created_at=now - timedelta(minutes=5),
    )

    row = build_job_rows([job], [video], now=now)[0]

    assert row.filename == "session.mov"
    assert row.age_label == "5m ago"
    assert row.job_type == "Cv Reconstruction"


def test_confidence_and_age_boundaries() -> None:
    now = datetime.now(UTC)
    assert confidence_label(None) == "Not scored"
    assert confidence_label(0.85) == "High"
    assert confidence_label(0.6) == "Medium"
    assert confidence_label(0.59) == "Low"
    assert format_age(now - timedelta(seconds=20), now) == "Just now"
    assert format_age(now - timedelta(hours=2), now) == "2h ago"
