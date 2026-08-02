from datetime import UTC, datetime, timedelta

from poker_tracker.persistence.models import Hand, ProcessingJob, Session, VideoRecord
from poker_tracker.ui.view_models import (
    build_hand_rows,
    build_job_rows,
    build_portfolio_summary,
    build_session_rows,
    confidence_label,
    format_age,
    result_basis_label,
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
    assert row.evidence_class == "cv_draft"
    assert row.evidence_label == "CV draft"


def test_hand_rows_carry_the_evidence_class_the_metric_denominator_counts() -> None:
    """One classification, so a row and the Insights denominator cannot disagree.

    ``source_label`` still reports provenance; ``evidence_class`` is the stronger
    statement, and a reviewed CV hand belongs with the reviewed hands in both
    places or the parts stop summing to the whole.
    """
    session = Session(id=1, name="Mixed")
    hands = [
        Hand(id=1, session_id=1, hand_number=1),
        Hand(id=2, session_id=1, hand_number=2, source_type="cv_import"),
        Hand(id=3, session_id=1, hand_number=3, source_type="corrected_cv"),
        Hand(
            id=4,
            session_id=1,
            hand_number=4,
            source_type="cv_import",
            review_status="reviewed",
        ),
    ]

    rows = build_hand_rows([session], hands)

    assert [row.evidence_class for row in rows] == [
        "manual",
        "cv_draft",
        "corrected_cv",
        "reviewed",
    ]
    assert rows[3].source_label == "Cv Import"
    assert rows[3].evidence_label == "Reviewed"


def test_a_derived_result_is_never_rendered_as_an_observed_one() -> None:
    """``derived_result_substituted`` had no display consumer at all.

    A hand whose ``hero_bb_won`` column is NULL but whose reconciliation derives
    -18 BB showed -18 BB in the Hands library indistinguishable from a figure the
    operator recorded.
    """
    session = Session(id=1, name="Mixed")
    observed = Hand(id=1, session_id=1, hand_number=1, hero_bb_won=-18)
    derived = observed.model_copy(update={"derived_result_substituted": True})
    blank = Hand(id=2, session_id=1, hand_number=2)

    rows = build_hand_rows([session], [observed, derived, blank])

    assert rows[0].result_bb == rows[1].result_bb == -18
    assert rows[0].result_basis_label == "Recorded on the hand"
    assert rows[1].result_basis_label == "Derived from the reconciled ledger"
    assert rows[2].result_basis_label == "No result recorded"
    assert result_basis_label(derived) == "Derived from the reconciled ledger"


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
