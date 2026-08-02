"""End-to-end proof for the Overview and Sessions surfaces.

Every assertion here is against SEEDED rows, not against copy. The phase these
tests belong to is about counts and provenance, and a test that only checks a
heading passes just as happily on a page whose numbers are wrong.

The governing rule, applied to a screen: a figure without its denominator, a CV
draft rendered like confirmed data, and a badge that survived the correction that
invalidated it are all the same defect. So the checks below are mostly of the
form "this number is X, and the page says what X is out of, and what kind of
evidence produced it".
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandIssue,
    HandPlayer,
    HandSettlement,
    ProcessingJob,
    Session,
    SettlementEntry,
    VideoRecord,
    utc_now,
)
from poker_tracker.services.hand_accounting import persist_reconciliation
from poker_tracker.services.validated_hand_import import CV_TIMELINE_IDENTITY_KEY
from poker_tracker.ui.jobs import hands_committed_by_job
from poker_tracker.ui.navigation import Page
from poker_tracker.ui.ui_theme import _THEME_CSS
from tests.conftest import attest_declared_assumptions

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

CLEAN_EVIDENCE: dict[str, object] = {
    "evidence_version": 1,
    "partial_start": False,
    "partial_end": False,
    "terminal_event": "showdown",
    "first_source_timestamp_s": 1.0,
    "last_source_timestamp_s": 9.0,
    "boundary_confidence": 0.91,
    "source_frames": ["frames/start.png", "frames/end.png"],
    "layout_profile": "clubwpt-6max",
    "layout_supported": True,
    "table_size": 6,
    "pipeline_version": "two-model-v7",
    "model_versions": {"detector": "v7"},
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _drop_cached_database() -> Iterator[None]:
    """Never hand the next test a connection to this test's temporary database.

    ``get_database`` is ``@st.cache_resource``, keyed on the function rather than
    on the path, so a cached handle outlives the run that created it and the next
    AppTest renders whatever this one seeded. That reaches other modules, not just
    this one, which makes it this module's job to clean up after itself.
    """
    yield
    st.cache_resource.clear()


def _open(path: Path) -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def _configure(path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()


def _run(path: Path, monkeypatch, page: Page | None = None) -> AppTest:
    _configure(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=90).run()
    assert not list(app.exception), [str(item) for item in app.exception]
    if page is not None:
        app.radio[0].set_value(page)
        app.run()
        assert not list(app.exception), [str(item) for item in app.exception]
    return app


def _page_text(app: AppTest) -> str:
    """Every rendered string on the page, as one haystack.

    Deliberately includes captions and expander labels: this phase's failures are
    almost always a missing qualifier rather than a missing headline.
    """
    parts: list[str] = []
    for group in (
        app.markdown,
        app.caption,
        app.info,
        app.warning,
        app.error,
        app.success,
        app.code,
        app.subheader,
    ):
        parts.extend(str(item.value) for item in group)
    parts.extend(str(item.label) for item in app.button)
    parts.extend(str(item.label) for item in app.expander)
    for frame in app.dataframe:
        parts.append(str(frame.value))
    return "\n".join(parts)


def _seed_hand(
    db: PokerDatabase,
    session_id: int,
    hand_number: int,
    *,
    source_type: str = "manual",
    completion_status: str | None = None,
    review_status: str = "unreviewed",
    hero_bb_won: float | None = 5.0,
    tags: list[str] | None = None,
    evidence: dict[str, object] | None = None,
) -> Hand:
    return db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=hand_number,
            game_type="No-limit Hold'em",
            table_size=6,
            hero_position="BTN",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=20,
            hero_bb_won=hero_bb_won,
            review_status=review_status,
            source_type=source_type,
            completion_status=(
                completion_status
                if completion_status is not None
                else ("not_applicable" if source_type == "manual" else "uncertain")
            ),
            completion_evidence=(
                evidence
                if evidence is not None
                else (dict(CLEAN_EVIDENCE) if source_type != "manual" else {})
            ),
            tags=tags or [],
        )
    )


def _seed_reconciled_hand(
    db: PokerDatabase, session_id: int, hand_number: int, *, review_status: str = "unreviewed"
) -> Hand:
    """A hand whose hero result exists only because the ledger derived it.

    ``hero_bb_won`` is left NULL on purpose: the display substitution then has a
    derivation to publish and marks the copy, which is exactly the case the
    Overview has to keep apart from an observed figure.

    Lands unreviewed whatever the caller asks for, because writing settlement
    rows demotes a reviewed hand at the store level. That is the store being
    right; the fixture just records that it happens.
    """
    hand = _seed_hand(
        db,
        session_id,
        hand_number,
        source_type="manual",
        review_status=review_status,
        hero_bb_won=None,
    )
    assert hand.id is not None
    hero = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            seat_index=0,
            player_name="Hero",
            position="BTN",
            starting_stack=100,
            is_hero=True,
        )
    )
    villain = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="villain",
            seat_index=1,
            player_name="Villain",
            position="BB",
            starting_stack=100,
        )
    )
    for actor, action_type in ((hero, "bet"), (villain, "call")):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=actor.player_key,
                player_name=actor.player_name,
                position=actor.position,
                street="river",
                action_type=action_type,
                amount=10,
                amount_semantics="incremental",
            )
        )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled", rake_rate=0.0))
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key=hero.player_key,
                player_name=hero.player_name,
                amount=20,
                entry_order=1,
            )
        ],
    )
    persist_reconciliation(db, hand.id)
    attest_declared_assumptions(db, hand.id)
    return db.fetch_hand(hand.id)


def _seed_issue(db: PokerDatabase, hand_id: int, description: str = "Board misread") -> HandIssue:
    return db.create_hand_issue(
        HandIssue(hand_id=hand_id, issue_types=["cards"], description=description)
    )


def _seed_stale_coaching(db: PokerDatabase, hand_id: int) -> None:
    db.create_coaching_response(
        CoachingResponse(
            hand_id=hand_id,
            review_type="hand",
            provider_name="test",
            model_name="test",
            raw_prompt="prompt",
            raw_response="response",
            is_stale=True,
            stale_reason="Hand facts were corrected after this review.",
        )
    )


def _seed_video(db: PokerDatabase, session_id: int | None, name: str) -> VideoRecord:
    return db.create_video(
        VideoRecord(
            session_id=session_id,
            original_filename=name,
            stored_path=f"/tmp/{name}",
            file_size_bytes=1024,
            content_sha256=f"sha-{name}",
        )
    )


def _seed_job(
    db: PokerDatabase,
    video_id: int,
    *,
    status: str = "completed",
    progress_percent: float = 100.0,
    job_type: str = "cv_reconstruction",
    error_message: str = "",
    live: bool = False,
) -> ProcessingJob:
    """Seed one job. ``live=True`` gives it a running worker's pid and heartbeat.

    Without those, ``reconcile_stuck_jobs`` correctly fails a 'running' row at
    startup because no worker holds it, and a fixture that did not say so would
    look like a bug in the page rather than the recovery path doing its job.
    """
    return db.create_processing_job(
        ProcessingJob(
            video_id=video_id,
            job_type=job_type,
            status=status,
            progress_percent=progress_percent,
            message="",
            error_message=error_message,
            pid=os.getpid() if live else None,
            heartbeat_at=utc_now() if live else None,
            started_at=utc_now() if live else None,
        )
    )


# ---------------------------------------------------------------------------
# Overview: counts
# ---------------------------------------------------------------------------


def test_overview_states_session_hand_review_issue_reconciliation_and_job_counts(
    tmp_path, monkeypatch
) -> None:
    """Six counts, each checked against the rows that produced it.

    Pre-repair the page carried three of them (sessions, hands, review percent)
    and no denominator on the review figure.
    """
    path = tmp_path / "counts.sqlite3"
    db = _open(path)
    first = db.create_session(Session(name="Alpha"))
    second = db.create_session(Session(name="Beta"))
    _seed_hand(db, first.id, 1, review_status="reviewed")
    _seed_hand(db, first.id, 2)
    flagged = _seed_hand(db, first.id, 3, review_status="needs_correction")
    _seed_reconciled_hand(db, second.id, 1)
    _seed_issue(db, flagged.id, "Hero cards unreadable")
    _seed_issue(db, flagged.id, "Pot size disagrees with stacks")
    video = _seed_video(db, first.id, "alpha.mp4")
    _seed_job(db, video.id, status="completed")
    _seed_job(db, video.id, status="failed", progress_percent=82, error_message="worker died")
    _seed_job(db, video.id, status="queued", progress_percent=0)
    db.close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert '"pt-kpi-label">Sessions</div><div class="pt-kpi-value">2<' in text
    assert '"pt-kpi-label">Hands</div><div class="pt-kpi-value">4<' in text
    # Review coverage carries its denominator, not a bare count.
    assert "1 of 4 hands marked reviewed" in text
    assert '"pt-kpi-label">Open issues</div><div class="pt-kpi-value">2<' in text
    assert "Saved debugging issues awaiting resolution" in text
    # One reconciled result (the ledger-derived hand), three observed.
    assert "Reconciled results" in text
    assert "3 were recorded as observed" in text
    # Job counts cover every job, not the recent window.
    assert "3 total · 1 in flight · 1 completed · 1 stopped without succeeding" in text


def test_overview_open_issue_count_names_the_orphans_it_cannot_show(
    tmp_path, monkeypatch
) -> None:
    """An issue whose hand is gone is reported separately, never folded into the count.

    Reached here the only way it is reachable: a database whose foreign keys were
    off when the hand went. That is the same degraded store ``audit_data_health``
    exists to find, and the rule is the one this whole surface follows -- a count
    the page cannot substantiate with a row must say so rather than be quietly
    larger than what it shows.
    """
    path = tmp_path / "orphan.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    kept = _seed_hand(db, session.id, 1)
    doomed = _seed_hand(db, session.id, 2)
    _seed_issue(db, kept.id, "Still listed")
    _seed_issue(db, doomed.id, "Hand later deleted")
    db.close()

    raw = sqlite3.connect(path)
    raw.execute("PRAGMA foreign_keys = OFF")
    raw.execute("DELETE FROM hands WHERE id = ?", (doomed.id,))
    raw.commit()
    raw.close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert '"pt-kpi-label">Open issues</div><div class="pt-kpi-value">1<' in text
    assert "1 more on hands no longer in the library" in text


# ---------------------------------------------------------------------------
# Overview: the six data states
# ---------------------------------------------------------------------------


def test_overview_distinguishes_all_six_data_states(tmp_path, monkeypatch) -> None:
    """completed / partial / uncertain / corrected / stale / reviewed, each with a count.

    Six labels over one flag would pass a check for the six words, so each is
    asserted with the number of seeded rows behind it.
    """
    path = tmp_path / "states.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    _seed_hand(db, session.id, 1, source_type="cv_import", completion_status="complete")
    _seed_hand(db, session.id, 2, source_type="cv_import", completion_status="partial")
    _seed_hand(db, session.id, 3, source_type="cv_import", completion_status="uncertain")
    corrected = _seed_hand(
        db,
        session.id,
        4,
        source_type="corrected_cv",
        completion_status="complete",
        review_status="reviewed",
    )
    _seed_stale_coaching(db, corrected.id)
    db.close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert "Completed</strong>" in text or "Completed <strong>2</strong>" in text
    for label, count in (
        ("Completed", 2),
        ("Partial", 1),
        ("Uncertain", 1),
        ("Corrected CV", 1),
        ("CV draft", 3),
        ("Marked reviewed", 1),
    ):
        assert f"{label} <strong>{count}</strong>" in text, (label, count)
    # Staleness is the reading that cuts across the other axes; it gets a count
    # and a denominator of its own.
    assert "1 of 4 carry retained analysis" in text
    assert "Stale analysis" in text
    # And the page says out loud that the axes do not sum.
    assert "do not add up to 4" in text


def test_overview_axes_carry_their_own_denominator(tmp_path, monkeypatch) -> None:
    path = tmp_path / "denominator.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    for number in range(1, 4):
        _seed_hand(db, session.id, number, source_type="cv_import", completion_status="partial")
    db.close()

    app = _run(path, monkeypatch)
    captions = "\n".join(item.value for item in app.caption)

    assert captions.count("3 saved hands") >= 2


# ---------------------------------------------------------------------------
# Overview: CV drafts are never confirmed analytics
# ---------------------------------------------------------------------------


def test_overview_never_adds_cv_draft_results_into_the_confirmed_figure(
    tmp_path, monkeypatch
) -> None:
    """A draft's hero result is separated from a reviewed hand's, not summed with it.

    Pre-repair the single 'Recorded result' KPI showed ``-40 BB`` for exactly
    this seed: a reviewed +10 BB hand and an unreviewed CV draft of -50 BB, with
    nothing on the page distinguishing the two contributions.
    """
    path = tmp_path / "drafts.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    _seed_hand(db, session.id, 1, review_status="reviewed", hero_bb_won=10)
    _seed_hand(
        db,
        session.id,
        2,
        source_type="cv_import",
        completion_status="partial",
        review_status="unreviewed",
        hero_bb_won=-50,
    )
    db.close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert "Confirmed result" in text
    assert "+10 BB" in text
    assert "From 1 reviewed hands with a recorded result" in text
    assert "Unconfirmed draft result" in text
    assert "-50 BB" in text
    assert "not study evidence" in text
    # The blended total must appear nowhere on the page -- not as a KPI, and not
    # in the session listing, which is where one Result column used to hide it.
    assert "-40 BB" not in text
    assert "Confirmed covers hands marked reviewed" in text


def test_overview_separates_derived_results_from_observed_ones(tmp_path, monkeypatch) -> None:
    path = tmp_path / "derived.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    _seed_hand(db, session.id, 1, review_status="reviewed", hero_bb_won=7)
    _seed_reconciled_hand(db, session.id, 2)
    db.close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert "Reconciled results" in text
    assert "Derived by the accounting ledger" in text
    assert "1 were recorded as observed" in text


# ---------------------------------------------------------------------------
# Overview: jobs that stopped without succeeding
# ---------------------------------------------------------------------------


def test_overview_failed_job_offers_a_next_action_and_shows_no_percentage(
    tmp_path, monkeypatch
) -> None:
    """A terminal failure gets an outcome and one click to the recording it failed on."""
    path = tmp_path / "failed_job.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    video = _seed_video(db, session.id, "clubwpt-2026-01-01.mp4")
    job = _seed_job(
        db,
        video.id,
        status="failed",
        progress_percent=82,
        error_message="Worker exited with code 1",
    )
    db.close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert "Jobs that stopped without succeeding" in text
    assert "clubwpt-2026-01-01.mp4" in text
    # The progress reading is not an outcome once the job has stopped.
    assert "82%" not in text
    assert "No hands were imported" in text

    open_button = next(
        button
        for button in app.button
        if button.label == "Open this recording in Import"
    )
    open_button.click()
    app.run()

    assert not list(app.exception), [str(item) for item in app.exception]
    assert app.session_state["video_context_id"] == video.id
    assert app.session_state["primary_navigation"] == Page.IMPORT
    assert job.id is not None


def test_overview_job_counts_are_not_taken_from_the_recent_window(
    tmp_path, monkeypatch
) -> None:
    """The table is bounded; the count is not. A count over six rows is not a count."""
    path = tmp_path / "many_jobs.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    video = _seed_video(db, session.id, "alpha.mp4")
    for _ in range(7):
        _seed_job(db, video.id, status="completed")
    for _ in range(2):
        _seed_job(db, video.id, status="cancelled", progress_percent=40)
    db.close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert "9 total · 0 in flight · 7 completed · 2 stopped without succeeding" in text
    assert "9 JOBS" in text
    # Only the recent window is tabulated.
    assert len(app.dataframe[-1].value) == 6


def test_overview_unrecognised_terminal_status_is_treated_as_stopped(
    tmp_path, monkeypatch
) -> None:
    """A status this build does not know falls to the safe side on this surface too.

    The classification comes from the shared predicates rather than a status list
    respelled on the page, so a future terminal state cannot inherit a progress
    bar by being forgotten here.
    """
    path = tmp_path / "unknown_status.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    video = _seed_video(db, session.id, "alpha.mp4")
    job = _seed_job(db, video.id, status="completed", progress_percent=63)
    db._execute("UPDATE processing_jobs SET status = 'evaporated'", ())
    db._commit()
    db.close()
    assert job.id is not None

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert "1 stopped without succeeding" in text
    assert "63%" not in text


# ---------------------------------------------------------------------------
# Overview: storage and database health
# ---------------------------------------------------------------------------


def test_overview_claims_nothing_about_health_until_the_check_is_run(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "health_idle.sqlite3"
    db = _open(path)
    db.create_session(Session(name="Alpha"))
    db.close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert "Storage and database health" in text
    assert "Nothing below is claimed about the store until you run one." in text
    assert "Schema version" in text


def test_overview_health_check_reports_each_check_with_a_word_not_only_a_colour(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "health_run.sqlite3"
    db = _open(path)
    db.create_session(Session(name="Alpha"))
    db.close()

    app = _run(path, monkeypatch)
    next(button for button in app.button if button.label == "Run health check").click()
    app.run()
    assert not list(app.exception), [str(item) for item in app.exception]

    text = _page_text(app)
    assert "Database File" in text
    assert "Foreign Keys" in text or "Foreign Key" in text
    # Every state is spelled, so the badge colour is never the only signal.
    assert "OK" in text
    assert "checks" in text


def test_overview_health_check_hides_secrets_and_absolute_paths(
    tmp_path, monkeypatch
) -> None:
    """A health readout must not leak a credential or the operator's filesystem."""
    path = tmp_path / "health_secret.sqlite3"
    db = _open(path)
    db.create_session(Session(name="Alpha"))
    db.close()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-do-not-leak-abcdefghijklmnop")
    monkeypatch.setenv("POKERTRAINER_DB_PASSWORD", "hunter2-please-hide-me")

    app = _run(path, monkeypatch)
    next(button for button in app.button if button.label == "Run health check").click()
    app.run()
    assert not list(app.exception), [str(item) for item in app.exception]

    text = _page_text(app)
    assert "sk-do-not-leak-abcdefghijklmnop" not in text
    assert "hunter2-please-hide-me" not in text
    # The database is identified, but not by a path that carries the home directory.
    assert str(path) not in text
    assert path.name in text


def test_overview_health_check_is_idempotent_across_reruns(tmp_path, monkeypatch) -> None:
    """Pressing the button twice leaves one report; a plain rerun re-runs no audit."""
    path = tmp_path / "health_idempotent.sqlite3"
    db = _open(path)
    db.create_session(Session(name="Alpha"))
    db.close()

    app = _run(path, monkeypatch)
    button = next(item for item in app.button if item.label == "Run health check")
    button.click()
    app.run()
    first = app.session_state["storage_health_report_overview"]
    assert first is not None

    app.run()
    assert not list(app.exception), [str(item) for item in app.exception]
    # A rerun that did not press the button reuses the stored report verbatim.
    assert app.session_state["storage_health_report_overview"] is first
    assert "Database File" in _page_text(app)


# ---------------------------------------------------------------------------
# Sessions: create and edit from a clean database
# ---------------------------------------------------------------------------


def test_create_session_from_a_clean_database(tmp_path, monkeypatch) -> None:
    """init_db and nothing else: the empty-state form has to actually work."""
    path = tmp_path / "clean.sqlite3"
    _open(path).close()

    app = _run(path, monkeypatch)
    next(
        item for item in app.text_input if item.label == "Custom name (optional)"
    ).set_value("Friday grind")
    next(item for item in app.text_input if item.label == "Platform").set_value("ClubWPT")
    next(item for item in app.text_input if item.label == "Stakes").set_value("1/2 NL")
    next(item for item in app.button if item.label == "Create session").click()
    app.run()
    assert not list(app.exception), [str(item) for item in app.exception]

    verifier = _open(path)
    sessions = verifier.fetch_sessions()
    verifier.close()
    assert [session.name for session in sessions] == ["Friday grind"]
    assert sessions[0].platform == "ClubWPT"
    assert sessions[0].stakes == "1/2 NL"
    assert "Friday grind" in _page_text(app)


def test_sessions_page_on_a_clean_database_offers_creation_rather_than_throwing(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "clean_sessions.sqlite3"
    _open(path).close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    text = _page_text(app)

    assert "Create your first session" in text
    assert any(item.label == "Create session" for item in app.button)


def test_overview_on_a_clean_database_draws_no_empty_data_state_section(
    tmp_path, monkeypatch
) -> None:
    """A section header with nothing under it is a claim the page cannot support."""
    path = tmp_path / "clean_overview.sqlite3"
    _open(path).close()

    app = _run(path, monkeypatch)
    text = _page_text(app)

    assert "Data states" not in text
    assert "No sessions yet" in text
    assert "No processing jobs" in text
    # The health strip still stands, because it describes the store, not the data.
    assert "Storage and database health" in text


def test_edit_session_renames_and_a_blank_name_keeps_the_old_one(
    tmp_path, monkeypatch
) -> None:
    """The rename path, and the refusal to blank a name, from the running page."""
    path = tmp_path / "edit.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Original", platform="Manual"))
    _seed_hand(db, session.id, 1)
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    next(item for item in app.text_input if item.label == "Name").set_value("Renamed")
    next(item for item in app.button if item.label == "Save session changes").click()
    app.run()
    assert not list(app.exception), [str(item) for item in app.exception]

    verifier = _open(path)
    assert verifier.fetch_session(session.id).name == "Renamed"
    verifier.close()

    next(item for item in app.text_input if item.label == "Name").set_value("   ")
    next(item for item in app.button if item.label == "Save session changes").click()
    app.run()
    assert not list(app.exception), [str(item) for item in app.exception]

    verifier = _open(path)
    assert verifier.fetch_session(session.id).name == "Renamed"
    verifier.close()


# ---------------------------------------------------------------------------
# Sessions: the evidence panel
# ---------------------------------------------------------------------------


def test_session_dashboard_shows_issues_stale_coaching_and_unresolved_hands(
    tmp_path, monkeypatch
) -> None:
    """The four questions the session page could not previously answer about itself."""
    path = tmp_path / "session_panel.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    reviewed = _seed_hand(db, session.id, 1, review_status="reviewed")
    flagged = _seed_hand(db, session.id, 2, review_status="reviewed")
    _seed_hand(db, session.id, 3, review_status="unreviewed")
    _seed_issue(db, flagged.id, "Villain stack unreadable")
    _seed_stale_coaching(db, reviewed.id)
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    text = _page_text(app)

    assert "Open work and evidence in this session" in text
    assert "Open debugging issues" in text
    assert "On 1 of 3 hands in this session" in text
    assert "Stale analysis" in text
    assert "Coaching or review a later correction invalidated" in text
    # Unresolved = not reviewed, OR reviewed but still carrying an open issue.
    assert "2 of 3" in text
    assert "Not marked reviewed, or carrying an open issue" in text


def test_session_dashboard_states_how_much_of_the_hero_result_was_derived(
    tmp_path, monkeypatch
) -> None:
    """SessionStats has carried this split all along with nothing reading it."""
    path = tmp_path / "provenance.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    _seed_hand(db, session.id, 1, hero_bb_won=4)
    _seed_reconciled_hand(db, session.id, 2)
    _seed_hand(db, session.id, 3, hero_bb_won=None)
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    text = _page_text(app)

    assert "Result provenance" in text
    assert "1 reconciled" in text
    assert "1 recorded as observed" in text
    assert "1 with no result" in text
    # And the headline card says the same thing rather than implying a measurement.
    assert "1 derived by the ledger, 1 observed" in text


def test_session_dashboard_shows_the_same_data_state_axes_as_overview(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "session_states.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    _seed_hand(db, session.id, 1, source_type="cv_import", completion_status="partial")
    _seed_hand(db, session.id, 2, source_type="corrected_cv", completion_status="complete")
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    text = _page_text(app)

    assert "Partial <strong>1</strong>" in text
    assert "Corrected CV <strong>1</strong>" in text
    assert "hands in this session" in text


# ---------------------------------------------------------------------------
# Sessions: multiple recordings, ordering, empty, large, partially processed
# ---------------------------------------------------------------------------


def test_two_recordings_in_one_session_keep_distinct_hands_and_attribution(
    tmp_path, monkeypatch
) -> None:
    """The multi-recording claim, with two DIFFERENT jobs landing into one session.

    Every existing dedup test uses one video and one job, so nothing proved that
    two recordings in one session neither collide on hand numbers nor cross-
    attribute their hands.
    """
    path = tmp_path / "two_videos.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    first_video = _seed_video(db, session.id, "part-one.mp4")
    second_video = _seed_video(db, session.id, "part-two.mp4")
    first_job = _seed_job(db, first_video.id)
    second_job = _seed_job(db, second_video.id)
    for number, job in ((1, first_job), (2, first_job), (3, second_job)):
        evidence = dict(CLEAN_EVIDENCE)
        evidence[CV_TIMELINE_IDENTITY_KEY] = {
            "job_id": job.id,
            "timeline_hand_number": number,
        }
        _seed_hand(
            db,
            session.id,
            number,
            source_type="cv_import",
            completion_status="complete",
            evidence=evidence,
        )
    hands = db.fetch_hands_by_session(session.id)
    numbers = [hand.hand_number for hand in hands]
    assert numbers == [1, 2, 3]
    assert len(set(hand.id for hand in hands)) == 3
    assert [hand.hand_number for hand in hands_committed_by_job(db, first_job)] == [1, 2]
    assert [hand.hand_number for hand in hands_committed_by_job(db, second_job)] == [3]
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    text = _page_text(app)

    assert "part-one.mp4" in text
    assert "part-two.mp4" in text
    assert "2 VIDEOS" in text


def test_session_hand_order_is_stable_across_reruns(tmp_path, monkeypatch) -> None:
    path = tmp_path / "ordering.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    for number in (3, 1, 2):
        _seed_hand(db, session.id, number)
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    first_pass = [
        item.value for item in app.markdown if item.value.startswith("**Hand #")
    ]
    app.run()
    second_pass = [
        item.value for item in app.markdown if item.value.startswith("**Hand #")
    ]

    assert first_pass == second_pass
    assert [value.split("#")[1][0] for value in first_pass] == ["1", "2", "3"]


def test_empty_session_renders_an_empty_state_and_can_still_be_deleted(
    tmp_path, monkeypatch
) -> None:
    """The session most likely created by mistake was the one that could not be removed."""
    path = tmp_path / "empty_session.sqlite3"
    db = _open(path)
    db.create_session(Session(name="Empty"))
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    text = _page_text(app)

    assert "No hands recorded yet" in text
    assert "Danger zone: delete this session" in text

    next(
        item
        for item in app.checkbox
        if item.label.startswith("I understand this permanently deletes")
    ).set_value(True)
    app.run()
    next(item for item in app.button if item.label == "Delete session").click()
    app.run()
    assert not list(app.exception), [str(item) for item in app.exception]

    verifier = _open(path)
    assert verifier.fetch_sessions() == []
    verifier.close()


def test_large_session_paginates_and_states_its_denominator(tmp_path, monkeypatch) -> None:
    path = tmp_path / "large.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    for number in range(1, 61):
        _seed_hand(db, session.id, number, review_status="reviewed" if number <= 10 else "unreviewed")
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    text = _page_text(app)

    assert "60 hands · showing 1–15" in text
    assert "10 of 60" in text
    assert "50 of 60" in text  # unresolved hands
    rendered = [item.value for item in app.markdown if item.value.startswith("**Hand #")]
    assert len(rendered) == 15


def test_partially_processed_session_renders_its_partial_and_uncertain_states(
    tmp_path, monkeypatch
) -> None:
    """A session mid-reconstruction reports what it has, labelled as unfinished."""
    path = tmp_path / "partial.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    video = _seed_video(db, session.id, "half.mp4")
    _seed_job(db, video.id, status="running", progress_percent=45, live=True)
    _seed_hand(db, session.id, 1, source_type="cv_import", completion_status="complete")
    _seed_hand(db, session.id, 2, source_type="cv_import", completion_status="partial")
    _seed_hand(db, session.id, 3, source_type="cv_import", completion_status="uncertain")
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    text = _page_text(app)

    assert "Partial <strong>1</strong>" in text
    assert "Uncertain <strong>1</strong>" in text
    assert "3 of 3" in text  # every hand still unresolved

    overview = _run(path, monkeypatch)
    overview_text = _page_text(overview)
    assert "1 in flight" in overview_text


def test_new_metric_strips_stack_at_narrow_widths(tmp_path, monkeypatch) -> None:
    """Every count strip added here is covered by the responsive rules, not just the old ones.

    AppTest has no viewport, so real layout is not verifiable here. What IS
    verifiable is the pairing that the layout depends on: each multi-column strip
    is wrapped in a keyed container, and the stylesheet carries a stacking rule
    for that key at both breakpoints. A four-column strip with no rule is the
    failure this catches -- counts squeezed to a few characters on a phone, which
    is where a denominator gets lost first.
    """
    path = tmp_path / "narrow.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Alpha"))
    _seed_hand(db, session.id, 1, source_type="cv_import", completion_status="partial")
    db.close()

    theme = str(_THEME_CSS)
    source = Path(APP_PATH).read_text(encoding="utf-8")
    for key in ("data_state_axes", "session_evidence", "storage_health"):
        assert f'st.container(key=f"{key}_' in source or f'st.container(key="{key}"' in source, key
        assert f"st-key-{key}" in theme, key
    wide = theme.split("@media (max-width: 900px)")[1].split("@media (max-width: 720px)")[0]
    narrow = theme.split("@media (max-width: 720px)")[1]
    for key in ("data_state_axes", "session_evidence", "storage_health"):
        assert f"st-key-{key}" in wide, (key, "900px")
        assert f"st-key-{key}" in narrow, (key, "720px")

    # And the page still renders with the containers in place.
    assert "Reconstruction completeness" in _page_text(_run(path, monkeypatch))
    assert "Open work and evidence" in _page_text(_run(path, monkeypatch, Page.SESSIONS))


def test_manual_hand_entry_tab_cannot_declare_a_review_state(tmp_path, monkeypatch) -> None:
    """The Add hands tab offers no control that could land a hand pre-approved."""
    path = tmp_path / "manual_entry.sqlite3"
    db = _open(path)
    db.create_session(Session(name="Alpha"))
    db.close()

    app = _run(path, monkeypatch, Page.SESSIONS)
    labels = [str(item.label) for item in app.selectbox] + [
        str(item.label) for item in app.radio
    ]

    assert any("Entry mode" in label for label in labels)
    assert not any("Review status" in label for label in labels)
    assert not any("Source type" in label for label in labels)
