"""The clean-database acceptance script, plus Insights, Import and Settings proof.

Two kinds of test live here and they answer different questions.

The acceptance script is the phase's exit gate: from ``init_db`` forward, every
one of the seven surfaces has to render, first with nothing in the store and
then with a corpus deliberately built out of the states this product has to keep
apart -- a manual reviewed hand, an unconfirmed CV draft, a corrected CV hand
with an open issue, a hand whose result exists only as a ledger derivation, a
stale coaching review, and a job that stopped without succeeding. On the empty
pass the surfaces must render an empty state rather than a number; on the seeded
pass every claim asserted is a claim about seeded rows.

The rest verify one rule each. A figure without its denominator, a CV draft
counted as confirmed data, a stale conclusion still voting, and a destructive
control with no rollback point are the same defect wearing four costumes, and
each has its own test below.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.backup import backups_dir_for, find_snapshots
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
from poker_tracker.ui.navigation import Page
from tests.conftest import attest_declared_assumptions

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

ALL_PAGES: tuple[Page, ...] = (
    Page.OVERVIEW,
    Page.SESSIONS,
    Page.HANDS,
    Page.STUDY,
    Page.INSIGHTS,
    Page.IMPORT,
    Page.SETTINGS,
)

CLEAN_EVIDENCE: dict[str, object] = {
    "evidence_version": 1,
    "partial_start": False,
    "partial_end": False,
    "terminal_event": "showdown",
    "first_source_timestamp_s": 1.0,
    "last_source_timestamp_s": 9.0,
    "boundary_confidence": 0.9,
    "source_frames": ["frames/start.png", "frames/end.png"],
    "layout_profile": "1272x896",
    "layout_supported": True,
    "table_size": 6,
    "pipeline_version": "two-model-v7",
    "model_versions": {"detector": "v7", "classifier": "cards-v1"},
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _drop_cached_database() -> Iterator[None]:
    """Never hand the next test a connection to this test's temporary database.

    ``get_database`` is ``@st.cache_resource`` keyed on the function rather than
    on the path, so a cached handle outlives the run that created it.
    """
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def _open(path: Path) -> PokerDatabase:
    db = PokerDatabase(path)
    db.init_db()
    return db


def _configure(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()


def _run(path: Path, monkeypatch: pytest.MonkeyPatch, page: Page | None = None) -> AppTest:
    _configure(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    assert not list(app.exception), [str(item) for item in app.exception]
    if page is not None:
        app.radio[0].set_value(page)
        app.run()
        assert not list(app.exception), [str(item) for item in app.exception]
    return app


def _page_text(app: AppTest) -> str:
    """Every rendered string on the page, as one haystack.

    Captions and expander labels are included on purpose: the failures this phase
    is about are almost always a missing qualifier rather than a missing headline.
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
    parts.extend(str(item.label) for item in app.radio)
    for frame in app.dataframe:
        parts.append(str(frame.value))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


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


def _seed_reconciled_hand(db: PokerDatabase, session_id: int, hand_number: int) -> Hand:
    """A hand whose hero result exists only because the ledger derived it.

    ``hero_bb_won`` is NULL on purpose so the derivation has something to publish
    and the result-basis split has a ``reconciled`` member to report.
    """
    hand = _seed_hand(
        db, session_id, hand_number, source_type="manual", hero_bb_won=None
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
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="settled", rake_rate=0.0)
    )
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


def _seed_corpus(path: Path) -> dict[str, object]:
    """One store holding every data state the product must keep apart."""
    db = _open(path)
    empty = db.create_session(Session(name="Empty session", date_played="2026-01-01"))
    studied = db.create_session(Session(name="Studied", date_played="2026-01-02"))
    drafts = db.create_session(Session(name="Drafts", date_played="2026-01-03"))

    reviewed = [
        _seed_hand(
            db,
            studied.id,
            number,
            source_type="manual",
            review_status="reviewed",
            hero_bb_won=value,
            tags=["BIG_POT"] if number == 1 else ["RIVER_DECISION"],
        )
        for number, value in ((1, 12.0), (2, -4.0))
    ]
    derived = _seed_reconciled_hand(db, studied.id, 3)
    draft = _seed_hand(
        db,
        drafts.id,
        1,
        source_type="cv_import",
        completion_status="partial",
        hero_bb_won=999.0,
        tags=["BIG_POT"],
    )
    corrected = _seed_hand(
        db,
        drafts.id,
        2,
        source_type="corrected_cv",
        completion_status="complete",
        hero_bb_won=-30.0,
        tags=["BIG_POT"],
        evidence={
            **CLEAN_EVIDENCE,
            CV_TIMELINE_IDENTITY_KEY: {"job_id": 1, "timeline_hand_number": 2},
        },
    )
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=corrected.id,
            issue_types=["cards"],
            description="Turn card misread as Th",
        )
    )
    db.create_coaching_response(
        CoachingResponse(
            hand_id=reviewed[0].id,
            review_type="hand",
            provider_name="test",
            model_name="test",
            raw_prompt="prompt",
            raw_response="response",
            parsed_sections={"Study Lesson": "Stop bluffing the river out of position."},
            is_stale=True,
            stale_reason="Hand facts were corrected after this review.",
        )
    )
    db.create_coaching_response(
        CoachingResponse(
            hand_id=reviewed[1].id,
            review_type="hand",
            provider_name="test",
            model_name="test",
            raw_prompt="prompt",
            raw_response="response",
            parsed_sections={"Study Lesson": "Size down on paired boards."},
            is_stale=False,
        )
    )
    video = db.create_video(
        VideoRecord(
            session_id=drafts.id,
            original_filename="clubwpt_session_07.mov",
            stored_path="/videos/clubwpt_session_07.mov",
            file_size_bytes=2048,
            content_sha256="sha-07",
        )
    )
    completed_job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id,
            job_type="cv_reconstruction",
            status="completed",
            progress_percent=100.0,
            message="Reconstruction finished.",
            created_at=utc_now(),
            completed_at=utc_now(),
        )
    )
    failed_job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id,
            job_type="cv_reconstruction",
            status="failed",
            progress_percent=82.0,
            message="Worker exited",
            error_message="Pipeline crashed at frame 4102",
            created_at=utc_now(),
            completed_at=utc_now(),
        )
    )
    ids = {
        "empty_session": empty.id,
        "studied_session": studied.id,
        "drafts_session": drafts.id,
        "reviewed": [hand.id for hand in reviewed],
        "derived": derived.id,
        "draft": draft.id,
        "corrected": corrected.id,
        "issue": issue.id,
        "video": video.id,
        "completed_job": completed_job.id,
        "failed_job": failed_job.id,
    }
    db.close()
    return ids


# ---------------------------------------------------------------------------
# Exit gate: the clean-database end-to-end acceptance script
# ---------------------------------------------------------------------------


def test_clean_database_renders_every_surface_without_claiming_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From init_db forward, each surface renders an empty state, not a number.

    The dangerous failure on an empty store is not a traceback -- that is loud.
    It is a page that computes a confident 0% or a 0.00 bb/100 over nothing and
    presents it in the same type as a real figure.
    """
    path = tmp_path / "clean.sqlite3"
    _open(path).close()

    for page in ALL_PAGES:
        app = _run(path, monkeypatch, page)
        text = _page_text(app)
        assert text.strip(), f"{page} rendered nothing at all"
        assert "bb/100" not in text or "Not enough evidence" in text, (
            f"{page} printed a rate on an empty database"
        )
        st.cache_resource.clear()

    insights = _run(path, monkeypatch, Page.INSIGHTS)
    assert "Insights appear after completed hands" in _page_text(insights)


def test_seeded_database_renders_every_surface_with_data_backed_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every surface renders over the full corpus, and says something true about it.

    One data-derived assertion per surface. A smoke test that only checks a
    heading passes just as happily on a page whose numbers are wrong, which is
    the failure mode this phase exists to close.
    """
    path = tmp_path / "corpus.sqlite3"
    ids = _seed_corpus(path)

    rendered: dict[Page, str] = {}
    for page in ALL_PAGES:
        app = _run(path, monkeypatch, page)
        rendered[page] = _page_text(app)
        st.cache_resource.clear()

    # Overview: the corpus holds 3 sessions and 5 hands, and a job that stopped.
    assert "5 hands" in rendered[Page.OVERVIEW] or "<div class=\"pt-kpi-value\">5</div>" in rendered[Page.OVERVIEW]
    assert "stopped without succeeding" in rendered[Page.OVERVIEW]

    # Sessions: the session library names every seeded session.
    assert "Studied" in rendered[Page.SESSIONS]

    # Hands: the open issue on the corrected hand is visible without entering Study.
    assert "open issue" in rendered[Page.HANDS]

    # Study: the workspace read the store rather than rendering a static shell.
    assert any(
        name in rendered[Page.STUDY]
        for name in ("Empty session", "Studied", "Drafts")
    )

    # Insights: the default population is the confirmed one, with its denominator.
    assert "of 5 saved hands" in rendered[Page.INSIGHTS]

    # Import: the post-session boundary is stated on the page that ingests video.
    assert "never captures a live table" in rendered[Page.IMPORT]

    # Settings: the store's own schema version, not a hardcoded string.
    assert str(db_module.SCHEMA_VERSION) in rendered[Page.SETTINGS]

    db = _open(path)
    assert len(db.fetch_all_hands()) == 5
    assert db.fetch_hand(ids["draft"]).source_type == "cv_import"
    db.close()


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


def test_insights_default_population_excludes_unconfirmed_evidence_and_counts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CV draft must not enter the confirmed population, and must be accounted for.

    The draft carries ``hero_bb_won=999``. Before the population layer that
    number went straight into the headline result beside two reviewed hands.
    """
    path = tmp_path / "population.sqlite3"
    _seed_corpus(path)
    app = _run(path, monkeypatch, Page.INSIGHTS)
    text = _page_text(app)

    assert "Confirmed hands" in text
    assert "2 of 5 saved hands" in text
    assert "+999" not in text and "999.00" not in text
    # The four hands left out are reported by reason rather than dropped.
    assert "Not marked reviewed" in text


def test_insights_metrics_agree_with_a_recomputation_from_the_hand_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recompute from the raw rows; the surface has to print the same figures.

    This is the check the phase asks for by name. A population filter that
    silently drops a hand shows up here as a disagreement, rather than as a
    slightly different number nobody compares against anything.
    """
    path = tmp_path / "verify.sqlite3"
    _seed_corpus(path)

    db = _open(path)
    expected = [
        hand.hero_bb_won
        for hand in db.fetch_all_hands()
        if hand.review_status == "reviewed"
        and hand.study_inclusion != "skip"
        and hand.completion_status in ("complete", "not_applicable")
        and hand.hero_bb_won is not None
    ]
    db.close()
    assert sorted(expected) == [-4.0, 12.0]
    net = sum(expected)

    app = _run(path, monkeypatch, Page.INSIGHTS)
    text = _page_text(app)
    assert f"{net:+,.2f} BB" in text
    # The mean is below the rate floor, so the product refuses to print it.
    assert "Not enough evidence" in text


def test_insights_refuses_a_win_rate_the_sample_cannot_carry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two hands is not a win rate, and the surface has to say which it is."""
    path = tmp_path / "smallsample.sqlite3"
    _seed_corpus(path)
    app = _run(path, monkeypatch, Page.INSIGHTS)
    text = _page_text(app)

    assert "Win rate" in text
    assert "Not enough evidence" in text
    assert "30-hand floor" in text
    # No bb/100 figure is printed anywhere at this sample size.
    assert "bb/100" not in text.replace("Win rate", "")


def test_insights_themes_carry_a_denominator_and_drop_stale_coaching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A theme on 1 of 2 hands must not render like a theme on 1 of 200.

    The stale coaching lesson seeded on the first reviewed hand is a conclusion a
    correction already invalidated. It must not vote, and the page must say how
    much was excluded rather than quietly shrinking.
    """
    path = tmp_path / "themes.sqlite3"
    _seed_corpus(path)
    app = _run(path, monkeypatch, Page.INSIGHTS)
    text = _page_text(app)

    # Asserted on the bar's own markup, not on the string appearing somewhere on
    # the page: a KPI elsewhere also reads "1 of 2", and a denominator that is
    # only present by coincidence is not a denominator.
    assert 'aria-label="Size down on paired boards: 1 of 2"' in text
    assert "Stop bluffing the river out of position" not in text
    assert "stale coaching review(s)" in text
    assert "removing 1 theme(s) that no current evidence supports" in text


def test_insights_all_saved_population_names_what_it_mixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Widening the population must widen the caveat, not just the number."""
    path = tmp_path / "allsaved.sqlite3"
    _seed_corpus(path)
    _configure(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    app.radio[0].set_value(Page.INSIGHTS)
    app.run()
    population = next(item for item in app.radio if item.key == "insights_population")
    population.set_value("all_saved")
    app.run()
    assert not list(app.exception), [str(item) for item in app.exception]
    text = _page_text(app)

    assert "5 of 5 saved hands" in text
    assert "mixes confirmed records with unconfirmed CV drafts" in text
    assert "CV draft" in text


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_page_states_the_post_session_only_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one non-negotiable constraint, stated on the page that could break it.

    Asserted on an EMPTY database as well, because the empty-state branch returns
    before the ingest UI and used to skip every statement of scope with it.
    """
    empty = tmp_path / "empty.sqlite3"
    _open(empty).close()
    text = _page_text(_run(empty, monkeypatch, Page.IMPORT))
    assert "Completed sessions only" in text
    assert "never captures a live table" in text
    assert "never advises on a hand in progress" in text
    st.cache_resource.clear()

    seeded = tmp_path / "seeded.sqlite3"
    _seed_corpus(seeded)
    text = _page_text(_run(seeded, monkeypatch, Page.IMPORT))
    assert "Completed sessions only" in text


def _run_snippet(path: Path, monkeypatch: pytest.MonkeyPatch, body: list[str]) -> AppTest:
    """Drive one app.py renderer directly, the way the validation-panel suite does."""
    _configure(path, monkeypatch)
    script = path.parent / f"_snippet_{abs(hash(tuple(body))) % 10**8}.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "from poker_tracker.persistence.db import PokerDatabase",
                "import app as app_module",
                f"db = PokerDatabase(r'{path}')",
                "db.init_db()",
                *body,
            ]
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=90).run()
    assert not list(app.exception), [str(item) for item in app.exception]
    return app


def test_an_unsupported_table_layout_is_warned_not_left_to_be_inferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The geometry that breaks every downstream read must be named where it matters."""
    path = tmp_path / "layout.sqlite3"
    _open(path).close()
    timeline = {"metadata": {"layout_profile": "640x448-unsupported"}}
    app = _run_snippet(
        path,
        monkeypatch,
        [
            f"timeline = json.loads(r'''{json.dumps(timeline)}''')",
            "app_module._render_timeline_layout_support(timeline)",
        ],
    )
    text = _page_text(app)
    assert "640x448-unsupported" in text
    assert "outside the calibrated geometries" in text
    assert "drafts" in text


def test_a_corrected_cv_hand_still_names_the_recording_it_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The video link has to survive a correction, and has to be visible.

    ``update_hand_facts`` rewrites completion evidence and flips ``source_type``
    to ``corrected_cv``; the retained CV timeline identity is what carries the
    link across that write. Until now nothing rendered it, so the link existed
    only as a navigation target.
    """
    path = tmp_path / "linkage.sqlite3"
    db = _open(path)
    session = db.create_session(Session(name="Recorded", date_played="2026-02-01"))
    video = db.create_video(
        VideoRecord(
            session_id=session.id,
            original_filename="clubwpt_session_09.mov",
            stored_path="/videos/clubwpt_session_09.mov",
            file_size_bytes=4096,
            content_sha256="sha-09",
        )
    )
    job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id,
            job_type="cv_reconstruction",
            status="completed",
            progress_percent=100.0,
        )
    )
    hand = _seed_hand(
        db,
        session.id,
        1,
        source_type="cv_import",
        completion_status="complete",
        evidence={
            **CLEAN_EVIDENCE,
            CV_TIMELINE_IDENTITY_KEY: {
                "job_id": job.id,
                "timeline_hand_number": 1,
            },
        },
    )
    db.update_hand_facts(
        hand.model_copy(update={"hero_cards": "Kd Kc"}), correction_notes="Hero cards"
    )
    corrected = db.fetch_hand(hand.id)
    assert corrected.source_type == "corrected_cv"
    db.close()

    app = _run_snippet(
        path,
        monkeypatch,
        [
            f"hand = db.fetch_hand({hand.id})",
            "app_module.render_hand_source_recording(db, hand)",
        ],
    )
    text = _page_text(app)
    assert "clubwpt_session_09.mov" in text
    assert f"job #{job.id}" in text


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_reports_paths_and_configuration_without_any_env_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guidance names the variable and whether it is set. Never what it is set to."""
    path = tmp_path / "settings.sqlite3"
    _seed_corpus(path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-do-not-leak-abcdef123456")
    monkeypatch.setenv("TEXAS_SOLVER_PATH", "/opt/private-build/console_solver")

    text = _page_text(_run(path, monkeypatch, Page.SETTINGS))

    assert "POKER_DB_PATH" in text
    assert "ANTHROPIC_API_KEY" in text
    assert "sk-test-do-not-leak" not in text
    assert str(path) not in text, "the absolute database path identifies the operator"
    assert "Storage and database health" in text
    assert "Run health check" in text

    # The table's own data, so the guarantee is checked at the source rather than
    # inferred from one page's rendering. The Solver tab separately echoes the
    # TEXAS_SOLVER_PATH it could not find, which is that tab's own diagnostic.
    from poker_tracker.maintenance.diagnostics import environment_variable_report

    for entry in environment_variable_report():
        assert set(entry) == {"name", "purpose", "configured"}
        assert isinstance(entry["configured"], bool)
        assert "sk-test-do-not-leak" not in json.dumps(entry)
        assert "/opt/private-build" not in json.dumps(entry)


def test_settings_shows_model_hashes_and_the_layouts_this_build_supports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reconstruction verdict is only reproducible if its weights are identified."""
    path = tmp_path / "identity.sqlite3"
    _seed_corpus(path)
    text = _page_text(_run(path, monkeypatch, Page.SETTINGS))

    assert "Reconstruction build identity" in text
    assert "SHA-256" in text
    assert "Supported table layouts" in text
    # The corpus reconstructed at a calibrated geometry; the readout says so.
    assert "1272x896" in text


def test_diagnostics_bundle_carries_no_secret_and_no_operator_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Built through the same helper the button uses, then searched for the secret."""
    from poker_tracker.maintenance.diagnostics import (
        build_diagnostics_payload,
        serialize_diagnostics,
    )

    path = tmp_path / "bundle.sqlite3"
    _seed_corpus(path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-do-not-leak-abcdef123456")
    monkeypatch.setenv("POKER_DB_PATH", str(path))

    db = _open(path)
    payload = build_diagnostics_payload(
        db, repo_root=Path(APP_PATH).parent, database_path=path
    )
    blob = serialize_diagnostics(payload)
    db.close()

    assert b"sk-test-do-not-leak" not in blob
    assert str(path).encode() not in blob
    assert b"clubwpt_session_07.mov" not in blob
    assert b"Turn card misread" not in blob
    assert b"Stop bluffing the river" not in blob

    decoded = json.loads(blob)
    assert decoded["store"]["hands"] == 5
    assert decoded["store"]["open_hand_issues"] == 1
    assert decoded["schema"]["expected_version"] == db_module.SCHEMA_VERSION
    assert any(
        entry["name"] == "ANTHROPIC_API_KEY" and entry["configured"] is True
        for entry in decoded["environment_variables"]
    )


def test_diagnostics_are_scrubbed_before_serialization_not_after() -> None:
    """The order is the mechanism, so it gets its own test rather than an assumption.

    A credential written as JSON inside a collected string field -- a health-check
    detail carrying a config fragment, say -- survives a pass made over the
    finished document, because ``json.dumps`` escapes the quotes the assignment
    pattern matches on. Scrubbing the structure first sidesteps the encoder. See
    ``redact_structure``'s own docstring.
    """
    from poker_tracker.maintenance.diagnostics import serialize_diagnostics

    blob = serialize_diagnostics({"health": {"detail": '{"api_key": "hunter2secret"}'}})

    assert b"hunter2secret" not in blob
    assert b"<redacted>" in blob


def test_diagnostics_bundle_is_offered_only_after_it_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two presses of the build button are two equivalent bundles and no other change.

    Bundling shells out to git and hashes 30 MB of weights, so it must not run on
    a page repaint; and pressing it twice must not accumulate anything.
    """
    path = tmp_path / "download.sqlite3"
    _seed_corpus(path)
    _configure(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    app.radio[0].set_value(Page.SETTINGS)
    app.run()
    assert not [item for item in app.button if "Download diagnostics" in item.label]

    build = next(item for item in app.button if item.key == "settings_build_diagnostics")
    build.click().run()
    assert not list(app.exception), [str(item) for item in app.exception]
    first = app.session_state["settings_diagnostics_bundle"]
    assert first

    build = next(item for item in app.button if item.key == "settings_build_diagnostics")
    build.click().run()
    assert not list(app.exception), [str(item) for item in app.exception]
    second = app.session_state["settings_diagnostics_bundle"]
    assert json.loads(second)["store"] == json.loads(first)["store"]


def test_health_check_runs_only_when_asked_and_survives_a_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read that costs a full audit must be a request, and a rerun must redraw it.

    The audit opens the database, integrity-checks it, walks every recorded
    artifact and restores each retained snapshot. Paying that on a page repaint
    would make Settings unusable; re-running it on a repaint would also mean the
    report on screen was never the one the operator asked for.
    """
    import poker_tracker.maintenance.data_health as health_module

    calls: list[int] = []
    real = health_module.audit_data_health

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(health_module, "audit_data_health", counted)

    path = tmp_path / "health.sqlite3"
    _seed_corpus(path)
    _configure(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    app.radio[0].set_value(Page.SETTINGS)
    app.run()
    assert calls == []

    run_check = next(item for item in app.button if item.key == "settings_run_health_check")
    run_check.click().run()
    assert not list(app.exception), [str(item) for item in app.exception]
    assert len(calls) == 1
    assert "checks" in _page_text(app)

    app.run()
    assert len(calls) == 1, "a rerun re-audited the store instead of redrawing"
    assert "checks" in _page_text(app)


def test_deleting_a_session_leaves_a_restorable_snapshot_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Destructive means recoverable here, or it does not happen.

    Asserted on the snapshot's CONTENTS, not only its existence: a rollback point
    that does not still hold the deleted hands is a file, not a rollback point.
    """
    path = tmp_path / "delete.sqlite3"
    ids = _seed_corpus(path)
    backups = backups_dir_for(path)
    assert find_snapshots(backups, purpose="predelete") == []

    _configure(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()
    app.radio[0].set_value(Page.SESSIONS)
    app.run()
    session_id = ids["studied_session"]
    # Select the session whose dashboard carries the danger zone.
    selector = next(
        (item for item in app.button if item.key == f"session_context_{session_id}"),
        None,
    )
    if selector is not None:
        selector.click().run()
    confirm = next(
        item
        for item in app.checkbox
        if item.key == f"confirm_delete_session_{session_id}"
    )
    confirm.set_value(True).run()
    delete = next(
        item for item in app.button if item.key == f"delete_session_{session_id}"
    )
    delete.click().run()
    assert not list(app.exception), [str(item) for item in app.exception]

    snapshots = find_snapshots(backups, purpose="predelete", scope=f"session{session_id}")
    assert snapshots, "a session was deleted with no rollback point"

    db = _open(path)
    assert db.fetch_session(session_id) is None
    db.close()

    restored = sqlite3.connect(str(snapshots[0]))
    try:
        surviving = restored.execute(
            "SELECT COUNT(*) FROM hands WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    finally:
        restored.close()
    assert surviving == 3, "the snapshot does not hold the hands it was taken to protect"


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------


def test_frequency_bars_render_the_denominator_they_are_a_share_of() -> None:
    """1 of 4 and 1 of 400 must not draw the same bar with the same number."""
    from poker_tracker.ui.components import frequency_bars_html

    small = frequency_bars_html([("Big Pot", 1)], denominator=4)
    large = frequency_bars_html([("Big Pot", 1)], denominator=400)

    assert "1 of 4" in small
    assert "1 of 400" in large
    assert small != large
    assert 'aria-label="Big Pot: 1 of 400"' in large


def test_the_history_deletions_snapshot_through_one_helper() -> None:
    """One writer for rollback points, and one call site per deletion that uses it.

    Scoped to what it actually covers: the four deletions that remove study
    history a person produced — a session, a hand, a recording, an ROI
    calibration. Narrower deletions in app.py (an action, a solver run, a solver
    range) are corrections within a hand rather than removals of the hand, and are
    deliberately not in this set; a future decision to cover them belongs in the
    list below.

    ``db.delete_video(`` joined the list when the session's recording panel began
    offering a delete. Before that there was one control, on Import, and it wrote
    no snapshot at all — so the second caller is what forced the rule to be
    honoured rather than merely stated.

    A source scan rather than a behavioural test, because the risk is a NEW
    control that forgets the snapshot and no behavioural test can fail for a
    control nobody has written yet.
    """
    source = Path(APP_PATH).read_text(encoding="utf-8")
    assert source.count("def snapshot_before_destructive(") == 1
    for writer in (
        "db.delete_session(",
        "db.delete_roi_profile(",
        "db.delete_hand(",
        "db.delete_video(",
    ):
        assert source.count(writer) == 1, f"{writer} has more than one call site"
    # Each of the four carries its own scoped snapshot request.
    for marker in (
        "session{session.id}",
        "hand{hand_id}",
        "roi{profile.id}",
        "video{video_id}",
    ):
        assert f'scope=f"{marker}"' in source


def test_no_hand_is_deleted_without_a_snapshot_in_hand() -> None:
    """The batch delete must not become a way to reach db.delete_hand unsnapshotted.

    Splitting the per-hand writer so a batch could take ONE snapshot moved
    ``db.delete_hand`` behind a private helper. The call-site count above still
    passes either way, so it can no longer be what guarantees a rollback point
    exists — the helper's required ``snapshot`` keyword is, and this pins it.
    """
    source = Path(APP_PATH).read_text(encoding="utf-8")
    assert source.count("def _remove_hand_and_artifacts(") == 1
    signature = source.split("def _remove_hand_and_artifacts(")[1].split(")")[0]
    assert "*, snapshot: Path" in signature, signature
    # Every caller passes one; none may construct the helper's work itself.
    assert source.count("_remove_hand_and_artifacts(") == 3, (
        "expected the definition and exactly two callers: the single-hand writer "
        "and the batch writer"
    )
