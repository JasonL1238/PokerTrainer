from __future__ import annotations

import streamlit as st
from streamlit.testing.v1 import AppTest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    CoachingResponse,
    Hand,
    HandPlayer,
    Session,
)
from poker_tracker.ui.navigation import Page
from tests.test_study_readiness_ui import (
    _open_fix_tool,
    _run_validation_editors,
)


def test_validation_fact_correction_updates_database_and_audit(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "study-correction.sqlite3"
    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Study correction"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
        )
    )
    db.create_hand_player(
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
    db.create_coaching_response(
        CoachingResponse(
            provider_name="fixture",
            model_name="deterministic",
            raw_prompt="post-session completed-hand review",
            raw_response="saved coaching",
            review_type="hand",
            hand_id=hand.id,
            session_id=session.id,
        )
    )
    hand_id = hand.id
    db.close()

    app = _open_fix_tool(
        _run_validation_editors(path, monkeypatch, hand_id),
        "Cards, board, or pot",
    )
    assert not list(app.exception)

    next(item for item in app.text_input if item.label == "Board cards").set_value(
        "Qd 7s 6c"
    )
    next(
        item
        for item in app.text_input
        if item.label == "Why is this correction needed?"
    ).set_value("Video frame shows 6c.")
    next(
        button for button in app.button if button.label == "Save corrected facts"
    ).click()
    app.run()

    assert not list(app.exception)
    verifier = PokerDatabase(path)
    verifier.init_db()
    saved = verifier.fetch_hand(hand_id)
    corrections = verifier.fetch_hand_corrections(hand_id)
    assert saved is not None
    assert saved.board_cards == "Qd 7s 6c"
    assert saved.source_type == "corrected_cv"
    assert len(corrections) == 1
    assert corrections[0].notes == "Video frame shows 6c."
    assert verifier.fetch_coaching_reviews_by_hand(hand_id)[0].is_stale is True
    verifier.close()
    st.cache_resource.clear()


def test_study_workspace_is_study_only_for_approved_hands(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "study-only.sqlite3"
    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Approved"))
    db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="manual",
            completion_status="not_applicable",
            review_status="reviewed",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
        )
    )
    db.close()

    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()

    app = AppTest.from_file("app.py", default_timeout=20).run()
    next(item for item in app.radio if "Study" in list(item.options)).set_value(
        Page.STUDY
    )
    app.run()

    assert not list(app.exception)
    rendered = "\n".join(str(item.value) for item in app.markdown)
    assert "#### Replay, then analyze" in rendered
    assert "TexasSolver postflop analysis" in rendered
    assert not any(
        "Looks good — Approve" in list(getattr(item, "options", []))
        for item in app.radio
    )
    st.cache_resource.clear()


def test_validation_can_save_hand_to_future_debugging_queue(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "study-issue.sqlite3"
    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Issue queue"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=4,
            source_type="cv_import",
            review_status="unreviewed",
        )
    )
    db.create_hand_player(
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
    hand_id = hand.id
    db.close()

    app = _run_validation_editors(path, monkeypatch, hand_id, frames_validated=False)
    assert not list(app.exception)

    next(item for item in app.multiselect if item.label == "What looks wrong?").set_value(
        ["cards", "actions"]
    )
    next(item for item in app.text_area if item.label == "What did you notice?").set_value(
        "The board and flop action do not match the recording."
    )
    next(
        button for button in app.button if button.label == "Save issue for later"
    ).click()
    app.run()

    assert not list(app.exception)
    verifier = PokerDatabase(path)
    verifier.init_db()
    issues = verifier.fetch_hand_issues(hand_id=hand_id)
    assert len(issues) == 1
    assert issues[0].issue_types == ["cards", "actions"]
    assert "flop action" in issues[0].description
    assert verifier.fetch_hand(hand_id).review_status == "needs_correction"
    verifier.close()
    st.cache_resource.clear()
