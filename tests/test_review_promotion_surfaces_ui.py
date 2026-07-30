"""The review-promotion surfaces that ``test_study_readiness_ui`` does not reach.

``test_study_readiness_ui`` covers the Study workspace, the Study coach button,
and the solver-explanation button. The remaining ways a hand can acquire
``review_status = 'reviewed'`` are the saved-hands expander, the Settings ->
Coaching tab, and the manual Add-hand form. Each is driven here through a real
Streamlit run against a real SQLite file.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import persist_reconciliation
from poker_tracker.ui.navigation import Page
from tests.conftest import attest_declared_assumptions

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

CLEAN_EVIDENCE: dict[str, object] = {
    "evidence_version": 1,
    "partial_start": False,
    "partial_end": False,
    "terminal_event": "showdown",
    "boundary_confidence": 0.9,
    "layout_profile": "clubwpt-6max",
    "layout_supported": True,
    "table_size": 6,
    "pipeline_version": "two-model-v7",
}


def _seed(
    path: Path,
    *,
    source_type: str,
    completion_status: str,
    completion_evidence: dict[str, object],
    review_status: str,
) -> int:
    """One reconciled, card-complete hand so only completion state varies."""
    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Promotion surfaces"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="No-limit Hold'em",
            table_size=6,
            hero_position="BTN",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=20,
            hero_bb_won=10,
            review_status=review_status,
            source_type=source_type,
            completion_status=completion_status,
            completion_evidence=completion_evidence,
        )
    )
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
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
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
    # The declared pot award is a measured declaration of its own on a
    # reconstructed hand, and the operator answers it in the Accounting
    # reconciliation panel; the fixture does the same.
    attest_declared_assumptions(db, hand.id)
    hand_id = hand.id
    db.close()
    return hand_id


def _seed_blocked_cv_hand(path: Path) -> int:
    return _seed(
        path,
        source_type="cv_import",
        completion_status="uncertain",
        completion_evidence={},
        review_status="needs_correction",
    )


def _seed_ready_manual_hand(path: Path) -> int:
    return _seed(
        path,
        source_type="manual",
        completion_status="not_applicable",
        completion_evidence={},
        review_status="unreviewed",
    )


def _isolate(path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()


def _saved_review_status(path: Path, hand_id: int) -> str:
    verifier = PokerDatabase(path)
    verifier.init_db()
    status = verifier.fetch_hand(hand_id).review_status
    verifier.close()
    return status


# --------------------------------------------------------------------------
# show_saved_hands: the per-hand "Update status" control
# --------------------------------------------------------------------------

_SAVED_HANDS_SCRIPT = """
import streamlit as st

import app as app_module
from poker_tracker.persistence.db import PokerDatabase

db = PokerDatabase({path!r})
db.init_db()
session = db.fetch_sessions()[0]
app_module.show_saved_hands(db, session)
"""


def _run_saved_hands(path: Path, monkeypatch) -> AppTest:
    _isolate(path, monkeypatch)
    app = AppTest.from_string(
        _SAVED_HANDS_SCRIPT.format(path=str(path)), default_timeout=30
    ).run()
    assert not list(app.exception)
    return app


def _status_widget(app: AppTest, label: str = "Review status"):
    return next(item for item in app.selectbox if item.label == label)


def test_saved_hands_does_not_offer_reviewed_for_a_blocked_hand(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "saved-blocked.sqlite3"
    _seed_blocked_cv_hand(path)

    app = _run_saved_hands(path, monkeypatch)

    assert "reviewed" not in _status_widget(app).options


def test_saved_hands_update_status_cannot_promote_a_blocked_hand(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "saved-bypass.sqlite3"
    hand_id = _seed_blocked_cv_hand(path)

    app = _run_saved_hands(path, monkeypatch)
    next(button for button in app.button if button.label == "Update status").click()
    app.run()

    assert not list(app.exception)
    assert _saved_review_status(path, hand_id) != "reviewed"


def test_saved_hands_can_still_promote_a_ready_manual_hand(tmp_path, monkeypatch) -> None:
    """Regression: the gate must not break the surface it protects."""
    path = tmp_path / "saved-manual.sqlite3"
    hand_id = _seed_ready_manual_hand(path)

    app = _run_saved_hands(path, monkeypatch)
    assert "reviewed" in _status_widget(app).options
    _status_widget(app).set_value("reviewed")
    app.run()
    next(button for button in app.button if button.label == "Update status").click()
    app.run()

    assert not list(app.exception)
    assert _saved_review_status(path, hand_id) == "reviewed"


def test_saved_hands_requires_confirmation_for_a_complete_cv_hand(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "saved-confirm.sqlite3"
    hand_id = _seed(
        path,
        source_type="cv_import",
        completion_status="complete",
        completion_evidence=CLEAN_EVIDENCE,
        review_status="needs_correction",
    )

    app = _run_saved_hands(path, monkeypatch)
    assert "reviewed" not in _status_widget(app).options
    next(
        item
        for item in app.checkbox
        if item.label == "I have read the evidence above and confirm this hand is correct"
    ).set_value(True)
    app.run()

    assert "reviewed" in _status_widget(app).options
    _status_widget(app).set_value("reviewed")
    app.run()
    next(button for button in app.button if button.label == "Update status").click()
    app.run()

    assert not list(app.exception)
    assert _saved_review_status(path, hand_id) == "reviewed"


# --------------------------------------------------------------------------
# Settings -> Coaching: show_hand_coach_review
# --------------------------------------------------------------------------


class _StubProvider:
    provider_name = "fixture"
    model_name = "deterministic"

    def generate_hand_review(self, prompt: str) -> str:
        return (
            "Hand Summary: Hero bet the river and was called.\n"
            "Theory Coach: Bet sizing is reasonable.\n"
            "Exploit Coach: Villain calls too wide.\n"
            "EV / Math Notes: Recorded facts only.\n"
            "Study Lesson: Size up against calling stations.\n"
            "Next Review Question: Would a larger bet still get called?\n"
        )

    def generate_session_review(self, prompt: str) -> str:
        return self.generate_hand_review(prompt)


def _run_settings_coaching(path: Path, monkeypatch) -> AppTest:
    _isolate(path, monkeypatch)
    monkeypatch.setattr(
        "poker_tracker.coaching.llm_providers.get_provider_from_env",
        lambda *args, **kwargs: _StubProvider(),
    )
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.radio[0].set_value(Page.SETTINGS)
    app.run()
    assert not list(app.exception)
    return app


def _settings_coach_button(app: AppTest):
    return next(
        item
        for item in app.button
        if item.label == "Generate and save post-session hand review"
    )


def test_settings_coaching_does_not_promote_a_blocked_hand(tmp_path, monkeypatch) -> None:
    path = tmp_path / "settings-blocked.sqlite3"
    hand_id = _seed_blocked_cv_hand(path)

    app = _run_settings_coaching(path, monkeypatch)
    _settings_coach_button(app).click()
    app.run()

    assert not list(app.exception)
    verifier = PokerDatabase(path)
    verifier.init_db()
    reviews = verifier.fetch_coaching_reviews_by_hand(hand_id)
    status = verifier.fetch_hand(hand_id).review_status
    verifier.close()
    # The generated review is retained; only the promotion is refused.
    assert len(reviews) == 1
    assert status != "reviewed"


def test_settings_coaching_renders_the_blockers_before_the_button(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "settings-panel.sqlite3"
    _seed_blocked_cv_hand(path)

    app = _run_settings_coaching(path, monkeypatch)
    rendered = "\n".join(item.value for item in app.markdown)

    assert "Study readiness" in rendered
    assert "Completion · " in rendered


def test_settings_coaching_promotes_a_ready_manual_hand(tmp_path, monkeypatch) -> None:
    path = tmp_path / "settings-manual.sqlite3"
    hand_id = _seed_ready_manual_hand(path)

    app = _run_settings_coaching(path, monkeypatch)
    _settings_coach_button(app).click()
    app.run()

    assert not list(app.exception)
    assert _saved_review_status(path, hand_id) == "reviewed"


# --------------------------------------------------------------------------
# The manual Add-hand form: the other way review_status is written
# --------------------------------------------------------------------------


def _run_add_hand_form(path: Path, monkeypatch) -> AppTest:
    """The manual Add-hands form lives in the Sessions workspace's "Add hands" tab."""
    _isolate(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.radio[0].set_value(Page.SESSIONS)
    app.run()
    assert not list(app.exception)
    return app


def test_add_hand_form_never_offers_reviewed_as_a_starting_status(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "add-hand.sqlite3"
    _seed_ready_manual_hand(path)

    app = _run_add_hand_form(path, monkeypatch)

    assert _status_widget(app).options == ["unreviewed", "needs_correction"]


def test_add_hand_form_forces_a_declared_cv_source_to_be_unproven(
    tmp_path, monkeypatch
) -> None:
    """A hand typed in by hand but declared reconstructed carries no evidence."""
    path = tmp_path / "add-hand-cv.sqlite3"
    _seed_ready_manual_hand(path)

    app = _run_add_hand_form(path, monkeypatch)
    next(item for item in app.number_input if item.label == "Hand number").set_value(2)
    next(item for item in app.selectbox if item.label == "Source").set_value("cv_import")
    _status_widget(app).set_value("unreviewed")
    app.run()
    next(
        item for item in app.button if item.label == "Save hand"
    ).click()
    app.run()

    assert not list(app.exception)
    verifier = PokerDatabase(path)
    verifier.init_db()
    created = next(
        hand for hand in verifier.fetch_all_hands() if hand.hand_number == 2
    )
    verifier.close()
    assert created.source_type == "cv_import"
    assert created.completion_status == "uncertain"
    assert created.review_status == "needs_correction"
    assert created.completion_evidence == {}
