"""The review-promotion surfaces that ``test_study_readiness_ui`` does not reach.

``test_study_readiness_ui`` covers the Study workspace, the Study coach button,
and the solver-explanation button. The remaining ways a hand can acquire
``review_status = 'reviewed'`` are the saved-hands expander and the Settings ->
Coaching tab; both are driven here through a real Streamlit run against a real
SQLite file. Manual entry is the one surface that writes a hand without ever
being able to promote it, and is covered structurally at the end of this file.
"""

from __future__ import annotations

import dataclasses
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
from poker_tracker.services.manual_spot_entry import (
    ManualSpotInput,
    PostflopActionInput,
    build_manual_spot,
    save_manual_spot,
)
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
# Manual spot entry: the other way a hand is written into a session
# --------------------------------------------------------------------------
#
# The free-form Add-hand form these two tests used to drive is gone. The Sessions
# "Add hands" tab is now compact solver-spot entry (``create_hand_form`` ->
# ``_create_single_hand_form``), which renders no Source and no Review status
# control at all, so there is no widget left to assert an option list on.
#
# The invariant those tests protected is now structural rather than presentational,
# and is asserted that way below: ``ManualSpotInput`` has no field that can declare
# either value, and ``build_manual_spot`` hardcodes the safe pair. A UI test could
# only prove that one form withholds the choice; this proves the entry path cannot
# express it.


def _manual_spot() -> ManualSpotInput:
    """One valid single-raised-pot spot; only the stamped labels matter here."""
    return ManualSpotInput(
        hand_number=2,
        hero_cards="Ah Qs",
        board_cards="Qd 7s 2c",
        hero_position="BB",
        villain_position="BTN",
        table_size=6,
        starting_stack=100.0,
        pot_type="single_raised",
        opener="villain",
        open_to=2.5,
        postflop_actions=(
            PostflopActionInput("flop", "hero", "check"),
            PostflopActionInput("flop", "villain", "bet", 3.75),
            PostflopActionInput("flop", "hero", "call", 3.75),
        ),
        winner="hero",
    )


def test_manual_spot_entry_cannot_declare_a_review_status_or_source() -> None:
    """The entry surface offers no field for either label, so neither can be forged."""
    fields = {field.name for field in dataclasses.fields(ManualSpotInput)}

    assert "review_status" not in fields
    assert "source_type" not in fields
    assert "completion_status" not in fields
    assert "completion_evidence" not in fields


def test_manual_spot_entry_always_lands_unreviewed_and_manual(tmp_path) -> None:
    """A typed-in spot is stamped unreviewed/manual on the way to SQLite."""
    built = build_manual_spot(_manual_spot())

    assert built.hand.review_status == "unreviewed"
    assert built.hand.source_type == "manual"
    assert built.hand.completion_status == "not_applicable"

    path = tmp_path / "manual-spot.sqlite3"
    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Manual entry"))
    saved, _reconciliation, _warnings = save_manual_spot(db, session.id, _manual_spot())
    db.close()

    verifier = PokerDatabase(path)
    verifier.init_db()
    persisted = next(
        hand for hand in verifier.fetch_all_hands() if hand.id == saved.id
    )
    verifier.close()

    assert persisted.review_status == "unreviewed"
    assert persisted.source_type == "manual"
    assert persisted.completion_status == "not_applicable"
