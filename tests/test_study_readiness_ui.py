"""UI-level proof that no blocked hand can reach ``reviewed`` through any surface."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

import poker_tracker.persistence.db as db_module
from poker_tracker.persistence.completion import (
    acknowledge_codes,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandIssue,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
    SolverRun,
)
from poker_tracker.services import hand_accounting as hand_accounting_module
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import StudyReadiness, evaluate_study_readiness
from poker_tracker.ui.navigation import Page
from tests.conftest import attest_declared_assumptions

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

CLEAN_EVIDENCE: dict[str, object] = {
    "evidence_version": 1,
    "partial_start": False,
    "partial_end": False,
    "terminal_event": "showdown",
    "first_source_timestamp_s": 1.0,
    "last_source_timestamp_s": 9.0,
    "preceding_boundary": {
        "kind": "hand_start",
        "timestamp_s": 1.0,
        "frame_ref": "frames/start.png",
        "confidence": 0.94,
        "codes": [],
    },
    "following_boundary": {
        "kind": "hand_end",
        "timestamp_s": 9.0,
        "frame_ref": "frames/end.png",
        "confidence": 0.91,
        "codes": [],
    },
    "boundary_confidence": 0.91,
    "source_frames": ["frames/start.png", "frames/end.png"],
    "warning_codes": [],
    "rejection_codes": [],
    "acknowledged_codes": [],
    "layout_profile": "clubwpt-6max",
    "layout_supported": True,
    "table_size": 6,
    "pipeline_version": "two-model-v7",
    "model_versions": {"detector": "v7"},
}


def _with_evidence(**overrides: object) -> dict[str, object]:
    evidence = dict(CLEAN_EVIDENCE)
    evidence.update(overrides)
    return evidence


def _seed_hand(
    path: Path,
    *,
    source_type: str = "cv_import",
    completion_status: str = "uncertain",
    completion_evidence: dict[str, object] | None = None,
    review_status: str = "needs_correction",
    reconcile: bool = True,
    hero_bb_won: float | None = 10,
    award_amount: float | None = 20,
    rake_rate: float = 0.0,
) -> int:
    """Seed one reconciled, card-complete hand so only completion state varies.

    The last three arguments default to the honest, unraked hand every other test
    here uses; only the settlement-assumption test moves them.
    """

    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Readiness"))
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
            hero_bb_won=hero_bb_won,
            review_status=review_status,
            source_type=source_type,
            completion_status=completion_status,
            completion_evidence=(
                CLEAN_EVIDENCE if completion_evidence is None else completion_evidence
            ),
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
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="settled", rake_rate=rake_rate)
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
                amount=award_amount,
                entry_order=1,
            )
        ],
    )
    if reconcile:
        persist_reconciliation(db, hand.id)
        # The declared pot award is its own measured declaration on a
        # reconstructed hand -- the CV exporter emits no settlement rows, so
        # nothing observed who won -- and it is answered here so each test below
        # is left with the declaration it is actually about.
        attest_declared_assumptions(db, hand.id, only="declared_pot_awards")
    hand_id = hand.id
    db.close()
    return hand_id


def _configure_app_env(path: Path, monkeypatch) -> None:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()


def _run_study(path: Path, monkeypatch) -> AppTest:
    _configure_app_env(path, monkeypatch)
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    next(item for item in app.radio if "Study" in list(item.options)).set_value(
        Page.STUDY
    )
    app.run()
    assert not list(app.exception)
    return app


def _run_validation_editors(
    path: Path,
    monkeypatch,
    hand_id: int,
    *,
    frames_validated: bool = True,
) -> AppTest:
    """Mount Import validation editors without a full CV timeline UI."""

    _configure_app_env(path, monkeypatch)
    script = path.parent / f"_validation_editors_{hand_id}.py"
    script.write_text(
        "\n".join(
            [
                "from poker_tracker.persistence.db import PokerDatabase",
                "import app as app_module",
                f"db = PokerDatabase(r'{path}')",
                "db.init_db()",
                f"hand = db.fetch_hand({hand_id})",
                "app_module.render_validation_edit_and_approve(",
                "    db,",
                "    hand,",
                f"    frames_validated={frames_validated!r},",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not list(app.exception)
    return app


def _open_fix_tool(app: AppTest, tool_label: str) -> AppTest:
    tool_box = next(
        item for item in app.selectbox if item.label == "What else needs fixing?"
    )
    assert tool_label in list(tool_box.options), list(tool_box.options)
    tool_box.set_value(tool_label)
    app.run()
    assert not list(app.exception)
    return app


def _saved_review_status(path: Path, hand_id: int) -> str:
    verifier = PokerDatabase(path)
    verifier.init_db()
    status = verifier.fetch_hand(hand_id).review_status
    verifier.close()
    return status


def test_validation_finish_refuses_blocked_hand(tmp_path, monkeypatch) -> None:
    path = tmp_path / "blocked.sqlite3"
    hand_id = _seed_hand(path)

    app = _run_validation_editors(path, monkeypatch, hand_id)
    finish = next(
        button
        for button in app.button
        if button.label == "Finish validation — send to Study"
    )
    finish.click()
    app.run()
    assert not list(app.exception)
    assert _saved_review_status(path, hand_id) != "reviewed"
    st.cache_resource.clear()


def test_validation_cannot_promote_uncertain_cv_hand_even_via_guard(
    tmp_path, monkeypatch
) -> None:
    """Finish validation and the store both refuse a blocked hand."""

    path = tmp_path / "bypass.sqlite3"
    hand_id = _seed_hand(path)

    app = _run_validation_editors(path, monkeypatch, hand_id)
    next(
        button
        for button in app.button
        if button.label == "Finish validation — send to Study"
    ).click()
    app.run()
    assert not list(app.exception)
    assert _saved_review_status(path, hand_id) != "reviewed"
    st.cache_resource.clear()

    import app as app_module

    db = PokerDatabase(path)
    db.init_db()
    hand = db.fetch_hand(hand_id)
    blocked = evaluate_study_readiness(hand, accounting=None)
    assert app_module.guarded_update_hand_status(db, hand, blocked, "reviewed") is False
    forced_ready = StudyReadiness(
        is_ready=True,
        completion_status=hand.completion_status,
        blockers=(),
    )
    assert (
        app_module.guarded_update_hand_status(db, hand, forced_ready, "reviewed") is False
    )
    assert db.fetch_hand(hand_id).review_status != "reviewed"
    db.close()


def test_validation_shows_blockers_grouped_by_category(tmp_path, monkeypatch) -> None:
    path = tmp_path / "grouped.sqlite3"
    hand_id = _seed_hand(path, completion_evidence={})

    app = _run_validation_editors(path, monkeypatch, hand_id)
    rendered = "\n".join(item.value for item in app.markdown)

    assert "What's blocking Study" in rendered or "Not study-ready" in rendered
    assert "Completion · " in rendered or "completion" in rendered.lower()
    reasons = "\n".join(item.value for item in app.caption)
    assert "Clears when:" in reasons or "Import validation" in reasons
    st.cache_resource.clear()


def test_validation_never_renders_a_confidence_percentage(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "confidence.sqlite3"
    hand_id = _seed_hand(path)

    app = _run_validation_editors(path, monkeypatch, hand_id)
    confidence_lines = [
        item.value for item in app.caption if "Reconstruction confidence" in item.value
    ]
    text = "\n".join(
        [item.value for item in app.markdown]
        + [item.value for item in app.caption]
    )
    for line in confidence_lines:
        assert "%" not in line
        assert not re.search(r"\d", line)
    assert "%" not in text or "Reconstruction confidence" not in text
    st.cache_resource.clear()


def test_validation_finish_confirms_a_cv_hand(tmp_path, monkeypatch) -> None:
    path = tmp_path / "confirm.sqlite3"
    hand_id = _seed_hand(path, completion_status="complete")

    # frames_validated=False keeps auto-approve from racing the Finish click.
    app = _run_validation_editors(
        path, monkeypatch, hand_id, frames_validated=False
    )
    assert any(
        button.label == "Finish validation — send to Study" for button in app.button
    )

    next(
        button
        for button in app.button
        if button.label == "Finish validation — send to Study"
    ).click()
    app.run()

    assert not list(app.exception)
    assert _saved_review_status(path, hand_id) == "reviewed"
    st.cache_resource.clear()


def test_validation_editors_surface_is_mounted_for_ready_hand(
    tmp_path, monkeypatch
) -> None:
    """Validation hosts edit + finish; Study no longer owns side-by-side Approve."""

    path = tmp_path / "side_by_side.sqlite3"
    hand_id = _seed_hand(path, completion_status="complete")

    app = _run_validation_editors(
        path, monkeypatch, hand_id, frames_validated=False
    )
    rendered = "\n".join(item.value for item in app.markdown)
    captions = "\n".join(item.value for item in app.caption)

    assert "### Fix this hand" in rendered
    assert "Jump a blocker" in captions
    assert any(
        button.label == "Finish validation — send to Study" for button in app.button
    )
    st.cache_resource.clear()


def test_study_confirming_a_settlement_assumption_clears_its_blocker(
    tmp_path, monkeypatch
) -> None:
    """The clearing action ACCOUNTING_ASSUMPTION_DEPENDENT names, pressed on the page.

    Several earlier rounds found blockers naming an action the product could not
    perform, so this drives the real control on the real page: the hand's recorded
    hero result of 0 is true only because a declared 50% rake destroys 10 chips,
    and under a neutral policy the same records derive +10.
    """
    path = tmp_path / "assumption.sqlite3"
    hand_id = _seed_hand(
        path,
        completion_status="complete",
        hero_bb_won=0,
        award_amount=None,
        rake_rate=0.5,
    )
    # The pipeline-warning half of the disclosure, already accepted, so what is
    # left on the page is the assumption itself.
    accepter = PokerDatabase(path)
    accepter.init_db()
    stored = accepter.fetch_hand(hand_id)
    accepter.update_hand_completion(
        hand_id,
        completion_evidence=dump_completion_evidence(
            acknowledge_codes(
                parse_completion_evidence(stored.completion_evidence),
                list(parse_completion_evidence(stored.completion_evidence).unresolved_codes),
            )
        ),
        notes="Acknowledged in test.",
    )
    accepter.close()

    app = _open_fix_tool(
        _run_validation_editors(path, monkeypatch, hand_id),
        "Chip stacks / accounting",
    )
    rendered = "\n".join(item.value for item in app.markdown)
    assert "rake_policy · unconfirmed" in rendered
    assert "settlement inputs you declared" in rendered

    next(
        button for button in app.button if button.label == "Confirm this assumption"
    ).click()
    app.run()
    assert not list(app.exception)

    verifier = PokerDatabase(path)
    verifier.init_db()
    confirmed = verifier.fetch_hand(hand_id)
    accounting = reconcile_persisted_hand(verifier, hand_id)
    verifier.close()
    assert accounting.assumption_dependence
    assert (
        evaluate_study_readiness(
            confirmed, accounting=accounting, user_confirmed=True
        ).has("ACCOUNTING_ASSUMPTION_DEPENDENT")
        is False
    )

    # Confirming the last blocker with frames_validated=True auto-approves.
    st.cache_resource.clear()
    if _saved_review_status(path, hand_id) != "reviewed":
        app = _run_validation_editors(
            path, monkeypatch, hand_id, frames_validated=False
        )
        next(
            button
            for button in app.button
            if button.label == "Finish validation — send to Study"
        ).click()
        app.run()
        assert not list(app.exception)
    assert _saved_review_status(path, hand_id) == "reviewed"
    st.cache_resource.clear()


def test_the_confirm_control_reports_a_refused_write_instead_of_flashing(
    tmp_path, monkeypatch
) -> None:
    """A control must never say "Confirmed" over a write that was discarded.

    Round 10 found the live version of this: the writer was scoped on
    ``source_type == 'manual'`` while the control was scoped on the reconstructed
    pair, so the page flashed "Confirmed the declared rake_policy for this hand"
    over a discarded write. The guard was added; nothing exercised its False
    branch, and forcing the flash unconditionally left the whole suite green --
    the only survivor of 45 mutants.

    The writer is driven to its refusing answer directly, because that is the
    only thing this control's branch is about: whether the page tells the truth
    about a write that did not happen. (In the product the answer comes from
    ``attest_assumption`` re-measuring, so it refuses a code this hand no longer
    measures -- a settlement saved from another tab between the render and the
    press -- rather than recording an attestation to a quantity that is gone.)
    """
    path = tmp_path / "refused.sqlite3"
    hand_id = _seed_hand(
        path,
        completion_status="complete",
        hero_bb_won=0,
        award_amount=None,
        rake_rate=0.5,
    )

    import app as app_module

    monkeypatch.setattr(
        hand_accounting_module, "attest_assumption", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(
        app_module, "attest_assumption", lambda *args, **kwargs: False
    )
    app = _open_fix_tool(
        _run_validation_editors(path, monkeypatch, hand_id),
        "Chip stacks / accounting",
    )
    next(
        item for item in app.button if item.label == "Confirm this assumption"
    ).click()
    app.run()
    assert not list(app.exception)
    assert any("Nothing was recorded" in item.value for item in app.error)
    assert not any("Confirmed the declared" in item.value for item in app.markdown)

    verifier = PokerDatabase(path)
    verifier.init_db()
    stored = verifier.fetch_hand(hand_id)
    verifier.close()
    assert stored is not None
    confirmed = parse_completion_evidence(
        stored.completion_evidence
    ).confirmed_assumption_codes
    assert not [code for code in confirmed if ":rake_policy:" in code]
    st.cache_resource.clear()


def test_study_manual_hand_can_still_be_marked_reviewed(tmp_path, monkeypatch) -> None:
    path = tmp_path / "manual.sqlite3"
    hand_id = _seed_hand(
        path,
        source_type="manual",
        completion_status="not_applicable",
        completion_evidence={},
        review_status="unreviewed",
    )

    app = _run_validation_editors(
        path, monkeypatch, hand_id, frames_validated=False
    )
    next(
        button
        for button in app.button
        if button.label == "Finish validation — send to Study"
    ).click()
    app.run()

    assert not list(app.exception)
    assert _saved_review_status(path, hand_id) == "reviewed"
    st.cache_resource.clear()


def test_study_partial_hand_is_still_inspectable_and_correctable(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "partial.sqlite3"
    hand_id = _seed_hand(
        path,
        completion_status="partial",
        completion_evidence=_with_evidence(partial_end=True),
    )

    app = _open_fix_tool(
        _run_validation_editors(path, monkeypatch, hand_id),
        "Cards, board, or pot",
    )
    next(item for item in app.text_input if item.label == "Board cards").set_value(
        "Qd 7s 6c"
    )
    next(
        item
        for item in app.text_input
        if item.label == "Why is this correction needed?"
    ).set_value("Frame shows 6c.")
    next(
        button for button in app.button if button.label == "Save corrected facts"
    ).click()
    app.run()

    assert not list(app.exception)
    verifier = PokerDatabase(path)
    verifier.init_db()
    saved = verifier.fetch_hand(hand_id)
    verifier.close()
    assert saved.board_cards == "Qd 7s 6c"
    # A correction can never un-truncate a recording.
    assert saved.completion_status == "partial"
    assert saved.review_status == "needs_correction"
    st.cache_resource.clear()


def test_study_acknowledging_a_source_warning_promotes_uncertain_to_complete(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "acknowledge.sqlite3"
    hand_id = _seed_hand(
        path,
        completion_evidence=_with_evidence(warning_codes=["pot_not_reconciled"]),
    )

    app = _open_fix_tool(
        _run_validation_editors(path, monkeypatch, hand_id),
        "Source warnings",
    )
    next(button for button in app.button if button.label == "Acknowledge").click()
    app.run()

    assert not list(app.exception)
    verifier = PokerDatabase(path)
    verifier.init_db()
    saved = verifier.fetch_hand(hand_id)
    corrections = verifier.fetch_hand_corrections(hand_id)
    verifier.close()
    assert saved.completion_status == "complete"
    assert corrections
    assert "pot_not_reconciled" in corrections[0].after_state["acknowledged_codes"]
    st.cache_resource.clear()


def test_the_panel_offers_no_acknowledge_button_for_a_rejection_code(
    tmp_path, monkeypatch
) -> None:
    """A rejection is the pipeline refusing the hand, so there is nothing to accept."""
    path = tmp_path / "rejection.sqlite3"
    hand_id = _seed_hand(
        path,
        completion_evidence=_with_evidence(
            rejection_codes=["duplicate_card_detected"]
        ),
    )

    app = _open_fix_tool(
        _run_validation_editors(path, monkeypatch, hand_id),
        "Source warnings",
    )

    assert not list(app.exception)
    assert not [button for button in app.button if button.label == "Acknowledge"]
    rendered = "\n".join(item.value for item in app.markdown)
    assert "duplicate_card_detected · rejected by the pipeline" in rendered
    st.cache_resource.clear()


def test_acknowledgement_never_promotes_a_partial_hand(tmp_path, monkeypatch) -> None:
    path = tmp_path / "acknowledge-partial.sqlite3"
    hand_id = _seed_hand(
        path,
        completion_status="partial",
        completion_evidence=_with_evidence(
            partial_start=True, warning_codes=["pot_not_reconciled"]
        ),
    )

    app = _open_fix_tool(
        _run_validation_editors(path, monkeypatch, hand_id),
        "Source warnings",
    )
    next(button for button in app.button if button.label == "Acknowledge").click()
    app.run()

    assert not list(app.exception)
    verifier = PokerDatabase(path)
    verifier.init_db()
    saved = verifier.fetch_hand(hand_id)
    verifier.close()
    assert saved.completion_status == "partial"
    st.cache_resource.clear()


class _StubProvider:
    """Deterministic stand-in so the coaching path can be driven without a network."""

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


def _run_coach_surface(path: Path, monkeypatch, hand_id: int) -> AppTest:
    """Mount Analyze coaching without requiring the hand to be Study-queued."""

    _configure_app_env(path, monkeypatch)
    script = path.parent / f"_coach_surface_{hand_id}.py"
    script.write_text(
        "\n".join(
            [
                "from poker_tracker.persistence.db import PokerDatabase",
                "from poker_tracker.services.study_readiness import evaluate_study_readiness",
                "import app as app_module",
                f"db = PokerDatabase(r'{path}')",
                "db.init_db()",
                f"hand = db.fetch_hand({hand_id})",
                "session = db.fetch_session(hand.session_id)",
                "actions = db.fetch_actions_by_hand(hand.id)",
                "players = db.fetch_players_by_hand(hand.id)",
                "accounting, accounting_error = app_module._reconcile_cached(db, hand.id, None)",
                "readiness = evaluate_study_readiness(",
                "    hand,",
                "    accounting=accounting,",
                "    accounting_error=accounting_error,",
                "    hand_issues=db.fetch_hand_issues(hand_id=hand.id),",
                "    coaching_reviews=db.fetch_coaching_reviews_by_hand(hand.id),",
                "    hand_reviews=db.fetch_reviews_by_hand(hand.id),",
                "    solver_runs=db.fetch_solver_runs_by_hand(hand.id),",
                "    user_confirmed=True,",
                ")",
                "app_module.show_study_coach_review(",
                "    db, session, hand, actions, players, accounting,",
                "    accounting_error, db.fetch_coaching_reviews_by_hand(hand.id), readiness,",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not list(app.exception)
    return app


def test_coaching_generation_does_not_promote_a_blocked_hand(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "coaching.sqlite3"
    hand_id = _seed_hand(path)
    monkeypatch.setattr(
        "app.get_provider_from_env",
        lambda *args, **kwargs: _StubProvider(),
    )

    app = _run_coach_surface(path, monkeypatch, hand_id)
    next(
        button
        for button in app.button
        if button.label == "Generate and save corrected-hand coaching"
    ).click()
    app.run()

    assert not list(app.exception)
    verifier = PokerDatabase(path)
    verifier.init_db()
    reviews = verifier.fetch_coaching_reviews_by_hand(hand_id)
    status = verifier.fetch_hand(hand_id).review_status
    verifier.close()
    # The coaching is retained; only the promotion is refused.
    assert len(reviews) == 1
    assert status != "reviewed"
    st.cache_resource.clear()


def test_coaching_generation_promotes_a_ready_hand(tmp_path, monkeypatch) -> None:
    """Regression: the guard must not break the legitimate promotion path."""

    path = tmp_path / "coaching-ready.sqlite3"
    hand_id = _seed_hand(
        path,
        source_type="manual",
        completion_status="not_applicable",
        completion_evidence={},
        review_status="unreviewed",
    )
    monkeypatch.setattr(
        "app.get_provider_from_env",
        lambda *args, **kwargs: _StubProvider(),
    )

    app = _run_coach_surface(path, monkeypatch, hand_id)
    next(
        button
        for button in app.button
        if button.label == "Generate and save corrected-hand coaching"
    ).click()
    app.run()

    assert not list(app.exception)
    assert _saved_review_status(path, hand_id) == "reviewed"
    st.cache_resource.clear()


def test_coaching_generation_is_blocked_by_an_open_debugging_issue(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "coaching-issue.sqlite3"
    hand_id = _seed_hand(
        path,
        source_type="manual",
        completion_status="not_applicable",
        completion_evidence={},
        review_status="unreviewed",
    )
    db = PokerDatabase(path)
    db.init_db()
    db.create_hand_issue(
        HandIssue(
            hand_id=hand_id,
            issue_types=["cards"],
            description="The river card does not match the recording.",
        )
    )
    db.close()
    monkeypatch.setattr(
        "app.get_provider_from_env",
        lambda *args, **kwargs: _StubProvider(),
    )

    app = _run_coach_surface(path, monkeypatch, hand_id)
    button = next(
        item
        for item in app.button
        if item.label == "Generate and save corrected-hand coaching"
    )

    assert button.disabled is True
    assert _saved_review_status(path, hand_id) != "reviewed"
    st.cache_resource.clear()


SOLVER_EVIDENCE: dict[str, object] = {
    "backend": "TexasSolver",
    "backend_version": "pinned",
    "street": "flop",
    "board": "Qd 7s 2c",
    "pot": 20.0,
    "effective_stack": 100.0,
    "hero_player": "Hero",
    "hero_combo": "AhQs",
    "range_ip_name": "BTN open",
    "range_oop_name": "BB defend",
}


def _seed_completed_solver_run(path: Path, hand_id: int) -> None:
    db = PokerDatabase(path)
    db.init_db()
    db.create_solver_run(
        SolverRun(
            hand_id=hand_id,
            status="completed",
            input_hash="fixture-hash",
            evidence=SOLVER_EVIDENCE,
        )
    )
    db.close()


def _run_solver_surface(path: Path, monkeypatch, hand_id: int) -> AppTest:
    _configure_app_env(path, monkeypatch)
    script = path.parent / f"_solver_surface_{hand_id}.py"
    script.write_text(
        "\n".join(
            [
                "from poker_tracker.persistence.db import PokerDatabase",
                "from poker_tracker.services.study_readiness import evaluate_study_readiness",
                "import app as app_module",
                f"db = PokerDatabase(r'{path}')",
                "db.init_db()",
                f"hand = db.fetch_hand({hand_id})",
                "session = db.fetch_session(hand.session_id)",
                "actions = db.fetch_actions_by_hand(hand.id)",
                "players = db.fetch_players_by_hand(hand.id)",
                "accounting, accounting_error = app_module._reconcile_cached(db, hand.id, None)",
                "readiness = evaluate_study_readiness(",
                "    hand,",
                "    accounting=accounting,",
                "    accounting_error=accounting_error,",
                "    hand_issues=db.fetch_hand_issues(hand_id=hand.id),",
                "    coaching_reviews=db.fetch_coaching_reviews_by_hand(hand.id),",
                "    hand_reviews=db.fetch_reviews_by_hand(hand.id),",
                "    solver_runs=db.fetch_solver_runs_by_hand(hand.id),",
                "    user_confirmed=True,",
                ")",
                "app_module.show_solver_review(",
                "    db, session, hand, actions, players, accounting,",
                "    accounting_error, readiness,",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not list(app.exception)
    return app


def test_solver_explanation_does_not_promote_a_blocked_hand(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "solver.sqlite3"
    hand_id = _seed_hand(path)
    _seed_completed_solver_run(path, hand_id)
    monkeypatch.setattr(
        "app.get_provider_from_env",
        lambda *args, **kwargs: _StubProvider(),
    )

    app = _run_solver_surface(path, monkeypatch, hand_id)
    next(
        button
        for button in app.button
        if button.label == "Explain solver result with AI"
    ).click()
    app.run()

    assert not list(app.exception)
    verifier = PokerDatabase(path)
    verifier.init_db()
    reviews = verifier.fetch_coaching_reviews_by_hand(hand_id)
    status = verifier.fetch_hand(hand_id).review_status
    verifier.close()
    assert len(reviews) == 1
    assert status != "reviewed"
    st.cache_resource.clear()


def test_every_review_status_write_is_routed_through_the_guard() -> None:
    """Regression for the enumerated bypass paths: one writer, one gate."""

    source = Path(APP_PATH).read_text()
    assert source.count("db.update_hand_status(") == 1
    guard_start = source.index("def guarded_update_hand_status(")
    guard_end = source.index("\ndef ", guard_start + 1)
    assert "db.update_hand_status(" in source[guard_start:guard_end]
    # Every remaining promotion site must call the guard rather than the store.
    assert source.count("guarded_update_hand_status(") >= 6
    assert "def approve_hand_for_study(" in source
    assert "def load_study_session_hands(" in source


def test_load_study_session_hands_scopes_to_one_session_without_all_hands(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "scoped.sqlite3"
    db = PokerDatabase(path)
    db.init_db()
    first = db.create_session(Session(name="First"))
    second = db.create_session(Session(name="Second"))
    db.create_hand(
        Hand(
            session_id=first.id,
            hand_number=1,
            source_type="manual",
            completion_status="not_applicable",
            study_inclusion="auto",
            review_status="reviewed",
        )
    )
    db.create_hand(
        Hand(
            session_id=second.id,
            hand_number=1,
            source_type="manual",
            completion_status="not_applicable",
            study_inclusion="auto",
            review_status="reviewed",
        )
    )
    sessions = db.fetch_sessions()
    calls: list[str] = []
    original = db.fetch_all_hands

    def _boom() -> list[Hand]:
        calls.append("fetch_all_hands")
        return original()

    monkeypatch.setattr(db, "fetch_all_hands", _boom)
    import app as app_module

    hand_session, ordered, forced = app_module.load_study_session_hands(
        db, sessions, second, None
    )
    db.close()
    assert forced is None
    assert hand_session is not None and hand_session.id == second.id
    assert len(ordered) == 1
    assert ordered[0].session_id == second.id
    assert calls == []


def test_batch_approve_promotes_ready_and_skips_blocked(tmp_path) -> None:
    path = tmp_path / "batch.sqlite3"
    ready_id = _seed_hand(
        path,
        source_type="manual",
        completion_status="not_applicable",
        completion_evidence={},
        review_status="unreviewed",
    )
    db = PokerDatabase(path)
    db.init_db()
    ready = db.fetch_hand(ready_id)
    assert ready is not None
    blocked = db.create_hand(
        Hand(
            session_id=ready.session_id,
            hand_number=2,
            source_type="manual",
            hero_cards="Ah Kd",
            board_cards="2c 3d 4h",
            review_status="unreviewed",
            completion_status="not_applicable",
            completion_evidence={},
        )
    )
    db.create_hand_issue(
        HandIssue(
            hand_id=blocked.id,
            issue_types=["cards"],
            description="Board looks wrong on the recording.",
        )
    )
    ordered = [
        hand
        for hand in db.fetch_hands_by_session(ready.session_id)
        if hand.study_inclusion != "skip"
    ]
    import app as app_module

    cache = app_module.new_accounting_cache()
    candidates = app_module._batch_approve_candidates(ordered)
    assert {hand.id for hand in candidates} == {ready_id, blocked.id}
    approved, skipped = app_module.approve_ready_hands_in_session(
        db, candidates, cache
    )
    assert approved == 1
    assert skipped == 1
    assert db.fetch_hand(ready_id).review_status == "reviewed"
    assert db.fetch_hand(blocked.id).review_status != "reviewed"
    db.close()
