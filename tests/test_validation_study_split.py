"""Validate-and-edit on Import; Study queue is approved hands only."""

from __future__ import annotations

from pathlib import Path

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandIssue,
    HandPlayer,
    HandSettlement,
    ProcessingJob,
    Session,
    SettlementEntry,
    VideoRecord,
)
from poker_tracker.services import validated_hand_import
from poker_tracker.services.hand_accounting import persist_reconciliation
from poker_tracker.services.validated_hand_import import ensure_draft_for_review
from tests.conftest import attest_declared_assumptions
from tests.test_study_readiness_ui import CLEAN_EVIDENCE


def _seed_ready_cv_hand(path: Path) -> int:
    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Split"))
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
            review_status="unreviewed",
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=CLEAN_EVIDENCE,
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
    attest_declared_assumptions(db, hand.id, only="declared_pot_awards")
    hand_id = hand.id
    db.close()
    return hand_id


def test_ensure_draft_for_review_is_idempotent(tmp_path, monkeypatch) -> None:
    from tests.test_validated_hand_import import _make_db, _seed_job, _spine_hand

    db = _make_db(tmp_path)
    timeline = {
        "states": [{"image": "a.jpg", "time_s": 1.0}],
        "hands": [_spine_hand(hand_number=1, source_images=["a.jpg"])],
    }
    job_id, session_id, _timelines = _seed_job(db, tmp_path, timeline, monkeypatch=monkeypatch)
    first = ensure_draft_for_review(db, job_id, 1)
    second = ensure_draft_for_review(db, job_id, 1)
    assert first.status == "imported"
    assert first.hand_id is not None
    assert second.status == "already_present"
    assert second.hand_id == first.hand_id
    assert second.session_id == session_id
    db.close()


def test_try_approve_after_validation_promotes_ready_hand(tmp_path) -> None:
    import app as app_module

    path = tmp_path / "approve.sqlite3"
    hand_id = _seed_ready_cv_hand(path)
    db = PokerDatabase(path)
    db.init_db()
    hand = db.fetch_hand(hand_id)
    assert hand is not None
    assert app_module.try_approve_hand_after_validation(db, hand, announce=False)
    assert db.fetch_hand(hand_id).review_status == "reviewed"
    db.close()


def test_try_approve_after_validation_refuses_open_issue(tmp_path) -> None:
    import app as app_module

    path = tmp_path / "issue.sqlite3"
    hand_id = _seed_ready_cv_hand(path)
    db = PokerDatabase(path)
    db.init_db()
    db.create_hand_issue(
        HandIssue(
            hand_id=hand_id,
            issue_types=["cards"],
            description="Board looks wrong.",
        )
    )
    hand = db.fetch_hand(hand_id)
    assert hand is not None
    assert (
        app_module.try_approve_hand_after_validation(db, hand, announce=False) is False
    )
    assert db.fetch_hand(hand_id).review_status != "reviewed"
    db.close()


def test_study_queue_includes_only_reviewed_hands(tmp_path) -> None:
    import app as app_module

    path = tmp_path / "queue.sqlite3"
    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Queue"))
    reviewed = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="manual",
            completion_status="not_applicable",
            review_status="reviewed",
        )
    )
    db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=2,
            source_type="manual",
            completion_status="not_applicable",
            review_status="unreviewed",
        )
    )
    skipped = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=3,
            source_type="manual",
            completion_status="not_applicable",
            review_status="reviewed",
        )
    )
    db.update_study_inclusion(skipped.id, "skip")
    ordered = app_module._study_queue_for_session(
        db.fetch_hands_by_session(session.id)
    )
    assert [hand.id for hand in ordered] == [reviewed.id]
    db.close()


def test_open_hand_for_validation_sets_import_session_state(tmp_path, monkeypatch) -> None:
    import app as app_module

    path = tmp_path / "open.sqlite3"
    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Open"))
    video = db.create_video(
        VideoRecord(
            original_filename="clip.mp4",
            stored_path=str(tmp_path / "clip.mp4"),
            file_size_bytes=0,
            session_id=session.id,
        )
    )
    job = db.create_processing_job(
        ProcessingJob(
            video_id=video.id,
            job_type="cv_reconstruction",
            status="completed",
        )
    )
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=4,
            source_type="cv_import",
            notes=f"job_{job.id}_timeline timeline_hand_number=4",
            completion_evidence={
                "evidence_version": 1,
                validated_hand_import.CV_TIMELINE_IDENTITY_KEY: {
                    "job_id": job.id,
                    "timeline_hand_number": 4,
                },
            },
        )
    )
    monkeypatch.setattr(app_module.st, "session_state", {})
    navigated: list[object] = []
    monkeypatch.setattr(
        app_module, "navigate_to", lambda page: navigated.append(page)
    )
    monkeypatch.setattr(app_module, "flash", lambda message: None)
    monkeypatch.setattr(app_module, "_activate_session", lambda session_id: None)
    app_module._open_hand_for_validation(db, hand)
    assert navigated
    assert app_module.st.session_state.get("video_context_id") == video.id
    assert app_module.st.session_state.get(f"cv_review_job_{video.id}") == job.id
    assert app_module.st.session_state.get(f"evidence_hand_pending_{job.id}") == 4
    db.close()


def test_study_workspace_source_has_no_fix_confirm_tab() -> None:
    source = Path("app.py").read_text(encoding="utf-8")
    start = source.index("def show_study_workspace(")
    end = source.index("\ndef study_hand_label(", start)
    body = source[start:end]
    assert '["1 · Replay", "2 · Analyze"]' in body
    assert "Fix & confirm" not in body
    assert "render_study_fix_and_confirm" not in body
