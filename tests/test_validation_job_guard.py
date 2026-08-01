"""The Import validation panel must only explain a hand with its own job's frames.

Round 10 found the guard wired to `hands.notes`, which the same panel lets the
operator edit, and pinned by a test that exercised the note parser rather than
the guard — deleting the guard outright left the suite green.
"""
from __future__ import annotations

import pytest

import app
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Action, Hand, Session
from poker_tracker.services.validated_hand_import import CV_TIMELINE_IDENTITY_KEY
from poker_tracker.ui.reconstruction_review import ValidationFrameContext

NOTES = "CV draft from YOLO card timeline. timeline=/x/job_1_timeline.json"


def _db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def _hand_with_identity(db: PokerDatabase, *, job_id: int | None, notes: str) -> Hand:
    session = db.create_session(Session(name="Guard"))
    evidence = {}
    if job_id is not None:
        evidence[CV_TIMELINE_IDENTITY_KEY] = {
            "job_id": job_id,
            "timeline_hand_number": 1,
        }
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            notes=notes,
            completion_evidence=evidence,
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            street="preflop",
            action_index=1,
            player_name="Seat1",
            position="UTG",
            action_type="call",
            amount=3.0,
        )
    )
    return hand


def test_identity_prefers_the_stamp_over_editable_notes() -> None:
    """The stamp is what chose this hand for this job; notes are editable from
    the very panel the guard protects."""
    db = _db()
    hand = _hand_with_identity(db, job_id=1, notes="operator rewrote these notes")
    assert app._cv_timeline_identity(hand)[0] == 1
    db.close()


def test_identity_falls_back_to_notes_when_unstamped() -> None:
    db = _db()
    hand = _hand_with_identity(db, job_id=None, notes=NOTES)
    assert app._cv_timeline_identity(hand)[0] == 1
    db.close()


def test_editing_the_notes_does_not_strand_a_stamped_hand() -> None:
    """A notes edit must not permanently block the backfill: the write is
    one-shot, so a blocked hand stays unexplained forever."""
    db = _db()
    hand = _hand_with_identity(db, job_id=1, notes="notes cleared by operator")
    assert app._cv_timeline_identity(hand)[0] == 1, (
        "a notes edit stranded a hand whose job is recorded in its evidence"
    )
    db.close()


def _context(job_id: int) -> ValidationFrameContext:
    return ValidationFrameContext(
        job_id=job_id,
        hand_number=1,
        timeline_hand={"hand_number": 1, "actions": []},
        states=[],
        reviews_by_image={},
        cursor_key="c",
        pending_hand_key="p",
    )


@pytest.mark.parametrize(
    ("viewing_job", "expected"),
    [(1, True), (2, False), (3, False)],
)
def test_only_the_owning_job_may_explain_the_hand(
    viewing_job: int, expected: bool
) -> None:
    """Several jobs run on one video and all resolve to the same hand."""
    db = _db()
    hand = _hand_with_identity(db, job_id=1, notes=NOTES)
    assert (
        app.frame_context_belongs_to_hand(hand, _context(viewing_job)) is expected
    )
    db.close()


def test_the_guard_ignores_notes_an_operator_rewrote() -> None:
    """The guard itself — not the parser — must read the stamp. Wiring it to
    the notes lets a notes edit either strand the hand forever (the write is
    one-shot) or hand it to the wrong job."""
    db = _db()
    # Stamped job 1, but the notes now name job 3.
    hand = _hand_with_identity(
        db,
        job_id=1,
        notes="CV draft from YOLO card timeline. timeline=/x/job_3_timeline.json",
    )
    assert app.frame_context_belongs_to_hand(hand, _context(1)) is True
    assert app.frame_context_belongs_to_hand(hand, _context(3)) is False
    db.close()


def test_a_foreign_jobs_context_is_dropped_before_anything_is_derived() -> None:
    """A10/A11: the guard block could be deleted outright with a green suite,
    because every test pinned the helper and none pinned the decision the
    panel actually makes."""
    db = _db()
    hand = _hand_with_identity(db, job_id=1, notes=NOTES)

    kept, notice = app.usable_frame_context(hand, _context(1))
    assert kept is not None
    assert notice == ""

    dropped, notice = app.usable_frame_context(hand, _context(3))
    assert dropped is None, "a foreign job's frames were kept"
    assert "imported from job 1" in notice
    assert "viewing job 3" in notice
    db.close()


def test_no_context_is_passed_through_untouched() -> None:
    db = _db()
    hand = _hand_with_identity(db, job_id=1, notes=NOTES)
    assert app.usable_frame_context(hand, None) == (None, "")
    db.close()


def test_an_unstamped_hand_is_not_stranded() -> None:
    db = _db()
    hand = _hand_with_identity(db, job_id=None, notes=NOTES)
    assert app.frame_context_belongs_to_hand(hand, _context(1)) is True
    db.close()


def _timeline_hand(job: int) -> dict:
    return {
        "hand_number": 1,
        "actions": [
            {
                "street": "preflop", "action_index": 1, "seat": 1,
                "player_name": "Seat1", "position": "UTG",
                "action_type": "call", "amount": 3.0,
                "source_image": f"/frames/cv_job_{job}/t000000.00.jpg",
            }
        ],
    }


def _job_context(job: int) -> ValidationFrameContext:
    return ValidationFrameContext(
        job_id=job,
        hand_number=1,
        timeline_hand=_timeline_hand(job),
        states=[],
        reviews_by_image={},
        cursor_key="c",
        pending_hand_key="p",
    )


def test_the_panel_never_writes_a_foreign_jobs_frames() -> None:
    """A12 round 12 F1: the guard function was well tested, but deleting its
    CALL SITE left the suite green — and the next line writes to the database.
    A cross-job backfill is one-shot, so it cannot be undone through the app."""
    db = _db()
    hand = _hand_with_identity(db, job_id=1, notes=NOTES)

    context, notice = app.prepare_hand_frames(db, hand, _job_context(3))
    assert context is None
    assert "imported from job 1" in notice
    stored = db.fetch_actions_by_hand(hand.id)[0].source_image
    assert stored is None, f"a foreign job's frame was written: {stored}"
    db.close()


def test_the_panel_repairs_provenance_from_the_owning_job() -> None:
    """A12 round 12 F2: deleting the backfill call also survived, and 31 of
    the live database's 33 rows get provenance only from it."""
    db = _db()
    hand = _hand_with_identity(db, job_id=1, notes=NOTES)
    assert db.fetch_actions_by_hand(hand.id)[0].source_image is None

    context, notice = app.prepare_hand_frames(db, hand, _job_context(1))
    assert context is not None
    assert notice == ""
    assert (
        db.fetch_actions_by_hand(hand.id)[0].source_image
        == "/frames/cv_job_1/t000000.00.jpg"
    ), "the panel did not repair this hand's provenance"
    db.close()
