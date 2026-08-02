"""Regressions for the Phase 1 study-readiness bypasses found in adversarial round 1.

Every test here encodes one concrete way a hand that its own stored evidence
proves is unproven was able to reach ``is_ready`` / ``review_status='reviewed'``.
They are grouped in one file because they all defend the same invariant:

    ``completion_status == 'complete'`` is only ever true when
    ``derive_completion_status`` says so, on every writer and every reader.
"""

from __future__ import annotations

from typing import Any

import pytest

from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    CompletionEvidence,
    acknowledge_codes,
    derive_completion_status,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import import_session
from poker_tracker.persistence.models import (
    Hand,
    HandIssue,
    HandReview,
    Session,
)
from poker_tracker.services.hand_accounting import AccountingReconciliation
from poker_tracker.services.study_readiness import evaluate_study_readiness


def _reconciled() -> AccountingReconciliation:
    return AccountingReconciliation(
        ledger=None,  # type: ignore[arg-type]
        settlement=None,
        entries=(),
        issues=(),
        is_authoritative=True,
    )


def _evidence(**overrides: Any) -> dict[str, object]:
    values: dict[str, Any] = {
        "evidence_version": EVIDENCE_SCHEMA_VERSION,
        "partial_start": False,
        "partial_end": False,
        "terminal_event": "showdown",
        "boundary_confidence": 0.93,
        "layout_supported": True,
        "table_size": 6,
    }
    values.update(overrides)
    return dump_completion_evidence(CompletionEvidence(**values))


def _payload(**hand_overrides: Any) -> dict[str, Any]:
    hand: dict[str, Any] = {
        "session_id": 0,
        "hand_number": 1,
        "hero_cards": "Ah Kd",
        "board_cards": "2c 7d 9s",
        "table_size": 6,
        "review_status": "reviewed",
        "source_type": "cv_import",
        "completion_status": "complete",
        "completion_evidence": _evidence(),
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    hand.update(hand_overrides)
    return {
        "export_version": 5,
        "session": {"name": "Attack", "date_played": "2026-01-01"},
        "hands": [{"hand": hand, "players": [], "actions": [], "reviews": []}],
    }


def _memory_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


# --------------------------------------------------------------------------- #
# Import must re-derive completion, never trust a declared one
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "evidence", "expected_status"),
    [
        ("truncated at both ends", _evidence(partial_start=True, partial_end=True), "partial"),
        ("truncated at the end", _evidence(partial_end=True), "partial"),
        ("no readable evidence", {}, "uncertain"),
        (
            "pipeline rejected the hand",
            _evidence(rejection_codes=("duplicate_card_detected",)),
            "uncertain",
        ),
        ("terminal event unobserved", _evidence(terminal_event="unobserved"), "uncertain"),
    ],
)
def test_import_rederives_a_declared_complete_status_from_the_evidence(
    label: str, evidence: dict[str, object], expected_status: str
) -> None:
    """A v5 payload may declare anything; only the evidence decides."""
    db = _memory_db()
    session = import_session(db, _payload(completion_evidence=evidence))
    hand = db.fetch_hands_by_session(session.id)[0]

    assert hand.completion_status == expected_status, label
    assert hand.review_status == "needs_correction", label

    readiness = evaluate_study_readiness(
        hand, accounting=_reconciled(), user_confirmed=True
    )
    assert readiness.is_ready is False, label
    assert readiness.has("COMPLETION_NOT_COMPLETE"), label
    db.close()


def test_import_demotes_reviewed_even_on_a_genuinely_complete_reconstructed_hand() -> None:
    """Completion is re-derived and kept; the promotion is not.

    Readiness requires explicit user confirmation, which is derived per render and
    never persisted, so it cannot travel in a payload -- the importing operator
    has not seen this hand's evidence. The v13 migration applies the same rule to
    every reconstructed row it finds.
    """
    db = _memory_db()
    session = import_session(db, _payload())
    hand = db.fetch_hands_by_session(session.id)[0]

    assert hand.completion_status == "complete"
    assert hand.review_status == "needs_correction"
    db.close()


def test_import_refuses_reviewed_when_the_payload_carries_an_open_issue() -> None:
    """update_hand_status refuses this exact state; create_hand must not walk around it."""
    payload = _payload(
        source_type="manual", completion_status="not_applicable", completion_evidence={}
    )
    payload["hands"][0]["issues"] = [
        {
            "hand_id": 0,
            "status": "open",
            "issue_types": ["pot_or_result"],
            "description": "winner looks wrong",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    db = _memory_db()
    session = import_session(db, payload)
    hand = db.fetch_hands_by_session(session.id)[0]

    assert [issue.status for issue in db.fetch_hand_issues(hand_id=hand.id)] == ["open"]
    assert hand.review_status == "needs_correction"
    db.close()


# --------------------------------------------------------------------------- #
# Readiness and the store must cross-check the column against the evidence
# --------------------------------------------------------------------------- #
def test_readiness_blocks_a_complete_column_that_the_evidence_contradicts() -> None:
    """Defence in depth for rows written outside import_session."""
    hand = Hand(
        id=3,
        session_id=1,
        hand_number=1,
        table_size=6,
        hero_cards="Ah Kd",
        board_cards="2c 7d 9s",
        source_type="cv_import",
        completion_status="complete",
        completion_evidence=_evidence(partial_end=True),
    )
    assert derive_completion_status(
        parse_completion_evidence(hand.completion_evidence), source_type="cv_import"
    ) == "partial"

    readiness = evaluate_study_readiness(
        hand, accounting=_reconciled(), user_confirmed=True
    )

    assert readiness.is_ready is False
    assert readiness.has("COMPLETION_NOT_COMPLETE")


def test_store_refuses_reviewed_when_the_evidence_contradicts_the_column() -> None:
    db = _memory_db()
    session = db.create_session(Session(name="Forged", date_played="2026-01-01"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_evidence(partial_start=True),
        )
    )

    with pytest.raises(ValueError, match="cannot be marked reviewed"):
        db.update_hand_status(hand.id, "reviewed")
    assert db.fetch_hand(hand.id).review_status != "reviewed"
    db.close()


# --------------------------------------------------------------------------- #
# A rejection code is the pipeline refusing the hand; acknowledging cannot clear it
# --------------------------------------------------------------------------- #
def test_acknowledging_a_rejection_code_never_promotes_a_hand_to_complete() -> None:
    db = _memory_db()
    session = db.create_session(Session(name="Ack", date_played="2026-01-01"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            source_type="cv_import",
            completion_evidence=_evidence(
                warning_codes=("pot_not_reconciled",),
                rejection_codes=("duplicate_card_detected", "stack_overcommit"),
            ),
        )
    )
    assert hand.completion_status == "uncertain"

    evidence = parse_completion_evidence(hand.completion_evidence)
    for code in ("pot_not_reconciled", "duplicate_card_detected", "stack_overcommit"):
        evidence = acknowledge_codes(evidence, [code])
        hand = db.update_hand_completion(
            hand.id, completion_evidence=dump_completion_evidence(evidence)
        )

    assert hand.completion_status == "uncertain"
    assert "duplicate_card_detected" not in evidence.acknowledged_codes
    readiness = evaluate_study_readiness(
        hand, accounting=_reconciled(), user_confirmed=True
    )
    assert readiness.is_ready is False
    with pytest.raises(ValueError, match="cannot be marked reviewed"):
        db.update_hand_status(hand.id, "reviewed")
    db.close()


def test_a_hand_edited_acknowledgement_cannot_launder_a_rejection_code() -> None:
    """acknowledge_codes refuses rejection codes, but the stored blob is just JSON.

    A row whose acknowledged_codes already lists a rejection -- written by a
    hand-edited database or a future producer -- must still derive 'uncertain',
    which is why derive_completion_status checks rejection_codes directly.
    """
    laundered = parse_completion_evidence(
        {
            **_evidence(rejection_codes=("duplicate_card_detected",)),
            "acknowledged_codes": ["duplicate_card_detected"],
        }
    )

    assert laundered.unresolved_codes == ()
    assert derive_completion_status(laundered, source_type="cv_import") == "uncertain"


def test_acknowledging_a_warning_code_still_promotes_to_complete() -> None:
    """The rejection-code fix must not break the documented warning workflow."""
    db = _memory_db()
    session = db.create_session(Session(name="Warn", date_played="2026-01-01"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            source_type="cv_import",
            completion_evidence=_evidence(warning_codes=("pot_not_reconciled",)),
        )
    )
    evidence = acknowledge_codes(
        parse_completion_evidence(hand.completion_evidence), ["pot_not_reconciled"]
    )

    updated = db.update_hand_completion(
        hand.id, completion_evidence=dump_completion_evidence(evidence)
    )

    assert updated.completion_status == "complete"
    db.close()


# --------------------------------------------------------------------------- #
# Legacy hand_reviews coaching is retained evidence too
# --------------------------------------------------------------------------- #
def test_stale_legacy_hand_review_blocks_study_readiness() -> None:
    db = _memory_db()
    session = db.create_session(Session(name="Stale", date_played="2026-01-01"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            source_type="manual",
        )
    )
    db.create_hand_review(
        HandReview(
            hand_id=hand.id,
            hand_summary="summary",
            theory_coach="Bet 2/3 pot for value.",
            exploit_coach="exploit",
            study_lesson="lesson",
        )
    )
    corrected = db.update_hand_facts(hand.model_copy(update={"board_cards": "Qd 7s 3c"}))
    assert [review.is_stale for review in db.fetch_reviews_by_hand(hand.id)] == [True]

    readiness = evaluate_study_readiness(
        corrected,
        accounting=_reconciled(),
        hand_reviews=db.fetch_reviews_by_hand(hand.id),
        user_confirmed=True,
    )

    assert readiness.is_ready is False
    assert readiness.has("STALE_COACHING_EVIDENCE")
    db.close()


# --------------------------------------------------------------------------- #
# Unpinned readiness rules (surviving mutants)
# --------------------------------------------------------------------------- #
def _reconstructed(**overrides: Any) -> Hand:
    values: dict[str, Any] = {
        "id": 11,
        "session_id": 1,
        "hand_number": 1,
        "table_size": 6,
        "hero_cards": "Ah Kd",
        "board_cards": "2c 7d 9s",
        "source_type": "cv_import",
        "completion_status": "complete",
        "completion_evidence": _evidence(),
    }
    values.update(overrides)
    return Hand(**values)


@pytest.mark.parametrize("table_size", [1, 11, 1000])
def test_layout_blocker_rejects_a_table_size_outside_the_supported_window(
    table_size: int,
) -> None:
    hand = _reconstructed(completion_evidence=_evidence(table_size=table_size))

    readiness = evaluate_study_readiness(
        hand, accounting=_reconciled(), user_confirmed=True
    )

    assert readiness.has("UNSUPPORTED_TABLE_LAYOUT")


@pytest.mark.parametrize("table_size", [2, 6, 10])
def test_layout_blocker_accepts_a_table_size_inside_the_supported_window(
    table_size: int,
) -> None:
    # The hand's own column carries the same seat count: readiness now also
    # requires the two to agree, and this test is about the supported window.
    hand = _reconstructed(
        table_size=table_size,
        completion_evidence=_evidence(table_size=table_size),
    )

    readiness = evaluate_study_readiness(
        hand, accounting=_reconciled(), user_confirmed=True
    )

    assert not readiness.has("UNSUPPORTED_TABLE_LAYOUT")


@pytest.mark.parametrize("hero_cards", ["", "Ah", "Ah Kd Qs"])
def test_card_blocker_requires_exactly_two_hero_cards_on_a_reconstructed_hand(
    hero_cards: str,
) -> None:
    """Hand validation allows 0 or 2 hero cards, and rows written outside the model
    may hold 1 or 3; a reconstructed hand always needs exactly 2."""
    hand = _reconstructed().model_copy(update={"hero_cards": hero_cards})

    readiness = evaluate_study_readiness(
        hand, accounting=_reconciled(), user_confirmed=True
    )

    assert readiness.has("INVALID_HERO_OR_BOARD_CARDS")
    assert any(
        "exactly 2 cards" in detail
        for blocker in readiness.blockers
        for detail in blocker.detail
    )


def test_card_blocker_accepts_two_hero_cards_on_a_reconstructed_hand() -> None:
    readiness = evaluate_study_readiness(
        _reconstructed(), accounting=_reconciled(), user_confirmed=True
    )

    assert not readiness.has("INVALID_HERO_OR_BOARD_CARDS")


def test_accounting_blocker_fires_on_an_error_even_when_the_ledger_reconciles() -> None:
    hand = _reconstructed()

    readiness = evaluate_study_readiness(
        hand,
        accounting=_reconciled(),
        accounting_error="Settlement could not be loaded.",
        user_confirmed=True,
    )

    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")


# --------------------------------------------------------------------------- #
# A damaged evidence blob must degrade, never raise out of the store
# --------------------------------------------------------------------------- #
def test_a_binary_evidence_blob_degrades_instead_of_breaking_the_hand_list() -> None:
    db = _memory_db()
    session = db.create_session(Session(name="Blob", date_played="2026-01-01"))
    db.create_hand(Hand(session_id=session.id, hand_number=1, source_type="manual"))
    damaged = db.create_hand(
        Hand(session_id=session.id, hand_number=2, source_type="manual")
    )
    db._execute(
        "UPDATE hands SET completion_evidence = x'deadbeef' WHERE id = ?", (damaged.id,)
    )
    db._commit()

    hands = db.fetch_hands_by_session(session.id)

    assert len(hands) == 2
    assert parse_completion_evidence(hands[-1].completion_evidence).is_known is False
    # Degrading is not the same as forgetting: the damaged bytes are recorded as
    # an unreadable column, so the hand cannot pass as one that simply carries no
    # evidence, and the undamaged hand beside it is untouched.
    assert hands[-1].unreadable_columns == ("completion_evidence",)
    assert hands[-1].review_status == "needs_correction"
    assert hands[0].unreadable_columns == ()
    db.close()


@pytest.mark.parametrize("version", [1.9, 1.0000001, 0.5])
def test_a_non_integer_evidence_version_is_unreadable(version: float) -> None:
    parsed = parse_completion_evidence(
        {"evidence_version": version, "partial_start": False, "partial_end": False}
    )

    assert parsed.evidence_version == 0
    assert parsed.is_known is False


def test_the_ui_guard_refuses_a_promotion_the_store_would_accept() -> None:
    """The case guarded_update_hand_status uniquely covers.

    The store's floor only reads completion_status, source_type, and open issues,
    so it accepts this hand. Readiness blocks it on accounting. Deleting the
    guard's refusal branch must therefore turn this test red.
    """
    import app as app_module

    db = _memory_db()
    session = db.create_session(Session(name="Guard", date_played="2026-01-01"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Kd",
            board_cards="2c 7d 9s",
            source_type="cv_import",
            review_status="needs_correction",
            completion_status="complete",
            completion_evidence=_evidence(),
        )
    )
    blocked = evaluate_study_readiness(hand, accounting=None, user_confirmed=True)
    assert blocked.codes() == ("ACCOUNTING_NOT_AUTHORITATIVE",)

    # The store on its own would say yes, which is exactly why the guard exists.
    db.update_hand_status(hand.id, "reviewed")
    assert db.fetch_hand(hand.id).review_status == "reviewed"
    db.update_hand_status(hand.id, "needs_correction")

    assert app_module.guarded_update_hand_status(db, hand, blocked, "reviewed") is False
    assert db.fetch_hand(hand.id).review_status == "needs_correction"
    db.close()


def test_open_issue_from_import_is_visible_to_readiness() -> None:
    """Sanity: the imported issue still blocks, so the demotion is not the only guard."""
    payload = _payload(
        source_type="manual", completion_status="not_applicable", completion_evidence={}
    )
    payload["hands"][0]["issues"] = [
        {
            "hand_id": 0,
            "status": "open",
            "issue_types": ["pot_or_result"],
            "description": "winner looks wrong",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    db = _memory_db()
    session = import_session(db, payload)
    hand = db.fetch_hands_by_session(session.id)[0]

    readiness = evaluate_study_readiness(
        hand,
        accounting=_reconciled(),
        hand_issues=db.fetch_hand_issues(hand_id=hand.id),
        user_confirmed=True,
    )

    assert readiness.has("OPEN_DEBUGGING_ISSUE")
    assert isinstance(db.fetch_hand_issues(hand_id=hand.id)[0], HandIssue)
    db.close()
