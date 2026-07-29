"""Regressions for the round-4 adversarial findings against Phase 1.

Every test here failed before its fix. The round-4 theme is *self-satisfying
evidence*: a settlement field that was also the tolerance every cross-check was
judged against, a payload that supplied that field itself, a non-finite float
that derived ``complete`` and then left the export unparseable, and a hand-edited
card column that took the whole session's hand list down instead of blocking.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import (
    _completion_evidence_for_hand,
)
from poker_tracker.maintenance.data_health import audit_data_health
from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence.backup import (
    BACKUP_KEEP_COUNT,
    PINNED_KEEP_COUNT,
    PINNED_PREFIX,
    backup_database,
)
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    CompletionEvidence,
    acknowledge_codes,
    derive_completion_status,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import DECLARED_DEAD_MONEY_CODE, PokerDatabase
from poker_tracker.persistence.import_export import (
    export_session_json,
    import_session,
)
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandIssue,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
    SolverRun,
)
from poker_tracker.services.hand_accounting import persist_reconciliation
from poker_tracker.services.study_readiness import evaluate_study_readiness
from tests.conftest import attest_declared_assumptions


def _clean_evidence_blob(**overrides: object) -> dict[str, object]:
    payload = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="showdown",
            boundary_confidence=0.92,
            layout_supported=True,
            table_size=6,
        )
    )
    payload.update(overrides)
    return payload


def _open_db(tmp_path: Path) -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / "round4.db"))
    db.init_db()
    return db


def _seed_contradicted_hand(db: PokerDatabase) -> Hand:
    """A hand whose recorded pot (95) and hero result (+85) contradict its ledger.

    The observed action line is a river bet/call of 10 each: 20 chips gross and
    +10 for the hero. One award of 95 is declared to the hero.
    """
    session = db.create_session(Session(name="Round 4", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=95.0,
            hero_bb_won=85.0,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence_blob(),
        )
    )
    assert hand.id is not None
    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            player_name="Hero",
            is_hero=True,
            starting_stack=1000,
        )
    )
    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="villain",
            player_name="Villain",
            starting_stack=1000,
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="hero",
            street="river",
            action_index=1,
            player_name="Hero",
            action_type="bet",
            amount=10,
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="villain",
            street="river",
            action_index=2,
            player_name="Villain",
            action_type="call",
            amount=10,
        )
    )
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="hero",
                player_name="Hero",
                amount=95.0,
                entry_order=1,
            )
        ],
    )
    return hand


# ---------------------------------------------------------------------------
# Findings B1 and B2 -- the chip denomination was also the correctness tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chip_unit", [0.01, 1.0, 200.0, 100000.0])
def test_the_chip_unit_cannot_reconcile_a_contradicted_hand(
    tmp_path: Path, chip_unit: float
) -> None:
    """`rake_rounding_unit` is a chip denomination, not a licence to disagree.

    It used to be fed straight into one tolerance that gated every accounting
    cross-check, so raising it silenced the pot, hero-result, refund and award
    comparisons at once: at 200 the hand below reconciled, became authoritative,
    and rendered study-ready with zero blockers.
    """
    db = _open_db(tmp_path)
    hand = _seed_contradicted_hand(db)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rounding_unit=chip_unit)
    )
    result = persist_reconciliation(db, hand.id)

    assert result.settlement is not None
    assert result.settlement.status == "needs_correction"
    assert result.is_authoritative is False
    assert "Observed final pot does not match the derived gross pot." in result.issues

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(
        stored, accounting=result, user_confirmed=True
    )
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")


def test_an_honest_rake_rounding_unit_still_absorbs_its_own_rounding(
    tmp_path: Path,
) -> None:
    """An honest rake policy still reconciles, and no recorded figure is excused.

    Rounds 4-6 granted a recorded ``rake_amount`` a tolerance derived from the
    settlement's own ``rake_rounding_unit`` so that "the same policy read one
    rounding step earlier" was not an issue. Round 7 removed that tolerance:
    every input it was derived from also arrives in an import payload, and
    ``import_session`` never rewrites the recorded pair, so the payload set both
    sides of its own comparison. What this test still protects is the original
    intent -- rake rounding itself must not become a mismatch -- verified through
    the path the product actually writes: the settlement editor nulls the
    recorded figures and ``persist_reconciliation`` re-derives them.
    """
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Rake", date_played=date(2026, 1, 1)))
    assert session.id is not None
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
    assert hand.id is not None
    for key, name, hero in (("hero", "Hero", True), ("villain", "Villain", False)):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                is_hero=hero,
                starting_stack=1000,
            )
        )
    for index, (key, name, kind) in enumerate(
        (("hero", "Hero", "bet"), ("villain", "Villain", "call")), start=1
    ):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=name,
                action_type=kind,
                amount=33,
            )
        )
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="hero",
                player_name="Hero",
                amount=63.0,
                entry_order=1,
            )
        ],
    )
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            rake_rate=0.05,
            rake_rounding_unit=1.0,
            # 66 * 0.05 = 3.3, rounded down to 3. The stored policy and the
            # stored amount disagree, and nothing here can know which is right.
            rake_amount=3.3,
        )
    )
    from poker_tracker.services.hand_accounting import reconcile_persisted_hand

    # The read-only path every readiness surface uses.
    result = reconcile_persisted_hand(db, hand.id)
    assert result.ledger.rake == pytest.approx(3.0)
    assert "Recorded rake does not match the derived ledger." in result.issues
    assert result.is_authoritative is False

    # And the product refuses to call it reconciled...
    result = persist_reconciliation(db, hand.id)
    assert result.settlement is not None
    assert result.settlement.status == "needs_correction"

    # ...while the honest policy itself still reconciles once the recorded
    # figure is the product's own, re-derived from the ledger.
    result = persist_reconciliation(db, hand.id)
    assert result.ledger.rake == pytest.approx(3.0)
    assert result.settlement is not None
    assert result.settlement.rake_amount == pytest.approx(3.0)
    assert result.settlement.status == "reconciled"
    assert result.is_authoritative is True


def test_an_imported_settlement_cannot_set_its_own_reconciliation_tolerance(
    tmp_path: Path,
) -> None:
    """One import call used to land a study-ready hand with a fabricated pot.

    The payload declared `rake_rounding_unit: 4000.0`, which became a tolerance
    of 2000 and swallowed every mismatch, so `_enforce_review_status_floor`'s
    stated safety argument -- that readiness still blocks what the store floor
    cannot see -- was false.
    """
    db = _open_db(tmp_path)
    payload = {
        "export_version": 5,
        "session": {
            "name": "Forged",
            "date_played": "2026-01-01",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        "hands": [
            {
                "hand": {
                    "hand_number": 1,
                    "table_size": 6,
                    "hero_cards": "Ah Qs",
                    "board_cards": "Qd 7s 2c",
                    "review_status": "reviewed",
                    "source_type": "manual",
                    "completion_status": "not_applicable",
                    "completion_evidence": {},
                    "pot_size": 900.0,
                    "hero_bb_won": 900.0,
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                "players": [
                    {
                        "player_key": "hero",
                        "player_name": "Hero",
                        "is_hero": True,
                        "starting_stack": 1000,
                    },
                    {
                        "player_key": "villain",
                        "player_name": "Villain",
                        "is_hero": False,
                        "starting_stack": 1000,
                    },
                ],
                "actions": [
                    {
                        "player_key": "hero",
                        "street": "river",
                        "action_index": 1,
                        "player_name": "Hero",
                        "action_type": "bet",
                        "amount": 10,
                        "amount_semantics": "incremental",
                    },
                    {
                        "player_key": "villain",
                        "street": "river",
                        "action_index": 2,
                        "player_name": "Villain",
                        "action_type": "call",
                        "amount": 10,
                        "amount_semantics": "incremental",
                    },
                ],
                "settlement": {"status": "reconciled", "rake_rounding_unit": 4000.0},
                "settlement_entries": [
                    {
                        "entry_type": "award",
                        "pot_index": 0,
                        "player_key": "hero",
                        "player_name": "Hero",
                        "amount": 900.0,
                        "entry_order": 1,
                    }
                ],
                "reviews": [],
                "coaching_reviews": [],
                "corrections": [],
                "issues": [],
            }
        ],
        "coaching_reviews": [],
    }
    session = import_session(db, payload)
    assert session.id is not None
    imported = db.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None

    from poker_tracker.services.hand_accounting import reconcile_persisted_hand

    accounting = reconcile_persisted_hand(db, imported.id)
    assert accounting.is_authoritative is False
    readiness = evaluate_study_readiness(imported, accounting=accounting)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")


def test_declared_dead_money_is_disclosed_on_a_reconstructed_hand(
    tmp_path: Path,
) -> None:
    """Dead money can always be tuned until a fabricated pot reconciles exactly.

    75 unobserved chips move the derived gross pot from 20 to exactly 95 and the
    hero's derived net from +10 to exactly +85, so every mismatch vanishes at the
    strict default tolerance and the hand became study-ready with zero blockers
    and nothing recording that the verdict rested on chips nothing observed.
    """
    db = _open_db(tmp_path)
    hand = _seed_contradicted_hand(db)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, dead_money=75.0))
    result = persist_reconciliation(db, hand.id)
    # The ledger models the declaration faithfully -- that is not the defect.
    assert result.ledger.gross_pot == pytest.approx(95.0)
    assert result.is_authoritative is True

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(
        stored, accounting=result, user_confirmed=True
    )
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_ASSUMPTION_DEPENDENT")

    # AMENDED in round 12. The disclosure is recorded in the OPERATOR's evidence
    # channel, never the pipeline's: writing it into `warning_codes` made a hand
    # whose reconstruction evidence was complete and clean report that "the
    # pipeline could not prove this hand was fully reconstructed", and offered the
    # generic Acknowledge as the answer to a declared chip movement.
    evidence = parse_completion_evidence(stored.completion_evidence)
    assert DECLARED_DEAD_MONEY_CODE in evidence.declared_settlement_codes
    assert DECLARED_DEAD_MONEY_CODE not in evidence.warning_codes
    assert DECLARED_DEAD_MONEY_CODE not in evidence.unresolved_codes
    assert readiness.has("UNRESOLVED_SOURCE_WARNING") is False

    # And the generic Acknowledge is not the answer to it: only an attestation to
    # the measured chip movement clears the blocker.
    db.update_hand_completion(
        hand.id,
        completion_evidence=dump_completion_evidence(
            acknowledge_codes(evidence, [DECLARED_DEAD_MONEY_CODE])
        ),
        notes="Attempted to acknowledge the declared dead money.",
    )
    unmoved = db.fetch_hand(hand.id)
    assert unmoved is not None
    assert (
        evaluate_study_readiness(
            unmoved, accounting=result, user_confirmed=True
        ).has("ACCOUNTING_ASSUMPTION_DEPENDENT")
        is True
    )
    attest_declared_assumptions(db, hand.id)
    attested = db.fetch_hand(hand.id)
    assert attested is not None
    assert (
        evaluate_study_readiness(
            attested, accounting=result, user_confirmed=True
        ).has("ACCOUNTING_ASSUMPTION_DEPENDENT")
        is False
    )

    # Withdrawing the declaration withdraws the disclosure with it.
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, dead_money=0.0))
    cleared = db.fetch_hand(hand.id)
    assert cleared is not None
    cleared_evidence = parse_completion_evidence(cleared.completion_evidence)
    assert DECLARED_DEAD_MONEY_CODE not in cleared_evidence.declared_settlement_codes
    assert DECLARED_DEAD_MONEY_CODE not in cleared_evidence.warning_codes


def test_a_manual_hand_may_declare_dead_money_without_a_source_warning(
    tmp_path: Path,
) -> None:
    """Antes and dead blinds are real. The disclosure is a reconstruction claim."""
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Ante", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, source_type="manual")
    )
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, dead_money=3.0))
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.completion_status == "not_applicable"
    assert stored.completion_evidence == {}


# ---------------------------------------------------------------------------
# Finding A1 -- non-finite floats in completion evidence
# ---------------------------------------------------------------------------


def test_non_finite_boundary_confidence_is_unreadable_not_close_enough() -> None:
    """NaN is not a confidence. It used to pass the `is None` gate and derive complete."""
    for value in (float("nan"), float("inf"), float("-inf")):
        evidence = parse_completion_evidence(
            _clean_evidence_blob(boundary_confidence=value)
        )
        assert evidence.boundary_confidence is None
        assert derive_completion_status(evidence, source_type="cv_import") == "uncertain"


def test_a_version_5_export_is_always_strict_json(tmp_path: Path) -> None:
    """Bare NaN/Infinity tokens are readable by Python and by nothing else.

    Python's `json.loads` is lenient, so the app could re-read its own broken
    export and restore `boundary_confidence: nan` with `completion_status:
    complete` -- the bad value was self-perpetuating.
    """
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="NaN", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="As Ks",
            board_cards="2h 7d 3c",
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence_blob(
                boundary_confidence=float("nan"),
                first_source_timestamp_s=float("inf"),
            ),
        )
    )
    assert hand.id is not None

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    # The unreadable value can never justify a promotion: the stored column and
    # the stored evidence now disagree, which both the store and readiness catch.
    assert parse_completion_evidence(stored.completion_evidence).boundary_confidence is None
    assert evaluate_study_readiness(
        stored, accounting=None, user_confirmed=True
    ).has("COMPLETION_NOT_COMPLETE")
    with pytest.raises(ValueError):
        db.update_hand_status(hand.id, "reviewed")

    destination = tmp_path / "export.json"
    export_session_json(db, session.id, destination)
    text = destination.read_text(encoding="utf-8")

    def _reject(constant: str) -> float:
        raise AssertionError(f"export emitted a non-JSON token: {constant}")

    json.loads(text, parse_constant=_reject)


# ---------------------------------------------------------------------------
# Finding C1 -- a hand-edited card column took the whole session down
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("board_cards", "2c 7d"),
        ("board_cards", "Ah 7d 9s"),  # duplicates a hero card
        ("hero_cards", "Ah Kd Qs"),
        ("hero_cards", "zz xx"),
    ],
)
def test_a_hand_edited_card_row_blocks_instead_of_hiding_the_session(
    tmp_path: Path, column: str, value: str
) -> None:
    """INVALID_HERO_OR_BOARD_CARDS claims to guard rows written outside the model.

    It could not: `Hand` refused those values on read, so `fetch_hands_by_session`
    raised a pydantic ValidationError and every other hand in the session
    disappeared with it. Two of the blocker's three branches were unreachable.
    """
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Edited", date_played=date(2026, 1, 1)))
    assert session.id is not None
    corrupt = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="Ah Kd",
            board_cards="2c 7d 9s",
            source_type="manual",
        )
    )
    healthy = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=2,
            hero_cards="Js Td",
            board_cards="2c 7d 9s",
            source_type="manual",
        )
    )
    assert corrupt.id is not None and healthy.id is not None
    db._execute(  # noqa: SLF001 - simulating a hand-edited database
        f"UPDATE hands SET {column} = ? WHERE id = ?", (value, corrupt.id)
    )
    db._commit()  # noqa: SLF001

    hands = db.fetch_hands_by_session(session.id)
    assert [hand.hand_number for hand in hands] == [1, 2]

    blocked = next(hand for hand in hands if hand.hand_number == 1)
    readiness = evaluate_study_readiness(blocked, accounting=None, user_confirmed=True)
    assert readiness.has("INVALID_HERO_OR_BOARD_CARDS")

    intact = next(hand for hand in hands if hand.hand_number == 2)
    assert evaluate_study_readiness(
        intact, accounting=None, user_confirmed=True
    ).has("INVALID_HERO_OR_BOARD_CARDS") is False


@pytest.mark.parametrize(
    ("hero", "board", "expected"),
    [
        ("Ah Kd", "2c 7d", "board must hold"),
        ("Ah Kd", "Ah 7d 9s", "Duplicate"),
        ("Ah Kd Qs", "2c 7d 9s", "exactly 2"),
    ],
)
def test_the_card_blocker_branches_hold_if_the_model_ever_stops_refusing(
    hero: str, board: str, expected: str
) -> None:
    """Defence in depth, exercised directly rather than asserted in a docstring.

    `Hand` refuses these values on write and `_hand_from_row` blanks them on read,
    so no reachable path can hand them to `_card_problem`. Deleting the board-count
    branch outright therefore left the whole suite green. `model_construct` skips
    validation to reach the branches the way a future lenient parser would.
    """
    hand = Hand.model_construct(
        session_id=1,
        hand_number=1,
        source_type="cv_import",
        completion_status="uncertain",
        completion_evidence={},
        hero_cards=hero,
        board_cards=board,
    )
    readiness = evaluate_study_readiness(hand, accounting=None, user_confirmed=True)
    blocker = next(
        item for item in readiness.blockers if item.code == "INVALID_HERO_OR_BOARD_CARDS"
    )
    assert any(expected in line for line in blocker.detail)


# ---------------------------------------------------------------------------
# Finding C3 -- the open-only issue filter is the blocker's clearing action
# ---------------------------------------------------------------------------


def test_resolving_a_debugging_issue_clears_the_blocker() -> None:
    """OPEN_DEBUGGING_ISSUE tells the operator to resolve the issue.

    Nothing proved that doing so worked: no test anywhere passed a non-open
    issue into readiness, so removing the status filter left the suite green and
    the blocker would have named an action the product could not perform.
    """
    hand = Hand(
        session_id=1,
        hand_number=1,
        source_type="manual",
        completion_status="not_applicable",
        hero_cards="Ah Kd",
    )
    open_issue = HandIssue(
        hand_id=1, issue_types=["pot_or_result"], status="open", description="Bad pot."
    )
    resolved_issue = HandIssue(
        hand_id=1,
        issue_types=["pot_or_result"],
        status="resolved",
        description="Bad pot.",
        resolution_notes="Corrected the pot and re-reconciled.",
    )
    assert evaluate_study_readiness(
        hand, accounting=None, hand_issues=(open_issue,)
    ).has("OPEN_DEBUGGING_ISSUE")
    assert (
        evaluate_study_readiness(
            hand, accounting=None, hand_issues=(resolved_issue,)
        ).has("OPEN_DEBUGGING_ISSUE")
        is False
    )


# ---------------------------------------------------------------------------
# Finding C4 -- the live-run refusal disagreed with the store's own definition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["queued", "running", "cancelling"])
def test_delete_solver_run_refuses_every_run_the_store_calls_active(
    tmp_path: Path, status: str
) -> None:
    """`fetch_active_solver_runs` counts 'cancelling' as live; the refusal did not.

    A mid-cancel row could therefore vanish underneath the background worker, and
    the refusal had no test at all -- removing it left the whole suite green.
    """
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Solver", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, source_type="manual")
    )
    assert hand.id is not None
    run = db.create_solver_run(
        SolverRun(hand_id=hand.id, status=status, input_hash="hash")
    )
    assert run.id is not None
    assert [active.id for active in db.fetch_active_solver_runs()] == [run.id]
    with pytest.raises(ValueError, match="Cancel the solver run before deleting it."):
        db.delete_solver_run(run.id)
    assert db.fetch_solver_run(run.id) is not None


# ---------------------------------------------------------------------------
# Finding C9 -- the two stale-evidence tie-breaks disagreed with each other
# ---------------------------------------------------------------------------


def test_a_rerun_at_the_same_timestamp_clears_both_stale_blockers() -> None:
    """Coaching cleared on `>=` and the solver on `>`, with neither pinned.

    On equal timestamps the solver blocker stood while naming 'Re-run the solve'
    as its clearing action -- which had just been done.
    """
    moment = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)
    hand = Hand(
        session_id=1,
        hand_number=1,
        source_type="manual",
        completion_status="not_applicable",
        hero_cards="Ah Kd",
    )
    runs = [
        SolverRun(hand_id=1, status="stale", input_hash="a", created_at=moment),
        SolverRun(hand_id=1, status="completed", input_hash="b", created_at=moment),
    ]
    assert (
        evaluate_study_readiness(hand, accounting=None, solver_runs=runs).has(
            "STALE_SOLVER_EVIDENCE"
        )
        is False
    )
    shared = dict(
        hand_id=1,
        review_type="hand",
        provider_name="claude",
        model_name="model",
        raw_prompt="prompt",
        raw_response="response",
        theory_coach="theory",
        exploit_coach="exploit",
        hand_summary="summary",
        created_at=moment,
    )
    reviews = [
        CoachingResponse(is_stale=True, **shared),
        CoachingResponse(is_stale=False, **shared),
    ]
    assert (
        evaluate_study_readiness(hand, accounting=None, coaching_reviews=reviews).has(
            "STALE_COACHING_EVIDENCE"
        )
        is False
    )


# ---------------------------------------------------------------------------
# Finding A3 -- a blocker must not invent a fact about the operator's recording
# ---------------------------------------------------------------------------


def test_a_partial_column_its_evidence_contradicts_states_the_disagreement(
    tmp_path: Path,
) -> None:
    """The import ceiling honours a declared `partial` over a weaker re-derivation.

    The blocker then told the operator the recording "starts mid-hand" about a
    hand whose own evidence records partial_start=False and partial_end=False,
    and named re-importing from a complete recording -- exactly what they did.
    """
    db = _open_db(tmp_path)
    payload = {
        "export_version": 5,
        "session": {
            "name": "Ceiling",
            "date_played": "2026-01-01",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        "hands": [
            {
                "hand": {
                    "hand_number": 1,
                    "table_size": 6,
                    "hero_cards": "As Ks",
                    "board_cards": "2h 7d 3c",
                    "review_status": "unreviewed",
                    "source_type": "cv_import",
                    "completion_status": "partial",
                    "completion_evidence": _clean_evidence_blob(),
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                "players": [],
                "actions": [],
                "settlement": None,
                "settlement_entries": [],
                "reviews": [],
                "coaching_reviews": [],
                "corrections": [],
                "issues": [],
            }
        ],
        "coaching_reviews": [],
    }
    session = import_session(db, payload)
    assert session.id is not None
    hand = db.fetch_hands_by_session(session.id)[0]
    assert hand.completion_status == "partial"
    blocker = next(
        item
        for item in evaluate_study_readiness(
            hand, accounting=None, user_confirmed=True
        ).blockers
        if item.code == "COMPLETION_NOT_COMPLETE"
    )
    assert "starts mid-hand" not in blocker.reason
    assert "ends mid-hand" not in blocker.reason
    assert "both ends are truncated" not in blocker.reason
    assert "does not agree" in blocker.reason


# ---------------------------------------------------------------------------
# Finding C5 -- the exporter's layout rule had no negative case
# ---------------------------------------------------------------------------


def _export_evidence(**hand_overrides: object) -> CompletionEvidence:
    hand: dict[str, object] = {
        "source_images": [],
        "warnings": [],
        "t_start": 1.0,
        "t_end": 9.0,
        "hero_cards": ["Ah", "Kd"],
        "showdown": True,
        # Emitted by the reconstruction spine on every hand: at least one state
        # POSITIVELY placed the hero zone on seat 0. layout_supported requires it
        # because the hero_seat_mismatch check below is a majority vote over the
        # hero zone's own cards, so with an EMPTY hero zone it evaluates `0 > 0`
        # and reports "confirmed" on no evidence -- and an empty hero zone is
        # exactly what a layout drift produces.
        "hero_seat_confirmed": True,
    }
    hand.update(hand_overrides)
    return _completion_evidence_for_hand(
        hand,
        preceded_by_hand=True,
        followed_by_hand=True,
        validation_codes=[],
        table_size=hand.pop("_table_size", 6),  # type: ignore[arg-type]
        metadata={},
    )


def test_the_exporter_refuses_layout_support_without_a_resolved_table_size() -> None:
    """Hard-coding `layout_supported=True` used to pass the whole suite."""
    assert _export_evidence(_table_size=None).layout_supported is False


def test_the_exporter_refuses_layout_support_on_a_hero_seat_mismatch() -> None:
    assert (
        _export_evidence(warnings=["hero_seat_mismatch"]).layout_supported is False
    )


def test_the_exporter_refuses_layout_support_without_positive_hero_evidence() -> None:
    """The third half. Absence of a mismatch is not confirmation: the mismatch
    vote is taken over the hero zone's own cards, so an empty hero zone -- which
    is what a layout drift produces -- returned False and the export asserted
    layout_supported=True on zero evidence."""
    assert _export_evidence(hero_seat_confirmed=False).layout_supported is False


def test_the_exporter_confirms_layout_support_when_both_halves_hold() -> None:
    assert _export_evidence().layout_supported is True


# ---------------------------------------------------------------------------
# Finding A2 -- the pinned snapshot was invisible to the audit and never rotated
# ---------------------------------------------------------------------------


def test_the_pinned_pre_migration_snapshot_is_audited_and_restore_drilled(
    tmp_path: Path,
) -> None:
    """The only rollback point for the irreversible v13 migration was unverified.

    `PINNED_PREFIX` deliberately sits outside the rotation glob, which also put it
    outside `data_health.BACKUP_GLOB`: immediately after a migration the operator's
    health report said no backups existed about a directory holding exactly that
    snapshot, and the restore drill never opened it.
    """
    backups = tmp_path / "backups"
    backups.mkdir()
    database = tmp_path / "live.db"
    live = PokerDatabase(str(database))
    live.init_db()
    live.create_session(Session(name="Live", date_played=date(2026, 1, 1)))
    live.close()
    backup_database(database, backups, pinned=True)
    written = sorted(backups.glob(f"{PINNED_PREFIX}*.sqlite3"))
    assert written
    snapshot = written[0]

    report = audit_data_health(
        database_path=database,
        data_dir=tmp_path,
        backup_dir=backups,
        restore_backups=True,
    )
    check = next(item for item in report.checks if item.name == "backups")
    assert check.status == "pass"
    assert "No retained PokerTrainer backups" not in check.message

    # A corrupt rollback point must be reported before it is needed.
    snapshot.write_bytes(b"this is not a SQLite database")
    broken = audit_data_health(
        database_path=database,
        data_dir=tmp_path,
        backup_dir=backups,
        restore_backups=True,
    )
    broken_check = next(item for item in broken.checks if item.name == "backups")
    assert broken_check.status == "fail"


def test_pinned_snapshots_have_their_own_bounded_retention(tmp_path: Path) -> None:
    """Pinned snapshots were exempt from every retention policy and grew forever.

    A persistently failing migration snapshots the whole database on every start,
    so the exemption from the five-slot rotation must not also mean "unbounded".
    """
    backups = tmp_path / "backups"
    database = tmp_path / "live.db"
    sqlite3.connect(database).close()
    for _ in range(PINNED_KEEP_COUNT + 3):
        backup_database(database, backups, pinned=True)
    pinned = sorted(backups.glob(f"{PINNED_PREFIX}*.sqlite3"))
    assert len(pinned) == PINNED_KEEP_COUNT

    for _ in range(BACKUP_KEEP_COUNT + 2):
        backup_database(database, backups)
    # Rotation still cannot touch the pinned set.
    assert len(sorted(backups.glob(f"{PINNED_PREFIX}*.sqlite3"))) == PINNED_KEEP_COUNT
    assert len(sorted(backups.glob("poker_tracker_*.sqlite3"))) == BACKUP_KEEP_COUNT


# ---------------------------------------------------------------------------
# Finding A4 -- a failed pre-migration snapshot read as database corruption
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unwritable_backup_directory_names_the_backup_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`unable to open database file` reads as "your poker_tracker.db is broken".

    The live database is fine; only the backups mount is unwritable. Every other
    init_db refusal names the real cause and the fact that nothing was migrated.
    """
    database = tmp_path / "live.db"
    seed = sqlite3.connect(database)
    seed.executescript(
        """
        CREATE TABLE schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_metadata VALUES ('schema_version', '12');
        CREATE TABLE hands (id INTEGER PRIMARY KEY, source_type TEXT, review_status TEXT);
        INSERT INTO hands VALUES (1, 'manual', 'reviewed');
        """
    )
    seed.commit()
    seed.close()

    read_only = tmp_path / "locked"
    read_only.mkdir()
    monkeypatch.setattr(backup_module, "BACKUPS_DIR", read_only / "backups")
    (read_only / "backups").mkdir()
    os.chmod(read_only / "backups", stat.S_IRUSR | stat.S_IXUSR)
    try:
        db = PokerDatabase(str(database))
        with pytest.raises(RuntimeError) as caught:
            db.init_db()
    finally:
        os.chmod(read_only / "backups", stat.S_IRWXU)
    message = str(caught.value)
    assert "pre-migration backup" in message
    assert str(read_only / "backups") in message
    assert "was not applied" in message

    # Fails closed: the live database is untouched at its old version.
    check = sqlite3.connect(database)
    stored = check.execute(
        "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
    ).fetchone()
    columns = {row[1] for row in check.execute("PRAGMA table_info(hands)")}
    check.close()
    assert stored[0] == "12"
    assert "completion_status" not in columns


# ---------------------------------------------------------------------------
# Finding C10 -- documented constants nothing pinned
# ---------------------------------------------------------------------------


def test_documented_retention_and_evidence_constants_are_pinned() -> None:
    """README states "the newest five are retained"; nothing asserted it.

    EVIDENCE_SCHEMA_VERSION is the on-disk contract `is_known` gates on, so a
    silent bump would make every hand this build wrote unreadable to it.
    """
    assert BACKUP_KEEP_COUNT == 5
    assert EVIDENCE_SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# Finding C8 -- the evidence parser's BLOB branch
# ---------------------------------------------------------------------------


def test_the_evidence_parser_reads_a_blob_and_refuses_an_undecodable_one() -> None:
    """The branch exists for a hand-edited database and had no test at all."""
    payload = json.dumps(_clean_evidence_blob()).encode("utf-8")
    for blob in (payload, bytearray(payload), memoryview(payload)):
        assert parse_completion_evidence(blob).evidence_version == (
            EVIDENCE_SCHEMA_VERSION
        )
    undecodable = parse_completion_evidence(b"\xff\xfe{'evidence_version': 1}")
    assert undecodable.evidence_version == 0
    assert undecodable.is_known is False


def test_no_finite_check_regression_in_dump() -> None:
    """dump_completion_evidence must never emit a value json.dumps cannot encode."""
    evidence = parse_completion_evidence(
        _clean_evidence_blob(boundary_confidence=float("nan"))
    )
    text = json.dumps(dump_completion_evidence(evidence), allow_nan=False)
    assert math.isnan(float("nan"))  # sanity: the input really was NaN
    assert "NaN" not in text
