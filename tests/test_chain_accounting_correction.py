"""The correction chain, joined end to end:

    YOLO reconstruction timeline
        -> export_timeline                        (cv_lab exporter)
        -> import_session                         (poker_tracker)
        -> PokerDatabase (sqlite file on tmp_path)
        -> persist_reconciliation                 (accounting)
        -> retained coaching / hand review / solver run
        -> update_hand_facts                      (correction)
        -> every derivative of that hand goes stale
        -> rerun                                  (accounting + coaching + solver)

The segments each had tests; the joins did not. Accounting in particular was
never joined to correction, and that is the seam a wrong answer survives: a
correction changes what the ledger is about, so everything derived from the old
ledger has to stop being presented as current, and the rerun has to land back on
something the corrected records actually support.

Two halves, and both are load-bearing:

* every derivative goes stale -- reviews, coaching, settlement, solver runs, and
  the staleness set the list surfaces badge from. A derivative that quietly
  survives a correction is a wrong answer labelled current;
* nothing else does. Targeted invalidation is a stated Phase 10 requirement, and
  an invalidation wider than the change destroys work the operator has to redo.
  The sibling hand, the neighbouring session, and -- proven here for the first
  time -- an accounting rerun that writes an identical settlement row are all on
  that side of the line.

Every assertion is on DATA read back through the product's own readers. An
integration test that only asserts no exception was raised passes when a seam
silently drops everything through it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cv_lab.scripts.pipeline.export_yolo_card_hands_for_app import export_timeline
from poker_tracker.coaching.coaching_prompts import build_hand_review_prompt
from poker_tracker.persistence.completion import (
    acknowledge_codes,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import import_session
from poker_tracker.persistence.models import (
    CoachingResponse,
    Hand,
    HandReview,
    HandSettlement,
    SettlementEntry,
    SolverRun,
)
from poker_tracker.services.hand_accounting import (
    AccountingReconciliation,
    attest_assumption,
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import (
    BlockerCode,
    StudyReadiness,
    evaluate_study_readiness,
)

# What the timeline below records, restated independently so a seam that
# recomputes a figure instead of carrying it has something to disagree with.
GROUND_TRUTH_HERO = "As Th"
GROUND_TRUTH_BOARD = "Qd 7s 2c 9h Kc"
GROUND_TRUTH_GROSS_POT = 38.0
GROUND_TRUTH_HERO_RESULT = 19.0
HERO_KEY = "seat:0"
VILLAIN_KEY = "seat:5"

CORRECTED_HERO = "Ad Th"


def _action_line() -> list[dict]:
    """One legal heads-up hand: blinds, a preflop raise, and three bet streets.

    Legality is not decoration here. The whole chain rests on a ledger the
    cross-check accepts, so the blinds are posted as forced bets and every wager
    is the increment the accounting adapter expects.
    """
    hero, villain = 0, 5
    return [
        {"street": "preflop", "action_index": 1, "seat": villain, "position": "BTN",
         "player_name": "Villain", "action_type": "post_blind", "amount": 0.5,
         "forced_bet_type": "small_blind"},
        {"street": "preflop", "action_index": 2, "seat": hero, "position": "BB",
         "player_name": "Hero", "action_type": "post_blind", "amount": 1.0,
         "forced_bet_type": "big_blind"},
        {"street": "preflop", "action_index": 3, "seat": villain, "position": "BTN",
         "player_name": "Villain", "action_type": "raise", "amount": 2.5},
        {"street": "preflop", "action_index": 4, "seat": hero, "position": "BB",
         "player_name": "Hero", "action_type": "call", "amount": 2.0},
        {"street": "flop", "action_index": 1, "seat": hero, "position": "BB",
         "player_name": "Hero", "action_type": "check", "amount": None},
        {"street": "flop", "action_index": 2, "seat": villain, "position": "BTN",
         "player_name": "Villain", "action_type": "bet", "amount": 4.0},
        {"street": "flop", "action_index": 3, "seat": hero, "position": "BB",
         "player_name": "Hero", "action_type": "call", "amount": 4.0},
        {"street": "turn", "action_index": 1, "seat": hero, "position": "BB",
         "player_name": "Hero", "action_type": "check", "amount": None},
        {"street": "turn", "action_index": 2, "seat": villain, "position": "BTN",
         "player_name": "Villain", "action_type": "check", "amount": None},
        {"street": "river", "action_index": 1, "seat": hero, "position": "BB",
         "player_name": "Hero", "action_type": "bet", "amount": 12.0},
        {"street": "river", "action_index": 2, "seat": villain, "position": "BTN",
         "player_name": "Villain", "action_type": "call", "amount": 12.0},
    ]


def _timeline_hand(number: int, t_start: float) -> dict:
    return {
        "hand_number": number,
        "t_start": t_start,
        "t_end": t_start + 30.0,
        "hero": ["AS", "10H"],
        "board": ["QD", "7S", "2C", "9H", "KC"],
        "complete_cards": True,
        "warnings": [],
        # Both are what lets the exported evidence derive a COMPLETE hand with a
        # supported layout, which is what makes a study-ready hand reachable and
        # therefore what makes "the correction took readiness away" observable.
        "hero_seat_confirmed": True,
        "terminal_event": "showdown",
        "players": [
            {"seat": 0, "position": "BB", "player_name": "Hero",
             "starting_stack": 100, "is_hero": True},
            {"seat": 5, "position": "BTN", "player_name": "Villain",
             "starting_stack": 100, "is_hero": False},
        ],
        "actions": _action_line(),
        "pot": GROUND_TRUTH_GROSS_POT,
        "winner_seat": 0,
        "result": "Hero wins",
        "hero_bb_won": GROUND_TRUTH_HERO_RESULT,
        "reconciled": True,
        "source_images": [f"images/val/frame_{number:06d}.jpg"],
    }


def _timeline() -> dict:
    """Three hands, because only a hand with neighbours proves its own boundaries.

    The exporter derives partial_start from "no hand precedes this one", so the
    middle hand is the only one whose completion evidence can read `complete`.
    Hand 1 doubles as the same-session control for targeted invalidation.
    """
    return {"hands": [_timeline_hand(n, 40.0 * (n - 1)) for n in (1, 2, 3)]}


def _import(db: PokerDatabase, tmp_path: Path, name: str) -> list[Hand]:
    timeline_path = tmp_path / f"timeline_{name}.json"
    timeline_path.write_text(json.dumps(_timeline()), encoding="utf-8")
    payload = export_timeline(
        timeline_path, tmp_path / f"draft_{name}.json", session_name=name
    )
    assert payload["cv_import_summary"]["exported_hands"] == 3
    session = import_session(db, payload)
    return db.fetch_hands_by_session(session.id)


def _declare_award_and_reconcile(
    db: PokerDatabase, hand_id: int
) -> AccountingReconciliation:
    """Do what the operator does in the Accounting reconciliation panel.

    The CV exporter emits no settlement rows at all, so the winner of the pot is
    a declared input; the attestation loop is the 'Confirm this assumption'
    control, which the hand cannot become study-ready without.
    """
    derived = reconcile_persisted_hand(db, hand_id)
    db.upsert_hand_settlement(HandSettlement(hand_id=hand_id, status="settled"))
    db.replace_settlement_entries(
        hand_id,
        [
            SettlementEntry(
                hand_id=hand_id,
                entry_type="award",
                pot_index=0,
                player_key=HERO_KEY,
                player_name="Hero",
                amount=derived.ledger.gross_pot,
                entry_order=1,
            )
        ],
    )
    reconciled = persist_reconciliation(db, hand_id)
    for dependence in reconciled.assumption_dependence:
        assert attest_assumption(db, hand_id, dependence.code)
    return reconcile_persisted_hand(db, hand_id)


def _write_coaching(
    db: PokerDatabase, hand_id: int | None, session_id: int, *, body: str = "review"
) -> CoachingResponse:
    return db.create_coaching_response(
        CoachingResponse(
            provider_name="fixture",
            model_name="deterministic",
            raw_prompt="post-session completed hands; do not provide real-time advice",
            raw_response=body,
            review_type="hand" if hand_id is not None else "session",
            hand_id=hand_id,
            session_id=session_id,
        )
    )


def _write_solver_run(db: PokerDatabase, hand_id: int, input_hash: str) -> SolverRun:
    return db.create_solver_run(
        SolverRun(
            hand_id=hand_id,
            status="completed",
            input_hash=input_hash,
            spot={"street": "flop", "board": GROUND_TRUTH_BOARD},
            evidence={"hero_action": "check", "solver_frequency": 0.62},
            exploitability_pct=0.4,
        )
    )


def _readiness(db: PokerDatabase, hand_id: int) -> StudyReadiness:
    """Readiness derived from the DATABASE, not from hand-authored model lists."""
    hand = db.fetch_hand(hand_id)
    assert hand is not None
    return evaluate_study_readiness(
        hand,
        accounting=reconcile_persisted_hand(db, hand_id),
        hand_issues=db.fetch_hand_issues(hand_id=hand_id),
        coaching_reviews=db.fetch_coaching_reviews_by_hand(hand_id),
        hand_reviews=db.fetch_reviews_by_hand(hand_id),
        solver_runs=db.fetch_solver_runs_by_hand(hand_id),
        user_confirmed=True,
    )


def _derivative_state(db: PokerDatabase, hand_id: int) -> dict[str, object]:
    """Every stored derivative of one hand, in one comparable snapshot.

    A sampled derivative is what lets one of them survive a correction unnoticed,
    so the whole set is read at once and compared as a whole.
    """
    hand = db.fetch_hand(hand_id)
    assert hand is not None
    settlement = db.fetch_hand_settlement(hand_id)
    assert settlement is not None
    return {
        "review_status": hand.review_status,
        "completion_status": hand.completion_status,
        "settlement_status": settlement.status,
        "settlement_is_balanced": settlement.is_balanced,
        "settlement_gross_pot": settlement.gross_pot,
        "accounting_authoritative": reconcile_persisted_hand(
            db, hand_id
        ).is_authoritative,
        "coaching_stale": [
            review.is_stale for review in db.fetch_coaching_reviews_by_hand(hand_id)
        ],
        "hand_reviews_stale": [
            review.is_stale for review in db.fetch_reviews_by_hand(hand_id)
        ],
        "solver_statuses": [
            run.status for run in db.fetch_solver_runs_by_hand(hand_id)
        ],
        "badged_stale": hand_id in db.fetch_stale_review_hand_ids(),
        "blockers": tuple(blocker.code for blocker in _readiness(db, hand_id).blockers),
    }


@pytest.fixture
def chain(tmp_path):
    """A reconciled, study-ready reconstructed hand with every derivative saved.

    Plus the two controls the targeted-invalidation half needs: a sibling in the
    same session and a hand in a second, unrelated session, each carrying the
    same derivatives.
    """
    db = PokerDatabase(str(tmp_path / "poker.db"))
    db.init_db()

    session_a = _import(db, tmp_path, "Session A")
    session_b = _import(db, tmp_path, "Session B")
    subject, sibling, neighbour = session_a[1], session_a[0], session_b[1]

    for hand, tag in ((subject, "subject"), (sibling, "sibling"), (neighbour, "neighbour")):
        assert _declare_award_and_reconcile(db, hand.id).is_authoritative
        _write_coaching(db, hand.id, hand.session_id, body=f"{tag} coaching")
        db.create_hand_review(
            HandReview(
                hand_id=hand.id,
                hand_summary=f"{tag} summary",
                theory_coach="theory",
                exploit_coach="exploit",
                study_lesson="lesson",
            )
        )
        _write_solver_run(db, hand.id, f"hash-{tag}")
    _write_coaching(db, None, subject.session_id)
    _write_coaching(db, None, neighbour.session_id)

    try:
        yield db, subject, sibling, neighbour
    finally:
        db.close()


def test_the_timeline_reaches_sqlite_and_an_authoritative_ledger_intact(chain) -> None:
    """Seams 1-3: the exporter, the importer, and the accounting cross-check.

    Asserted as data rather than as "it ran": the cards the detector emitted in
    label form, the seats, the per-seat contributions the ledger derived from the
    imported action rows, and the settlement columns the reconciliation wrote.
    """
    db, subject, _, _ = chain

    hand = db.fetch_hand(subject.id)
    assert hand.hero_cards == GROUND_TRUTH_HERO
    assert hand.board_cards == GROUND_TRUTH_BOARD
    assert hand.source_type == "cv_import"
    assert hand.completion_status == "complete"
    assert [player.player_key for player in db.fetch_players_by_hand(hand.id)] == [
        HERO_KEY,
        VILLAIN_KEY,
    ]
    assert len(db.fetch_actions_by_hand(hand.id)) == len(_action_line())

    accounting = reconcile_persisted_hand(db, hand.id)
    assert accounting.issues == ()
    assert accounting.is_authoritative is True
    assert accounting.ledger.gross_pot == pytest.approx(GROUND_TRUTH_GROSS_POT)
    assert accounting.ledger.contributions[HERO_KEY] == pytest.approx(19.0)
    assert accounting.ledger.contributions[VILLAIN_KEY] == pytest.approx(19.0)
    assert accounting.ledger.net_results[HERO_KEY] == pytest.approx(
        GROUND_TRUTH_HERO_RESULT
    )
    assert accounting.settlement.status == "reconciled"
    assert accounting.settlement.gross_pot == pytest.approx(GROUND_TRUTH_GROSS_POT)
    assert _readiness(db, hand.id).blockers == ()


def test_a_source_correction_stales_every_derivative_of_the_corrected_hand(
    chain,
) -> None:
    """Seam 4: the correction, and the whole derivative set it invalidates.

    The set is asserted as a set. Sampling one derivative is exactly how another
    one survives a correction and keeps being presented as current.
    """
    db, subject, _, _ = chain
    session_id = subject.session_id
    assert _derivative_state(db, subject.id)["blockers"] == ()

    corrected = db.update_hand_facts(
        db.fetch_hand(subject.id).model_copy(update={"hero_cards": CORRECTED_HERO}),
        correction_notes="Reviewed the showdown frames; hero's ace is a diamond.",
    )

    assert corrected.hero_cards == CORRECTED_HERO
    assert corrected.source_type == "corrected_cv"
    audit = db.fetch_hand_corrections(subject.id)[0]
    assert audit.correction_type == "hand_facts"
    assert audit.before_state["hero_cards"] == GROUND_TRUTH_HERO
    assert audit.after_state["hero_cards"] == CORRECTED_HERO

    state = _derivative_state(db, subject.id)
    assert state["review_status"] == "needs_correction"
    assert state["completion_status"] == "uncertain"
    assert state["settlement_status"] == "needs_correction"
    assert state["settlement_is_balanced"] is False
    assert state["accounting_authoritative"] is False
    assert state["coaching_stale"] == [True]
    assert state["hand_reviews_stale"] == [True]
    assert state["solver_statuses"] == ["stale"]
    assert state["badged_stale"] is True

    evidence = parse_completion_evidence(
        db.fetch_hand(subject.id).completion_evidence
    )
    assert "source_facts_corrected" in evidence.warning_codes

    # The session-level coaching describes a set of hands one of which just
    # changed, so it is stale too -- and it is the derivative most easily missed,
    # because it hangs off the session rather than off the hand.
    session_reviews = [
        review
        for review in db.fetch_coaching_reviews_by_session(session_id)
        if review.review_type == "session"
    ]
    assert [review.is_stale for review in session_reviews] == [True]

    assert set(state["blockers"]) >= {
        "ACCOUNTING_NOT_AUTHORITATIVE",
        "UNRESOLVED_SOURCE_WARNING",
        "STALE_COACHING_EVIDENCE",
        "STALE_SOLVER_EVIDENCE",
    }


def test_a_correction_leaves_the_hands_and_sessions_it_does_not_bear_on_alone(
    chain,
) -> None:
    """The inverse half. An invalidation wider than the change destroys work.

    Targeted invalidation is a stated Phase 10 requirement, and the two ways to
    over-reach are by session and by neighbour: a correction that swept the whole
    session, or the whole database, would take a study-ready hand's coaching,
    solver result and reconciliation with it and force the operator to redo work
    the correction says nothing about.
    """
    db, subject, sibling, neighbour = chain
    before = {
        "sibling": _derivative_state(db, sibling.id),
        "neighbour": _derivative_state(db, neighbour.id),
    }
    neighbour_session_before = [
        (review.review_type, review.is_stale)
        for review in db.fetch_coaching_reviews_by_session(neighbour.session_id)
    ]
    # Not zero: confirming a declared assumption files its own audit row, so the
    # question is whether the correction added one, not whether any exist.
    corrections_before = {
        hand.id: len(db.fetch_hand_corrections(hand.id))
        for hand in (sibling, neighbour)
    }

    db.update_hand_facts(
        db.fetch_hand(subject.id).model_copy(update={"hero_cards": CORRECTED_HERO}),
        correction_notes="Reviewed the showdown frames; hero's ace is a diamond.",
    )

    assert _derivative_state(db, sibling.id) == before["sibling"]
    assert _derivative_state(db, neighbour.id) == before["neighbour"]
    assert db.fetch_stale_review_hand_ids() == {subject.id}
    # The other session's session-level review is about hands none of which moved.
    assert [
        (review.review_type, review.is_stale)
        for review in db.fetch_coaching_reviews_by_session(neighbour.session_id)
    ] == neighbour_session_before
    assert {
        hand.id: len(db.fetch_hand_corrections(hand.id))
        for hand in (sibling, neighbour)
    } == corrections_before


def test_rerunning_the_analysis_after_a_correction_restores_a_study_ready_hand(
    chain,
) -> None:
    """Seam 5: the rerun, driven through the database rather than model lists.

    Clearing the two stale blockers has only ever been asserted against in-memory
    lists handed straight to ``evaluate_study_readiness``. Here the fresh rows are
    written, read back, and the blockers are re-derived from what the store holds
    -- and the fresh coaching is checked to be about the CORRECTED hand, because a
    rerun that re-derives its prompt from the old facts clears the blocker while
    answering the wrong question.
    """
    db, subject, _, _ = chain
    db.update_hand_facts(
        db.fetch_hand(subject.id).model_copy(update={"hero_cards": CORRECTED_HERO}),
        correction_notes="Reviewed the showdown frames; hero's ace is a diamond.",
    )

    # Accounting first: re-derive the ledger from the corrected records.
    reconciled = persist_reconciliation(db, subject.id)
    assert reconciled.is_authoritative is True
    assert reconciled.issues == ()
    # The corrected fact does not move a chip, so the rerun must land on the same
    # ledger. A rerun that produced different figures from an unchanged action
    # line would mean the first reconciliation had been reading something else.
    assert reconciled.ledger.gross_pot == pytest.approx(GROUND_TRUTH_GROSS_POT)
    assert reconciled.ledger.net_results[HERO_KEY] == pytest.approx(
        GROUND_TRUTH_HERO_RESULT
    )

    hand = db.fetch_hand(subject.id)
    prompt = build_hand_review_prompt(
        db.fetch_session(hand.session_id),
        hand,
        db.fetch_actions_by_hand(hand.id),
        db.fetch_players_by_hand(hand.id),
        ledger=reconciled.ledger,
        accounting_issues=list(reconciled.issues),
        accounting_authoritative=reconciled.is_authoritative,
    )
    assert CORRECTED_HERO in prompt
    assert GROUND_TRUTH_HERO not in prompt
    _write_coaching(db, hand.id, hand.session_id, body="rerun coaching")
    _write_solver_run(db, hand.id, "hash-subject-rerun")

    acknowledged = acknowledge_codes(
        parse_completion_evidence(db.fetch_hand(subject.id).completion_evidence),
        ["source_facts_corrected"],
    )
    db.update_hand_completion(
        subject.id,
        completion_evidence=dump_completion_evidence(acknowledged),
        notes="Correction reviewed against the recording.",
    )

    state = _derivative_state(db, subject.id)
    assert state["completion_status"] == "complete"
    assert state["settlement_status"] == "reconciled"
    assert state["accounting_authoritative"] is True
    assert sorted(state["solver_statuses"]) == ["completed", "stale"]
    assert state["blockers"] == ()
    # The invalidated rows stay as retained history; what changed is that current
    # evidence now exists beside them.
    assert sorted(state["coaching_stale"]) == [False, True]
    assert sorted(state["hand_reviews_stale"]) == [True]


def test_rerunning_accounting_that_changes_nothing_keeps_the_analysis_it_did_not_change(
    chain,
) -> None:
    """An idempotent reconciliation is not an evidence change.

    ``persist_reconciliation`` recomputes and re-saves the settlement row every
    time it is called, and the settlement writer invalidated the hand's retained
    analysis unconditionally -- so re-reconciling a hand nothing had happened to
    staled its coaching, staled its saved hand review, flipped its completed
    solver run to ``stale``, demoted it out of ``reviewed``, and handed a
    study-ready hand two blockers.

    In the correction chain that is an ordering trap with no signal: an operator
    who reruns coaching and then reconciles the accounting throws away the
    coaching they just paid for, while the same two actions in the other order
    succeed, and nothing in the blocker text says so. The same press also
    cancels a solve that is still running.
    """
    db, subject, _, _ = chain
    before = _derivative_state(db, subject.id)
    assert before["blockers"] == ()

    again = persist_reconciliation(db, subject.id)

    assert again.is_authoritative is True
    assert again.settlement.gross_pot == pytest.approx(before["settlement_gross_pot"])
    assert _derivative_state(db, subject.id) == before


def test_a_rerun_of_coaching_survives_the_accounting_rerun_the_same_correction_needs(
    chain,
) -> None:
    """The operator-visible shape of the same defect, through the whole chain.

    Correct the hand, rerun both derivatives, then reconcile the accounting the
    correction also invalidated. All three reruns are required to clear the
    hand's blockers, so no ordering of them may destroy another's result.
    """
    db, subject, _, _ = chain
    db.update_hand_facts(
        db.fetch_hand(subject.id).model_copy(update={"hero_cards": CORRECTED_HERO}),
        correction_notes="Reviewed the showdown frames; hero's ace is a diamond.",
    )
    _write_coaching(db, subject.id, subject.session_id, body="rerun coaching")
    fresh_run = _write_solver_run(db, subject.id, "hash-subject-rerun")

    persist_reconciliation(db, subject.id)

    fresh_coaching = [
        review
        for review in db.fetch_coaching_reviews_by_hand(subject.id)
        if review.raw_response == "rerun coaching"
    ]
    assert [review.is_stale for review in fresh_coaching] == [False]
    assert db.fetch_solver_run(fresh_run.id).status == "completed"
    assert "STALE_COACHING_EVIDENCE" not in _readiness(db, subject.id).codes()
    assert "STALE_SOLVER_EVIDENCE" not in _readiness(db, subject.id).codes()


def test_a_re_declared_pot_winner_still_stales_the_analysis_built_on_the_old_one(
    chain,
) -> None:
    """The guard on the fix above: a settlement write that DOES change something.

    Who won the pot is the single input the derived payouts and the reported hero
    result come from, so coaching and solver output built on the old winner are
    stale the moment it moves. Narrowing the invalidation to real changes must not
    turn into not invalidating at all.
    """
    db, subject, _, _ = chain
    assert _derivative_state(db, subject.id)["blockers"] == ()

    db.replace_settlement_entries(
        subject.id,
        [
            SettlementEntry(
                hand_id=subject.id,
                entry_type="award",
                pot_index=0,
                player_key=VILLAIN_KEY,
                player_name="Villain",
                amount=GROUND_TRUTH_GROSS_POT,
                entry_order=1,
            )
        ],
    )

    state = _derivative_state(db, subject.id)
    assert state["coaching_stale"] == [True]
    assert state["hand_reviews_stale"] == [True]
    assert state["solver_statuses"] == ["stale"]
    assert state["review_status"] == "needs_correction"
    assert state["badged_stale"] is True
    assert set(state["blockers"]) >= {
        "STALE_COACHING_EVIDENCE",
        "STALE_SOLVER_EVIDENCE",
    }
    # The re-declaration is audited like any other source-fact change. Newest
    # first, because the fixture's assumption attestation filed a row of its own.
    assert (
        db.fetch_hand_corrections(subject.id)[0].correction_type
        == "settlement_award_update"
    )


def test_a_declared_rake_change_still_stales_the_analysis_built_on_the_old_ledger(
    chain,
) -> None:
    """The same guard for the other settlement writer.

    ``upsert_hand_settlement`` carries the declared rake policy and dead money.
    Changing either moves the net pot and the hero result, so the retained
    analysis has to go stale through this writer too -- the narrowing must be
    "nothing changed", never "this writer no longer invalidates".
    """
    db, subject, _, _ = chain
    settlement = db.fetch_hand_settlement(subject.id)
    assert settlement.rake_rate == pytest.approx(0.0)

    db.upsert_hand_settlement(settlement.model_copy(update={"rake_rate": 0.05}))

    state = _derivative_state(db, subject.id)
    assert state["coaching_stale"] == [True]
    assert state["hand_reviews_stale"] == [True]
    assert state["solver_statuses"] == ["stale"]
    assert state["badged_stale"] is True


def test_a_correction_the_ledger_contradicts_is_not_laundered_by_rerunning_it(
    chain,
) -> None:
    """A rerun re-derives the verdict; it does not restore the old one.

    Correcting the recorded hero result to a figure the action line does not
    support has to leave the hand blocked no matter how many times the accounting
    is re-reconciled. This is the release-blocking direction of the chain: a
    corrected fact that disagrees with the ledger must be visibly refused, not
    absorbed into a settlement row that reads `reconciled`.
    """
    db, subject, _, _ = chain
    db.update_hand_facts(
        db.fetch_hand(subject.id).model_copy(update={"hero_bb_won": 5.0}),
        correction_notes="Transcribed the hero result from the session notes.",
    )

    reconciled = persist_reconciliation(db, subject.id)
    reconciled = persist_reconciliation(db, subject.id)

    assert reconciled.is_authoritative is False
    assert reconciled.settlement.status == "needs_correction"
    assert any("hero result" in issue.lower() for issue in reconciled.issues)
    assert reconciled.ledger.net_results[HERO_KEY] == pytest.approx(
        GROUND_TRUTH_HERO_RESULT
    )
    assert "ACCOUNTING_NOT_AUTHORITATIVE" in _readiness(db, subject.id).codes()


def test_every_blocker_code_this_chain_names_is_a_real_blocker_code() -> None:
    """A misspelled code in an ``in``/``>=`` assertion is an assertion that passes.

    Every code this module asserts on is checked against the readiness vocabulary
    once, here, so a renamed blocker fails loudly instead of quietly disarming the
    checks above.
    """
    named = {
        "ACCOUNTING_NOT_AUTHORITATIVE",
        "UNRESOLVED_SOURCE_WARNING",
        "STALE_COACHING_EVIDENCE",
        "STALE_SOLVER_EVIDENCE",
    }
    assert named <= set(BlockerCode.__args__)
