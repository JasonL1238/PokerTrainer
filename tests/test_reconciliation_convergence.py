"""What one save of a reconciliation settles, and what it leaves for the next one.

``persist_reconciliation`` rewrites the recorded summary figures that its own
cross-check compares against, so the state it leaves behind was one derivation
behind itself: the row it stored carried a verdict about the row it replaced. The
visible cost was a hand blocked with an empty issue list -- ``app.py`` renders
``accounting.issues``, which is re-derived from the REPAIRED row and therefore
empty -- so the product refused a hand and would not say why, and pressing the
same button again, changing nothing, cleared it.

Two halves are pinned here.

The first is the fixed point that IS reachable, and it is the ordinary one: a save
that repairs nothing changes nothing, down to ``updated_at``. That covers every
edit path once its repairing save has run, and it covers the operator's actual
clearing action -- the Accounting reconciliation panel nulls the recorded figures
before reconciling, so its save repairs nothing and converges immediately.

The second is the state that still needs another pass, which is pinned as SAID
rather than as absent. Rounds 4-6 require the save that replaces a recorded rake
disagreeing with the policy beside it to report ``needs_correction``, and round 10
requires the save that rewrites an unreadable settlement row to rewrite it; both
exist because a recorded figure disagreeing with the ledger is evidence and the
operator gets one refusal before the product replaces it. That contract and
one-save convergence cannot both hold -- the second save is the pinned one -- so
what this file pins is that the intermediate state names itself: the ledger
reconciles, the stored verdict does not say so, and every surface says exactly
that instead of claiming the ledger is broken.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.settlement_sync import (
    SettlementSyncRefused,
    sync_recorded_figures_from_ledger,
)
from poker_tracker.services.study_readiness import (
    accounting_verdict_predates_record,
    evaluate_study_readiness,
)


def _open_db(tmp_path: Path, name: str) -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _seed(
    db: PokerDatabase,
    *,
    bet: float = 10.0,
    award: float | None = 20.0,
    pot_size: float | None = None,
    hero_bb_won: float | None = None,
) -> tuple[Hand, HandPlayer, HandPlayer, list[Action]]:
    """A heads-up river bet and call, settled by one declared award."""

    session = db.create_session(Session(name="Convergence"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="No-limit Hold'em",
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type="manual",
        )
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
    actions = [
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=actor.player_key,
                player_name=actor.player_name,
                position=actor.position,
                street="river",
                action_index=index,
                action_type=kind,
                amount=bet,
                amount_semantics="incremental",
            )
        )
        for index, (actor, kind) in enumerate(((hero, "bet"), (villain, "call")), start=1)
    ]
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
                amount=award,
                entry_order=1,
            )
        ],
    )
    return hand, hero, villain, actions


def _state(db: PokerDatabase, hand_id: int) -> tuple:
    """Everything a save of the reconciliation can move, timestamps included."""

    settlement = db.fetch_hand_settlement(hand_id)
    hand = db.fetch_hand(hand_id)
    assert settlement is not None and hand is not None
    return (
        settlement.status,
        settlement.gross_pot,
        settlement.rake_amount,
        settlement.net_pot,
        settlement.is_balanced,
        tuple(settlement.warnings),
        settlement.dead_money,
        settlement.rake_rate,
        settlement.rake_cap,
        settlement.rake_rounding_unit,
        settlement.no_flop_no_drop,
        settlement.created_at,
        settlement.updated_at,
        hand.pot_size,
        hand.hero_bb_won,
        hand.review_status,
        tuple(
            (
                entry.entry_type,
                entry.pot_index,
                entry.player_key,
                entry.player_name,
                entry.amount,
                entry.entry_order,
            )
            for entry in db.fetch_settlement_entries(hand_id)
        ),
    )


def _panel_save(db: PokerDatabase, hand_id: int, **declaration: object):
    """The Accounting reconciliation panel's save, as ``app.py`` performs it.

    It nulls the three recorded summary figures before reconciling, which is what
    makes replacing them an act the operator took in a form rather than a side
    effect of any internal call.
    """
    settlement = db.fetch_hand_settlement(hand_id) or HandSettlement(hand_id=hand_id)
    db.upsert_hand_settlement(
        settlement.model_copy(
            update={
                "status": "settled",
                "gross_pot": None,
                "rake_amount": None,
                "net_pot": None,
                "is_balanced": False,
                "warnings": [],
                **declaration,
            }
        )
    )
    return persist_reconciliation(db, hand_id)


# ---------------------------------------------------------------------------
# The fixed point a save must reach, over every edit that reaches reconciliation
# ---------------------------------------------------------------------------


def _edit_the_action_line(db: PokerDatabase, hand, hero, villain, actions) -> None:
    for action in actions:
        db.update_action(
            action.model_copy(update={"amount": 20.0}),
            correction_notes="video shows 20, not 10",
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
                amount=40.0,
                entry_order=1,
            )
        ],
    )


def _edit_the_declared_award(db: PokerDatabase, hand, hero, villain, actions) -> None:
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key=villain.player_key,
                player_name=villain.player_name,
                amount=20.0,
                entry_order=1,
            )
        ],
    )


def _edit_the_rake_policy(db: PokerDatabase, hand, hero, villain, actions) -> None:
    stored = db.fetch_hand_settlement(hand.id)
    assert stored is not None
    db.upsert_hand_settlement(stored.model_copy(update={"rake_rate": 0.05}))
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key=hero.player_key,
                player_name=hero.player_name,
                amount=19.0,
                entry_order=1,
            )
        ],
    )


def _delete_a_recorded_action(db: PokerDatabase, hand, hero, villain, actions) -> None:
    db.delete_action(actions[-1].id, correction_notes="the call was never made")


_EDITS = {
    "action amount": _edit_the_action_line,
    "declared award": _edit_the_declared_award,
    "rake policy": _edit_the_rake_policy,
    "deleted action": _delete_a_recorded_action,
}


@pytest.mark.parametrize("edit", sorted(_EDITS))
def test_a_repeat_save_after_any_edit_path_changes_nothing(
    tmp_path: Path, edit: str
) -> None:
    """Save, save again, and the second save moves nothing -- rows, figures, timestamp.

    Parametrized over every edit that reaches reconciliation rather than the one
    an adversary happened to use, because a save that settles after one edit type
    and not another is the same defect wearing a different hat. Each of these
    moves a different input: the action line, the declared winner, the rake
    policy, and the existence of a recorded action.

    ``updated_at`` is in ``_state`` on purpose. A save that changes nothing used
    to restate the row's age anyway, so "did this save do something?" could not be
    answered from the record -- which is the same question the status was failing
    to answer.
    """
    db = _open_db(tmp_path, f"repeat_{abs(hash(edit))}.db")
    hand, hero, villain, actions = _seed(db)
    assert hand.id is not None
    _panel_save(db, hand.id)

    _EDITS[edit](db, hand, hero, villain, actions)
    _panel_save(db, hand.id)

    settled = _state(db, hand.id)
    persist_reconciliation(db, hand.id)
    assert _state(db, hand.id) == settled
    persist_reconciliation(db, hand.id)
    assert _state(db, hand.id) == settled
    db.close()


def test_the_operators_own_save_settles_a_corrected_hand_in_one_pass(
    tmp_path: Path,
) -> None:
    """The panel's save reaches the fixed point immediately, and repeats are no-ops.

    This is the path the operator actually has, and it is the reason the
    remaining two-pass case is a wording problem rather than a workflow one: the
    panel nulls the recorded summary figures, so its save repairs nothing and has
    nothing to be one step behind.
    """
    db = _open_db(tmp_path, "panel.db")
    hand, hero, villain, actions = _seed(db)
    assert hand.id is not None
    _panel_save(db, hand.id)
    _edit_the_action_line(db, hand, hero, villain, actions)

    saved = _panel_save(db, hand.id)
    assert saved.is_authoritative is True
    assert saved.settlement is not None
    assert saved.settlement.gross_pot == pytest.approx(40.0)
    assert saved.issues == ()

    settled = _state(db, hand.id)
    assert persist_reconciliation(db, hand.id).is_authoritative is True
    assert _state(db, hand.id) == settled
    db.close()


def test_a_save_that_derives_refund_rows_still_settles_in_one_pass(
    tmp_path: Path,
) -> None:
    """The refund derivation writes rows mid-save, and the verdict must survive it.

    ``persist_reconciliation`` files derived refund rows and re-derives before
    deciding, so this path was already convergent; it is pinned because it is the
    one place the function deliberately changes the record it is judging, and a
    future repair of the summary-figure ordering must not break it.
    """
    db = _open_db(tmp_path, "refund.db")
    session = db.create_session(Session(name="Refund"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="No-limit Hold'em")
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
            starting_stack=30,
        )
    )
    for index, (actor, kind, amount) in enumerate(
        ((hero, "bet", 50.0), (villain, "call", 30.0)), start=1
    ):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=actor.player_key,
                player_name=actor.player_name,
                position=actor.position,
                street="river",
                action_index=index,
                action_type=kind,
                amount=amount,
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
                amount=60.0,
                entry_order=1,
            )
        ],
    )

    first = persist_reconciliation(db, hand.id)
    assert first.is_authoritative is True
    refunds = [
        entry for entry in db.fetch_settlement_entries(hand.id)
        if entry.entry_type == "refund"
    ]
    assert [entry.amount for entry in refunds] == [pytest.approx(20.0)]

    settled = _state(db, hand.id)
    persist_reconciliation(db, hand.id)
    assert _state(db, hand.id) == settled
    db.close()


# ---------------------------------------------------------------------------
# The state that still needs another pass says so
# ---------------------------------------------------------------------------


def test_the_stored_warnings_describe_the_row_they_are_stored_on(
    tmp_path: Path,
) -> None:
    """A save may not file a complaint about a record it repaired in the same call.

    Correcting the action line leaves ``gross_pot`` and ``net_pot`` describing the
    old line. The save replaces both AND used to store "Recorded gross pot does
    not match the derived ledger" beside the corrected figures, so the row
    asserted a mismatch it did not contain, and an operator following it went
    looking for a disagreement that was not there.
    """
    db = _open_db(tmp_path, "warnings.db")
    hand, hero, villain, actions = _seed(db)
    assert hand.id is not None
    _panel_save(db, hand.id)
    _edit_the_action_line(db, hand, hero, villain, actions)

    saved = persist_reconciliation(db, hand.id)
    assert saved.settlement is not None
    assert saved.settlement.gross_pot == pytest.approx(40.0)
    assert saved.settlement.status == "needs_correction"
    assert not [
        note
        for note in saved.settlement.warnings
        if "does not match the derived ledger" in note
    ]
    db.close()


def test_a_hand_whose_verdict_predates_its_record_is_told_exactly_that(
    tmp_path: Path,
) -> None:
    """The blocker said the ledger does not reconcile. The ledger reconciles.

    This is the operator-visible face of the two-pass save: after the repairing
    pass the hand is blocked, ``accounting.issues`` is EMPTY because the figures
    the cross-check compares have just been corrected, and ``app.py`` renders
    that issue list. So the product refused the hand, gave a reason that was
    false, and offered no evidence at all -- indistinguishable from a hand with a
    real accounting defect, which wants the opposite action.
    """
    db = _open_db(tmp_path, "predates.db")
    hand, hero, villain, actions = _seed(db)
    assert hand.id is not None
    _panel_save(db, hand.id)
    _edit_the_action_line(db, hand, hero, villain, actions)

    saved = persist_reconciliation(db, hand.id)
    assert saved.is_authoritative is False
    assert saved.issues == ()
    assert saved.ledger.is_settled and saved.ledger.is_balanced and saved.ledger.is_legal
    assert accounting_verdict_predates_record(saved) is True

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=saved)
    blocker = next(
        item for item in readiness.blockers if item.code == "ACCOUNTING_NOT_AUTHORITATIVE"
    )
    assert "chip ledger reconciles" in blocker.reason
    assert "does not reconcile" not in blocker.reason
    assert blocker.detail, "a refused hand must carry evidence for the refusal"
    assert "needs_correction" in blocker.detail[0]
    assert "Save and reconcile" in blocker.clearing_action

    # And the pass it asks for is the pass that clears it.
    assert persist_reconciliation(db, hand.id).is_authoritative is True
    db.close()


def test_a_real_accounting_defect_is_still_reported_as_one(tmp_path: Path) -> None:
    """The honest wording must not become a blanket excuse for a broken ledger.

    A hand whose recorded pot contradicts its own action line has a genuine
    defect, states it, and must keep the original reason and the original detail
    -- otherwise the new branch would swallow every accounting blocker in the
    product.
    """
    db = _open_db(tmp_path, "realdefect.db")
    hand, hero, villain, actions = _seed(db, pot_size=999.0)
    assert hand.id is not None

    saved = persist_reconciliation(db, hand.id)
    assert saved.is_authoritative is False
    assert accounting_verdict_predates_record(saved) is False

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    blocker = next(
        item
        for item in evaluate_study_readiness(stored, accounting=saved).blockers
        if item.code == "ACCOUNTING_NOT_AUTHORITATIVE"
    )
    assert "does not reconcile" in blocker.reason
    assert "Observed final pot does not match the derived gross pot." in blocker.detail
    db.close()


def test_a_never_reconciled_hand_is_not_reported_as_a_broken_ledger(
    tmp_path: Path,
) -> None:
    """The same state arises with no edit at all: nobody has saved the settlement yet.

    ``import_session`` never calls ``persist_reconciliation``, so a freshly
    imported hand sits here permanently. It was told its ledger does not
    reconcile, which is false and names a defect that does not exist.
    """
    db = _open_db(tmp_path, "never.db")
    hand, hero, villain, actions = _seed(db)
    assert hand.id is not None

    fresh = reconcile_persisted_hand(db, hand.id)
    assert fresh.is_authoritative is False
    assert accounting_verdict_predates_record(fresh) is True

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    blocker = next(
        item
        for item in evaluate_study_readiness(stored, accounting=fresh).blockers
        if item.code == "ACCOUNTING_NOT_AUTHORITATIVE"
    )
    assert "chip ledger reconciles" in blocker.reason
    assert "'settled'" in blocker.detail[0]
    db.close()


def test_the_sync_refusal_names_the_stale_verdict_instead_of_the_ledger(
    tmp_path: Path,
) -> None:
    """"Reconcile a legal, balanced ledger first" is the wrong instruction here.

    ``sync_recorded_figures_from_ledger`` reconciles before it decides, so on a
    hand whose recorded summary figures are stale its own first pass is the
    repairing one and its own verdict is the pre-repair one -- it then refused
    with a sentence about a ledger that is already legal and already balanced.
    The operator's next action is the same save once more, and that is now what
    it says.
    """
    db = _open_db(tmp_path, "sync.db")
    hand, hero, villain, actions = _seed(db)
    assert hand.id is not None
    _panel_save(db, hand.id)
    _edit_the_action_line(db, hand, hero, villain, actions)

    with pytest.raises(SettlementSyncRefused) as refusal:
        sync_recorded_figures_from_ledger(db, hand.id)
    message = str(refusal.value)
    assert "balances and is legal" in message
    assert "Reconcile a legal, balanced ledger first." not in message
    assert "once more" in message

    # And the pass it names is the pass that lets the replacement through.
    synced = sync_recorded_figures_from_ledger(db, hand.id)
    assert synced.is_authoritative is True
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.pot_size == pytest.approx(40.0)
    db.close()


def test_a_degraded_settlement_row_is_not_called_a_stale_verdict(
    tmp_path: Path,
) -> None:
    """An unreadable row has a real, named defect and must keep saying so.

    ``_degraded_hand_settlement`` forces ``status`` off ``reconciled`` and files
    the unreadable columns as an issue, so the shape resembles a stale verdict.
    It is not one, and the issue naming the column to fix is what the operator
    needs; the wording branch must not replace it.
    """
    db = _open_db(tmp_path, "degraded.db")
    hand, hero, villain, actions = _seed(db)
    assert hand.id is not None
    persist_reconciliation(db, hand.id)

    connection = sqlite3.connect(str(tmp_path / "degraded.db"))
    connection.execute(
        "UPDATE hand_settlements SET rake_rate = ? WHERE hand_id = ?", (-0.5, hand.id)
    )
    connection.commit()
    connection.close()

    degraded = reconcile_persisted_hand(db, hand.id)
    assert degraded.is_authoritative is False
    assert any("could not be read" in issue for issue in degraded.issues)
    assert accounting_verdict_predates_record(degraded) is False
    db.close()
