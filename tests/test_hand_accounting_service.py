from __future__ import annotations

import pytest

from poker_tracker.math.analytics import compute_session_stats
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


def _make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def _create_heads_up_value_hand(
    db: PokerDatabase,
    session_id: int,
    *,
    hand_number: int = 1,
    observed_pot: float | None = 20,
    observed_hero_result: float | None = 10,
) -> tuple[Hand, HandPlayer, HandPlayer, list[Action]]:
    hand = db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=hand_number,
            game_type="No-limit Hold'em",
            pot_size=observed_pot,
            hero_bb_won=observed_hero_result,
        )
    )
    hero = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key=f"hero-{hand_number}",
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
            player_key=f"villain-{hand_number}",
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
                player_key=hero.player_key,
                player_name=hero.player_name,
                position=hero.position,
                street="river",
                action_type="bet",
                amount=10,
                amount_semantics="incremental",
            )
        ),
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=villain.player_key,
                player_name=villain.player_name,
                position=villain.position,
                street="river",
                action_type="call",
                amount=10,
                amount_semantics="incremental",
            )
        ),
    ]
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            status="settled",
        )
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
    return hand, hero, villain, actions


def test_valid_completed_hand_becomes_authoritative_after_reconciliation() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Authoritative"))
    hand, hero, _, _ = _create_heads_up_value_hand(db, session.id)

    before = reconcile_persisted_hand(db, hand.id)
    assert before.is_authoritative is False
    assert before.settlement.status == "settled"
    assert before.issues == ()

    result = persist_reconciliation(db, hand.id)

    assert result.is_authoritative is True
    assert result.issues == ()
    assert result.settlement.status == "reconciled"
    assert result.settlement.gross_pot == pytest.approx(20)
    assert result.settlement.rake_amount == pytest.approx(0)
    assert result.settlement.net_pot == pytest.approx(20)
    assert result.settlement.is_balanced is True
    assert result.ledger.is_settled is True
    assert result.ledger.is_balanced is True
    assert result.ledger.is_legal is True
    assert result.ledger.net_results[hero.player_key] == pytest.approx(10)
    assert sum(result.ledger.net_results.values()) + result.ledger.rake == pytest.approx(0)
    db.close()


def test_observed_pot_and_result_mismatches_block_authority() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Mismatch"))
    hand, _, _, _ = _create_heads_up_value_hand(
        db,
        session.id,
        observed_pot=21,
        observed_hero_result=11,
    )

    result = persist_reconciliation(db, hand.id)

    assert result.is_authoritative is False
    assert result.ledger.is_balanced is True
    assert result.ledger.is_legal is True
    assert result.settlement.status == "needs_correction"
    assert result.settlement.is_balanced is True
    assert any("observed final pot" in issue.lower() for issue in result.issues)
    assert any("observed hero result" in issue.lower() for issue in result.issues)
    # Derived summaries remain truthful even while the observed evidence
    # prevents the hand from being authoritative.
    assert result.settlement.gross_pot == pytest.approx(20)
    assert result.settlement.net_pot == pytest.approx(20)
    db.close()


def test_player_and_action_edits_invalidate_a_reconciled_settlement() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Invalidation"))
    hand, hero, _, actions = _create_heads_up_value_hand(db, session.id)
    assert persist_reconciliation(db, hand.id).is_authoritative is True

    db.update_hand_player(hero.model_copy(update={"notes": "reviewed identity"}))
    after_player_edit = db.fetch_hand_settlement(hand.id)
    assert after_player_edit.status == "needs_correction"
    assert after_player_edit.is_balanced is False
    assert any("changed" in warning.lower() for warning in after_player_edit.warnings)

    assert persist_reconciliation(db, hand.id).is_authoritative is True
    db.update_action(actions[0].model_copy(update={"notes": "verified sizing"}))
    after_action_edit = db.fetch_hand_settlement(hand.id)
    assert after_action_edit.status == "needs_correction"
    assert after_action_edit.is_balanced is False
    assert any("changed" in warning.lower() for warning in after_action_edit.warnings)
    assert reconcile_persisted_hand(db, hand.id).is_authoritative is False
    db.close()


def test_reconciliation_persists_a_derived_uncalled_refund() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Derived refund"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            pot_size=120,
            hero_bb_won=-60,
        )
    )
    hero = db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="hero",
            seat_index=0,
            player_name="Hero",
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
            starting_stack=60,
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key=hero.player_key,
            player_name=hero.player_name,
            street="river",
            action_type="all-in",
            amount=100,
            amount_semantics="incremental",
        )
    )
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key=villain.player_key,
            player_name=villain.player_name,
            street="river",
            action_type="call",
            amount=60,
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
                player_key=villain.player_key,
                player_name=villain.player_name,
                amount=120,
                entry_order=1,
            )
        ],
    )

    result = persist_reconciliation(db, hand.id)
    entries = db.fetch_settlement_entries(hand.id)
    refunds = [entry for entry in entries if entry.entry_type == "refund"]

    assert result.is_authoritative is True
    assert result.ledger.refunds == pytest.approx({"hero": 40, "villain": 0})
    assert len(refunds) == 1
    assert refunds[0].player_key == hero.player_key
    assert refunds[0].player_name == hero.player_name
    assert refunds[0].amount == pytest.approx(40)
    assert refunds[0].pot_index is None
    assert result.ledger.net_results == pytest.approx({"hero": -60, "villain": 60})
    db.close()


def test_analytics_uses_authoritative_result_and_observed_fallback() -> None:
    db = _make_db()
    session = db.create_session(Session(name="Mixed evidence"))
    authoritative, hero, _, _ = _create_heads_up_value_hand(
        db,
        session.id,
        hand_number=1,
        observed_pot=20,
        observed_hero_result=None,
    )
    assert persist_reconciliation(db, authoritative.id).is_authoritative is True
    observed = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=2,
            hero_bb_won=-4,
        )
    )

    stats = compute_session_stats(db, session.id)

    assert stats.hand_count == 2
    assert stats.hands_with_result == 2
    assert stats.reconciled_result_count == 1
    assert stats.observed_result_count == 1
    assert stats.total_hero_bb == pytest.approx(6)
    assert stats.average_hero_bb == pytest.approx(3)
    assert stats.bb_per_100 == pytest.approx(300)
    assert stats.biggest_winning_hands[0].id == authoritative.id
    assert stats.biggest_winning_hands[0].hero_bb_won == pytest.approx(
        10
    )
    assert stats.biggest_losing_hands[0].id == observed.id
    assert stats.biggest_losing_hands[0].hero_bb_won == pytest.approx(-4)
    assert hero.player_key == "hero-1"
    db.close()


# ---------------------------------------------------------------------------
# A refund that nobody was in a position to answer is not a fold win
# ---------------------------------------------------------------------------


def _hand_with_actions(
    db: PokerDatabase,
    session_id: int,
    rows: list[tuple[str, str, str, float | None]],
    *,
    winner: str,
    hero_stack: float = 100.0,
    villain_stack: float = 100.0,
) -> Hand:
    """Persist a heads-up hand from (player, street, action_type, amount) rows.

    Deliberately writes the records directly rather than going through
    ``manual_spot_entry``: this is the backstop for every OTHER way a settlement
    reaches the store -- a reconstructed hand whose closing action was never
    observed, an import payload, the settlement editor -- and it has to hold when
    the manual-entry validator was never involved.
    """
    hand = db.create_hand(
        Hand(session_id=session_id, hand_number=99, game_type="No-limit Hold'em")
    )
    seats = {
        "hero": db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key="hero",
                seat_index=0,
                player_name="Hero",
                position="BB",
                starting_stack=hero_stack,
                is_hero=True,
            )
        ),
        "villain": db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key="villain",
                seat_index=1,
                player_name="Villain",
                position="BTN",
                starting_stack=villain_stack,
            )
        ),
    }
    for key, street, action_type, amount in rows:
        player = seats[key]
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=player.player_key,
                player_name=player.player_name,
                position=player.position,
                street=street,
                action_type=action_type,
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
                player_key=seats[winner].player_key,
                player_name=seats[winner].player_name,
                entry_order=1,
            )
        ],
    )
    return hand


_TRUNCATED = [
    ("villain", "preflop", "bet", 2.5),
    ("hero", "preflop", "call", 2.5),
    ("hero", "flop", "check", None),
    ("villain", "flop", "bet", 3.5),
]


@pytest.mark.parametrize("winner", ["hero", "villain"])
def test_a_hand_that_stops_mid_wager_does_not_reconcile(winner: str) -> None:
    """Villain's 3.5 is refunded as uncalled while Hero still has cards and chips.

    The ledger cannot tell that from a fold win by looking at contributions, so
    it did both at once: refunded the wager AND paid the pot to the declared
    winner. That settlement asserts two things that cannot both be true, and it
    used to certify as reconciled, authoritative and study-ready with an empty
    issue tuple.
    """
    db = _make_db()
    session = db.create_session(Session(name="Mid-wager"))
    hand = _hand_with_actions(db, session.id, _TRUNCATED, winner=winner)

    result = persist_reconciliation(db, hand.id)

    assert result.ledger.refunds["villain"] == pytest.approx(3.5)
    assert result.issues, "a refunded wager nobody answered reconciled silently"
    assert any("never closed the betting" in issue for issue in result.issues)
    assert result.is_authoritative is False
    assert result.settlement.status == "needs_correction"
    db.close()


def test_a_recorded_fold_is_what_makes_the_same_refund_legitimate() -> None:
    """The control: identical chips, plus the one action that explains them."""
    db = _make_db()
    session = db.create_session(Session(name="Fold win"))
    hand = _hand_with_actions(
        db,
        session.id,
        [*_TRUNCATED, ("hero", "flop", "fold", None)],
        winner="villain",
    )

    result = persist_reconciliation(db, hand.id)

    assert result.ledger.refunds["villain"] == pytest.approx(3.5)
    assert result.issues == ()
    assert result.is_authoritative is True
    db.close()


def test_an_opponent_who_is_all_in_cannot_answer_and_is_not_a_missing_action() -> None:
    """The other legitimate source of an uncalled wager: nobody could cover it.

    Hero overbets a 20-chip stack; Villain is all-in for less and can still win
    the pot at showdown. Refusing this would turn the guard into a false blocker
    on an ordinary side-pot hand.
    """
    db = _make_db()
    session = db.create_session(Session(name="All-in"))
    hand = _hand_with_actions(
        db,
        session.id,
        [
            ("villain", "preflop", "bet", 2.5),
            ("hero", "preflop", "call", 2.5),
            ("hero", "flop", "bet", 50.0),
            ("villain", "flop", "all_in", 17.5),
        ],
        winner="villain",
        villain_stack=20.0,
    )

    result = persist_reconciliation(db, hand.id)

    assert result.ledger.refunds["hero"] == pytest.approx(32.5)
    assert result.issues == ()
    assert result.is_authoritative is True
    db.close()


def test_a_seat_that_never_put_a_chip_in_is_not_read_as_a_missing_decision() -> None:
    """A dealt-out or pre-recording-fold seat must not raise a false blocker.

    The ledger already leaves a zero-contribution seat out of every pot's
    eligible set; reading it as "a player who never answered" would block every
    ordinary hand carrying a seat row for someone who was not in it.
    """
    db = _make_db()
    session = db.create_session(Session(name="Ghost seat"))
    hand = _hand_with_actions(
        db,
        session.id,
        [
            ("villain", "preflop", "bet", 2.5),
            ("hero", "preflop", "call", 2.5),
            ("hero", "flop", "check", None),
            ("villain", "flop", "bet", 3.5),
            ("hero", "flop", "fold", None),
        ],
        winner="villain",
    )
    db.create_hand_player(
        HandPlayer(
            hand_id=hand.id,
            player_key="ghost",
            seat_index=2,
            player_name="Ghost",
            position="CO",
            starting_stack=100.0,
        )
    )

    result = persist_reconciliation(db, hand.id)

    assert result.issues == ()
    assert result.is_authoritative is True
    db.close()


def test_the_block_clears_by_recording_the_action_that_was_missing() -> None:
    """Fail closed onto a reachable fix, and file nothing fabricated on the way.

    ``persist_reconciliation`` derives and PERSISTS refund rows for a hand that
    has none. On an unfinished line the refund it would derive is the arithmetic
    of a fold nobody made, so writing it would file the fabrication in the store
    and leave a stale row to contradict the corrected line afterwards.
    """
    db = _make_db()
    session = db.create_session(Session(name="Corrected"))
    hand = _hand_with_actions(db, session.id, _TRUNCATED, winner="hero")

    blocked = persist_reconciliation(db, hand.id)
    assert blocked.is_authoritative is False
    assert [entry.entry_type for entry in blocked.entries] == ["award"]

    # The operator records what actually happened: Hero called.
    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="hero",
            player_name="Hero",
            position="BB",
            street="flop",
            action_type="call",
            amount=3.5,
            amount_semantics="incremental",
        )
    )
    # Then saves the settlement, which is what the Accounting reconciliation
    # panel does: it nulls the three recorded restatements of the rake policy so
    # they are re-derived rather than compared against the figures the blocked
    # save left behind (app.py, "gross_pot": None / "rake_amount": None /
    # "net_pot": None).
    db.upsert_hand_settlement(
        blocked.settlement.model_copy(
            update={"gross_pot": None, "rake_amount": None, "net_pot": None}
        )
    )
    fixed = persist_reconciliation(db, hand.id)

    assert fixed.issues == ()
    assert fixed.is_authoritative is True
    assert fixed.settlement.status == "reconciled"
    assert fixed.ledger.net_results["hero"] == pytest.approx(6.0)
    db.close()
