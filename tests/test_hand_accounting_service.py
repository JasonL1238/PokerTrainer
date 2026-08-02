from __future__ import annotations

import pytest
from pydantic import ValidationError

from poker_tracker.math.accounting import LedgerError
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
    BLIND_STRUCTURE_INPUT,
    STALE_AWARD_PREFIX,
    attest_assumption,
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


# --- Adversarial round 17: a durable ordinal into a derived structure ---------
#
# `settlement_entries.pot_index` is an ordinal into the pot layering this
# product derives, and the derivation moved: commit 3c3144e began cutting levels
# at dead contributions as well as live ones, changing both the count and the
# numbering of the layers of any hand containing a forced post. Awards already
# stored against the previous layering can name a pot this build does not
# produce. `_validate_winners` raises for that, and `reconcile_persisted_hand`
# does not catch it -- so a hand that was reconciled a release ago becomes an
# unhandled traceback for any caller that is not app.py.


def _stale_award_hand(db: PokerDatabase, session_id: int, *, pot_index: int) -> Hand:
    """A settled heads-up hand whose stored award names ``pot_index``.

    The row is written directly because that is what makes it the real case: an
    award row carries its ordinal and nothing about the layering that produced
    it, so a row saved under an older derivation and a row imported from another
    store are the same bytes here.
    """
    hand, hero, _villain, _actions = _create_heads_up_value_hand(db, session_id)
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=pot_index,
                player_key=hero.player_key,
                player_name=hero.player_name,
                amount=20,
                entry_order=1,
            )
        ],
    )
    return hand


def test_an_award_naming_a_pot_this_hand_no_longer_has_is_a_correction() -> None:
    """The stale ordinal is reported, not raised, and nothing is renumbered.

    This hand derives one layer. The stored award names pot 1. Before, that was
    `LedgerError: Winner declaration references missing pot 1` propagating out of
    `reconcile_persisted_hand` -- past `persist_reconciliation`, past every
    non-app.py caller -- so a hand that reconciled under the previous layering
    could not be read at all.
    """
    db = _make_db()
    session = db.create_session(Session(name="Stale ordinal"))
    hand = _stale_award_hand(db, session.id, pot_index=1)

    result = reconcile_persisted_hand(db, hand.id)

    assert result.is_authoritative is False
    assert any(issue.startswith(STALE_AWARD_PREFIX) for issue in result.issues)
    # Named precisely enough to act on: which pot was claimed, and how many the
    # hand actually has.
    assert any("missing pot 1" in issue for issue in result.issues)
    assert any("1 pot layer(s)" in issue for issue in result.issues)
    # The hand is still fully inspectable -- only the settlement is withdrawn.
    assert len(result.ledger.pots) == 1
    assert result.ledger.gross_pot == pytest.approx(20)
    assert result.ledger.is_settled is False
    # And the stored declaration is untouched, so re-declaring is an edit rather
    # than a data repair.
    assert [entry.pot_index for entry in result.entries] == [1]
    db.close()


def test_a_stale_award_persists_as_needs_correction_rather_than_reconciled() -> None:
    """The blocked verdict reaches the durable row the operator reads."""
    db = _make_db()
    session = db.create_session(Session(name="Stale ordinal persisted"))
    hand = _stale_award_hand(db, session.id, pot_index=1)

    result = persist_reconciliation(db, hand.id)

    assert result.settlement.status == "needs_correction"
    assert result.is_authoritative is False
    assert any(
        warning.startswith(STALE_AWARD_PREFIX) for warning in result.settlement.warnings
    )
    db.close()


def test_re_declaring_the_award_against_the_derived_layers_clears_the_block() -> None:
    """The correction is an ordinary settlement edit, not a migration."""
    db = _make_db()
    session = db.create_session(Session(name="Stale ordinal cleared"))
    hand = _stale_award_hand(db, session.id, pot_index=1)
    assert persist_reconciliation(db, hand.id).is_authoritative is False

    stored = db.fetch_settlement_entries(hand.id)
    db.replace_settlement_entries(
        hand.id,
        [entry.model_copy(update={"pot_index": 0}) for entry in stored],
    )
    db.upsert_hand_settlement(
        db.fetch_hand_settlement(hand.id).model_copy(
            update={"gross_pot": None, "rake_amount": None, "net_pot": None}
        )
    )

    fixed = persist_reconciliation(db, hand.id)

    assert fixed.issues == ()
    assert fixed.is_authoritative is True
    assert fixed.settlement.status == "reconciled"
    db.close()


def test_an_award_naming_a_seat_that_layer_no_longer_admits_is_the_same_correction() -> None:
    """The other half of the family: the index exists, the eligibility does not.

    A seat all-in for nothing but its ante is capped at the layer holding the
    antes. An award declaring it the winner of the live betting above that layer
    was recordable before 3c3144e and is refused now, which is the same durable
    row meeting a changed derivation.
    """
    db = _make_db()
    session = db.create_session(Session(name="Stale eligibility"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=7, game_type="No-limit Hold'em")
    )
    seats = [
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                seat_index=index,
                player_name=key.title(),
                position=position,
                starting_stack=stack,
                is_hero=(key == "short"),
            )
        )
        for index, (key, position, stack) in enumerate(
            [("short", "BTN", 1.0), ("alpha", "SB", 100.0), ("beta", "BB", 100.0)]
        )
    ]
    rows = [
        ("short", "ante", 1.0),
        ("alpha", "ante", 1.0),
        ("beta", "ante", 1.0),
        ("alpha", "bet", 10.0),
        ("beta", "call", 10.0),
    ]
    by_key = {seat.player_key: seat for seat in seats}
    for key, action_type, amount in rows:
        seat = by_key[key]
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=seat.player_key,
                player_name=seat.player_name,
                position=seat.position,
                street="preflop",
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
                pot_index=index,
                player_key="short",
                player_name="Short",
                entry_order=index + 1,
            )
            for index in (0, 1)
        ],
    )

    result = reconcile_persisted_hand(db, hand.id)

    assert result.is_authoritative is False
    assert any(issue.startswith(STALE_AWARD_PREFIX) for issue in result.issues)
    assert any("not eligible for pot 1" in issue for issue in result.issues)
    assert [pot.amount for pot in result.ledger.pots] == pytest.approx([3, 20])
    db.close()


def test_a_hand_that_is_impossible_without_its_awards_still_raises() -> None:
    """Withdrawing the awards is the test for whether the awards were the problem.

    A record whose action line cannot be reduced at all is not a settlement
    mismatch, and reporting it as one would file a misleading correction against
    a hand whose actual defect is elsewhere. The award-free rebuild raises the
    same error, so it propagates unchanged.
    """
    db = _make_db()
    session = db.create_session(Session(name="Impossible"))
    hand, hero, _villain, _actions = _create_heads_up_value_hand(db, session.id)
    db.update_hand_player(hero.model_copy(update={"starting_stack": 1.0}))

    with pytest.raises(LedgerError):
        reconcile_persisted_hand(db, hand.id)
    db.close()


# --- Adversarial round 17: closure is not the same fact as a refund -----------


def _three_handed_truncated_preflop(db: PokerDatabase, session_id: int) -> Hand:
    """Small blind 1, big blind 2, button calls 2, and the small blind never acts.

    Two seats tie at the top of the street, so nothing is uncalled and no refund
    is produced. The small blind is still in the hand, owes a chip and has
    ninety-eight behind.
    """
    hand = db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=42,
            game_type="No-limit Hold'em",
            pot_size=5,
            hero_bb_won=2,
        )
    )
    seats = [("sb", "SB", 0, True), ("bb", "BB", 1, False), ("btn", "BTN", 2, False)]
    for key, position, index, is_hero in seats:
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                seat_index=index,
                player_name=key.upper(),
                position=position,
                starting_stack=100,
                is_hero=is_hero,
            )
        )
    for key, action_type, amount in [
        ("sb", "post_blind", 1.0),
        ("bb", "post_blind", 2.0),
        ("btn", "call", 2.0),
    ]:
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=key.upper(),
                position=key.upper(),
                street="preflop",
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
                player_key="sb",
                player_name="SB",
                amount=3,
                entry_order=1,
            ),
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=1,
                player_key="bb",
                player_name="BB",
                amount=2,
                entry_order=2,
            ),
        ],
    )
    return hand


def test_a_line_that_stops_with_no_refund_to_key_on_still_does_not_reconcile() -> None:
    """The unanswered-wager guard read the symptom, so a tie hid the disease.

    The previous guard started from a REFUND: find a seat handed uncalled money
    back, then ask whether anybody who could have called it was still there. A
    refund only exists when one seat is uniquely ahead, so the moment two seats
    tie at the top of the street the guard never ran at all. This hand -- the
    small blind never acting after the button flats -- reconciled as
    authoritative, balanced, legal and warning-free, and recorded the small blind
    winning two big blinds it never paid to see.
    """
    db = _make_db()
    session = db.create_session(Session(name="Tie at the top"))
    hand = _three_handed_truncated_preflop(db, session.id)

    result = persist_reconciliation(db, hand.id)

    assert all(refund == 0 for refund in result.ledger.refunds.values())
    assert result.is_authoritative is False
    assert result.settlement.status == "needs_correction"
    assert any("never closed the betting" in issue for issue in result.issues)
    assert any("SB" in issue for issue in result.issues)
    db.close()


def test_recording_the_missing_call_closes_the_line_and_the_hand_reconciles() -> None:
    """The guard fails closed onto an action the operator can actually record."""
    db = _make_db()
    session = db.create_session(Session(name="Tie at the top, corrected"))
    hand = _three_handed_truncated_preflop(db, session.id)
    assert persist_reconciliation(db, hand.id).is_authoritative is False

    db.create_action(
        Action(
            hand_id=hand.id,
            player_key="sb",
            player_name="SB",
            position="SB",
            street="preflop",
            action_type="call",
            amount=1.0,
            amount_semantics="incremental",
        )
    )
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="sb",
                player_name="SB",
                amount=6,
                entry_order=1,
            )
        ],
    )
    db.upsert_hand_settlement(
        db.fetch_hand_settlement(hand.id).model_copy(
            update={"gross_pot": None, "rake_amount": None, "net_pot": None}
        )
    )
    db.update_hand_facts(
        db.fetch_hand(hand.id).model_copy(update={"pot_size": 6.0, "hero_bb_won": 4.0})
    )

    fixed = persist_reconciliation(db, hand.id)

    assert fixed.issues == ()
    assert fixed.is_authoritative is True
    assert fixed.ledger.net_results["sb"] == pytest.approx(4)
    db.close()


# --- Adversarial round 18: the phantom side pot, end to end ------------------


def _phantom_dead_money_hand(db: PokerDatabase, session_id: int) -> Hand:
    """Four seats, unequal dead money, one live wager everybody matched.

    ``A`` posts a 5 ante and ``B`` a 3 dead blind; ``C`` and ``D`` owe nothing
    dead. All four put in 20 live and the betting closes. Hero is ``C``, the seat
    that owes no dead money, and ``C`` wins the hand: gross 88, hero net +68.
    """
    hand = db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=181,
            game_type="No-limit Hold'em",
            pot_size=88,
            hero_bb_won=68,
        )
    )
    for key, index in (("a", 0), ("b", 1), ("c", 2), ("d", 3)):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                seat_index=index,
                player_name=key.upper(),
                starting_stack=100,
                is_hero=(key == "c"),
            )
        )
    rows = [
        ("a", "ante", 5.0, None),
        ("b", "post_blind", 3.0, False),
        ("a", "bet", 20.0, None),
        ("b", "call", 20.0, None),
        ("c", "call", 20.0, None),
        ("d", "call", 20.0, None),
    ]
    for key, action_type, amount, live_post in rows:
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=key.upper(),
                street="preflop",
                action_type=action_type,
                amount=amount,
                amount_semantics="incremental",
                is_live_post=live_post,
            )
        )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
    return hand


def test_unequal_dead_money_no_longer_funnels_the_operator_into_a_wrong_payout() -> None:
    """The whole defect, measured where the operator meets it.

    Cutting a layer at every distinct TOTAL commitment derived a main pot of 80
    plus an 8-chip "side pot" only the two seats that owed dead money could win.
    Every exit from that was worse than the last. The truthful award -- C takes
    everything -- raised "C is not eligible for pot 1"; leaving pot 1 undeclared
    left the hand unsettled and unbalanced; and declaring pot 1 to A or B was the
    only thing the product accepted, reconciling as authoritative with C's net
    short by the dead money and no issue, no warning and no correction anywhere
    on the record.

    There is one pot now, the operator declares the winner of it, and the derived
    hero result is the one the hand actually produced.
    """
    db = _make_db()
    session = db.create_session(Session(name="Phantom side pot"))
    hand = _phantom_dead_money_hand(db, session.id)

    derived = reconcile_persisted_hand(db, hand.id)
    assert [pot.amount for pot in derived.ledger.pots] == pytest.approx([88])
    assert set(derived.ledger.pots[0].eligible_players) == {"a", "b", "c", "d"}

    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="c",
                player_name="C",
                amount=88,
                entry_order=1,
            )
        ],
    )
    result = persist_reconciliation(db, hand.id)

    assert result.issues == ()
    assert result.is_authoritative is True
    assert result.settlement.status == "reconciled"
    assert result.settlement.gross_pot == pytest.approx(88)
    assert result.ledger.net_results["c"] == pytest.approx(68)
    assert sum(result.ledger.net_results.values()) + result.ledger.rake == pytest.approx(0)
    db.close()


def test_the_dead_money_cannot_be_awarded_to_a_seat_that_did_not_win_the_hand() -> None:
    """The declaration the product used to accept is the one it now refuses.

    Paying A the 8 chips of dead money as a "side pot" was the resting state the
    operator was funnelled into. There is no pot 1 to declare it against, so the
    award is reported as a stale claim naming a layer this hand does not have
    rather than being reconciled into a wrong hero result.
    """
    db = _make_db()
    session = db.create_session(Session(name="No layer to hide in"))
    hand = _phantom_dead_money_hand(db, session.id)

    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="c",
                player_name="C",
                amount=80,
                entry_order=1,
            ),
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=1,
                player_key="a",
                player_name="A",
                amount=8,
                entry_order=2,
            ),
        ],
    )
    result = persist_reconciliation(db, hand.id)

    assert result.is_authoritative is False
    assert result.settlement.status == "needs_correction"
    assert any(issue.startswith(STALE_AWARD_PREFIX) for issue in result.issues)
    db.close()


def test_a_short_all_in_does_not_reopen_the_phantom_at_the_product_boundary() -> None:
    """The reported funnel, measured with one extra seat at the table.

    Removing the phantom only from hands where nobody is short leaves it in every
    hand that has a short all-in anywhere in it, which is most hands with an
    all-in. Here ``e`` is all-in having wagered 16 live behind a 4-chip dead
    blind, while ``a``'s ante and ``b``'s dead blind sit above it in the ladder.

    Under the live-level model the seat that must be refused is ``e`` and only
    ``e``. ``c`` and ``d`` wagered the full 20 exactly as ``a`` and ``b`` did, so
    nothing separates the four of them and any of them may be declared the winner
    of all 108; only dead money made them look different, and dead money opens no
    boundary. ``e`` wagered 16, the table matched 16 of it, and the layer above
    that is 16 chips ``e`` cannot reach -- so the declaration this refuses at the
    product boundary is ``e`` taking the hand outright, which is 16 chips more
    than anybody wagered against it.

    Both halves are asserted, because a repair that satisfies one by discarding
    the other is exactly how this module produced five consecutive criticals.
    """
    db = _make_db()
    session = db.create_session(Session(name="Phantom behind an all-in"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=182,
            game_type="No-limit Hold'em",
            pot_size=108,
            hero_bb_won=88,
        )
    )
    for key, index, stack in (("a", 0, 100), ("b", 1, 100), ("c", 2, 100), ("d", 3, 100), ("e", 4, 20)):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                seat_index=index,
                player_name=key.upper(),
                starting_stack=stack,
                is_hero=(key == "c"),
            )
        )
    rows = [
        ("a", "ante", 5.0, None),
        ("b", "post_blind", 3.0, False),
        ("e", "post_blind", 4.0, False),
        ("a", "bet", 20.0, None),
        ("b", "call", 20.0, None),
        ("c", "call", 20.0, None),
        ("d", "call", 20.0, None),
        ("e", "all-in", 16.0, None),
    ]
    for key, action_type, amount, live_post in rows:
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=key.upper(),
                street="preflop",
                action_type=action_type,
                amount=amount,
                amount_semantics="incremental",
                is_live_post=live_post,
            )
        )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))

    derived = reconcile_persisted_hand(db, hand.id)
    assert [pot.amount for pot in derived.ledger.pots] == pytest.approx([92, 16])
    # The service orders seats hero-first, so compare membership.
    assert set(derived.ledger.pots[0].eligible_players) == {"a", "b", "c", "d", "e"}
    assert set(derived.ledger.pots[1].eligible_players) == {"a", "b", "c", "d"}

    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=index,
                player_key="e",
                player_name="E",
                amount=amount,
                entry_order=index + 1,
            )
            for index, amount in ((0, 92.0), (1, 16.0))
        ],
    )
    result = persist_reconciliation(db, hand.id)

    assert result.is_authoritative is False
    assert result.settlement.status == "needs_correction"
    assert any("not eligible for pot 1" in issue for issue in result.issues), result.issues
    db.close()


# ---------------------------------------------------------------------------
# The blind structure, end to end through the store
# ---------------------------------------------------------------------------


def _create_short_blind_hand(db: PokerDatabase, session_id: int) -> Hand:
    """Blinds 5/10, big blind all-in for 4, button calls the real 10.

    The reported hand. Before the blind structure existed the reducer told the
    button that the amount to call was 5, and a hand built around that reconciled
    around a 14-chip pot whose truth is 24.
    """
    hand = db.create_hand(
        Hand(session_id=session_id, hand_number=1, blinds_antes="5/10 NL")
    )
    seats = [
        ("sb", "SB", "SB", 200.0, False),
        ("bb", "BB", "BB", 4.0, False),
        ("btn", "BTN", "BTN", 200.0, True),
    ]
    for index, (key, name, position, stack, is_hero) in enumerate(seats):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                seat_index=index,
                player_name=name,
                position=position,
                starting_stack=stack,
                is_hero=is_hero,
            )
        )
    line = [
        ("sb", "SB", "post_blind", 5.0),
        ("bb", "BB", "post_blind", 4.0),
        ("btn", "BTN", "call", 10.0),
        ("sb", "SB", "call", 5.0),
    ]
    for key, name, action_type, amount in line:
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                street="preflop",
                action_type=action_type,
                amount=amount,
                amount_semantics="incremental",
                is_live_post=True if action_type == "post_blind" else None,
            )
        )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=index,
                player_key="btn",
                player_name="BTN",
                entry_order=index + 1,
            )
            for index in range(2)
        ],
    )
    return hand


def test_a_hand_with_an_unreadable_forced_post_cannot_reconcile_undeclared() -> None:
    """No declaration, no authority -- and the operator is told exactly why."""
    db = _make_db()
    session = db.create_session(Session(name="blinds"))
    hand = _create_short_blind_hand(db, session.id)

    result = persist_reconciliation(db, hand.id)

    assert result.is_authoritative is False
    assert result.settlement is not None
    assert result.settlement.status == "needs_correction"
    assert any("Declare the blind structure" in issue for issue in result.issues)
    # Blocked, not blanked: the pot is still the truthful 24.
    assert result.ledger.gross_pot == pytest.approx(24)
    db.close()


def test_declaring_the_blind_structure_clears_it_and_is_measured() -> None:
    """The clearing action is one ordinary settlement edit, and it is disclosed.

    The structure moves no chip figure, so it is measured as verdict-only -- the
    neutral pass stops reconciling while every reported number stands still. That
    is precisely the case ``_is_dependent``'s verdict half exists for, and it is
    what stops "somebody typed 5/10" from being an invisible input.
    """
    db = _make_db()
    session = db.create_session(Session(name="blinds"))
    hand = _create_short_blind_hand(db, session.id)

    stored = db.fetch_hand_settlement(hand.id)
    db.upsert_hand_settlement(
        stored.model_copy(update={"small_blind": 5.0, "big_blind": 10.0})
    )
    result = persist_reconciliation(db, hand.id)

    assert result.is_authoritative is True
    assert result.ledger.gross_pot == pytest.approx(24)
    assert not any("Declare the blind structure" in issue for issue in result.issues)

    named = [
        item
        for item in result.assumption_dependence
        if item.input_name == BLIND_STRUCTURE_INPUT
    ]
    assert len(named) == 1
    assert named[0].declared == "5/10"
    assert named[0].neutral == "no declared blind structure"
    assert named[0].deltas == ()
    assert "verdict-only" in named[0].code

    # The attestation door refuses a code naming no current dependence, so the
    # measurement -- not the shape of the string -- is what a confirmation binds
    # to. (This hand is manual, and a manual hand is exempt from attesting at
    # all, so the honest expectation for both calls here is a refusal.)
    assert attest_assumption(db, hand.id, "not-a-measured-code") is False
    db.close()


def test_re_declaring_the_blind_structure_lapses_the_attestation() -> None:
    """A different room is a different claim, so the confirmation cannot carry."""
    db = _make_db()
    session = db.create_session(Session(name="blinds"))
    hand = _create_short_blind_hand(db, session.id)
    stored = db.fetch_hand_settlement(hand.id)
    db.upsert_hand_settlement(
        stored.model_copy(update={"small_blind": 5.0, "big_blind": 10.0})
    )
    first = persist_reconciliation(db, hand.id)
    first_code = next(
        item.code
        for item in first.assumption_dependence
        if item.input_name == BLIND_STRUCTURE_INPUT
    )

    # A different room, chosen so the hand still reconciles: the floor is the
    # big blind either way, so nothing about the verdict or a figure moves and
    # ONLY the declared text differs. That is the case an attestation is most
    # likely to be inherited across, and it must not be.
    db.upsert_hand_settlement(
        db.fetch_hand_settlement(hand.id).model_copy(
            update={"small_blind": 2.0, "big_blind": 10.0}
        )
    )
    second = persist_reconciliation(db, hand.id)
    second_code = next(
        item.code
        for item in second.assumption_dependence
        if item.input_name == BLIND_STRUCTURE_INPUT
    )
    assert second_code != first_code
    db.close()


def test_a_hand_whose_posts_were_all_made_in_full_is_disclosed_nothing() -> None:
    """The silent half, which matters as much as the blocking half.

    Where every forced post was made in full the structure is not load-bearing:
    the observed maximum already IS it. Naming a dependence there would train the
    operator to click through the disclosures that do mean something.
    """
    db = _make_db()
    session = db.create_session(Session(name="full posts"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    for key, name, stack, is_hero in (
        ("sb", "SB", 200.0, False),
        ("bb", "BB", 200.0, True),
    ):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                position=name,
                starting_stack=stack,
                is_hero=is_hero,
            )
        )
    for key, name, action_type, amount, live in (
        ("sb", "SB", "post_blind", 5.0, True),
        ("bb", "BB", "post_blind", 10.0, True),
        ("sb", "SB", "call", 5.0, None),
        ("bb", "BB", "check", None, None),
    ):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                street="preflop",
                action_type=action_type,
                amount=amount,
                amount_semantics="incremental",
                is_live_post=live,
            )
        )
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id, status="settled", small_blind=5.0, big_blind=10.0
        )
    )
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="bb",
                player_name="BB",
                entry_order=1,
            )
        ],
    )

    result = persist_reconciliation(db, hand.id)
    assert result.is_authoritative is True
    assert [
        item
        for item in result.assumption_dependence
        if item.input_name == BLIND_STRUCTURE_INPUT
    ] == []
    db.close()


def test_a_half_declared_blind_structure_cannot_be_stored() -> None:
    """"Declared" and "usable" have to be the same state, or a saved row blocks forever."""
    with pytest.raises(ValidationError):
        HandSettlement(hand_id=1, small_blind=5.0)
    with pytest.raises(ValidationError):
        HandSettlement(hand_id=1, straddles=[20.0])
    with pytest.raises(ValidationError):
        HandSettlement(hand_id=1, small_blind=10.0, big_blind=5.0)
    with pytest.raises(ValidationError):
        HandSettlement(hand_id=1, small_blind=5.0, big_blind=10.0, straddles=[10.0])
    # Unstated small blind is legitimate: it moves nothing and 0 would be a claim.
    assert HandSettlement(hand_id=1, big_blind=10.0).small_blind is None


def test_the_blind_structure_survives_a_store_round_trip() -> None:
    db = _make_db()
    session = db.create_session(Session(name="round trip"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id, small_blind=2.0, big_blind=5.0, straddles=[10.0, 20.0]
        )
    )
    stored = db.fetch_hand_settlement(hand.id)
    assert stored.small_blind == pytest.approx(2.0)
    assert stored.big_blind == pytest.approx(5.0)
    assert stored.straddles == pytest.approx([10.0, 20.0])
    db.close()


def test_an_unreadable_straddles_column_blocks_instead_of_shrinking_the_floor() -> None:
    """Degrading a declaration must not quietly make it weaker.

    ``hand_settlements`` has no CHECK constraint, so a hand-edited or forged row
    can hold junk in ``straddles``. Dropping it to an empty list -- which is what
    the neighbouring ``warnings`` column does, correctly, because losing a note
    only removes noise -- would LOWER the structural forced bet the amount to
    call is floored at. So the row is reported as degraded instead: the column is
    named, the status cannot read ``reconciled``, and the hand blocks.
    """
    db = _make_db()
    session = db.create_session(Session(name="corrupt"))
    hand = _create_short_blind_hand(db, session.id)
    db.upsert_hand_settlement(
        db.fetch_hand_settlement(hand.id).model_copy(
            update={"small_blind": 5.0, "big_blind": 10.0, "straddles": [20.0]}
        )
    )
    db._execute(
        "UPDATE hand_settlements SET straddles = ? WHERE hand_id = ?",
        ("not json at all", hand.id),
    )
    db._commit()

    stored = db.fetch_hand_settlement(hand.id)
    assert stored.unreadable_columns == ("straddles",)
    assert stored.straddles == []
    assert any("straddles" in note for note in stored.warnings)
    assert reconcile_persisted_hand(db, hand.id).is_authoritative is False
    db.close()


def test_a_transposed_blind_structure_cannot_reach_the_disk() -> None:
    """``model_copy(update=...)`` skips validators, so the WRITE has to validate.

    Every editor in the app assembles the row it saves with ``model_copy``, which
    does not run ``HandSettlement``'s model validators. A "5/10" typed into the
    fields in the order an operator says it -- small 10, big 5 -- therefore
    reached the disk as a structure the class itself refuses. Re-validating at
    the write makes the class's rules true of every stored row however the caller
    built it.
    """
    db = _make_db()
    session = db.create_session(Session(name="transposed"))
    hand = _create_short_blind_hand(db, session.id)
    stored = db.fetch_hand_settlement(hand.id)

    impossible = stored.model_copy(update={"small_blind": 10.0, "big_blind": 5.0})
    assert impossible.small_blind == 10.0  # model_copy really did accept it

    with pytest.raises(ValidationError):
        db.upsert_hand_settlement(impossible)

    # Nothing was written, so the hand is still blocked on the undeclared
    # structure rather than reconciled around a smaller one.
    unchanged = db.fetch_hand_settlement(hand.id)
    assert (unchanged.small_blind, unchanged.big_blind) == (None, None)
    assert reconcile_persisted_hand(db, hand.id).is_authoritative is False
    db.close()


def test_an_unreadable_blind_structure_is_dropped_whole_not_salvaged_in_half() -> None:
    """Half a refused declaration is worse than none of it.

    ``hand_settlements`` carries no CHECK constraint, so a forged or hand-edited
    row can hold a transposed structure the writer would refuse. Probing the
    three columns one at a time kept whichever half validated alone: from small
    10 / big 5 the reader kept the 5 -- a perfectly valid structure, half the
    real size, and low enough that a big blind all-in for 4 clears it. The
    refusal the declaration exists to raise then never fired and the hand
    reconciled around a 14-chip pot whose truth is 24. A declaration that cannot
    be read together was not made, so all three columns go.
    """
    db = _make_db()
    session = db.create_session(Session(name="forged"))
    hand = _create_short_blind_hand(db, session.id)
    db._execute(
        "UPDATE hand_settlements SET small_blind = ?, big_blind = ? WHERE hand_id = ?",
        (10.0, 5.0, hand.id),
    )
    db._commit()

    stored = db.fetch_hand_settlement(hand.id)
    assert (stored.small_blind, stored.big_blind, stored.straddles) == (None, None, [])
    assert any("big_blind" in note for note in stored.warnings)

    result = reconcile_persisted_hand(db, hand.id)
    assert result.is_authoritative is False
    assert result.ledger.is_legal is False
    assert any(
        "Declare the blind structure" in issue
        for issue in result.ledger.legality_issues
    )
    db.close()


def test_a_readable_blind_structure_survives_a_row_degraded_for_another_reason() -> None:
    """Dropping the structure as a unit must not drop an intact one.

    The group probe is a refusal, not a blanket. A row degraded by an unrelated
    column keeps a declaration that reads perfectly well beside it, so an
    operator does not lose a structure they correctly declared because the rake
    cell was corrupted.
    """
    db = _make_db()
    session = db.create_session(Session(name="partly-corrupt"))
    hand = _create_short_blind_hand(db, session.id)
    db.upsert_hand_settlement(
        db.fetch_hand_settlement(hand.id).model_copy(
            update={"small_blind": 5.0, "big_blind": 10.0}
        )
    )
    db._execute(
        "UPDATE hand_settlements SET rake_rounding_unit = ? WHERE hand_id = ?",
        (-1.0, hand.id),
    )
    db._commit()

    stored = db.fetch_hand_settlement(hand.id)
    assert (stored.small_blind, stored.big_blind) == (5.0, 10.0)
    assert any("rake_rounding_unit" in note for note in stored.warnings)
    db.close()


def test_a_short_blind_booked_as_an_all_in_still_blocks_the_persisted_hand() -> None:
    """The refusal has to survive the action type a recording chose.

    A blind that takes its poster's last chip is commonly written as an all-in.
    While the recording still names the forced-bet type, the persisted hand is
    refused exactly as the ``post_blind`` shape is -- otherwise the whole
    declaration is one relabel away from being skipped.
    """
    db = _make_db()
    session = db.create_session(Session(name="relabelled"))
    hand = _create_short_blind_hand(db, session.id)
    blind_row = [
        action
        for action in db.fetch_actions_by_hand(hand.id)
        if action.player_key == "bb"
    ][0]
    db.update_action(
        blind_row.model_copy(
            update={"action_type": "all-in", "forced_bet_type": "big_blind"}
        )
    )

    result = reconcile_persisted_hand(db, hand.id)
    assert result.ledger.is_legal is False
    assert any(
        "Declare the blind structure" in issue
        for issue in result.ledger.legality_issues
    )
    db.close()


def _short_ante_hand(db: PokerDatabase, session_id: int, c_row: tuple) -> Hand:
    """Worked example (c) at 10x, with the short seat's ante row spelled by caller.

    A 10-chip ante and 5/10 blinds, four seats, ``c`` holding exactly the ante.
    Truth: main 40 [a, b, c, d], side 300 [a, b, d]; ``c`` wins the main pot for
    net +30 and ``a`` takes the side pot.
    """
    hand = db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=182,
            game_type="No-limit Hold'em",
            pot_size=340,
            hero_bb_won=-110,
        )
    )
    for key, index, stack in (("a", 0, 200), ("b", 1, 200), ("c", 2, 10), ("d", 3, 200)):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                seat_index=index,
                player_name=key.upper(),
                starting_stack=stack,
                is_hero=(key == "d"),
            )
        )
    rows = [
        ("a", "ante", 10.0, None, None, "preflop"),
        ("b", "ante", 10.0, None, None, "preflop"),
        c_row,
        ("d", "ante", 10.0, None, None, "preflop"),
        ("a", "post_blind", 5.0, None, None, "preflop"),
        ("b", "post_blind", 10.0, None, None, "preflop"),
        ("d", "raise", 100.0, None, None, "preflop"),
        ("a", "call", 95.0, None, None, "preflop"),
        ("b", "call", 90.0, None, None, "preflop"),
        ("d", "check", None, None, None, "flop"),
        ("a", "check", None, None, None, "flop"),
        ("b", "check", None, None, None, "flop"),
    ]
    for key, action_type, amount, live_post, forced, street in rows:
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=key.upper(),
                street=street,
                action_type=action_type,
                amount=amount,
                amount_semantics="incremental" if amount is not None else "unknown",
                is_live_post=live_post,
                forced_bet_type=forced,
            )
        )
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id, status="settled", small_blind=5, big_blind=10
        )
    )
    return hand


def test_relabelling_an_ante_row_cannot_move_a_chip_at_the_product_boundary() -> None:
    """The two durable columns the hand editor writes must not change the chips.

    ``actions.forced_bet_type`` and ``actions.is_live_post`` are set from two
    selectboxes on EVERY action row, including all-in, and a forced post that
    took its poster's last chip is routinely booked as an all-in. The money
    classifier read only ``action_type``, so the same event spelled the second way
    was counted as chosen live money -- and under the live-level model that moved
    30 chips into the main pot and paid them to a seat whose live commitment is
    zero. It reconciled as authoritative with no issue and no warning.
    """
    db = _make_db()
    session = db.create_session(Session(name="Ante spelling"))
    plain = _short_ante_hand(
        db, session.id, ("c", "ante", 10.0, None, None, "preflop")
    )
    relabelled = _short_ante_hand(
        db, session.id, ("c", "all_in", 10.0, False, "ante", "preflop")
    )

    for hand in (plain, relabelled):
        db.replace_settlement_entries(
            hand.id,
            [
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=0,
                    player_key="c",
                    player_name="C",
                    amount=40,
                    entry_order=1,
                ),
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=1,
                    player_key="a",
                    player_name="A",
                    amount=300,
                    entry_order=2,
                ),
            ],
        )

    truth = persist_reconciliation(db, plain.id)
    other = persist_reconciliation(db, relabelled.id)

    assert [pot.amount for pot in truth.ledger.pots] == pytest.approx([40, 300])
    assert truth.ledger.net_results["c"] == pytest.approx(30)
    assert truth.is_authoritative is True
    # Same hand, other spelling: same chips, same verdict.
    assert [pot.amount for pot in other.ledger.pots] == pytest.approx([40, 300])
    assert other.ledger.contributions == pytest.approx(truth.ledger.contributions)
    assert other.ledger.net_results == pytest.approx(truth.ledger.net_results)
    assert other.is_authoritative is True
    assert other.issues == ()
    db.close()


def test_a_forced_post_no_seat_could_cover_is_refused_as_study_ready() -> None:
    """Rule 2's undecided case reaches the operator instead of the study queue.

    Antes of 100 with a 40-chip stack short of its own ante: the model pays that
    stack all five opponents' full antes, 300 chips more than any of them covered
    of it, and the four worked examples do not decide whether that is right.
    Nothing here changes a chip. What it changes is that the hand can no longer be
    published as authoritative while the question is open.
    """
    db = _make_db()
    session = db.create_session(Session(name="Short of the ante"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=183,
            game_type="No-limit Hold'em",
            pot_size=740,
        )
    )
    seats = (("btn", 0, 40, True), ("sb", 1, 5000, False), ("bb", 2, 5000, False))
    for key, index, stack, hero in seats:
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                seat_index=index,
                player_name=key.upper(),
                starting_stack=stack,
                is_hero=hero,
            )
        )
    rows = [
        ("btn", "ante", 40.0),
        ("sb", "ante", 100.0),
        ("bb", "ante", 100.0),
        ("sb", "post_blind", 100.0),
        ("bb", "post_blind", 200.0),
        ("sb", "fold", None),
    ]
    for key, action_type, amount in rows:
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=key.upper(),
                street="preflop",
                action_type=action_type,
                amount=amount,
                amount_semantics="incremental" if amount is not None else "unknown",
            )
        )
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="settled", small_blind=100, big_blind=200)
    )
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="btn",
                player_name="BTN",
                amount=240,
                entry_order=1,
            ),
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=1,
                player_key="bb",
                player_name="BB",
                amount=200,
                entry_order=2,
            ),
        ],
    )

    result = persist_reconciliation(db, hand.id)

    # The awards declared above ARE the derived ladder, to the chip, so the only
    # thing standing between this hand and "reconciled" is the open question.
    assert [pot.amount for pot in result.ledger.pots] == pytest.approx([240, 200])
    assert result.ledger.is_legal is True
    assert result.ledger.is_settled is True
    assert result.ledger.is_balanced is True
    assert result.is_authoritative is False
    assert result.settlement.status == "needs_correction"
    assert any("forced post" in issue for issue in result.issues)
    assert result.ledger.net_results["btn"] == pytest.approx(200)
    db.close()
