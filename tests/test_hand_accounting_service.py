from __future__ import annotations

import pytest

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
    STALE_AWARD_PREFIX,
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
