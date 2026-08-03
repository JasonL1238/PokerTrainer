"""The ante mode as a PERSISTED declaration, and the migration it forces.

Ruling 2 says to model this on the blind structure, which is the established
precedent in this codebase for exactly this shape: a fact about the room the
action line cannot demonstrate, declared, refused when absent, persisted,
attestable, and stale-making when edited. These are the tests for the parts of
that sentence that only exist once the declaration is on disk.

THE MIGRATION IS THE PART THAT NEEDS THE MOST CARE, so it is the first section.
Every hand already in the store that contains an ante has no declared mode, and
under the ruling those hands are ambiguous and must be refused rather than
inferred -- which means PREVIOUSLY-RECONCILED HANDS START BLOCKING. Every hand
with NO antes must be untouched, because ``NONE`` is not a guess for them.
"""

from __future__ import annotations

import pytest

from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import (
    ANTE_MODE_INPUT,
    persist_reconciliation,
    reconcile_persisted_hand,
)


@pytest.fixture()
def db(tmp_path):
    database = PokerDatabase(tmp_path / "ante_mode.db")
    database.init_db()
    yield database
    database.close()


def _seed(
    database: PokerDatabase,
    *,
    rows,
    stacks,
    settlement_kwargs=None,
    awards=(),
) -> Hand:
    session = database.create_session(Session(name="Ante mode"))
    hand = database.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="No-limit Hold'em")
    )
    for index, (key, stack) in enumerate(stacks):
        database.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                seat_index=index,
                player_name=key.upper(),
                starting_stack=stack,
            )
        )
    for key, action_type, amount, forced, live_post in rows:
        database.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                player_name=key.upper(),
                street="preflop",
                action_type=action_type,
                amount=amount,
                amount_semantics="incremental" if amount is not None else "unknown",
                forced_bet_type=forced,
                is_live_post=live_post,
            )
        )
    database.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="settled", **(settlement_kwargs or {}))
    )
    if awards:
        database.replace_settlement_entries(
            hand.id,
            [
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=index,
                    player_key=key,
                    player_name=key.upper(),
                    amount=None,
                    entry_order=index + 1,
                )
                for index, key in enumerate(awards)
            ],
        )
    return hand


# The reported ante hand: three seats ante 1, one of them all-in from its ante.
_ANTE_ROWS = (
    ("a", "ante", 1.0, "ante", False),
    ("b", "ante", 1.0, "ante", False),
    ("c", "ante", 1.0, "ante", False),
    ("a", "bet", 10.0, None, None),
    ("b", "call", 10.0, None, None),
)
_ANTE_STACKS = (("a", 100.0), ("b", 100.0), ("c", 1.0))

# The same shape with no ante anywhere: a plain blinds hand.
_NO_ANTE_ROWS = (
    ("a", "post_blind", 5.0, "small_blind", True),
    ("b", "post_blind", 10.0, "big_blind", True),
    ("c", "call", 10.0, None, None),
    ("a", "call", 5.0, None, None),
    ("b", "check", None, None, None),
)
_NO_ANTE_STACKS = (("a", 100.0), ("b", 100.0), ("c", 100.0))


# ---------------------------------------------------------------------------
# MIGRATION IMPACT
# ---------------------------------------------------------------------------


def test_the_schema_carries_the_declaration_and_defaults_it_to_undeclared(db):
    """Schema 20 adds one nullable column and backfills nothing.

    NULL is the honest value and the only safe one: every alternative is a claim
    about a room nobody recorded. ``NONE`` is a lie on any hand with an ante;
    ``PER_PLAYER`` asserts that every stored hand anted individually, when a
    big-blind ante is the commonest ante structure in modern tournaments; and
    reading "exactly one seat anted" as a table ante is the inference ruling 2
    forbids by name.
    """

    assert SCHEMA_VERSION >= 20
    hand = _seed(db, rows=_ANTE_ROWS, stacks=_ANTE_STACKS)
    stored = db.fetch_hand_settlement(hand.id)
    assert stored is not None
    assert stored.ante_mode is None


def test_a_stored_hand_with_antes_starts_blocking_and_names_its_clearing_action(db):
    """THE DEMOTION, stated as the operator meets it.

    A hand that reconciled before this column existed now lands in
    ``needs_correction`` with a named issue. That is the ruling and not a side
    effect. What makes it acceptable rather than a regression is the other three
    things asserted here: the chip figures are unchanged, the issue names the
    anteing seats, and the clearing action is one ordinary settlement edit.
    """

    hand = _seed(db, rows=_ANTE_ROWS, stacks=_ANTE_STACKS, awards=("c", "a"))
    result = persist_reconciliation(db, hand.id)

    assert result.settlement.status == "needs_correction"
    assert result.is_authoritative is False
    blocker = next(
        note for note in result.issues if "Declare the ante mode" in note
    )
    assert "'a'" in blocker and "'b'" in blocker and "'c'" in blocker
    assert "Declare the ante mode" in blocker

    # Nothing about the chips moved; only the verdict.
    assert [pot.amount for pot in result.ledger.pots] == pytest.approx([3, 20])
    assert result.ledger.net_results["c"] == pytest.approx(2)

    # ONE settlement edit clears it, and it is the same save the operator already
    # makes for the blind structure or the rake policy.
    db.upsert_hand_settlement(
        result.settlement.model_copy(update={"ante_mode": "PER_PLAYER"})
    )
    cleared = persist_reconciliation(db, hand.id)
    assert cleared.settlement.status == "reconciled"
    assert cleared.is_authoritative is True
    assert [pot.amount for pot in cleared.ledger.pots] == pytest.approx([3, 20])
    assert cleared.ledger.net_results["c"] == pytest.approx(2)


def test_a_stored_hand_with_no_antes_is_not_demoted_by_the_migration(db):
    """The half of the migration that must change NOTHING.

    ``NONE`` is not a guess for a hand with no antes -- it is the only thing such
    a hand can be -- so the absent declaration is resolved silently. Every
    ordinary cash-game hand in the store therefore reconciles on the day this
    column ships exactly as it did the day before, with no new blocker, no new
    warning, and no re-save required.
    """

    hand = _seed(
        db,
        rows=_NO_ANTE_ROWS,
        stacks=_NO_ANTE_STACKS,
        settlement_kwargs={"small_blind": 5.0, "big_blind": 10.0},
        awards=("b",),
    )
    result = persist_reconciliation(db, hand.id)

    assert result.settlement.status == "reconciled"
    assert result.is_authoritative is True
    assert result.issues == ()
    assert not any("ante" in note.lower() for note in result.ledger.warnings)
    # The shape this test certifies must be the shape the migration genuinely
    # cannot reach: no ante rows AND no declared dead money. Asserted rather
    # than assumed, because the previous version of this test seeded a hand
    # with dead_money defaulted to 0 and read "no antes" as the whole
    # exemption, which certified a claim it did not exercise -- ruling 5
    # re-derives every stored hand carrying declared dead money, ante or no
    # ante. See test_v20_stales_the_analysis_beside_the_hands_ruling_5_recomputes.
    assert db.fetch_hand_settlement(hand.id).dead_money == 0

    # And declaring any mode over it moves nothing at all, which is the property
    # that says the migration cannot reach these hands in either direction.
    baseline = [pot.amount for pot in result.ledger.pots]
    for mode in ("NONE", "PER_PLAYER", "SINGLE_PAYER_TABLE_ANTE"):
        db.upsert_hand_settlement(
            db.fetch_hand_settlement(hand.id).model_copy(update={"ante_mode": mode})
        )
        again = persist_reconciliation(db, hand.id)
        assert again.settlement.status == "reconciled"
        assert [pot.amount for pot in again.ledger.pots] == pytest.approx(baseline)
        assert again.ledger.net_results == pytest.approx(result.ledger.net_results)


def test_v20_stales_the_analysis_beside_the_hands_ruling_5_recomputes(db):
    """THE OTHER HALF OF THE MIGRATION, and the one nobody had measured.

    Ruling 5 ships in the same release as the ante column and it moves stored
    chips on hands that contain NO ANTE AT ALL: external dead money used to drop
    WHOLE into the lowest layer and is now capped against the collecting seat's
    own total commitment.  The gross pot, the pot count and every eligible set
    are unchanged, so the operator's stored award rows still resolve by
    ``pot_index`` and every cross-check -- recorded gross, recorded net,
    ``is_balanced``, ``_validate_winners`` -- still passes.  Only the
    DISTRIBUTION moves, which is exactly the change no existing guard can see.

    Seeded here: hero all-in for a 3-chip small blind against two 120-chip
    seats, with 75 chips of declared dead money.  Schema 19 put all 75 in the
    main pot -- 84 / 234, hero paid 84 for a net of +81.  Ruling 5 caps it at
    hero's own 3-chip commitment, so the same recording is 12 / 306 and hero
    nets +9.  Gross is 318 either way, both pots keep their index, and both
    eligible sets are unchanged.

    The arithmetic is the operator's ruling and is not in question.  What this
    test pins is that the migration does not perform it in silence: the retained
    coaching that was written against the old figure stops being presented as
    current, so study readiness blocks on STALE_COACHING_EVIDENCE instead of
    republishing a hero result derived from a rule this build no longer applies.

    ``review_status`` is deliberately NOT demoted -- see ``_migrate_to_v20``.
    Staling retained analysis and discarding an operator's confirmation are two
    different acts, and the codebase keeps them in two methods for that reason.
    """

    from poker_tracker.persistence import db as db_module
    from poker_tracker.persistence.models import CoachingResponse

    rows = (
        ("a", "post_blind", 3.0, "small_blind", True),
        ("b", "post_blind", 10.0, "big_blind", True),
        ("c", "raise", 120.0, None, None),
        ("b", "call", 110.0, None, None),
    )
    hand = _seed(
        db,
        rows=rows,
        stacks=(("a", 3.0), ("b", 400.0), ("c", 400.0)),
        settlement_kwargs={
            "small_blind": 3.0,
            "big_blind": 10.0,
            "dead_money": 75.0,
        },
        awards=("a", "b"),
    )
    result = persist_reconciliation(db, hand.id)

    # Ruling 5's arithmetic, stated in chips so a revert is visible here too.
    # Schema 19 gave 84 / 234 with hero paid 84; both readings gross 318.
    assert [pot.amount for pot in result.ledger.pots] == pytest.approx([12.0, 306.0])
    assert result.ledger.gross_pot == pytest.approx(318.0)
    assert result.ledger.net_results["a"] == pytest.approx(9.0)
    # And it happens with nothing raised and nothing warned, which is why the
    # migration has to be the thing that speaks.
    assert result.ledger.warnings == ()
    assert result.is_authoritative is True

    db.create_coaching_response(
        CoachingResponse(
            hand_id=hand.id,
            provider_name="claude",
            model_name="m",
            review_type="hand",
            raw_prompt="p",
            raw_response="you won the 99-chip main pot",
        )
    )
    db.update_hand_status(hand.id, "reviewed")
    assert db.fetch_hand(hand.id).review_status == "reviewed"
    assert [r.is_stale for r in db.fetch_coaching_reviews_by_hand(hand.id)] == [False]

    db_module._migrate_to_v20(db)

    # The analysis derived under the old rule stops being presented as current.
    assert [r.is_stale for r in db.fetch_coaching_reviews_by_hand(hand.id)] == [True]
    # Nothing is destroyed: the coaching text, the awards, the actions, the
    # settlement AND the operator's own confirmation all survive. The last of
    # those is the line between this and _invalidate_hand_derivatives.
    assert db.fetch_hand(hand.id).review_status == "reviewed"
    assert db.fetch_hand_settlement(hand.id).dead_money == pytest.approx(75.0)
    assert len(db.fetch_actions_by_hand(hand.id)) == len(rows)
    assert db.fetch_coaching_reviews_by_hand(hand.id)[0].raw_response == (
        "you won the 99-chip main pot"
    )


def test_v20_leaves_a_hand_with_no_dead_money_and_no_antes_alone(db):
    """The over-strict predicate must still not reach the ordinary hand.

    ``dead_money > 0`` is deliberately coarser than "the declared amount exceeds
    the floor", because a schema migration cannot run the reducer.  What it must
    not be is coarser still: a hand that declares nothing keeps its review and
    its coaching.
    """

    from poker_tracker.persistence import db as db_module
    from poker_tracker.persistence.models import CoachingResponse

    hand = _seed(
        db,
        rows=_NO_ANTE_ROWS,
        stacks=_NO_ANTE_STACKS,
        settlement_kwargs={"small_blind": 5.0, "big_blind": 10.0},
        awards=("b",),
    )
    persist_reconciliation(db, hand.id)
    db.create_coaching_response(
        CoachingResponse(
            hand_id=hand.id,
            provider_name="claude",
            model_name="m",
            review_type="hand",
            raw_prompt="p",
            raw_response="r",
        )
    )
    db.update_hand_status(hand.id, "reviewed")

    db_module._migrate_to_v20(db)

    assert [r.is_stale for r in db.fetch_coaching_reviews_by_hand(hand.id)] == [False]
    assert db.fetch_hand(hand.id).review_status == "reviewed"


def test_v20_stales_retained_analysis_without_discarding_the_confirmation(db):
    """The line between staling and demotion, asserted as a pair.

    ``dead_money > 0`` is coarser than "the amount exceeds the floor" because a
    schema migration cannot run the reducer, so this population contains hands
    whose figures did not move at all.  Spending that imprecision on
    ``is_stale`` -- which already means "may have been derived from something
    that changed", and which the settlement writers set on every save without
    checking either -- costs a coaching rerun.  Spending it on ``review_status``
    would discard confirmations the operator gave, on hands where nothing
    happened, and ``_migrate_to_v13``'s fixture seeds a reviewed complete hand
    specifically to catch a migration that does that.
    """

    from poker_tracker.persistence import db as db_module
    from poker_tracker.persistence.models import CoachingResponse

    hand = _seed(
        db,
        rows=_NO_ANTE_ROWS,
        stacks=_NO_ANTE_STACKS,
        # Under the floor: this hand's figures do not move at all.
        settlement_kwargs={"small_blind": 5.0, "big_blind": 10.0, "dead_money": 0.5},
        awards=("b",),
    )
    before = [pot.amount for pot in persist_reconciliation(db, hand.id).ledger.pots]
    db.create_coaching_response(
        CoachingResponse(
            hand_id=hand.id,
            provider_name="claude",
            model_name="m",
            review_type="hand",
            raw_prompt="p",
            raw_response="r",
        )
    )
    db.update_hand_status(hand.id, "reviewed")

    db_module._migrate_to_v20(db)

    assert [r.is_stale for r in db.fetch_coaching_reviews_by_hand(hand.id)] == [True]
    assert db.fetch_hand(hand.id).review_status == "reviewed"
    assert [
        pot.amount for pot in reconcile_persisted_hand(db, hand.id).ledger.pots
    ] == pytest.approx(before)


def test_a_dead_blind_only_hand_is_not_asked_for_an_ante_declaration(db):
    """The mode names ANTES. A returning player's dead blind is not one.

    Blocking here would demand a declaration about a structure the hand does not
    have, and an operator asked to answer a question that does not apply to their
    hand learns to answer it at random -- which is how the declaration stops
    meaning anything.
    """

    rows = (
        ("b", "post_blind", 3.0, "dead_blind", False),
        ("a", "bet", 20.0, None, None),
        ("b", "call", 20.0, None, None),
        ("c", "call", 20.0, None, None),
    )
    hand = _seed(db, rows=rows, stacks=_NO_ANTE_STACKS, awards=("a",))
    result = persist_reconciliation(db, hand.id)

    assert result.settlement.status == "reconciled"
    assert result.issues == ()
    assert result.is_authoritative is True


# ---------------------------------------------------------------------------
# THE DECLARATION AS AN INPUT: round trip, dependence, staling
# ---------------------------------------------------------------------------


def test_the_declaration_survives_a_store_round_trip(db):
    hand = _seed(
        db,
        rows=_ANTE_ROWS,
        stacks=_ANTE_STACKS,
        settlement_kwargs={"ante_mode": "SINGLE_PAYER_TABLE_ANTE"},
    )
    stored = db.fetch_hand_settlement(hand.id)
    assert stored.ante_mode == "SINGLE_PAYER_TABLE_ANTE"

    db.upsert_hand_settlement(stored.model_copy(update={"ante_mode": "PER_PLAYER"}))
    assert db.fetch_hand_settlement(hand.id).ante_mode == "PER_PLAYER"

    db.upsert_hand_settlement(stored.model_copy(update={"ante_mode": None}))
    assert db.fetch_hand_settlement(hand.id).ante_mode is None


def test_a_hand_whose_pot_rests_on_the_declaration_measures_it_and_blocks(db):
    """The mode is a declared input, so it goes through the dependence rule.

    It is the one that goes furthest of the six: unlike the blind structure it
    moves CHIPS as well as the verdict. Worked example (f)'s hand reconciles to a
    5-chip main pot under ``SINGLE_PAYER_TABLE_ANTE`` and a 4-chip one under
    ``PER_PLAYER``, and withdrawing the declaration entirely puts back the
    refusal, so removing it changes the verdict AND the figures.
    """

    rows = (
        ("sb", "post_blind", 1.0, "small_blind", True),
        ("bb", "post_blind", 2.0, "big_blind", True),
        ("bb", "ante", 2.0, "big_blind_ante", False),
        ("btn", "call", 2.0, None, None),
    )
    stacks = (("sb", 1.0), ("bb", 4.0), ("btn", 2.0))
    hand = _seed(
        db,
        rows=rows,
        stacks=stacks,
        settlement_kwargs={
            "small_blind": 1.0,
            "big_blind": 2.0,
            "ante_mode": "SINGLE_PAYER_TABLE_ANTE",
        },
        awards=("sb", "bb"),
    )
    result = persist_reconciliation(db, hand.id)

    assert result.settlement.status == "reconciled"
    assert [pot.amount for pot in result.ledger.pots] == pytest.approx([5, 2])
    assert result.ledger.net_results["sb"] == pytest.approx(4)

    measured = reconcile_persisted_hand(db, hand.id)
    dependence = [
        item
        for item in measured.assumption_dependence
        if item.input_name == ANTE_MODE_INPUT
    ]
    assert dependence, "the declared ante mode must be measured as a dependence"
    assert "consolidated table ante" in dependence[0].declared
    assert "no declared ante mode" in dependence[0].neutral


def test_a_hand_with_no_antes_is_disclosed_nothing_about_the_mode(db):
    """A declaration that provably reaches no chip must stay silent.

    The same exemption a rake policy that provably takes nothing already gets. An
    operator trained to press "Confirm this assumption" on hands where the
    assumption moves nothing is an operator who will press it on the hand where it
    moves everything.
    """

    hand = _seed(
        db,
        rows=_NO_ANTE_ROWS,
        stacks=_NO_ANTE_STACKS,
        settlement_kwargs={
            "small_blind": 5.0,
            "big_blind": 10.0,
            "ante_mode": "SINGLE_PAYER_TABLE_ANTE",
        },
        awards=("b",),
    )
    persist_reconciliation(db, hand.id)
    measured = reconcile_persisted_hand(db, hand.id)

    assert not [
        item
        for item in measured.assumption_dependence
        if item.input_name == ANTE_MODE_INPUT
    ]


def test_editing_the_declaration_stales_what_was_derived_under_the_old_one(db):
    """The mode is in ``_declared_settlement_inputs``, and it has to be.

    Retained coaching, a saved review, a completed solver run: all of them rest on
    a hero result that the mode can move. Analysis produced under one reading is
    not evidence about the other, so changing the declaration invalidates it in
    exactly the way changing the rake policy does. Re-saving the SAME mode is
    still a no-op, so an idempotent save does not stale anything.
    """

    hand = _seed(
        db,
        rows=_ANTE_ROWS,
        stacks=_ANTE_STACKS,
        settlement_kwargs={"ante_mode": "PER_PLAYER"},
    )
    db._execute(
        "UPDATE hands SET review_status = ? WHERE id = ?", ("reviewed", hand.id)
    )
    db._commit()
    assert db.fetch_hand(hand.id).review_status == "reviewed"

    stored = db.fetch_hand_settlement(hand.id)
    db.upsert_hand_settlement(stored.model_copy(update={"updated_at": stored.updated_at}))
    assert db.fetch_hand(hand.id).review_status == "reviewed", (
        "re-saving the same declaration is not an evidence change"
    )

    db.upsert_hand_settlement(
        stored.model_copy(update={"ante_mode": "SINGLE_PAYER_TABLE_ANTE"})
    )
    assert db.fetch_hand(hand.id).review_status != "reviewed"


def test_an_unreadable_mode_column_degrades_to_undeclared_rather_than_to_a_guess(db):
    """A corrupt column becomes a REFUSAL, never a working declaration.

    ``hand_settlements`` carries no CHECK constraint -- the same threat model the
    dependence rule exists for -- so a hand-edited or imported row can hold
    anything. The reader drops a value it cannot validate, which on a hand with
    antes leaves the mode undeclared and the hand blocked. Keeping a
    half-readable value would let a mode nobody typed decide whether a
    consolidated ante is capped.
    """

    hand = _seed(
        db,
        rows=_ANTE_ROWS,
        stacks=_ANTE_STACKS,
        settlement_kwargs={"ante_mode": "PER_PLAYER"},
        awards=("c", "a"),
    )
    db._execute(
        "UPDATE hand_settlements SET ante_mode = ? WHERE hand_id = ?",
        ("BIG_BLIND_ANTE", hand.id),
    )
    db._commit()

    stored = db.fetch_hand_settlement(hand.id)
    assert stored.ante_mode is None
    assert any("ante_mode" in note for note in stored.warnings)

    result = reconcile_persisted_hand(db, hand.id)
    assert result.is_authoritative is False
    assert any("ante mode" in note for note in result.issues)
