"""Regressions for the round-6 adversarial findings against Phase 1.

Every test here failed before its fix. The round-6 themes are *a correctness gate
still priced in operator-typed fields*, *a check that switches itself off when one
optional cell is blank*, *a source fact re-declared with no audit trail*, *a
producer that disappears across an export/import round trip*, and *a first start
that could brick a brand-new database*.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

import app as app_module
from poker_tracker.persistence import backup as backup_module
from poker_tracker.persistence import db as db_module
from poker_tracker.persistence.backup import BACKUP_KEEP_COUNT, backup_database
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    UNREADABLE_CARDS_KEY,
    CompletionEvidence,
    dump_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import export_session, import_session
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
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import (
    BLOCKER_ORDER,
    evaluate_study_readiness,
)


def _clean_evidence(**overrides: object) -> dict[str, object]:
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


def _open_db(tmp_path: Path, name: str = "round6.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _seed_hand(
    db: PokerDatabase,
    *,
    pot_size: float | None = 20.0,
    hero_bb_won: float | None = 0.0,
    award: float | None = 20.0,
    board: str = "Qd 7s 2c",
    table_size: int | None = 6,
    evidence: dict[str, object] | None = None,
) -> Hand:
    """Two players, bet 10 / call 10.

    The derived gross pot is 20 before any rake. With the whole pot awarded to the
    hero the derived hero net result is 0 whatever rake policy is stored, because
    the hero contributed 10 and is pushed the net pot of 10 when the rake is 10.
    """
    session = db.create_session(Session(name="Round 6", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=table_size,
            hero_cards="Ah Qs",
            board_cards=board,
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence() if evidence is None else evidence,
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
                amount=10.0,
            )
        )
    if award is not None:
        db.replace_settlement_entries(
            hand.id,
            [
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=0,
                    player_key="hero",
                    player_name="Hero",
                    amount=award,
                    entry_order=1,
                )
            ],
        )
    return hand


# ---------------------------------------------------------------------------
# Finding 1 -- an interrupted FIRST start must not brick a brand-new database
# ---------------------------------------------------------------------------


def test_an_interrupted_first_start_leaves_a_database_the_next_start_can_open(
    tmp_path: Path,
) -> None:
    """The fresh-file path had no rollback point and no recovery.

    ``_create_base_tables`` ran outside the migration transaction and committed on
    its own. On a brand-new file that commit wrote the CURRENT schema, including
    ``hands.completion_status``, so any interruption after it -- power loss, an
    OOM kill, a container restart, Ctrl-C on the first launch -- left a file whose
    physical schema floor read 13 with no version stamp. Every later start refused
    it forever and told the operator to restore a backup that was never taken:
    ``_backup_before_migration`` returns early for a fresh file, correctly,
    because there is nothing yet to preserve. On a container with a persistent
    data mount that is a permanent startup loop.
    """

    path = tmp_path / "fresh.sqlite3"

    def boom(_db: PokerDatabase) -> None:
        raise RuntimeError("simulated crash inside migration 13")

    original = db_module._MIGRATIONS[13]
    db_module._MIGRATIONS[13] = boom
    try:
        first = PokerDatabase(str(path))
        with pytest.raises(RuntimeError, match="simulated crash"):
            first.init_db()
        first.close()
    finally:
        db_module._MIGRATIONS[13] = original

    # Nothing on disk may be ahead of its own stamp, which is the state
    # _assert_stamp_matches_schema refuses.
    raw = sqlite3.connect(path)
    try:
        floor = db_module._physical_schema_floor(raw)
        stamp = db_module._readable_schema_version(raw)
    finally:
        raw.close()
    assert stamp is not None
    assert floor <= stamp

    second = PokerDatabase(str(path))
    second.init_db()
    assert second.schema_version() == db_module.SCHEMA_VERSION
    session = second.create_session(Session(name="After", date_played=date(2026, 1, 1)))
    assert session.id is not None
    second.close()


# ---------------------------------------------------------------------------
# Findings 2, 4 and 10 -- the reconciliation slack is still operator-priced, and
# it still gates the hero result and declared awards
# ---------------------------------------------------------------------------


def test_the_chip_unit_cannot_buy_slack_on_the_observed_hero_result(
    tmp_path: Path,
) -> None:
    """``min(unit, rake taken, gross - rake) / 2`` peaks at a QUARTER of the pot.

    Rounds 4 and 5 made the pre-rake quantities exact and bounded the slack from
    ``gross / 2`` to ``gross / 4``; they left it gating the hero result. Three
    fields the operator types -- and an import payload supplies verbatim -- still
    maximise it: ``rake_rate`` up to 1.0, a ``rake_cap`` of half the pot, and a
    ``rake_rounding_unit`` to match. A recorded hero result a quarter of the pot
    away from the hand's own action line reconciled with zero blockers.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=5.0, award=10.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.5, rake_rounding_unit=10.0)
    )
    result = persist_reconciliation(db, hand.id)

    assert result.ledger.net_results["hero"] == pytest.approx(0.0)
    assert "Observed Hero result does not match the derived ledger result." in result.issues
    assert result.is_authoritative is False
    assert result.settlement is not None
    assert result.settlement.status == "needs_correction"

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=result, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE")


def test_the_chip_unit_cannot_buy_slack_on_a_declared_award(tmp_path: Path) -> None:
    """A declared award says how many chips a seat was pushed. That is an
    observation of the hand, not a restatement of the rake policy."""
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=0.0, award=15.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.5, rake_rounding_unit=10.0)
    )
    result = persist_reconciliation(db, hand.id)

    assert result.ledger.payouts["hero"] == pytest.approx(10.0)
    assert "Declared awards for 'hero' do not match derived payouts." in result.issues
    assert result.is_authoritative is False


def test_an_imported_settlement_cannot_buy_slack_on_the_hero_result(
    tmp_path: Path,
) -> None:
    """One import_session call used to land a hand that presents as reconciled."""
    source = _open_db(tmp_path, "slack-src.db")
    hand = _seed_hand(source, pot_size=20.0, hero_bb_won=5.0, award=10.0)
    assert hand.id is not None
    source.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            status="reconciled",
            rake_rate=0.5,
            rake_rounding_unit=10.0,
        )
    )
    payload = export_session(source, hand.session_id)
    source.close()

    target = _open_db(tmp_path, "slack-tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    result = reconcile_persisted_hand(target, imported.id)

    assert result.settlement is not None
    assert result.settlement.rake_rounding_unit == pytest.approx(10.0)
    assert "Observed Hero result does not match the derived ledger result." in result.issues
    assert result.is_authoritative is False
    readiness = evaluate_study_readiness(
        imported, accounting=result, user_confirmed=True
    )
    assert readiness.is_ready is False
    target.close()


def test_an_honest_whole_chip_rake_still_absorbs_its_own_rounding(
    tmp_path: Path,
) -> None:
    """An honest rake policy still reconciles, and no recorded figure is excused.

    Rounds 4-6 kept a tolerance for the RECORDED rake on the argument that it is
    the same policy read one rounding step earlier, and round 6 argued that
    narrowing it to nothing would be a silent revert. Round 7 narrowed it to
    nothing deliberately, and this is the reasoning that replaces round 6's:
    every field the tolerance was derived from also arrives in an import payload,
    and `import_session` -- unlike the settlement editor -- never rewrites the
    recorded pair, so the payload set both sides of its own comparison and landed
    a recorded rake 24.5% of the pot away from the hand's own action line as
    study-ready. No honest producer writes a disagreeing row: the editor nulls
    the recorded figures and `persist_reconciliation` writes them from the
    ledger. So the original intent -- rake rounding must not itself become a
    mismatch -- is now verified through that path, below, and a stored policy
    that disagrees with the stored amount beside it fails closed onto the same
    one-click clearing action.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=9.0, award=19.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            rake_rate=0.05,
            rake_rounding_unit=1.0,
            rake_amount=1.3,
        )
    )
    # The read-only path every readiness surface uses.
    result = reconcile_persisted_hand(db, hand.id)
    assert result.ledger.rake == pytest.approx(1.0)
    assert "Recorded rake does not match the derived ledger." in result.issues
    assert result.is_authoritative is False

    # And the product refuses to call it reconciled...
    result = persist_reconciliation(db, hand.id)
    assert result.settlement is not None
    assert result.settlement.status == "needs_correction"

    # ...while the honest policy itself still reconciles once the recorded
    # figure is the product's own, re-derived from the ledger.
    result = persist_reconciliation(db, hand.id)
    assert result.ledger.rake == pytest.approx(1.0)
    assert result.settlement is not None
    assert result.settlement.rake_amount == pytest.approx(1.0)
    assert result.settlement.status == "reconciled"
    assert result.is_authoritative is True


@pytest.mark.parametrize(
    ("quantity", "expected_issue"),
    [
        ("gross_pot", "Recorded gross pot does not match the derived ledger."),
        ("observed_pot", "Observed final pot does not match the derived gross pot."),
        ("refund", "Declared refund for 'hero' does not match the derived refund."),
    ],
)
def test_pre_rake_quantities_are_compared_exactly(
    tmp_path: Path, quantity: str, expected_issue: str
) -> None:
    """PLAN.md names these three comparisons as the whole round-4 fix, and no test
    failed when any of them was handed the slack instead.

    The round-5 test that was meant to pin them is parametrized over rate/unit
    pairs that all make ``min(unit, rake, gross - rake)`` exactly zero, so it
    passes whether the pot check is exact or slack-gated. Here the stored policy
    takes a real rake of 10 on a 20-chip pot with a chip unit of 10, so the slack
    is 5 and every discrepancy below is 2 -- comfortably inside it.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(
        db,
        pot_size=22.0 if quantity == "observed_pot" else 20.0,
        hero_bb_won=0.0,
        award=10.0,
    )
    assert hand.id is not None
    if quantity == "refund":
        db.replace_settlement_entries(
            hand.id,
            [
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="award",
                    pot_index=0,
                    player_key="hero",
                    player_name="Hero",
                    amount=10.0,
                    entry_order=1,
                ),
                SettlementEntry(
                    hand_id=hand.id,
                    entry_type="refund",
                    player_key="hero",
                    player_name="Hero",
                    amount=2.0,
                    entry_order=2,
                ),
            ],
        )
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id,
            rake_rate=0.5,
            rake_rounding_unit=10.0,
            gross_pot=22.0 if quantity == "gross_pot" else None,
        )
    )
    result = reconcile_persisted_hand(db, hand.id)

    assert result.ledger.gross_pot == pytest.approx(20.0)
    assert result.ledger.rake == pytest.approx(10.0)
    assert expected_issue in result.issues
    assert result.is_authoritative is False


# ---------------------------------------------------------------------------
# Finding 5 -- one blank 'Observed payout' disabled every other award check
# ---------------------------------------------------------------------------


def _seed_side_pot_hand(db: PokerDatabase, *, pot0: float | None, pot1: float | None) -> Hand:
    session = db.create_session(Session(name="Side pot", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=250.0,
            hero_bb_won=150.0,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(),
        )
    )
    assert hand.id is not None
    for key, name, hero, stack in (
        ("hero", "Hero", True, 1000.0),
        ("short", "Short", False, 50.0),
        ("big", "Big", False, 1000.0),
    ):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                is_hero=hero,
                starting_stack=stack,
            )
        )
    for index, (key, name, kind, amount) in enumerate(
        (
            ("hero", "Hero", "bet", 100.0),
            ("short", "Short", "all_in", 50.0),
            ("big", "Big", "call", 100.0),
        ),
        start=1,
    ):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=name,
                action_type=kind,
                amount=amount,
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
                amount=pot0,
                entry_order=1,
            ),
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=1,
                player_key="hero",
                player_name="Hero",
                amount=pot1,
                entry_order=2,
            ),
        ],
    )
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.0, rake_rounding_unit=0.01)
    )
    return hand


def test_one_blank_observed_payout_does_not_disable_the_other_award_checks(
    tmp_path: Path,
) -> None:
    """'Observed payout' is an optional column, so a blank cell is ordinary input.

    The guard was ``if awards and all(entry.amount is not None ...)``: one blank
    amount anywhere in the award set skipped the whole per-identity comparison,
    so a declared payout of 9999 against a derived 250 reconciled and rendered
    study-ready. Not knowing pot 1's payout is not evidence about pot 0's.
    """
    db = _open_db(tmp_path)
    hand = _seed_side_pot_hand(db, pot0=9999.0, pot1=None)
    assert hand.id is not None
    result = persist_reconciliation(db, hand.id)

    assert result.ledger.payouts["hero"] == pytest.approx(250.0)
    assert "Declared awards for 'hero' do not match derived payouts." in result.issues
    assert result.is_authoritative is False
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert evaluate_study_readiness(
        stored, accounting=result, user_confirmed=True
    ).has("ACCOUNTING_NOT_AUTHORITATIVE")


def test_an_award_set_may_still_leave_an_observed_payout_blank(tmp_path: Path) -> None:
    """The column stays optional: declaring less than the derived payout, or
    nothing at all, is 'not known yet' and must not be reported as a mismatch."""
    db = _open_db(tmp_path)
    hand = _seed_side_pot_hand(db, pot0=150.0, pot1=None)
    assert hand.id is not None
    result = persist_reconciliation(db, hand.id)

    assert result.issues == ()
    assert result.is_authoritative is True


# ---------------------------------------------------------------------------
# Finding 6 -- re-declaring the pot winner is a source-fact correction
# ---------------------------------------------------------------------------


def test_re_declaring_the_pot_winner_is_recorded_as_an_auditable_correction(
    tmp_path: Path,
) -> None:
    """The declared winner is the sole input the derived payouts, and therefore
    the hero-result cross-check, are computed from.

    Flipping it in the Accounting reconciliation panel cleared
    ACCOUNTING_NOT_AUTHORITATIVE while leaving no HandCorrection, no
    completion-evidence disclosure and completion_status still 'complete' -- so
    the recorded hero result was cross-checked against a freely editable,
    unaudited, undisclosed declaration. Correcting one board card left a
    permanent auditable record; re-assigning who won the pot left none.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db, pot_size=20.0, hero_bb_won=10.0, award=20.0)
    assert hand.id is not None
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="villain",
                player_name="Villain",
                amount=20.0,
                entry_order=1,
            )
        ],
    )
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.0, rake_rounding_unit=0.01)
    )
    before = persist_reconciliation(db, hand.id)
    assert before.is_authoritative is False

    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="hero",
                player_name="Hero",
                amount=20.0,
                entry_order=1,
            )
        ],
    )
    after = persist_reconciliation(db, hand.id)

    corrections = db.fetch_hand_corrections(hand.id)
    assert [item.correction_type for item in corrections].count(
        "settlement_award_update"
    ) >= 1
    latest = next(
        item for item in corrections if item.correction_type == "settlement_award_update"
    )
    assert "villain" in str(latest.before_state)
    assert "hero" in str(latest.after_state)

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.completion_status == "uncertain"
    assert "source_facts_corrected" in stored.completion_evidence["warning_codes"]
    readiness = evaluate_study_readiness(stored, accounting=after, user_confirmed=True)
    assert readiness.is_ready is False
    assert readiness.has("UNRESOLVED_SOURCE_WARNING")


def test_deriving_refunds_during_reconciliation_is_not_an_operator_correction(
    tmp_path: Path,
) -> None:
    """``persist_reconciliation`` writes the ledger's own refunds back through
    ``replace_settlement_entries``. Recording that as a re-declaration would demote
    every hand the reconciler touched and make the disclosure meaningless."""
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Refund", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=20.0,
            hero_bb_won=10.0,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(),
        )
    )
    assert hand.id is not None
    for key, name, hero, stack in (
        ("hero", "Hero", True, 1000.0),
        ("villain", "Villain", False, 10.0),
    ):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=name,
                is_hero=hero,
                starting_stack=stack,
            )
        )
    for index, (key, name, kind, amount) in enumerate(
        (("hero", "Hero", "bet", 30.0), ("villain", "Villain", "all_in", 10.0)), start=1
    ):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=name,
                action_type=kind,
                amount=amount,
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
                amount=20.0,
                entry_order=1,
            )
        ],
    )
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, rake_rate=0.0, rake_rounding_unit=0.01)
    )
    result = persist_reconciliation(db, hand.id)

    assert result.ledger.refunds["hero"] == pytest.approx(20.0)
    assert db.fetch_hand_corrections(hand.id) == []
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.completion_status == "complete"


# ---------------------------------------------------------------------------
# Finding 7 -- an export/import round trip erased INVALID_HERO_OR_BOARD_CARDS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "corrupt_board",
    [
        "Qd Qd 7s",  # a card repeated inside the board
        "Qs 7s 2c",  # a board card the hero also holds -- the annotated branch
    ],
)
def test_an_unreadable_card_column_still_blocks_after_a_round_trip(
    tmp_path: Path, corrupt_board: str
) -> None:
    """The blocker's producer is the corrupt column, which the exporter blanks;
    the surviving record of it is the marker, which the importer strips.

    Between them a board that "could not be read" silently became a hand with "no
    board recorded" -- a legitimate, unblocked state for a preflop hand -- and the
    text that proved the corruption was gone from the database entirely. The
    marker must stay derived-only (round-5 finding 3), so the round trip carries
    the offending VALUE instead.
    """
    source = _open_db(tmp_path, "cards-src.db")
    hand = _seed_hand(source, pot_size=20.0, hero_bb_won=0.0, award=10.0)
    assert hand.id is not None
    source._execute(
        "UPDATE hands SET board_cards = ? WHERE id = ?", (corrupt_board, hand.id)
    )
    source._commit()
    degraded = source.fetch_hand(hand.id)
    assert degraded is not None
    assert evaluate_study_readiness(
        degraded, accounting=None, user_confirmed=True
    ).has("INVALID_HERO_OR_BOARD_CARDS")
    payload = export_session(source, hand.session_id)
    source.close()

    target = _open_db(tmp_path, "cards-tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    assert evaluate_study_readiness(
        imported, accounting=None, user_confirmed=True
    ).has("INVALID_HERO_OR_BOARD_CARDS")

    # A second round trip must be stable: the carried value re-derives to itself
    # rather than accumulating annotations or decaying into "no board recorded".
    again = _open_db(tmp_path, "cards-tgt2.db")
    twice = again.fetch_hands_by_session(
        import_session(again, export_session(target, session.id)).id
    )[0]
    assert twice.completion_evidence[UNREADABLE_CARDS_KEY] == (
        imported.completion_evidence[UNREADABLE_CARDS_KEY]
    )
    assert evaluate_study_readiness(twice, accounting=None, user_confirmed=True).has(
        "INVALID_HERO_OR_BOARD_CARDS"
    )
    again.close()

    # Still derived, never stored: the stored evidence carries no marker, so the
    # blocker clears the moment the column is corrected.
    stored_row = target._execute(
        "SELECT completion_evidence FROM hands WHERE id = ?", (imported.id,)
    ).fetchone()
    assert UNREADABLE_CARDS_KEY not in stored_row["completion_evidence"]
    target.update_hand_facts(imported.model_copy(update={"board_cards": "Qd 7s 2c"}))
    fixed = target.fetch_hand(imported.id)
    assert fixed is not None
    assert fixed.board_cards == "Qd 7s 2c"
    assert not evaluate_study_readiness(
        fixed, accounting=None, user_confirmed=True
    ).has("INVALID_HERO_OR_BOARD_CARDS")
    target.close()


# ---------------------------------------------------------------------------
# Finding 8 -- a rejection is not an acknowledgeable source warning
# ---------------------------------------------------------------------------


def test_a_rejection_is_not_reported_as_an_acknowledgeable_source_warning(
    tmp_path: Path,
) -> None:
    """``unresolved_codes`` mixes warnings and rejections, and every rejection code
    is permanently unresolved because ``acknowledge_codes`` refuses one.

    The blocker called a pipeline refusal a "source warning" and told the operator
    to acknowledge it in a panel that draws no Acknowledge button for a rejection
    -- directly contradicting COMPLETION_NOT_COMPLETE on the same page.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(
        db,
        evidence=_clean_evidence(rejection_codes=["board_unreadable"]),
    )
    assert hand.id is not None
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(stored, accounting=None, user_confirmed=True)
    blocker = next(
        item for item in readiness.blockers if item.code == "UNRESOLVED_SOURCE_WARNING"
    )

    assert "board_unreadable" in blocker.detail
    assert "acknowledge the remaining codes" not in blocker.clearing_action
    assert "Run CV reconstruction" in blocker.clearing_action
    assert "rejection cannot be acknowledged" in blocker.clearing_action


def test_a_warning_beside_a_rejection_keeps_its_own_clearing_action(
    tmp_path: Path,
) -> None:
    """Mixed evidence must describe both halves truthfully, not collapse to one."""
    db = _open_db(tmp_path)
    hand = _seed_hand(
        db,
        evidence=_clean_evidence(
            rejection_codes=["board_unreadable"], warning_codes=["low_confidence_seat"]
        ),
    )
    assert hand.id is not None
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    blocker = next(
        item
        for item in evaluate_study_readiness(
            stored, accounting=None, user_confirmed=True
        ).blockers
        if item.code == "UNRESOLVED_SOURCE_WARNING"
    )

    assert "Run CV reconstruction" in blocker.clearing_action
    assert "low_confidence_seat" in blocker.clearing_action
    assert "acknowledge" in blocker.clearing_action.lower()


# ---------------------------------------------------------------------------
# Finding 9 -- a ledger error names a panel that cannot perform the fix
# ---------------------------------------------------------------------------


def test_a_ledger_error_names_a_control_that_can_actually_clear_it(
    tmp_path: Path,
) -> None:
    """When the ledger REFUSES to build, nothing in the Accounting reconciliation
    panel -- dead money, rake %, rake cap, chip unit, awards, refunds -- can fix
    it, so naming that panel is an action the product cannot perform."""
    db = _open_db(tmp_path)
    hand = _seed_hand(db)
    assert hand.id is not None
    db._execute(
        "UPDATE hand_players SET starting_stack = 4 WHERE hand_id = ? AND player_key = 'hero'",
        (hand.id,),
    )
    db._commit()
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    readiness = evaluate_study_readiness(
        stored,
        accounting=None,
        accounting_error="Player 'hero' commits 10 with only 4 remaining.",
        user_confirmed=True,
    )
    blocker = next(
        item
        for item in readiness.blockers
        if item.code == "ACCOUNTING_NOT_AUTHORITATIVE"
    )

    assert "stack sizes" in blocker.clearing_action
    assert "cannot change them" in blocker.clearing_action


# ---------------------------------------------------------------------------
# Finding 11 -- an unrecorded table size is the sole trigger of its blocker
# ---------------------------------------------------------------------------


def test_a_reconstructed_hand_with_no_recorded_table_size_is_not_study_ready(
    tmp_path: Path,
) -> None:
    """Reachable from the UI: 'Correct hand facts' offers Table size with a blank
    'Unknown' placeholder and writes NULL. Seat, position and hero attribution
    cannot be checked against a table size that is not there."""
    db = _open_db(tmp_path)
    hand = _seed_hand(db, table_size=None)
    assert hand.id is not None
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.table_size is None

    readiness = evaluate_study_readiness(stored, accounting=None, user_confirmed=True)
    blocker = next(
        item for item in readiness.blockers if item.code == "UNSUPPORTED_TABLE_LAYOUT"
    )

    assert blocker.detail == ("hand.table_size is not recorded",)
    assert readiness.is_ready is False


# ---------------------------------------------------------------------------
# Finding 12 -- the composer behind every readiness surface had no test
# ---------------------------------------------------------------------------


def test_hand_study_readiness_feeds_every_store_backed_blocker_to_the_service(
    tmp_path: Path,
) -> None:
    """``app.hand_study_readiness`` is the only place the store's issue, coaching,
    legacy hand_review and solver rows are fetched and handed to
    ``evaluate_study_readiness``.

    It feeds the Study page, the Hands workspace, the coaching surface and the
    session 'Not study-ready' KPI, and every other test exercises the service with
    hand-built arguments, so the store-to-service wiring was never checked. With
    it broken a hand carrying an open debugging issue rendered 'Study-ready' on
    every surface.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(db)
    assert hand.id is not None
    db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["actions"],
            description="Action amounts look wrong.",
        )
    )
    db.create_coaching_response(
        CoachingResponse(
            hand_id=hand.id,
            session_id=hand.session_id,
            review_type="hand",
            provider_name="test",
            model_name="test",
            raw_prompt="p",
            raw_response="r",
            is_stale=True,
            stale_reason="Hand evidence changed; rerun coaching.",
        )
    )
    db.create_solver_run(
        SolverRun(hand_id=hand.id, status="stale", input_hash="abc123")
    )
    stored = db.fetch_hand(hand.id)
    assert stored is not None

    readiness = app_module.hand_study_readiness(
        db, stored, None, None, user_confirmed=True
    )

    assert readiness.has("OPEN_DEBUGGING_ISSUE")
    assert readiness.has("STALE_COACHING_EVIDENCE")
    assert readiness.has("STALE_SOLVER_EVIDENCE")
    assert readiness.is_ready is False


# ---------------------------------------------------------------------------
# Finding 13 -- the documented blocker emission order
# ---------------------------------------------------------------------------


def test_blockers_are_emitted_in_the_documented_order(tmp_path: Path) -> None:
    """PLAN.md publishes the order as a contract and ``codes()`` exposes it.

    This pins the observable order, which today is produced by the sequence of
    ``blockers.extend(...)`` calls; the ``BLOCKER_ORDER`` sort behind it is
    defence in depth for a future reordering of those calls.
    """
    db = _open_db(tmp_path)
    hand = _seed_hand(
        db,
        pot_size=999.0,
        table_size=None,
        evidence=_clean_evidence(warning_codes=["low_confidence_seat"]),
    )
    assert hand.id is not None
    db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["actions"],
            description="Action amounts look wrong.",
        )
    )
    db.create_solver_run(
        SolverRun(hand_id=hand.id, status="stale", input_hash="abc123")
    )
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    codes = app_module.hand_study_readiness(db, stored, None, None).codes()

    assert len(codes) >= 6
    assert list(codes) == sorted(codes, key=BLOCKER_ORDER.index)


# ---------------------------------------------------------------------------
# Finding 14 -- the multi-hero cross-check
# ---------------------------------------------------------------------------


def test_more_than_one_hero_seat_is_reported(tmp_path: Path) -> None:
    """``_validate_single_hero`` refuses a second hero on write, so this is
    defence in depth for a hand-edited database -- and it was the only untested
    branch of a chain whose other two branches were hardened this round."""
    db = _open_db(tmp_path)
    hand = _seed_hand(db)
    assert hand.id is not None
    db._execute(
        "UPDATE hand_players SET is_hero = 1 WHERE hand_id = ?", (hand.id,)
    )
    db._commit()
    result = reconcile_persisted_hand(db, hand.id)

    assert "More than one player is marked as Hero." in result.issues
    assert result.is_authoritative is False


# ---------------------------------------------------------------------------
# Finding 3 -- rotation must never delete a snapshot the product did not write
# ---------------------------------------------------------------------------


def test_rotation_never_deletes_a_backup_the_product_did_not_write(
    tmp_path: Path,
) -> None:
    """``poker_tracker_manual_keepme.sqlite3`` -- an operator's own snapshot, in
    the operator's own backup directory -- matched the rotating glob and was
    evicted alongside the product's own. PLAN.md's promise is that nothing here
    deletes a rollback point it did not write."""
    source = tmp_path / "poker_tracker.db"
    sqlite3.connect(source).close()
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    operator_files = (
        "poker_tracker_manual_keepme.sqlite3",
        "poker_tracker_KEEP.sqlite3",
        "poker_tracker-v7-20260724.db",
        f"{backup_module.PINNED_PREFIX}keepme.sqlite3",
    )
    for name in operator_files:
        (backup_dir / name).write_bytes(b"operator snapshot")

    for _ in range(BACKUP_KEEP_COUNT + 2):
        backup_database(source, backup_dir)
    for _ in range(backup_module.PINNED_KEEP_COUNT + 2):
        backup_database(source, backup_dir, pinned=True)

    survivors = {path.name for path in backup_dir.iterdir()}
    assert set(operator_files) <= survivors
    # ...and the product's own snapshots are still rotated.
    assert (
        len([name for name in survivors if backup_module.ROUTINE.name.match(name)])
        == BACKUP_KEEP_COUNT
    )
    assert (
        len([name for name in survivors if backup_module.PREMIGRATION.name.match(name)])
        == backup_module.PINNED_KEEP_COUNT
    )


def test_a_backup_written_by_the_product_still_rotates(tmp_path: Path) -> None:
    """The strict name match must not turn rotation off."""
    source = tmp_path / "poker_tracker.db"
    sqlite3.connect(source).close()
    backup_dir = tmp_path / "backups"
    written = [
        backup_database(source, backup_dir) for _ in range(BACKUP_KEEP_COUNT + 3)
    ]

    assert not written[0].exists()
    assert written[-1].exists()
    assert len(list(backup_dir.glob("poker_tracker_*.sqlite3"))) == BACKUP_KEEP_COUNT
