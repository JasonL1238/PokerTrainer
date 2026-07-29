"""Round-12 regressions.

Each test below fails on the pre-round-12 tree and states what it did there.
They are written as families rather than as the shapes the round demonstrated:
round 12's whole finding about rounds 7-9 was that a repair scoped to the
demonstrated shape buys one round.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from poker_tracker.maintenance.data_health import audit_data_health
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    CompletionEvidence,
    acknowledge_codes,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import (
    DECLARED_DEAD_MONEY_CODE,
    DECLARED_RAKE_CODE,
    PokerDatabase,
)
from poker_tracker.persistence.import_export import (
    export_session,
    import_hands_into_session,
    import_session,
)
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services import study_readiness
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import evaluate_study_readiness


def _open_db(tmp_path: Path, name: str = "round12.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _clean_evidence(**overrides: object) -> dict[str, object]:
    payload = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="fold_win",
            boundary_confidence=0.95,
            layout_supported=True,
            table_size=6,
        )
    )
    payload.update(overrides)
    return payload


def _readiness(db: PokerDatabase, hand_id: int) -> tuple[str, ...]:
    hand = db.fetch_hand(hand_id)
    assert hand is not None
    try:
        accounting = reconcile_persisted_hand(db, hand_id)
        error = None
    except Exception as exc:  # noqa: BLE001 - mirrors the UI's own fallback
        accounting, error = None, str(exc)
    result = evaluate_study_readiness(
        hand, accounting=accounting, accounting_error=error, user_confirmed=True
    )
    return tuple(blocker.code for blocker in result.blockers)


# ---------------------------------------------------------------------------
# 1. A correction whose correct value equals the DEGRADED view of the column
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "source_type", "hero_cards", "board_cards", "column", "corrupt", "correct"),
    [
        # The demonstrated shape: a board that "could not be read" on a hand that
        # ended preflop, whose correct board is 0 cards.
        ("preflop board", "cv_import", "Ah Qs", "", "board_cards", "Qd 7s", ""),
        # The same defect on the other card column, on a manual hand, where empty
        # hero cards are a legal state.
        ("manual hero cards", "manual", "", "Qd 7s 2c", "hero_cards", "Ah Ah Kd", ""),
        # A card visible in two places at once: the reader blanks the BOARD and
        # records the collision, and the hand's correct board is again empty.
        ("card seen twice", "cv_import", "Ah Qs", "", "board_cards", "Ah 7s 2c", ""),
        # Readable but not normalized. No blocker here, so nothing hid it -- but
        # it is the same blind spot, and it proves the repair is about the
        # comparison and not about unreadable cards.
        ("unnormalized board", "cv_import", "Ah Qs", "", "board_cards", "qd 7s 2c", "Qd 7s 2c"),
    ],
)
def test_correcting_a_fact_to_the_value_its_column_only_appears_to_hold_is_written(
    tmp_path: Path,
    case: str,
    source_type: str,
    hero_cards: str,
    board_cards: str,
    column: str,
    corrupt: str,
    correct: str,
) -> None:
    """`update_hand_facts` decided "nothing changed" against a DEGRADED view.

    ``_hand_from_row`` blanks a card column it cannot read and normalizes one it
    can, so the model the form is filled from is a projection of the row rather
    than the row. Comparing the submitted hand against that projection made the
    writer skip exactly the corrections that had to be written: the operator's
    answer equalled the projection, so ``before_state == after_state`` held, the
    UPDATE never fired, no ``hand_corrections`` row was recorded -- and the UI
    reported "Corrected facts saved".

    BEFORE this change, on all four rows: the raw column still held ``corrupt``
    afterwards and zero corrections were recorded. On the first three that left
    INVALID_HERO_OR_BOARD_CARDS permanent, with no reachable action anywhere in
    the product that clears it, because nothing else writes a card column.
    """
    db = _open_db(tmp_path, f"{column}.db")
    session = db.create_session(Session(name=case, date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards=hero_cards,
            board_cards=board_cards,
            source_type=source_type,  # type: ignore[arg-type]
            completion_status="not_applicable" if source_type == "manual" else "complete",
            completion_evidence={} if source_type == "manual" else _clean_evidence(),
        )
    )
    assert hand.id is not None
    db.restore_unreadable_card_columns(hand.id, {column: corrupt})

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    # The form is pre-filled from the model, which is why the operator cannot see
    # the corrupt text: submitting the value they are shown is the whole scenario.
    assert getattr(stored, column) == correct

    db.update_hand_facts(
        stored.model_copy(update={column: correct}),
        correction_notes=f"{case}: the column's true value.",
    )

    raw = sqlite3.connect(db.db_path).execute(
        f"SELECT {column} FROM hands WHERE id = ?", (hand.id,)  # noqa: S608 - literal
    ).fetchone()
    assert raw[0] == correct, f"{case}: the stored column still holds {raw[0]!r}"
    corrections = db.fetch_hand_corrections(hand.id)
    assert len(corrections) == 1, f"{case}: the correction was not recorded"
    assert corrections[0].before_state[column] == corrupt
    assert corrections[0].after_state[column] == correct
    assert "INVALID_HERO_OR_BOARD_CARDS" not in _readiness(db, hand.id)
    db.close()


# ---------------------------------------------------------------------------
# 2. An operator's own settlement declaration in the PIPELINE's channel
# ---------------------------------------------------------------------------


def _seed_reconstructed(
    db: PokerDatabase,
    *,
    hero_bb_won: float | None = None,
    pot_size: float | None = None,
) -> Hand:
    session = db.create_session(Session(name="Declared", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(terminal_event="showdown"),
        )
    )
    assert hand.id is not None
    for key in ("hero", "villain"):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=key.capitalize(),
                is_hero=key == "hero",
                starting_stack=1000,
            )
        )
    for index, key in enumerate(("hero", "villain"), start=1):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=key.capitalize(),
                action_type="bet" if index == 1 else "call",
                amount=40.0,
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
                entry_order=1,
            )
        ],
    )
    return hand


@pytest.mark.parametrize(
    ("case", "settlement", "code"),
    [
        ("rake", {"rake_rate": 0.5}, DECLARED_RAKE_CODE),
        ("dead money", {"dead_money": 25.0}, DECLARED_DEAD_MONEY_CODE),
        (
            "both",
            {"rake_rate": 0.25, "dead_money": 10.0},
            DECLARED_RAKE_CODE,
        ),
    ],
)
def test_an_operator_declaration_is_never_reported_as_a_pipeline_finding(
    tmp_path: Path, case: str, settlement: dict[str, float], code: str
) -> None:
    """The disclosure half of the channel separation round 10 made for the attestation.

    ``_record_declared_chip_adjustment`` wrote the operator's own settlement
    declaration into ``CompletionEvidence.warning_codes`` -- the PIPELINE's
    channel. ``derive_completion_status`` demotes on an unresolved entry there,
    so declaring a rake on a hand whose reconstruction evidence was complete and
    clean turned that hand ``uncertain`` and produced two blockers reading "The
    pipeline could not prove this hand was fully reconstructed" and "The pipeline
    flagged 1 unresolved source warning(s)" -- about a figure the pipeline never
    claimed and never observed -- while naming Correct hand facts, a form with no
    rake field in it, as the place to fix a value that exists only in the
    Accounting reconciliation panel.

    BEFORE this change, on every row: ``completion_status`` went complete ->
    uncertain, and the blocker tuple gained COMPLETION_NOT_COMPLETE and
    UNRESOLVED_SOURCE_WARNING.
    """
    db = _open_db(tmp_path, f"{case}.db")
    hand = _seed_reconstructed(db)
    assert hand.id is not None
    before = db.fetch_hand(hand.id)
    assert before is not None
    assert before.completion_status == "complete"

    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, **settlement))
    persist_reconciliation(db, hand.id)

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    assert code in evidence.declared_settlement_codes
    assert code not in evidence.warning_codes
    assert code not in evidence.acknowledged_codes
    assert code not in evidence.rejection_codes
    assert evidence.unresolved_codes == ()
    assert stored.completion_status == "complete"

    blockers = _readiness(db, hand.id)
    assert "COMPLETION_NOT_COMPLETE" not in blockers
    assert "UNRESOLVED_SOURCE_WARNING" not in blockers
    # The declaration is still disclosed, by the measurement that actually
    # weighs it.
    assert "ACCOUNTING_ASSUMPTION_DEPENDENT" in blockers
    db.close()


@pytest.mark.parametrize("channel", ["warning_codes", "acknowledged_codes", "rejection_codes"])
def test_no_writer_or_payload_can_file_a_declaration_in_the_pipeline_channel(
    channel: str,
) -> None:
    """Enforced on every read, like the attestation's separation, not per writer.

    A hand-edited row, a legacy database written before this channel existed, and
    an import payload all reach the same parser, so the channel a code ends up in
    is decided there rather than by whoever wrote it. A misfiled declaration is
    relocated rather than dropped: it is an audit record with nothing resting on
    it, and every database written before round 12 holds its declarations in
    ``warning_codes``.
    """
    evidence = parse_completion_evidence(
        {
            "evidence_version": 1,
            "partial_start": False,
            "partial_end": False,
            "terminal_event": "showdown",
            "boundary_confidence": 0.9,
            channel: [DECLARED_RAKE_CODE, "hero_seat_mismatch"],
        }
    )
    assert DECLARED_RAKE_CODE not in evidence.warning_codes
    assert DECLARED_RAKE_CODE not in evidence.acknowledged_codes
    assert DECLARED_RAKE_CODE not in evidence.rejection_codes
    assert DECLARED_RAKE_CODE not in evidence.unresolved_codes
    assert evidence.declared_settlement_codes == (DECLARED_RAKE_CODE,)
    # The genuine pipeline code in the same list is untouched.
    assert "hero_seat_mismatch" in getattr(evidence, channel)
    # And it is not acknowledgeable, so no generic control can answer it.
    assert (
        DECLARED_RAKE_CODE
        not in acknowledge_codes(evidence, [DECLARED_RAKE_CODE]).acknowledged_codes
    )


def test_a_declaration_survives_an_export_import_round_trip_in_its_own_channel(
    tmp_path: Path,
) -> None:
    """The round trip is how round 10's attestation defect became reachable."""
    source = _open_db(tmp_path, "src.db")
    hand = _seed_reconstructed(source)
    assert hand.id is not None
    source.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=0.5))
    persist_reconciliation(source, hand.id)
    payload = export_session(source, hand.session_id)
    source.close()

    target = _open_db(tmp_path, "tgt.db")
    session = import_session(target, payload)
    assert session.id is not None
    imported = target.fetch_hands_by_session(session.id)[0]
    assert imported.id is not None
    evidence = parse_completion_evidence(imported.completion_evidence)

    assert DECLARED_RAKE_CODE in evidence.declared_settlement_codes
    assert DECLARED_RAKE_CODE not in evidence.warning_codes
    assert evidence.unresolved_codes == ()
    blockers = _readiness(target, imported.id)
    assert "UNRESOLVED_SOURCE_WARNING" not in blockers
    assert "ACCOUNTING_ASSUMPTION_DEPENDENT" in blockers
    target.close()


# ---------------------------------------------------------------------------
# 3. A clearing action that names an operation which never acts on this hand
# ---------------------------------------------------------------------------


_RECONSTRUCTION_PHRASE = "Run CV reconstruction"


def test_only_one_literal_in_the_readiness_module_names_a_new_reconstruction() -> None:
    """The family guard: no branch may spell its own version of this action.

    Seven blocker branches each wrote their own sentence, and every one of them
    described an operation the product does not perform -- "Re-import this hand",
    when ``import_hands_into_session`` appends a rebuilt copy beside the blocked
    hand and renumbers the collision. Fixing the seven sentences would leave the
    eighth branch free to reintroduce the defect, so there is one literal and the
    branches compose it.
    """
    source = (
        Path(study_readiness.__file__).read_text(encoding="utf-8")
    )
    tree = ast.parse(source)
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and _RECONSTRUCTION_PHRASE in node.value
    ]
    assert literals == [study_readiness.NEW_RECONSTRUCTION_STEPS]


def test_every_blocker_naming_a_new_reconstruction_states_what_it_actually_does(
    tmp_path: Path,
) -> None:
    """Behavioural half, over the shapes that reach each of those branches.

    BEFORE this change every one of these read "re-import" or "Re-import this
    hand" and stopped there, so an operator who performed the action verbatim
    kept the blocker, gained a duplicate of every hand in the session, and had a
    completed hand counted twice in that session's statistics.
    """
    shapes: dict[str, dict[str, object]] = {
        "rejection": _clean_evidence(rejection_codes=["board_unreadable"]),
        "unreadable evidence": {},
        "no codes to acknowledge": _clean_evidence(terminal_event=""),
        "partial": _clean_evidence(partial_end=True),
        "partial with no truncation": _clean_evidence(partial_start=None, partial_end=None),
        "unsupported layout": _clean_evidence(layout_supported=False),
        "unresolved rejection code": _clean_evidence(
            warning_codes=["board_unreadable"], rejection_codes=["board_unreadable"]
        ),
    }
    named = 0
    for case, evidence in shapes.items():
        db = _open_db(tmp_path, f"action-{abs(hash(case))}.db")
        session = db.create_session(Session(name=case, date_played=date(2026, 1, 1)))
        assert session.id is not None
        hand = db.create_hand(
            Hand(
                session_id=session.id,
                hand_number=1,
                table_size=6,
                hero_cards="Ah Qs",
                board_cards="Qd 7s 2c",
                source_type="cv_import",
                completion_status="partial" if "partial" in case else "uncertain",
                completion_evidence=evidence,
            )
        )
        assert hand.id is not None
        stored = db.fetch_hand(hand.id)
        assert stored is not None
        result = evaluate_study_readiness(stored, accounting=None, user_confirmed=True)
        for blocker in result.blockers:
            if _RECONSTRUCTION_PHRASE not in blocker.clearing_action:
                continue
            named += 1
            assert study_readiness.NEW_RECONSTRUCTION_STEPS in blocker.clearing_action, (
                f"{case}/{blocker.code} names a reconstruction in its own words"
            )
        db.close()
    assert named >= len(shapes), "the matrix did not reach the branches under test"


def test_the_named_reconstruction_action_appends_and_the_blocker_says_so(
    tmp_path: Path,
) -> None:
    """Performed verbatim, the action leaves this hand blocked and adds a second one.

    This is the behaviour the text now discloses, pinned so the text cannot drift
    away from it: if a future import ever REPLACES instead of appending, this
    fails and the sentence has to be rewritten.
    """
    db = _open_db(tmp_path, "reimport.db")
    hand = _seed_reconstructed(db)
    assert hand.id is not None
    db.update_hand_completion(
        hand.id,
        completion_evidence=_clean_evidence(rejection_codes=["board_unreadable"]),
        notes="Pipeline rejected the board.",
    )
    blocked = db.fetch_hand(hand.id)
    assert blocked is not None
    before = evaluate_study_readiness(blocked, accounting=None, user_confirmed=True)
    action = next(
        item.clearing_action
        for item in before.blockers
        if item.code == "COMPLETION_NOT_COMPLETE"
    )
    assert "delete this one" in action
    assert "never replaces a hand" in action

    # The operator does exactly what the action says: a new reconstruction of the
    # same session, imported into it.
    payload = export_session(db, hand.session_id)
    import_hands_into_session(db, payload, hand.session_id)

    hands = db.fetch_hands_by_session(hand.session_id)
    assert len(hands) == 2, "the import replaced the hand instead of appending"
    still = db.fetch_hand(hand.id)
    assert still is not None
    after = evaluate_study_readiness(still, accounting=None, user_confirmed=True)
    assert tuple(item.code for item in after.blockers) == tuple(
        item.code for item in before.blockers
    )
    db.close()


# ---------------------------------------------------------------------------
# 4. A declaration the RECORDING forces is not an assumption
# ---------------------------------------------------------------------------


def _seed_award_hand(
    db: PokerDatabase,
    *,
    name: str,
    seats: int,
    folds: tuple[str, ...],
    winners: tuple[str, ...],
    hero_bb_won: float | None,
    pot_size: float | None,
    bet: float = 40.0,
) -> Hand:
    """``seats`` seats each commit ``bet``; the seats in ``folds`` then fold."""
    keys = ["hero", "villain", "third", "fourth"][:seats]
    session = db.create_session(Session(name=name, date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=_clean_evidence(terminal_event="showdown"),
        )
    )
    assert hand.id is not None
    for key in keys:
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=key.capitalize(),
                is_hero=key == "hero",
                starting_stack=1000,
            )
        )
    index = 0
    for key in keys:
        index += 1
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=key.capitalize(),
                action_type="bet" if index == 1 else "call",
                amount=bet,
            )
        )
    for key in folds:
        index += 1
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=key.capitalize(),
                action_type="fold",
                amount=None,
            )
        )
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key=key,
                player_name=key.capitalize(),
                entry_order=order,
            )
            for order, key in enumerate(winners, start=1)
        ],
    )
    return hand


@pytest.mark.parametrize(
    ("case", "seats", "folds", "winners", "hero_bb_won", "expected"),
    [
        # The action line leaves exactly one seat standing, so the winner is not
        # a declaration at all -- and this hand declares nothing else, which is
        # the shape that used to produce a mandatory, contentless press of
        # "Confirm this assumption".
        ("heads up fold-out", 2, ("villain",), ("hero",), 40.0, []),
        # Same rule with more seats folding into one.
        ("three-way fold-out", 3, ("villain", "third"), ("hero",), 80.0, []),
        # ...and when the hero is the folder, so the forced winner is not the hero.
        ("hero folds out", 3, ("hero", "third"), ("villain",), -40.0, []),
        # A showdown IS a declaration: two seats are eligible and nothing in the
        # recording says which was pushed the pot.
        ("showdown", 2, (), ("hero",), 40.0, ["declared_pot_awards"]),
        ("three-way showdown", 3, ("third",), ("hero",), 80.0, ["declared_pot_awards"]),
    ],
)
def test_an_award_the_action_line_forces_is_not_a_declared_assumption(
    tmp_path: Path,
    case: str,
    seats: int,
    folds: tuple[str, ...],
    winners: tuple[str, ...],
    hero_bb_won: float,
    expected: list[str],
) -> None:
    """The withdrawn state must be one the recording could actually produce.

    Withdrawing the awards to "nobody won anything" is not such a state: an
    award-less ledger is never ``is_settled``, so ``_is_dependent`` was
    unconditionally True for the awards on EVERY hand whose baseline reconciles.
    A sweep of 1440 authoritative states named ``declared_pot_awards`` on 1440 of
    them, and on 300 it was the only dependence -- a compulsory "Confirm this
    assumption" press on a hand declaring no rake and no dead money at all. The
    module's own argument is that an operator trained to click through
    disclosures that mean nothing will click through the one that means
    something.

    BEFORE this change all five rows measured ``declared_pot_awards``.
    """
    db = _open_db(tmp_path, f"awards-{abs(hash(case))}.db")
    hand = _seed_award_hand(
        db,
        name=case,
        seats=seats,
        folds=folds,
        winners=winners,
        hero_bb_won=hero_bb_won,
        pot_size=float(seats) * 40.0,
    )
    assert hand.id is not None
    result = persist_reconciliation(db, hand.id)
    assert result.ledger.is_settled and result.ledger.is_balanced
    assert [item.input_name for item in result.assumption_dependence] == expected
    blockers = _readiness(db, hand.id)
    assert ("ACCOUNTING_ASSUMPTION_DEPENDENT" in blockers) is bool(expected)
    db.close()


def test_a_forced_award_beside_a_declared_rake_names_only_the_rake(
    tmp_path: Path,
) -> None:
    """The two inputs are attributed separately, so silencing one must not silence the other."""
    db = _open_db(tmp_path, "forced-rake.db")
    hand = _seed_award_hand(
        db,
        name="forced award, declared rake",
        seats=2,
        folds=("villain",),
        winners=("hero",),
        hero_bb_won=0.0,
        pot_size=80.0,
    )
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=0.5))
    result = persist_reconciliation(db, hand.id)
    assert [item.input_name for item in result.assumption_dependence] == ["rake_policy"]
    assert "ACCOUNTING_ASSUMPTION_DEPENDENT" in _readiness(db, hand.id)
    db.close()


def test_the_round_11_award_defect_is_still_measured_on_a_contested_pot(
    tmp_path: Path,
) -> None:
    """The anti-regression for the repair above.

    Round 11's critical was a freshly imported hand -- null ``pot_size``, null
    ``hero_bb_won``, award rows with no amount -- where one dropdown moved the
    reported hero result by the whole pot with an empty blocker tuple. Two seats
    are eligible there, so the winner is a declaration and must stay measured:
    the fold-out exemption must not reach it.
    """
    db = _open_db(tmp_path, "round11.db")
    hand = _seed_award_hand(
        db,
        name="contested",
        seats=2,
        folds=(),
        winners=("hero",),
        hero_bb_won=None,
        pot_size=None,
    )
    assert hand.id is not None
    result = persist_reconciliation(db, hand.id)
    (award,) = [
        item
        for item in result.assumption_dependence
        if item.input_name == "declared_pot_awards"
    ]
    assert dict(award.deltas)["hero"] == pytest.approx(80.0)
    assert "ACCOUNTING_ASSUMPTION_DEPENDENT" in _readiness(db, hand.id)
    db.close()


# ---------------------------------------------------------------------------
# 5. The attestation is the half that is not re-derived, so it is corroborated
# ---------------------------------------------------------------------------


def test_a_forged_attestation_is_reported_by_the_data_health_audit(
    tmp_path: Path,
) -> None:
    """A hand-edited attestation cannot be disproven on read, but it can be audited.

    The dependence is re-measured from the chips on every read; the operator's
    ANSWER to it is a string in a column, and the codes are deterministic enough
    to compute without ever seeing the product. Storing a human assertion means
    it can be forged -- that is symmetric with every attestation here. What is
    checkable is corroboration: the writer files a ``hand_corrections`` row
    naming the code in the same transaction, so an attestation with no such row
    was not written by this product.
    """
    db = _open_db(tmp_path, "forged.db")
    hand = _seed_award_hand(
        db,
        name="forged",
        seats=2,
        folds=(),
        winners=("hero",),
        hero_bb_won=0.0,
        pot_size=80.0,
    )
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, rake_rate=0.5))
    result = persist_reconciliation(db, hand.id)
    codes = [item.code for item in result.assumption_dependence]
    assert codes
    path = db.db_path
    db.close()

    healthy = audit_data_health(path, data_dir=tmp_path, restore_backups=False)
    attestations = next(
        check for check in healthy.checks if check.name == "settlement_attestations"
    )
    assert attestations.status == "pass"

    connection = sqlite3.connect(path)
    stored = connection.execute(
        "SELECT completion_evidence FROM hands WHERE id = ?", (hand.id,)
    ).fetchone()
    evidence = json.loads(stored[0])
    evidence["confirmed_assumption_codes"] = codes
    connection.execute(
        "UPDATE hands SET completion_evidence = ?, completion_status = 'complete' "
        "WHERE id = ?",
        (json.dumps(evidence), hand.id),
    )
    connection.commit()
    connection.close()

    # The forgery does exactly what the finding said it does: the hand reads back
    # unblocked, because no reader can tell a real attestation from a typed one.
    assert "ACCOUNTING_ASSUMPTION_DEPENDENT" not in _readiness(PokerDatabase(path), hand.id)

    # And the audit says so.
    report = audit_data_health(path, data_dir=tmp_path, restore_backups=False)
    flagged = next(
        check for check in report.checks if check.name == "settlement_attestations"
    )
    assert flagged.status == "warning"
    assert any(str(hand.id) in detail for detail in flagged.details)
    assert report.has_warnings is True


def test_an_unchanged_correction_form_still_writes_nothing(tmp_path: Path) -> None:
    """The other half of the same rule: the short-circuit must survive the repair.

    Comparing in the column's space must not turn every re-save into a
    correction. A hand whose row was written through the model has a row that
    already equals what the writer would write, so re-submitting it is still a
    no-op -- no UPDATE, no correction row, no demotion.
    """
    db = _open_db(tmp_path, "noop.db")
    session = db.create_session(Session(name="No-op", date_played=date(2026, 1, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=80.0,
            hero_bb_won=40.0,
            tags=["PREFLOP_3BET_SPOT", "RIVER_DECISION"],
            notes="Kept.",
            review_status="reviewed",
            source_type="manual",
        )
    )
    assert hand.id is not None
    stored = db.fetch_hand(hand.id)
    assert stored is not None

    unchanged = db.update_hand_facts(stored, correction_notes="Nothing changed.")

    assert unchanged.review_status == "reviewed"
    assert db.fetch_hand_corrections(hand.id) == []
    db.close()
