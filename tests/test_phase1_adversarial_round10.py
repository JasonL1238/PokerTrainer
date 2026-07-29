"""Round-10 regressions: the assumption rule's channel, scope, and measurement.

Round 10 attacked the dependence rule that replaced eight rounds of per-field
disclosure. It did not find a per-field hole in it -- there is no field list left
to hole -- but it found four holes AROUND it, and each is a family rather than a
shape:

* **The attestation shared the pipeline channel.** It was written into
  ``warning_codes`` / ``acknowledged_codes``, so the generic one-click
  "Acknowledge" in the Source warnings panel cleared it, reachable through an
  ordinary export/import round trip and through a forged payload alike.
* **The measurement asked about the verdict, not the figures.** A hand that
  records none of the cross-checked figures reconciles under every policy, so a
  declared 90% rake was measured as assumption-INDEPENDENT while it moved the
  hero result the product displays by 90% of the pot.
* **The scope was a string a payload could choose.** ``source_type: manual``
  with the evidence removed claimed the manual exemption for a reconstructed
  hand, and no guard can disprove the claim.
* **The attestation named the chips but not the declaration.** A different rake
  policy over a corrected action line that moved the same chips inherited it.

Plus three defects the same round found elsewhere in Phase 1: the derived hero
result leaking into the observed column through 'Correct hand facts', a
settlement row this build cannot validate raising a ValidationError out of a
fetch, and Coach Review presenting an assumption-dependent reconciliation as
established fact.

Every test below fails on the pre-repair tree; the docstrings say what it did
there. Each covers the family, and each family is exercised at several
independent instances rather than at the one shape the adversary demonstrated.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from poker_tracker.math.accounting import LedgerError
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    IMPORTED_HAND_KEY,
    CompletionEvidence,
    acknowledge_codes,
    confirm_assumption,
    dump_completion_evidence,
    is_assumption_dependence_code,
    parse_completion_evidence,
    requires_assumption_attestation,
)
from poker_tracker.persistence.db import (
    DECLARED_RAKE_CODE,
    UNREADABLE_SETTLEMENT_PREFIX,
    PokerDatabase,
)
from poker_tracker.persistence.import_export import export_session, import_session
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services import hand_accounting
from poker_tracker.services.hand_accounting import (
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import (
    BLOCKER_ORDER,
    evaluate_study_readiness,
    hand_requires_assumption_attestation,
)
from tests.conftest import attest_declared_assumptions

ASSUMPTION_BLOCKER = "ACCOUNTING_ASSUMPTION_DEPENDENT"


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


def _open_db(tmp_path: Path, name: str = "round10.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _seed(
    db: PokerDatabase,
    *,
    seats: int = 2,
    bet: float = 40.0,
    winners: tuple[str, ...] = ("hero",),
    hero_bb_won: float | None = None,
    pot_size: float | None = None,
    award_amounts: tuple[float | None, ...] | None = None,
    source_type: str = "cv_import",
    completion_status: str = "complete",
    evidence: dict[str, object] | None = None,
    session_name: str = "Round 10",
) -> Hand:
    """``seats`` seats commit ``bet`` each; ``winners`` share pot 0."""
    session = db.create_session(Session(name=session_name, date_played=date(2026, 1, 1)))
    assert session.id is not None
    keys = ["hero", "villain", "third", "fourth", "fifth"][:seats]
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c",
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type=source_type,  # type: ignore[arg-type]
            completion_status=completion_status,  # type: ignore[arg-type]
            completion_evidence=_clean_evidence() if evidence is None else evidence,
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
    for index, key in enumerate(keys, start=1):
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
    amounts = award_amounts or tuple(None for _ in winners)
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key=key,
                player_name=key.capitalize(),
                amount=amount,
                entry_order=order,
            )
            for order, (key, amount) in enumerate(
                zip(winners, amounts, strict=True), start=1
            )
        ],
    )
    return hand


def _press_every_generic_acknowledge(db: PokerDatabase, hand_id: int) -> Hand:
    """Exactly what ``app.show_source_warning_controls`` does, for every offered code."""
    for _ in range(6):
        stored = db.fetch_hand(hand_id)
        assert stored is not None
        evidence = parse_completion_evidence(stored.completion_evidence)
        if not evidence.unresolved_codes:
            break
        db.update_hand_completion(
            hand_id,
            completion_evidence=dump_completion_evidence(
                acknowledge_codes(evidence, list(evidence.unresolved_codes))
            ),
            notes="Acknowledged in test.",
        )
    refreshed = db.fetch_hand(hand_id)
    assert refreshed is not None
    return refreshed


def _named(result: object, input_name: str = "rake_policy"):
    """The one measured dependence naming ``input_name``.

    A reconstructed hand's award rows are typed into the Accounting
    reconciliation panel by the operator -- the CV exporter emits no settlement
    key -- so they are a declared settlement input, measured on their own beside
    the rake and the dead money. Selecting by name keeps each test below pinning
    the declaration it is about rather than the size of the tuple.
    """
    (found,) = [
        item
        for item in result.assumption_dependence  # type: ignore[attr-defined]
        if item.input_name == input_name
    ]
    return found


def _readiness(db: PokerDatabase, hand_id: int, *, confirmed: bool = True):
    stored = db.fetch_hand(hand_id)
    assert stored is not None
    return evaluate_study_readiness(
        stored,
        accounting=reconcile_persisted_hand(db, hand_id),
        user_confirmed=confirmed,
    )


# ---------------------------------------------------------------------------
# Family A. The attestation has its own channel, and only its own control
# ---------------------------------------------------------------------------


def test_a_generic_acknowledge_cannot_clear_an_assumption_after_a_round_trip(
    tmp_path: Path,
) -> None:
    """The legitimate route: attest here, export, import, press Acknowledge there.

    Pre-repair: ``acknowledge_accounting_assumption`` wrote the code into
    ``warning_codes`` as well as ``acknowledged_codes``; import reset the
    acknowledgements and kept the warnings, so the attestation arrived as an
    unacknowledged PIPELINE WARNING -- it demoted ``completion_status`` to
    ``uncertain``, was reported by UNRESOLVED_SOURCE_WARNING as a field to fix in
    'Correct hand facts', and one press of the generic Acknowledge cleared
    ACCOUNTING_ASSUMPTION_DEPENDENT and re-promoted the hand to study-ready with
    an empty blocker tuple, its recorded hero result half a pot from its own
    action line.
    """
    source = _open_db(tmp_path, "source.db")
    hand = _seed(source, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    source.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    result = persist_reconciliation(source, hand.id)
    dependence = _named(result)
    attest_declared_assumptions(source, hand.id, only="declared_pot_awards")
    # The pipeline warnings are answered first, so the hand is `complete` and the
    # attestation below is the only thing that changes.
    _press_every_generic_acknowledge(source, hand.id)
    promoted = source.fetch_hand(hand.id)
    assert promoted is not None
    assert promoted.completion_status == "complete"
    assert _readiness(source, hand.id).has(ASSUMPTION_BLOCKER) is True

    assert source.acknowledge_accounting_assumption(hand.id, dependence.code) is True
    assert _readiness(source, hand.id).has(ASSUMPTION_BLOCKER) is False

    stored = source.fetch_hand(hand.id)
    assert stored is not None
    attested = parse_completion_evidence(stored.completion_evidence)
    assert dependence.code in attested.confirmed_assumption_codes
    # The attestation is not a pipeline code and never enters the pipeline sets,
    # so it cannot be offered to, or answered by, a control that clears those.
    assert dependence.code not in attested.warning_codes
    assert dependence.code not in attested.acknowledged_codes
    assert dependence.code not in attested.unresolved_codes
    assert stored.completion_status == "complete"

    session_payload = export_session(source, stored.session_id)
    source.close()

    target = _open_db(tmp_path, "target.db")
    imported = import_session(target, json.loads(json.dumps(session_payload)))
    assert imported.id is not None
    landed = target.fetch_hands_by_session(imported.id)[0]
    assert landed.id is not None
    landed_evidence = parse_completion_evidence(landed.completion_evidence)
    # The attestation does not travel: it is this operator's statement.
    assert landed_evidence.confirmed_assumption_codes == ()
    assert not any(
        is_assumption_dependence_code(code)
        for code in (
            *landed_evidence.warning_codes,
            *landed_evidence.rejection_codes,
            *landed_evidence.acknowledged_codes,
        )
    )
    assert _readiness(target, landed.id).has(ASSUMPTION_BLOCKER) is True

    after = _press_every_generic_acknowledge(target, landed.id)
    readiness = _readiness(target, landed.id)
    assert readiness.has(ASSUMPTION_BLOCKER) is True
    assert readiness.is_ready is False
    assert after.hero_bb_won == 0.0
    target.close()


@pytest.mark.parametrize(
    "channel", ["warning_codes", "acknowledged_codes", "rejection_codes"]
)
def test_a_forged_payload_cannot_put_a_dependence_code_in_a_pipeline_channel(
    tmp_path: Path, channel: str
) -> None:
    """Forgery, in each of the three pipeline channels.

    Pre-repair: listing the measured code in ``warning_codes`` was enough --
    import copied it verbatim, the Study page drew an Acknowledge button for it,
    and one press cleared the blocker. No ``acknowledged_codes`` forgery was
    needed, which is the one thing import already stripped.
    """
    db = _open_db(tmp_path, f"forged_{channel}.db")
    probe = _seed(db, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert probe.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=probe.id, status="reconciled", rake_rate=0.5)
    )
    measured = persist_reconciliation(db, probe.id)
    dependence = _named(measured)
    db.close()

    target = _open_db(tmp_path, f"forged_target_{channel}.db")
    payload = {
        "export_version": 5,
        "session": {"name": "Forged", "date_played": "2026-01-01"},
        "hands": [
            {
                "hand": {
                    "hand_number": 1,
                    "table_size": 6,
                    "hero_cards": "Ah Qs",
                    "board_cards": "Qd 7s 2c",
                    "pot_size": 80.0,
                    "hero_bb_won": 0.0,
                    "source_type": "cv_import",
                    "completion_status": "complete",
                    "completion_evidence": _clean_evidence(
                        **{channel: [dependence.code]}
                    ),
                    "tags": [],
                },
                "players": [
                    {"player_key": "hero", "player_name": "Hero", "is_hero": True,
                     "starting_stack": 1000},
                    {"player_key": "villain", "player_name": "Villain",
                     "is_hero": False, "starting_stack": 1000},
                ],
                "actions": [
                    {"player_key": "hero", "player_name": "Hero", "street": "river",
                     "action_index": 1, "action_type": "bet", "amount": 40.0,
                     "amount_semantics": "incremental"},
                    {"player_key": "villain", "player_name": "Villain",
                     "street": "river", "action_index": 2, "action_type": "call",
                     "amount": 40.0, "amount_semantics": "incremental"},
                ],
                "settlement": {
                    "status": "reconciled", "rake_rate": 0.5, "gross_pot": 80.0,
                    "rake_amount": 40.0, "net_pot": 40.0, "is_balanced": True,
                },
                "settlement_entries": [
                    {"entry_type": "award", "pot_index": 0, "player_key": "hero",
                     "player_name": "Hero", "amount": 40.0, "entry_order": 1},
                ],
            }
        ],
    }
    imported = import_session(target, payload)
    assert imported.id is not None
    landed = target.fetch_hands_by_session(imported.id)[0]
    assert landed.id is not None
    _press_every_generic_acknowledge(target, landed.id)
    readiness = _readiness(target, landed.id)
    assert readiness.has(ASSUMPTION_BLOCKER) is True
    assert readiness.is_ready is False
    target.close()


def test_a_hand_edited_evidence_column_cannot_answer_the_assumption(
    tmp_path: Path,
) -> None:
    """The same forgery written straight into SQLite, bypassing import entirely.

    ``hands`` has no CHECK constraint on ``completion_evidence``. The channel
    separation therefore has to be enforced by the READER, which every surface
    goes through, and not by import.
    """
    db = _open_db(tmp_path, "handedited.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    dependence = _named(persist_reconciliation(db, hand.id))

    forged = _clean_evidence(
        warning_codes=[dependence.code],
        acknowledged_codes=[dependence.code],
    )
    connection = sqlite3.connect(str(tmp_path / "handedited.db"))
    connection.execute(
        "UPDATE hands SET completion_evidence = ? WHERE id = ?",
        (json.dumps(forged), hand.id),
    )
    connection.commit()
    connection.close()

    readiness = _readiness(db, hand.id)
    assert readiness.has(ASSUMPTION_BLOCKER) is True
    assert readiness.is_ready is False
    db.close()


def test_a_dependence_code_is_never_acknowledgeable_as_a_pipeline_warning() -> None:
    """The leaf rule, stated directly: the two channels never mix, in either direction."""
    code = "declared_settlement_dependence:rake_policy:abc123:rake+40|hero-40"
    evidence = parse_completion_evidence(
        {
            "evidence_version": 1,
            "warning_codes": [code, "declared_unobserved_rake"],
            "rejection_codes": [code],
            "acknowledged_codes": [code],
            "confirmed_assumption_codes": [code, "hero_seat_mismatch"],
        }
    )
    # AMENDED in round 12: `declared_unobserved_rake` is an operator declaration,
    # not a pipeline warning, and it has its own channel too. The leaf rule now
    # separates three kinds of claim rather than two, in every direction.
    assert evidence.warning_codes == ()
    assert evidence.rejection_codes == ()
    assert evidence.acknowledged_codes == ()
    assert evidence.confirmed_assumption_codes == (code,)
    assert evidence.declared_settlement_codes == ("declared_unobserved_rake",)
    # Even handed the code directly, the pipeline acknowledger refuses it.
    reconstructed = CompletionEvidence(evidence_version=1, warning_codes=(code,))
    assert acknowledge_codes(reconstructed, [code]).acknowledged_codes == ()


# ---------------------------------------------------------------------------
# Family B. Dependence is measured on the reported figures, not just the verdict
# ---------------------------------------------------------------------------


DECLARATIONS_THAT_MOVE_CHIPS = [
    pytest.param({"rake_rate": 0.9}, id="rake-90pct"),
    pytest.param({"rake_rate": 0.5}, id="rake-50pct"),
    pytest.param({"rake_rate": 0.25}, id="rake-25pct"),
    pytest.param({"dead_money": 75.0}, id="dead-money"),
    pytest.param({"rake_rate": 0.5, "dead_money": 20.0}, id="rake-and-dead-money"),
    pytest.param({"rake_rate": 1.0, "rake_cap": 12.0}, id="rake-capped"),
    pytest.param(
        {"rake_rate": 1.0, "rake_rounding_unit": 7.0}, id="rake-coarse-unit"
    ),
    pytest.param(
        {"rake_rate": 0.5, "no_flop_no_drop": True}, id="rake-with-waiver-and-a-flop"
    ),
]


@pytest.mark.parametrize("declaration", DECLARATIONS_THAT_MOVE_CHIPS)
def test_a_hand_recording_no_figures_still_measures_its_declaration(
    tmp_path: Path, declaration: dict[str, float | bool]
) -> None:
    """The family: nothing recorded, so BOTH passes reconcile -- and chips still move.

    A freshly imported hand records none of the figures the cross-check compares:
    ``import_session`` never calls ``persist_reconciliation``, so ``gross_pot``,
    ``rake_amount`` and ``net_pot`` stay NULL, and a pipeline that could not read
    the hero's result leaves ``hands.hero_bb_won`` NULL beside a blank award
    amount. Pre-repair the neutral pass had nothing left to disagree with, so it
    reconciled too and the rule reported ``()`` -- while the declared policy moved
    the hero result that ``_hands_with_accounting_results`` and
    ``math.analytics`` substitute into every list, every stat and every prompt.
    """
    db = _open_db(tmp_path, "nofigures.db")
    hand = _seed(db, hero_bb_won=None, pot_size=None, award_amounts=(None,))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", **declaration)  # type: ignore[arg-type]
    )

    result = reconcile_persisted_hand(db, hand.id)
    assert result.is_authoritative is True
    assert result.assumption_dependence, "a declaration that moves chips is never silent"
    hero_declared = result.ledger.net_results["hero"]
    assert hero_declared != pytest.approx(40.0), "this declaration must move the hero result"

    _press_every_generic_acknowledge(db, hand.id)
    readiness = _readiness(db, hand.id)
    assert readiness.has(ASSUMPTION_BLOCKER) is True
    assert readiness.is_ready is False

    # And the attestation clears it, so the blocker names an action that works.
    for dependence in result.assumption_dependence:
        assert db.acknowledge_accounting_assumption(hand.id, dependence.code) is True
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is False
    db.close()


@pytest.mark.parametrize(
    "declaration",
    [
        pytest.param({"rake_rate": 1.0, "rake_cap": 0.0}, id="zero-cap"),
        pytest.param(
            {"rake_rate": 1.0, "rake_rounding_unit": 1e6}, id="unit-coarser-than-the-pot"
        ),
        pytest.param(
            {"rake_rate": 0.5, "no_flop_no_drop": True}, id="waiver-on-a-preflop-hand"
        ),
    ],
)
def test_a_declaration_that_moves_no_chips_stays_silent_even_with_nothing_recorded(
    tmp_path: Path, declaration: dict[str, float | bool]
) -> None:
    """The other half, and it matters as much: no chips move, so nothing is disclosed.

    Measuring the FIGURES rather than the verdict must not turn into disclosing
    every declaration. An operator trained to click through disclosures that mean
    nothing is an operator who clicks through the one that means something.
    """
    db = _open_db(tmp_path, "silent.db")
    no_flop = declaration.get("no_flop_no_drop") is True
    hand = _seed(db, hero_bb_won=None, pot_size=None, award_amounts=(None,))
    assert hand.id is not None
    if no_flop:
        # A hand that never saw a flop, so the waiver genuinely takes nothing.
        db.update_hand_facts(
            (db.fetch_hand(hand.id) or hand).model_copy(update={"board_cards": ""}),
            correction_notes="No flop was dealt.",
        )
        connection = sqlite3.connect(str(tmp_path / "silent.db"))
        connection.execute(
            "UPDATE actions SET street = 'preflop' WHERE hand_id = ?", (hand.id,)
        )
        connection.commit()
        connection.close()
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", **declaration)  # type: ignore[arg-type]
    )

    result = reconcile_persisted_hand(db, hand.id)
    assert result.ledger.rake == pytest.approx(0.0)
    # The rake is never named. The declared pot award is a separate declaration,
    # measured on its own, and answering it is what leaves the hand unblocked.
    assert [item.input_name for item in result.assumption_dependence] == [
        "declared_pot_awards"
    ]
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True
    attest_declared_assumptions(db, hand.id, only="declared_pot_awards")
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is False
    db.close()


def test_a_chopped_pot_with_nothing_recorded_is_measured_too(tmp_path: Path) -> None:
    """A third seat contributing into a two-way chop, with every figure NULL.

    The shape rounds 8 and 9 repaired one at a time, now crossed with the
    "records nothing" family: three seats commit 40 each, two share the pot, and
    the declared rake moves what each of them is paid.
    """
    db = _open_db(tmp_path, "chop.db")
    hand = _seed(
        db,
        seats=3,
        winners=("hero", "villain"),
        hero_bb_won=None,
        pot_size=None,
        award_amounts=(None, None),
    )
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    result = reconcile_persisted_hand(db, hand.id)
    dependence = _named(result)
    assert "payout" in dependence.code
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True
    db.close()


# ---------------------------------------------------------------------------
# Family C. The manual exemption is about who entered the hand, not a string
# ---------------------------------------------------------------------------


def test_an_imported_hand_declaring_manual_cannot_claim_the_exemption(
    tmp_path: Path,
) -> None:
    """A payload can declare anything; it cannot have been typed in here.

    Pre-repair: declaring ``source_type: manual`` with ``completion_evidence``
    removed satisfied import's manual-payload guard (which only refuses READABLE
    reconstruction evidence), derived ``not_applicable``, and the hand landed
    study-ready with an EMPTY blocker tuple before any click at all -- exempt from
    the assumption blocker and from every other reconstructed-hand blocker --
    while recording a hero result 40 chips from its own action line.
    """
    db = _open_db(tmp_path, "manualclaim.db")
    payload = {
        "export_version": 5,
        "session": {"name": "Forged manual", "date_played": "2026-01-01"},
        "hands": [
            {
                "hand": {
                    "hand_number": 1,
                    "table_size": 6,
                    "hero_cards": "Ah Qs",
                    "board_cards": "Qd 7s 2c",
                    "pot_size": 80.0,
                    "hero_bb_won": 0.0,
                    "source_type": "manual",
                    "completion_status": "not_applicable",
                    "completion_evidence": {},
                    "tags": [],
                },
                "players": [
                    {"player_key": "hero", "player_name": "Hero", "is_hero": True,
                     "starting_stack": 1000},
                    {"player_key": "villain", "player_name": "Villain",
                     "is_hero": False, "starting_stack": 1000},
                ],
                "actions": [
                    {"player_key": "hero", "player_name": "Hero", "street": "river",
                     "action_index": 1, "action_type": "bet", "amount": 40.0,
                     "amount_semantics": "incremental"},
                    {"player_key": "villain", "player_name": "Villain",
                     "street": "river", "action_index": 2, "action_type": "call",
                     "amount": 40.0, "amount_semantics": "incremental"},
                ],
                "settlement": {
                    "status": "reconciled", "rake_rate": 0.5, "gross_pot": 80.0,
                    "rake_amount": 40.0, "net_pot": 40.0, "is_balanced": True,
                },
                "settlement_entries": [
                    {"entry_type": "award", "pot_index": 0, "player_key": "hero",
                     "player_name": "Hero", "amount": 40.0, "entry_order": 1},
                ],
            }
        ],
    }
    imported = import_session(db, payload)
    assert imported.id is not None
    landed = db.fetch_hands_by_session(imported.id)[0]
    assert landed.id is not None
    assert landed.source_type == "manual"
    assert landed.completion_status == "not_applicable"
    assert hand_requires_assumption_attestation(landed) is True

    readiness = _readiness(db, landed.id)
    assert readiness.has(ASSUMPTION_BLOCKER) is True
    assert readiness.is_ready is False

    # And the control the blocker names works on it.
    dependence = _named(reconcile_persisted_hand(db, landed.id))
    assert dependence.code in attest_declared_assumptions(db, landed.id)
    assert _readiness(db, landed.id).has(ASSUMPTION_BLOCKER) is False
    db.close()


def test_a_manual_hand_entered_here_keeps_its_exemption(tmp_path: Path) -> None:
    """The control. The exemption is not weakened for the workflow it exists for.

    A hand typed into this database by its own operator: the declared rake is
    that operator's own observation, there is no pipeline claim for it to
    outrank, and it is disclosed but never blocked.
    """
    db = _open_db(tmp_path, "genuine_manual.db")
    hand = _seed(
        db,
        hero_bb_won=0.0,
        pot_size=80.0,
        award_amounts=(40.0,),
        source_type="manual",
        completion_status="not_applicable",
        evidence={},
    )
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    result = persist_reconciliation(db, hand.id)
    assert result.assumption_dependence, "still measured, and still reported"
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert hand_requires_assumption_attestation(stored) is False
    readiness = _readiness(db, hand.id)
    assert readiness.has(ASSUMPTION_BLOCKER) is False
    assert readiness.is_ready is True
    # A writer that is never called for this hand records nothing for it.
    assert (
        db.acknowledge_accounting_assumption(
            hand.id, result.assumption_dependence[0].code
        )
        is False
    )
    refreshed = db.fetch_hand(hand.id)
    assert refreshed is not None
    assert refreshed.completion_evidence == {}
    db.close()


def test_an_exported_manual_hand_is_attested_to_in_the_database_it_lands_in(
    tmp_path: Path,
) -> None:
    """No forgery at all: the same rule, reached by exporting your own manual hand.

    The importing operator did not enter it, so they attest to its declared rake
    once, in their own database. The stamp is idempotent, so a further round trip
    changes nothing.
    """
    source = _open_db(tmp_path, "manual_source.db")
    hand = _seed(
        source,
        hero_bb_won=0.0,
        pot_size=80.0,
        award_amounts=(40.0,),
        source_type="manual",
        completion_status="not_applicable",
        evidence={},
    )
    assert hand.id is not None
    source.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    persist_reconciliation(source, hand.id)
    payload = export_session(source, hand.session_id)
    source.close()

    target = _open_db(tmp_path, "manual_target.db")
    imported = import_session(target, json.loads(json.dumps(payload)))
    assert imported.id is not None
    landed = target.fetch_hands_by_session(imported.id)[0]
    assert landed.id is not None
    assert landed.completion_evidence.get(IMPORTED_HAND_KEY) is True
    assert parse_completion_evidence(landed.completion_evidence).is_known is False
    assert _readiness(target, landed.id).has(ASSUMPTION_BLOCKER) is True

    second_payload = export_session(target, imported.id)
    third = _open_db(tmp_path, "manual_third.db")
    third_session = import_session(third, json.loads(json.dumps(second_payload)))
    assert third_session.id is not None
    twice_landed = third.fetch_hands_by_session(third_session.id)[0]
    assert twice_landed.completion_evidence == landed.completion_evidence
    target.close()
    third.close()


def test_the_attestation_scope_predicate_is_one_rule() -> None:
    """The blocker, the control, and the writer consult the same function."""
    entered_here = CompletionEvidence()
    imported = CompletionEvidence(extra={IMPORTED_HAND_KEY: True})
    assert (
        requires_assumption_attestation(
            source_type="manual",
            completion_status="not_applicable",
            evidence=entered_here,
        )
        is False
    )
    for source_type, status in (
        ("cv_import", "complete"),
        ("corrected_cv", "uncertain"),
        ("manual", "complete"),
        ("manual", "partial"),
    ):
        assert (
            requires_assumption_attestation(
                source_type=source_type,
                completion_status=status,
                evidence=entered_here,
            )
            is True
        )
    assert (
        requires_assumption_attestation(
            source_type="manual", completion_status="not_applicable", evidence=imported
        )
        is True
    )


# ---------------------------------------------------------------------------
# Family D. The attestation is bound to the declaration, not only to the chips
# ---------------------------------------------------------------------------


def test_a_different_policy_moving_the_same_chips_does_not_inherit_the_attestation(
    tmp_path: Path,
) -> None:
    """A 50% rake on 80 chips and a 25% rake on 160 chips both destroy 40.

    Pre-repair the code carried the movement alone, so both produced a
    byte-identical string: the operator's confirmation of the first was inherited
    by the second -- a policy they never saw, over an action line that had
    doubled, with the recorded hero result moved from 0 to +40.
    """
    db = _open_db(tmp_path, "collision.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    first = _named(persist_reconciliation(db, hand.id))
    assert first.code in attest_declared_assumptions(db, hand.id)
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is False

    for action in db.fetch_actions_by_hand(hand.id):
        assert action.id is not None
        db.update_action(
            action.model_copy(update={"amount": 80.0}),
            correction_notes="Corrected the river amounts.",
        )
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.25)
    )
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    db.update_hand_facts(
        stored.model_copy(update={"pot_size": 160.0, "hero_bb_won": 40.0}),
        correction_notes="Corrected the recorded pot and result.",
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
                amount=120.0,
                entry_order=1,
            )
        ],
    )
    second = _named(persist_reconciliation(db, hand.id))
    assert second.deltas == first.deltas, "the same chips move; only the policy differs"
    assert second.code != first.code
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True
    db.close()


def test_an_unchanged_declaration_keeps_its_attestation_across_a_re_save(
    tmp_path: Path,
) -> None:
    """The other side of the same coin: no churn, or the attestation is worthless.

    Re-saving the settlement, and re-deriving the code on every read, must
    produce the identical string. An attestation that expires for no reason
    trains the operator to re-click it without reading.
    """
    db = _open_db(tmp_path, "stable.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    first = _named(persist_reconciliation(db, hand.id))
    assert first.code in attest_declared_assumptions(db, hand.id)
    for _ in range(3):
        again = _named(persist_reconciliation(db, hand.id))
        assert again.code == first.code
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is False
    db.close()


def test_a_re_measured_input_replaces_its_predecessor_in_the_evidence() -> None:
    """One attestation per declared input, never two contradictory chip figures.

    Also the reason it is a replacement and not an append: a stale attestation
    left behind would re-clear the blocker by itself if the hand were later
    corrected back to the earlier measurement.
    """
    small = "declared_settlement_dependence:rake_policy:aaaaaaaaaa:rake+0.01|hero-0.01"
    large = "declared_settlement_dependence:rake_policy:bbbbbbbbbb:rake+80.01|hero-80.01"
    other = "declared_settlement_dependence:dead_money:cccccccccc:gross+5|hero+5"
    evidence = CompletionEvidence(evidence_version=1)
    after_small = confirm_assumption(evidence, small)
    assert after_small is not None
    after_other = confirm_assumption(after_small, other)
    assert after_other is not None
    after_large = confirm_assumption(after_other, large)
    assert after_large is not None
    assert after_large.confirmed_assumption_codes == (other, large)
    # Idempotent: confirming the same measurement twice changes nothing.
    assert confirm_assumption(after_large, large) is None
    with pytest.raises(ValueError):
        confirm_assumption(evidence, "declared_unobserved_rake")


# ---------------------------------------------------------------------------
# Family E. An unreadable settlement row degrades; it never raises into a fetch
# ---------------------------------------------------------------------------


UNREADABLE_SETTLEMENT_VALUES = [
    pytest.param("rake_rate", -0.5, id="negative-rake-rate"),
    pytest.param("rake_rate", 4.0, id="rake-rate-above-one"),
    pytest.param("rake_rate", "NaN", id="nan-rake-rate"),
    pytest.param("dead_money", -25.0, id="negative-dead-money"),
    pytest.param("rake_rounding_unit", 0.0, id="zero-rounding-unit"),
    pytest.param("rake_cap", -1.0, id="negative-rake-cap"),
    pytest.param("rake_amount", -40.0, id="negative-rake-amount"),
    pytest.param("net_pot", -1.0, id="negative-net-pot"),
    pytest.param("status", "definitely-reconciled", id="unknown-status"),
    pytest.param("created_at", "never", id="unreadable-timestamp"),
]


@pytest.mark.parametrize("column,value", UNREADABLE_SETTLEMENT_VALUES)
def test_an_unreadable_settlement_row_degrades_instead_of_raising(
    tmp_path: Path, column: str, value: object
) -> None:
    """One bad row must cost that hand its authority, not the whole page.

    Pre-repair ``_hand_settlement_from_row`` fed the raw row to a validating
    model, so a ValidationError escaped ``fetch_hand_settlement`` ->
    ``reconcile_persisted_hand`` -> ``app._reconcile_cached`` (which catches
    ``LedgerError`` only) and reached Streamlit unhandled. Its immediate
    neighbours already degrade: the warnings column in the same function, and
    ``source_type``, ``completion_status`` and the card columns in
    ``_hand_from_row``.
    """
    db = _open_db(tmp_path, "badrow.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    other = _seed(db, hero_bb_won=None, pot_size=None, session_name="Second hand")
    assert other.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    persist_reconciliation(db, hand.id)

    connection = sqlite3.connect(str(tmp_path / "badrow.db"))
    connection.execute(
        f"UPDATE hand_settlements SET {column} = ? WHERE hand_id = ?",  # noqa: S608
        (value, hand.id),
    )
    connection.commit()
    connection.close()

    settlement = db.fetch_hand_settlement(hand.id)
    assert settlement is not None
    assert settlement.status != "reconciled"
    assert settlement.is_balanced is False
    assert any(
        note.startswith(UNREADABLE_SETTLEMENT_PREFIX) for note in settlement.warnings
    )

    result = reconcile_persisted_hand(db, hand.id)
    assert result.is_authoritative is False
    assert any(issue.startswith(UNREADABLE_SETTLEMENT_PREFIX) for issue in result.issues)
    readiness = _readiness(db, hand.id)
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE") is True
    assert readiness.is_ready is False

    # The rest of the database is unaffected: one row does not take a list down.
    assert sorted(row.id or 0 for row in db.fetch_all_hands()) == sorted(
        [hand.id, other.id]
    )
    reconcile_persisted_hand(db, other.id)
    db.close()


def test_the_unreadable_marker_is_never_persisted(tmp_path: Path) -> None:
    """It describes the row a reader could not validate, so no writer may store it."""
    db = _open_db(tmp_path, "badrow_write.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
    connection = sqlite3.connect(str(tmp_path / "badrow_write.db"))
    connection.execute(
        "UPDATE hand_settlements SET rake_rounding_unit = 0.0 WHERE hand_id = ?",
        (hand.id,),
    )
    connection.commit()
    connection.close()

    degraded = db.fetch_hand_settlement(hand.id)
    assert degraded is not None
    assert any(
        note.startswith(UNREADABLE_SETTLEMENT_PREFIX) for note in degraded.warnings
    )
    persist_reconciliation(db, hand.id)
    saved = db.fetch_hand_settlement(hand.id)
    assert saved is not None
    assert not any(
        note.startswith(UNREADABLE_SETTLEMENT_PREFIX) for note in saved.warnings
    )
    assert saved.rake_rounding_unit == pytest.approx(0.01)
    db.close()


# ---------------------------------------------------------------------------
# Family F. A derived hero result may never be written into the observed column
# ---------------------------------------------------------------------------


def test_a_display_copy_of_a_hand_is_refused_by_every_writer(tmp_path: Path) -> None:
    """The structural half of the 'Correct hand facts' leak.

    ``app._hands_with_accounting_results`` replaces ``hero_bb_won`` with the
    DERIVED ledger result for display, and that object was indistinguishable from
    a stored hand -- it reached the fact editor, where saving an unrelated field
    persisted the derivation as an observation and recorded it in
    ``hand_corrections`` as a fact the operator stated. Fixing the call site
    fixes one call site; refusing the object at the writer covers all of them.
    """
    db = _open_db(tmp_path, "displaycopy.db")
    hand = _seed(db, hero_bb_won=None, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.derived_result_substituted is False

    display_copy = stored.model_copy(
        update={"hero_bb_won": 40.0, "derived_result_substituted": True}
    )
    with pytest.raises(ValueError, match="substituted"):
        db.update_hand_facts(display_copy, correction_notes="Unrelated note.")
    with pytest.raises(ValueError, match="substituted"):
        db.create_hand(
            display_copy.model_copy(update={"id": None, "hand_number": 2})
        )
    refreshed = db.fetch_hand(hand.id)
    assert refreshed is not None
    assert refreshed.hero_bb_won is None
    # The marker is read-time only: it is excluded from every dump, so no export
    # or payload can carry it and no stored row can be forged with it.
    assert "derived_result_substituted" not in display_copy.model_dump()
    db.close()


def test_the_fact_editor_never_writes_the_derived_result_into_the_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The UI half, on the real Study page.

    Pre-repair the 'Hero result (BB)' input rendered pre-filled with the DERIVED
    ledger result while ``hands.hero_bb_won`` was NULL, and correcting an
    unrelated field -- here, the hand notes -- persisted that derivation as an
    observation, recorded in ``hand_corrections`` as a fact the operator stated.
    With a rake declared the number written was the operator's own rake policy
    applied to the action line: a settlement assumption laundered into the column
    ``hand_accounting._cross_check`` compares EXACTLY because it is supposed to be
    independent evidence of what the hero won.
    """
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import poker_tracker.persistence.db as db_module
    from poker_tracker.ui.navigation import Page

    path = tmp_path / "facteditor_ui.db"
    db = _open_db(tmp_path, "facteditor_ui.db")
    hand = _seed(db, hero_bb_won=None, pot_size=80.0, award_amounts=(None,))
    assert hand.id is not None
    hand_id = hand.id
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand_id, status="reconciled", rake_rate=0.5)
    )
    result = persist_reconciliation(db, hand_id)
    assert result.is_authoritative is True
    for dependence in result.assumption_dependence:
        db.acknowledge_accounting_assumption(hand_id, dependence.code)
    db.close()

    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()
    app = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60
    ).run()
    app.radio[0].set_value(Page.STUDY)
    app.run()
    assert not list(app.exception)

    widget = next(item for item in app.number_input if item.label == "Hero result (BB)")
    assert widget.value is None, "the form shows the observation, never the derivation"

    next(item for item in app.text_area if item.label == "Hand notes").set_value(
        "checked the turn card against the video"
    )
    next(
        item
        for item in app.text_input
        if item.label == "Why is this correction needed?"
    ).set_value("turn card was misread")
    next(item for item in app.button if item.label == "Save corrected facts").click()
    app.run()
    assert not list(app.exception)

    verifier = PokerDatabase(str(path))
    verifier.init_db()
    stored = verifier.fetch_hand(hand_id)
    assert stored is not None
    assert stored.hero_bb_won is None, "the hero result was never observed"
    assert stored.notes == "checked the turn card against the video"
    verifier.close()
    st.cache_resource.clear()


# ---------------------------------------------------------------------------
# Family G. No surface presents an assumption-dependent reconciliation as fact
# ---------------------------------------------------------------------------


def test_an_assumption_dependent_hand_is_not_established_for_any_consumer(
    tmp_path: Path,
) -> None:
    """Study refuses the hand; the coaching prompt used to call it 'reconciled'.

    Pre-repair the Coach Review button was gated on ``is_authoritative`` alone,
    so it was enabled, and the prompt was built with
    ``accounting_authoritative=True`` and no accounting issues: the provider was
    told "Final pot: 80 BB (reconciled) / Rake: 40 BB / Result: 0.00 BB /
    Accounting: reconciled", with nothing saying the 40 BB was an operator
    declaration the recording does not support. The saved review was then
    retained as coaching evidence about a hand Study refuses.
    """
    import app as app_module

    db = _open_db(tmp_path, "coach.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    result = persist_reconciliation(db, hand.id)
    assert result.is_authoritative is True
    assert result.assumption_dependence

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert app_module._accounting_is_established(stored, result) is False
    assert app_module._accounting_prompt_math_facts(stored, result) == {}
    issues = app_module._accounting_prompt_issues(result, None)
    assert any("declared settlement assumption" in issue for issue in issues)
    assert any("rake_policy" in issue for issue in issues)

    # AMENDED, and the amendment is a finding this test used to pin shut.
    #
    # It asserted that the predicate stays False after every dependence has been
    # confirmed -- i.e. that the ONLY way to re-enable coaching is to withdraw the
    # declaration. That is a dead end with the operator's own truthful entry on
    # the wrong side of it: an ordinary room rake really does take chips, so on
    # every raked hand 'Generate and save corrected-hand coaching' stayed disabled
    # forever above a message naming an action already performed, and on a manual
    # hand -- where no attestation control is ever drawn -- nothing could clear it
    # at all. The measurement is a MEASUREMENT; the attestation is the answer to
    # it, and an answered declaration establishes the figures.
    attest_declared_assumptions(db, hand.id)
    answered = reconcile_persisted_hand(db, hand.id)
    attested = db.fetch_hand(hand.id)
    assert attested is not None
    assert answered.assumption_dependence, "still measured, and still displayed"
    assert app_module._accounting_is_established(attested, answered) is True
    assert app_module._accounting_prompt_math_facts(attested, answered)

    # The attestation covers the declaration it was given against and no other:
    # re-declaring the rake lapses it and the predicate closes again.
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.25)
    )
    relapsed = persist_reconciliation(db, hand.id)
    reread = db.fetch_hand(hand.id)
    assert reread is not None
    assert app_module._accounting_is_established(reread, relapsed) is False

    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.0)
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
                amount=80.0,
                entry_order=1,
            )
        ],
    )
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    db.update_hand_facts(
        stored.model_copy(update={"hero_bb_won": 40.0}),
        correction_notes="The rake was not taken after all.",
    )
    withdrawn = persist_reconciliation(db, hand.id)
    assert [item.input_name for item in withdrawn.assumption_dependence] == [
        "declared_pot_awards"
    ]
    final = db.fetch_hand(hand.id)
    assert final is not None
    # The rake is gone, so it is named nowhere; the award is still declared, so
    # it is still measured, and answering it is what establishes the figures.
    assert app_module._accounting_is_established(final, withdrawn) is False
    attest_declared_assumptions(db, hand.id)
    final = db.fetch_hand(hand.id)
    assert final is not None
    assert app_module._accounting_is_established(final, withdrawn) is True
    assert app_module._accounting_prompt_math_facts(final, withdrawn)
    db.close()


# ---------------------------------------------------------------------------
# Family H. Mechanisms the round-10 mutation pass found unprotected
# ---------------------------------------------------------------------------


def test_the_writer_never_moves_the_completion_status(tmp_path: Path) -> None:
    """An accounting attestation is not a completion promotion.

    ``derive_completion_status`` does not read the attestation channel, so
    re-deriving the status here could only ever move it for another reason: on a
    ``manual`` row stored with ``completion_status='complete'`` -- a pair
    ``create_hand`` accepts -- one press re-derived ``not_applicable``, the manual
    exemption, and three unrelated blockers vanished with it.
    """
    db = _open_db(tmp_path, "nopromote.db")
    hand = _seed(
        db,
        hero_bb_won=0.0,
        pot_size=80.0,
        award_amounts=(40.0,),
        source_type="manual",
        completion_status="complete",
        evidence={},
    )
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    result = persist_reconciliation(db, hand.id)
    dependence = _named(result)
    before = _readiness(db, hand.id)
    assert before.has(ASSUMPTION_BLOCKER) is True

    assert dependence.code in attest_declared_assumptions(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert stored.completion_status == "complete"
    after = _readiness(db, hand.id)
    assert after.has(ASSUMPTION_BLOCKER) is False
    assert after.is_ready is False
    assert set(before.codes()) - set(after.codes()) == {ASSUMPTION_BLOCKER}
    db.close()


def test_an_unbuildable_ledger_still_discloses_its_declared_rake(
    tmp_path: Path,
) -> None:
    """``_declared_chips_taken`` fails closed onto the declaration itself.

    A ledger that refuses to build tells us nothing about how many chips the
    policy takes, so an unbuildable hand with a declared rate must stay disclosed
    rather than be silently cleared. Replacing that branch with a zero passed the
    entire suite before this test existed.
    """
    db = _open_db(tmp_path, "unbuildable.db")
    hand = _seed(db, hero_bb_won=None, pot_size=None)
    assert hand.id is not None
    connection = sqlite3.connect(str(tmp_path / "unbuildable.db"))
    connection.execute("DELETE FROM hand_players WHERE hand_id = ?", (hand.id,))
    connection.commit()
    connection.close()
    with pytest.raises(LedgerError):
        reconcile_persisted_hand(db, hand.id)

    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="settled", rake_rate=0.5)
    )
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    evidence = parse_completion_evidence(stored.completion_evidence)
    assert DECLARED_RAKE_CODE in evidence.declared_settlement_codes
    assert DECLARED_RAKE_CODE not in evidence.warning_codes
    db.close()


def test_the_assumption_blocker_carries_the_measurement_it_names(
    tmp_path: Path,
) -> None:
    """``detail`` is the only place the code and the chip movement reach the screen."""
    db = _open_db(tmp_path, "detail.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    dependence = _named(persist_reconciliation(db, hand.id))
    attest_declared_assumptions(db, hand.id, only="declared_pot_awards")
    blocker = next(
        item for item in _readiness(db, hand.id).blockers if item.code == ASSUMPTION_BLOCKER
    )
    assert blocker.detail, "a blocker that states no measurement is a rumour"
    (line,) = blocker.detail
    assert dependence.code in line
    assert dependence.describe() in line
    assert "rake_policy" in blocker.reason
    assert "Confirm this assumption" in blocker.clearing_action
    # The measured code names every figure that moved, including the payout, in
    # a form that round-trips to the float it was measured from.
    assert dependence.code.endswith("rake+40.0|net-40.0|payout+40.0|hero-40.0")
    db.close()


def test_the_blocker_order_matches_the_documented_table() -> None:
    """PLAN.md publishes this order; a silent reshuffle changes what an operator reads first."""
    assert BLOCKER_ORDER == (
        "COMPLETION_NOT_COMPLETE",
        "COMPLETION_EVIDENCE_MISSING",
        "INVALID_HERO_OR_BOARD_CARDS",
        "UNREADABLE_HAND_COLUMNS",
        "UNSUPPORTED_TABLE_LAYOUT",
        "ACCOUNTING_NOT_AUTHORITATIVE",
        "ACCOUNTING_ASSUMPTION_DEPENDENT",
        "OPEN_DEBUGGING_ISSUE",
        "UNRESOLVED_SOURCE_WARNING",
        "STALE_COACHING_EVIDENCE",
        "STALE_SOLVER_EVIDENCE",
        "USER_CONFIRMATION_MISSING",
    )


def test_a_neutral_pass_that_cannot_be_built_is_the_strongest_dependence(
    tmp_path: Path,
) -> None:
    """``_is_dependent`` fails closed on an unbuildable neutralisation.

    Removing the declaration left an impossible hand, which is the strongest
    possible form of "the reconciliation rested on it". Inverting that branch
    passed the entire suite before this test existed.
    """
    db = _open_db(tmp_path, "failclosed.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0, award_amounts=(40.0,))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    persist_reconciliation(db, hand.id)
    records = hand_accounting._load_hand_records(db, hand.id)
    baseline = hand_accounting._cross_check(records, records.declaration)
    assert hand_accounting._is_dependent(records, baseline, None) is True
    assert hand_accounting._is_dependent(records, baseline, baseline) is False
    db.close()


def test_the_joint_fallback_still_names_a_dependence_no_half_can_explain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defence in depth, pinned by injection rather than by a hand shape.

    No hand shape is known to reach the joint branch -- rake and dead money
    compose additively, so a pair that moves chips has a half that moves chips.
    The branch exists because its absence would silently return ``()``, and this
    test drives it directly: both single-input neutralisations are made to look
    harmless while the whole-policy one is not.
    """
    db = _open_db(tmp_path, "joint.db")
    hand = _seed(db, hero_bb_won=None, pot_size=None, award_amounts=(None,))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(
            hand_id=hand.id, status="reconciled", rake_rate=0.5, dead_money=10.0
        )
    )
    persist_reconciliation(db, hand.id)

    real_is_dependent = hand_accounting._is_dependent
    calls: list[int] = []

    def _only_the_pair_is_dependent(records, baseline, neutralised):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) == 1:  # the whole-policy pass
            return real_is_dependent(records, baseline, neutralised)
        return False  # each half, on its own, looks harmless

    monkeypatch.setattr(hand_accounting, "_is_dependent", _only_the_pair_is_dependent)
    result = reconcile_persisted_hand(db, hand.id)
    (dependence,) = result.assumption_dependence
    assert dependence.input_name == hand_accounting.JOINT_INPUT
    assert dependence.deltas, "the joint measurement still carries its chip movement"
    db.close()
