"""Round-14 regressions (workflow round 5): readers, evidence writes, and gates.

Round 14 found no hole in the dependence rule itself. What it found, again, was
enumerated-list decay AROUND the rules — each repaired here as a family, with
several independent instances per family rather than the one shape the
adversary demonstrated:

* **Row readers raised instead of degrading.** ``_hand_from_row`` guarded four
  columns by hand while ``table_size = 99`` in the fifth raised a
  ValidationError out of ``fetch_hands_by_session`` and took the entire
  application down on load; ``_action_from_row`` and its siblings were not
  guarded at all. Every reader now salvages column-by-column through one
  helper (``db._salvaged_row``) driven by the model's own field set, and every
  degradation is conservative: it can only ever add blockers.
* **``update_hand_completion`` took the caller's blob as the base.** Pinning
  three code channels from the stored row left every OTHER field
  caller-writable: a blob manufactured the pipeline's boundary observations and
  promoted an unprovable hand, and a blob that dropped the
  ``imported_from_payload`` stamp walked an imported hand into the manual
  exemption. The merge is now inverted: the stored evidence is the base and the
  caller may only ADD codes.
* **The promotion gate deny-listed values instead of allow-listing them.** Any
  unrecognised ``terminal_event`` counted as observed, and any finite
  ``boundary_confidence`` — including 0.0 and 42.0 — counted as a measurement.
* **Non-finite floats passed the validating boundary.** ``ge=0`` admits ``inf``,
  so a hostile payload landed ``dead_money=Infinity`` and the session export
  stopped being RFC 8259 JSON. ``PersistedModel`` sets ``allow_inf_nan=False``
  once, for every float field of every persisted model, present and future.
* **The raw attestation writer trusted a shape-valid code.** A forged
  ``declared_settlement_dependence:...`` string was recorded, evicted the
  operator's genuine attestation for the same input, and filed an audit row for
  an attestation nobody made. The writer itself now re-measures and refuses a
  code naming no current dependence, failing closed when the measurement cannot
  be taken.
* **The dependence tolerance was unpinned.** Widening ``_FLOAT_TOLERANCE`` from
  1e-9 to 1e-6 survived the whole suite; the constant and a sub-microchip
  behavioural case are now both pinned.

Every test below fails on the pre-repair tree.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    IMPORTED_HAND_KEY,
    UNREADABLE_HAND_COLUMNS_KEY,
    CompletionEvidence,
    derive_completion_status,
    dump_completion_evidence,
    parse_completion_evidence,
    strip_derived_evidence_markers,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandIssue,
    HandPlayer,
    HandSettlement,
    PersistedModel,
    Session,
    SettlementEntry,
)
from poker_tracker.services import hand_accounting
from poker_tracker.services.hand_accounting import (
    attest_assumption,
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import evaluate_study_readiness


def _open_db(tmp_path: Path, name: str = "round14.db") -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


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


def _seed(
    db: PokerDatabase,
    *,
    seats: int = 2,
    bet: float = 40.0,
    hero_bb_won: float | None = None,
    pot_size: float | None = None,
    evidence: dict[str, object] | None = None,
    session_name: str = "Round 14",
) -> Hand:
    session = db.create_session(Session(name=session_name, date_played=date(2026, 2, 1)))
    assert session.id is not None
    keys = ["hero", "villain", "third", "fourth"][:seats]
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
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=0,
                player_key="hero",
                player_name="Hero",
                amount=None,
                entry_order=1,
            )
        ],
    )
    return hand


def _readiness(db: PokerDatabase, hand_id: int, *, user_confirmed: bool = True):
    stored = db.fetch_hand(hand_id)
    assert stored is not None
    try:
        accounting = reconcile_persisted_hand(db, hand_id)
        error = None
    except hand_accounting.LedgerError as exc:  # type: ignore[attr-defined]
        accounting, error = None, str(exc)
    return evaluate_study_readiness(
        stored,
        accounting=accounting,
        accounting_error=error,
        hand_issues=db.fetch_hand_issues(hand_id=hand_id),
        user_confirmed=user_confirmed,
    )


# ---------------------------------------------------------------------------
# Family 1: every row reader degrades instead of raising, conservatively
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("table_size", "99"),
        ("review_status", "'REVIEWED'"),
        ("pot_size", "-1.0"),
        ("confidence_score", "1.7"),
        ("hand_number", "0"),
        ("effective_stack", "-5.0"),
        ("tags", "'[\"not_a_real_tag\"]'"),
        ("created_at", "'not a timestamp'"),
        ("hero_bb_won", "'forty'"),
    ],
)
def test_a_hand_row_this_build_cannot_validate_degrades_and_blocks(
    tmp_path: Path, column: str, value: str
) -> None:
    """Pre-repair: every one of these raised ValidationError out of
    ``fetch_hands_by_session`` at app load and took the whole product down."""
    db = _open_db(tmp_path, f"hand_{column}.db")
    hand = _seed(db)
    assert hand.id is not None
    db._execute(f"UPDATE hands SET {column} = {value} WHERE id = ?", (hand.id,))
    db._commit()

    (fetched,) = db.fetch_hands_by_session(hand.session_id)
    assert fetched.review_status == "needs_correction"
    marker = fetched.completion_evidence.get(UNREADABLE_HAND_COLUMNS_KEY)
    assert isinstance(marker, dict) and column in marker

    readiness = _readiness(db, hand.id)
    assert readiness.is_ready is False
    assert readiness.has("UNREADABLE_HAND_COLUMNS") is True
    blocker = next(
        item for item in readiness.blockers if item.code == "UNREADABLE_HAND_COLUMNS"
    )
    # The stored text is reported, never silently replaced by the fallback.
    assert any(column in line for line in blocker.detail)
    db.close()


def test_the_unreadable_column_marker_is_derived_and_never_persisted(
    tmp_path: Path,
) -> None:
    """The marker appears while the row is unreadable and vanishes when it is
    corrected; no writer may store it (same lifecycle as UNREADABLE_CARDS_KEY)."""
    db = _open_db(tmp_path)
    hand = _seed(db)
    assert hand.id is not None
    db._execute("UPDATE hands SET table_size = 99 WHERE id = ?", (hand.id,))
    db._commit()
    degraded = db.fetch_hand(hand.id)
    assert degraded is not None
    assert UNREADABLE_HAND_COLUMNS_KEY in degraded.completion_evidence
    assert (
        UNREADABLE_HAND_COLUMNS_KEY
        not in strip_derived_evidence_markers(degraded.completion_evidence)
    )
    # Correcting the column clears the marker on the next read.
    db._execute("UPDATE hands SET table_size = 6 WHERE id = ?", (hand.id,))
    db._commit()
    corrected = db.fetch_hand(hand.id)
    assert corrected is not None
    assert UNREADABLE_HAND_COLUMNS_KEY not in corrected.completion_evidence
    db.close()


def test_a_degraded_hero_result_cannot_pass_the_cross_check_silently(
    tmp_path: Path,
) -> None:
    """hero_bb_won this build cannot read degrades to None, which would make the
    hero-result cross-check trivially pass — so the degradation itself must
    block the accounting verdict."""
    db = _open_db(tmp_path)
    hand = _seed(db, hero_bb_won=40.0, pot_size=80.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    assert persist_reconciliation(db, hand.id).is_authoritative is True

    db._execute("UPDATE hands SET hero_bb_won = 'forty' WHERE id = ?", (hand.id,))
    db._commit()
    result = reconcile_persisted_hand(db, hand.id)
    assert result.is_authoritative is False
    assert any("hand row" in issue and "hero_bb_won" in issue for issue in result.issues)
    readiness = _readiness(db, hand.id)
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE") is True
    assert readiness.has("UNREADABLE_HAND_COLUMNS") is True
    db.close()


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("actions", "street", "'moon'"),
        ("actions", "action_type", "'levitate'"),
        ("actions", "amount", "-5.0"),
        ("hand_players", "starting_stack", "-50.0"),
        ("settlement_entries", "entry_type", "'bribe'"),
        ("settlement_entries", "amount", "-10.0"),
        ("settlement_entries", "pot_index", "-2"),
    ],
)
def test_a_sibling_row_this_build_cannot_validate_blocks_the_accounting(
    tmp_path: Path, table: str, column: str, value: str
) -> None:
    """Pre-repair: each of these raised ValidationError out of its fetch. Now
    the row degrades, and the degradation forces the reconciliation off
    authoritative — a default is not an observation."""
    db = _open_db(tmp_path, f"{table}_{column}.db")
    hand = _seed(db, hero_bb_won=40.0, pot_size=80.0)
    assert hand.id is not None
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id))
    assert persist_reconciliation(db, hand.id).is_authoritative is True

    db._execute(f"UPDATE {table} SET {column} = {value} WHERE hand_id = ?", (hand.id,))
    db._commit()
    # The fetch itself must not raise.
    db.fetch_actions_by_hand(hand.id)
    db.fetch_players_by_hand(hand.id)
    db.fetch_settlement_entries(hand.id)
    try:
        result = reconcile_persisted_hand(db, hand.id)
    except hand_accounting.LedgerError:
        # The ledger refused to build over the degraded row, which is the
        # fail-closed branch every readiness surface reports as
        # ACCOUNTING_NOT_AUTHORITATIVE via accounting_error.
        pass
    else:
        assert result.is_authoritative is False
        assert any("could not be fully read" in issue for issue in result.issues)
    readiness = _readiness(db, hand.id)
    assert readiness.is_ready is False
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE") is True
    db.close()


def test_an_unreadable_issue_row_reads_back_open_never_resolved(
    tmp_path: Path,
) -> None:
    db = _open_db(tmp_path)
    hand = _seed(db)
    assert hand.id is not None
    db.create_hand_issue(
        HandIssue(hand_id=hand.id, issue_types=["cards"], description="check later"),
        apply_workflow=False,
    )
    db._execute("UPDATE hand_issues SET status = 'zombie' WHERE hand_id = ?", (hand.id,))
    db._commit()
    (issue,) = db.fetch_hand_issues(hand_id=hand.id)
    assert issue.status == "open"
    assert _readiness(db, hand.id).has("OPEN_DEBUGGING_ISSUE") is True

    # A resolution recorded in a row this build cannot fully read is reopened.
    db._execute(
        "UPDATE hand_issues SET status = 'resolved', resolved_at = 'garbage' "
        "WHERE hand_id = ?",
        (hand.id,),
    )
    db._commit()
    (issue,) = db.fetch_hand_issues(hand_id=hand.id)
    assert issue.status == "open"
    db.close()


def test_an_unreadable_session_row_does_not_take_the_session_list_down(
    tmp_path: Path,
) -> None:
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Good", date_played=date(2026, 2, 1)))
    db._execute(
        "UPDATE sessions SET date_played = 'someday', name = '' WHERE id = ?",
        (session.id,),
    )
    db._commit()
    (fetched,) = db.fetch_sessions()
    assert fetched.name == "(unreadable)"
    db.close()


# ---------------------------------------------------------------------------
# Family 2: update_hand_completion may only add codes — never observations
# ---------------------------------------------------------------------------


_CODE_CHANNELS = {"warning_codes", "rejection_codes", "acknowledged_codes"}


def test_update_hand_completion_cannot_manufacture_boundary_evidence(
    tmp_path: Path,
) -> None:
    """Pre-repair: a caller blob stating boundary_confidence=0.92 and
    terminal_event='showdown' promoted a hand the pipeline never finished
    observing straight to 'complete' and study-ready."""
    db = _open_db(tmp_path)
    hand = _seed(
        db,
        evidence=_clean_evidence(terminal_event="", boundary_confidence=None),
    )
    assert hand.id is not None
    db._execute(
        "UPDATE hands SET completion_status = 'uncertain' WHERE id = ?", (hand.id,)
    )
    db._commit()
    before = db.fetch_hand(hand.id)
    assert before is not None
    assert _readiness(db, hand.id).has("COMPLETION_NOT_COMPLETE") is True

    updated = db.update_hand_completion(
        hand.id, completion_evidence=_clean_evidence()
    )
    assert updated.completion_status == "uncertain"
    assert _readiness(db, hand.id).has("COMPLETION_NOT_COMPLETE") is True
    stored = parse_completion_evidence(updated.completion_evidence)
    assert stored.terminal_event == ""
    assert stored.boundary_confidence is None
    db.close()


def test_update_hand_completion_pins_every_field_that_is_not_a_code_channel(
    tmp_path: Path,
) -> None:
    """The family, not the shape: after a write with a wildly different blob,
    the stored evidence is byte-identical to the previous evidence outside the
    three code channels — for every field that exists or is added later."""
    db = _open_db(tmp_path)
    hand = _seed(db, evidence=_clean_evidence(warning_codes=["pot_not_reconciled"]))
    assert hand.id is not None
    before = db.fetch_hand(hand.id)
    assert before is not None
    previous = dump_completion_evidence(
        parse_completion_evidence(before.completion_evidence)
    )

    hostile = _clean_evidence(
        evidence_version=EVIDENCE_SCHEMA_VERSION + 5,
        partial_start=True,
        partial_end=True,
        terminal_event="chop",
        boundary_confidence=42.0,
        layout_supported=False,
        table_size=2,
        source_frames=["forged.png"],
        pipeline_version="fabricated",
        model_versions={"detector": "v0"},
        first_source_timestamp_s=1.0,
        last_source_timestamp_s=2.0,
        confirmed_assumption_codes=[
            "declared_settlement_dependence:rake_policy:0000000000:rake+1"
        ],
        declared_settlement_codes=["declared_unobserved_chips"],
        acknowledged_codes=["pot_not_reconciled"],
        rejection_codes=["board_unreadable"],
    )
    hostile["some_future_key"] = "forged"
    updated = db.update_hand_completion(hand.id, completion_evidence=hostile)

    after = dump_completion_evidence(
        parse_completion_evidence(updated.completion_evidence)
    )
    for key, value in previous.items():
        if key in _CODE_CHANNELS:
            continue
        assert after[key] == value, key
    assert "some_future_key" not in after
    # The code channels moved only by addition.
    assert after["acknowledged_codes"] == ["pot_not_reconciled"]
    assert after["rejection_codes"] == ["board_unreadable"]
    db.close()


def test_update_hand_completion_cannot_erase_the_imported_stamp(
    tmp_path: Path,
) -> None:
    """Pre-repair: a blob without the ``imported_from_payload`` key flipped
    ``is_imported`` to False and dropped ACCOUNTING_ASSUMPTION_DEPENDENT and
    USER_CONFIRMATION_MISSING together — the manual exemption, claimed by an
    evidence write."""
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Imported", date_played=date(2026, 2, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="manual",
            completion_status="not_applicable",
            completion_evidence={IMPORTED_HAND_KEY: True},
        )
    )
    assert hand.id is not None
    before = db.fetch_hand(hand.id)
    assert before is not None
    assert parse_completion_evidence(before.completion_evidence).is_imported is True

    db.update_hand_completion(hand.id, completion_evidence={})
    after = db.fetch_hand(hand.id)
    assert after is not None
    assert parse_completion_evidence(after.completion_evidence).is_imported is True
    readiness = evaluate_study_readiness(after, accounting=None, user_confirmed=False)
    assert readiness.has("USER_CONFIRMATION_MISSING") is True
    db.close()


# ---------------------------------------------------------------------------
# Family 3: the promotion gate allow-lists observed values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event", ["chop", " ", "PREFLOP", "SHOWDOWN", "fold_win ", "unknown", "", "unobserved"]
)
def test_an_unrecognised_terminal_event_is_not_an_observed_one(event: str) -> None:
    evidence = parse_completion_evidence(_clean_evidence(terminal_event=event))
    assert derive_completion_status(evidence, source_type="cv_import") == "uncertain"


@pytest.mark.parametrize("event", ["showdown", "fold_win", "hero_fold"])
def test_the_three_observed_terminal_events_still_promote(event: str) -> None:
    evidence = parse_completion_evidence(_clean_evidence(terminal_event=event))
    assert derive_completion_status(evidence, source_type="cv_import") == "complete"


@pytest.mark.parametrize("confidence", [0.0, -0.0, -3.0, 1.0001, 42.0])
def test_an_implausible_boundary_confidence_is_not_a_measurement(
    confidence: float,
) -> None:
    evidence = parse_completion_evidence(
        _clean_evidence(boundary_confidence=confidence)
    )
    assert derive_completion_status(evidence, source_type="cv_import") == "uncertain"


@pytest.mark.parametrize("confidence", [1e-9, 0.5, 1.0])
def test_a_plausible_boundary_confidence_still_promotes(confidence: float) -> None:
    evidence = parse_completion_evidence(
        _clean_evidence(boundary_confidence=confidence)
    )
    assert derive_completion_status(evidence, source_type="cv_import") == "complete"


# ---------------------------------------------------------------------------
# Family 4: non-finite floats are refused at the validating boundary
# ---------------------------------------------------------------------------


def test_every_persisted_model_refuses_non_finite_floats() -> None:
    """The family pin: the refusal is model-wide configuration, not a per-field
    constraint list, so a float field added later inherits it."""
    seen = set()
    stack = [PersistedModel]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    assert seen, "no persisted models found"
    for cls in seen:
        assert cls.model_config.get("allow_inf_nan") is False, cls.__name__


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dead_money": float("inf")},
        {"rake_cap": float("inf")},
        {"gross_pot": float("inf")},
        {"rake_amount": float("nan")},
        {"net_pot": float("-inf")},
    ],
)
def test_hand_settlement_refuses_non_finite_declarations(kwargs: dict) -> None:
    """Pre-repair: ``ge=0`` admitted ``inf``; the row landed, the ledger failed
    closed, and the session export emitted a bare Infinity token no RFC 8259
    parser could read."""
    with pytest.raises(ValidationError):
        HandSettlement(hand_id=1, **kwargs)


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (Hand, {"session_id": 1, "hand_number": 1, "hero_bb_won": float("nan")}),
        (Hand, {"session_id": 1, "hand_number": 1, "effective_stack": float("inf")}),
        (
            Action,
            {
                "hand_id": 1,
                "street": "flop",
                "player_name": "V",
                "action_type": "bet",
                "amount": float("inf"),
            },
        ),
        (HandPlayer, {"hand_id": 1, "player_name": "V", "starting_stack": float("inf")}),
        (
            SettlementEntry,
            {
                "hand_id": 1,
                "entry_type": "refund",
                "player_name": "V",
                "amount": float("nan"),
            },
        ),
    ],
)
def test_sibling_models_refuse_non_finite_floats(model: type, kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_a_session_export_stays_strict_json_end_to_end(tmp_path: Path) -> None:
    """The consequence the family test protects: exports parse under a strict
    RFC 8259 reader even after every write path has run."""
    from poker_tracker.persistence.import_export import export_session_json

    db = _open_db(tmp_path)
    hand = _seed(db, hero_bb_won=40.0, pot_size=80.0)
    assert hand.id is not None
    persist_reconciliation(db, hand.id)

    def _refuse(constant: str) -> None:
        raise AssertionError(f"non-RFC token {constant!r} in export")

    out = tmp_path / "export.json"
    export_session_json(db, hand.session_id, out)
    json.loads(out.read_text(encoding="utf-8"), parse_constant=_refuse)
    db.close()


# ---------------------------------------------------------------------------
# Family 5: the raw attestation writer verifies the code it records
# ---------------------------------------------------------------------------


def test_the_raw_writer_refuses_a_forged_code_and_keeps_the_genuine_attestation(
    tmp_path: Path,
) -> None:
    """Pre-repair: the raw writer accepted any shape-valid string, EVICTED the
    genuine same-input attestation, and filed a hand_corrections row for an
    attestation nobody made."""
    db = _open_db(tmp_path)
    hand = _seed(db, hero_bb_won=None, pot_size=None)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    result = persist_reconciliation(db, hand.id)
    (genuine,) = [
        item for item in result.assumption_dependence if item.input_name == "rake_policy"
    ]
    assert attest_assumption(db, hand.id, genuine.code) is True
    corrections_before = len(db.fetch_hand_corrections(hand.id))

    forged = "declared_settlement_dependence:rake_policy:0000000000:rake+1|hero-1"
    assert db.acknowledge_accounting_assumption(hand.id, forged) is False

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    confirmed = parse_completion_evidence(stored.completion_evidence)
    assert genuine.code in confirmed.confirmed_assumption_codes
    assert forged not in confirmed.confirmed_assumption_codes
    assert len(db.fetch_hand_corrections(hand.id)) == corrections_before
    db.close()


def test_the_raw_writer_fails_closed_when_the_measurement_cannot_be_taken(
    tmp_path: Path,
) -> None:
    """A hand whose ledger refuses to build cannot verify any code: refuse,
    write nothing, raise nothing."""
    db = _open_db(tmp_path)
    session = db.create_session(Session(name="Empty", date_played=date(2026, 2, 1)))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            completion_status="uncertain",
            completion_evidence=_clean_evidence(),
        )
    )
    assert hand.id is not None
    code = "declared_settlement_dependence:rake_policy:0000000000:rake+1"
    assert db.acknowledge_accounting_assumption(hand.id, code) is False
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert parse_completion_evidence(stored.completion_evidence).confirmed_assumption_codes == ()
    db.close()


# ---------------------------------------------------------------------------
# Family 6: the dependence tolerance is pinned, by value and by behaviour
# ---------------------------------------------------------------------------


def test_the_dependence_tolerance_is_the_float_noise_floor(tmp_path: Path) -> None:
    """1e-9 is float-representation noise and nothing wider. A declared input
    moving 1e-8 chips — above the noise floor, below the 1e-6 a widened mutant
    would use — must still be measured as a movement, with the figure in the
    code, not collapsed to 'verdict-only'."""
    assert hand_accounting._FLOAT_TOLERANCE == 1e-9

    db = _open_db(tmp_path)
    hand = _seed(db, hero_bb_won=None, pot_size=None)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", dead_money=1e-8)
    )
    result = persist_reconciliation(db, hand.id)
    (dependence,) = [
        item for item in result.assumption_dependence if item.input_name == "dead_money"
    ]
    moved = dict(dependence.deltas)
    assert "gross" in moved, "a 1e-8 chip movement was not measured"
    assert moved["gross"] == pytest.approx(1e-8, abs=1e-10)
    assert "verdict-only" not in dependence.code
    db.close()
