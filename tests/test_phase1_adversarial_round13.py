"""Round-13 regressions: the manual exemption is evidence, not a pair of strings.

Round 13 (the fourth round of this workflow) found one family in two costumes and
four defects around it:

* **The exemption could be claimed by writing two strings.** Import refuses a
  payload that declares ``source_type: manual`` while carrying READABLE
  reconstruction evidence, and refuses the reverse pair outright -- but the READER
  accepted a hand-edited row claiming ``manual`` with the evidence still attached,
  so one UPDATE statement walked a blocked CV hand out of the dependence rule and
  every other reconstructed-hand blocker. And an import payload claiming
  ``manual`` with the evidence blanked (or its ``evidence_version`` bumped past
  what this build reads) was byte-identical to a genuine manual export, landed
  ``reviewed`` straight from JSON, and was exempt from every blocker at once.
  The repair is one argument applied everywhere the exemption is consulted: a
  hand that was not entered in this database may not claim an exemption that
  rests on being entered here. The reader reaches the same verdict import does; a
  nonzero ``evidence_version`` -- readable or not -- is a reconstruction claim;
  no imported hand lands ``reviewed`` from a payload; and every imported hand
  owes this operator's explicit confirmation, whatever ``source_type`` it
  declares.
* **An evidence write could promote a hand INTO the exemption.**
  ``update_hand_completion`` re-derived ``completion_status`` from
  ``source_type``, which returns ``not_applicable`` for any manual row, so one
  press of the generic Acknowledge walked a hand-edited ``('manual','complete')``
  row into the exemption and dropped its blockers with it.
* **The pipeline's code channels were writable through the evidence blob.**
  ``update_hand_completion`` trusted a caller-supplied blob for
  ``rejection_codes`` and ``warning_codes``, so a blob with a rejection removed
  promoted the hand the pipeline had refused. Both channels are now preserved
  from the stored row; a caller can add codes (which only ever demote) and can
  never remove one.
* **The attestation fingerprint did not bind the action line.** Two seats
  committing 40 each and four seats committing 20 each produced byte-identical
  dependence codes -- the contributions cancel out of every measured delta -- so
  an attestation survived a rewrite that changed the derived hero result. The
  settled contribution vector and the hero identity are now context terms.
* **Nine clearing actions named a deletion no control performed.**
  ``NEW_RECONSTRUCTION_STEPS`` told the operator to "delete this one from the
  session's hand list" while the running app had no reachable delete-hand
  control at all. The control now exists on every hand row.

Plus the round-4 mutation finding: the dependence rule's non-reconciling-baseline
short-circuit had no killing test.
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
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import export_session, import_session
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
from poker_tracker.services.study_readiness import (
    evaluate_study_readiness,
    is_reconstructed_hand,
)

ASSUMPTION_BLOCKER = "ACCOUNTING_ASSUMPTION_DEPENDENT"
CONFIRMATION_BLOCKER = "USER_CONFIRMATION_MISSING"


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


def _open_db(tmp_path: Path, name: str = "round13.db") -> PokerDatabase:
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
    source_type: str = "cv_import",
    completion_status: str = "complete",
    evidence: dict[str, object] | None = None,
    session_name: str = "Round 13",
) -> Hand:
    """``seats`` seats commit ``bet`` each on the river; ``winners`` share pot 0."""
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
                player_key=key,
                player_name=key.capitalize(),
                amount=None,
                entry_order=order,
            )
            for order, key in enumerate(winners, start=1)
        ],
    )
    return hand


def _readiness(db: PokerDatabase, hand_id: int, *, confirmed: bool = True):
    stored = db.fetch_hand(hand_id)
    assert stored is not None
    try:
        accounting = reconcile_persisted_hand(db, hand_id)
        accounting_error = None
    except LedgerError as exc:
        # What every readiness surface in app.py does with a ledger refusal.
        accounting, accounting_error = None, str(exc)
    return evaluate_study_readiness(
        stored,
        accounting=accounting,
        accounting_error=accounting_error,
        user_confirmed=confirmed,
    )


def _launder_to_manual(
    db_path: Path,
    hand_id: int,
    *,
    completion_status: str = "not_applicable",
    evidence_version: int | None = None,
) -> None:
    """The one UPDATE statement of the round-13 finding, against the raw file."""
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT completion_evidence FROM hands WHERE id = ?", (hand_id,)
        ).fetchone()
        blob = json.loads(row[0]) if row and row[0] else {}
        blob.pop(IMPORTED_HAND_KEY, None)
        if evidence_version is not None:
            blob["evidence_version"] = evidence_version
        connection.execute(
            "UPDATE hands SET source_type = 'manual', completion_status = ?, "
            "completion_evidence = ? WHERE id = ?",
            (completion_status, json.dumps(blob), hand_id),
        )
        connection.commit()
    finally:
        connection.close()


def _forged_manual_payload(
    *, completion_evidence: object, completion_status: str = "not_applicable"
) -> dict[str, object]:
    """The round-13 payload: a reconstructed hand relabelled as somebody's own entry.

    The strong variant of the finding: a fold-out, so pot 0 has exactly one
    eligible seat, the declared award is measured as forced and silent, there is
    no rake and no dead money, and nothing is left for the assumption blocker to
    say. Every remaining reconstructed-hand blocker is bypassed by the two
    relabelled strings, so pre-repair the hand landed ``reviewed`` and
    study-ready with an EMPTY blocker tuple before any click at all.
    """
    return {
        "export_version": 5,
        "session": {"name": "Forged manual", "date_played": "2026-01-01"},
        "hands": [
            {
                "hand": {
                    "hand_number": 1,
                    "table_size": 6,
                    "hero_cards": "Ah Qs",
                    "board_cards": "",
                    "pot_size": None,
                    "hero_bb_won": None,
                    "source_type": "manual",
                    "completion_status": completion_status,
                    "completion_evidence": completion_evidence,
                    "review_status": "reviewed",
                    "tags": [],
                },
                "players": [
                    {
                        "player_key": "hero",
                        "player_name": "Hero",
                        "is_hero": True,
                        "starting_stack": 1000,
                    },
                    {
                        "player_key": "villain",
                        "player_name": "Villain",
                        "is_hero": False,
                        "starting_stack": 1000,
                    },
                ],
                "actions": [
                    {
                        "player_key": "hero",
                        "player_name": "Hero",
                        "street": "preflop",
                        "action_index": 1,
                        "action_type": "bet",
                        "amount": 40.0,
                        "amount_semantics": "incremental",
                    },
                    {
                        "player_key": "villain",
                        "player_name": "Villain",
                        "street": "preflop",
                        "action_index": 2,
                        "action_type": "call",
                        "amount": 40.0,
                        "amount_semantics": "incremental",
                    },
                    {
                        "player_key": "hero",
                        "player_name": "Hero",
                        "street": "flop",
                        "action_index": 1,
                        "action_type": "bet",
                        "amount": 20.0,
                        "amount_semantics": "incremental",
                    },
                    {
                        "player_key": "villain",
                        "player_name": "Villain",
                        "street": "flop",
                        "action_index": 2,
                        "action_type": "fold",
                        "amount": None,
                        "amount_semantics": "unknown",
                    },
                ],
                "settlement": {"status": "reconciled"},
                "settlement_entries": [
                    {
                        "entry_type": "award",
                        "pot_index": 0,
                        "player_key": "hero",
                        "player_name": "Hero",
                        "amount": None,
                        "entry_order": 1,
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Family A. The reader reaches the same verdict import does: a manual claim
# carrying a reconstruction claim is a reconstructed hand
# ---------------------------------------------------------------------------


def _seed_blocked_cv_hand(db: PokerDatabase) -> int:
    """A CV hand whose recorded hero result is true only under a declared rake."""
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    persist_reconciliation(db, hand.id)
    readiness = _readiness(db, hand.id)
    assert readiness.has(ASSUMPTION_BLOCKER) is True
    assert readiness.is_ready is False
    return hand.id


def test_a_hand_edited_manual_relabel_keeps_every_reconstructed_blocker(
    tmp_path: Path,
) -> None:
    """The round-13 HIGH, verbatim: one UPDATE against the database file.

    Pre-repair: the row read back as ``('manual', 'not_applicable')`` with its
    reconstruction evidence still attached, ``is_reconstructed_hand`` said
    False, every reconstructed-hand blocker stopped being emitted at once, the
    measured dependence was simply not acted on, and ``update_hand_status``
    accepted the ``reviewed`` promotion. Import refused the same row verbatim.
    """
    path = tmp_path / "launder.db"
    db = PokerDatabase(str(path))
    db.init_db()
    hand_id = _seed_blocked_cv_hand(db)
    db.close()

    _launder_to_manual(path, hand_id)

    db = PokerDatabase(str(path))
    laundered = db.fetch_hand(hand_id)
    assert laundered is not None
    # The reader reaches import's verdict: evidence carrying a reconstruction
    # claim outranks the two relabelled strings.
    assert is_reconstructed_hand(laundered) is True
    readiness = _readiness(db, hand_id)
    assert readiness.has(ASSUMPTION_BLOCKER) is True
    assert readiness.is_ready is False
    with pytest.raises(ValueError):
        db.update_hand_status(hand_id, "reviewed")
    db.close()


@pytest.mark.parametrize(
    ("completion_status", "evidence_version"),
    [
        ("complete", None),  # keep the stored pair's other half instead
        ("not_applicable", 2),  # a version this build cannot read is still a claim
        ("uncertain", None),
    ],
)
def test_the_relabel_is_normalised_for_every_pair_and_version_shape(
    tmp_path: Path, completion_status: str, evidence_version: int | None
) -> None:
    """Three more instances of the same family, constructed rather than found."""
    path = tmp_path / f"launder_{completion_status}_{evidence_version}.db"
    db = PokerDatabase(str(path))
    db.init_db()
    hand_id = _seed_blocked_cv_hand(db)
    db.close()

    _launder_to_manual(
        path, hand_id, completion_status=completion_status, evidence_version=evidence_version
    )

    db = PokerDatabase(str(path))
    laundered = db.fetch_hand(hand_id)
    assert laundered is not None
    assert is_reconstructed_hand(laundered) is True
    assert _readiness(db, hand_id).is_ready is False
    with pytest.raises(ValueError):
        db.update_hand_status(hand_id, "reviewed")
    db.close()


# ---------------------------------------------------------------------------
# Family B. An import payload claiming `manual` cannot bypass the blockers
# ---------------------------------------------------------------------------


def test_a_manual_payload_with_an_unreadable_evidence_version_is_refused(
    tmp_path: Path,
) -> None:
    """The round-13 CRITICAL's worse variant.

    Pre-repair: bumping ``evidence_version`` to 2 defeated the manual-payload
    refusal (which checked ``is_known``, 1..1) while KEEPING the pipeline's
    ``board_unreadable`` rejection in the stored row -- the hand landed
    ``reviewed``, study-ready, with the pipeline's refusal sitting in its own
    completion evidence. A nonzero version is a reconstruction claim whether or
    not this build can read it.
    """
    db = _open_db(tmp_path, "forged_v2.db")
    payload = _forged_manual_payload(
        completion_evidence=_clean_evidence(
            evidence_version=2, rejection_codes=["board_unreadable"]
        )
    )
    with pytest.raises(ValueError, match="reconstruction completion evidence"):
        import_session(db, payload)
    assert db.fetch_sessions() == []
    db.close()


def test_a_manual_payload_with_blank_evidence_stays_blocked_and_unpromoted(
    tmp_path: Path,
) -> None:
    """The round-13 CRITICAL's primary variant.

    A payload declaring ``source_type: manual`` with blank evidence is
    byte-identical to a genuine manual export, so it cannot be refused. What it
    cannot claim is having been entered here: it lands unable to declare itself
    ``reviewed``, and it owes this operator's explicit confirmation exactly as a
    reconstructed hand does. Pre-repair it landed ``reviewed`` with an empty
    blocker tuple before any click at all.
    """
    db = _open_db(tmp_path, "forged_blank.db")
    imported = import_session(db, _forged_manual_payload(completion_evidence={}))
    assert imported.id is not None
    landed = db.fetch_hands_by_session(imported.id)[0]
    assert landed.id is not None

    # `reviewed` is this database operator's attestation; it cannot travel in a
    # payload -- for any declared source_type.
    assert landed.review_status != "reviewed"

    readiness = _readiness(db, landed.id, confirmed=False)
    assert readiness.has(CONFIRMATION_BLOCKER) is True
    assert readiness.is_ready is False

    # The clearing action is performable: this operator confirms the hand and
    # the blocker clears -- the exemption is decided by who vouched, not by JSON.
    assert _readiness(db, landed.id, confirmed=True).has(CONFIRMATION_BLOCKER) is False
    db.close()


def test_a_genuine_manual_round_trip_gets_the_same_honest_treatment(
    tmp_path: Path,
) -> None:
    """The control: the rule is about provenance, so a genuine export hits it too.

    A real manual hand exported from one database and imported into another is
    indistinguishable from the forgery above, so it lands unpromoted and owing
    confirmation -- and the path back to ``reviewed`` is one tick and one save,
    performed by the operator who now vouches for it.
    """
    source = _open_db(tmp_path, "manual_source.db")
    session = source.create_session(Session(name="My manual session"))
    assert session.id is not None
    hand = source.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            hero_cards="Ah Qs",
            source_type="manual",
            completion_status="not_applicable",
            review_status="reviewed",
        )
    )
    assert hand.id is not None
    for key in ("hero", "villain"):
        source.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=key.capitalize(),
                is_hero=key == "hero",
                starting_stack=1000,
            )
        )
    # A fold-out, so the declared award is forced by the action line and the
    # imported hand's readiness question is confirmation alone.
    for street, index, key, action_type, amount in (
        ("preflop", 1, "hero", "bet", 40.0),
        ("preflop", 2, "villain", "call", 40.0),
        ("flop", 1, "hero", "bet", 20.0),
        ("flop", 2, "villain", "fold", None),
    ):
        source.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street=street,
                action_index=index,
                player_name=key.capitalize(),
                action_type=action_type,
                amount=amount,
                amount_semantics="incremental" if amount is not None else "unknown",
            )
        )
    source.upsert_hand_settlement(HandSettlement(hand_id=hand.id, status="settled"))
    source.replace_settlement_entries(
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
    persist_reconciliation(source, hand.id)
    payload = export_session(source, session.id)
    source.close()

    target = _open_db(tmp_path, "manual_target.db")
    imported = import_session(target, payload)
    assert imported.id is not None
    landed = target.fetch_hands_by_session(imported.id)[0]
    assert landed.id is not None
    assert landed.review_status != "reviewed"
    assert (
        _readiness(target, landed.id, confirmed=False).has(CONFIRMATION_BLOCKER) is True
    )
    # After this operator confirms, the store accepts the promotion: the manual
    # pair still passes the floor, so the named clearing action really clears it.
    assert _readiness(target, landed.id, confirmed=True).is_ready is True
    target.update_hand_status(landed.id, "reviewed")
    refreshed = target.fetch_hand(landed.id)
    assert refreshed is not None
    assert refreshed.review_status == "reviewed"
    target.close()


def test_a_manual_hand_entered_here_keeps_its_exemption(tmp_path: Path) -> None:
    """The other control: the workflow the exemption exists for is unchanged."""
    db = _open_db(tmp_path, "entered_here.db")
    session = db.create_session(Session(name="Typed in here"))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="manual",
            completion_status="not_applicable",
        )
    )
    assert hand.id is not None
    readiness = _readiness(db, hand.id, confirmed=False)
    assert readiness.has(CONFIRMATION_BLOCKER) is False
    db.update_hand_status(hand.id, "reviewed")
    refreshed = db.fetch_hand(hand.id)
    assert refreshed is not None
    assert refreshed.review_status == "reviewed"
    db.close()


# ---------------------------------------------------------------------------
# Family C. No evidence write walks a hand INTO the manual exemption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stored_status", ["complete", "uncertain"])
def test_update_hand_completion_never_promotes_into_the_exemption(
    tmp_path: Path, stored_status: str
) -> None:
    """The round-13 MEDIUM: one Acknowledge press was a promotion.

    ``derive_completion_status`` returns ``not_applicable`` for ANY manual row,
    so re-deriving on a hand-edited ``('manual', 'complete')`` pair walked it
    into the exemption: ``requires_assumption_attestation`` flipped to False and
    COMPLETION_NOT_COMPLETE and ACCOUNTING_ASSUMPTION_DEPENDENT vanished on one
    press. The write may record evidence; it may never change which side of the
    exemption the hand is on.
    """
    db = _open_db(tmp_path, f"ack_{stored_status}.db")
    session = db.create_session(Session(name="Manual pair"))
    assert session.id is not None
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="manual",
            completion_status=stored_status,  # type: ignore[arg-type]
        )
    )
    assert hand.id is not None
    before = _readiness(db, hand.id)
    assert before.has("COMPLETION_NOT_COMPLETE") is True

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    db.update_hand_completion(
        hand.id,
        completion_evidence=dump_completion_evidence(
            parse_completion_evidence(stored.completion_evidence)
        ),
        notes="One press of Acknowledge.",
    )

    refreshed = db.fetch_hand(hand.id)
    assert refreshed is not None
    assert refreshed.completion_status == stored_status
    after = _readiness(db, hand.id)
    assert after.has("COMPLETION_NOT_COMPLETE") is True
    db.close()


# ---------------------------------------------------------------------------
# Family D. The pipeline's code channels are not writable through the blob
# ---------------------------------------------------------------------------


def _seed_rejected_hand(db: PokerDatabase) -> int:
    hand = _seed(
        db,
        completion_status="uncertain",
        evidence=_clean_evidence(rejection_codes=["board_unreadable"]),
    )
    assert hand.id is not None
    return hand.id


def test_update_hand_completion_cannot_remove_a_rejection_code(tmp_path: Path) -> None:
    """The round-13 defence-in-depth MEDIUM, verbatim.

    A rejection is the pipeline refusing the hand: ``acknowledge_codes`` refuses
    to accept one and ``derive_completion_status`` checks rejections before the
    acknowledged set. This writer already pinned ``confirmed_assumption_codes``
    and ``partial`` against exactly this shape of laundering; the two pipeline
    code channels are the same kind of claim.
    """
    db = _open_db(tmp_path, "rej.db")
    hand_id = _seed_rejected_hand(db)
    stored = db.fetch_hand(hand_id)
    assert stored is not None
    blob = dump_completion_evidence(parse_completion_evidence(stored.completion_evidence))
    blob["rejection_codes"] = []

    db.update_hand_completion(hand_id, completion_evidence=blob)

    refreshed = db.fetch_hand(hand_id)
    assert refreshed is not None
    evidence = parse_completion_evidence(refreshed.completion_evidence)
    assert "board_unreadable" in evidence.rejection_codes
    assert refreshed.completion_status == "uncertain"
    assert _readiness(db, hand_id).has("COMPLETION_NOT_COMPLETE") is True
    db.close()


def test_update_hand_completion_cannot_remove_or_swap_pipeline_codes(
    tmp_path: Path,
) -> None:
    """Three more instances of the family: strip a warning, swap a rejection,
    and check additions still land (they can only ever demote)."""
    db = _open_db(tmp_path, "codes.db")
    hand = _seed(
        db,
        completion_status="uncertain",
        evidence=_clean_evidence(
            warning_codes=["hero_seat_mismatch"],
            rejection_codes=["board_unreadable"],
        ),
    )
    assert hand.id is not None
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    blob = dump_completion_evidence(parse_completion_evidence(stored.completion_evidence))
    blob["warning_codes"] = []
    blob["rejection_codes"] = ["something_else"]

    db.update_hand_completion(hand.id, completion_evidence=blob)

    refreshed = db.fetch_hand(hand.id)
    assert refreshed is not None
    evidence = parse_completion_evidence(refreshed.completion_evidence)
    assert "hero_seat_mismatch" in evidence.warning_codes
    assert "board_unreadable" in evidence.rejection_codes
    # An addition is allowed: a new code can only ever demote.
    assert "something_else" in evidence.rejection_codes
    db.close()


def test_acknowledging_a_warning_still_promotes_through_the_same_door(
    tmp_path: Path,
) -> None:
    """The control: the one production caller's flow is not broken by the pin."""
    db = _open_db(tmp_path, "ack_flow.db")
    hand = _seed(
        db,
        completion_status="uncertain",
        evidence=_clean_evidence(warning_codes=["hero_seat_mismatch"]),
    )
    assert hand.id is not None
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    from poker_tracker.persistence.completion import acknowledge_codes

    db.update_hand_completion(
        hand.id,
        completion_evidence=dump_completion_evidence(
            acknowledge_codes(
                parse_completion_evidence(stored.completion_evidence),
                ["hero_seat_mismatch"],
            )
        ),
    )
    refreshed = db.fetch_hand(hand.id)
    assert refreshed is not None
    assert refreshed.completion_status == "complete"
    db.close()


# ---------------------------------------------------------------------------
# Family E. The attestation fingerprint binds the settled contribution vector
# ---------------------------------------------------------------------------


def test_the_dependence_code_separates_hands_with_different_contribution_vectors(
    tmp_path: Path,
) -> None:
    """The round-13 LOW: 2 seats x 40 chips and 4 seats x 20 chips shared a code.

    The contributions cancel out of every measured delta -- ``_ledger_deltas``
    measures declared minus neutral -- and the fingerprint digested only the
    declaration texts, the gross pot and the dead money, all identical across
    the two shapes. The derived hero result differs by 20 chips, so an
    attestation to one is not an attestation to the other.
    """
    db = _open_db(tmp_path, "vector.db")
    two_way = _seed(db, seats=2, bet=40.0, session_name="Two seats")
    four_way = _seed(db, seats=4, bet=20.0, session_name="Four seats")
    for hand in (two_way, four_way):
        assert hand.id is not None
        db.upsert_hand_settlement(
            HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
        )

    def _codes(hand_id: int) -> dict[str, str]:
        return {
            item.input_name: item.code
            for item in reconcile_persisted_hand(db, hand_id).assumption_dependence
        }

    assert two_way.id is not None and four_way.id is not None
    first = _codes(two_way.id)
    second = _codes(four_way.id)
    assert set(first) == set(second)
    for input_name, code in first.items():
        assert code != second[input_name], input_name
    db.close()


def test_the_dependence_code_separates_which_seat_is_the_hero(tmp_path: Path) -> None:
    """A second instance: same chips, same declaration, different hero.

    A symmetric chop -- both seats commit 40, both are declared winners, an even
    rake -- moves every measured delta identically whichever seat carries the
    hero flag, so pre-repair the code was byte-identical across the swap. Whose
    result the product reports is part of what the operator attested to.
    """
    db = _open_db(tmp_path, "hero_moves.db")
    hand = _seed(db, seats=2, bet=40.0, winners=("hero", "villain"))
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    before = {
        item.input_name: item.code
        for item in reconcile_persisted_hand(db, hand.id).assumption_dependence
    }

    players = db.fetch_players_by_hand(hand.id)
    # Unset the current hero first so the single-hero rule is never violated.
    for player in sorted(players, key=lambda item: item.player_key != "hero"):
        assert player.id is not None
        db.update_hand_player(
            player.model_copy(update={"is_hero": player.player_key == "villain"})
        )
    after = {
        item.input_name: item.code
        for item in reconcile_persisted_hand(db, hand.id).assumption_dependence
    }
    assert set(before) == set(after)
    assert any(before[name] != after[name] for name in before)
    db.close()


# ---------------------------------------------------------------------------
# Family F. The non-reconciling-baseline short-circuit has a killing test
# ---------------------------------------------------------------------------


def test_the_session_hand_list_performs_the_deletion_the_blockers_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round-13 HIGH: nine clearing actions named a deletion with no control.

    ``NEW_RECONSTRUCTION_STEPS`` tells the operator to delete the superseded
    hand from the session's hand list after importing a rebuilt copy. Pre-repair
    the only ``db.delete_hand`` call site sat in ``show_saved_hands``, which
    nothing invokes: across all pages of the running app the only delete
    controls were 'Delete session' and 'Delete action', so the instruction was
    not performable. This drives the real app to the page the text names and
    performs it.
    """
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import poker_tracker.persistence.db as db_module
    from poker_tracker.ui.navigation import Page

    path = tmp_path / "delete_control.db"
    db = PokerDatabase(str(path))
    db.init_db()
    hand = _seed(
        db,
        completion_status="partial",
        evidence=_clean_evidence(
            partial_start=True, rejection_codes=["board_unreadable"]
        ),
    )
    assert hand.id is not None
    hand_id = hand.id
    session_id = hand.session_id
    db.close()

    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    app = AppTest.from_file(app_path, default_timeout=30).run()
    assert not list(app.exception)
    app.radio[0].set_value(Page.SESSIONS)
    app.run()
    assert not list(app.exception)

    confirm_key = f"session_{session_id}_confirm_delete_{hand_id}"
    delete_key = f"session_{session_id}_delete_{hand_id}"
    confirm = next(item for item in app.checkbox if item.key == confirm_key)
    delete = next(item for item in app.button if item.key == delete_key)
    assert delete.label == "Delete hand"

    confirm.set_value(True)
    app.run()
    next(item for item in app.button if item.key == delete_key).click()
    app.run()
    assert not list(app.exception)

    verifier = PokerDatabase(str(path))
    assert verifier.fetch_hand(hand_id) is None
    verifier.close()


def test_validation_finish_is_the_confirmation_control_for_an_imported_manual_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blocker's clearing action must be performable on the hand it names.

    USER_CONFIRMATION_MISSING fires for every imported hand; finishing Import
    validation is the control that clears it. A manual hand entered here still
    does not require that confirmation.
    """
    import streamlit as st

    from tests.test_study_readiness_ui import _run_validation_editors

    path = tmp_path / "imported_manual_ui.db"
    db = PokerDatabase(str(path))
    db.init_db()
    import_session(db, _forged_manual_payload(completion_evidence={}))
    hand_id = db.fetch_all_hands()[0].id
    assert hand_id is not None
    db.close()

    app = _run_validation_editors(
        path, monkeypatch, hand_id, frames_validated=False
    )
    assert not list(app.exception)
    assert any(
        button.label == "Finish validation — send to Study" for button in app.button
    )
    st.cache_resource.clear()


def test_a_hand_that_does_not_reconcile_measures_no_dependence(tmp_path: Path) -> None:
    """Kills mutant HA11 (round 4's surviving mutant).

    Case 3 of the dependence rule's docstring: a hand that does not reconcile
    under its stored assumptions measures NO dependence -- it is already
    ACCOUNTING_NOT_AUTHORITATIVE, no Confirm-this-assumption control is drawn on
    a broken hand, and no attestation can be recorded against one. Removing the
    ``if not baseline.reconciles: return ()`` gate changed observable behaviour
    (Confirm buttons drawn on a hand whose chips do not balance) with no test
    failing anywhere in the suite.
    """
    db = _open_db(tmp_path, "broken_baseline.db")
    # Recorded pot 999 contradicts the derived 80-chip pot, so the baseline
    # cross-check fails while the declared 50% rake still takes chips.
    hand = _seed(db, hero_bb_won=None, pot_size=999.0)
    assert hand.id is not None
    db.upsert_hand_settlement(
        HandSettlement(hand_id=hand.id, status="reconciled", rake_rate=0.5)
    )
    result = reconcile_persisted_hand(db, hand.id)
    assert result.is_authoritative is False
    assert result.assumption_dependence == ()
    readiness = _readiness(db, hand.id)
    assert readiness.has("ACCOUNTING_NOT_AUTHORITATIVE") is True
    assert readiness.has(ASSUMPTION_BLOCKER) is False
    db.close()
