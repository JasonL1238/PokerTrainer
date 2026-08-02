"""A damaged JSON blob column must degrade like a damaged scalar, in every table.

Every scalar column the reader cannot validate is named on the fetched model's
``unreadable_columns``, recorded with its stored text under
``UNREADABLE_HAND_COLUMNS_KEY``, and forces ``review_status='needs_correction'``
-- the reader's stated contract that "degradation can only ever ADD
study-readiness blockers".

The two JSON blob columns were outside that contract. ``completion_evidence`` is
the worst case, because it is the CHANNEL the markers travel in: its own damage
destroyed the only record that could have reported it, so a hand whose evidence
had been truncated kept its ``Reviewed`` / ``Complete`` / confidence-High badges
and stayed inside the Overview "Confirmed result ... from N reviewed hands" KPI
-- a status ``update_hand_status`` re-derives from the same evidence and refuses
to issue. ``tags`` was the same silence one column over.

``solver_runs.spot`` is the same silence one table over: a run whose spot could
not be read came back ``completed`` -- the one status study evidence is granted
on -- holding an empty dict.

These tests are keyed on the MODEL's blob columns, not on the two hands columns
that exist today, so a blob column added later cannot re-open the hole.
"""

from __future__ import annotations

import pytest

from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    UNREADABLE_HAND_COLUMNS_KEY,
    CompletionEvidence,
    dump_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase, _blob_columns
from poker_tracker.persistence.models import (
    Hand,
    HandIssue,
    HandSettlement,
    Session,
    SolverRun,
)
from poker_tracker.services.study_readiness import evaluate_study_readiness


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def _reviewed_reconstructed_hand(db: PokerDatabase, session_id: int) -> Hand:
    """A hand carrying every badge the damaged read must not keep silently."""
    evidence = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="showdown",
            boundary_confidence=0.95,
        )
    )
    hand = db.create_hand(
        Hand(
            session_id=session_id,
            hand_number=1,
            source_type="cv_import",
            completion_status="complete",
            completion_evidence=evidence,
            review_status="reviewed",
            confidence_score=0.9,
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c 5h 9d",
            hero_bb_won=10.0,
            tags=["MISSED_VALUE"],
        )
    )
    assert hand.id is not None
    assert hand.review_status == "reviewed"
    assert hand.completion_status == "complete"
    return hand


def _damage(db: PokerDatabase, hand_id: int, column: str, text: str) -> None:
    db._execute(
        f"UPDATE hands SET {column} = ? WHERE id = ?",  # noqa: S608
        (text, hand_id),
    )
    db._commit()


def _blockers(hand: Hand) -> dict[str, tuple[str, ...]]:
    readiness = evaluate_study_readiness(hand, accounting=None)
    return {blocker.code: blocker.detail for blocker in readiness.blockers}


@pytest.mark.parametrize("column", [name for name, _ in _blob_columns(Hand)])
def test_every_json_blob_column_degrades_like_a_scalar(column: str) -> None:
    """The family: any blob column of the hands row, damaged, must be reported.

    Parametrised off ``_blob_columns(Hand)``, which is derived from
    ``Hand.model_fields``, so a blob column added to the model later is covered
    here without anyone editing this test.
    """
    db = make_db()
    session = db.create_session(Session(name="Damaged blob"))
    hand = _reviewed_reconstructed_hand(db, session.id)
    _damage(db, hand.id, column, "{not valid json")

    saved = db.fetch_hand(hand.id)

    assert saved is not None
    assert column in saved.unreadable_columns
    assert saved.review_status == "needs_correction"
    recorded = saved.completion_evidence[UNREADABLE_HAND_COLUMNS_KEY]
    assert isinstance(recorded, dict)
    assert "{not valid json" in str(recorded[column])
    detail = _blockers(saved)["UNREADABLE_HAND_COLUMNS"]
    assert any(column in line and "{not valid json" in line for line in detail)
    db.close()


def test_damaged_evidence_no_longer_keeps_a_status_the_store_would_refuse() -> None:
    """The reported failure, end to end.

    Before the repair the row came back ``reviewed`` / ``complete`` with
    ``unreadable_columns == ()``, while ``update_hand_status`` -- which re-derives
    completion from the same stored evidence -- refused to grant that very status.
    """
    db = make_db()
    session = db.create_session(Session(name="Silently confirmed"))
    hand = _reviewed_reconstructed_hand(db, session.id)
    _damage(db, hand.id, "completion_evidence", "{not json")

    saved = db.fetch_hand(hand.id)

    assert saved is not None
    assert saved.unreadable_columns == ("completion_evidence",)
    assert saved.review_status == "needs_correction"
    # The store's own verdict on the same row, which the library used to
    # contradict.
    with pytest.raises(ValueError):
        db.update_hand_status(hand.id, "reviewed")
    db.close()


def test_damaged_tags_are_reported_rather_than_read_as_no_tags() -> None:
    """An empty tag list is a legitimate state, so the loss needs its own record."""
    db = make_db()
    session = db.create_session(Session(name="Lost tags"))
    hand = _reviewed_reconstructed_hand(db, session.id)
    _damage(db, hand.id, "tags", '["MISSED_VALUE"')

    saved = db.fetch_hand(hand.id)

    assert saved is not None
    assert saved.tags == []
    assert saved.unreadable_columns == ("tags",)
    assert saved.review_status == "needs_correction"
    assert "MISSED_VALUE" in str(
        saved.completion_evidence[UNREADABLE_HAND_COLUMNS_KEY]["tags"]
    )
    db.close()


def test_a_damaged_blob_and_a_damaged_scalar_on_one_row_both_survive() -> None:
    """``_degraded_hand`` must merge into the marker, never replace it.

    The scalar salvage writes ``UNREADABLE_HAND_COLUMNS_KEY`` too; assigning it
    dropped whatever the blob pass had already recorded, so damaging both columns
    reported only the scalar and lost the blob's blocker entirely.
    """
    db = make_db()
    session = db.create_session(Session(name="Both halves"))
    hand = _reviewed_reconstructed_hand(db, session.id)
    _damage(db, hand.id, "tags", "not json at all")
    _damage(db, hand.id, "hero_bb_won", "abc")

    saved = db.fetch_hand(hand.id)

    assert saved is not None
    assert set(saved.unreadable_columns) == {"tags", "hero_bb_won"}
    recorded = saved.completion_evidence[UNREADABLE_HAND_COLUMNS_KEY]
    assert set(recorded) == {"tags", "hero_bb_won"}
    detail = _blockers(saved)["UNREADABLE_HAND_COLUMNS"]
    assert len(detail) == 2
    db.close()


@pytest.mark.parametrize("blank", [True, False])
def test_an_empty_blob_column_is_not_a_degradation(blank: bool) -> None:
    """A hand with no tags and a hand with no evidence are ordinary states."""
    db = make_db()
    session = db.create_session(Session(name="Empty blobs"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    for column, empty in (("tags", "[]"), ("completion_evidence", "{}")):
        value = "" if blank else empty
        db._execute(
            f"UPDATE hands SET {column} = ? WHERE id = ?",  # noqa: S608
            (value, hand.id),
        )
    db._commit()

    saved = db.fetch_hand(hand.id)

    assert saved is not None
    assert saved.unreadable_columns == ()
    assert saved.tags == []
    assert saved.completion_evidence == {}
    assert "UNREADABLE_HAND_COLUMNS" not in _blockers(saved)
    db.close()


def test_a_pathologically_large_damaged_blob_is_recorded_bounded() -> None:
    """A blob column is the one place a hands row can hold megabytes.

    The marker is rendered into a study blocker's detail and travels through
    export and import, so the recorded text is capped; every realistic value is
    still recorded whole, which is what keeps the import restore exact.
    """
    db = make_db()
    session = db.create_session(Session(name="Huge"))
    hand = _reviewed_reconstructed_hand(db, session.id)
    _damage(db, hand.id, "completion_evidence", "[" * 60000)

    saved = db.fetch_hand(hand.id)

    assert saved is not None
    assert saved.unreadable_columns == ("completion_evidence",)
    recorded = str(
        saved.completion_evidence[UNREADABLE_HAND_COLUMNS_KEY]["completion_evidence"]
    )
    assert len(recorded) < 4200
    assert "truncated" in recorded
    db.close()


def test_a_completed_solver_run_with_a_damaged_blob_is_not_offered_as_evidence() -> None:
    """The same silence, one table over.

    ``completed`` is the one solver status study evidence is granted on. A run
    whose spot, ranges or frequencies could not be read used to come back
    ``completed`` holding empty dicts, so a solve was presented as evidence for a
    spot nobody could read. ``stale`` is what the reader already forces on an
    unreadable STATUS, for the same reason.
    """
    db = make_db()
    session = db.create_session(Session(name="Solver"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    run = db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            input_hash="abc",
            status="completed",
            spot={"board": "Qd 7s 2c"},
        )
    )
    assert db.fetch_solver_run(run.id).status == "completed"

    db._execute("UPDATE solver_runs SET spot = ? WHERE id = ?", ("{not json", run.id))
    db._commit()
    damaged = db.fetch_solver_run(run.id)

    assert damaged is not None
    assert damaged.unreadable_columns == ("spot",)
    assert damaged.status == "stale"
    db.close()


def test_a_damaged_issue_evidence_snapshot_is_named() -> None:
    """An issue's evidence snapshot is deliberately immutable; losing it must show.

    ``issue_types`` was already reported, but only because ``min_length=1``
    happens to reject its own empty degraded value. Whether a blob column carries
    such a constraint must not decide whether its loss is reported.
    """
    db = make_db()
    session = db.create_session(Session(name="Issues"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["pot_or_result"],
            description="pot does not reconcile",
            evidence_snapshot={"pot": 12.5},
        )
    )
    db._execute(
        "UPDATE hand_issues SET evidence_snapshot = ? WHERE id = ?",
        ("{not json", issue.id),
    )
    db._commit()

    stored = db.fetch_hand_issues(hand_id=hand.id)[0]

    assert stored.evidence_snapshot == {}
    assert stored.unreadable_columns == ("evidence_snapshot",)
    assert stored.status == "open"
    db.close()


def test_a_damaged_settlement_warnings_blob_blocks_the_accounting_verdict() -> None:
    """``_unreadable_row_issues`` reads the same marker for every record type."""
    db = make_db()
    session = db.create_session(Session(name="Settlement"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, warnings=["rake_guess"]))
    db._execute(
        "UPDATE hand_settlements SET warnings = ? WHERE hand_id = ?",
        ("[not json", hand.id),
    )
    db._commit()

    stored = db.fetch_hand_settlement(hand.id)

    assert stored is not None
    assert stored.warnings == []
    assert stored.unreadable_columns == ("warnings",)
    db.close()


def test_a_damaged_blob_is_not_erased_by_an_export_import_round_trip() -> None:
    """The recorded text must reach the importing database, as a scalar's does.

    ``completion_evidence`` is deliberately not restored -- it is the channel the
    marker travels in -- so the imported hand re-derives the blocker from the
    marker itself rather than from a rewritten column.
    """
    from poker_tracker.persistence.import_export import export_session, import_session

    source = make_db()
    session = source.create_session(Session(name="Round trip"))
    hand = _reviewed_reconstructed_hand(source, session.id)
    _damage(source, hand.id, "tags", "not json at all")

    payload = export_session(source, session.id)
    target = make_db()
    imported = import_session(target, payload)
    landed = target.fetch_hands_by_session(imported.id)[0]

    assert landed.review_status == "needs_correction"
    detail = _blockers(landed)["UNREADABLE_HAND_COLUMNS"]
    assert any("tags" in line for line in detail)
    source.close()
    target.close()
