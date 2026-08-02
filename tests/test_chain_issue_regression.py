"""Phase 14 integration chain: issue save -> regression -> fix -> resolution -> export/import.

Each half of this loop already has a test. What none of them covers is the
whole of it: whether an issue, its frozen evidence, the correction that fixed
the hand, the regression that proves the fix and the resolution all survive a
round trip with the links that make the gate mean something.

The gate is the point. A release-blocking issue may only close on a regression
observed FAILING for the original defect and PASSING after the fix
(``PokerDatabase._regression_blocker``). If that requirement can be lost --
because a seam drops the categories the gate is keyed on, or because a payload
can manufacture the proof -- then a hand the operator flagged as wrong is
re-admitted to study by a file copy, with nothing refusing it. So every
assertion below is about DATA at a seam, never about an absence of exceptions:
an integration test that only proves nothing raised passes just as well when
every seam silently drops what it was carrying.

Two behaviours are pinned that look like losses and are not. The exported
payload deliberately carries NO regression rows: proof is an observation made
in the exporting database about runs that happened there, and the importer
cannot verify it, so it is dropped exactly as an imported resolution, an
acknowledgement and a ``reviewed`` promotion are. The consequence, asserted
here, is that the reopened issue is unresolvable on the other side until
somebody proves a regression again.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import (
    IMPORTED_ISSUE_REOPEN_NOTE,
    export_session,
    import_session,
)
from poker_tracker.persistence.models import (
    RELEASE_BLOCKING_ISSUE_TYPES,
    Action,
    Hand,
    HandCorrection,
    HandIssue,
    HandPlayer,
    Session,
)
from poker_tracker.services.hand_accounting import reconcile_persisted_hand
from poker_tracker.services.regression_promotion import (
    fetch_regression_case,
    promote_issue_to_regression,
    record_regression_observation,
    regression_summary,
    regressions_for_issue,
    resolution_blocker,
)
from poker_tracker.services.study_readiness import (
    StudyReadiness,
    evaluate_study_readiness,
)

FIXTURE = "tests/test_chain_issue_regression.py::test_the_whole_loop_keeps_its_links"


@pytest.fixture
def db() -> Iterator[PokerDatabase]:
    database = PokerDatabase(":memory:")
    database.init_db()
    yield database
    database.close()


def _new_db() -> PokerDatabase:
    database = PokerDatabase(":memory:")
    database.init_db()
    return database


def _seed_hand(database: PokerDatabase, *, name: str = "Chain") -> Hand:
    """One reconstructed hand with a hero and a river bet, as the CV path lands it."""
    session = database.create_session(Session(name=name))
    assert session.id is not None
    hand = database.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="cv_import",
            hero_cards="Ah Qs",
            board_cards="2c 7d 9h Ts 4c",
            pot_size=47.0,
        )
    )
    assert hand.id is not None
    database.create_hand_player(
        HandPlayer(
            hand_id=hand.id, player_key="hero", player_name="Hero", is_hero=True
        )
    )
    database.create_hand_player(
        HandPlayer(hand_id=hand.id, player_key="p2", player_name="Villain")
    )
    database.create_action(
        Action(
            hand_id=hand.id,
            player_key="hero",
            street="river",
            player_name="Hero",
            action_type="bet",
            amount=12.0,
        )
    )
    return hand


def _readiness(
    database: PokerDatabase, hand_id: int, *, user_confirmed: bool = False
) -> StudyReadiness:
    hand = database.fetch_hand(hand_id)
    assert hand is not None
    try:
        accounting = reconcile_persisted_hand(database, hand_id)
        error: str | None = None
    except Exception as exc:  # noqa: BLE001 - the readiness surface's own contract
        accounting, error = None, str(exc)
    return evaluate_study_readiness(
        hand,
        accounting=accounting,
        accounting_error=error,
        hand_issues=database.fetch_hand_issues(hand_id=hand_id),
        coaching_reviews=database.fetch_coaching_reviews_by_hand(hand_id),
        hand_reviews=database.fetch_reviews_by_hand(hand_id),
        solver_runs=database.fetch_solver_runs_by_hand(hand_id),
        user_confirmed=user_confirmed,
    )


def _codes(readiness: StudyReadiness) -> set[str]:
    return {blocker.code for blocker in readiness.blockers}


def _round_trip(source: PokerDatabase, session_id: int) -> tuple[PokerDatabase, Hand]:
    """Export, re-parse as JSON text, import into a fresh database."""
    payload = json.loads(json.dumps(export_session(source, session_id)))
    target = _new_db()
    imported = import_session(target, payload)
    assert imported.id is not None
    hand = target.fetch_hands_by_session(imported.id)[0]
    assert hand.id is not None
    return target, hand


def _damage(database: PokerDatabase, issue_id: int, stored: str) -> None:
    """Write an ``issue_types`` value this build cannot read, as damage would."""
    database._execute(
        "UPDATE hand_issues SET issue_types = ? WHERE id = ?", (stored, issue_id)
    )
    database._commit()


# --- The spine --------------------------------------------------------------


def test_the_whole_loop_keeps_its_links(db: PokerDatabase) -> None:
    """Every seam, asserted on the data it is supposed to be carrying."""

    hand = _seed_hand(db)
    assert hand.id is not None and hand.session_id is not None

    # Seam 1 -- issue save. The evidence is frozen at flag time and the hand is
    # taken out of study until somebody deals with it.
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["pot_or_result"],
            description="The 47 pot was awarded to Villain; the recording shows Hero.",
        )
    )
    assert issue.id is not None
    assert issue.status == "open"
    assert issue.evidence_snapshot["hand"]["pot_size"] == 47.0
    assert issue.evidence_snapshot["hand"]["board_cards"] == "2c 7d 9h Ts 4c"
    assert [player["player_key"] for player in issue.evidence_snapshot["players"]] == [
        "hero",
        "p2",
    ]
    assert db.fetch_hand(hand.id).review_status == "needs_correction"
    assert "OPEN_DEBUGGING_ISSUE" in _codes(_readiness(db, hand.id))

    # Seam 2 -- promotion. The case is born unproven and says so, and it is the
    # gate's answer that says so, not a status column read on its own.
    correction = db.create_hand_correction(
        HandCorrection(
            hand_id=hand.id,
            correction_type="settlement_award_update",
            before_state={"winner": "Villain"},
            after_state={"winner": "Hero"},
            notes="Showdown frame re-read at 00:41.",
        )
    )
    assert correction.id is not None
    case = promote_issue_to_regression(
        db,
        issue.id,
        kind="cached_state",
        fixture_path=FIXTURE,
        correction_id=correction.id,
        notes="Award mapping for a single-winner river.",
    )
    assert case.id is not None
    assert case.issue_id == issue.id
    assert case.correction_id == correction.id
    assert case.kind == "cached_state"
    assert case.fixture_path == FIXTURE
    assert case.status == "proposed"
    assert case.is_proven is False
    assert "not been proven" in (resolution_blocker(db, issue.id) or "")
    with pytest.raises(ValueError, match="not proven"):
        db.resolve_hand_issue(issue.id, resolution_notes="Award mapping repaired.")

    # Seam 3 -- the fix, recorded as two independent observations. Seeing it
    # fail is what proves the test touches the defect; seeing it pass alone
    # would not.
    record_regression_observation(db, case.id, failing_before=True)
    assert fetch_regression_case(db, case.id).status == "failing"
    with pytest.raises(ValueError, match="not proven"):
        db.resolve_hand_issue(issue.id, resolution_notes="Award mapping repaired.")
    record_regression_observation(
        db,
        case.id,
        passing_after=True,
        fixing_commit="1a2b3c4",
        report_path="data/release_reports/release_gate_report.json",
    )
    proven = fetch_regression_case(db, case.id)
    assert proven.status == "proven"
    assert proven.is_proven is True
    assert proven.failing_before is True and proven.passing_after is True
    assert proven.fixing_commit == "1a2b3c4"
    assert proven.report_path.endswith("release_gate_report.json")

    # Seam 4 -- resolution. The gate opens, and the debugging evidence is kept.
    assert resolution_blocker(db, issue.id) is None
    resolved = db.resolve_hand_issue(
        issue.id, resolution_notes="Award mapping repaired; pot goes to Hero."
    )
    assert resolved.status == "resolved"
    assert resolved.resolved_at is not None
    assert resolved.evidence_snapshot["hand"]["pot_size"] == 47.0
    assert "OPEN_DEBUGGING_ISSUE" not in _codes(_readiness(db, hand.id))

    summary = regression_summary(db, issue.id)
    assert summary["release_blocking"] is True
    assert summary["blocker"] is None
    assert summary["cases"][0]["fixture_path"] == FIXTURE
    assert summary["cases"][0]["correction_id"] == correction.id
    assert summary["cases"][0]["proven"] is True

    # Seam 5 -- export. The issue, its frozen evidence and the correction all
    # travel; the regression rows deliberately do not.
    payload = export_session(db, hand.session_id)
    exported_hand = payload["hands"][0]
    (exported_issue,) = exported_hand["issues"]
    assert exported_issue["status"] == "resolved"
    assert exported_issue["issue_types"] == ["pot_or_result"]
    assert exported_issue["resolution_notes"].startswith("Award mapping repaired")
    assert exported_issue["evidence_snapshot"]["hand"]["pot_size"] == 47.0
    (exported_correction,) = exported_hand["corrections"]
    assert exported_correction["after_state"] == {"winner": "Hero"}
    assert "regression" not in json.dumps(payload)

    # Seam 6 -- import. The resolution is reopened because the importing
    # operator has verified nothing, and the prior resolution travels as history
    # rather than being discarded.
    target, imported_hand = _round_trip(db, hand.session_id)
    (landed,) = target.fetch_hand_issues(hand_id=imported_hand.id)
    assert landed.id is not None
    assert landed.status == "open"
    assert landed.resolved_at is None
    assert landed.resolution_notes == ""
    assert landed.issue_types == ["pot_or_result"]
    assert landed.evidence_snapshot["hand"]["pot_size"] == 47.0
    assert "awarded to Villain" in landed.description
    assert IMPORTED_ISSUE_REOPEN_NOTE in landed.description
    assert "pot goes to Hero" in landed.description
    assert [c.after_state for c in target.fetch_hand_corrections(imported_hand.id)] == [
        {"winner": "Hero"}
    ]
    assert imported_hand.review_status != "reviewed"
    assert "OPEN_DEBUGGING_ISSUE" in _codes(
        _readiness(target, imported_hand.id, user_confirmed=True)
    )

    # Seam 7 -- the gate on the other side. The proof did not travel, so the
    # requirement is back, and it is satisfied the same way it was originally.
    assert regressions_for_issue(target, landed.id) == []
    with pytest.raises(ValueError, match="needs a permanent regression"):
        target.resolve_hand_issue(landed.id, resolution_notes="Already fixed upstream.")
    reproven = promote_issue_to_regression(
        target, landed.id, kind="cached_state", fixture_path=FIXTURE
    )
    record_regression_observation(
        target, reproven.id, failing_before=True, passing_after=True
    )
    assert (
        target.resolve_hand_issue(
            landed.id, resolution_notes="Re-proved against the imported records."
        ).status
        == "resolved"
    )
    target.close()


@pytest.mark.parametrize("issue_type", sorted(RELEASE_BLOCKING_ISSUE_TYPES))
def test_every_release_blocking_category_is_still_gated_after_a_round_trip(
    issue_type: str,
) -> None:
    """The gate is keyed on the categories, so each one has to survive the trip."""
    source = _new_db()
    hand = _seed_hand(source, name=f"Gate {issue_type}")
    assert hand.id is not None and hand.session_id is not None
    issue = source.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=[issue_type],
            description=f"Flagged for {issue_type}.",
        )
    )
    assert issue.id is not None
    case = promote_issue_to_regression(
        source, issue.id, kind="cached_state", fixture_path=FIXTURE
    )
    record_regression_observation(
        source, case.id, failing_before=True, passing_after=True
    )
    source.resolve_hand_issue(issue.id, resolution_notes="Fixed and proved.")

    target, imported_hand = _round_trip(source, hand.session_id)
    (landed,) = target.fetch_hand_issues(hand_id=imported_hand.id)
    assert landed.id is not None
    assert landed.issue_types == [issue_type]
    with pytest.raises(ValueError, match="needs a permanent regression"):
        target.resolve_hand_issue(landed.id, resolution_notes="Fixed upstream.")
    source.close()
    target.close()


def test_a_non_blocking_category_still_closes_after_a_round_trip() -> None:
    """The control: widening the gate to every category is also a failure."""
    source = _new_db()
    hand = _seed_hand(source, name="Coaching wording")
    assert hand.id is not None and hand.session_id is not None
    source.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["coaching"],
            description="The lesson repeats itself.",
        )
    )

    target, imported_hand = _round_trip(source, hand.session_id)
    (landed,) = target.fetch_hand_issues(hand_id=imported_hand.id)
    assert landed.id is not None
    assert resolution_blocker(target, landed.id) is None
    assert (
        target.resolve_hand_issue(landed.id, resolution_notes="Reworded.").status
        == "resolved"
    )
    source.close()
    target.close()


@pytest.mark.parametrize(
    ("failing_before", "passing_after"),
    [(True, False), (False, True)],
)
def test_half_a_proof_does_not_close_the_reopened_issue(
    failing_before: bool, passing_after: bool
) -> None:
    """The fail-before/pass-after pair has to be re-established, not just a row."""
    source = _new_db()
    hand = _seed_hand(source, name="Half proof")
    assert hand.id is not None and hand.session_id is not None
    source.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["accounting"],
            description="Chips do not conserve across the river.",
        )
    )

    target, imported_hand = _round_trip(source, hand.session_id)
    (landed,) = target.fetch_hand_issues(hand_id=imported_hand.id)
    assert landed.id is not None
    case = promote_issue_to_regression(
        target, landed.id, kind="cached_state", fixture_path=FIXTURE
    )
    record_regression_observation(
        target, case.id, failing_before=failing_before, passing_after=passing_after
    )

    with pytest.raises(ValueError, match="not proven"):
        target.resolve_hand_issue(landed.id, resolution_notes="Close enough.")
    source.close()
    target.close()


def test_a_payload_cannot_carry_its_own_proof() -> None:
    """A fabricated regression in the file must not satisfy the gate it is under.

    The exporter emits no regression rows, so a hand-written ``regression_cases``
    block is the obvious next forgery: it would make the proof travel in the same
    file as the claim it is supposed to check.
    """
    source = _new_db()
    hand = _seed_hand(source, name="Forged proof")
    assert hand.id is not None and hand.session_id is not None
    source.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["cards"],
            description="The turn read as a 3, it was a 9.",
        )
    )
    payload: dict[str, Any] = export_session(source, hand.session_id)
    forged = {
        "issue_id": 1,
        "kind": "cropped_frame",
        "fixture_path": FIXTURE,
        "status": "proven",
        "failing_before": True,
        "passing_after": True,
        "fixing_commit": "deadbee",
    }
    payload["regression_cases"] = [forged]
    payload["hands"][0]["regression_cases"] = [forged]
    payload["hands"][0]["issues"][0]["regression_cases"] = [forged]

    target = _new_db()
    imported = import_session(target, payload)
    assert imported.id is not None
    imported_hand = target.fetch_hands_by_session(imported.id)[0]
    assert imported_hand.id is not None
    (landed,) = target.fetch_hand_issues(hand_id=imported_hand.id)
    assert landed.id is not None

    assert regressions_for_issue(target, landed.id) == []
    assert (
        target._execute("SELECT COUNT(*) AS n FROM regression_cases").fetchone()["n"]
        == 0
    )
    with pytest.raises(ValueError, match="needs a permanent regression"):
        target.resolve_hand_issue(landed.id, resolution_notes="Proof is in the file.")
    source.close()
    target.close()


# --- The gate must not be clearable by row damage ---------------------------
#
# ``_hand_issue_from_row`` salvages a row it cannot validate and forces its
# status to ``open``, on the stated rule that a degraded row may only ever ADD
# blockers. ``issue_types`` was the exception: the whole column was replaced by
# ``["other"]``, which is precisely the category the regression gate does not
# cover, so damage to the one field the gate is keyed on CLEARED it.


def test_a_partly_damaged_category_list_keeps_its_readable_categories(
    db: PokerDatabase,
) -> None:
    """A category this build does not know must not cost the ones it does.

    ``["cards", "solver_output"]`` is what a database written by a build with one
    more issue category looks like to this one. Pre-repair the row read back as
    ``["other"]``, so ``resolve_hand_issue`` closed a card misread with no
    regression at all.
    """
    hand = _seed_hand(db, name="Partly damaged")
    assert hand.id is not None
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["cards"],
            description="The turn read as a 3, it was a 9.",
        )
    )
    assert issue.id is not None
    _damage(db, issue.id, '["cards", "solver_output"]')

    landed = db.fetch_hand_issue(issue.id)
    assert landed is not None
    assert landed.issue_types == ["cards"]
    assert "issue_types" in landed.unreadable_columns
    assert landed.status == "open"
    with pytest.raises(ValueError, match="needs a permanent regression"):
        db.resolve_hand_issue(issue.id, resolution_notes="Nothing to see here.")


@pytest.mark.parametrize("stored", ["not json at all", "[]", '["nope"]', '"cards"'])
def test_unreadable_categories_are_gated_like_release_blocking_ones(
    db: PokerDatabase, stored: str
) -> None:
    """Damage to the field the gate is keyed on may not be read as ``other``.

    Pre-repair every parametrisation salvaged to ``["other"]``, which is outside
    ``RELEASE_BLOCKING_ISSUE_TYPES``, and the issue closed on a resolution note
    alone -- the gate removed by the same damage that is the reason to distrust
    the row.
    """
    hand = _seed_hand(db, name=f"Damaged {stored}")
    assert hand.id is not None
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["pot_or_result"],
            description="The pot went to the wrong seat.",
        )
    )
    assert issue.id is not None
    _damage(db, issue.id, stored)

    landed = db.fetch_hand_issue(issue.id)
    assert landed is not None
    assert "issue_types" in landed.unreadable_columns

    with pytest.raises(ValueError, match="could not be read"):
        db.resolve_hand_issue(issue.id, resolution_notes="Nothing to see here.")

    # ...and the gate is satisfiable in the ordinary way, so a damaged row is
    # gated rather than stranded. The stored resolution still reads back OPEN,
    # which is the separate, older rule that an unverifiable resolution recorded
    # in a row this build cannot read is not one anybody can act on.
    case = promote_issue_to_regression(
        db, issue.id, kind="cached_state", fixture_path=FIXTURE
    )
    record_regression_observation(db, case.id, failing_before=True, passing_after=True)
    db.resolve_hand_issue(issue.id, resolution_notes="Award mapping repaired.")
    assert (
        db._execute(
            "SELECT status, resolution_notes FROM hand_issues WHERE id = ?",
            (issue.id,),
        ).fetchone()["status"]
        == "resolved"
    )
    reread = db.fetch_hand_issue(issue.id)
    assert reread is not None and reread.status == "open"


def test_the_explanation_agrees_with_the_writer_that_refuses(db: PokerDatabase) -> None:
    """``resolution_blocker`` explains the refusal ``resolve_hand_issue`` makes.

    Two implementations of one rule: an explanation that reported a damaged issue
    as closable while the writer refused it would be read as a bug in the gate.
    """
    hand = _seed_hand(db, name="Explanation")
    assert hand.id is not None
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["stacks"],
            description="Villain's stack jumps by 40 between two frames.",
        )
    )
    assert issue.id is not None
    _damage(db, issue.id, "not json at all")

    blocker = resolution_blocker(db, issue.id)
    assert blocker is not None
    assert "could not be read" in blocker
    summary = regression_summary(db, issue.id)
    assert summary["release_blocking"] is True
    assert summary["blocker"] == blocker


def test_a_damaged_category_list_does_not_export_as_an_ordinary_other_issue() -> None:
    """The downgrade used to outlive the damaged row it came from.

    The exporter emits what the reader returns, so a ``cards`` issue salvaged to
    ``other`` landed in the importing database as a well-formed, ungated
    ``other`` issue -- a hand flagged as wrong, admitted to study on the other
    side by a file copy.
    """
    source = _new_db()
    hand = _seed_hand(source, name="Damaged travel")
    assert hand.id is not None and hand.session_id is not None
    issue = source.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["cards"],
            description="The turn read as a 3, it was a 9.",
        )
    )
    assert issue.id is not None
    _damage(source, issue.id, '["cards", "solver_output"]')

    target, imported_hand = _round_trip(source, hand.session_id)
    (landed,) = target.fetch_hand_issues(hand_id=imported_hand.id)
    assert landed.id is not None
    assert landed.issue_types == ["cards"]
    assert landed.status == "open"
    with pytest.raises(ValueError, match="needs a permanent regression"):
        target.resolve_hand_issue(landed.id, resolution_notes="Fixed upstream.")
    source.close()
    target.close()
