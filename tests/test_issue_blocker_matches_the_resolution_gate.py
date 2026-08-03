"""OPEN_DEBUGGING_ISSUE must describe the gate ``resolve_hand_issue`` actually applies.

PLAN.md carries a standing rule: "A blocker never names an action the product
cannot perform." The existing guard for it,
``test_every_control_a_clearing_action_names_exists_in_the_app``, proves the
named control is drawn. It cannot prove the writer behind that control will
accept the submission, and that is the gap this file covers: the issue blocker
named "resolve each issue with resolution notes" for all nine offered
categories, while ``resolve_hand_issue`` refuses seven of them until a linked
regression has been observed both failing before the fix and passing after it —
a promotion no control in the app performs.

The assertion is an equivalence rather than two separate checks. Disclosing the
gate on a category the writer does not gate is as wrong as staying silent on one
it does: the first sends an operator off to write a regression for a coaching
complaint that would have closed on a note.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import get_args

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    RELEASE_BLOCKING_ISSUE_TYPES,
    Hand,
    HandIssue,
    HandIssueType,
    Session,
)
from poker_tracker.services.study_readiness import (
    StudyBlocker,
    evaluate_study_readiness,
)

# Every category the flagging control offers, read from the type rather than
# retyped, so a category added to the product is covered by having been added.
OFFERED_ISSUE_TYPES: tuple[str, ...] = tuple(get_args(HandIssueType))


@pytest.fixture
def db() -> Iterator[PokerDatabase]:
    database = PokerDatabase(":memory:")
    database.init_db()
    yield database
    database.close()


def _seed_hand(database: PokerDatabase) -> Hand:
    session = database.create_session(Session(name="Gate"))
    assert session.id is not None
    hand = database.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            source_type="manual",
            completion_status="not_applicable",
            hero_cards="Ah Qs",
        )
    )
    assert hand.id is not None
    return hand


def _issue_blocker(database: PokerDatabase, hand_id: int) -> StudyBlocker:
    hand = database.fetch_hand(hand_id)
    assert hand is not None
    readiness = evaluate_study_readiness(
        hand,
        accounting=None,
        hand_issues=database.fetch_hand_issues(hand_id=hand_id),
    )
    return next(
        blocker
        for blocker in readiness.blockers
        if blocker.code == "OPEN_DEBUGGING_ISSUE"
    )


def _writer_refuses(database: PokerDatabase, issue_id: int) -> str | None:
    try:
        database.resolve_hand_issue(issue_id, resolution_notes="Corrected the hand.")
    except ValueError as exc:
        return str(exc)
    return None


@pytest.mark.parametrize("issue_type", OFFERED_ISSUE_TYPES)
def test_the_blocker_promises_a_closure_the_writer_will_accept(
    db: PokerDatabase, issue_type: str
) -> None:
    """Pre-repair: every release-blocking category fails here.

    The blocker promised a note-only closure and the writer answered "Promote it
    to a regression case first", with no control able to do that.
    """
    hand = _seed_hand(db)
    assert hand.id is not None
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=[issue_type],  # type: ignore[list-item]
            description="Something about this hand is wrong.",
        )
    )
    assert issue.id is not None

    blocker = _issue_blocker(db, hand.id)
    disclosed = "regression" in blocker.clearing_action
    refusal = _writer_refuses(db, issue.id)

    assert disclosed == (refusal is not None), (
        f"{issue_type}: blocker disclosed a regression requirement={disclosed}, "
        f"writer refusal={refusal!r}"
    )
    assert (issue_type in RELEASE_BLOCKING_ISSUE_TYPES) == disclosed


def test_a_row_whose_categories_cannot_be_read_is_gated_and_says_so(
    db: PokerDatabase,
) -> None:
    """Damage salvages to ``other``, which is outside the set; the writer gates it anyway."""
    hand = _seed_hand(db)
    assert hand.id is not None
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["coaching"],
            description="Wording is off.",
        )
    )
    assert issue.id is not None
    db._execute(
        "UPDATE hand_issues SET issue_types = ? WHERE id = ?", ("{not json", issue.id)
    )
    db._commit()

    stored = db.fetch_hand_issues(hand_id=hand.id)[0]
    assert "issue_types" in stored.unreadable_columns

    blocker = _issue_blocker(db, hand.id)
    refusal = _writer_refuses(db, issue.id)

    assert refusal is not None
    assert "regression" in blocker.clearing_action
    assert any("categories unreadable" in item for item in blocker.detail)


def test_the_disclosure_names_a_procedure_that_exists(db: PokerDatabase) -> None:
    """The runbook step the clearing action sends the operator to must be real.

    Naming a section that does not exist would reproduce the defect one level
    out: an operator following the blocker would again find nothing to do.
    """
    from pathlib import Path

    hand = _seed_hand(db)
    assert hand.id is not None
    db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=["accounting"],
            description="The pot does not add up.",
        )
    )
    blocker = _issue_blocker(db, hand.id)

    runbooks = (
        Path(__file__).resolve().parents[1].joinpath("docs/RUNBOOKS.md").read_text()
    )
    assert "docs/RUNBOOKS.md section 12" in blocker.clearing_action
    assert "## 12. Issue-to-regression debugging workflow" in runbooks
    for symbol in ("promote_issue_to_regression", "record_regression_observation"):
        assert symbol in blocker.clearing_action
        assert symbol in runbooks
