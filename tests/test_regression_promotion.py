"""Phase 6: a closed release-blocking issue must have a proven regression.

The rule being defended is narrow and easy to weaken by accident: a regression
counts only when it was observed FAILING for the original defect and PASSING
after the fix. A test that only ever passed proves nothing — it may not exercise
the defect at all — and accepting it would turn the gate into paperwork.
"""

from __future__ import annotations

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Hand,
    HandCorrection,
    HandIssue,
    Session,
)
from poker_tracker.services.regression_promotion import (
    fetch_regression_case,
    is_release_blocking,
    promote_issue_to_regression,
    record_regression_observation,
    regression_summary,
    regressions_for_issue,
    resolution_blocker,
    suggested_kind,
)


@pytest.fixture
def db():
    database = PokerDatabase(":memory:")
    database.init_db()
    yield database
    database.close()


def _issue(db, *, issue_types=("cards",)) -> int:
    session = db.create_session(Session(name="Regression"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1))
    issue = db.create_hand_issue(
        HandIssue(
            hand_id=hand.id,
            issue_types=list(issue_types),
            description="The turn card was read as a 3, it was a 9.",
        )
    )
    return issue.id


# --- The gate ---------------------------------------------------------------


def test_a_release_blocking_issue_cannot_be_closed_without_a_regression(db):
    issue_id = _issue(db)
    with pytest.raises(ValueError, match="needs a permanent regression"):
        db.resolve_hand_issue(issue_id, resolution_notes="fixed it")
    assert db.fetch_hand_issue(issue_id).status == "open"


def test_a_regression_that_only_ever_passed_is_not_enough(db):
    """The failing observation is the one that proves the test touches the bug."""
    issue_id = _issue(db)
    case = promote_issue_to_regression(
        db, issue_id, kind="cropped_frame", fixture_path="tests/t.py::test_nine"
    )
    record_regression_observation(db, case.id, passing_after=True)

    with pytest.raises(ValueError, match="not proven"):
        db.resolve_hand_issue(issue_id, resolution_notes="fixed it")


def test_a_regression_that_only_ever_failed_is_not_enough(db):
    issue_id = _issue(db)
    case = promote_issue_to_regression(
        db, issue_id, kind="cropped_frame", fixture_path="tests/t.py::test_nine"
    )
    record_regression_observation(db, case.id, failing_before=True)

    with pytest.raises(ValueError, match="not proven"):
        db.resolve_hand_issue(issue_id, resolution_notes="fixed it")


def test_both_observations_close_the_issue(db):
    issue_id = _issue(db)
    case = promote_issue_to_regression(
        db, issue_id, kind="cropped_frame", fixture_path="tests/t.py::test_nine"
    )
    record_regression_observation(
        db,
        case.id,
        failing_before=True,
        passing_after=True,
        fixing_commit="1a2b3c4",
        report_path="data/release_reports/release_gate_report.json",
    )

    resolved = db.resolve_hand_issue(issue_id, resolution_notes="OCR scale repaired.")
    assert resolved.status == "resolved"

    stored = fetch_regression_case(db, case.id)
    assert stored.status == "proven"
    assert stored.fixing_commit == "1a2b3c4"
    assert stored.report_path.endswith("release_gate_report.json")


def test_a_non_release_blocking_issue_closes_without_one(db):
    """Gating a coaching-wording complaint would push everything into "other"."""
    issue_id = _issue(db, issue_types=("coaching",))
    assert resolution_blocker(db, issue_id) is None
    assert db.resolve_hand_issue(issue_id, resolution_notes="Reworded.").status == (
        "resolved"
    )


def test_one_proven_case_is_enough_among_several(db):
    issue_id = _issue(db)
    unproven = promote_issue_to_regression(
        db, issue_id, kind="cached_state", fixture_path="tests/a.py::test_a"
    )
    proven = promote_issue_to_regression(
        db, issue_id, kind="cropped_frame", fixture_path="tests/b.py::test_b"
    )
    record_regression_observation(db, proven.id, failing_before=True, passing_after=True)

    assert resolution_blocker(db, issue_id) is None
    assert len(regressions_for_issue(db, issue_id)) == 2
    assert fetch_regression_case(db, unproven.id).status == "proposed"


# --- Observations are historical facts --------------------------------------


def test_observations_accumulate_and_cannot_be_unset(db):
    issue_id = _issue(db)
    case = promote_issue_to_regression(
        db, issue_id, kind="cached_state", fixture_path="tests/t.py::t"
    )
    record_regression_observation(db, case.id, failing_before=True)
    assert fetch_regression_case(db, case.id).status == "failing"

    # Passing False must not erase the recorded failure: it happened.
    record_regression_observation(db, case.id, failing_before=False, passing_after=True)
    stored = fetch_regression_case(db, case.id)
    assert stored.failing_before is True
    assert stored.passing_after is True
    assert stored.status == "proven"


# --- Fixture selection ------------------------------------------------------


@pytest.mark.parametrize(
    ("issue_types", "expected"),
    [
        (["accounting"], "cached_state"),
        (["cards"], "cropped_frame"),
        (["hand_boundary"], "full_video"),
        # The most expensive fixture wins: the cheap one cannot reproduce the
        # boundary half of a hand flagged for both.
        (["cards", "hand_boundary"], "full_video"),
        (["accounting", "cards"], "cropped_frame"),
    ],
)
def test_suggested_fixture_matches_the_layer_the_bug_lives_in(issue_types, expected):
    assert suggested_kind(issue_types) == expected


def test_release_blocking_classification():
    assert is_release_blocking(["cards"]) is True
    assert is_release_blocking(["coaching"]) is False
    assert is_release_blocking(["other"]) is False
    # One blocking type among several still blocks.
    assert is_release_blocking(["coaching", "accounting"]) is True


# --- Validation -------------------------------------------------------------


def test_an_unknown_fixture_kind_is_rejected(db):
    issue_id = _issue(db)
    with pytest.raises(ValueError, match="Unsupported regression kind"):
        promote_issue_to_regression(
            db, issue_id, kind="vibes", fixture_path="tests/t.py::t"
        )


def test_a_regression_needs_a_fixture_path(db):
    issue_id = _issue(db)
    with pytest.raises(ValueError, match="needs the path"):
        promote_issue_to_regression(db, issue_id, kind="cached_state", fixture_path="  ")


def test_promoting_an_unknown_issue_is_rejected(db):
    with pytest.raises(ValueError, match="was not found"):
        promote_issue_to_regression(
            db, 9999, kind="cached_state", fixture_path="tests/t.py::t"
        )


# --- The retained linkage ---------------------------------------------------


def test_summary_carries_the_linkage_plan_asks_to_retain(db):
    issue_id = _issue(db)
    hand_id = db.fetch_hand_issue(issue_id).hand_id
    correction = db.create_hand_correction(
        HandCorrection(
            hand_id=hand_id,
            correction_type="hand_facts",
            before_state={"board_cards": "2c 7d 3h"},
            after_state={"board_cards": "2c 7d 9h"},
            notes="Turn card re-read from the frame.",
        )
    )
    case = promote_issue_to_regression(
        db,
        issue_id,
        kind="cropped_frame",
        fixture_path="tests/t.py::test_nine",
        correction_id=correction.id,
    )
    record_regression_observation(
        db,
        case.id,
        failing_before=True,
        passing_after=True,
        fixing_commit="1a2b3c4",
        report_path="reports/x.json",
    )
    summary = regression_summary(db, issue_id)

    assert summary["issue_id"] == issue_id
    assert summary["release_blocking"] is True
    assert summary["blocker"] is None
    entry = summary["cases"][0]
    assert entry["proven"] is True
    assert entry["fixing_commit"] == "1a2b3c4"
    assert entry["report_path"] == "reports/x.json"
    assert entry["fixture_path"] == "tests/t.py::test_nine"
    # The correction that fixed the hand is linked to the regression.
    assert entry["correction_id"] == correction.id


def test_deleting_the_issue_removes_its_regression_rows(db):
    """A cascade gap would leave orphaned cases pointing at nothing."""
    issue_id = _issue(db)
    promote_issue_to_regression(
        db, issue_id, kind="cached_state", fixture_path="tests/t.py::t"
    )
    hand_id = db.fetch_hand_issue(issue_id).hand_id
    db._execute("DELETE FROM hands WHERE id = ?", (hand_id,))
    db._commit()
    assert regressions_for_issue(db, issue_id) == []
