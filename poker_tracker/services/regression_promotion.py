"""Turn a confirmed issue into a permanent regression, and gate its closure.

PLAN.md: "Do not resolve the issue until the new regression fails before the fix
and passes after it," and "Every closed release-blocking issue has a passing
regression."

Before this, resolving an issue meant typing a note. Nothing recorded whether a
test existed, whether it had ever failed for the original defect, or which
change fixed it — so a fix that silently stopped working reopened the same bug
with no alarm, and a "resolved" issue was indistinguishable from an abandoned
one.

The fail-before/pass-after pair is the part that matters. A regression that only
ever passed proves nothing: it may not exercise the defect at all. Recording the
two observations separately, and requiring both, is what makes the test evidence
rather than decoration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import RELEASE_BLOCKING_ISSUE_TYPES

# Which fixture a defect deserves, per PLAN.md. The choice follows the layer the
# bug lives in: reproducing an OCR misread does not need a video, and a boundary
# bug cannot be reproduced without one.
RegressionKind = Literal["cached_state", "cropped_frame", "full_video"]
REGRESSION_KINDS: tuple[RegressionKind, ...] = (
    "cached_state",
    "cropped_frame",
    "full_video",
)



# Which fixture suits which issue category, used to advise rather than enforce:
# an operator who knows the bug is reconstruction-only should not be forced to
# retain a video for it.
SUGGESTED_KIND_BY_ISSUE_TYPE: dict[str, RegressionKind] = {
    "hand_boundary": "full_video",
    "cards": "cropped_frame",
    "stacks": "cropped_frame",
    "players": "cached_state",
    "actions": "cached_state",
    "pot_or_result": "cached_state",
    "accounting": "cached_state",
}


@dataclass(frozen=True)
class RegressionCase:
    id: int | None
    issue_id: int
    correction_id: int | None
    kind: str
    fixture_path: str
    status: str
    failing_before: bool
    passing_after: bool
    fixing_commit: str
    report_path: str
    notes: str

    @property
    def is_proven(self) -> bool:
        """Observed failing for the defect and passing after the fix.

        Both, not either. A regression that only ever passed may not touch the
        defect at all.
        """
        return self.failing_before and self.passing_after


def suggested_kind(issue_types: list[str]) -> RegressionKind:
    """The most expensive fixture any of these issue types needs.

    A hand flagged for both a card misread and a boundary error needs the video:
    the cheaper fixture cannot reproduce the boundary half.
    """
    ranking = {"cached_state": 0, "cropped_frame": 1, "full_video": 2}
    best: RegressionKind = "cached_state"
    for issue_type in issue_types:
        candidate = SUGGESTED_KIND_BY_ISSUE_TYPE.get(issue_type)
        if candidate and ranking[candidate] > ranking[best]:
            best = candidate
    return best


def is_release_blocking(issue_types: list[str]) -> bool:
    return bool(RELEASE_BLOCKING_ISSUE_TYPES.intersection(issue_types))


def _row_to_case(row: Any) -> RegressionCase:
    return RegressionCase(
        id=row["id"],
        issue_id=row["issue_id"],
        correction_id=row["correction_id"],
        kind=row["kind"],
        fixture_path=row["fixture_path"],
        status=row["status"],
        failing_before=bool(row["failing_before"]),
        passing_after=bool(row["passing_after"]),
        fixing_commit=row["fixing_commit"],
        report_path=row["report_path"],
        notes=row["notes"],
    )


def promote_issue_to_regression(
    db: PokerDatabase,
    issue_id: int,
    *,
    kind: RegressionKind,
    fixture_path: str,
    correction_id: int | None = None,
    notes: str = "",
) -> RegressionCase:
    """Register the regression that will keep this issue closed."""

    if kind not in REGRESSION_KINDS:
        raise ValueError(
            f"Unsupported regression kind {kind!r}; expected one of {REGRESSION_KINDS}."
        )
    if not fixture_path.strip():
        raise ValueError("A regression case needs the path of its fixture or test.")
    issue = db.fetch_hand_issue(issue_id)
    if issue is None:
        raise ValueError(f"Issue {issue_id} was not found.")

    now = datetime.now(UTC).isoformat()
    with db.transaction():
        cursor = db._execute(
            """
            INSERT INTO regression_cases (
                issue_id, correction_id, kind, fixture_path, status,
                failing_before, passing_after, fixing_commit, report_path,
                notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'proposed', 0, 0, '', '', ?, ?, ?)
            """,
            (issue_id, correction_id, kind, fixture_path.strip(), notes.strip(), now, now),
        )
        case_id = cursor.lastrowid
    return fetch_regression_case(db, int(case_id))


def record_regression_observation(
    db: PokerDatabase,
    case_id: int,
    *,
    failing_before: bool | None = None,
    passing_after: bool | None = None,
    fixing_commit: str = "",
    report_path: str = "",
) -> RegressionCase:
    """Record that the regression was seen failing, or seen passing.

    The two observations are recorded independently and neither can be unset
    here: they are historical facts about runs that happened. A regression that
    later breaks is handled by re-running it, not by rewriting the record that
    it once passed.
    """
    case = fetch_regression_case(db, case_id)
    updates: dict[str, Any] = {"updated_at": datetime.now(UTC).isoformat()}
    if failing_before:
        updates["failing_before"] = 1
    if passing_after:
        updates["passing_after"] = 1
    if fixing_commit.strip():
        updates["fixing_commit"] = fixing_commit.strip()
    if report_path.strip():
        updates["report_path"] = report_path.strip()

    merged_failing = bool(updates.get("failing_before", int(case.failing_before)))
    merged_passing = bool(updates.get("passing_after", int(case.passing_after)))
    updates["status"] = (
        "proven" if merged_failing and merged_passing
        else "failing" if merged_failing
        else "passing" if merged_passing
        else "proposed"
    )

    assignments = ", ".join(f"{column} = ?" for column in updates)
    with db.transaction():
        db._execute(
            f"UPDATE regression_cases SET {assignments} WHERE id = ?",
            (*updates.values(), case_id),
        )
    return fetch_regression_case(db, case_id)


def fetch_regression_case(db: PokerDatabase, case_id: int) -> RegressionCase:
    row = db._execute(
        "SELECT * FROM regression_cases WHERE id = ?", (case_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Regression case {case_id} was not found.")
    return _row_to_case(row)


def regressions_for_issue(db: PokerDatabase, issue_id: int) -> list[RegressionCase]:
    rows = db._execute(
        "SELECT * FROM regression_cases WHERE issue_id = ? ORDER BY id",
        (issue_id,),
    ).fetchall()
    return [_row_to_case(row) for row in rows]


def resolution_blocker(db: PokerDatabase, issue_id: int) -> str | None:
    """Why this issue may not be resolved yet, or None when it may.

    Only release-blocking categories are gated. A coaching-wording complaint
    does not admit a wrong hand into study, and requiring a permanent regression
    for one would push operators to file everything as "other" — which would
    cost the gate its meaning on the categories that matter.

    An issue whose stored categories could not be read is gated regardless: the
    reader's fallback is ``other``, which is outside the set, so believing it
    would let row damage clear the gate. This mirrors ``_regression_blocker``,
    the writer that enforces it — an explanation that disagreed with the refusal
    would be worse than none.
    """
    issue = db.fetch_hand_issue(issue_id)
    if issue is None:
        raise ValueError(f"Issue {issue_id} was not found.")
    categories_unreadable = "issue_types" in issue.unreadable_columns
    if not categories_unreadable and not is_release_blocking(list(issue.issue_types)):
        return None
    cases = regressions_for_issue(db, issue_id)
    if not cases:
        if categories_unreadable:
            return (
                "This issue's stored categories could not be read, so it is "
                "treated as release-blocking and closing it needs a permanent "
                f"regression. Suggested fixture: {suggested_kind(list(issue.issue_types))}."
            )
        return (
            "This issue is release-blocking, so closing it needs a permanent "
            f"regression. Suggested fixture: {suggested_kind(list(issue.issue_types))}."
        )
    if not any(case.is_proven for case in cases):
        unproven = [
            f"#{case.id} ({'saw it fail' if case.failing_before else 'never seen failing'}"
            f", {'saw it pass' if case.passing_after else 'never seen passing'})"
            for case in cases
        ]
        return (
            "A regression exists but has not been proven: it must be observed "
            "failing for the original defect and passing after the fix. "
            + "; ".join(unproven)
        )
    return None


def regression_summary(db: PokerDatabase, issue_id: int) -> dict[str, Any]:
    """The linkage PLAN asks to be retained, in one readable structure."""
    issue = db.fetch_hand_issue(issue_id)
    if issue is None:
        raise ValueError(f"Issue {issue_id} was not found.")
    cases = regressions_for_issue(db, issue_id)
    return {
        "issue_id": issue_id,
        "issue_types": list(issue.issue_types),
        # Reports what the gate does, not what the salvaged categories say: a row
        # whose categories are unreadable is gated, and a summary claiming
        # otherwise while `blocker` refuses the close would be read as a bug in
        # the gate rather than as damage in the row.
        "release_blocking": is_release_blocking(list(issue.issue_types))
        or "issue_types" in issue.unreadable_columns,
        "suggested_kind": suggested_kind(list(issue.issue_types)),
        "blocker": resolution_blocker(db, issue_id),
        "cases": [
            {
                "id": case.id,
                "kind": case.kind,
                "fixture_path": case.fixture_path,
                "status": case.status,
                "correction_id": case.correction_id,
                "failing_before": case.failing_before,
                "passing_after": case.passing_after,
                "proven": case.is_proven,
                "fixing_commit": case.fixing_commit,
                "report_path": case.report_path,
            }
            for case in cases
        ],
    }


def serialize_regression_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)
