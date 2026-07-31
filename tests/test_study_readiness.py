from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    UNREADABLE_HAND_COLUMNS_KEY,
    CompletionEvidence,
    dump_completion_evidence,
)
from poker_tracker.persistence.models import (
    CoachingResponse,
    Hand,
    HandIssue,
    SolverRun,
)
from poker_tracker.services.hand_accounting import (
    AccountingReconciliation,
    AssumptionDependence,
)
from poker_tracker.services.study_readiness import (
    BLOCKER_ORDER,
    CATEGORY_ORDER,
    evaluate_study_readiness,
)

_BASE = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _dependence() -> AssumptionDependence:
    """One measured settlement-assumption dependence, as reconcile_persisted_hand emits it."""
    return AssumptionDependence(
        input_name="rake_policy",
        declared="rate=0.5, cap=None, unit=0.01, no_flop_no_drop=False",
        neutral="rate=0, cap=None, unit=0.01, no_flop_no_drop=False",
        deltas=(("rake", 10.0), ("hero", -10.0)),
        code="declared_settlement_dependence:rake_policy:rake+10|hero-10",
    )


def _authoritative(
    is_authoritative: bool = True,
    *,
    assumption_dependence: tuple[AssumptionDependence, ...] = (),
) -> AccountingReconciliation:
    return AccountingReconciliation(
        ledger=None,  # type: ignore[arg-type]
        settlement=None,
        entries=(),
        issues=("Chip conservation fails.",),
        is_authoritative=is_authoritative,
        assumption_dependence=assumption_dependence,
    )


def _clean_evidence(**overrides: Any) -> CompletionEvidence:
    values: dict[str, Any] = {
        "evidence_version": EVIDENCE_SCHEMA_VERSION,
        "partial_start": False,
        "partial_end": False,
        "terminal_event": "showdown",
        "boundary_confidence": 0.93,
        "layout_profile": "6-max",
        "layout_supported": True,
        "table_size": 6,
        "pipeline_version": "two-model-v7",
    }
    values.update(overrides)
    return CompletionEvidence(**values)


def _cv_hand(**overrides: Any) -> Hand:
    values: dict[str, Any] = {
        "id": 7,
        "session_id": 1,
        "hand_number": 1,
        "table_size": 6,
        "hero_cards": "Ah Kd",
        "board_cards": "2c 3d 4h",
        "source_type": "cv_import",
        "completion_status": "complete",
        "completion_evidence": dump_completion_evidence(_clean_evidence()),
    }
    values.update(overrides)
    return Hand(**values)


def _manual_hand(**overrides: Any) -> Hand:
    values: dict[str, Any] = {
        "id": 9,
        "session_id": 1,
        "hand_number": 2,
        "table_size": 6,
        "hero_cards": "Ah Kd",
        "board_cards": "2c 3d 4h",
        "source_type": "manual",
    }
    values.update(overrides)
    return Hand(**values)


def _issue(status: str = "open") -> HandIssue:
    return HandIssue(
        id=3,
        hand_id=7,
        status=status,
        issue_types=["pot_or_result"],
        description="Pot does not match the recording.",
        resolution_notes="" if status == "open" else "Fixed.",
    )


def _coaching(*, is_stale: bool, minutes: int) -> CoachingResponse:
    return CoachingResponse(
        provider_name="test",
        model_name="fixture",
        raw_prompt="prompt",
        raw_response="response",
        review_type="hand",
        hand_id=7,
        is_stale=is_stale,
        created_at=_BASE + timedelta(minutes=minutes),
    )


def _solver(*, status: str, minutes: int) -> SolverRun:
    return SolverRun(
        hand_id=7,
        status=status,
        input_hash="hash",
        created_at=_BASE + timedelta(minutes=minutes),
    )


def _evaluate(hand: Hand, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "accounting": _authoritative(),
        "accounting_error": None,
        "hand_issues": (),
        "coaching_reviews": (),
        "solver_runs": (),
        "user_confirmed": True,
    }
    kwargs.update(overrides)
    return evaluate_study_readiness(hand, **kwargs)


_SINGLE_BLOCKER_CASES: tuple[tuple[str, dict[str, Any], dict[str, Any]], ...] = (
    ("STUDY_EXCLUDED_BY_OPERATOR", {"study_inclusion": "skip"}, {}),
    ("COMPLETION_NOT_COMPLETE", {"completion_status": "uncertain"}, {}),
    ("INVALID_HERO_OR_BOARD_CARDS", {"hero_cards": ""}, {}),
    (
        # The reader could not validate a stored column and degraded it to a
        # fallback (db._salvaged_row); the marker rides the evidence's open
        # extra mapping exactly as the unreadable-card marker does.
        "UNREADABLE_HAND_COLUMNS",
        {
            "completion_evidence": {
                **dump_completion_evidence(_clean_evidence()),
                UNREADABLE_HAND_COLUMNS_KEY: {"table_size": "'99'"},
            }
        },
        {},
    ),
    (
        "UNSUPPORTED_TABLE_LAYOUT",
        {
            "completion_evidence": dump_completion_evidence(
                _clean_evidence(layout_supported=False)
            )
        },
        {},
    ),
    ("ACCOUNTING_NOT_AUTHORITATIVE", {}, {"accounting": _authoritative(False)}),
    (
        # An authoritative ledger that only reconciles because of a declared
        # settlement assumption. It fires alone: nothing about the hand is wrong,
        # which is exactly why it needed its own code rather than being folded
        # into a blocker whose reason says the ledger does not reconcile.
        "ACCOUNTING_ASSUMPTION_DEPENDENT",
        {},
        {"accounting": _authoritative(assumption_dependence=(_dependence(),))},
    ),
    ("OPEN_DEBUGGING_ISSUE", {}, {"hand_issues": (_issue(),)}),
    (
        "STALE_COACHING_EVIDENCE",
        {},
        {"coaching_reviews": (_coaching(is_stale=True, minutes=10),)},
    ),
    (
        "STALE_SOLVER_EVIDENCE",
        {},
        {"solver_runs": (_solver(status="stale", minutes=10),)},
    ),
    ("USER_CONFIRMATION_MISSING", {}, {"user_confirmed": False}),
)


@pytest.mark.parametrize(
    ("code", "hand_overrides", "call_overrides"),
    _SINGLE_BLOCKER_CASES,
    ids=[case[0] for case in _SINGLE_BLOCKER_CASES],
)
def test_blocker_fires_alone(
    code: str, hand_overrides: dict[str, Any], call_overrides: dict[str, Any]
) -> None:
    readiness = _evaluate(_cv_hand(**hand_overrides), **call_overrides)

    assert readiness.codes() == (code,)
    assert readiness.is_ready is False
    assert readiness.has(code) is True


def test_every_declared_blocker_code_is_covered_by_the_isolation_matrix() -> None:
    """Two codes cannot fire alone; both describe evidence that cannot derive complete.

    Unreadable evidence and an unresolved source code each make
    derive_completion_status return 'uncertain', so a stored 'complete' column is
    unproven and COMPLETION_NOT_COMPLETE necessarily co-fires.
    """
    covered = {case[0] for case in _SINGLE_BLOCKER_CASES}

    assert set(BLOCKER_ORDER) - covered == {
        "COMPLETION_EVIDENCE_MISSING",
        "UNRESOLVED_SOURCE_WARNING",
    }


def test_evidence_missing_blocks_together_with_unconfirmed_layout() -> None:
    """Layout support is only ever carried by the evidence blob, so it co-fires."""
    readiness = _evaluate(_cv_hand(completion_evidence={}))

    assert readiness.codes() == (
        "COMPLETION_NOT_COMPLETE",
        "COMPLETION_EVIDENCE_MISSING",
        "UNSUPPORTED_TABLE_LAYOUT",
    )


def test_an_unresolved_warning_never_leaves_a_complete_column_unchallenged() -> None:
    """Regression: a 'complete' column standing over an unresolved code is unproven."""
    readiness = _evaluate(
        _cv_hand(
            completion_evidence=dump_completion_evidence(
                _clean_evidence(warning_codes=("pot_not_reconciled",))
            )
        )
    )

    assert readiness.codes() == (
        "COMPLETION_NOT_COMPLETE",
        "UNRESOLVED_SOURCE_WARNING",
    )


def test_evidence_missing_suppresses_the_duplicate_warning_blocker() -> None:
    readiness = _evaluate(_cv_hand(completion_evidence={"warning_codes": ["boom"]}))

    assert readiness.has("COMPLETION_EVIDENCE_MISSING") is True
    assert readiness.has("UNRESOLVED_SOURCE_WARNING") is False


# Every clearing action must name one of these, and every entry that names a
# control must be a label the running app actually renders. The two assertions
# together are what make "the exact clearing action" checkable rather than prose.
_NAMED_CONTROLS: tuple[str, ...] = (
    "Hand facts",
    "Accounting reconciliation",
    "Saved debugging issue queue",
    "Source warnings",
    "Analyze → AI coach",
    "Analyze → TexasSolver",
    "ROI calibration",
    "Run CV reconstruction",
    "Finish validation — send to Study",
    "Import validation",
    "Re-import this hand",
)

# The literal widget labels behind those names, as rendered by app.py.
_APP_CONTROL_LABELS: tuple[str, ...] = (
    "Hand facts",
    "Accounting reconciliation",
    "Saved debugging issue queue",
    "Source warnings",
    "AI coach",
    "TexasSolver",
    "Finish validation — send to Study",
    "ROI calibration",
    "Run CV reconstruction",
    "Videos",
)


def test_every_blocker_carries_a_reason_and_a_clearing_action() -> None:
    readiness = _evaluate(
        _cv_hand(completion_status="partial", hero_cards="", completion_evidence={}),
        accounting=None,
        accounting_error="Ledger could not be built.",
        hand_issues=(_issue(),),
        coaching_reviews=(_coaching(is_stale=True, minutes=1),),
        solver_runs=(_solver(status="stale", minutes=1),),
        user_confirmed=False,
    )

    assert readiness.blockers
    for blocker in readiness.blockers:
        assert blocker.reason.strip()
        assert blocker.clearing_action.strip()
        assert "%" not in blocker.reason
        assert any(
            control in blocker.clearing_action for control in _NAMED_CONTROLS
        ), blocker.code


def test_every_control_a_clearing_action_names_exists_in_the_app() -> None:
    """Regression: four clearing actions named controls the app never rendered."""
    from pathlib import Path

    import poker_tracker.services.study_readiness as module

    app_source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text()
    missing = [label for label in _APP_CONTROL_LABELS if f'"{label}' not in app_source]
    assert missing == []

    service_source = Path(module.__file__).read_text()
    unused = [name for name in _NAMED_CONTROLS if name not in service_source]
    assert unused == []


def test_blockers_are_emitted_in_declaration_order() -> None:
    readiness = _evaluate(
        _cv_hand(completion_status="uncertain", hero_cards="", completion_evidence={}),
        accounting=_authoritative(False),
        hand_issues=(_issue(),),
        coaching_reviews=(_coaching(is_stale=True, minutes=1),),
        solver_runs=(_solver(status="stale", minutes=1),),
        user_confirmed=False,
    )

    codes = readiness.codes()
    assert list(codes) == [code for code in BLOCKER_ORDER if code in set(codes)]


def test_by_category_covers_every_emitted_blocker() -> None:
    readiness = _evaluate(
        _cv_hand(completion_status="uncertain", hero_cards="", completion_evidence={}),
        accounting=_authoritative(False),
        hand_issues=(_issue(),),
        coaching_reviews=(_coaching(is_stale=True, minutes=1),),
        solver_runs=(_solver(status="stale", minutes=1),),
        user_confirmed=False,
    )
    grouped = readiness.by_category()

    flattened = [blocker for group in grouped.values() for blocker in group]
    assert len(flattened) == len(readiness.blockers)
    assert set(flattened) == set(readiness.blockers)
    assert list(grouped) == [
        category for category in CATEGORY_ORDER if category in grouped
    ]
    assert list(grouped) == [
        "completion",
        "cards",
        "layout",
        "accounting",
        "issues",
        "coaching",
        "solver",
        "confirmation",
    ]


def test_complete_reconstructed_hand_is_ready_with_confirmation() -> None:
    readiness = _evaluate(_cv_hand())

    assert readiness.is_ready is True
    assert readiness.blockers == ()
    assert readiness.completion_status == "complete"


def test_complete_reconstructed_hand_is_blocked_without_confirmation() -> None:
    readiness = _evaluate(_cv_hand(), user_confirmed=False)

    assert readiness.codes() == ("USER_CONFIRMATION_MISSING",)


def test_manual_hand_is_ready_when_reconciled_and_clean() -> None:
    readiness = _evaluate(_manual_hand(), user_confirmed=False)

    assert readiness.is_ready is True
    assert readiness.completion_status == "not_applicable"


def test_manual_hand_does_not_require_confirmation() -> None:
    assert _evaluate(_manual_hand(), user_confirmed=False).has(
        "USER_CONFIRMATION_MISSING"
    ) is False


@pytest.mark.parametrize(
    "hand_overrides",
    [
        {"completion_evidence": {}},
        {"completion_evidence": {"warning_codes": ["pot_not_reconciled"]}},
        {"table_size": None},
        {
            "completion_evidence": dump_completion_evidence(
                _clean_evidence(layout_supported=False, warning_codes=("boom",))
            )
        },
    ],
)
def test_manual_hand_never_emits_completion_layout_or_warning_blockers(
    hand_overrides: dict[str, Any],
) -> None:
    readiness = _evaluate(_manual_hand(**hand_overrides), user_confirmed=False)

    assert readiness.has("COMPLETION_NOT_COMPLETE") is False
    assert readiness.has("COMPLETION_EVIDENCE_MISSING") is False
    assert readiness.has("UNSUPPORTED_TABLE_LAYOUT") is False
    assert readiness.has("UNRESOLVED_SOURCE_WARNING") is False
    assert readiness.has("USER_CONFIRMATION_MISSING") is False


def test_manual_hand_with_empty_hero_cards_is_not_blocked() -> None:
    readiness = _evaluate(
        _manual_hand(hero_cards="", board_cards=""), user_confirmed=False
    )

    assert readiness.is_ready is True


def test_reconstructed_hand_with_empty_hero_cards_is_blocked() -> None:
    readiness = _evaluate(_cv_hand(hero_cards=""))

    assert readiness.has("INVALID_HERO_OR_BOARD_CARDS") is True


def test_stale_coaching_cleared_by_a_newer_current_review() -> None:
    readiness = _evaluate(
        _cv_hand(),
        coaching_reviews=(
            _coaching(is_stale=True, minutes=10),
            _coaching(is_stale=False, minutes=20),
        ),
    )

    assert readiness.has("STALE_COACHING_EVIDENCE") is False


def test_stale_coaching_blocks_when_the_current_review_is_older() -> None:
    readiness = _evaluate(
        _cv_hand(),
        coaching_reviews=(
            _coaching(is_stale=False, minutes=5),
            _coaching(is_stale=True, minutes=10),
        ),
    )

    assert readiness.has("STALE_COACHING_EVIDENCE") is True


def test_stale_coaching_does_not_block_with_no_reviews_at_all() -> None:
    assert _evaluate(_cv_hand(), coaching_reviews=()).has("STALE_COACHING_EVIDENCE") is False


def test_stale_solver_cleared_by_a_newer_completed_run() -> None:
    readiness = _evaluate(
        _cv_hand(),
        solver_runs=(
            _solver(status="stale", minutes=10),
            _solver(status="completed", minutes=20),
        ),
    )

    assert readiness.has("STALE_SOLVER_EVIDENCE") is False


def test_stale_solver_blocks_when_the_completed_run_is_older() -> None:
    readiness = _evaluate(
        _cv_hand(),
        solver_runs=(
            _solver(status="completed", minutes=5),
            _solver(status="stale", minutes=10),
        ),
    )

    assert readiness.has("STALE_SOLVER_EVIDENCE") is True


@pytest.mark.parametrize("status", ["failed", "cancelled", "queued", "running"])
def test_failed_or_cancelled_solver_run_alone_does_not_block(status: str) -> None:
    readiness = _evaluate(_cv_hand(), solver_runs=(_solver(status=status, minutes=10),))

    assert readiness.has("STALE_SOLVER_EVIDENCE") is False


def test_cancelling_solver_run_blocks() -> None:
    readiness = _evaluate(
        _cv_hand(), solver_runs=(_solver(status="cancelling", minutes=10),)
    )

    assert readiness.has("STALE_SOLVER_EVIDENCE") is True


_TRUTH_TABLE: tuple[tuple[str, Hand, dict[str, Any], set[str]], ...] = (
    ("all clear reconstructed", _cv_hand(), {}, set()),
    (
        "all blocked",
        _cv_hand(completion_status="partial", hero_cards="", completion_evidence={}),
        {
            "accounting": _authoritative(False),
            "hand_issues": (_issue(),),
            "coaching_reviews": (_coaching(is_stale=True, minutes=1),),
            "solver_runs": (_solver(status="stale", minutes=1),),
            "user_confirmed": False,
        },
        {
            "COMPLETION_NOT_COMPLETE",
            "COMPLETION_EVIDENCE_MISSING",
            "INVALID_HERO_OR_BOARD_CARDS",
            "UNSUPPORTED_TABLE_LAYOUT",
            "ACCOUNTING_NOT_AUTHORITATIVE",
            "OPEN_DEBUGGING_ISSUE",
            "STALE_COACHING_EVIDENCE",
            "STALE_SOLVER_EVIDENCE",
            "USER_CONFIRMATION_MISSING",
        },
    ),
    (
        "same category pair: completion status and missing evidence",
        _cv_hand(completion_status="uncertain", completion_evidence={}),
        {},
        {
            "COMPLETION_NOT_COMPLETE",
            "COMPLETION_EVIDENCE_MISSING",
            "UNSUPPORTED_TABLE_LAYOUT",
        },
    ),
    (
        "same category pair: completion status and unresolved warning",
        _cv_hand(
            completion_status="uncertain",
            completion_evidence=dump_completion_evidence(
                _clean_evidence(warning_codes=("pot_not_reconciled",))
            ),
        ),
        {},
        {"COMPLETION_NOT_COMPLETE", "UNRESOLVED_SOURCE_WARNING"},
    ),
    (
        "same category pair: stale coaching and stale solver",
        _cv_hand(),
        {
            "coaching_reviews": (_coaching(is_stale=True, minutes=1),),
            "solver_runs": (_solver(status="stale", minutes=1),),
        },
        {"STALE_COACHING_EVIDENCE", "STALE_SOLVER_EVIDENCE"},
    ),
    (
        "partial + reconciled + confirmed",
        _cv_hand(
            completion_status="partial",
            completion_evidence=dump_completion_evidence(
                _clean_evidence(partial_end=True)
            ),
        ),
        {},
        {"COMPLETION_NOT_COMPLETE"},
    ),
    (
        "uncertain + unreconciled",
        _cv_hand(completion_status="uncertain"),
        {"accounting": _authoritative(False)},
        {"COMPLETION_NOT_COMPLETE", "ACCOUNTING_NOT_AUTHORITATIVE"},
    ),
    (
        "complete + open issue",
        _cv_hand(),
        {"hand_issues": (_issue(),)},
        {"OPEN_DEBUGGING_ISSUE"},
    ),
    (
        "complete + stale coaching + confirmed",
        _cv_hand(),
        {"coaching_reviews": (_coaching(is_stale=True, minutes=1),)},
        {"STALE_COACHING_EVIDENCE"},
    ),
    (
        "complete + stale solver + confirmed",
        _cv_hand(),
        {"solver_runs": (_solver(status="stale", minutes=1),)},
        {"STALE_SOLVER_EVIDENCE"},
    ),
    (
        "complete + unsupported layout + confirmed",
        _cv_hand(
            completion_evidence=dump_completion_evidence(
                _clean_evidence(layout_supported=None)
            )
        ),
        {},
        {"UNSUPPORTED_TABLE_LAYOUT"},
    ),
    (
        "complete + duplicate cards",
        # model_copy skips validation, standing in for a row written outside the model.
        _cv_hand().model_copy(update={"board_cards": "Ah 3d 4h"}),
        {},
        {"INVALID_HERO_OR_BOARD_CARDS"},
    ),
    (
        "manual + unreconciled",
        _manual_hand(),
        {"accounting": _authoritative(False), "user_confirmed": False},
        {"ACCOUNTING_NOT_AUTHORITATIVE"},
    ),
    (
        "manual + open issue",
        _manual_hand(),
        {"hand_issues": (_issue(),), "user_confirmed": False},
        {"OPEN_DEBUGGING_ISSUE"},
    ),
    (
        "manual + stale solver",
        _manual_hand(),
        {"solver_runs": (_solver(status="stale", minutes=1),), "user_confirmed": False},
        {"STALE_SOLVER_EVIDENCE"},
    ),
    (
        # A 'complete' column the evidence cannot justify is itself a blocker.
        "declared complete + evidence missing",
        _cv_hand(completion_evidence={}),
        {},
        {
            "COMPLETION_NOT_COMPLETE",
            "COMPLETION_EVIDENCE_MISSING",
            "UNSUPPORTED_TABLE_LAYOUT",
        },
    ),
    (
        "declared complete + unacknowledged warning + confirmed",
        _cv_hand(
            completion_evidence=dump_completion_evidence(
                _clean_evidence(warning_codes=("boom",))
            )
        ),
        {},
        {"COMPLETION_NOT_COMPLETE", "UNRESOLVED_SOURCE_WARNING"},
    ),
)


@pytest.mark.parametrize(
    ("hand", "call_overrides", "expected"),
    [(case[1], case[2], case[3]) for case in _TRUTH_TABLE],
    ids=[case[0] for case in _TRUTH_TABLE],
)
def test_readiness_truth_table(
    hand: Hand, call_overrides: dict[str, Any], expected: set[str]
) -> None:
    readiness = _evaluate(hand, **call_overrides)

    assert set(readiness.codes()) == expected
    assert readiness.is_ready is (not expected)


# ---------------------------------------------------------------------------
# Exhaustive combination coverage
#
# The named cases above are representative. These two tests walk the full
# power set of the ten blocker triggers, so no combination is unexercised.
# ---------------------------------------------------------------------------


def _build_case(
    codes: frozenset[str], *, manual: bool
) -> tuple[Hand, dict[str, Any]]:
    """Turn a set of blocker codes into the hand and call arguments that trigger them."""
    hand_overrides: dict[str, Any] = {}
    call_overrides: dict[str, Any] = {"user_confirmed": True}
    evidence_overrides: dict[str, Any] = {}

    if "STUDY_EXCLUDED_BY_OPERATOR" in codes:
        hand_overrides["study_inclusion"] = "skip"
    if "COMPLETION_NOT_COMPLETE" in codes:
        hand_overrides["completion_status"] = "uncertain"
    if "UNSUPPORTED_TABLE_LAYOUT" in codes:
        evidence_overrides["layout_supported"] = False
    if "UNRESOLVED_SOURCE_WARNING" in codes:
        evidence_overrides["warning_codes"] = ("pot_not_reconciled",)
    marker: dict[str, Any] = (
        {UNREADABLE_HAND_COLUMNS_KEY: {"table_size": "'99'"}}
        if "UNREADABLE_HAND_COLUMNS" in codes
        else {}
    )
    if "COMPLETION_EVIDENCE_MISSING" in codes:
        # An unreadable blob replaces the whole envelope, so it subsumes the
        # evidence-carried triggers rather than composing with them. The
        # unreadable-column marker still composes: it rides the open extra
        # mapping and carries no evidence_version.
        hand_overrides["completion_evidence"] = dict(marker)
    else:
        hand_overrides["completion_evidence"] = {
            **dump_completion_evidence(_clean_evidence(**evidence_overrides)),
            **marker,
        }
    call_overrides["accounting"] = _authoritative(
        "ACCOUNTING_NOT_AUTHORITATIVE" not in codes,
        assumption_dependence=(
            (_dependence(),) if "ACCOUNTING_ASSUMPTION_DEPENDENT" in codes else ()
        ),
    )
    if "OPEN_DEBUGGING_ISSUE" in codes:
        call_overrides["hand_issues"] = (_issue(),)
    if "STALE_COACHING_EVIDENCE" in codes:
        call_overrides["coaching_reviews"] = (_coaching(is_stale=True, minutes=1),)
    if "STALE_SOLVER_EVIDENCE" in codes:
        call_overrides["solver_runs"] = (_solver(status="stale", minutes=1),)
    if "USER_CONFIRMATION_MISSING" in codes:
        call_overrides["user_confirmed"] = False

    if manual:
        hand_overrides.pop("completion_status", None)
        hand = _manual_hand(**hand_overrides)
    else:
        hand = _cv_hand(**hand_overrides)
    if "INVALID_HERO_OR_BOARD_CARDS" in codes:
        # Ah appears in both hero and board. model_copy skips validation, standing
        # in for a row written outside the model, and fires for manual hands too
        # (an empty hero_cards is legitimate for a manual hand and must not fire).
        hand = hand.model_copy(update={"board_cards": "Ah 3d 4h"})
    return hand, call_overrides


def _expected_reconstructed(codes: frozenset[str]) -> set[str]:
    expected = set(codes)
    if "COMPLETION_EVIDENCE_MISSING" in codes:
        # Layout support is only ever carried by the evidence blob, and the
        # warning blocker is suppressed to avoid restating the same fact.
        expected.add("UNSUPPORTED_TABLE_LAYOUT")
        expected.discard("UNRESOLVED_SOURCE_WARNING")
    if codes & {"COMPLETION_EVIDENCE_MISSING", "UNRESOLVED_SOURCE_WARNING"}:
        # Neither unreadable evidence nor an unresolved code derives 'complete',
        # so a stored 'complete' column is unproven and co-fires.
        expected.add("COMPLETION_NOT_COMPLETE")
    return expected


_MANUAL_APPLICABLE = {
    "STUDY_EXCLUDED_BY_OPERATOR",
    "INVALID_HERO_OR_BOARD_CARDS",
    "UNREADABLE_HAND_COLUMNS",
    "ACCOUNTING_NOT_AUTHORITATIVE",
    "OPEN_DEBUGGING_ISSUE",
    "STALE_COACHING_EVIDENCE",
    "STALE_SOLVER_EVIDENCE",
}


def _power_set(codes: tuple[str, ...]):
    for mask in range(1 << len(codes)):
        yield frozenset(code for index, code in enumerate(codes) if mask & (1 << index))


def test_reconstructed_readiness_is_exhaustive_over_every_blocker_combination() -> None:
    """Every subset of the declared triggers, not just the named representatives."""
    failures: list[str] = []

    for subset in _power_set(BLOCKER_ORDER):
        hand, call_overrides = _build_case(subset, manual=False)
        readiness = _evaluate(hand, **call_overrides)
        expected = _expected_reconstructed(subset)
        if set(readiness.codes()) != expected:
            failures.append(
                f"{sorted(subset)} -> {sorted(readiness.codes())} != {sorted(expected)}"
            )
        elif readiness.is_ready is not (not expected):
            failures.append(f"{sorted(subset)} -> is_ready {readiness.is_ready}")

    assert failures == []


def test_manual_readiness_is_exhaustive_and_never_emits_reconstructed_blockers() -> None:
    """A manual hand can only ever emit the cards, accounting, issue, and staleness codes."""
    failures: list[str] = []

    for subset in _power_set(BLOCKER_ORDER):
        hand, call_overrides = _build_case(subset, manual=True)
        readiness = _evaluate(hand, **call_overrides)
        expected = subset & _MANUAL_APPLICABLE
        if set(readiness.codes()) != expected:
            failures.append(
                f"{sorted(subset)} -> {sorted(readiness.codes())} != {sorted(expected)}"
            )

    assert failures == []
    assert _MANUAL_APPLICABLE < set(BLOCKER_ORDER)
