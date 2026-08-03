"""Derived, never-persisted judgement of whether one completed hand is safe to study.

This module is pure: it takes already-fetched records and never touches
``PokerDatabase``, so it is unit-testable without a database and cannot turn a
list view into one reconciliation per hand.

It lives in ``services/`` rather than ``ui/view_models.py`` because it composes
accounting, issue, coaching, and solver evidence; ``view_models`` is documented as
pure display transformation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from poker_tracker.coaching.grounding import UNGROUNDED_STALE_PREFIX
from poker_tracker.math.cards import (
    CardParseError,
    parse_board_cards,
    parse_hero_cards,
    parse_visible_cards,
)
from poker_tracker.persistence.completion import (
    UNREADABLE_CARDS_KEY,
    UNREADABLE_HAND_COLUMNS_KEY,
    CompletionEvidence,
    derive_completion_status,
    has_operator_manual_completion,
    is_reconstructed,
    parse_completion_evidence,
    requires_assumption_attestation,
    requires_user_confirmation,
)
from poker_tracker.persistence.models import (
    CoachingResponse,
    CompletionStatus,
    Hand,
    HandIssue,
    HandReview,
    HandSettlement,
    SolverRun,
)
from poker_tracker.services.hand_accounting import (
    AccountingReconciliation,
    AssumptionDependence,
)
from poker_tracker.services.regression_promotion import is_release_blocking

BlockerCategory = Literal[
    "completion",
    "cards",
    "facts",
    "layout",
    "accounting",
    "issues",
    "coaching",
    "solver",
    "confirmation",
    "study_preference",
]
BlockerCode = Literal[
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
    "STUDY_EXCLUDED_BY_OPERATOR",
]

# Blockers are emitted in this order, and categories render in this order.
BLOCKER_ORDER: tuple[BlockerCode, ...] = (
    "STUDY_EXCLUDED_BY_OPERATOR",
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
CATEGORY_ORDER: tuple[BlockerCategory, ...] = (
    "study_preference",
    "completion",
    "cards",
    "facts",
    "layout",
    "accounting",
    "issues",
    "coaching",
    "solver",
    "confirmation",
)

_VALID_BOARD_COUNTS = {0, 3, 4, 5}
_STALE_SOLVER_STATUSES = {"stale", "cancelling"}


class RetainedReview(Protocol):
    """Any saved coaching artefact the correction workflow can invalidate.

    Both retained coaching tables satisfy it: ``coaching_reviews`` (the current
    provider path) and the legacy ``hand_reviews`` rows, which
    ``_invalidate_hand_derivatives`` stales identically and which the Hands
    workspace still renders.

    ``stale_reason`` is part of the protocol because ``is_stale`` alone no longer
    says what happened: a review is staled both by a correction to the hand and
    by failing its own grounding check, and that column is the only thing that
    tells the two apart.
    """

    @property
    def is_stale(self) -> bool: ...

    @property
    def stale_reason(self) -> str: ...

    @property
    def created_at(self) -> datetime: ...


@dataclass(frozen=True)
class StudyBlocker:
    """One concrete reason a hand is not study-ready, and the action that clears it."""

    code: BlockerCode
    category: BlockerCategory
    reason: str  # why this blocks, in plain language, never a percentage
    clearing_action: str  # the exact action that clears it
    detail: tuple[str, ...] = ()  # concrete offending values


@dataclass(frozen=True)
class _Cause:
    """One condition that made a blocker fire, carrying its own explanation.

    A blocker code covers a FAMILY of conditions -- STALE_COACHING_EVIDENCE
    covers a review a correction invalidated and a review that failed its own
    grounding check, UNSUPPORTED_TABLE_LAYOUT covers five -- and the sentence an
    operator acts on has to come from the condition that actually fired. Stating
    it where the blocker is raised looks equivalent and is not: the second
    condition routed to a code silently inherits the first one's explanation, and
    nothing fails. That is how STALE_COACHING_EVIDENCE came to tell an operator
    "a later correction invalidated this", which means re-run coaching, about an
    answer that was rejected for naming a card the hand never held, which means
    something quite different about the answer they just paid for.

    Binding the sentence to the condition makes that mistake require deleting
    text rather than merely forgetting to add it.
    """

    reason: str
    clearing_action: str
    detail: tuple[str, ...] = ()


@dataclass(frozen=True)
class StudyReadiness:
    is_ready: bool
    completion_status: CompletionStatus
    blockers: tuple[StudyBlocker, ...]

    def has(self, code: BlockerCode) -> bool:
        return any(blocker.code == code for blocker in self.blockers)

    def codes(self) -> tuple[BlockerCode, ...]:
        return tuple(blocker.code for blocker in self.blockers)

    def by_category(self) -> dict[BlockerCategory, tuple[StudyBlocker, ...]]:
        grouped: dict[BlockerCategory, tuple[StudyBlocker, ...]] = {}
        for category in CATEGORY_ORDER:
            matches = tuple(
                blocker for blocker in self.blockers if blocker.category == category
            )
            if matches:
                grouped[category] = matches
        return grouped


def _joined(values: Iterable[str]) -> str:
    """Join distinct, non-empty sentences in the order their conditions were found."""
    return " ".join(dict.fromkeys(value for value in values if value))


def _blocker_from_causes(
    code: BlockerCode, category: BlockerCategory, causes: list[_Cause]
) -> list[StudyBlocker]:
    """Compose one blocker out of the conditions that actually fired, or none.

    Repeated sentences are dropped, so two conditions that share a clearing
    action -- both evidence-borne layout faults name the same reconstruction --
    do not print it twice.
    """
    if not causes:
        return []
    return [
        StudyBlocker(
            code=code,
            category=category,
            reason=_joined(cause.reason for cause in causes),
            clearing_action=_joined(cause.clearing_action for cause in causes),
            detail=tuple(item for cause in causes for item in cause.detail),
        )
    ]


def is_reconstructed_hand(hand: Hand) -> bool:
    """The single predicate deciding which blockers and which controls apply.

    Both halves are required: import validation rejects the inconsistent pair and
    ``_hand_from_row`` normalises it, so this cannot be laundered by editing one
    field in an export. Exported so the UI cannot drift to a different rule --
    the Study page used to gate its confirmation checkbox on the completion
    column alone, which meant a reconstructed hand could emit
    USER_CONFIRMATION_MISSING naming a checkbox that was never drawn.

    The rule itself lives in ``persistence.completion`` so the persistence
    writers can share it without importing this service.
    """
    return is_reconstructed(hand.source_type, hand.completion_status)


def hand_requires_assumption_attestation(hand: Hand) -> bool:
    """Whether THIS operator must attest to this hand's declared settlement inputs.

    Exported so the control that clears ACCOUNTING_ASSUMPTION_DEPENDENT is drawn
    under exactly the condition the blocker is emitted under.
    """
    return requires_assumption_attestation(
        source_type=hand.source_type,
        completion_status=hand.completion_status,
        evidence=parse_completion_evidence(hand.completion_evidence),
    )


def hand_requires_user_confirmation(hand: Hand) -> bool:
    """Whether the whole-hand confirmation checkbox and blocker apply to this hand.

    Exported for the same reason ``hand_requires_assumption_attestation`` is: the
    checkbox that clears USER_CONFIRMATION_MISSING must be drawn under exactly
    the condition the blocker is emitted under. Scoped by
    ``requires_user_confirmation`` -- reconstructed OR imported -- because "you
    entered this hand yourself" is false for a hand that arrived as
    user-supplied JSON, whatever ``source_type`` it declares; gating the blocker
    on ``is_reconstructed_hand`` alone let a payload relabelled ``manual`` land
    study-ready with an empty blocker tuple before any click at all.
    """
    return requires_user_confirmation(
        source_type=hand.source_type,
        completion_status=hand.completion_status,
        evidence=parse_completion_evidence(hand.completion_evidence),
    )


def unattested_assumption_dependence(
    hand: Hand, accounting: AccountingReconciliation | None
) -> tuple[AssumptionDependence, ...]:
    """Measured settlement-assumption dependences this hand still owes an answer for.

    ONE definition of "outstanding", consulted by the readiness blocker, by
    ``accounting_is_established``, and therefore by every surface that prints a
    derived figure. Two things resolve a measured dependence and nothing else
    does:

    * the operator attested to that exact code (``confirmed_assumption_codes``),
      which is bound to the declaration and to the measured chips, so it lapses
      the moment either changes; or
    * this hand does not require an attestation at all -- it was entered in this
      database, so a declared ante, dead blind or rake is the same person's own
      observation and there is no pipeline claim for it to outrank. The
      dependence is still measured and still displayed on such a hand; it is
      simply already answered.

    Before this existed the two questions were answered by two expressions in two
    modules. Readiness consulted the attestation and app-level consumers did not,
    so confirming an assumption cleared the Study blocker while every coaching
    control stayed disabled forever -- and on a manual hand, where no attestation
    control is ever drawn, an ordinary room rake disabled coaching, emptied the
    provider's math facts and printed "Accounting: unsettled" on a hand the same
    page rendered as reconciled, with no action anywhere that could clear it.
    """
    if accounting is None:
        return ()
    if not hand_requires_assumption_attestation(hand):
        return ()
    confirmed = set(
        parse_completion_evidence(hand.completion_evidence).confirmed_assumption_codes
    )
    return tuple(
        dependence
        for dependence in accounting.assumption_dependence
        if dependence.code not in confirmed
    )


def stale_accounting_verdict(
    accounting: AccountingReconciliation | None,
) -> HandSettlement | None:
    """The stored settlement whose verdict predates the record, if there is one.

    Returns the settlement rather than a yes/no because every caller that acts on
    this state then quotes ``settlement.status`` back to the operator, and a bare
    predicate cannot carry the guarantee that the settlement is there: the
    callers were re-deriving ``accounting.settlement.status`` behind a boolean,
    which no reader -- human or type checker -- can confirm is safe without
    re-deriving this function's body. ``accounting_verdict_predates_record``
    below is the yes/no form for callers that only need the question answered.

    ``is_authoritative`` needs two things: a ledger that reconciles NOW, derived
    on every read, and a ``hand_settlements.status`` reading ``reconciled``,
    which is a label written by a past save. They come apart whenever the record
    moves after the label was written -- correct an action amount, an award, a
    starting stack or the board -- and on a hand that has simply never been
    reconciled, where the ledger has always been fine and nobody has saved yet.

    Both are the same state and it needs its own name, because the blocker built
    for it said "The chip ledger does not reconcile", which is the one thing that
    is not true here, and carried no detail: ``persist_reconciliation`` had
    already rewritten the recorded figures the cross-check compares, so the
    issue list an operator would be shown was EMPTY. A hand blocked with a false
    reason and no evidence is indistinguishable from a hand with a real
    accounting defect, and the two want opposite actions -- one wants the ledger
    investigated, the other wants the settlement saved once.

    Deliberately not part of ``accounting_is_established``: a stale verdict is
    still not a verdict, and no derived figure may be published on one. This only
    decides how the refusal is worded.

    ``is_authoritative`` is deliberately not read, and not because of the
    allowlist that guards it: a stored status other than ``reconciled`` already
    forces that flag False, so consulting it would restate a term below rather
    than decide anything.
    """
    if accounting is None or accounting.issues:
        return None
    settlement = accounting.settlement
    if settlement is None or settlement.status == "reconciled":
        return None
    ledger = accounting.ledger
    if not (ledger.is_settled and ledger.is_balanced and ledger.is_legal):
        return None
    return settlement


def accounting_verdict_predates_record(
    accounting: AccountingReconciliation | None,
) -> bool:
    """Yes/no form of :func:`stale_accounting_verdict`.

    Callers that go on to quote the stored status should call that instead, so
    the settlement they quote is the one this decision was made about.
    """

    return stale_accounting_verdict(accounting) is not None


def accounting_is_established(
    hand: Hand, accounting: AccountingReconciliation | None
) -> bool:
    """Reconciled by the RECORDING (or by an answered declaration), not by an assumption.

    ``AccountingReconciliation.is_authoritative`` answers the narrower question
    "do the chips balance under the stored settlement declaration?" and is
    deliberately unchanged. This is the question every consumer that prints a
    derived figure as fact is really asking: is this number established, or does
    it rest on an unanswered declaration somebody typed in?

    Every surface outside the Study accounting panel must call this rather than
    reading ``is_authoritative``, and
    ``test_no_consumer_decides_on_is_authoritative_alone`` fails on any new
    reader of the raw flag, so a consumer added later cannot quietly rejoin the
    old predicate. That is not hypothetical: the session win rate, the hero-result
    column of every list view, the Overview featured pot, the Math Review
    defaults and solver eligibility were all still on the raw flag, and published
    a 72-chip fabrication as a reconciled result on a hand Study refused.
    """
    return (
        accounting is not None
        and accounting.is_authoritative
        and not unattested_assumption_dependence(hand, accounting)
    )


def evaluate_study_readiness(
    hand: Hand,
    *,
    accounting: AccountingReconciliation | None,
    accounting_error: str | None = None,
    hand_issues: Iterable[HandIssue] = (),
    coaching_reviews: Iterable[CoachingResponse] = (),
    hand_reviews: Iterable[HandReview] = (),
    solver_runs: Iterable[SolverRun] = (),
    user_confirmed: bool = False,
) -> StudyReadiness:
    """Derive, without persisting, whether one completed hand is safe to study."""

    is_reconstructed = is_reconstructed_hand(hand)
    evidence = parse_completion_evidence(hand.completion_evidence)
    issues = [issue for issue in hand_issues]
    reviews: list[RetainedReview] = [*coaching_reviews, *hand_reviews]
    runs = [run for run in solver_runs]

    blockers: list[StudyBlocker] = []
    if hand.study_inclusion == "skip":
        blockers.append(
            StudyBlocker(
                code="STUDY_EXCLUDED_BY_OPERATOR",
                category="study_preference",
                reason="You marked this hand as non-study.",
                clearing_action=(
                    "Set Study inclusion to Auto or Study on this hand's row in "
                    "Sessions → Hands (or Hands library)."
                ),
            )
        )
    if is_reconstructed:
        blockers.extend(_completion_blockers(hand, evidence))
    blockers.extend(
        _card_blockers(hand, evidence, is_reconstructed=is_reconstructed)
    )
    blockers.extend(_unreadable_column_blockers(evidence))
    if is_reconstructed:
        blockers.extend(_layout_blockers(hand, evidence))
    blockers.extend(_accounting_blockers(accounting, accounting_error))
    blockers.extend(_assumption_blockers(hand, accounting))
    blockers.extend(_issue_blockers(issues))
    if is_reconstructed and evidence.is_known:
        blockers.extend(_source_warning_blockers(evidence))
    blockers.extend(_coaching_blockers(reviews))
    blockers.extend(_solver_blockers(runs))
    if hand_requires_user_confirmation(hand) and not user_confirmed:
        # Reconstructed OR imported, never `is_reconstructed` alone: an import
        # payload relabelled `source_type: manual` with blank evidence satisfies
        # the reconstructed predicate's two strings, and this blocker was the
        # last line between that payload and an empty blocker tuple. What a
        # payload cannot manufacture is having been entered here.
        blockers.extend(
            _blocker_from_causes(
                "USER_CONFIRMATION_MISSING",
                "confirmation",
                [
                    _Cause(
                        reason=(
                            "You have not confirmed that this reconstructed hand "
                            "is correct."
                            if is_reconstructed
                            else (
                                "This hand arrived in an import payload, and you "
                                "have not confirmed that it is correct."
                            )
                        ),
                        clearing_action=(
                            "Press 'Finish validation — send to Study' on Import "
                            "validation."
                        ),
                    )
                ],
            )
        )

    ordered = tuple(sorted(blockers, key=lambda blocker: BLOCKER_ORDER.index(blocker.code)))
    return StudyReadiness(
        is_ready=not ordered,
        completion_status=hand.completion_status,
        blockers=ordered,
    )


def _completion_blockers(
    hand: Hand, evidence: CompletionEvidence
) -> list[StudyBlocker]:
    blockers: list[StudyBlocker] = []
    # The stored column is never trusted on its own. Any row written outside
    # import_session -- a hand-edited database, a future writer, a payload whose
    # declared status was laundered -- must still be justified by its own
    # evidence, and derive_completion_status is the only thing that can justify
    # it. The pair is reported under the existing blocker code so the documented
    # vocabulary does not grow.
    derived = derive_completion_status(evidence, source_type=hand.source_type)
    if hand.completion_status != "complete" or derived != "complete":
        effective = hand.completion_status if derived == "complete" else derived
        detail = [f"completion_status={hand.completion_status}"]
        if derived != hand.completion_status:
            detail.append(f"evidence derives completion_status={derived}")
        if evidence.rejection_codes:
            detail.append(f"rejection_codes={', '.join(evidence.rejection_codes)}")
        blockers.append(
            StudyBlocker(
                code="COMPLETION_NOT_COMPLETE",
                category="completion",
                reason=_completion_reason(effective, evidence),
                clearing_action=_completion_clearing_action(effective, evidence),
                detail=tuple(detail),
            )
        )
    if not evidence.is_known:
        blockers.append(
            StudyBlocker(
                code="COMPLETION_EVIDENCE_MISSING",
                category="completion",
                reason=(
                    "No readable reconstruction evidence is attached, so completion "
                    "cannot be proven."
                ),
                clearing_action=(
                    "Only a new reconstruction attaches evidence to a hand that has "
                    "none -- hands stored before schema 13 carry none, and no writer "
                    "reachable from the UI attaches it after the fact: "
                    f"{NEW_RECONSTRUCTION_STEPS} The hand can still be inspected, "
                    "corrected, and given a debugging issue in the meantime."
                ),
                detail=(f"evidence_version={evidence.evidence_version}",),
            )
        )
    return blockers


# What "run a new reconstruction and import it" actually does, written once and
# composed into every blocker that names it.
#
# ``import_hands_into_session`` APPENDS. It imports the payload into a temporary
# session, moves every hand into the target session and renumbers on collision
# (``db.move_hand_to_session``); nothing is matched, replaced, or de-duplicated,
# and no writer in this product re-imports an EXISTING hand. Seven blocker
# branches used to say "Re-import this hand", so an operator who performed the
# named action verbatim watched the blocker stay byte-identical on the hand it
# named, gained a second copy of every hand in the session, and had a completed
# hand counted twice in that session's own statistics -- with no text anywhere on
# the Study page telling them to delete the original. A blocker never names an
# action the product cannot perform, and it does not get to name half of one
# either: the deletion is part of the action -- which is why every hand row
# rendered by ``app.render_hand_results`` (Sessions -> Hands, and the Hand
# library) carries a 'Delete hand' control. This text named the deletion for one
# round while the running app had no reachable delete-hand control at all: the
# only ``db.delete_hand`` call site sat in an unreferenced function, so the
# operator who performed the first half of the action verbatim ended holding a
# duplicated session and an instruction the product could not perform.
NEW_RECONSTRUCTION_STEPS = (
    "open the session's Videos tab, press Run CV reconstruction, and import the "
    "session again. The import ADDS the rebuilt hands beside the existing ones and "
    "renumbers any hand-number collision; it never replaces a hand. So this hand "
    "stays blocked, and stays in the session's statistics, until you compare it "
    "against its rebuilt copy and delete this one with the 'Delete hand' control "
    "on its row in the session's hand list (Sessions → Hands)."
)


def _completion_reason(status: CompletionStatus, evidence: CompletionEvidence) -> str:
    if status == "partial":
        if evidence.partial_start is True and evidence.partial_end is True:
            truncation = "both ends are truncated"
        elif evidence.partial_end is True:
            truncation = "it ends mid-hand"
        elif evidence.partial_start is True:
            truncation = "it starts mid-hand"
        else:
            # The import ceiling deliberately honours a declared `partial` over a
            # weaker re-derivation, and update_hand_completion is sticky on it, so
            # the column can legitimately outrank its own evidence. Asserting a
            # truncation the evidence denies invented a fact about the operator's
            # recording, and then told them to re-import from a complete one --
            # which is exactly what they had already done.
            return (
                "This hand is classified partial, but its stored evidence does "
                "not agree: it records neither a truncated start nor a truncated "
                "end. The more restrictive classification stands."
            )
        return (
            f"The recording does not contain the whole hand; {truncation}. "
            "That is fine if you reconstructed every action yourself — finalize "
            "the draft after filling facts."
        )
    if status == "not_applicable":
        return "This hand claims a reconstructed source but declares no completion evidence."
    return "The pipeline could not prove this hand was fully reconstructed."


def _completion_clearing_action(
    status: CompletionStatus, evidence: CompletionEvidence
) -> str:
    if status == "partial":
        if evidence.partial_start is not True and evidence.partial_end is not True:
            # No truncation is recorded, so naming a fuller recording is not an
            # action the operator can take. `partial` is sticky by design.
            return (
                "Nothing clears this on this hand. A partial classification is "
                "permanent, and importing the same source again cannot weaken it. To "
                "produce new evidence, "
                f"{NEW_RECONSTRUCTION_STEPS} The hand can still be inspected and "
                "corrected in the meantime."
            )
        return (
            "Fill in the missing facts under Import validation → Edit this hand → "
            "Cards, board, or pot (and edit any missing actions), acknowledge "
            "remaining source warnings, then use Other fixes → Finalize incomplete "
            "hand to attest that you reconstructed the whole hand yourself — even "
            "when the recording joined late. Sticky truncation and pipeline "
            "rejection codes stay in the audit trail; only this finalize clears "
            f"them for study. Alternatively, {NEW_RECONSTRUCTION_STEPS}"
        )
    if status == "not_applicable":
        return (
            "Only a new reconstruction clears this: a reconstructed hand cannot be "
            "exempted from completion evidence, and its source cannot be changed to "
            f"manual after the fact. To rebuild it, {NEW_RECONSTRUCTION_STEPS}"
        )
    if evidence.rejection_codes:
        # Acknowledge cannot clear a rejection. Operator finalize can, once the
        # operator has reconstructed the gaps (late join, OCR holes, etc.).
        return (
            "The pipeline rejected "
            f"{', '.join(evidence.rejection_codes)}. A rejection cannot be "
            "acknowledged away. If you reconstructed the whole hand yourself, "
            "acknowledge any remaining source warnings, then use Import "
            "validation → Edit this hand → Finalize incomplete hand. "
            f"Alternatively, only a new reconstruction clears this: "
            f"{NEW_RECONSTRUCTION_STEPS} If the reconstruction reproduces the "
            "same code and you cannot fill the gaps, record a debugging issue "
            "instead of promoting the hand."
        )
    if not evidence.is_known:
        # Every hand the v13 migration classified is here: it leaves
        # completion_evidence at '{}' rather than fabricating evidence for
        # historical hands. There is nothing to correct and nothing to
        # acknowledge, and no writer reachable from the UI can attach evidence to
        # an existing hand, so naming Hand facts and the Source warnings
        # panel promised two actions that cannot clear it -- and the panel is not
        # even drawn, because it renders only when the evidence carries a code.
        return (
            "Only a new reconstruction clears this. No readable completion "
            "evidence is attached — hands stored before schema 13 carry none — so "
            f"there is nothing to correct or acknowledge here: {NEW_RECONSTRUCTION_STEPS} "
            "The hand can still be inspected, corrected, and given a debugging "
            "issue in the meantime; it just cannot become a study record."
        )
    if not evidence.unresolved_codes:
        # Readable evidence, no codes to acknowledge: what is missing is an
        # observed boundary, a terminal event, or a boundary confidence, and only
        # the pipeline writes those. Naming the Source warnings panel here sent the
        # operator to a panel that renders only when a code is present.
        return (
            "Only a new reconstruction clears this. The stored evidence records no "
            "warning to acknowledge — what it is missing is an observed hand "
            "boundary, a terminal event, or a boundary confidence, and only a "
            f"reconstruction writes those: {NEW_RECONSTRUCTION_STEPS}"
        )
    return (
        "Fix the flagged fields in Hand facts, then acknowledge each remaining "
        "source warning in the Source warnings panel. The hand becomes complete when "
        "both boundaries are observed and no unresolved warning remains."
    )


def _card_blockers(
    hand: Hand, evidence: CompletionEvidence, *, is_reconstructed: bool
) -> list[StudyBlocker]:
    """Defense in depth: Hand validates cards on write, so this guards rows written
    outside the model. Manual hands may legitimately omit hero cards.

    A row written outside the model cannot reach ``Hand`` with its bad value
    intact -- the model refuses it -- so ``_hand_from_row`` blanks the column and
    records what it read under ``UNREADABLE_CARDS_KEY``. That record is checked
    first: without it a hand-edited board silently became "no board recorded",
    which is a legitimate state for a manual hand and blocked nothing.

    Three unlike conditions raise this one code, and one sentence covered all
    three. "The hero and board cards are not a valid, unique set" is false of a
    column this build could not read back -- nobody knows whether that value is a
    valid set -- and false of a five-card board requirement that a legal
    three-card flop fails, where the cards are fine and it is the attested ending
    they contradict. The old clearing action was worse than the reason there: it
    said the board must hold 0, 3, 4, or 5 cards, which a three-card board
    already does, so following it verbatim could not clear the blocker.
    """
    causes: list[_Cause] = []
    unreadable = _unreadable_card_columns(evidence)
    if unreadable is not None:
        causes.append(
            _Cause(
                reason=(
                    "A stored card column of this hand holds a value this build "
                    "cannot read, so the cards shown for it are a blank fallback "
                    "rather than the stored record."
                ),
                clearing_action=(
                    "Open Hand facts and re-enter the hero and board cards; "
                    "saving the correction rewrites the column with a value this "
                    "build can read."
                ),
                detail=(unreadable,),
            )
        )
    else:
        problem = _card_problem(hand, is_reconstructed=is_reconstructed)
        if problem is not None:
            causes.append(
                _Cause(
                    reason="The hero and board cards are not a valid, unique set.",
                    clearing_action=(
                        "Open Hand facts and fix the hero and board cards; every "
                        "visible card must appear exactly once and the board must "
                        "hold 0, 3, 4, or 5 cards."
                    ),
                    detail=(problem,),
                )
            )
        elif has_operator_manual_completion(evidence):
            # Only enforce terminal/board agreement for operator-attested
            # terminals. Pipeline-observed showdown with a missing board is
            # already handled by completion_status; applying it broadly broke
            # every clean-hand fixture.
            op_terminal = evidence.extra.get("operator_terminal_event")
            effective_terminal = (
                op_terminal
                if isinstance(op_terminal, str) and op_terminal
                else evidence.terminal_event
            )
            if (
                effective_terminal == "showdown"
                and len((hand.board_cards or "").split()) != 5
            ):
                causes.append(
                    _Cause(
                        reason=(
                            "You attested that this hand ended in a showdown, but "
                            "its board does not hold five cards, so the cards "
                            "recorded and the ending recorded cannot both be right."
                        ),
                        clearing_action=(
                            "Open Hand facts and record the full five-card board "
                            "the showdown was played to. If the hand did not reach "
                            "a showdown, re-run Import validation → Edit this hand "
                            "→ Other fixes → Finalize incomplete hand and attest "
                            "the terminal event that actually ended it."
                        ),
                        detail=(
                            "operator terminal event is showdown but board_cards "
                            f"does not hold five cards (board={hand.board_cards!r})",
                        ),
                    )
                )
    return _blocker_from_causes("INVALID_HERO_OR_BOARD_CARDS", "cards", causes)


def _unreadable_column_blockers(evidence: CompletionEvidence) -> list[StudyBlocker]:
    """A stored column this build could not read blocks study, for every hand.

    ``db._hand_from_row`` degrades an unreadable non-card column to the model
    default and records the column with its stored text under
    ``UNREADABLE_HAND_COLUMNS_KEY`` — the same read-time channel the card
    columns use — so this blocker names the exact value instead of silently
    presenting a conservative fallback as the record. Unconditional, exactly as
    ``_card_blockers`` is: a hand-edited row is not anyone's own entry,
    whatever ``source_type`` it claims.
    """
    recorded = evidence.extra.get(UNREADABLE_HAND_COLUMNS_KEY)
    if not isinstance(recorded, dict) or not recorded:
        return []
    detail = tuple(
        f"{column} could not be read: {value}"
        for column, value in sorted((str(key), item) for key, item in recorded.items())
    )
    return [
        StudyBlocker(
            code="UNREADABLE_HAND_COLUMNS",
            category="facts",
            reason=(
                f"{len(recorded)} stored column(s) of this hand hold values this "
                "build cannot read, so the values shown for them are "
                "conservative fallbacks, not the stored record."
            ),
            clearing_action=(
                "Open Hand facts and re-enter the listed fields; saving "
                "the correction rewrites every editable column. A listed column "
                "that form does not edit (for example confidence_score or "
                "created_at) cannot be repaired in the product: keep the hand "
                "for inspection, or remove it with the 'Delete hand' control on "
                "its row in the session's hand list (Sessions → Hands)."
            ),
            detail=detail,
        )
    ]


def _unreadable_card_columns(evidence: CompletionEvidence) -> str | None:
    """Report a card column the store could not read back, with what it held."""
    recorded = evidence.extra.get(UNREADABLE_CARDS_KEY)
    if not isinstance(recorded, dict) or not recorded:
        return None
    return "; ".join(
        f"{column} could not be read: {value!r}"
        for column, value in sorted((str(key), item) for key, item in recorded.items())
    )


def _card_problem(hand: Hand, *, is_reconstructed: bool) -> str | None:
    try:
        if hand.hero_cards and hand.board_cards:
            parse_visible_cards(hand.hero_cards, hand.board_cards)
        elif hand.hero_cards:
            parse_hero_cards(hand.hero_cards)
        if hand.board_cards and len(parse_board_cards(hand.board_cards)) not in (
            _VALID_BOARD_COUNTS
        ):
            return "The board must hold 0, 3, 4, or 5 cards."
    except CardParseError as exc:
        return str(exc)
    if not is_reconstructed:
        return None
    try:
        # parse_hero_cards already refuses any count other than two, so the
        # explicit length check below is redundant today. It is kept as a second
        # line of defence: if the parser ever became lenient, a reconstructed hand
        # with one or three hero cards must still block rather than pass silently.
        if len(parse_hero_cards(hand.hero_cards)) != 2:
            return "A reconstructed hand must record exactly 2 cards for the hero."
    except CardParseError as exc:
        return str(exc)
    return None


_RECONSTRUCTION_ACTION = (
    "Only a new reconstruction clears this. Under Settings → ROI calibration, "
    "activate an ROI profile whose table layout matches this recording, then "
    f"{NEW_RECONSTRUCTION_STEPS} Correcting the table size by hand does not clear "
    "it: the layout claim lives in the pipeline's evidence, which only a "
    "reconstruction writes."
)


def _layout_blockers(hand: Hand, evidence: CompletionEvidence) -> list[StudyBlocker]:
    """Layout support is carried by the pipeline's evidence, and by the row beside it.

    The clearing action is composed from the causes actually present, because it
    used to be one fixed sentence covering four of them and was false for two.
    "Only a new reconstruction clears this ... Correcting the table size by hand
    does not clear it" is exactly right about the evidence-borne causes -- no
    writer reachable from the UI rewrites ``layout_supported`` or the evidence's
    own ``table_size`` -- and it was drawn verbatim over two causes that are not
    evidence-borne at all:

    * ``hand.table_size`` is an ordinary editable column, so typing it in Correct
      hand facts removes that detail and, when it was the only one, cleared the
      whole blocker with the text still saying that action does nothing;
    * ``hero_seat_mismatch`` is an acknowledgeable warning, so one press of
      Acknowledge in the Source warnings panel dropped it out of
      ``unresolved_codes`` and cleared the blocker the same way.

    Naming an action the product cannot perform, while withholding the one it
    can, is the failure PLAN.md's "a blocker never names an action the product
    cannot perform" rule exists to prevent, and it read as "this hand is beyond
    repair" to an operator holding the fix.

    ``hero_seat_mismatch`` is only an acknowledgeable warning when the pipeline
    raised it AS a warning. The same code can arrive in ``rejection_codes``, where
    nothing can accept it -- ``acknowledge_codes`` drops it and ``app.py`` draws no
    Acknowledge button -- and keying this line on ``unresolved_codes``, which mixes
    both kinds, printed "Accept hero_seat_mismatch with Acknowledge" next to
    COMPLETION_NOT_COMPLETE and UNRESOLVED_SOURCE_WARNING on the same page, both
    correctly saying a rejection cannot be acknowledged or corrected away. That
    was ``_source_warning_blockers``' repaired defect surviving in a second
    consumer, which is why the split now lives on ``CompletionEvidence`` and is
    enforced (``test_no_consumer_prescribes_an_action_from_unresolved_codes``)
    rather than applied one consumer at a time.

    The recorded table size is also compared against the evidence's, which
    nothing did: the two columns could disagree outright -- a hand recording 9
    seats against evidence for 6 -- and the gate was satisfied by any typed
    value, so "record the table size" was a box to tick rather than a fact to
    state.

    The reason is now composed the same way, for the same reason the action had
    to be. "The seating layout for this hand was not confirmed" describes the
    evidence-borne causes and is false of the other two: a hand whose only fault
    is a blank ``hand.table_size`` column has a layout the reconstruction DID
    confirm, and a hand whose two seat counts disagree has two confirmations
    rather than none. An operator told the layout was never confirmed reaches for
    a re-run; an operator told the two records disagree reaches for the one that
    is wrong.
    """
    causes: list[_Cause] = []
    evidence_detail: list[str] = []
    if evidence.layout_supported is not True:
        evidence_detail.append(f"layout_supported={evidence.layout_supported}")
    if evidence.table_size is None or not 2 <= evidence.table_size <= 10:
        evidence_detail.append(f"evidence.table_size={evidence.table_size}")
    if evidence_detail:
        causes.append(
            _Cause(
                reason=(
                    "The reconstruction did not confirm the seating layout for "
                    "this hand, so seat, position, and hero attribution cannot be "
                    "trusted."
                ),
                clearing_action=_RECONSTRUCTION_ACTION,
                detail=tuple(evidence_detail),
            )
        )
    if hand.table_size is None:
        causes.append(
            _Cause(
                reason=(
                    "This hand does not record how many seats were at the table, "
                    "so position cannot be derived from its seat numbers."
                ),
                clearing_action=(
                    "Record the table size in Hand facts: that column is the "
                    "hand's own, and typing it clears this line."
                ),
                detail=("hand.table_size is not recorded",),
            )
        )
    elif evidence.table_size is not None and hand.table_size != evidence.table_size:
        causes.append(
            _Cause(
                reason=(
                    "The seat count this hand records and the seat count the "
                    "reconstruction observed disagree, so seat, position, and hero "
                    "attribution rest on whichever of the two is right."
                ),
                clearing_action=(
                    "The recorded table size and the reconstructed one disagree. Set the "
                    "table size in Hand facts to the seat count the recording "
                    "shows, or re-run the reconstruction if the evidence is the wrong one."
                ),
                detail=(
                    f"hand.table_size={hand.table_size} disagrees with "
                    f"evidence.table_size={evidence.table_size}",
                ),
            )
        )
    if "hero_seat_mismatch" in evidence.unresolved_warning_codes:
        causes.append(
            _Cause(
                reason=(
                    "The reconstruction flagged that the hero was not in the seat "
                    "it expected, so hero attribution is unconfirmed."
                ),
                clearing_action=(
                    "Accept hero_seat_mismatch with Acknowledge in the Source warnings "
                    "panel, which records it as an auditable correction, or re-run the "
                    "reconstruction to remove it."
                ),
                detail=("hero_seat_mismatch",),
            )
        )
    elif "hero_seat_mismatch" in evidence.unresolved_rejection_codes:
        causes.append(
            _Cause(
                reason=(
                    "The reconstruction REJECTED this hand's hero seat, so hero "
                    "attribution is unconfirmed."
                ),
                clearing_action=(
                    "The pipeline REJECTED hero_seat_mismatch, which is a refusal rather "
                    f"than a note you can accept: {_RECONSTRUCTION_ACTION}"
                ),
                detail=("hero_seat_mismatch (rejected)",),
            )
        )
    return _blocker_from_causes("UNSUPPORTED_TABLE_LAYOUT", "layout", causes)


# The settlement editor, named once for the conditions whose fix really is
# "correct what the cross-check flagged, then save".
_SETTLEMENT_ACTION = (
    "Open Import validation → Edit this hand → Other fixes → "
    "Chip stacks / accounting, fix the flagged contributions or awards, "
    "and save the settlement until its status reads reconciled."
)


def _accounting_blockers(
    accounting: AccountingReconciliation | None, accounting_error: str | None
) -> list[StudyBlocker]:
    """One code, several conditions, and the operator is told which one fired.

    ``stale_accounting_verdict`` split the first of these out already, for
    exactly the reason repeated below: "The chip ledger does not reconcile" was
    being printed over a hand whose ledger reconciles perfectly, with an empty
    issue list beneath it, and the operator could not tell that from a genuine
    chip defect. The same sentence was still being printed over two more
    conditions it is not true of.

    * A ``LedgerError`` means the ledger REFUSED TO BUILD, which is not a failure
      to balance; the clearing action already branched on it while the reason did
      not.
    * A hand with no settlement row at all -- a hand entered here whose
      accounting panel has never been opened -- has nothing to disagree with, and
      its chips usually balance. Telling that operator the ledger does not
      reconcile sends them hunting a defect that is not there, and the action
      told them to "fix the flagged contributions or awards" when nothing is
      flagged and nothing needs fixing but the missing save.

    A ledger that balances against a RECORDED figure that contradicts it keeps
    the original sentence. That is a deliberate ruling, pinned by
    ``test_a_real_accounting_defect_is_still_reported_as_one``: a hand whose
    recorded pot contradicts its own action line has a genuine defect and the
    reconciliation, not merely the label, is what failed.
    """
    if accounting is not None and accounting.is_authoritative and not accounting_error:
        return []
    if accounting_error:
        # A LedgerError means the ledger REFUSED to build -- a player commits more
        # than their recorded stack, an action references an identity that is not
        # seated -- and none of that is editable in the Accounting reconciliation
        # panel, which only edits dead money, the rake policy, awards and refunds. The
        # blocker used to name that panel for both branches, so following it literally
        # could not clear the blocker; the panel's own inline caption already said so.
        return _blocker_from_causes(
            "ACCOUNTING_NOT_AUTHORITATIVE",
            "accounting",
            [
                _Cause(
                    reason=(
                        "This hand's chip ledger could not be built at all, so the "
                        "pot, result, and every derived number are unproven."
                    ),
                    clearing_action=(
                        "Open Import validation → Edit this hand and correct the "
                        "stack sizes, action amounts, or players the ledger "
                        "rejected — the Accounting reconciliation panel cannot "
                        "change them. Reopen Other fixes → Chip stacks / accounting "
                        "afterwards and save the settlement until its status reads "
                        "reconciled."
                    ),
                    detail=(accounting_error,),
                )
            ],
        )
    if accounting is None:
        return _blocker_from_causes(
            "ACCOUNTING_NOT_AUTHORITATIVE",
            "accounting",
            [
                _Cause(
                    reason=(
                        "No accounting reconciliation was produced for this hand, "
                        "so nothing has established its pot or its result."
                    ),
                    clearing_action=(
                        "Open Import validation → Edit this hand → Other fixes → "
                        "Chip stacks / accounting (Accounting reconciliation) and "
                        "save the settlement. If no reconciliation appears at all, "
                        "record a debugging issue against the hand instead of "
                        "promoting it."
                    ),
                )
            ],
        )
    stale_verdict = stale_accounting_verdict(accounting)
    if stale_verdict is not None:
        # Said in the blocker rather than left to the operator to infer, because
        # the alternative is what shipped: a hand blocked by an accounting
        # verdict, no issue to show, and a save that appears to do nothing.
        # `persist_reconciliation` repairs the recorded figures and records the
        # verdict it reached BEFORE that repair, so the pass that fixes the
        # record and the pass that blesses it are two passes, and only one of
        # them is announced.
        cause = _Cause(
            reason=(
                "This hand's chip ledger reconciles, but no saved settlement "
                "records that verdict, so the pot and result are not yet "
                "proven by anything durable."
            ),
            clearing_action=(
                "Open Import validation → Edit this hand → Other fixes → "
                "Chip stacks / accounting and press Save and reconcile once. "
                "The ledger already balances; the stored settlement status is "
                "what is out of date."
            ),
            detail=(
                f"Settlement status reads {stale_verdict.status!r}, not 'reconciled'.",
            ),
        )
        return _blocker_from_causes("ACCOUNTING_NOT_AUTHORITATIVE", "accounting", [cause])
    detail = tuple(accounting.issues[:4])
    ledger = accounting.ledger
    ledger_reconciles = ledger is not None and (
        ledger.is_settled and ledger.is_balanced and ledger.is_legal
    )
    # `_cross_check` contributes exactly one issue for an absent settlement, so a
    # second issue means there is a real finding underneath and the established
    # sentence stands. `reconcile_persisted_hand` always supplies a ledger; a
    # reconciliation that arrives without one tells us nothing about the chips,
    # and also keeps the sentence that claims the least.
    if ledger_reconciles and accounting.settlement is None and len(detail) <= 1:
        cause = _Cause(
            reason=(
                "The chips themselves balance, but no settlement has ever been "
                "saved for this hand, so nothing durable records who was paid "
                "what and the pot and result stay unproven."
            ),
            clearing_action=(
                "Open Import validation → Edit this hand → Other fixes → "
                "Chip stacks / accounting (Accounting reconciliation) and press "
                "Save and reconcile once. There is nothing to correct first; what "
                "is missing is the saved settlement itself."
            ),
            detail=detail,
        )
    else:
        cause = _Cause(
            reason=(
                "The chip ledger does not reconcile, so the pot, result, and every "
                "derived number are unproven."
            ),
            clearing_action=_SETTLEMENT_ACTION,
            detail=detail,
        )
    return _blocker_from_causes("ACCOUNTING_NOT_AUTHORITATIVE", "accounting", [cause])


def _assumption_blockers(
    hand: Hand, accounting: AccountingReconciliation | None
) -> list[StudyBlocker]:
    """Block a reconstructed hand whose reconciliation rests on a declared assumption.

    Scoped, through ``unattested_assumption_dependence``, to hands this operator
    did not enter. On a hand typed in here the operator IS the source of truth:
    an ante, a dead blind, a straddle from a seat that left, and the room's rake
    are all the same person's own entry, so there is no pipeline claim for a
    declaration to outrank. The dependence is still MEASURED on such a hand and
    still recorded on the reconciliation, so it can be disclosed; it is only
    never blocked.

    "Manual" is not by itself that argument. An import payload declaring
    ``source_type: manual`` with no evidence is byte-identical to a genuine
    manual export, so it landed a fabricated hero result exempt from this blocker
    -- and from every other reconstructed-hand blocker -- by claiming to be
    something no guard can disprove. What it cannot claim is that this operator
    entered it: import stamps every hand it lands, which is the same reason it
    refuses to land ``reviewed`` and resets acknowledged codes.

    ``accounting.assumption_dependence`` is DERIVED on every read (see
    ``hand_accounting._derive_assumption_dependence``), so no writer can bypass
    it. That matters more than it sounds: the previous disclosure lived entirely
    in ``upsert_hand_settlement``, so a settlement row written any other way --
    a hand-edited database, a future writer -- carried a reconciled, authoritative,
    study-ready hand with no disclosure at all. The verdict now lives with the
    reader, which every readiness surface goes through.

    Whole-hand confirmation deliberately does not clear it. USER_CONFIRMATION_MISSING
    asks "is this hand correct?"; this asks the narrower question "do you assert
    these specific unobserved chips?", and one tick answering the first was
    exactly how eight rounds of declared-chip disclosures were cleared without
    anyone reading them. The attestation is keyed on the code, and the code
    carries both the declaration and the measured chip movement, so it covers
    that declaration and no other -- an attestation earned against 0.01 chips of
    rake does not survive the same policy taking 80.01 chips off a grown pot, and
    one earned against a 50% rake does not survive a 25% rake that removes the
    same 40 chips off a doubled action line.

    It is answered from ``confirmed_assumption_codes`` and NEVER from
    ``acknowledged_codes``. Sharing the pipeline channel meant the generic
    one-click "Acknowledge" in the Source warnings panel -- drawn for every
    unacknowledged warning code, captioned as a pipeline note, saying nothing
    about chips -- cleared this blocker as well, which is the same one-tick
    bypass in a second costume.
    """
    pending = unattested_assumption_dependence(hand, accounting)
    if not pending:
        return []
    names = ", ".join(dependence.input_name for dependence in pending)
    return [
        StudyBlocker(
            code="ACCOUNTING_ASSUMPTION_DEPENDENT",
            category="accounting",
            reason=(
                "What this hand reports rests on settlement inputs you declared "
                f"({names}). Withdrawing them stops it reconciling or changes the "
                "figures it reports, so the pot, the rake, who was paid, and the "
                "hero result are not established by the recording alone."
            ),
            clearing_action=(
                "Open Import validation → Edit this hand → Other fixes → "
                "Chip stacks / accounting (Accounting reconciliation) and press "
                "'Confirm this assumption' beside each listed assumption, which "
                "records the exact chip movement you are attesting to. Finishing "
                "validation as a whole does not clear this. If the chips did not "
                "move that way, correct the declared winner, the rake policy or "
                "the dead money there instead and save the settlement — a "
                "declaration that changes nothing is never disclosed."
            ),
            detail=tuple(
                f"{dependence.describe()} [{dependence.code}]" for dependence in pending
            ),
        )
    ]


def _requires_proven_regression(issue: HandIssue) -> bool:
    """Will the writer refuse to close this issue on a resolution note alone?

    The category half is ``regression_promotion.is_release_blocking`` rather
    than a fourth copy of the set, so a category added to
    ``RELEASE_BLOCKING_ISSUE_TYPES`` cannot be enforced without also being
    disclosed. The unreadable half mirrors ``PokerDatabase._regression_blocker``
    and ``resolution_blocker``, which both gate a row whose categories could not
    be read: the reader's fallback is ``other``, outside the set, so believing it
    would let row damage clear the gate. That branch cannot be shared with them
    because both need the database; this predicate answers from the record the
    readiness pass was already handed.
    """
    if "issue_types" in issue.unreadable_columns:
        return True
    return is_release_blocking(list(issue.issue_types))


def _issue_detail(issue: HandIssue) -> str:
    categories = ", ".join(issue.issue_types)
    if "issue_types" in issue.unreadable_columns:
        return (
            f"#{issue.id}: categories unreadable (salvaged as {categories}), so it "
            "is gated as release-blocking and needs a proven regression to close"
        )
    if _requires_proven_regression(issue):
        return (
            f"#{issue.id}: {categories} — release-blocking; needs a proven "
            "regression to close"
        )
    return f"#{issue.id}: {categories}"


def _issue_blockers(issues: list[HandIssue]) -> list[StudyBlocker]:
    """Name the whole precondition on closing an issue, not the half a note covers.

    Seven of the nine categories the flagging control offers are in
    ``RELEASE_BLOCKING_ISSUE_TYPES``, and ``resolve_hand_issue`` refuses one of
    those on a resolution note alone until a linked regression has been observed
    both failing before the fix and passing after it. This blocker used to say
    only "resolve each issue with resolution notes", so following it literally
    produced a refusal naming a promotion no control performs -- the same shape
    as the four clearing actions PLAN.md already records as having named an
    action the product could not perform, and the reason
    ``test_every_control_a_clearing_action_names_exists_in_the_app`` is not
    enough on its own: it proves the control is drawn, not that the writer
    behind it will accept the submission.

    Until the promotion has a control, the honest thing to name is the procedure
    that does exist. Saying so here also stops the operator reading a permanent
    gate as a step they have not found yet.
    """
    open_issues = [issue for issue in issues if issue.status == "open"]
    if not open_issues:
        return []
    gated = [issue for issue in open_issues if _requires_proven_regression(issue)]
    clearing_action = (
        "Resolve each issue in the Saved debugging issue queue with resolution "
        "notes; the issue and its evidence snapshot are retained as history."
    )
    if gated:
        falls = "falls" if len(gated) == 1 else "fall"
        clearing_action += (
            f" {len(gated)} of them {falls} in a release-blocking category, and a "
            "resolution note alone will not close those: each needs a regression "
            "case linked to the issue and observed BOTH failing before the fix "
            "and passing after it. No control in the app creates one — the "
            "procedure is docs/RUNBOOKS.md section 12 "
            "(promote_issue_to_regression, then record_regression_observation "
            "twice) — so this hand stays out of study until that is done."
        )
    return [
        StudyBlocker(
            code="OPEN_DEBUGGING_ISSUE",
            category="issues",
            reason=(
                f"{len(open_issues)} unresolved debugging issue(s) are recorded "
                "against this hand."
            ),
            clearing_action=clearing_action,
            detail=tuple(_issue_detail(issue) for issue in open_issues),
        )
    ]


def _source_warning_blockers(evidence: CompletionEvidence) -> list[StudyBlocker]:
    """A rejection is not a warning, and it is not acknowledgeable.

    ``unresolved_codes`` mixes both kinds -- every rejection code is permanently
    unresolved, because ``acknowledge_codes`` refuses one -- so this blocker used
    to call a pipeline refusal a "source warning" and tell the operator to
    "acknowledge the remaining codes in the Source warnings panel", which
    ``acknowledge_codes`` refuses and for which ``app.py`` draws no Acknowledge
    button. It also directly contradicted COMPLETION_NOT_COMPLETE, rendered on
    the same page, which reports the same code correctly. The blocker still fires
    on exactly the same evidence; only what it says about it is now true.

    After operator finalize, rejection codes remain in the audit trail but no
    longer block study — that attestation is the override for late-join /
    operator-filled reconstructions.
    """
    warnings = evidence.unresolved_warning_codes
    rejections = (
        ()
        if has_operator_manual_completion(evidence)
        else evidence.unresolved_rejection_codes
    )
    causes: list[_Cause] = []
    if rejections:
        causes.append(
            _Cause(
                reason=(
                    f"The pipeline REJECTED {len(rejections)} of this hand's source "
                    "fact(s), which is a refusal rather than a note you can accept."
                ),
                clearing_action=(
                    "A rejection cannot be acknowledged away. If you reconstructed the "
                    "whole hand yourself, acknowledge remaining warnings then use "
                    "Finalize incomplete hand. Alternatively, only a new reconstruction "
                    f"clears {', '.join(rejections)}: {NEW_RECONSTRUCTION_STEPS}"
                ),
                detail=tuple(rejections),
            )
        )
    if warnings:
        causes.append(
            _Cause(
                reason=(
                    f"The pipeline flagged {len(warnings)} unresolved source warning(s)."
                ),
                clearing_action=(
                    f"For {', '.join(warnings)}: fix each listed field in Correct hand "
                    "facts, then acknowledge the remaining codes in the Source warnings "
                    "panel. Acknowledging records the accepted code as an auditable "
                    "correction."
                ),
                detail=tuple(warnings),
            )
        )
    return _blocker_from_causes("UNRESOLVED_SOURCE_WARNING", "completion", causes)


def _coaching_blockers(reviews: list[RetainedReview]) -> list[StudyBlocker]:
    """A current review that predates the staling event no longer clears the block.

    Both retained coaching tables are considered: the legacy ``hand_reviews`` rows
    are staled by the same correction path and are still rendered in the Hands
    workspace, so a stale one is stale evidence presented as current.

    ``is_stale`` carries more than one cause and only ``stale_reason`` tells them
    apart, so that column decides what this says. A correction invalidated the
    answer, which means the facts moved under it and re-running is the point; or
    the answer failed its own grounding check -- it named a card the hand never
    held, or quoted a solver-shaped frequency with no retained solver evidence --
    which says something about the answer itself, not about the hand; or the row
    records nothing, which is what a hand-edited or pre-migration row looks like
    and is not an invitation to guess. All three leave the same two ways out, so
    only the sentence differs.

    The recorded text is placed in the detail rather than inlined into the
    reason: a grounding failure quotes the rejected claim back, and a rejected
    claim is routinely a percentage, which a blocker reason may never contain.
    """
    stale = [review for review in reviews if review.is_stale]
    current = [review.created_at for review in reviews if not review.is_stale]
    if not stale:
        return []
    governing = max(stale, key=lambda review: review.created_at)
    newest_current = max(current) if current else None
    if newest_current is not None and newest_current >= governing.created_at:
        return []
    recorded = (governing.stale_reason or "").strip()
    if recorded.startswith(UNGROUNDED_STALE_PREFIX):
        # The marker the writer stamps on a rejected answer, imported rather than
        # restated, because a second copy of the sentence drifts from the first.
        reason = (
            "The saved coaching for this hand failed its own grounding check: it "
            "asserted facts the prompt it was generated from does not support, so "
            "it is retained as history rather than as analysis."
        )
    elif recorded:
        reason = (
            "The saved coaching for this hand was invalidated by a later change "
            "to the hand or its session, and has not been re-run."
        )
    else:
        reason = (
            "The saved coaching for this hand is marked not current and has not "
            "been re-run; the review does not record what made it stale."
        )
    return _blocker_from_causes(
        "STALE_COACHING_EVIDENCE",
        "coaching",
        [
            _Cause(
                reason=reason,
                clearing_action=(
                    "Re-run coaching in Analyze → AI coach, or press Discard stale coaching "
                    "there. Re-running keeps the stale review visible as retained "
                    "history; discarding is the way out when no coaching provider is "
                    "configured, which is every imported hand's starting state."
                ),
                detail=(recorded,) if recorded else (),
            )
        ],
    )


def _solver_blockers(runs: list[SolverRun]) -> list[StudyBlocker]:
    """A failed or cancelled solve is not stale evidence being shown as current.

    ``status == 'stale'`` is reached by four unlike routes and the blocker
    asserted one of them. A correction to the hand or its session stales a
    completed run and records why in ``error_message``; so does flagging the hand
    for debugging, and so did the v20 dead-money re-derivation. A cancellation
    that finishes lands in the same status with no message and no result -- there
    was nothing to invalidate. And a completed run whose stored row this build
    cannot read is degraded to ``stale`` deliberately, because ``completed`` would
    grant study evidence over a blob nobody can read.

    So the sentence comes from the run's own record -- the message each staling
    writer leaves, and the columns the reader had to give up -- rather than from
    this line. The recorded message goes in the detail, not the reason, for the
    same reason the coaching one does: a solver message may carry a percentage.
    """
    stale = [run for run in runs if run.status in _STALE_SOLVER_STATUSES]
    if not stale:
        return []
    newest_stale = max(run.created_at for run in stale)
    # ``>=``, matching _coaching_blockers. The two describe the same situation and
    # used to disagree on ties: a re-run recorded at the same timestamp as the
    # staling event cleared the coaching blocker but not this one, which then named
    # "Re-run the solve" as the action for something already done.
    if any(
        run.status == "completed" and run.created_at >= newest_stale for run in runs
    ):
        return []
    finished = [run for run in stale if run.status == "stale"]
    if not finished:
        # Only a cancellation is in flight. Nothing was invalidated by a
        # correction and there is no saved result, so saying so would be false --
        # and the Delete stale run control is not drawn while a run is cancelling.
        return _blocker_from_causes(
            "STALE_SOLVER_EVIDENCE",
            "solver",
            [
                _Cause(
                    reason=(
                        "A solver run for this hand is still being cancelled, so no "
                        "solver evidence for it is current."
                    ),
                    clearing_action=(
                        "Wait for the cancellation to finish in Analyze → TexasSolver, then "
                        "either re-run the solve or press Delete stale run beside the "
                        "cancelled run."
                    ),
                )
            ],
        )
    governing = max(finished, key=lambda run: run.created_at)
    recorded = (governing.error_message or "").strip()
    detail: tuple[str, ...] = (recorded,) if recorded else ()
    if governing.unreadable_columns:
        reason = (
            "A saved solver run for this hand holds stored values this build "
            "cannot read, so it cannot stand as current solver evidence."
        )
        detail = (
            *detail,
            *(f"{column} could not be read" for column in governing.unreadable_columns),
        )
    elif recorded:
        reason = (
            "A saved solver result for this hand was invalidated by a later change "
            "to the hand or its session, and has not been re-run."
        )
    else:
        reason = (
            "A solver run for this hand ended without leaving a usable result, so "
            "there is no current solver evidence for it; the run does not record why."
        )
    return _blocker_from_causes(
        "STALE_SOLVER_EVIDENCE",
        "solver",
        [
            _Cause(
                reason=reason,
                clearing_action=(
                    "Re-run the solve in Analyze → TexasSolver, or press Delete stale run "
                    "beside it. Deleting is the only clearing action when the hand is "
                    "no longer solver-eligible, which is why the control exists."
                ),
                detail=detail,
            )
        ],
    )
