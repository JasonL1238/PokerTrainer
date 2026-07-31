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
    SolverRun,
)
from poker_tracker.services.hand_accounting import (
    AccountingReconciliation,
    AssumptionDependence,
)

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
    """

    @property
    def is_stale(self) -> bool: ...

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
        blockers.append(
            StudyBlocker(
                code="USER_CONFIRMATION_MISSING",
                category="confirmation",
                reason=(
                    "You have not confirmed that this reconstructed hand is correct."
                    if is_reconstructed
                    else (
                        "This hand arrived in an import payload, and you have not "
                        "confirmed that it is correct."
                    )
                ),
                clearing_action=(
                    "Tick 'I have read the evidence above and confirm this hand is "
                    "correct' under Fix & confirm → Looks good — Approve."
                ),
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
            "Fill in the missing facts under Fix & confirm → Edit the hand — Fix → "
            "Other fixes → Hand facts (and edit any missing actions), acknowledge "
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
            "acknowledge any remaining source warnings, then use Fix & confirm → "
            "Edit the hand — Fix → Other fixes → Finalize incomplete hand. "
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
    """
    problem = _unreadable_card_columns(evidence) or _card_problem(
        hand, is_reconstructed=is_reconstructed
    )
    if problem is None and has_operator_manual_completion(evidence):
        # Only enforce terminal/board agreement for operator-attested terminals.
        # Pipeline-observed showdown with a missing board is already handled by
        # completion_status; applying it broadly broke every clean-hand fixture.
        op_terminal = evidence.extra.get("operator_terminal_event")
        effective_terminal = (
            op_terminal
            if isinstance(op_terminal, str) and op_terminal
            else evidence.terminal_event
        )
        if effective_terminal == "showdown" and len((hand.board_cards or "").split()) != 5:
            problem = (
                "operator terminal event is showdown but board_cards does not "
                f"hold five cards (board={hand.board_cards!r})"
            )
    if problem is None:
        return []
    return [
        StudyBlocker(
            code="INVALID_HERO_OR_BOARD_CARDS",
            category="cards",
            reason="The hero and board cards are not a valid, unique set.",
            clearing_action=(
                "Open Hand facts and fix the hero and board cards; every "
                "visible card must appear exactly once and the board must hold 0, 3, "
                "4, or 5 cards."
            ),
            detail=(problem,),
        )
    ]


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
    """
    detail: list[str] = []
    actions: list[str] = []
    if evidence.layout_supported is not True:
        detail.append(f"layout_supported={evidence.layout_supported}")
    if evidence.table_size is None or not 2 <= evidence.table_size <= 10:
        detail.append(f"evidence.table_size={evidence.table_size}")
    if detail:
        actions.append(_RECONSTRUCTION_ACTION)
    if hand.table_size is None:
        detail.append("hand.table_size is not recorded")
        actions.append(
            "Record the table size in Hand facts: that column is the "
            "hand's own, and typing it clears this line."
        )
    elif evidence.table_size is not None and hand.table_size != evidence.table_size:
        detail.append(
            f"hand.table_size={hand.table_size} disagrees with "
            f"evidence.table_size={evidence.table_size}"
        )
        actions.append(
            "The recorded table size and the reconstructed one disagree. Set the "
            "table size in Hand facts to the seat count the recording "
            "shows, or re-run the reconstruction if the evidence is the wrong one."
        )
    if "hero_seat_mismatch" in evidence.unresolved_warning_codes:
        detail.append("hero_seat_mismatch")
        actions.append(
            "Accept hero_seat_mismatch with Acknowledge in the Source warnings "
            "panel, which records it as an auditable correction, or re-run the "
            "reconstruction to remove it."
        )
    elif "hero_seat_mismatch" in evidence.unresolved_rejection_codes:
        detail.append("hero_seat_mismatch (rejected)")
        actions.append(
            "The pipeline REJECTED hero_seat_mismatch, which is a refusal rather "
            f"than a note you can accept: {_RECONSTRUCTION_ACTION}"
        )
    if not detail:
        return []
    return [
        StudyBlocker(
            code="UNSUPPORTED_TABLE_LAYOUT",
            category="layout",
            reason=(
                "The seating layout for this hand was not confirmed, so seat, "
                "position, and hero attribution cannot be trusted."
            ),
            clearing_action=" ".join(actions),
            detail=tuple(detail),
        )
    ]


def _accounting_blockers(
    accounting: AccountingReconciliation | None, accounting_error: str | None
) -> list[StudyBlocker]:
    if accounting is not None and accounting.is_authoritative and not accounting_error:
        return []
    detail: tuple[str, ...] = (accounting_error,) if accounting_error else ()
    if not detail and accounting is not None:
        detail = tuple(accounting.issues[:4])
    # A LedgerError means the ledger REFUSED to build -- a player commits more
    # than their recorded stack, an action references an identity that is not
    # seated -- and none of that is editable in the Accounting reconciliation
    # panel, which only edits dead money, the rake policy, awards and refunds. The
    # blocker used to name that panel for both branches, so following it literally
    # could not clear the blocker; the panel's own inline caption already said so.
    clearing_action = (
        "Open Fix & confirm → Edit the hand — Fix and correct the stack sizes, "
        "action amounts, or players the ledger rejected — the Accounting "
        "reconciliation panel cannot change them. Reopen Other fixes → "
        "Chip stacks / accounting afterwards and save the settlement until its "
        "status reads reconciled."
        if accounting_error
        else (
            "Open Fix & confirm → Edit the hand — Fix → Other fixes → "
            "Chip stacks / accounting, fix the flagged contributions or awards, "
            "and save the settlement until its status reads reconciled."
        )
    )
    return [
        StudyBlocker(
            code="ACCOUNTING_NOT_AUTHORITATIVE",
            category="accounting",
            reason=(
                "The chip ledger does not reconcile, so the pot, result, and every "
                "derived number are unproven."
            ),
            clearing_action=clearing_action,
            detail=detail,
        )
    ]


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
                "Open Fix & confirm → Edit the hand — Fix → Other fixes → "
                "Chip stacks / accounting (Accounting reconciliation) and press "
                "'Confirm this assumption' beside each listed assumption, which "
                "records the exact chip movement you are attesting to. Approving "
                "the hand as a whole does not clear this. If the chips did not "
                "move that way, correct the declared winner, the rake policy or "
                "the dead money there instead and save the settlement — a "
                "declaration that changes nothing is never disclosed."
            ),
            detail=tuple(
                f"{dependence.describe()} [{dependence.code}]" for dependence in pending
            ),
        )
    ]


def _issue_blockers(issues: list[HandIssue]) -> list[StudyBlocker]:
    open_issues = [issue for issue in issues if issue.status == "open"]
    if not open_issues:
        return []
    return [
        StudyBlocker(
            code="OPEN_DEBUGGING_ISSUE",
            category="issues",
            reason=(
                f"{len(open_issues)} unresolved debugging issue(s) are recorded "
                "against this hand."
            ),
            clearing_action=(
                "Resolve each issue in the Saved debugging issue queue with resolution "
                "notes; the issue and its evidence snapshot are retained as history."
            ),
            detail=tuple(
                f"#{issue.id}: {', '.join(issue.issue_types)}" for issue in open_issues
            ),
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
    unresolved = tuple([*rejections, *warnings])
    if not unresolved:
        return []

    reasons: list[str] = []
    actions: list[str] = []
    if rejections:
        reasons.append(
            f"The pipeline REJECTED {len(rejections)} of this hand's source "
            "fact(s), which is a refusal rather than a note you can accept."
        )
        actions.append(
            "A rejection cannot be acknowledged away. If you reconstructed the "
            "whole hand yourself, acknowledge remaining warnings then use "
            "Finalize incomplete hand. Alternatively, only a new reconstruction "
            f"clears {', '.join(rejections)}: {NEW_RECONSTRUCTION_STEPS}"
        )
    if warnings:
        reasons.append(
            f"The pipeline flagged {len(warnings)} unresolved source warning(s)."
        )
        actions.append(
            f"For {', '.join(warnings)}: fix each listed field in Correct hand "
            "facts, then acknowledge the remaining codes in the Source warnings "
            "panel. Acknowledging records the accepted code as an auditable "
            "correction."
        )
    return [
        StudyBlocker(
            code="UNRESOLVED_SOURCE_WARNING",
            category="completion",
            reason=" ".join(reasons),
            clearing_action=" ".join(actions),
            detail=unresolved,
        )
    ]


def _coaching_blockers(reviews: list[RetainedReview]) -> list[StudyBlocker]:
    """A current review that predates the staling event no longer clears the block.

    Both retained coaching tables are considered: the legacy ``hand_reviews`` rows
    are staled by the same correction path and are still rendered in the Hands
    workspace, so a stale one is stale evidence presented as current.
    """
    stale = [review.created_at for review in reviews if review.is_stale]
    current = [review.created_at for review in reviews if not review.is_stale]
    if not stale:
        return []
    newest_stale = max(stale)
    newest_current = max(current) if current else None
    if newest_current is not None and newest_current >= newest_stale:
        return []
    return [
        StudyBlocker(
            code="STALE_COACHING_EVIDENCE",
            category="coaching",
            reason=(
                "The saved coaching for this hand was invalidated by a later "
                "correction and has not been re-run."
            ),
            clearing_action=(
                "Re-run coaching in Analyze → AI coach, or press Discard stale coaching "
                "there. Re-running keeps the stale review visible as retained "
                "history; discarding is the way out when no coaching provider is "
                "configured, which is every imported hand's starting state."
            ),
        )
    ]


def _solver_blockers(runs: list[SolverRun]) -> list[StudyBlocker]:
    """A failed or cancelled solve is not stale evidence being shown as current."""
    stale = [run.created_at for run in runs if run.status in _STALE_SOLVER_STATUSES]
    if not stale:
        return []
    newest_stale = max(stale)
    # ``>=``, matching _coaching_blockers. The two describe the same situation and
    # used to disagree on ties: a re-run recorded at the same timestamp as the
    # staling event cleared the coaching blocker but not this one, which then named
    # "Re-run the solve" as the action for something already done.
    if any(
        run.status == "completed" and run.created_at >= newest_stale for run in runs
    ):
        return []
    if all(run.status != "stale" for run in runs):
        # Only a cancellation is in flight. Nothing was invalidated by a
        # correction and there is no saved result, so saying so would be false --
        # and the Delete stale run control is not drawn while a run is cancelling.
        return [
            StudyBlocker(
                code="STALE_SOLVER_EVIDENCE",
                category="solver",
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
        ]
    return [
        StudyBlocker(
            code="STALE_SOLVER_EVIDENCE",
            category="solver",
            reason=(
                "A saved solver result for this hand was invalidated by a later "
                "correction and has not been re-run."
            ),
            clearing_action=(
                "Re-run the solve in Analyze → TexasSolver, or press Delete stale run "
                "beside it. Deleting is the only clearing action when the hand is "
                "no longer solver-eligible, which is why the control exists."
            ),
        )
    ]
