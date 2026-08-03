"""Session stats and the metric-population layer the Insights surface reads.

The population layer exists because a single number computed over the whole
``hands`` table stands for four different epistemic states at once. A reviewed
manual hand, an unreconciled CV draft, a corrected CV hand nobody has confirmed
and a hand whose only "result" is a derivation from an unanswered settlement
declaration all contributed equally to one win rate, and the product printed it
as though it meant one thing. Nothing here computes a bare float: every metric
carries the population it was computed over, how much of that population
actually contributed, the evidence classes behind it, and a sample verdict, so a
caller cannot render the value without rendering what it stands for.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from poker_tracker.math.accounting import LedgerError
from poker_tracker.math.study_math import mean_confidence_interval
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import DUPLICATE_IMPORT_NOTE
from poker_tracker.persistence.models import CoachingResponse, Hand, HandReview, Session
from poker_tracker.services.hand_accounting import (
    AccountingReconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import accounting_is_established

AGGRESSIVE_ACTIONS = {"bet", "raise", "all-in"}
PASSIVE_ACTIONS = {"check", "call"}


# ---------------------------------------------------------------------------
# Re-imported copies
# ---------------------------------------------------------------------------

# Built from the note the importer writes rather than respelled, so the reader
# and the writer cannot drift into disagreeing about what a re-imported copy
# looks like. If ``DUPLICATE_IMPORT_NOTE`` is reworded, this pattern is reworded
# with it and a copy labelled by an older build stops matching -- which is the
# safe direction, because a session that fails to match is counted, exactly as it
# was before this existed, rather than silently vanishing from the totals.
_DUPLICATE_NOTE_PATTERN = re.compile(
    re.escape(DUPLICATE_IMPORT_NOTE.split("{other_id}")[0]) + r"(\d+)"
)


def duplicate_import_source(session: Session) -> int | None:
    """The session id this one was recorded as a re-imported copy of, if any.

    The marker is the note ``import_export._label_duplicate_import`` writes, and
    it is the product's own statement that these hands are already in this
    database under another session. Nothing else in the schema records it: import
    is an append and the copy is an ordinary session in every other respect.
    """
    match = _DUPLICATE_NOTE_PATTERN.search(session.notes or "")
    if match is None:
        return None
    duplicate_of = int(match.group(1))
    return None if duplicate_of == session.id else duplicate_of


def duplicate_import_sessions(sessions: Iterable[Session]) -> Mapping[int, int]:
    """``{copy session id: the session id it duplicates}`` over a session list.

    The copy is the one dropped from a total and the named session is the one
    kept, which is well defined however many copies exist: the importer always
    names the OLDEST matching session, so a third and fourth re-import both point
    at the original rather than at each other.
    """
    found: dict[int, int] = {}
    for session in sessions:
        if session.id is None:
            continue
        duplicate_of = duplicate_import_source(session)
        if duplicate_of is not None:
            found[session.id] = duplicate_of
    return found


@dataclass(frozen=True)
class SessionStats:
    """Basic manual stats, not HUD-grade poker statistics."""

    hand_count: int
    hands_with_result: int
    total_hero_bb: float
    average_hero_bb: float
    bb_per_100: float
    # 95% confidence interval on bb/100 (None with fewer than 2 recorded hands).
    # Session samples are small, so this is a reminder of variance, not proof of a winrate.
    bb_per_100_ci: tuple[float, float] | None = None
    biggest_winning_hands: list[Hand] = field(default_factory=list)
    biggest_losing_hands: list[Hand] = field(default_factory=list)
    hands_by_tag: dict[str, int] = field(default_factory=dict)
    hands_by_review_status: dict[str, int] = field(default_factory=dict)
    action_counts_by_type: dict[str, int] = field(default_factory=dict)
    aggression_count: int = 0
    passive_count: int = 0
    reconciled_result_count: int = 0
    observed_result_count: int = 0


def compute_session_stats(
    db: PokerDatabase,
    session_id: int,
    hands: list[Hand] | None = None,
) -> SessionStats:
    """Compute basic/manual session stats from stored hands and actions.

    When ``hands`` is provided, stats are computed from that list instead of
    every hand in the session (used to exclude non-study hands from coaching).
    """
    if hands is None:
        hands = db.fetch_hands_by_session(session_id)
    # Only hands with a recorded result count toward result stats; treating a
    # missing result as 0 BB would deflate the averages.
    result_hands: list[Hand] = []
    reconciled_result_count = 0
    observed_result_count = 0
    for hand in hands:
        # One resolver, shared with the population layer below, so the session
        # dashboard and Insights cannot disagree about which of a hand's two
        # possible results is the one being shown.
        resolved = resolve_hero_result(db, hand)
        if resolved.basis == "reconciled":
            reconciled_result_count += 1
        elif resolved.basis == "observed":
            observed_result_count += 1
        result_hands.append(hand.model_copy(update={"hero_bb_won": resolved.value}))
    recorded = [
        hand.hero_bb_won for hand in result_hands if hand.hero_bb_won is not None
    ]
    total = sum(recorded)
    hand_count = len(hands)
    tag_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()

    for hand in hands:
        tag_counts.update(hand.tags)
        status_counts.update([hand.review_status])
        if hand.id is None:
            continue
        action_counts.update(action.action_type for action in db.fetch_actions_by_hand(hand.id))

    if len(recorded) >= 2:
        _, ci_low, ci_high = mean_confidence_interval(recorded)
        bb_per_100_ci = (100 * ci_low, 100 * ci_high)
    else:
        bb_per_100_ci = None

    return SessionStats(
        hand_count=hand_count,
        hands_with_result=len(recorded),
        total_hero_bb=total,
        average_hero_bb=total / len(recorded) if recorded else 0,
        bb_per_100=100 * total / len(recorded) if recorded else 0,
        bb_per_100_ci=bb_per_100_ci,
        biggest_winning_hands=_top_hands(result_hands, reverse=True),
        biggest_losing_hands=_top_hands(result_hands, reverse=False),
        hands_by_tag=dict(tag_counts),
        hands_by_review_status=dict(status_counts),
        action_counts_by_type=dict(action_counts),
        aggression_count=sum(action_counts[action] for action in AGGRESSIVE_ACTIONS),
        passive_count=sum(action_counts[action] for action in PASSIVE_ACTIONS),
        reconciled_result_count=reconciled_result_count,
        observed_result_count=observed_result_count,
    )


def _top_hands(hands: list[Hand], *, reverse: bool) -> list[Hand]:
    eligible = [hand for hand in hands if hand.hero_bb_won is not None]
    sorted_hands = sorted(eligible, key=lambda hand: hand.hero_bb_won or 0, reverse=reverse)
    if reverse:
        return [hand for hand in sorted_hands if (hand.hero_bb_won or 0) > 0][:3]
    return [hand for hand in sorted_hands if (hand.hero_bb_won or 0) < 0][:3]


# ---------------------------------------------------------------------------
# Evidence classes
# ---------------------------------------------------------------------------

EvidenceClass = Literal["reviewed", "corrected_cv", "cv_draft", "manual"]

# Ordered strongest-evidence-first, which is also the order a caller should
# print them in.
EVIDENCE_CLASSES: tuple[EvidenceClass, ...] = (
    "reviewed",
    "corrected_cv",
    "cv_draft",
    "manual",
)

EVIDENCE_CLASS_LABELS: Mapping[EvidenceClass, str] = {
    "reviewed": "Reviewed",
    "corrected_cv": "Corrected CV",
    "cv_draft": "CV draft",
    "manual": "Manual entry",
}

EVIDENCE_CLASS_MEANING: Mapping[EvidenceClass, str] = {
    "reviewed": (
        "You marked this hand reviewed, which the promotion guard only allows "
        "once every readiness blocker is clear."
    ),
    "corrected_cv": (
        "Reconstructed by CV and then corrected by you, but not yet marked reviewed."
    ),
    "cv_draft": (
        "Reconstructed by CV and never corrected or confirmed. A draft, not a record."
    ),
    "manual": "Entered by hand and not yet marked reviewed.",
}


def classify_evidence(hand: Hand) -> EvidenceClass:
    """Which of the four evidence classes one stored hand belongs to.

    Exactly one, never two. The classes are a partition of the hands table, so a
    count of them sums to the hand count and a caller can print the parts beside
    the whole without the parts exceeding it.

    ``reviewed`` wins over provenance deliberately. ``source_type`` records where
    a hand's facts came from; ``review_status`` records that a person has since
    read them against the evidence and the store let the promotion through. The
    second is the stronger statement about whether the row can be believed, and
    it is the one a metric population is drawn on, so a reviewed CV hand belongs
    with the reviewed hands. Where a caller wants the provenance of a reviewed
    hand it is still on ``Hand.source_type``, unchanged.
    """
    if hand.review_status == "reviewed":
        return "reviewed"
    if hand.source_type == "corrected_cv":
        return "corrected_cv"
    if hand.source_type == "cv_import":
        return "cv_draft"
    return "manual"


# ---------------------------------------------------------------------------
# Hero result provenance
# ---------------------------------------------------------------------------

ResultBasis = Literal["reconciled", "observed", "unattributed", "none"]

RESULT_BASIS_LABELS: Mapping[ResultBasis, str] = {
    "reconciled": "Derived from the reconciled ledger",
    "observed": "Recorded on the hand",
    "unattributed": "Recorded, ledger could not attribute it to a hero seat",
    "none": "No result recorded",
}


@dataclass(frozen=True)
class HeroResult:
    """One hand's hero result and where the number came from.

    A derived figure and an observed one are the same float and mean different
    things, which is why they travel together. ``unattributed`` is its own basis
    rather than being folded into ``observed``: the accounting established, so
    the recorded column is not the authority, but no hero seat existed to read
    the ledger's answer off. It is counted as neither reconciled nor observed --
    the behaviour ``compute_session_stats`` has always had, now named.
    """

    value: float | None
    basis: ResultBasis


def resolve_hero_result(
    db: PokerDatabase,
    hand: Hand,
    accounting: AccountingReconciliation | None = None,
    *,
    reconciled: bool = False,
) -> HeroResult:
    """The hero result to report for one hand, with its provenance.

    Pass ``accounting`` with ``reconciled=True`` when the caller already holds
    this hand's reconciliation; a reconciliation is two ledger builds, and every
    surface that lists many hands must not pay for it twice. The two-argument
    form exists because ``accounting=None`` is a legitimate reconciliation
    result (``LedgerError``) and cannot double as "not supplied".

    ``accounting_is_established``, not ``is_authoritative``: a hand whose
    reported result exists only because of an unanswered operator declaration is
    not a fact. A declared 90% rake once turned a recorded NULL into -32 BB and a
    -3200 bb/100 win rate on a hand Study refuses for exactly that reason.
    """
    if hand.id is None:
        return HeroResult(
            value=hand.hero_bb_won,
            basis="observed" if hand.hero_bb_won is not None else "none",
        )
    if not reconciled:
        try:
            accounting = reconcile_persisted_hand(db, hand.id)
        except LedgerError:
            accounting = None
    if accounting_is_established(hand, accounting):
        assert accounting is not None
        hero = next(
            (player for player in db.fetch_players_by_hand(hand.id) if player.is_hero),
            None,
        )
        if hero is not None:
            return HeroResult(
                value=accounting.ledger.net_results.get(hero.player_key),
                basis="reconciled",
            )
        return HeroResult(value=hand.hero_bb_won, basis="unattributed")
    return HeroResult(
        value=hand.hero_bb_won,
        basis="observed" if hand.hero_bb_won is not None else "none",
    )


# ---------------------------------------------------------------------------
# Populations
# ---------------------------------------------------------------------------

PopulationKey = Literal["confirmed", "reconciled", "all_saved"]

ExclusionReason = Literal[
    "duplicate_session",
    "operator_excluded",
    "not_reviewed",
    "incomplete",
    "accounting_not_established",
]

EXCLUSION_REASON_LABELS: Mapping[ExclusionReason, str] = {
    "duplicate_session": "In a session the importer labelled a re-imported copy",
    "operator_excluded": "Marked non-study",
    "not_reviewed": "Not marked reviewed",
    "incomplete": "Reconstruction is partial or uncertain",
    "accounting_not_established": "Accounting is not reconciled",
}


@dataclass(frozen=True)
class PopulationSpec:
    """Which hands are eligible for a metric, and why.

    ``rule`` is written in the vocabulary of the columns the database actually
    records, not in product prose, because that is what makes the population
    checkable: a reader can take the sentence to the schema and confirm it.
    """

    key: PopulationKey
    label: str
    rule: str
    admits: str
    excludes: str


# The phase's language is "confirmed/reconciled". Made precise against the four
# things the database records about how far a hand has been taken:
#
#   review_status       unreviewed | reviewed | needs_correction
#   study_inclusion     auto | study | skip
#   completion_status   complete | partial | uncertain | not_applicable
#   settlement status   unsettled | settled | reconciled | needs_correction,
#                       cross-checked into `accounting_is_established`
#
# `completion_status` is tested even though `review_status == "reviewed"` should
# already imply it: the promotion guard enforces that for hands promoted through
# the product, but an import payload and a hand-edited row are not the product,
# and a population that trusts one column is a population one forged column can
# enter. Three populations rather than a free-form filter, because a metric that
# can be computed over an arbitrary subset is a metric whose caption is a
# promise nobody checked.
POPULATIONS: Mapping[PopulationKey, PopulationSpec] = {
    "confirmed": PopulationSpec(
        key="confirmed",
        label="Confirmed hands",
        rule=(
            "review_status = 'reviewed' AND study_inclusion <> 'skip' AND "
            "completion_status IN ('complete', 'not_applicable')"
        ),
        admits=(
            "Hands you marked reviewed, which the promotion guard only allows "
            "once every readiness blocker is clear."
        ),
        excludes=(
            "CV drafts, corrected-but-unconfirmed hands, hands you marked "
            "non-study, and anything whose reconstruction is partial or uncertain."
        ),
    ),
    "reconciled": PopulationSpec(
        key="reconciled",
        label="Confirmed and reconciled hands",
        rule=(
            "the confirmed rule, AND the ledger cross-check is established "
            "(settlement status 'reconciled', the chips balance, and no "
            "unattested declared assumption is holding the balance up)"
        ),
        admits="Confirmed hands whose chip accounting closes on its own evidence.",
        excludes=(
            "Confirmed hands that were never settled, that do not balance, or "
            "that only balance because of a declaration nobody has attested to."
        ),
    ),
    "all_saved": PopulationSpec(
        key="all_saved",
        label="All saved hands",
        rule=(
            "every row in the hands table, except rows belonging to a session the "
            "importer recorded as a re-imported copy"
        ),
        admits="Everything the database holds, whatever state it is in.",
        excludes=(
            "Only re-imported copies, which are a second stored copy of hands "
            "already counted under another session. This population still mixes "
            "confirmed records with unconfirmed CV drafts, so a rate computed "
            "over it is a description of the database rather than of your play."
        ),
    ),
}

# The default is the narrowest population that still answers a question about
# the operator's play. Showing less with provenance beats showing more without
# it, and "all saved hands" is exactly the population whose win rate silently
# mixed four epistemic states.
DEFAULT_POPULATION: PopulationKey = "confirmed"


@dataclass(frozen=True)
class HandEvidence:
    """One hand's epistemic state, resolved once so every metric reads the same one."""

    hand: Hand
    hand_id: int | None
    evidence_class: EvidenceClass
    review_status: str
    completion_status: str
    source_type: str
    study_inclusion: str
    settlement_status: str
    accounting_established: bool
    accounting_error: str | None
    result: HeroResult
    # The session id this hand's session was recorded as a re-imported copy of.
    # Set means the same poker is already in the database under another session,
    # so counting this row adds it to the operator's totals a second time.
    duplicate_of_session_id: int | None
    tags: tuple[str, ...]
    current_coaching_themes: tuple[str, ...]
    stale_coaching_themes: tuple[str, ...]
    current_coaching_count: int
    stale_coaching_count: int

    @property
    def result_value(self) -> float | None:
        return self.result.value

    @property
    def result_basis(self) -> ResultBasis:
        return self.result.basis


def population_exclusion(
    evidence: HandEvidence, population: PopulationKey
) -> ExclusionReason | None:
    """Why this hand is not in that population, or ``None`` when it is.

    Returns the first failing clause rather than all of them, so the caller can
    report a single actionable reason per hand. The order is by what the
    operator would do about it: a deliberate exclusion first, then the workflow
    step, then the evidence problems.

    The duplicate check precedes the ``all_saved`` shortcut deliberately. A
    re-imported copy is not a weaker grade of evidence that a wider population
    can admit -- it is the same poker stored twice -- so no population may count
    it, and ``all_saved`` says so in its rule rather than quietly doubling. That
    also removes the second, weaker answer to the question: with the check above
    the shortcut there is no population left that a caller could reach for to get
    the doubled figure back.
    """
    if evidence.duplicate_of_session_id is not None:
        return "duplicate_session"
    if population == "all_saved":
        return None
    if evidence.study_inclusion == "skip":
        return "operator_excluded"
    if evidence.review_status != "reviewed":
        return "not_reviewed"
    if evidence.completion_status not in ("complete", "not_applicable"):
        return "incomplete"
    if population == "reconciled" and not evidence.accounting_established:
        return "accounting_not_established"
    return None


@dataclass(frozen=True)
class PopulationSnapshot:
    """A declared population, its members, and everything it left out.

    Carrying the excluded hands rather than dropping them is the point: a filter
    that silently loses rows is indistinguishable from a filter that is right,
    and the exclusion counts are what let a caller say "12 of 47, and here is
    where the other 35 went".
    """

    spec: PopulationSpec
    members: tuple[HandEvidence, ...]
    considered: tuple[HandEvidence, ...]
    excluded_by_reason: Mapping[ExclusionReason, int]

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def considered_count(self) -> int:
        return len(self.considered)

    @property
    def evidence_mix(self) -> Mapping[EvidenceClass, int]:
        return _evidence_mix(self.members)

    @property
    def result_basis_mix(self) -> Mapping[ResultBasis, int]:
        counts: Counter[ResultBasis] = Counter(
            member.result_basis for member in self.members
        )
        return {basis: counts.get(basis, 0) for basis in RESULT_BASIS_LABELS}

    def coverage(self, included: int) -> Coverage:
        return Coverage(
            included=included,
            eligible=self.size,
            considered=self.considered_count,
            population_label=self.spec.label,
        )


def _evidence_mix(items: Sequence[HandEvidence]) -> Mapping[EvidenceClass, int]:
    counts: Counter[EvidenceClass] = Counter(item.evidence_class for item in items)
    return {name: counts.get(name, 0) for name in EVIDENCE_CLASSES}


def build_hand_evidence(
    db: PokerDatabase,
    hands: Iterable[Hand],
    *,
    reconcile: Callable[[int], tuple[AccountingReconciliation | None, str | None]]
    | None = None,
    coaching_by_hand: Mapping[int, Sequence[CoachingResponse | HandReview]]
    | None = None,
    load_coaching: bool = True,
    sessions: Iterable[Session] | None = None,
) -> tuple[HandEvidence, ...]:
    """Resolve every hand's epistemic state once, for any number of populations.

    Split from ``select_population`` so that switching the population selector on
    a rendered page re-filters an already-resolved list instead of reconciling
    the database again.

    ``reconcile`` lets a surface that already keeps a per-render reconciliation
    cache pass it in; without one this reconciles each hand itself, which is two
    ledger builds per hand. ``coaching_by_hand`` does the same for retained
    coaching: supplied, it costs nothing; omitted with ``load_coaching`` set, it
    is two queries per hand. ``sessions`` is the same optimisation for a caller
    that already holds the session list.

    The re-import lookup is done HERE, from the database, rather than taken as a
    required argument, because a caller that forgets it would get doubled totals
    back and nothing would say so. One query for the whole corpus is the price of
    making that unforgettable.
    """
    duplicates = duplicate_import_sessions(
        db.fetch_sessions() if sessions is None else sessions
    )
    resolved: list[HandEvidence] = []
    for hand in hands:
        accounting: AccountingReconciliation | None = None
        error: str | None = None
        if hand.id is not None:
            if reconcile is not None:
                accounting, error = reconcile(hand.id)
            else:
                try:
                    accounting = reconcile_persisted_hand(db, hand.id)
                except LedgerError as exc:
                    error = str(exc)
        established = accounting_is_established(hand, accounting)
        reviews: Sequence[CoachingResponse | HandReview] = ()
        if hand.id is not None:
            if coaching_by_hand is not None:
                reviews = coaching_by_hand.get(hand.id, ())
            elif load_coaching:
                reviews = [
                    *db.fetch_coaching_reviews_by_hand(hand.id),
                    *db.fetch_reviews_by_hand(hand.id),
                ]
        current: list[str] = []
        stale: list[str] = []
        current_count = 0
        stale_count = 0
        for review in reviews:
            if getattr(review, "is_stale", False):
                stale_count += 1
                stale.extend(coaching_themes(review))
            else:
                current_count += 1
                current.extend(coaching_themes(review))
        resolved.append(
            HandEvidence(
                hand=hand,
                hand_id=hand.id,
                evidence_class=classify_evidence(hand),
                review_status=hand.review_status,
                completion_status=hand.completion_status,
                source_type=hand.source_type,
                study_inclusion=hand.study_inclusion,
                settlement_status=(
                    accounting.settlement.status
                    if accounting is not None and accounting.settlement is not None
                    else "unsettled"
                ),
                accounting_established=established,
                accounting_error=error,
                result=resolve_hero_result(db, hand, accounting, reconciled=True),
                duplicate_of_session_id=duplicates.get(hand.session_id),
                tags=tuple(hand.tags),
                current_coaching_themes=tuple(dict.fromkeys(current)),
                stale_coaching_themes=tuple(dict.fromkeys(stale)),
                current_coaching_count=current_count,
                stale_coaching_count=stale_count,
            )
        )
    return tuple(resolved)


def select_population(
    evidence: Iterable[HandEvidence],
    population: PopulationKey = DEFAULT_POPULATION,
) -> PopulationSnapshot:
    """Filter resolved evidence down to one declared population."""
    considered = tuple(evidence)
    members: list[HandEvidence] = []
    excluded: Counter[ExclusionReason] = Counter()
    for item in considered:
        reason = population_exclusion(item, population)
        if reason is None:
            members.append(item)
        else:
            excluded[reason] += 1
    return PopulationSnapshot(
        spec=POPULATIONS[population],
        members=tuple(members),
        considered=considered,
        excluded_by_reason={
            reason: excluded[reason] for reason in EXCLUSION_REASON_LABELS if excluded[reason]
        },
    )


# ---------------------------------------------------------------------------
# Coverage, sample size and metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Coverage:
    """How much of the declared population actually fed a metric.

    Three numbers, not one. ``included`` is what the metric averaged or summed,
    ``eligible`` is the population it was drawn from, and ``considered`` is every
    hand examined before the population filter ran. A theme on 3 of 4 hands and
    a theme on 3 of 400 render identically without the second number, and a
    population filter that silently dropped half the database is invisible
    without the third.
    """

    included: int
    eligible: int
    considered: int
    population_label: str

    @property
    def is_complete(self) -> bool:
        return self.included == self.eligible

    def statement(self) -> str:
        label = self.population_label.lower()
        return (
            f"{self.included} of {self.eligible} {label} "
            f"· {self.considered} saved hands examined"
        )


# 30 hands is the floor at which the normal approximation behind
# `mean_confidence_interval` starts to describe the sample it is computed from.
# Below it the interval that function returns is narrower than the truth rather
# than wider, which is the dangerous direction: a tight interval on eight hands
# reads as precision.
#
# It is emphatically NOT the point at which a poker win rate becomes meaningful.
# With a per-hand standard deviation in the 10-15 BB range the 95% interval on
# bb/100 is still hundreds of bb/100 wide at n=1000, so `RATE_CAVEAT` is
# attached to every rate metric at every sample size and this threshold only
# decides whether the product is willing to print a rate at all. The verdict is
# returned rather than left to the caller because "is 8 hands enough" is the
# judgement call the surface kept making differently in each place.
MIN_HANDS_FOR_RATE = 30

# `mean_confidence_interval` raises below two values.
MIN_HANDS_FOR_INTERVAL = 2

# One hand is enough to state a count honestly; a count is not an estimate.
MIN_HANDS_FOR_COUNT = 1

RATE_CAVEAT = (
    "A study-corpus win rate is a description of this sample, not proof of a "
    "win rate. Poker variance is large enough that the interval stays wide at "
    "any sample size you will review by hand."
)

SampleSize = Literal["empty", "below_threshold", "adequate"]


@dataclass(frozen=True)
class SampleVerdict:
    """Whether the sample behind a metric is large enough to print it."""

    hands: int
    threshold: int
    verdict: SampleSize
    caveat: str

    @property
    def is_reportable(self) -> bool:
        return self.verdict == "adequate"


def sample_verdict(hands: int, threshold: int, *, noun: str = "hands") -> SampleVerdict:
    if hands <= 0:
        return SampleVerdict(
            hands=hands,
            threshold=threshold,
            verdict="empty",
            caveat=f"No {noun} in this population.",
        )
    if hands < threshold:
        return SampleVerdict(
            hands=hands,
            threshold=threshold,
            verdict="below_threshold",
            caveat=(
                f"{hands} {noun} is below the {threshold}-hand floor this product "
                "will print a rate from."
            ),
        )
    return SampleVerdict(hands=hands, threshold=threshold, verdict="adequate", caveat="")


@dataclass(frozen=True)
class Metric:
    """A number that cannot be read without what it was computed from.

    ``value`` is a float for arithmetic, but the two properties a surface renders
    -- ``headline`` and ``support`` -- are produced together and ``headline``
    refuses to name a figure the sample cannot carry. That is the defect this
    type exists to make impossible: a bare float invites a caption written from
    memory, and a caption written from memory is how a draft-inflated win rate
    got printed as a fact.
    """

    key: str
    label: str
    value: float | None
    unit: str
    numerator: float | None
    denominator: int
    population: PopulationSpec
    coverage: Coverage
    sample: SampleVerdict
    evidence_mix: Mapping[EvidenceClass, int]
    basis_mix: Mapping[ResultBasis, int]
    interval: tuple[float, float] | None = None
    caveats: tuple[str, ...] = ()

    @property
    def is_reportable(self) -> bool:
        return self.value is not None and self.sample.is_reportable

    @property
    def headline(self) -> str:
        if not self.is_reportable:
            return "Not enough evidence"
        assert self.value is not None
        return format_metric_value(self.value, self.unit)

    @property
    def support(self) -> str:
        return self.coverage.statement()

    @property
    def interval_label(self) -> str:
        if self.interval is None:
            return "No interval at this sample size"
        low, high = self.interval
        return f"95% interval {low:+,.1f} to {high:+,.1f} {self.unit}"


def format_metric_value(value: float, unit: str) -> str:
    if unit == "hands":
        return f"{value:,.0f} hands"
    if unit == "%":
        return f"{value:.0f}%"
    if unit in ("BB", "bb/100"):
        return f"{value:+,.2f} {unit}"
    return f"{value:,.2f} {unit}".strip()


@dataclass(frozen=True)
class PopulationMetrics:
    """Every headline metric for one population, each carrying its own context."""

    population: PopulationSpec
    snapshot: PopulationSnapshot
    hand_count: Metric
    net_result: Metric
    average_result: Metric
    bb_per_100: Metric
    evidence_mix: Mapping[EvidenceClass, int]
    result_basis_mix: Mapping[ResultBasis, int]
    excluded_by_reason: Mapping[ExclusionReason, int]

    @property
    def metrics(self) -> tuple[Metric, ...]:
        return (self.hand_count, self.net_result, self.average_result, self.bb_per_100)

    def by_key(self, key: str) -> Metric:
        for metric in self.metrics:
            if metric.key == key:
                return metric
        raise KeyError(key)


def compute_population_metrics(snapshot: PopulationSnapshot) -> PopulationMetrics:
    """Headline metrics for one population, recomputable from the hand rows.

    Every number here is a plain sum or mean over ``snapshot.members``, which is
    what makes the verification test possible: recompute from the raw rows and
    the two must agree, so a population filter that silently drops hands shows up
    as a disagreement rather than as a slightly different number nobody checks.
    """
    members = snapshot.members
    results = [
        member.result_value for member in members if member.result_value is not None
    ]
    evidence_mix = snapshot.evidence_mix
    basis_mix = snapshot.result_basis_mix
    spec = snapshot.spec

    with_results = [
        member for member in members if member.result_value is not None
    ]
    result_mix = _evidence_mix(with_results)
    result_basis_mix: Mapping[ResultBasis, int] = {
        basis: sum(1 for member in with_results if member.result_basis == basis)
        for basis in RESULT_BASIS_LABELS
    }

    total = sum(results)
    mean = total / len(results) if results else None
    per_hand_interval: tuple[float, float] | None = None
    rate_interval: tuple[float, float] | None = None
    if len(results) >= MIN_HANDS_FOR_INTERVAL:
        _, low, high = mean_confidence_interval(results)
        per_hand_interval = (low, high)
        rate_interval = (100 * low, 100 * high)

    population_note = (
        ()
        if spec.key != "all_saved"
        else (
            "This population mixes confirmed records with unconfirmed CV drafts.",
        )
    )

    hand_count = Metric(
        key="hand_count",
        label=f"{spec.label} in scope",
        value=float(len(members)),
        unit="hands",
        numerator=float(len(members)),
        denominator=snapshot.considered_count,
        population=spec,
        coverage=Coverage(
            included=len(members),
            eligible=len(members),
            considered=snapshot.considered_count,
            population_label=spec.label,
        ),
        sample=sample_verdict(len(members), MIN_HANDS_FOR_COUNT),
        evidence_mix=evidence_mix,
        basis_mix=basis_mix,
        caveats=population_note,
    )
    net_result = Metric(
        key="net_result",
        label="Net result",
        value=total if results else None,
        unit="BB",
        numerator=total if results else None,
        denominator=len(results),
        population=spec,
        coverage=snapshot.coverage(len(results)),
        sample=sample_verdict(len(results), MIN_HANDS_FOR_COUNT),
        evidence_mix=result_mix,
        basis_mix=result_basis_mix,
        caveats=population_note,
    )
    average_result = Metric(
        key="average_result",
        label="Average result per hand",
        value=mean,
        unit="BB",
        numerator=total if results else None,
        denominator=len(results),
        population=spec,
        coverage=snapshot.coverage(len(results)),
        sample=sample_verdict(len(results), MIN_HANDS_FOR_RATE),
        evidence_mix=result_mix,
        basis_mix=result_basis_mix,
        interval=per_hand_interval,
        caveats=(RATE_CAVEAT, *population_note),
    )
    bb_per_100 = Metric(
        key="bb_per_100",
        label="Win rate",
        value=None if mean is None else 100 * mean,
        unit="bb/100",
        numerator=total if results else None,
        denominator=len(results),
        population=spec,
        coverage=snapshot.coverage(len(results)),
        sample=sample_verdict(len(results), MIN_HANDS_FOR_RATE),
        evidence_mix=result_mix,
        basis_mix=result_basis_mix,
        interval=rate_interval,
        caveats=(RATE_CAVEAT, *population_note),
    )
    return PopulationMetrics(
        population=spec,
        snapshot=snapshot,
        hand_count=hand_count,
        net_result=net_result,
        average_result=average_result,
        bb_per_100=bb_per_100,
        evidence_mix=evidence_mix,
        result_basis_mix=basis_mix,
        excluded_by_reason=snapshot.excluded_by_reason,
    )


# ---------------------------------------------------------------------------
# Study themes
# ---------------------------------------------------------------------------

ThemeSource = Literal["tag", "coaching"]

# The one structured claim a coaching response makes that is countable across
# hands. `HandReview` carries the same thing in a column.
COACHING_THEME_SECTION = "Study Lesson"

_THEME_MAX_CHARS = 80
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s")


def normalize_theme(text: str) -> str:
    """Collapse one free-text study lesson into a countable theme label.

    The countable part of a lesson is its claim; the sentences after it are the
    argument for the claim and never repeat verbatim across hands, so keeping
    them would give every hand its own unique theme and a theme index where
    every count is 1.
    """
    collapsed = " ".join(str(text or "").split())
    if not collapsed:
        return ""
    first = _SENTENCE_SPLIT.split(collapsed)[0].strip().rstrip(".!?").strip()
    if not first:
        return ""
    if len(first) <= _THEME_MAX_CHARS:
        return first
    clipped = first[:_THEME_MAX_CHARS].rsplit(" ", 1)[0]
    return f"{clipped or first[:_THEME_MAX_CHARS]}…"


def coaching_themes(review: CoachingResponse | HandReview) -> tuple[str, ...]:
    """The countable themes one retained review states, stale or not.

    Staleness is decided by the caller, not here, so that the excluded set can be
    reported rather than silently vanishing.
    """
    sections = getattr(review, "parsed_sections", None)
    if isinstance(sections, Mapping):
        theme = normalize_theme(str(sections.get(COACHING_THEME_SECTION, "")))
        return (theme,) if theme else ()
    lesson = getattr(review, "study_lesson", "")
    theme = normalize_theme(str(lesson))
    return (theme,) if theme else ()


def tag_theme_label(tag: str) -> str:
    return tag.replace("_", " ").title()


@dataclass(frozen=True)
class ThemeCount:
    """One theme, with the population it is a share of.

    ``hands`` without ``denominator`` is the defect: a theme on 3 of 4 hands and
    a theme on 3 of 400 are the same bar.
    """

    theme: str
    source: ThemeSource
    hands: int
    denominator: int

    @property
    def share(self) -> float:
        return 100 * self.hands / self.denominator if self.denominator else 0.0

    @property
    def statement(self) -> str:
        return f"{self.hands} of {self.denominator} hands ({self.share:.0f}%)"


@dataclass(frozen=True)
class ThemeAggregate:
    """The theme index for one population, with what stale coaching cost it."""

    population: PopulationSpec
    denominator: int
    themes: tuple[ThemeCount, ...]
    coverage: Coverage
    sample: SampleVerdict
    stale_coaching_reviews_excluded: int
    stale_coaching_hands: int
    stale_only_themes_excluded: tuple[str, ...]

    @property
    def exclusion_statement(self) -> str:
        if not self.stale_coaching_reviews_excluded:
            return "No stale coaching evidence was excluded."
        themes = len(self.stale_only_themes_excluded)
        return (
            f"{self.stale_coaching_reviews_excluded} stale coaching review(s) on "
            f"{self.stale_coaching_hands} hand(s) were excluded"
            + (
                f", removing {themes} theme(s) that no current evidence supports."
                if themes
                else "."
            )
        )


def aggregate_study_themes(snapshot: PopulationSnapshot) -> ThemeAggregate:
    """Count themes over a declared population, excluding stale coaching.

    A theme index is a claim about what your CURRENT evidence says. A coaching
    review that a correction invalidated is retained deliberately -- the product
    keeps it as history rather than deleting it -- but retained history is not
    current evidence, and letting it into the index means a conclusion the
    operator has already overruled keeps voting. Excluded reviews are counted and
    reported rather than dropped, so the index can say what it left out.

    Tag themes are not filtered by coaching staleness: tags are written by the
    operator and the pipeline, not by the coach, so a stale coaching review says
    nothing about them.
    """
    members = snapshot.members
    denominator = len(members)
    tag_hands: Counter[str] = Counter()
    coaching_hands: Counter[str] = Counter()
    stale_themes: Counter[str] = Counter()
    stale_reviews = 0
    stale_hands = 0
    contributing: set[int] = set()

    for index, member in enumerate(members):
        for tag in dict.fromkeys(member.tags):
            tag_hands[tag_theme_label(tag)] += 1
            contributing.add(index)
        for theme in member.current_coaching_themes:
            coaching_hands[theme] += 1
            contributing.add(index)
        if member.stale_coaching_count:
            stale_reviews += member.stale_coaching_count
            stale_hands += 1
            for theme in member.stale_coaching_themes:
                stale_themes[theme] += 1

    themes = tuple(
        sorted(
            (
                *(
                    ThemeCount(theme=name, source="tag", hands=count, denominator=denominator)
                    for name, count in tag_hands.items()
                ),
                *(
                    ThemeCount(
                        theme=name, source="coaching", hands=count, denominator=denominator
                    )
                    for name, count in coaching_hands.items()
                ),
            ),
            key=lambda item: (-item.hands, item.source, item.theme),
        )
    )
    stale_only = tuple(
        sorted(name for name in stale_themes if name not in coaching_hands)
    )
    return ThemeAggregate(
        population=snapshot.spec,
        denominator=denominator,
        themes=themes,
        coverage=snapshot.coverage(len(contributing)),
        sample=sample_verdict(denominator, MIN_HANDS_FOR_COUNT),
        stale_coaching_reviews_excluded=stale_reviews,
        stale_coaching_hands=stale_hands,
        stale_only_themes_excluded=stale_only,
    )
