from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from poker_tracker.math.cards import CardParseError, parse_visible_cards
from poker_tracker.persistence.validation import normalize_cards, validate_tags

Street = Literal["preflop", "flop", "turn", "river", "showdown"]
ActionType = Literal[
    "fold",
    "check",
    "call",
    "bet",
    "raise",
    "all-in",
    "ante",
    "post_blind",
    "show",
    "win",
]
AmountSemantics = Literal["incremental", "raise_to", "unknown"]
SettlementStatus = Literal["unsettled", "settled", "reconciled", "needs_correction"]
SettlementEntryType = Literal["award", "refund"]
ForcedBetType = Literal[
    "small_blind",
    "big_blind",
    "ante",
    "big_blind_ante",
    "straddle",
    "dead_blind",
    "bring_in",
]
ReviewStatus = Literal["unreviewed", "reviewed", "needs_correction"]
FrameReviewStatus = Literal["unreviewed", "correct", "incorrect"]
SourceType = Literal["manual", "cv_import", "corrected_cv"]
CompletionStatus = Literal["complete", "partial", "uncertain", "not_applicable"]
ReviewType = Literal["hand", "session"]
SafetyMode = Literal["post_session_only"]
CorrectionType = Literal[
    "hand_facts",
    "player_update",
    "action_create",
    "action_update",
    "action_delete",
    # Re-declaring who took a pot is a source-fact correction like any other: the
    # declared winner is the sole input the derived payouts, and therefore the
    # hero-result cross-check, are computed from.
    "settlement_award_update",
]
HandIssueStatus = Literal["open", "resolved"]
HandIssueType = Literal[
    "hand_boundary",
    "cards",
    "players",
    "stacks",
    "actions",
    "pot_or_result",
    "accounting",
    "coaching",
    "other",
]
JobStatus = Literal["queued", "running", "completed", "failed"]
JobType = Literal["frame_extraction", "cv_reconstruction"]
SolverRunStatus = Literal[
    "queued",
    "running",
    "cancelling",
    "completed",
    "failed",
    "cancelled",
    "stale",
]
SolverRangeSource = Literal["builtin", "user"]
ROIType = Literal[
    "hero_card",
    "board_card",
    "pot",
    "player_stack",
    "player_bet",
    "player_name",
    "dealer_button",
    "active_indicator",
    "action_button",
    "table_area",
    "unknown",
]

HAND_TAGS = {
    "MISSED_VALUE",
    "BIG_POT",
    "MULTIWAY",
    "PREFLOP_3BET_SPOT",
    "RIVER_DECISION",
    "LOW_CONFIDENCE",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


class PersistedModel(BaseModel):
    """The validating boundary every stored record passes through.

    ``allow_inf_nan=False`` is set HERE, once, rather than per field: NaN and
    +/-Infinity are not measurements, they cannot be written to RFC 8259 JSON
    (``json.dumps`` emits bare tokens no standards-compliant consumer can read,
    so one ``dead_money=Infinity`` landed by a hostile import payload made the
    whole session export unparseable), and a per-field constraint list is
    exactly the enumerated-list decay that let ``ge=0`` admit ``inf`` while
    every merely out-of-range value was refused. A float field added to any
    model below inherits the refusal without anyone remembering this class
    exists — the same reason ``completion._as_float_or_none`` refuses non-finite
    evidence floats.
    """

    model_config = ConfigDict(from_attributes=True, allow_inf_nan=False)

    # Set only by the persistence layer's read-time degradation when a stored
    # row holds column values this build cannot validate (``db._salvaged_row``).
    # Excluded from every dump for the same reason ``derived_result_substituted``
    # is: it is a derivation about the CURRENT row, so no export, payload, or
    # database row can persist or forge it, and the stored response shape is
    # unchanged. Consumers that grant authority (the accounting cross-check)
    # treat a non-empty value as an issue, so a degraded row can only ever add
    # blockers, never clear one.
    unreadable_columns: tuple[str, ...] = Field(default=(), exclude=True)


class Session(PersistedModel):
    id: int | None = None
    name: str = Field(min_length=1)
    date_played: date = Field(default_factory=date.today)
    platform: str = "Manual"
    stakes: str = ""
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class Hand(PersistedModel):

    id: int | None = None
    session_id: int
    hand_number: int = Field(ge=1)
    game_type: str = ""
    blinds_antes: str = ""
    table_size: int | None = Field(default=None, ge=2, le=10)
    effective_stack: float | None = Field(default=None, ge=0)
    hero_position: str = ""
    hero_cards: str = ""
    board_cards: str = ""
    pot_size: float | None = Field(default=None, ge=0)
    result: str = ""
    hero_bb_won: float | None = None
    review_status: ReviewStatus = "unreviewed"
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    source_type: SourceType = "manual"
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    completion_status: CompletionStatus = "not_applicable"
    # Deliberately an open mapping rather than a nested model: unknown keys from a
    # newer producer must survive a round trip instead of being dropped by
    # extra="ignore", and a corrupt blob must never raise into the UI.
    completion_evidence: dict[str, object] = Field(default_factory=dict)
    # Set only by the READ-TIME display substitution that replaces `hero_bb_won`
    # with the derived ledger result on an authoritative hand, so a writer can
    # refuse an object whose "observation" is actually a derivation. Excluded from
    # every dump, so no export, payload, or database row can carry or forge it,
    # and the stored response shape is unchanged.
    #
    # It exists because a display copy is otherwise indistinguishable from a
    # stored hand: one reached the 'Correct hand facts' form, and saving an
    # unrelated field on that form persisted the derived result into the observed
    # column. Fixing the one call site fixes one call site; refusing the object at
    # the writer covers every present and future one.
    derived_result_substituted: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def default_completion_status(cls, data: Any) -> Any:
        """A reconstructed hand with no declared completion is unproven, not exempt."""
        if isinstance(data, dict) and not data.get("completion_status"):
            data = dict(data)
            source = data.get("source_type") or "manual"
            data["completion_status"] = (
                "not_applicable" if source == "manual" else "uncertain"
            )
        return data

    @field_validator("hero_cards")
    @classmethod
    def validate_hero_cards(cls, value: str) -> str:
        return normalize_cards(value, expected_counts={0, 2})

    @field_validator("board_cards")
    @classmethod
    def validate_board_cards(cls, value: str) -> str:
        return normalize_cards(value, expected_counts={0, 3, 4, 5})

    @field_validator("tags")
    @classmethod
    def validate_hand_tags(cls, value: list[str]) -> list[str]:
        return validate_tags(value, HAND_TAGS)

    @model_validator(mode="after")
    def validate_visible_card_uniqueness(self) -> Hand:
        if self.hero_cards and self.board_cards:
            try:
                parse_visible_cards(self.hero_cards, self.board_cards)
            except CardParseError as exc:
                raise ValueError(str(exc)) from exc
        return self


class HandPlayer(PersistedModel):

    id: int | None = None
    hand_id: int
    player_key: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    seat_index: int | None = Field(default=None, ge=0, le=9)
    player_name: str = Field(min_length=1)
    position: str = ""
    starting_stack: float | None = Field(default=None, ge=0)
    is_hero: bool = False
    notes: str = ""


class Action(PersistedModel):

    id: int | None = None
    hand_id: int
    player_key: str | None = Field(default=None, min_length=1)
    street: Street
    action_index: int | None = Field(default=None, ge=1)
    player_name: str = Field(min_length=1)
    position: str = ""
    action_type: ActionType
    amount: float | None = Field(default=None, ge=0)
    amount_semantics: AmountSemantics = "incremental"
    forced_bet_type: ForcedBetType | None = None
    is_live_post: bool | None = None
    pot_before: float | None = Field(default=None, ge=0)
    stack_before: float | None = Field(default=None, ge=0)
    notes: str = ""

    @field_validator("action_type", mode="before")
    @classmethod
    def normalize_action_type(cls, value: str) -> str:
        if value == "all_in":
            return "all-in"
        return value


class HandSettlement(PersistedModel):
    """Persisted assumptions and reconciliation summary for a completed hand."""

    hand_id: int
    status: SettlementStatus = "unsettled"
    dead_money: float = Field(default=0, ge=0)
    rake_rate: float = Field(default=0, ge=0, le=1)
    rake_cap: float | None = Field(default=None, ge=0)
    rake_rounding_unit: float = Field(default=0.01, gt=0)
    no_flop_no_drop: bool = False
    gross_pot: float | None = Field(default=None, ge=0)
    rake_amount: float | None = Field(default=None, ge=0)
    net_pot: float | None = Field(default=None, ge=0)
    is_balanced: bool = False
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SettlementEntry(PersistedModel):
    """A declared pot award or explicit uncalled-chip refund."""

    id: int | None = None
    hand_id: int
    entry_type: SettlementEntryType
    pot_index: int | None = Field(default=None, ge=0)
    player_key: str | None = Field(default=None, min_length=1)
    player_name: str = Field(min_length=1)
    amount: float | None = Field(default=None, ge=0)
    entry_order: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_pot_index(self) -> SettlementEntry:
        if self.entry_type == "award" and self.pot_index is None:
            raise ValueError("Award entries require a pot index.")
        if self.entry_type == "refund" and self.pot_index is not None:
            raise ValueError("Refund entries cannot reference a pot index.")
        return self


class HandReview(PersistedModel):

    id: int | None = None
    hand_id: int
    hand_summary: str
    theory_coach: str
    exploit_coach: str
    ev_math_notes: str = ""
    study_lesson: str
    next_review_question: str = ""
    notes: str = ""
    is_stale: bool = False
    stale_reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class HandCorrection(PersistedModel):
    """Auditable before/after evidence retained from post-session correction."""

    id: int | None = None
    hand_id: int
    correction_type: CorrectionType
    before_state: dict[str, object] = Field(default_factory=dict)
    after_state: dict[str, object] = Field(default_factory=dict)
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class HandIssue(PersistedModel):
    """A saved debug-later report with an immutable evidence snapshot."""

    id: int | None = None
    hand_id: int
    status: HandIssueStatus = "open"
    issue_types: list[HandIssueType] = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_snapshot: dict[str, object] = Field(default_factory=dict)
    resolution_notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class LLMProviderConfig(BaseModel):
    provider_name: str = "unconfigured"
    model_name: str = ""
    has_api_key: bool = False


class CoachingRequest(BaseModel):
    prompt: str
    review_type: ReviewType
    provider_name: str = "unconfigured"
    model_name: str = ""
    hand_id: int | None = None
    session_id: int | None = None
    safety_mode: SafetyMode = "post_session_only"


class CoachingResponse(PersistedModel):

    id: int | None = None
    provider_name: str
    model_name: str
    raw_prompt: str
    raw_response: str
    review_type: ReviewType
    safety_mode: SafetyMode = "post_session_only"
    hand_id: int | None = None
    session_id: int | None = None
    parsed_sections: dict[str, str] = Field(default_factory=dict)
    is_stale: bool = False
    stale_reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class SolverRangeProfile(PersistedModel):
    """Reusable user-owned range input for post-session solver work."""

    id: int | None = None
    name: str = Field(min_length=1, max_length=120)
    notation: str = Field(min_length=1)
    table_size: int | None = Field(default=None, ge=2, le=10)
    position: str = ""
    scenario: str = ""
    pot_type: str = ""
    stack_bb: float | None = Field(default=None, gt=0)
    description: str = ""
    source: SolverRangeSource = "user"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SolverRun(PersistedModel):
    """Durable metadata for one external postflop solve."""

    id: int | None = None
    hand_id: int
    status: SolverRunStatus = "queued"
    backend_name: str = "texassolver"
    backend_version: str = ""
    input_hash: str = Field(min_length=1)
    spot: dict[str, object] = Field(default_factory=dict)
    range_ip: dict[str, object] = Field(default_factory=dict)
    range_oop: dict[str, object] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    evidence: dict[str, object] = Field(default_factory=dict)
    command_path: str = ""
    result_path: str = ""
    log_path: str = ""
    exploitability_pct: float | None = Field(default=None, ge=0)
    runtime_seconds: float | None = Field(default=None, ge=0)
    error_message: str = ""
    pid: int | None = Field(default=None, ge=1)
    heartbeat_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class VideoRecord(PersistedModel):

    id: int | None = None
    session_id: int | None = None
    original_filename: str
    stored_path: str
    file_size_bytes: int = Field(ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    frame_count: int | None = Field(default=None, ge=0)
    uploaded_at: datetime = Field(default_factory=utc_now)
    notes: str = ""


class ProcessingJob(PersistedModel):

    id: int | None = None
    job_type: JobType
    status: JobStatus = "queued"
    video_id: int
    progress_percent: float = Field(default=0, ge=0, le=100)
    message: str = ""
    error_message: str = ""
    pid: int | None = Field(default=None, ge=1)
    heartbeat_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ExtractedFrame(PersistedModel):

    id: int | None = None
    video_id: int
    job_id: int
    timestamp_seconds: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    image_path: str
    created_at: datetime = Field(default_factory=utc_now)


class ReconstructionFrameReview(PersistedModel):
    """Human verdict for one source frame used by a reconstructed hand."""

    id: int | None = None
    job_id: int
    hand_number: int = Field(ge=1)
    source_image: str = Field(min_length=1)
    timestamp_seconds: float = Field(ge=0)
    status: FrameReviewStatus = "unreviewed"
    issue_types: list[str] = Field(default_factory=list)
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ROIProfile(PersistedModel):

    id: int | None = None
    name: str = Field(min_length=1)
    description: str = ""
    platform: str = "ClubWPT Gold"
    table_layout: str = ""
    video_width: int | None = Field(default=None, ge=1)
    video_height: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    is_active: bool = False


class ROIRegion(PersistedModel):

    id: int | None = None
    profile_id: int
    roi_key: str = Field(min_length=1)
    roi_type: ROIType = "unknown"
    label: str = ""
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    seat_index: int | None = Field(default=None, ge=1, le=10)
    card_index: int | None = Field(default=None, ge=1, le=5)
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ROICropResult(PersistedModel):

    roi_key: str
    roi_type: ROIType
    source_frame_id: int | None = None
    source_timestamp_seconds: float | None = Field(default=None, ge=0)
    source_image_path: str
    crop_path: str
    crop_width: int = Field(ge=0)
    crop_height: int = Field(ge=0)
