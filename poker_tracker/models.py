from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from poker_tracker.cards import CardParseError, parse_visible_cards
from poker_tracker.validation import normalize_cards, validate_tags


Street = Literal["preflop", "flop", "turn", "river", "showdown"]
ActionType = Literal[
    "fold",
    "check",
    "call",
    "bet",
    "raise",
    "all-in",
    "post_blind",
    "show",
    "win",
]
ReviewStatus = Literal["unreviewed", "reviewed", "needs_correction"]
SourceType = Literal["manual", "cv_import", "corrected_cv"]
ReviewType = Literal["hand", "session"]
SafetyMode = Literal["post_session_only"]
JobStatus = Literal["queued", "running", "completed", "failed"]
JobType = Literal["frame_extraction"]
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
    return datetime.now(timezone.utc)


class Session(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    name: str = Field(min_length=1)
    date_played: date = Field(default_factory=date.today)
    platform: str = "Manual"
    stakes: str = ""
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class Hand(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
    def validate_visible_card_uniqueness(self) -> "Hand":
        if self.hero_cards and self.board_cards:
            try:
                parse_visible_cards(self.hero_cards, self.board_cards)
            except CardParseError as exc:
                raise ValueError(str(exc)) from exc
        return self


class HandPlayer(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    hand_id: int
    player_name: str = Field(min_length=1)
    position: str = ""
    starting_stack: float | None = Field(default=None, ge=0)
    is_hero: bool = False
    notes: str = ""


class Action(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    hand_id: int
    street: Street
    action_index: int | None = Field(default=None, ge=1)
    player_name: str = Field(min_length=1)
    position: str = ""
    action_type: ActionType
    amount: float | None = Field(default=None, ge=0)
    pot_before: float | None = Field(default=None, ge=0)
    stack_before: float | None = Field(default=None, ge=0)
    notes: str = ""

    @field_validator("action_type", mode="before")
    @classmethod
    def normalize_action_type(cls, value: str) -> str:
        if value == "all_in":
            return "all-in"
        return value


class HandReview(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    hand_id: int
    hand_summary: str
    theory_coach: str
    exploit_coach: str
    ev_math_notes: str = ""
    study_lesson: str
    next_review_question: str = ""
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class LLMProviderConfig(BaseModel):
    provider_name: str = "mock"
    model_name: str = "mock-local"
    has_api_key: bool = False


class CoachingRequest(BaseModel):
    prompt: str
    review_type: ReviewType
    provider_name: str = "mock"
    model_name: str = "mock-local"
    hand_id: int | None = None
    session_id: int | None = None
    safety_mode: SafetyMode = "post_session_only"


class CoachingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
    created_at: datetime = Field(default_factory=utc_now)


class VideoRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


class ProcessingJob(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    job_type: JobType
    status: JobStatus = "queued"
    video_id: int
    progress_percent: float = Field(default=0, ge=0, le=100)
    message: str = ""
    error_message: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ExtractedFrame(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    video_id: int
    job_id: int
    timestamp_seconds: float = Field(ge=0)
    frame_index: int = Field(ge=0)
    image_path: str
    created_at: datetime = Field(default_factory=utc_now)


class ROIProfile(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


class ROIRegion(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


class ROICropResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    roi_key: str
    roi_type: ROIType
    source_frame_id: int | None = None
    source_timestamp_seconds: float | None = Field(default=None, ge=0)
    source_image_path: str
    crop_path: str
    crop_width: int = Field(ge=0)
    crop_height: int = Field(ge=0)
