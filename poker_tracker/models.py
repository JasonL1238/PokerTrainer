from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
