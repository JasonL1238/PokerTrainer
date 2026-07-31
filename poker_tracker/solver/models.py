from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SolverStreet = Literal["flop", "turn", "river"]


class EligibilityResult(BaseModel):
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SolverPlayer(BaseModel):
    player_key: str
    player_name: str
    position: str = ""
    role: Literal["ip", "oop"]
    is_hero: bool = False


class RecordedSolverAction(BaseModel):
    player_key: str
    player_name: str
    street: SolverStreet
    action_type: str
    amount: float | None = None
    pot_before: float | None = None


class SolverSpot(BaseModel):
    hand_id: int
    table_size: int
    street: SolverStreet
    board: str
    pot: float = Field(gt=0)
    effective_stack: float = Field(gt=0)
    pot_type: Literal["single_raised", "three_bet"]
    preflop_aggressor_key: str
    oop: SolverPlayer
    ip: SolverPlayer
    hero_cards: str
    recorded_line: list[RecordedSolverAction] = Field(default_factory=list)
    prior_multiway: bool = False
    rake_amount: float = Field(default=0, ge=0)


class ResolvedRange(BaseModel):
    player_key: str
    player_name: str
    position: str = ""
    role: Literal["ip", "oop"]
    source: Literal["default", "builtin", "user", "custom"]
    profile_name: str
    notation: str
    solver_notation: str
    combo_count: int = Field(ge=1)
    range_percent: float = Field(ge=0, le=1)
    requested_profile: str = ""
    mismatches: list[str] = Field(default_factory=list)


class ActionFrequency(BaseModel):
    action: str
    frequency: float = Field(ge=0, le=1)


class SolverEvidence(BaseModel):
    backend: str = "TexasSolver"
    backend_version: str = ""
    street: SolverStreet
    board: str
    pot: float
    effective_stack: float
    hero_player: str
    hero_combo: str
    recorded_action: str = ""
    mapped_action: str = ""
    action_frequencies: list[ActionFrequency] = Field(default_factory=list)
    range_action_frequencies: list[ActionFrequency] = Field(default_factory=list)
    exploitability_pct: float | None = Field(default=None, ge=0)
    runtime_seconds: float | None = Field(default=None, ge=0)
    range_ip_name: str
    range_oop_name: str
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
