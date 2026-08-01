from __future__ import annotations

import copy
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


class SolverRunParameters(BaseModel):
    """The solve settings a retained frequency vector cannot be read without.

    Frequencies only mean something relative to the tree that produced them: a
    33%/75% flop abstraction and a 25%/50%/pot abstraction give different mixes
    for the same spot, and a run that stopped short of its accuracy target is a
    different claim from one that reached it. Those settings used to exist only
    as text inside the run directory's command file, which the product deletes
    with the hand and operators prune by hand, leaving the row still presenting
    its frequencies as evidence with no record of what was actually solved.
    """

    backend: str = ""
    accuracy_target_pct: float | None = None
    max_iterations: int | None = None
    tree: dict[str, object] = Field(default_factory=dict)

    @property
    def is_retained(self) -> bool:
        """Whether enough was recorded to say what this run actually solved."""

        return bool(self.tree) and self.max_iterations is not None

    def summary_lines(self) -> list[str]:
        """Render the settings for readers who no longer have the artifacts."""

        if not self.is_retained:
            return []
        tree = "; ".join(
            f"{key} {_render_tree_value(self.tree[key])}" for key in sorted(self.tree)
        )
        convergence = (
            "not recorded"
            if self.accuracy_target_pct is None
            else f"{self.accuracy_target_pct:g}% of the starting pot"
        )
        return [
            f"Solved tree · {tree}",
            (
                f"Convergence target · exploitability below {convergence} "
                f"within {self.max_iterations} iterations"
            ),
        ]


def current_run_parameters() -> SolverRunParameters:
    """Snapshot the settings this build solves with, for stamping onto a run.

    The adapter is imported here rather than at module scope because it imports
    these models; the values are plain constants, so this reads no files and
    starts no processes and is safe to call while constructing a row.
    """

    from poker_tracker.solver import texassolver

    return SolverRunParameters(
        backend=texassolver.BACKEND_NAME,
        accuracy_target_pct=texassolver.DEFAULT_ACCURACY_PCT,
        max_iterations=texassolver.DEFAULT_MAX_ITERATIONS,
        tree=copy.deepcopy(texassolver.TREE_SPEC),
    )


def _render_tree_value(value: object) -> str:
    if isinstance(value, list | tuple):
        return ",".join(str(item) for item in value)
    return str(value)
