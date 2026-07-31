"""Versioned corpus / answer-key schema constants and loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
ANSWER_KEY_SCHEMA_VERSION = 1

VALID_SPLITS: frozenset[str] = frozenset({"development", "validation", "locked_test"})
VALID_COMPLETION_CLASSES: frozenset[str] = frozenset(
    {"complete", "partial", "uncertain"}
)
VALID_TERMINAL_EVENTS: frozenset[str] = frozenset(
    {"showdown", "fold_win", "unobserved", "recording_end", "unsupported"}
)
VALID_STREETS: frozenset[str] = frozenset(
    {"preflop", "flop", "turn", "river", "showdown"}
)
VALID_ACTIONS: frozenset[str] = frozenset(
    {
        "fold",
        "check",
        "call",
        "bet",
        "raise",
        "all-in",
        "post_blind",
        "ante",
        "show",
    }
)
AMOUNT_SEMANTICS: frozenset[str] = frozenset({"incremental", "to", "unknown"})

# Coverage tags required by PLAN.md Phase 2. A release corpus must eventually
# cover every tag at least once across committed cases.
REQUIRED_COVERAGE_TAGS: tuple[str, ...] = (
    "players_5",
    "players_6",
    "players_7",
    "players_8",
    "hero_fold_preflop",
    "opponent_fold_before_hero",
    "limped_pot",
    "single_raised_pot",
    "three_bet_pot",
    "postflop_fold",
    "flop_completion",
    "turn_completion",
    "river_completion",
    "showdown",
    "voluntary_shown_cards",
    "all_in",
    "uncalled_bet_return",
    "short_stack",
    "large_stack",
    "seat_join_leave",
    "table_lobby_transition",
    "long_pause_or_animation",
    "blurry_or_occluded_frames",
    "recording_start_mid_hand",
    "recording_end_mid_hand",
    "supported_layout_profile",
    "unsupported_geometry_rejection",
)

DEFAULT_MINIMUMS: dict[str, int] = {
    "sessions": 10,
    "completed_hands": 100,
    "partial_hands": 10,
    "development_sessions": 5,
    "validation_sessions": 2,
    "locked_test_sessions": 3,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value
