from __future__ import annotations

from dataclasses import dataclass


# Positional preflop baselines for 100bb 9-max no-limit hold'em cash games.
# These are GTO-informed study charts for post-session review, not solver output:
# they exist so villain/hero ranges in the Math Review tab can start from a
# defensible positional baseline instead of a one-size-fits-all label.

TOTAL_COMBOS = 1326


@dataclass(frozen=True)
class PreflopRange:
    position: str
    scenario: str
    notation: str
    description: str


POSITIONS = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
SCENARIOS = ["rfi", "vs_open_call", "vs_open_3bet"]


_RANGES: dict[tuple[str, str], PreflopRange] = {}


def _define(position: str, scenario: str, notation: str, description: str) -> None:
    _RANGES[(position, scenario)] = PreflopRange(
        position=position, scenario=scenario, notation=notation, description=description
    )


# --- Raise first in (opening) ranges -------------------------------------------------
_define(
    "UTG",
    "rfi",
    "22+,ATs+,A5s,KTs+,QTs+,JTs,T9s,AJo+,KQo",
    "Tight opening range from under the gun (~13% of hands).",
)
_define(
    "UTG+1",
    "rfi",
    "22+,ATs+,A5s,A4s,KTs+,QTs+,JTs,T9s,98s,ATo+,KQo",
    "Early-position open, slightly wider than UTG (~15% of hands).",
)
_define(
    "LJ",
    "rfi",
    "22+,A6s+,A5s-A4s,K9s+,Q9s+,J9s+,T9s,98s,87s,ATo+,KJo+",
    "Lojack open adding more suited aces and connectors (~17% of hands).",
)
_define(
    "HJ",
    "rfi",
    "22+,A2s+,K8s+,Q8s+,J9s+,T8s+,98s,87s,76s,ATo+,KJo+,QJo",
    "Hijack open with all suited aces and more suited broadways (~20% of hands).",
)
_define(
    "CO",
    "rfi",
    "22+,A2s+,K5s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,54s,A9o+,KTo+,QTo+,JTo",
    "Cutoff open (~26% of hands).",
)
_define(
    "BTN",
    "rfi",
    "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,96s+,86s+,75s+,65s,54s,A2o+,K9o+,Q9o+,J9o+,T9o,98o",
    "Button open, the widest profitable opening range (~41% of hands).",
)
_define(
    "SB",
    "rfi",
    "22+,A2s+,K5s+,Q7s+,J8s+,T8s+,97s+,86s+,75s+,65s,54s,A7o+,K9o+,Q9o+,J9o+,T9o",
    "Small-blind open vs the big blind only (~33% of hands).",
)

# --- Facing a single open ------------------------------------------------------------
_define(
    "BTN",
    "vs_open_call",
    "22-TT,ATs+,KTs+,QTs+,JTs,T9s,98s,87s,76s,65s,AQo",
    "Button flat vs an earlier open: pairs, suited broadways, suited connectors (~9%).",
)
_define(
    "BTN",
    "vs_open_3bet",
    "99+,AJs+,A5s-A4s,KQs,AQo+",
    "Button 3-bet vs an earlier open: strong value plus suited-ace bluffs (~6%).",
)
_define(
    "BB",
    "vs_open_call",
    "22+,A2s+,K2s+,Q2s+,J5s+,T6s+,96s+,85s+,74s+,64s+,53s+,43s,A5o+,K9o+,Q9o+,J9o+,T8o+,98o,87o",
    "Big-blind defend vs a late-position open; closing the action allows a wide flat (~42%).",
)
_define(
    "BB",
    "vs_open_3bet",
    "77+,ATs+,A5s-A4s,KJs+,QJs,JTs,AQo+,KQo",
    "Big-blind 3-bet vs an open: linear value plus the best suited aces (~9%).",
)


def get_preflop_range(position: str, scenario: str) -> PreflopRange:
    """Return the study baseline chart for a position/scenario combination.

    Raises ValueError for unknown positions/scenarios and for combinations with
    no defined chart (e.g. the big blind never raises first in).
    """
    if position not in POSITIONS:
        raise ValueError(f"Unknown position: {position}")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    chart = _RANGES.get((position, scenario))
    if chart is None:
        raise ValueError(f"No baseline chart defined for {position} {scenario}.")
    return chart


_POSITION_ALIASES: dict[str, str] = {
    "UTG1": "UTG+1",
    "UTG+1": "UTG+1",
    "UTG 1": "UTG+1",
    "MP": "LJ",
    "LOJACK": "LJ",
    "HIJACK": "HJ",
    "CUTOFF": "CO",
    "BUTTON": "BTN",
    "DEALER": "BTN",
    "BU": "BTN",
    "SMALLBLIND": "SB",
    "SMALL BLIND": "SB",
    "BIGBLIND": "BB",
    "BIG BLIND": "BB",
}


def normalize_position(label: str | None) -> str | None:
    """Map a free-form position label to a canonical POSITIONS entry, or None.

    Trims/uppercases the input and resolves common aliases (e.g. "button" -> "BTN").
    Returns None for empty or unrecognised labels so callers can omit gracefully.
    """
    if not label:
        return None
    key = " ".join(label.strip().upper().split())
    if key in POSITIONS:
        return key
    return _POSITION_ALIASES.get(key)


def resolve_preflop_range(position: str | None, scenario: str = "rfi") -> PreflopRange | None:
    """Return the baseline chart for a position/scenario, or None if undefined.

    Unlike get_preflop_range this never raises: it normalizes the position label,
    lowercases the scenario, and returns None when no chart exists so prompt
    builders can skip the block instead of erroring.
    """
    canonical = normalize_position(position)
    if canonical is None:
        return None
    scenario_key = (scenario or "").strip().lower()
    return _RANGES.get((canonical, scenario_key))


def available_ranges() -> list[PreflopRange]:
    """Return every defined positional chart in position order."""
    order = {position: index for index, position in enumerate(POSITIONS)}
    scenario_order = {scenario: index for index, scenario in enumerate(SCENARIOS)}
    return sorted(
        _RANGES.values(),
        key=lambda chart: (order[chart.position], scenario_order[chart.scenario]),
    )


def range_percent(notation: str) -> float:
    """Return the fraction of all 1326 starting combos covered by a notation."""
    import eval7

    return len(eval7.HandRange(notation).hands) / TOTAL_COMBOS
