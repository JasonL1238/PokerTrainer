from __future__ import annotations

from dataclasses import dataclass


RANGE_LABELS = {"tight", "standard", "loose", "very_loose", "premium", "unknown"}


@dataclass(frozen=True)
class RangeDescription:
    label: str
    description: str
    examples: list[str]


RANGES = {
    "premium": RangeDescription(
        label="premium",
        description="Very strong hands, often 3-bet/4-bet candidates.",
        examples=["AA", "KK", "QQ", "AKs", "AKo"],
    ),
    "tight": RangeDescription(
        label="tight",
        description="Narrow continuing range with stronger pairs, broadways, and suited aces.",
        examples=["TT+", "AQ+", "AJs+", "KQs"],
    ),
    "standard": RangeDescription(
        label="standard",
        description="Reasonable default range for unknown regulars.",
        examples=["77+", "ATs+", "KJs+", "QJs", "AQo+"],
    ),
    "loose": RangeDescription(
        label="loose",
        description="Wider range with more suited hands, broadways, and speculative calls.",
        examples=["55+", "Axs", "KTs+", "QTs+", "JTs", "T9s"],
    ),
    "very_loose": RangeDescription(
        label="very_loose",
        description="Very wide range; can include many weak suited, offsuit, and connected hands.",
        examples=["any pair", "many suited kings", "suited connectors", "broadway offsuit"],
    ),
    "unknown": RangeDescription(
        label="unknown",
        description="No reliable range estimate from the current manual data.",
        examples=[],
    ),
}


def get_range_description(label: str) -> RangeDescription:
    """Return the predefined range description for a label."""
    normalized = normalize_range_label(label)
    return RANGES[normalized]


def normalize_range_label(label: str | None) -> str:
    """Normalize unsupported or empty labels to `unknown`."""
    normalized = (label or "unknown").strip().lower()
    return normalized if normalized in RANGE_LABELS else "unknown"


def estimate_villain_range_label(tags: list[str] | None = None, notes: str = "") -> str:
    """Estimate a rough villain range label from manual tags/notes.

    This is a first-pass review aid, not a solver-derived range.
    """
    text = " ".join([*(tags or []), notes]).lower()
    if any(token in text for token in ("premium", "3bet", "3-bet", "three_bet", "4bet")):
        return "premium"
    if any(token in text for token in ("nit", "tight", "narrow")):
        return "tight"
    if any(token in text for token in ("very loose", "whale", "splashy", "maniac")):
        return "very_loose"
    if any(token in text for token in ("loose", "passive", "station", "wide")):
        return "loose"
    return "unknown"
