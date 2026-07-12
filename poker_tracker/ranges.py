from __future__ import annotations

from dataclasses import dataclass


RANGE_LABELS = {"tight", "standard", "loose", "very_loose", "premium", "unknown"}


@dataclass(frozen=True)
class RangeDescription:
    label: str
    description: str
    examples: list[str]
    # Machine-readable range in standard poker notation (eval7 `HandRange` syntax),
    # used by the equity engine to expand the label into concrete combos. `None`
    # for labels that carry no reliable definition (e.g. `unknown`).
    notation: str | None = None


RANGES = {
    "premium": RangeDescription(
        label="premium",
        description="Very strong hands, often 3-bet/4-bet candidates.",
        examples=["AA", "KK", "QQ", "AKs", "AKo"],
        notation="AA,KK,QQ,AKs,AKo",
    ),
    "tight": RangeDescription(
        label="tight",
        description="Narrow continuing range with stronger pairs, broadways, and suited aces.",
        examples=["TT+", "AQ+", "AJs+", "KQs"],
        notation="TT+,AJs+,AQo+,KQs",
    ),
    "standard": RangeDescription(
        label="standard",
        description="Reasonable default range for unknown regulars.",
        examples=["77+", "ATs+", "KJs+", "QJs", "AQo+"],
        notation="77+,ATs+,KJs+,QJs,AQo+",
    ),
    "loose": RangeDescription(
        label="loose",
        description="Wider range with more suited hands, broadways, and speculative calls.",
        examples=["55+", "Axs", "KTs+", "QTs+", "JTs", "T9s"],
        notation="55+,A2s+,KTs+,QTs+,JTs,T9s,ATo+,KJo+",
    ),
    "very_loose": RangeDescription(
        label="very_loose",
        description="Very wide range; can include many weak suited, offsuit, and connected hands.",
        examples=["any pair", "many suited kings", "suited connectors", "broadway offsuit"],
        notation="22+,A2s+,K2s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s,A2o+,K7o+,Q8o+,J8o+,T8o+,98o",
    ),
    "unknown": RangeDescription(
        label="unknown",
        description="No reliable range estimate from the current manual data.",
        examples=[],
        notation=None,
    ),
}


def get_range_description(label: str) -> RangeDescription:
    """Return the predefined range description for a label."""
    normalized = normalize_range_label(label)
    return RANGES[normalized]


def range_notation(label: str) -> str | None:
    """Return the machine-readable range notation for a label, or `None` if undefined."""
    return get_range_description(label).notation


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
