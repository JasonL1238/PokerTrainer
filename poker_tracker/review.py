from __future__ import annotations

from datetime import date

from poker_tracker.equity import EquityResult
from poker_tracker.models import Action, Hand, HandPlayer, HandReview
from poker_tracker.hand_history import format_hand_history
from poker_tracker.pot_odds import format_percentage
from poker_tracker.ranges import get_range_description, normalize_range_label


AGGRESSIVE_ACTIONS = {"bet", "raise", "all-in"}
PASSIVE_ACTIONS = {"check", "call"}


def generate_mock_review(
    hand: Hand,
    actions: list[Action],
    players: list[HandPlayer] | None = None,
    *,
    math_facts: dict[str, float | str] | None = None,
    equity_result: EquityResult | None = None,
    villain_range_label: str | None = None,
) -> HandReview:
    """Generate deterministic placeholder coaching for a completed hand."""
    session_stub = _SessionStub()
    history = format_hand_history(session_stub, hand, actions, players or [])
    aggressive_count = sum(1 for action in actions if action.action_type in AGGRESSIVE_ACTIONS)
    passive_count = sum(1 for action in actions if action.action_type in PASSIVE_ACTIONS)
    facts = math_facts or {}
    range_label = normalize_range_label(villain_range_label)

    return HandReview(
        hand_id=_require_hand_id(hand),
        hand_summary=history,
        theory_coach=_theory_notes(hand, aggressive_count, passive_count, facts, equity_result),
        exploit_coach=_exploit_notes(hand, range_label),
        ev_math_notes=_ev_math_notes(hand, aggressive_count, passive_count, facts, equity_result),
        study_lesson=_study_lesson(hand),
        next_review_question=_next_review_question(hand, facts, equity_result),
    )


def _require_hand_id(hand: Hand) -> int:
    if hand.id is None:
        raise ValueError("Cannot review a hand before it has been saved.")
    return hand.id


def _theory_notes(
    hand: Hand,
    aggressive_count: int,
    passive_count: int,
    math_facts: dict[str, float | str],
    equity_result: EquityResult | None,
) -> str:
    notes = [
        "Review each street against Hero's range, position, board texture, and sizing plan."
    ]
    if "MULTIWAY" in hand.tags:
        notes.append("Multiway pots require tighter value thresholds and less bluffing.")
    if "RIVER_DECISION" in hand.tags:
        notes.append("River decisions should be checked against pot odds and villain range.")
    if "MISSED_VALUE" in hand.tags:
        notes.append("Look for thin value opportunities when worse hands can still call.")
    if passive_count > aggressive_count:
        notes.append("The line contains more checks/calls than bets/raises; review whether passivity capped Hero's range.")
    if aggressive_count:
        notes.append("There are aggressive actions; review bet sizing and which worse hands continue.")
    if "required_equity_to_call" in math_facts:
        notes.append(
            f"The call needs about {_format_fact_percent(math_facts['required_equity_to_call'])} equity."
        )
    if equity_result is not None and equity_result.equity is not None:
        notes.append(
            f"Equity input is {format_percentage(equity_result.equity)} by {equity_result.method}; do not treat placeholder estimates as solver truth."
        )
    return " ".join(notes)


def _exploit_notes(hand: Hand, villain_range_label: str) -> str:
    notes = [
        "Use this only for post-session pattern study, not current-hand recommendations."
    ]
    range_description = get_range_description(villain_range_label)
    if range_description.label != "unknown":
        notes.append(f"Villain range label: {range_description.label} ({range_description.description})")
    if "MISSED_VALUE" in hand.tags:
        notes.append("If this pool calls too wide, missed river value is a likely leak.")
    if "PREFLOP_3BET_SPOT" in hand.tags:
        notes.append("Compare the 3-bet spot against opener position and stack depth.")
    if (hand.hero_bb_won or 0) <= -25:
        notes.append("Large losing hands deserve a call-down or stack-off necessity review.")
    if (hand.hero_bb_won or 0) >= 10:
        notes.append("For large wins, check whether earlier streets could have built a bigger pot.")
    return " ".join(notes)


def _ev_math_notes(
    hand: Hand,
    aggressive_count: int,
    passive_count: int,
    math_facts: dict[str, float | str],
    equity_result: EquityResult | None,
) -> str:
    result = hand.hero_bb_won or 0
    notes = [f"Recorded result: {result:g} BB."]
    if hand.pot_size is not None:
        notes.append(f"Final pot recorded as {hand.pot_size:g}.")
    notes.append(f"Aggressive actions: {aggressive_count}; passive actions: {passive_count}.")
    if math_facts:
        notes.append("Math facts: " + _format_math_facts(math_facts))
    if equity_result is not None:
        equity_text = (
            "unavailable"
            if equity_result.equity is None
            else format_percentage(equity_result.equity)
        )
        notes.append(
            f"Equity: {equity_text} using {equity_result.method}, confidence {format_percentage(equity_result.confidence)}. {equity_result.notes}"
        )
    notes.append("These are approximate review aids, not solver outputs.")
    return " ".join(notes)


def _study_lesson(hand: Hand) -> str:
    if "MISSED_VALUE" in hand.tags:
        return "Write the worse hands that could call a small river value bet."
    if "MULTIWAY" in hand.tags:
        return "List which hands remain strong enough to continue multiway by street."
    if (hand.hero_bb_won or 0) < -25:
        return "Mark the earliest street where Hero could control pot size or fold."
    return "Pick the highest-leverage decision and compare two alternative lines."


def _next_review_question(
    hand: Hand,
    math_facts: dict[str, float | str],
    equity_result: EquityResult | None,
) -> str:
    if "required_equity_to_call" in math_facts and equity_result is not None:
        return "Does the estimated equity clear the required calling equity, and how reliable is that estimate?"
    if "RIVER_DECISION" in hand.tags:
        return "What exact worse hands call, and what better hands fold, on the river?"
    if "PREFLOP_3BET_SPOT" in hand.tags:
        return "What is Hero's 3-bet range versus this opener position and stack depth?"
    return "What assumption about villain's range most changes the best line?"


def _format_math_facts(facts: dict[str, float | str]) -> str:
    parts = []
    for key, value in facts.items():
        if isinstance(value, float) and 0 <= value <= 1 and (
            "equity" in key or "frequency" in key or "threshold" in key
        ):
            parts.append(f"{key}={format_percentage(value)}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _format_fact_percent(value: float | str) -> str:
    if isinstance(value, float):
        return format_percentage(value)
    return str(value)


class _SessionStub:
    date_played = date.today()
    platform = "Manual Review"


# TODO: Replace this deterministic mock with a real coaching provider behind an interface.
# TODO: Add equity/solver annotations as separate services, not inside this mock generator.
