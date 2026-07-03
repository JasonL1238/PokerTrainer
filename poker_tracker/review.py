from __future__ import annotations

from datetime import date

from poker_tracker.models import Action, Hand, HandPlayer, HandReview
from poker_tracker.hand_history import format_hand_history


AGGRESSIVE_ACTIONS = {"bet", "raise", "all-in"}
PASSIVE_ACTIONS = {"check", "call"}


def generate_mock_review(
    hand: Hand,
    actions: list[Action],
    players: list[HandPlayer] | None = None,
) -> HandReview:
    """Generate deterministic placeholder coaching for a completed hand."""
    session_stub = _SessionStub()
    history = format_hand_history(session_stub, hand, actions, players or [])
    aggressive_count = sum(1 for action in actions if action.action_type in AGGRESSIVE_ACTIONS)
    passive_count = sum(1 for action in actions if action.action_type in PASSIVE_ACTIONS)

    return HandReview(
        hand_id=_require_hand_id(hand),
        hand_summary=history,
        theory_coach=_theory_notes(hand, aggressive_count, passive_count),
        exploit_coach=_exploit_notes(hand),
        ev_math_notes=_ev_math_notes(hand, aggressive_count, passive_count),
        study_lesson=_study_lesson(hand),
        next_review_question=_next_review_question(hand),
    )


def _require_hand_id(hand: Hand) -> int:
    if hand.id is None:
        raise ValueError("Cannot review a hand before it has been saved.")
    return hand.id


def _theory_notes(hand: Hand, aggressive_count: int, passive_count: int) -> str:
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
    return " ".join(notes)


def _exploit_notes(hand: Hand) -> str:
    notes = [
        "Use this only for post-session pattern study, not current-hand recommendations."
    ]
    if "MISSED_VALUE" in hand.tags:
        notes.append("If this pool calls too wide, missed river value is a likely leak.")
    if "PREFLOP_3BET_SPOT" in hand.tags:
        notes.append("Compare the 3-bet spot against opener position and stack depth.")
    if (hand.hero_bb_won or 0) <= -25:
        notes.append("Large losing hands deserve a call-down or stack-off necessity review.")
    if (hand.hero_bb_won or 0) >= 10:
        notes.append("For large wins, check whether earlier streets could have built a bigger pot.")
    return " ".join(notes)


def _ev_math_notes(hand: Hand, aggressive_count: int, passive_count: int) -> str:
    result = hand.hero_bb_won or 0
    notes = [f"Recorded result: {result:g} BB."]
    if hand.pot_size is not None:
        notes.append(f"Final pot recorded as {hand.pot_size:g}.")
    notes.append(f"Aggressive actions: {aggressive_count}; passive actions: {passive_count}.")
    notes.append("Future equity/pot-odds calculations should plug in here as a separate module.")
    return " ".join(notes)


def _study_lesson(hand: Hand) -> str:
    if "MISSED_VALUE" in hand.tags:
        return "Write the worse hands that could call a small river value bet."
    if "MULTIWAY" in hand.tags:
        return "List which hands remain strong enough to continue multiway by street."
    if (hand.hero_bb_won or 0) < -25:
        return "Mark the earliest street where Hero could control pot size or fold."
    return "Pick the highest-leverage decision and compare two alternative lines."


def _next_review_question(hand: Hand) -> str:
    if "RIVER_DECISION" in hand.tags:
        return "What exact worse hands call, and what better hands fold, on the river?"
    if "PREFLOP_3BET_SPOT" in hand.tags:
        return "What is Hero's 3-bet range versus this opener position and stack depth?"
    return "What assumption about villain's range most changes the best line?"


class _SessionStub:
    date_played = date.today()
    platform = "Manual Review"


# TODO: Replace this deterministic mock with a real coaching provider behind an interface.
# TODO: Add equity/solver annotations as separate services, not inside this mock generator.
