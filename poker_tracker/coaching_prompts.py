from __future__ import annotations

from poker_tracker.analytics import SessionStats
from poker_tracker.equity import EquityResult
from poker_tracker.hand_history import format_hand_history
from poker_tracker.models import Action, Hand, HandPlayer, Session
from poker_tracker.pot_odds import format_percentage
from poker_tracker.ranges import get_range_description, normalize_range_label


REQUIRED_REVIEW_SECTIONS = [
    "Hand Summary",
    "Theory Coach",
    "Exploit Coach",
    "EV / Math Notes",
    "Mistake Severity",
    "Best Alternative Line",
    "Study Lesson",
    "Next Review Question",
]

SESSION_REVIEW_SECTIONS = [
    "Session Summary",
    "Biggest Leaks",
    "Best Played Spots",
    "Theory Study Priorities",
    "Exploit Study Priorities",
    "Hands To Review Again",
    "Next Study Plan",
]

POST_SESSION_SAFETY = (
    "This is strictly post-session analysis of completed hands. Do not provide "
    "real-time assistance, current-hand recommendations, live table advice, "
    "live table capture, poker-client overlays, hotkeys, or capture guidance."
)


def build_hand_review_prompt(
    session: Session,
    hand: Hand,
    actions: list[Action],
    players: list[HandPlayer] | None = None,
    *,
    pot_odds_facts: dict[str, float | str] | None = None,
    equity_result: EquityResult | None = None,
    villain_range_label: str | None = None,
    coaching_mode: str = "Theory + Exploit",
) -> str:
    """Build a structured prompt for a future LLM hand review without calling one."""
    range_label = normalize_range_label(villain_range_label)
    range_description = get_range_description(range_label)
    facts = _format_math_facts(pot_odds_facts or {})
    equity = _format_equity(equity_result)
    sections = "\n".join(f"- {section}" for section in REQUIRED_REVIEW_SECTIONS)

    return f"""Post-session safety:
{POST_SESSION_SAFETY}

Do not invent equities, solver outputs, range facts, population reads, or exact math.
If an equity result is labeled placeholder/estimated, state that clearly.
Coaching mode: {coaching_mode}

Hand history:
{format_hand_history(session, hand, actions, players or [])}

Hand tags: {", ".join(hand.tags) if hand.tags else "none"}
Result: {hand.hero_bb_won if hand.hero_bb_won is not None else "unknown"} BB
Villain range label: {range_description.label}
Villain range description: {range_description.description}
Pot odds / math facts:
{facts}
Equity result:
{equity}

Return exactly these sections:
{sections}
"""


def build_session_review_prompt(
    session: Session,
    stats: SessionStats,
    hand_histories: list[str],
    *,
    coaching_mode: str = "Theory + Exploit",
) -> str:
    """Build a structured prompt for a future LLM session review without calling one."""
    sections = "\n".join(f"- {section}" for section in SESSION_REVIEW_SECTIONS)
    histories = "\n\n---\n\n".join(hand_histories)
    return f"""Post-session safety:
{POST_SESSION_SAFETY}

Do not invent equities, solver outputs, exact EV, range frequencies, or HUD-grade stats. The stats below
are basic/manual review stats from completed hands only.
Identify patterns from the supplied completed hands, but do not overclaim.
Coaching mode: {coaching_mode}

Session: {session.date_played.isoformat()} {session.platform} {session.name}
Hands: {stats.hand_count}
Total Hero result: {stats.total_hero_bb:g} BB
Average result: {stats.average_hero_bb:g} BB/hand
Tags: {stats.hands_by_tag}
Review statuses: {stats.hands_by_review_status}
Action counts: {stats.action_counts_by_type}

Hand histories:
{histories}

Return exactly these sections:
{sections}
"""


def _format_math_facts(facts: dict[str, float | str]) -> str:
    if not facts:
        return "- none provided"
    lines = []
    for key, value in facts.items():
        if isinstance(value, float) and 0 <= value <= 1 and (
            "equity" in key or "frequency" in key or "threshold" in key
        ):
            lines.append(f"- {key}: {format_percentage(value)}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _format_equity(equity_result: EquityResult | None) -> str:
    if equity_result is None:
        return "- none provided"
    equity = "unavailable" if equity_result.equity is None else format_percentage(equity_result.equity)
    return (
        f"- hero_hand: {equity_result.hero_hand}\n"
        f"- board: {equity_result.board or 'none'}\n"
        f"- villain_range_label: {equity_result.villain_range_label}\n"
        f"- equity: {equity}\n"
        f"- method: {equity_result.method}\n"
        f"- confidence: {format_percentage(equity_result.confidence)}\n"
        f"- notes: {equity_result.notes}"
    )
