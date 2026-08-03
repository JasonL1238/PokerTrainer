from __future__ import annotations

from poker_tracker.coaching.hand_history import format_hand_history
from poker_tracker.math.accounting import HandLedger
from poker_tracker.math.analytics import SessionStats
from poker_tracker.math.equity import EquityResult
from poker_tracker.math.pot_odds import format_percentage
from poker_tracker.math.preflop_ranges import range_percent, resolve_preflop_range
from poker_tracker.math.ranges import get_range_description, normalize_range_label
from poker_tracker.persistence.models import Action, Hand, HandPlayer, Session
from poker_tracker.solver.models import SolverEvidence

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

# The heading this module writes above the solver block, and the body it writes
# when there is no run to describe.
SOLVER_EVIDENCE_HEADING = "Solver evidence:"
NO_SOLVER_EVIDENCE = "- none provided"


def retained_solver_evidence(prompt: str) -> str | None:
    """Recover the solver block a prompt actually carried, or None if it carried none.

    A saved response has to stay checkable long after the run behind it was
    deleted or superseded, and the prompt is the one record that is certainly the
    response's own. Reading the evidence back out of the prompt is also what stops
    a checker from being handed evidence the model was never shown: there is no
    parameter to pass the wrong thing to.

    The last heading wins. Hero notes travel into the prompt inside the hand
    history, so an operator could write this heading there; the block this module
    appends is always the later one.
    """
    lines = prompt.splitlines()
    start = next(
        (
            index
            for index in range(len(lines) - 1, -1, -1)
            if lines[index] == SOLVER_EVIDENCE_HEADING
        ),
        None,
    )
    if start is None:
        return None
    block: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("- "):
            break
        block.append(line)
    if not block or block == [NO_SOLVER_EVIDENCE]:
        return None
    return "\n".join(block)


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
    hero_position: str | None = None,
    hero_range_scenario: str = "RFI",
    ledger: HandLedger | None = None,
    accounting_issues: list[str] | None = None,
    accounting_authoritative: bool = False,
    solver_evidence: SolverEvidence | None = None,
) -> str:
    """Build a structured prompt for a future LLM hand review without calling one."""
    range_label = normalize_range_label(villain_range_label)
    range_description = get_range_description(range_label)
    facts = _format_math_facts(pot_odds_facts or {})
    equity = _format_equity(equity_result)
    solver = _format_solver_evidence(solver_evidence)
    hero_range_block = _format_hero_range(
        hero_position if hero_position is not None else hand.hero_position,
        hero_range_scenario,
    )
    sections = "\n".join(f"- {section}" for section in REQUIRED_REVIEW_SECTIONS)
    result_bb = hand.hero_bb_won
    if accounting_authoritative and ledger is not None and players:
        hero = next((player for player in players if player.is_hero), None)
        if hero is not None:
            result_bb = ledger.net_results.get(hero.player_key, result_bb)

    return f"""Post-session safety:
{POST_SESSION_SAFETY}

Do not invent equities, solver outputs, range facts, population reads, exact math, or EV loss.
If an equity result is labeled placeholder/estimated, state that clearly.
If solver evidence is supplied, preserve every frequency exactly, explain mixed
strategies as mixes, and state its material assumptions. TexasSolver strategy JSON
does not establish exact action EV or BB loss, so do not invent either.
Coaching mode: {coaching_mode}

Hand history:
{format_hand_history(
    session,
    hand,
    actions,
    players or [],
    ledger=ledger,
    accounting_issues=accounting_issues,
    accounting_authoritative=accounting_authoritative,
)}

Hand tags: {", ".join(hand.tags) if hand.tags else "none"}
Result: {result_bb if result_bb is not None else "unknown"} BB
Villain range label: {range_description.label}
Villain range description: {range_description.description}{hero_range_block}
Pot odds / math facts:
{facts}
Equity result:
{equity}
Solver evidence:
{solver}

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


def _format_hero_range(hero_position: str | None, scenario: str) -> str:
    """Return a hero preflop-range block, or "" if no chart resolves.

    The block is a positional study baseline (not solver output), consistent
    with the anti-hallucination framing of the rest of the prompt. Returns an
    empty string when the position/scenario has no defined chart so the block
    is omitted gracefully rather than emitting an empty section.
    """
    chart = resolve_preflop_range(hero_position, scenario)
    if chart is None:
        return ""
    percent = format_percentage(range_percent(chart.notation))
    return (
        f"\nHero preflop range (study baseline, not solver output):\n"
        f"- position: {chart.position}\n"
        f"- scenario: {chart.scenario}\n"
        f"- notation: {chart.notation}\n"
        f"- percent of hands: {percent}\n"
        f"- description: {chart.description}"
    )


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


def _format_solver_evidence(evidence: SolverEvidence | None) -> str:
    if evidence is None:
        return "- none provided"
    combo_strategy = (
        ", ".join(
            f"{item.action} {format_percentage(item.frequency)}"
            for item in evidence.action_frequencies
        )
        or "unavailable"
    )
    range_strategy = (
        "omitted because combo-specific strategy is available"
        if evidence.action_frequencies
        else (
            ", ".join(
                f"{item.action} {format_percentage(item.frequency)}"
                for item in evidence.range_action_frequencies
            )
            or "unavailable"
        )
    )
    exploitability = (
        "unavailable"
        if evidence.exploitability_pct is None
        else f"{evidence.exploitability_pct:g}% of the starting pot"
    )
    assumptions = "; ".join(evidence.assumptions) or "none recorded"
    warnings = "; ".join(evidence.warnings) or "none"
    return (
        f"- source: {evidence.backend} {evidence.backend_version}\n"
        f"- spot: {evidence.street} on {evidence.board}; pot {evidence.pot:g} BB; "
        f"effective stack {evidence.effective_stack:g} BB\n"
        f"- hero: {evidence.hero_player} with {evidence.hero_combo}\n"
        f"- recorded_action: {evidence.recorded_action or 'unavailable'}\n"
        f"- mapped_solver_action: {evidence.mapped_action or 'unavailable'}\n"
        f"- hero_combo_strategy: {combo_strategy}\n"
        f"- input_weighted_range_strategy: {range_strategy}\n"
        f"- IP_range: {evidence.range_ip_name}\n"
        f"- OOP_range: {evidence.range_oop_name}\n"
        f"- final_exploitability: {exploitability}\n"
        f"- assumptions: {assumptions}\n"
        f"- warnings: {warnings}\n"
        "- action_ev_and_bb_loss: unavailable; do not infer or invent"
    )
