"""The four standalone study calculators from the Math review surface.

Realization, multiway pot share, outs, and ICM. They are grouped here because
they share a shape the rest of ``app.py`` does not: each renders its own inputs,
computes from ``poker_tracker.math`` alone, and writes nothing. None of them
touches the database, the session, readiness, or any stored fact, so none of them
can participate in the invariants the rest of the UI is checked against.

That independence is why this is the first slice taken out of ``app.py``: no
source-text guard counts a symbol in here, and no caller outside the Math review
tab reaches them. ``app.py`` re-exports every public name so
``app.show_icm_tool`` and ``app._icm_risk_premium_readout`` keep resolving --
``tests/test_icm_tool_readout.py`` reaches for both through the ``app`` module.
"""

from __future__ import annotations

import streamlit as st

from poker_tracker.math.equity import get_equity_calculator
from poker_tracker.math.icm import icm_equities, icm_risk_premium_range
from poker_tracker.math.pot_odds import format_percentage
from poker_tracker.math.study_math import (
    REALIZATION_FACTOR_GUIDE,
    outs_to_equity_exact,
    outs_to_equity_rule,
    realized_equity,
)
from poker_tracker.persistence.models import Hand


@st.cache_data(show_spinner=False)
def _cached_multiway_equity(hero_cards: str, board_cards: str, villain_ranges: tuple[str, ...]):
    calculator = get_equity_calculator()
    return calculator.calculate_equity_multiway(hero_cards, board_cards, list(villain_ranges))


def show_equity_realization_tool(equity_result) -> None:
    st.caption(
        "Raw equity assumes every hand reaches showdown. Out of position or with a capped "
        "range, Hero realizes less of it. Factors are study heuristics, not solver output."
    )
    if equity_result is None or equity_result.equity is None:
        st.info("Compute Hero equity vs a range above to estimate realized equity.")
        return
    scenario = st.selectbox(
        "Realization scenario",
        options=list(REALIZATION_FACTOR_GUIDE.keys()),
        format_func=lambda key: key.replace("_", " "),
    )
    factor = REALIZATION_FACTOR_GUIDE[scenario]
    realized = realized_equity(equity_result.equity, factor)
    raw_col, factor_col, realized_col = st.columns(3)
    raw_col.metric("Raw equity", format_percentage(equity_result.equity))
    factor_col.metric("Realization factor", f"{factor:.2f}×")
    realized_col.metric("Realized equity (est.)", format_percentage(realized))


def show_multiway_equity_tool(hand: Hand, range_label: str, range_display: str) -> None:
    st.caption(
        "Pot-share equity vs two or more ranges. Villain 1 uses the range selected above; "
        "add at least one more villain. Multiway pots need stronger hands to continue."
    )
    if not hand.hero_cards:
        st.info("This hand has no Hero cards recorded.")
        return
    second = st.text_input(
        "Villain 2 range", value="standard", help="A range label or standard notation."
    )
    third = st.text_input("Villain 3 range (optional)", value="")
    villain_ranges = [range_label, second.strip(), *([third.strip()] if third.strip() else [])]
    if not second.strip():
        st.info("Enter a Villain 2 range to compute multiway equity.")
        return
    try:
        with st.spinner("Computing multiway pot share..."):
            result = _cached_multiway_equity(
                hand.hero_cards, hand.board_cards, tuple(villain_ranges)
            )
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        return
    if result.equity is None:
        st.warning(f"Could not compute: {result.notes}")
        return
    share_col, fair_col = st.columns(2)
    help_text = result.notes
    if result.std_error:
        low, high = (
            max(0.0, result.equity - 1.96 * result.std_error),
            min(1.0, result.equity + 1.96 * result.std_error),
        )
        help_text += f" 95% CI: {format_percentage(low)}–{format_percentage(high)}."
    share_col.metric(
        f"Hero pot share ({len(villain_ranges) + 1}-way)",
        format_percentage(result.equity),
        help=help_text,
    )
    fair_col.metric(
        "Fair share",
        format_percentage(1 / (len(villain_ranges) + 1)),
        help="An equal split of the pot. Above this, Hero is profiting from the multiway pot.",
    )
    st.caption(f"Villain 1: {range_display} · ranges: {result.villain_range_label}")


def show_outs_tool() -> None:
    st.caption("Draw equity from counted outs — the rule of 2 and 4 next to the exact odds.")
    outs = st.number_input("Outs", min_value=0, max_value=20, value=9, step=1)
    street = st.radio(
        "Cards to come",
        options=["Flop → river (2 cards)", "Turn → river (1 card)"],
        horizontal=True,
    )
    streets_to_come = 2 if street.startswith("Flop") else 1
    unseen = 47 if streets_to_come == 2 else 46
    if outs == 0:
        st.info("Count Hero's outs to estimate draw equity.")
        return
    rule = outs_to_equity_rule(int(outs), streets_to_come)
    exact = outs_to_equity_exact(int(outs), unseen, streets_to_come)
    rule_col, exact_col = st.columns(2)
    rule_col.metric("Rule of 2 and 4", format_percentage(min(rule, 1.0)))
    exact_col.metric(
        "Exact",
        format_percentage(exact),
        help=f"{outs} outs among {unseen} unseen cards, {streets_to_come} card(s) to come.",
    )


def _icm_risk_premium_readout(
    stacks: list[float],
    payouts: list[float],
    hero_index: int,
    risk_amount: float,
) -> tuple[str, str]:
    """The risk premium as the span it is, plus what the span does not bound.

    The cost of losing the chips genuinely depends on which opponent takes them,
    so a single figure under prose reading "prize equity lost" states one of
    several answers as though it were the only one. The screen has room for the
    span and for who sits at each end, so it shows them.

    The span covers single winners only. Chips split between several opponents
    -- a multiway all-in with side pots -- can cost Hero more than its top end,
    so the help says so rather than letting two numbers read as bounds.
    """
    span = icm_risk_premium_range(stacks, payouts, hero_index, risk_amount)
    cheapest = min(span.by_opponent, key=lambda seat: (span.by_opponent[seat], seat))
    dearest = max(span.by_opponent, key=lambda seat: (span.by_opponent[seat], -seat))
    if f"{span.low:.2f}" == f"{span.high:.2f}":
        value = f"{span.high:.2f}"
        detail = "Every opponent costs Hero the same here."
    else:
        value = f"{span.low:.2f} – {span.high:.2f}"
        detail = (
            f"Cheapest if player {cheapest + 1} wins the chips, "
            f"dearest if player {dearest + 1} does."
        )
    help_text = (
        "Prize equity Hero loses, by which opponent wins the chips. "
        f"{detail} Compare against the prize equity gained by winning the same "
        "pot — the gap is the ICM risk premium. Chips split between several "
        "opponents can cost more than the top of this range."
    )
    return value, help_text


def show_icm_tool() -> None:
    st.caption(
        "Malmuth-Harville ICM: converts tournament chip stacks into prize equity. "
        "Chips lost hurt more than chips won help — the risk premium quantifies that."
    )
    stacks_text = st.text_input("Stacks (comma-separated chips)", value="5000, 3000, 2000")
    payouts_text = st.text_input("Payouts (comma-separated, best first)", value="50, 30, 20")
    try:
        stacks = [float(part) for part in stacks_text.split(",") if part.strip()]
        payouts = [float(part) for part in payouts_text.split(",") if part.strip()]
        equities = icm_equities(stacks, payouts)
    except ValueError as exc:
        st.error(f"Could not compute ICM: {exc}")
        return
    st.dataframe(
        [
            {
                "Player": index + 1,
                "Stack": f"{stack:g}",
                "Chip share": format_percentage(stack / sum(stacks)),
                "ICM equity": f"{equity:.2f}",
                "Prize share": format_percentage(equity / sum(payouts)),
            }
            for index, (stack, equity) in enumerate(zip(stacks, equities, strict=True))
        ],
        hide_index=True,
        width="stretch",
    )
    hero_col, risk_col = st.columns(2)
    hero_seat = hero_col.number_input(
        "Hero player #", min_value=1, max_value=len(stacks), value=1, step=1
    )
    max_risk = stacks[int(hero_seat) - 1]
    risk_amount = risk_col.number_input(
        "Chips at risk",
        min_value=0.0,
        max_value=float(max_risk),
        value=min(1000.0, max_risk / 2),
        step=100.0,
    )
    if risk_amount > 0 and risk_amount < max_risk:
        value, help_text = _icm_risk_premium_readout(
            stacks, payouts, int(hero_seat) - 1, risk_amount
        )
        st.metric("ICM cost of losing those chips", value, help=help_text)
