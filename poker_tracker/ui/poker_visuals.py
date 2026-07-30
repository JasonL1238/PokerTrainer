"""Accessible HTML renderers for completed-hand poker visualizations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape

import streamlit as st

from poker_tracker.math.accounting import (
    HandLedger,
    LedgerError,
    build_ledger_from_records,
)
from poker_tracker.persistence.models import Action, HandPlayer
from poker_tracker.player_labels import actor_label, distinct_position

_SUITS = {
    "s": ("♠", "spade"),
    "h": ("♥", "heart"),
    "d": ("♦", "diamond"),
    "c": ("♣", "club"),
}


def _card_tokens(cards: str) -> list[str]:
    return [token.strip() for token in cards.replace(",", " ").split() if token.strip()]


def playing_card_html(card: str, *, concealed: bool = False, delay: int = 0) -> str:
    """Return one compact playing card with safe rank and suit markup."""
    if concealed:
        return (
            f'<span class="pt-card pt-card-back" style="--deal-delay:{delay}ms" '
            'aria-label="Concealed card"><span aria-hidden="true"></span></span>'
        )
    token = card.strip()
    if len(token) < 2 or token[-1].lower() not in _SUITS:
        return (
            f'<span class="pt-card pt-card-unknown" style="--deal-delay:{delay}ms" '
            f'aria-label="Unknown card">{escape(token or "?")}</span>'
        )
    rank = token[:-1].upper()
    suit, suit_name = _SUITS[token[-1].lower()]
    color = "red" if token[-1].lower() in {"h", "d"} else "black"
    label = f"{rank} of {suit_name}s"
    return (
        f'<span class="pt-card pt-card-{color}" style="--deal-delay:{delay}ms" '
        f'aria-label="{escape(label)}"><span>{escape(rank)}</span>'
        f'<span aria-hidden="true">{suit}</span></span>'
    )


def cards_html(cards: str, *, empty_count: int = 0, delay_start: int = 0) -> str:
    """Return a row of cards, optionally filling missing community-card slots."""
    tokens = _card_tokens(cards)
    rendered = [
        playing_card_html(card, delay=delay_start + index * 55) for index, card in enumerate(tokens)
    ]
    rendered.extend(
        playing_card_html("", delay=delay_start + (len(tokens) + index) * 55)
        for index in range(max(0, empty_count - len(tokens)))
    )
    return '<span class="pt-card-row">' + "".join(rendered) + "</span>"


def _bb_amount_html(value: float | None, *, signed: bool = False) -> str:
    """Render a visible, accessible big-blind amount for table graphics."""

    display = "—" if value is None else f"{value:+g}" if signed else f"{value:g}"
    accessible = "Unknown big blinds" if value is None else f"{display} big blinds"
    return (
        f'<span class="pt-bb-amount" aria-label="{escape(accessible)}">'
        f"<span>{escape(display)}</span><small>BB</small></span>"
    )


def _seat_html(
    player: HandPlayer,
    index: int,
    total: int,
    *,
    actor_player_key: str | None = None,
    folded_player_keys: frozenset[str] = frozenset(),
) -> str:
    seat_class = f"pt-seat-{index + 1}-of-{min(max(total, 2), 9)}"
    hero_class = " pt-seat-hero" if player.is_hero else ""
    player_key = player.player_key or player.player_name
    actor_class = " pt-seat-acting" if player_key == actor_player_key else ""
    folded_class = " pt-seat-folded" if player_key in folded_player_keys else ""
    name = actor_label(player.player_name, None) or player.position or f"Seat {index + 1}"
    position = distinct_position(name, player.position)
    position_html = (
        f'<span class="pt-seat-position">{escape(position)}</span>' if position else ""
    )
    return (
        f'<div class="pt-seat {seat_class}{hero_class}{actor_class}{folded_class}">'
        f'{position_html}<strong>{escape(name)}</strong>'
        f'<span class="pt-seat-stack">{_bb_amount_html(player.starting_stack)}</span></div>'
    )


def poker_table_html(
    *,
    hero_cards: str,
    board_cards: str,
    pot_size: float | None,
    players: Sequence[HandPlayer],
    result_bb: float | None = None,
    label: str = "Completed hand replay",
    actor_player_key: str | None = None,
    folded_player_keys: frozenset[str] = frozenset(),
) -> str:
    """Return an oval digital table using only completed-hand display data."""
    table_players = list(players[:9])
    seats = "".join(
        _seat_html(
            player,
            index,
            len(table_players),
            actor_player_key=actor_player_key,
            folded_player_keys=folded_player_keys,
        )
        for index, player in enumerate(table_players)
    )
    result_class = (
        "pt-result-positive"
        if (result_bb or 0) > 0
        else "pt-result-negative"
        if (result_bb or 0) < 0
        else ""
    )
    result = (
        ""
        if result_bb is None
        else (
            f'<span class="pt-table-result {result_class}">'
            f"{_bb_amount_html(result_bb, signed=True)}</span>"
        )
    )
    return (
        '<figure class="pt-poker-stage">'
        f"<figcaption><span>{escape(label)}</span>{result}</figcaption>"
        '<div class="pt-table-shell"><div class="pt-table-felt">'
        f'{seats}<div class="pt-table-center"><div class="pt-board">'
        f'{cards_html(board_cards, empty_count=5, delay_start=100)}</div>'
        f'<div class="pt-pot"><span>POT</span><strong>{_bb_amount_html(pot_size)}</strong>'
        '<i aria-hidden="true"></i></div>'
        f'<div class="pt-hero-cards"><span>HERO</span>'
        f'{cards_html(hero_cards, empty_count=2)}</div></div></div></div></figure>'
    )


def render_poker_table(**kwargs) -> None:
    """Render a completed-hand poker table."""
    st.markdown(poker_table_html(**kwargs), unsafe_allow_html=True)


def _action_board_cards(board_cards: str, street: str) -> str:
    """Return only the community cards that had been dealt on ``street``."""

    visible_count = {
        "preflop": 0,
        "flop": 3,
        "turn": 4,
        "river": 5,
        "showdown": 5,
    }.get(street, 0)
    return " ".join(_card_tokens(board_cards)[:visible_count])


@dataclass(frozen=True)
class ActionReplayState:
    """One reconstructed table state selected from a completed action line."""

    players: tuple[HandPlayer, ...]
    board_cards: str
    pot_size: float | None
    actor_player_key: str
    folded_player_keys: frozenset[str]


def action_replay_state(
    actions: Sequence[Action],
    selected_index: int,
    *,
    players: Sequence[HandPlayer],
    board_cards: str,
    initial_pot: float | None = None,
    ledger: HandLedger | None = None,
) -> ActionReplayState:
    """Reconstruct the table immediately after one saved completed-hand action."""

    if not 0 <= selected_index < len(actions):
        raise IndexError("selected action is outside the saved action history")
    stacks = {
        player.player_key or player.player_name: player.starting_stack
        for player in players
    }
    names_to_keys: dict[str, list[str]] = {}
    for player in players:
        names_to_keys.setdefault(player.player_name, []).append(
            player.player_key or player.player_name
        )
    folded: set[str] = set()
    running_pot = initial_pot
    actor_key = ""
    for index, action in enumerate(actions[: selected_index + 1]):
        matching_keys = names_to_keys.get(action.player_name, [])
        actor_key = action.player_key or (
            matching_keys[0] if len(matching_keys) == 1 else action.player_name
        )
        snapshot = (
            ledger.snapshots[index]
            if ledger is not None and index < len(ledger.snapshots)
            else None
        )
        if snapshot is not None:
            stacks[actor_key] = snapshot.stack_after
            running_pot = snapshot.pot_after
        else:
            if action.stack_before is not None:
                stacks[actor_key] = action.stack_before
            pot_before = action.pot_before if action.pot_before is not None else running_pot
            is_money_action = action.action_type in {
                "ante",
                "post_blind",
                "call",
                "bet",
                "raise",
                "all-in",
            }
            stack_before = stacks.get(actor_key)
            if is_money_action and action.amount is not None and stack_before is not None:
                stacks[actor_key] = max(0.0, stack_before - action.amount)
            running_pot = (
                pot_before + action.amount
                if pot_before is not None and action.amount is not None and is_money_action
                else pot_before
            )
        if action.action_type == "fold":
            folded.add(actor_key)
    replay_players = tuple(
        player.model_copy(
            update={"starting_stack": stacks.get(player.player_key or player.player_name)}
        )
        for player in players
    )
    selected_action = actions[selected_index]
    return ActionReplayState(
        players=replay_players,
        board_cards=_action_board_cards(board_cards, selected_action.street),
        pot_size=running_pot,
        actor_player_key=actor_key,
        folded_player_keys=frozenset(folded),
    )


def action_timeline_html(
    actions: Iterable[Action],
    *,
    players: Sequence[HandPlayer] = (),
    effective_stack: float | None = None,
    initial_pot: float | None = None,
    ledger: HandLedger | None = None,
) -> str:
    """Return the complete saved action history without truncating row details."""
    items = list(actions)
    if not items:
        return (
            '<section class="pt-history-panel pt-timeline-empty">'
            "<strong>No decision history recorded</strong>"
            "<small>Add completed-hand actions to build the replay.</small></section>"
        )
    if ledger is None and players:
        opening_pot = (
            items[0].pot_before
            if items[0].pot_before is not None
            else (initial_pot or 0)
        )
        try:
            ledger = build_ledger_from_records(
                players,
                items,
                dead_money=opening_pot,
            )
        except LedgerError:
            # Incomplete legacy/CV drafts can still be reviewed. Their recorded
            # observations remain visible, but they are not presented as a
            # reconciled derived ledger.
            ledger = None
    nodes: list[str] = []
    stacks: dict[str, float | None] = {
        player.player_key or player.player_name: player.starting_stack
        for player in players
    }
    names_to_keys: dict[str, list[str]] = {}
    for player in players:
        names_to_keys.setdefault(player.player_name, []).append(
            player.player_key or player.player_name
        )
    active_players = set(stacks)
    running_pot = initial_pot
    for index, action in enumerate(items):
        actor_key = action.player_key
        if actor_key is None:
            matching_keys = names_to_keys.get(action.player_name, [])
            actor_key = matching_keys[0] if len(matching_keys) == 1 else action.player_name
        street = action.street.title()
        amount = "—" if action.amount is None else f"{action.amount:g} BB"
        tone = action.action_type.replace("all-in", "raise")
        snapshot = (
            ledger.snapshots[index]
            if ledger is not None and index < len(ledger.snapshots)
            else None
        )
        if snapshot is not None:
            effective_range = snapshot.effective_stack_range_before
            pot_before = snapshot.pot_before
            pot_after = snapshot.pot_after
            pot_is_estimated = action.pot_before is None
            stacks[actor_key] = snapshot.stack_after
        else:
            if action.stack_before is not None:
                stacks[actor_key] = action.stack_before
                active_players.add(actor_key)
            active_stacks: list[float] = []
            for name in active_players:
                stack = stacks.get(name)
                if stack is not None:
                    active_stacks.append(stack)
            row_effective_stack = min(active_stacks) if active_stacks else effective_stack
            if effective_stack is not None:
                row_effective_stack = (
                    effective_stack
                    if row_effective_stack is None
                    else min(row_effective_stack, effective_stack)
                )
            effective_range = (
                None
                if row_effective_stack is None
                else (row_effective_stack, row_effective_stack)
            )
            pot_before = action.pot_before if action.pot_before is not None else running_pot
            pot_is_estimated = action.pot_before is None and pot_before is not None
            is_money_action = action.action_type in {
                "ante",
                "post_blind",
                "call",
                "bet",
                "raise",
                "all-in",
            }
            pot_after = (
                pot_before + action.amount
                if pot_before is not None and action.amount is not None and is_money_action
                else pot_before
            )
        estimate_mark = "~" if pot_is_estimated else ""
        pot_context = "—"
        if pot_before is not None:
            pot_values = (
                f"{estimate_mark}{pot_before:g}"
                if pot_after == pot_before
                else f"{estimate_mark}{pot_before:g} → {estimate_mark}{pot_after:g}"
            )
            pot_context = f"{pot_values} BB"
        effective_context = "—"
        if effective_range is not None:
            low, high = effective_range
            effective_context = (
                f"{low:g} BB" if abs(high - low) < 0.001 else f"{low:g}–{high:g} BB"
            )
        is_decision = action.street != "showdown" and action.action_type not in {"show", "win"}
        if snapshot is not None:
            spr_range = snapshot.spr_range_before if is_decision else None
        elif (
            is_decision
            and effective_range is not None
            and pot_before is not None
            and pot_before > 0
        ):
            spr_range = (effective_range[0] / pot_before, effective_range[1] / pot_before)
        else:
            spr_range = None
        spr_context = "—"
        if spr_range is not None:
            low, high = spr_range
            spr_context = f"{low:.1f}" if abs(high - low) < 0.05 else f"{low:.1f}–{high:.1f}"
        note_parts = [action.notes] if action.notes else []
        if (
            snapshot is not None
            and action.pot_before is not None
            and abs(action.pot_before - snapshot.pot_before) > 0.05
        ):
            note_parts.append(
                f"Recorded pot {action.pot_before:g} BB differs from ledger "
                f"{snapshot.pot_before:g} BB."
            )
        note_text = " ".join(note_parts)
        notes = (
            f'<span class="pt-history-note" title="{escape(note_text)}">'
            f"<b>Notes</b> {escape(note_text)}</span>"
            if note_text
            else ""
        )
        actor = actor_label(action.player_name, None) or "Unknown player"
        position_value = distinct_position(actor, action.position)
        position = f"<small>{escape(position_value)}</small>" if position_value else ""
        if snapshot is None:
            if (
                action.amount is not None
                and is_money_action
                and actor_key in stacks
                and stacks[actor_key] is not None
            ):
                stacks[actor_key] = max(0.0, stacks[actor_key] - action.amount)
            if pot_after is not None:
                running_pot = pot_after
        if action.action_type == "fold":
            active_players.discard(actor_key)
        nodes.append(
            f'<li class="pt-action pt-action-{escape(tone)}" style="--action-delay:{index * 45}ms">'
            '<div class="pt-history-copy">'
            '<div class="pt-history-primary">'
            f'<span class="pt-history-sequence">{index + 1:02d}</span>'
            f'<span class="pt-history-street">{escape(street)}</span>'
            f'<span class="pt-history-actor"><strong>{escape(actor)}</strong>{position}</span>'
            f'<span class="pt-history-decision">{escape(action.action_type.replace("-", " ").title())}</span>'
            f'<span class="pt-history-size">{escape(amount)}</span></div>'
            '<span class="pt-history-context">'
            f"<span><b>Pot</b> {escape(pot_context)}</span>"
            f"<span><b>Effective stack</b> {escape(effective_context)}</span>"
            f"<span><b>SPR</b> {escape(spr_context)}</span>{notes}</span></div></li>"
        )
    return (
        '<section class="pt-history-panel">'
        f'<div class="pt-history-summary"><strong>All {len(items)} saved actions</strong>'
        "<span>No saved actions are hidden.</span></div>"
        '<ol class="pt-timeline pt-decision-history" aria-label="Completed hand decision history">'
        + "".join(nodes)
        + "</ol></section>"
    )


def render_action_timeline(actions: Iterable[Action], **kwargs) -> None:
    st.markdown(action_timeline_html(actions, **kwargs), unsafe_allow_html=True)


def equity_meter_html(
    value: float, *, label: str = "Equity", threshold: float | None = None
) -> str:
    """Return a bounded meter for a previously computed equity value."""
    bounded = max(0.0, min(1.0, value))
    marker = (
        ""
        if threshold is None
        else f'<i style="left:{max(0, min(100, threshold * 100)):.2f}%"></i>'
    )
    return (
        '<div class="pt-equity">'
        f"<div><span>{escape(label)}</span><strong>{bounded * 100:.1f}%</strong></div>"
        f'<div class="pt-equity-track" role="meter" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{bounded * 100:.1f}"><b style="width:{bounded * 100:.2f}%"></b>{marker}</div></div>'
    )


def range_matrix_html(cells: Mapping[str, str], *, label: str = "Preflop range") -> str:
    """Return a 13x13 canonical Hold'em range matrix.

    ``cells`` maps labels such as ``AA``, ``AKs`` and ``AKo`` to semantic tones.
    """
    ranks = "AKQJT98765432"
    rendered: list[str] = []
    for row, first in enumerate(ranks):
        for column, second in enumerate(ranks):
            if row == column:
                hand = first + second
            elif row < column:
                hand = first + second + "s"
            else:
                hand = second + first + "o"
            tone = cells.get(hand, "off")
            rendered.append(
                f'<span class="pt-range-cell pt-range-{escape(tone)}" title="{hand}" '
                f'aria-label="{hand}: {escape(tone)}">{hand}</span>'
            )
    return f'<div class="pt-range-wrap" aria-label="{escape(label)}"><div class="pt-range-grid">{"".join(rendered)}</div></div>'


def range_cells_from_notation(notation: str) -> dict[str, str]:
    """Expand eval7 notation into semantic 13x13 range-cell tones."""
    import eval7

    order = {rank: index for index, rank in enumerate("AKQJT98765432")}
    weights: dict[str, list[float]] = {}
    for (first, second), weight in eval7.HandRange(notation).hands:
        ranks = sorted(
            (str(first)[0].upper(), str(second)[0].upper()),
            key=lambda rank: order[rank],
        )
        if ranks[0] == ranks[1]:
            label = ranks[0] + ranks[1]
        else:
            suited = str(first)[1].lower() == str(second)[1].lower()
            label = ranks[0] + ranks[1] + ("s" if suited else "o")
        weights.setdefault(label, []).append(float(weight))
    return {
        label: "value" if sum(values) / len(values) >= 0.999 else "mixed"
        for label, values in weights.items()
    }


_POKER_CSS = r"""
<style>
.pt-poker-stage { margin: 0; min-width: 0; container-type: inline-size; }
.pt-poker-stage figcaption { display: flex; align-items: center; gap: .6rem; min-height: 28px; color: var(--pt-muted); font-size: .68rem; font-family: var(--pt-font-mono); }
.pt-preview-label { color: var(--pt-gold); letter-spacing: .1em; font-family: var(--pt-font-sans); font-size: .58rem; font-weight: 760; }
.pt-table-result { margin-left: auto; color: var(--pt-muted); font-weight: 700; }
.pt-result-positive { color: var(--pt-positive); }
.pt-result-negative { color: var(--pt-negative); }
.pt-table-shell { position: relative; min-height: 310px; display: grid; place-items: center; padding: 2.4rem 3.2rem; border: 1px solid var(--pt-border); border-radius: var(--pt-radius); overflow: hidden; background: linear-gradient(145deg, #080c0a, #0d1511); }
.pt-table-shell::before { content: ""; position: absolute; inset: 0; background: linear-gradient(rgba(53,208,127,.02) 1px, transparent 1px), linear-gradient(90deg, rgba(53,208,127,.02) 1px, transparent 1px); background-size: 24px 24px; mask-image: linear-gradient(to bottom, black, transparent); }
.pt-table-felt { position: relative; width: min(100%, 580px); aspect-ratio: 1.8; border: 8px solid #1D2A23; border-radius: 50%; background: radial-gradient(ellipse at center, #123D29 0%, #0D2D20 60%, #092016 100%); box-shadow: inset 0 0 0 2px #365344, inset 0 0 70px rgba(0,0,0,.36), 0 22px 40px rgba(0,0,0,.34); }
.pt-table-felt::after { content: ""; position: absolute; inset: 9%; border: 1px solid rgba(216,239,224,.08); border-radius: 50%; }
.pt-seat { position: absolute; z-index: 3; min-width: 88px; padding: .36rem .5rem; border: 1px solid #2B3B32; border-radius: 6px; background: #0A100E; box-shadow: 0 7px 18px rgba(0,0,0,.28); text-align: center; transform: translate(-50%, -50%); }
.pt-seat strong, .pt-seat span { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pt-seat strong { color: var(--pt-text); font-size: .66rem; }
.pt-seat-position { color: var(--pt-accent); font-size: .52rem; letter-spacing: .08em; }
.pt-seat-stack { color: var(--pt-muted); font-family: var(--pt-font-mono); font-size: .57rem; margin-top: .14rem; }
.pt-bb-amount { display: inline-flex; align-items: baseline; gap: .2rem; white-space: nowrap; }
.pt-bb-amount small { color: var(--pt-accent); font-family: var(--pt-font-mono); font-size: .72em; font-weight: 800; letter-spacing: .055em; }
.pt-seat .pt-bb-amount { display: inline-flex; overflow: visible; text-overflow: clip; }
.pt-seat .pt-bb-amount > span { display: inline; overflow: visible; text-overflow: clip; }
.pt-seat-hero { z-index: 5; border-color: #388057; box-shadow: 0 0 0 2px rgba(53,208,127,.1), 0 7px 18px rgba(0,0,0,.3); }
.pt-seat-acting { z-index: 6; border-color: var(--pt-gold); box-shadow: 0 0 0 2px rgba(213,168,75,.16), 0 7px 18px rgba(0,0,0,.3); }
.pt-seat-folded { opacity: .42; filter: grayscale(.7); }
.pt-seat-1-of-2, .pt-seat-1-of-3, .pt-seat-1-of-4, .pt-seat-1-of-5, .pt-seat-1-of-6, .pt-seat-1-of-7, .pt-seat-1-of-8, .pt-seat-1-of-9 { left: 50%; top: 102%; }
.pt-seat-2-of-2 { left: 50%; top: -2%; }
.pt-seat-2-of-3, .pt-seat-2-of-4, .pt-seat-2-of-5, .pt-seat-2-of-6, .pt-seat-2-of-7, .pt-seat-2-of-8, .pt-seat-2-of-9 { left: 11%; top: 58%; }
.pt-seat-3-of-3 { left: 89%; top: 42%; }
.pt-seat-3-of-4, .pt-seat-3-of-5, .pt-seat-3-of-6, .pt-seat-3-of-7, .pt-seat-3-of-8, .pt-seat-3-of-9 { left: 14%; top: 12%; }
.pt-seat-4-of-4 { left: 86%; top: 12%; }
.pt-seat-4-of-5, .pt-seat-4-of-6, .pt-seat-4-of-7, .pt-seat-4-of-8, .pt-seat-4-of-9 { left: 50%; top: -2%; }
.pt-seat-5-of-5 { left: 89%; top: 58%; }
.pt-seat-5-of-6, .pt-seat-5-of-7, .pt-seat-5-of-8, .pt-seat-5-of-9 { left: 86%; top: 12%; }
.pt-seat-6-of-6 { left: 89%; top: 58%; }
.pt-seat-6-of-7, .pt-seat-6-of-8, .pt-seat-6-of-9 { left: 90%; top: 55%; }
.pt-seat-7-of-7, .pt-seat-7-of-8, .pt-seat-7-of-9 { left: 78%; top: 94%; }
.pt-seat-8-of-8, .pt-seat-8-of-9 { left: 22%; top: 94%; }
.pt-seat-9-of-9 { left: 7%; top: 55%; }
.pt-table-center { position: absolute; z-index: 2; left: 50%; bottom: calc(24px - 2%); transform: translateX(-50%); display: grid; justify-items: center; gap: 8px; }
.pt-board { position: relative; z-index: 2; }
.pt-hero-cards { position: relative; z-index: 2; display: flex; flex-direction: column; align-items: center; gap: 3px; }
.pt-hero-cards > span:first-child { color: var(--pt-accent); font-size: .48rem; font-weight: 760; line-height: 1; letter-spacing: .08em; }
.pt-card-row { display: inline-flex; gap: .2rem; align-items: center; }
.pt-card { width: 30px; height: 42px; display: inline-flex; flex-direction: column; justify-content: space-between; padding: 3px 4px; border: 1px solid #D5DAD3; border-radius: 4px; background: #EEF1EC; box-shadow: 0 6px 12px rgba(0,0,0,.24); color: #121713; font-family: var(--pt-font-mono); font-size: .69rem; font-weight: 780; line-height: 1; animation: pt-deal 220ms var(--pt-ease) both; animation-delay: var(--deal-delay); }
.pt-card span:last-child { align-self: flex-end; font-size: .72rem; }
.pt-card-red { color: #B43238; }
.pt-card-unknown { justify-content: center; align-items: center; color: #788079; border-style: dashed; background: #BFC6C0; }
.pt-card-back { justify-content: center; align-items: center; color: #35D07F; border-color: #426653; background: repeating-linear-gradient(45deg, #14231B, #14231B 3px, #1B3326 3px, #1B3326 6px); }
.pt-pot { position: relative; z-index: 3; display: flex; align-items: center; gap: .4rem; padding: .28rem .5rem; border: 1px solid rgba(213,168,75,.34); border-radius: 5px; background: rgba(7,15,10,.8); }
.pt-pot span { color: var(--pt-gold); font-size: .48rem; letter-spacing: .1em; }
.pt-pot strong { color: var(--pt-text); font-family: var(--pt-font-mono); font-size: .64rem; }
.pt-pot .pt-bb-amount > span { color: var(--pt-text); font-size: 1em; letter-spacing: normal; }
.pt-pot .pt-bb-amount small { color: var(--pt-gold); }
.pt-table-result .pt-bb-amount small { color: inherit; }
.pt-pot i { width: 14px; height: 6px; border-radius: 50%; background: var(--pt-gold); box-shadow: 0 -3px 0 #8D702F, 0 -6px 0 #D5A84B; animation: pt-chip-in 300ms 260ms var(--pt-ease) both; }
.pt-history-panel { width: 100%; overflow: hidden; border: 1px solid var(--pt-border); border-radius: var(--pt-radius); background: var(--pt-surface-soft); }
.pt-history-summary { display: flex; justify-content: space-between; gap: .75rem; padding: .58rem .75rem; border-bottom: 1px solid var(--pt-border); background: var(--pt-surface); }
.pt-history-summary strong { color: var(--pt-text); font-size: .66rem; }
.pt-history-summary span { color: var(--pt-muted); font-size: .6rem; text-align: right; }
.stMarkdown ol.pt-decision-history { display: block !important; width: 100%; max-width: 100%; box-sizing: border-box; overflow: visible; margin: 0 !important; padding: 0 !important; list-style: none !important; }
.stMarkdown ol.pt-decision-history > li.pt-action { position: relative; min-height: 76px; box-sizing: border-box; margin: 0; padding: .62rem .7rem; border-bottom: 1px solid rgba(51,70,59,.58); list-style: none !important; animation: pt-action-in 180ms var(--pt-ease) both; animation-delay: var(--action-delay); }
.stMarkdown ol.pt-decision-history > li.pt-action:last-child { border-bottom: 0; }
.stMarkdown ol.pt-decision-history > li.pt-action::before { content: ""; position: absolute; inset: 9px auto 9px 0; width: 2px; background: var(--pt-muted); }
.pt-history-copy { min-width: 0; display: grid; gap: .65rem; padding-left: .2rem; }
.pt-history-primary { display: grid; grid-template-columns: 28px 58px minmax(70px, 1fr) 72px 60px; column-gap: .45rem; align-items: center; }
.pt-history-sequence, .pt-history-size { color: var(--pt-muted); font-family: var(--pt-font-mono); font-size: .59rem; }
.pt-history-street { color: var(--pt-gold); font-size: .58rem; font-weight: 720; letter-spacing: .045em; text-transform: uppercase; }
.pt-history-actor { min-width: 0; display: flex; align-items: baseline; gap: .35rem; overflow: hidden; }
.pt-history-actor strong, .pt-history-actor small { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pt-history-actor strong { color: var(--pt-text); font-size: .66rem; }
.pt-history-actor small { color: var(--pt-muted); font-size: .52rem; }
.pt-history-decision { color: var(--pt-text); font-size: .64rem; }
.pt-history-size { color: var(--pt-text); text-align: right; }
.pt-history-context { min-width: 0; display: flex; flex-wrap: wrap; align-items: center; gap: .45rem 1rem; padding-left: calc(28px + .45rem); color: var(--pt-muted); font-family: var(--pt-font-mono); font-size: .52rem; }
.pt-history-context b { color: #718078; font-family: var(--pt-font-sans); font-size: .49rem; font-weight: 680; letter-spacing: .045em; text-transform: uppercase; }
.pt-history-note { width: 100%; white-space: normal; line-height: 1.45; }
.stMarkdown ol.pt-decision-history > li.pt-action-fold::before { background: var(--pt-negative); }
.stMarkdown ol.pt-decision-history > li.pt-action-call::before, .stMarkdown ol.pt-decision-history > li.pt-action-check::before { background: var(--pt-warning); }
.stMarkdown ol.pt-decision-history > li.pt-action-bet::before, .stMarkdown ol.pt-decision-history > li.pt-action-raise::before { background: var(--pt-accent); }
.pt-timeline-empty { min-height: 68px; display: grid; align-content: center; padding: .8rem 1rem; }
.pt-timeline-empty strong, .pt-timeline-empty small { display: block; }
.pt-timeline-empty strong { color: var(--pt-text); font-size: .72rem; }
.pt-timeline-empty small { color: var(--pt-muted); font-size: .66rem; margin-top: .16rem; }
.pt-equity { padding: .75rem 0; }
.pt-equity > div:first-child { display: flex; justify-content: space-between; align-items: baseline; }
.pt-equity span { color: var(--pt-muted); font-size: .68rem; }
.pt-equity strong { color: var(--pt-text); font-family: var(--pt-font-mono); font-size: 1.05rem; }
.pt-equity-track { position: relative; height: 7px; margin-top: .45rem; overflow: visible; border-radius: 2px; background: #1C2821; }
.pt-equity-track b { display: block; height: 100%; border-radius: 2px; background: var(--pt-accent); transform-origin: left; animation: pt-meter 380ms var(--pt-ease) both; }
.pt-equity-track i { position: absolute; top: -3px; width: 1px; height: 13px; background: var(--pt-gold); }
.pt-range-wrap { overflow-x: auto; padding-bottom: .25rem; }
.pt-range-grid { display: grid; grid-template-columns: repeat(13, minmax(28px, 1fr)); min-width: 520px; gap: 2px; }
.pt-range-cell { aspect-ratio: 1; display: grid; place-items: center; border: 1px solid #223027; border-radius: 2px; background: #0B110E; color: #65716A; font-family: var(--pt-font-mono); font-size: clamp(.44rem, .7vw, .61rem); }
.pt-range-value { color: #07110B; background: var(--pt-accent); border-color: var(--pt-accent); }
.pt-range-mixed { color: #1B1508; background: var(--pt-gold); border-color: var(--pt-gold); }
.pt-range-fold { color: #B56A6A; background: #291617; border-color: #54292D; }
@keyframes pt-deal { from { opacity: 0; transform: translateY(-8px) rotate(-2deg); } to { opacity: 1; transform: translateY(0) rotate(0); } }
@keyframes pt-chip-in { from { opacity: 0; transform: translate(20px, 14px) scale(.7); } to { opacity: 1; transform: translate(0,0) scale(1); } }
@keyframes pt-action-in { from { opacity: 0; transform: translateX(-5px); } to { opacity: 1; transform: translateX(0); } }
@keyframes pt-meter { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (max-width: 720px) {
  .pt-poker-stage figcaption { flex-wrap: wrap; }
  .pt-table-shell { min-height: 250px; padding: 2.2rem 2rem; }
  .pt-table-felt { width: min(100%, 440px); aspect-ratio: 1.45; }
  .pt-seat { min-width: 70px; padding: .3rem .36rem; }
  .pt-seat strong { font-size: .58rem; }
  .pt-card { width: 25px; height: 36px; font-size: .6rem; }
  .pt-table-center { gap: 7px; }
}
@media (max-width: 420px) {
  .pt-table-shell { min-height: 232px; padding: 2.15rem .7rem; }
  .pt-table-felt { width: 100%; aspect-ratio: 1.42; border-width: 6px; }
  .pt-seat { min-width: 64px; max-width: 82px; padding: .28rem .32rem; }
  .pt-card-row { gap: .12rem; }
  .pt-card { width: 23px; height: 33px; padding: 2px 3px; }
  .pt-table-center { bottom: calc(22px - 2%); gap: 6px; }
  .pt-hero-cards { gap: 2px; }
  .pt-pot { padding: .24rem .38rem; }
  .pt-history-summary { display: grid; }
  .pt-history-summary span { text-align: left; }
  .pt-history-primary { grid-template-columns: 46px minmax(54px, 1fr) 62px 54px; column-gap: .35rem; }
  .pt-history-sequence { display: none; }
  .stMarkdown ol.pt-decision-history > li.pt-action { padding-inline: .55rem; }
  .pt-history-context { padding-left: 0; gap: .35rem .65rem; font-size: .49rem; }
  .pt-history-context b { font-size: .46rem; }
  .pt-range-grid { min-width: 455px; }
}
@container (max-width: 560px) {
  .pt-table-shell { min-height: 250px; padding: 2.2rem 2rem; }
  .pt-table-felt { width: min(100%, 440px); aspect-ratio: 1.45; }
  .pt-seat { min-width: 70px; padding: .3rem .36rem; }
  .pt-seat strong { font-size: .58rem; }
  .pt-card { width: 25px; height: 36px; font-size: .6rem; }
  .pt-table-center { gap: 7px; }
}
@container (max-width: 390px) {
  .pt-table-shell { min-height: 232px; padding: 2.15rem .7rem; }
  .pt-table-felt { width: 100%; aspect-ratio: 1.42; border-width: 6px; }
  .pt-seat { min-width: 64px; max-width: 82px; padding: .28rem .32rem; }
  .pt-card-row { gap: .12rem; }
  .pt-card { width: 23px; height: 33px; padding: 2px 3px; }
  .pt-table-center { bottom: calc(22px - 2%); gap: 6px; }
  .pt-hero-cards { gap: 2px; }
  .pt-pot { padding: .24rem .38rem; }
}
</style>
"""


def inject_poker_visual_styles() -> None:
    """Inject poker-specific styles once per Streamlit rerun."""
    st.markdown(_POKER_CSS, unsafe_allow_html=True)
