from __future__ import annotations

from collections import defaultdict

from poker_tracker.models import Action, Hand, HandPlayer, Session


STREET_LABELS = {
    "preflop": "Preflop",
    "flop": "Flop",
    "turn": "Turn",
    "river": "River",
    "showdown": "Showdown",
}


def format_hand_history(
    session: Session,
    hand: Hand,
    actions: list[Action],
    players: list[HandPlayer] | None = None,
) -> str:
    """Convert stored hand data into a readable post-session hand history."""
    lines = [
        f"Session: {session.date_played.isoformat()} {session.platform}".strip(),
        f"Hand #{hand.hand_number}",
        f"Game: {hand.game_type or 'Unknown'} {hand.blinds_antes}".strip(),
        f"Hero: {hand.hero_position or 'Unknown'}, {hand.hero_cards or 'unknown cards'}",
        f"Board: {hand.board_cards or 'none'}",
    ]
    if hand.pot_size is not None:
        lines.append(f"Final pot: {hand.pot_size:g}")
    if hand.result:
        lines.append(f"Outcome: {hand.result}")
    lines += [
        f"Result: {_format_bb_result(hand.hero_bb_won)}",
        f"Tags: {', '.join(hand.tags) if hand.tags else 'none'}",
    ]

    if players:
        lines.append("")
        lines.append("Players:")
        for player in players:
            lines.append(_format_player(player))

    grouped = _group_actions(actions)
    for street in STREET_LABELS:
        street_actions = grouped.get(street, [])
        if not street_actions:
            continue
        lines.append("")
        lines.append(f"{STREET_LABELS[street]}:")
        for action in street_actions:
            lines.append(_format_action(action))

    lines.append("")
    lines.append(f"Review status: {hand.review_status}")
    if hand.notes:
        lines.append(f"Notes: {hand.notes}")
    return "\n".join(lines)


def _group_actions(actions: list[Action]) -> dict[str, list[Action]]:
    grouped: dict[str, list[Action]] = defaultdict(list)
    for action in actions:
        grouped[action.street].append(action)
    for street_actions in grouped.values():
        street_actions.sort(key=lambda action: (action.action_index or 0, action.id or 0))
    return grouped


def _format_player(player: HandPlayer) -> str:
    hero_marker = " Hero" if player.is_hero else ""
    stack = "" if player.starting_stack is None else f", stack {player.starting_stack:g}"
    notes = "" if not player.notes else f", {player.notes}"
    return f"{player.player_name}{hero_marker}: {player.position or 'Unknown'}{stack}{notes}"


def _format_action(action: Action) -> str:
    position = f"{action.position} " if action.position else ""
    amount = "" if action.amount is None else f" {action.amount:g}"
    notes = "" if not action.notes else f" ({action.notes})"
    return f"{position}{action.player_name} {action.action_type}{amount}{notes}"


def _format_bb_result(value: float | None) -> str:
    if value is None:
        return "unknown"
    prefix = "+" if value > 0 else ""
    return f"{prefix}{value:g} BB"
