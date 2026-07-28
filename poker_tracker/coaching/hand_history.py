from __future__ import annotations

from collections import defaultdict

from poker_tracker.math.accounting import HandLedger
from poker_tracker.persistence.models import Action, Hand, HandPlayer, Session
from poker_tracker.player_labels import actor_label, labels_match

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
    *,
    ledger: HandLedger | None = None,
    accounting_issues: list[str] | None = None,
    accounting_authoritative: bool = False,
) -> str:
    """Convert stored hand data into a readable post-session hand history."""
    lines = [
        f"Session: {session.date_played.isoformat()} {session.platform}".strip(),
        f"Hand #{hand.hand_number}",
        f"Game: {hand.game_type or 'Unknown'} {hand.blinds_antes}".strip(),
        f"Hero: {hand.hero_position or 'Unknown'}, {hand.hero_cards or 'unknown cards'}",
        f"Board: {hand.board_cards or 'none'}",
    ]
    if accounting_authoritative and ledger is not None:
        lines.extend(
            [
                f"Final pot: {ledger.gross_pot:g} BB (reconciled)",
                f"Rake: {ledger.rake:g} BB",
                f"Net pot: {ledger.net_pot:g} BB",
            ]
        )
    elif hand.pot_size is not None:
        lines.append(f"Final pot: {hand.pot_size:g} BB (observed)")
    if hand.result:
        lines.append(f"Outcome: {hand.result}")
    result_bb = hand.hero_bb_won
    if accounting_authoritative and ledger is not None and players:
        hero = next((player for player in players if player.is_hero), None)
        if hero is not None:
            result_bb = ledger.net_results.get(hero.player_key, result_bb)
    lines += [
        f"Result: {_format_bb_result(result_bb)}",
        f"Tags: {', '.join(hand.tags) if hand.tags else 'none'}",
    ]
    if ledger is not None:
        status = (
            "reconciled"
            if accounting_authoritative
            else "unsettled / not authoritative"
        )
        lines.append(
            f"Accounting: {status}; balanced={ledger.is_balanced}; legal={ledger.is_legal}"
        )
    if accounting_issues:
        lines.append("Accounting issues: " + " | ".join(accounting_issues))

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
    hero_marker = " (Hero)" if player.is_hero and not labels_match(player.player_name, "Hero") else ""
    name = actor_label(player.player_name, None) or "Unknown player"
    details: list[str] = []
    if not labels_match(player.player_name, player.position):
        details.append(player.position or "Unknown position")
    if player.seat_index is not None:
        details.append(f"seat {player.seat_index}")
    if player.starting_stack is not None:
        details.append(f"stack {player.starting_stack:g}")
    if player.notes:
        details.append(player.notes)
    detail_text = f": {', '.join(details)}" if details else ""
    return f"{name}{hero_marker}{detail_text}"


def _format_action(action: Action) -> str:
    amount = "" if action.amount is None else f" {action.amount:g}"
    if action.amount is not None and action.action_type in {
        "ante",
        "post_blind",
        "call",
        "bet",
        "raise",
        "all-in",
    }:
        amount += (
            " additional"
            if action.amount_semantics == "incremental"
            else " total-this-street"
            if action.amount_semantics == "raise_to"
            else " [amount meaning unknown]"
        )
    context: list[str] = []
    if action.pot_before is not None:
        context.append(f"observed pot before {action.pot_before:g}")
    if action.stack_before is not None:
        context.append(f"observed stack before {action.stack_before:g}")
    if action.forced_bet_type:
        context.append(action.forced_bet_type.replace("_", " "))
    if action.is_live_post is False:
        context.append("dead post")
    notes = "" if not action.notes else f" ({action.notes})"
    context_text = f" [{'; '.join(context)}]" if context else ""
    actor = actor_label(action.player_name, action.position, position_first=True)
    return f"{actor} {action.action_type}{amount}{context_text}{notes}"


def _format_bb_result(value: float | None) -> str:
    if value is None:
        return "unknown"
    prefix = "+" if value > 0 else ""
    return f"{prefix}{value:g} BB"
