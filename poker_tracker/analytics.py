from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from poker_tracker.db import PokerDatabase
from poker_tracker.models import Hand


AGGRESSIVE_ACTIONS = {"bet", "raise", "all-in"}
PASSIVE_ACTIONS = {"check", "call"}


@dataclass(frozen=True)
class SessionStats:
    """Basic manual stats, not HUD-grade poker statistics."""

    hand_count: int
    total_hero_bb: float
    average_hero_bb: float
    biggest_winning_hands: list[Hand] = field(default_factory=list)
    biggest_losing_hands: list[Hand] = field(default_factory=list)
    hands_by_tag: dict[str, int] = field(default_factory=dict)
    hands_by_review_status: dict[str, int] = field(default_factory=dict)
    action_counts_by_type: dict[str, int] = field(default_factory=dict)
    aggression_count: int = 0
    passive_count: int = 0


def compute_session_stats(db: PokerDatabase, session_id: int) -> SessionStats:
    """Compute basic/manual session stats from stored hands and actions."""
    hands = db.fetch_hands_by_session(session_id)
    total = sum(hand.hero_bb_won or 0 for hand in hands)
    hand_count = len(hands)
    tag_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()

    for hand in hands:
        tag_counts.update(hand.tags)
        status_counts.update([hand.review_status])
        if hand.id is None:
            continue
        action_counts.update(action.action_type for action in db.fetch_actions_by_hand(hand.id))

    return SessionStats(
        hand_count=hand_count,
        total_hero_bb=total,
        average_hero_bb=total / hand_count if hand_count else 0,
        biggest_winning_hands=_top_hands(hands, reverse=True),
        biggest_losing_hands=_top_hands(hands, reverse=False),
        hands_by_tag=dict(tag_counts),
        hands_by_review_status=dict(status_counts),
        action_counts_by_type=dict(action_counts),
        aggression_count=sum(action_counts[action] for action in AGGRESSIVE_ACTIONS),
        passive_count=sum(action_counts[action] for action in PASSIVE_ACTIONS),
    )


def _top_hands(hands: list[Hand], *, reverse: bool) -> list[Hand]:
    eligible = [hand for hand in hands if hand.hero_bb_won is not None]
    sorted_hands = sorted(eligible, key=lambda hand: hand.hero_bb_won or 0, reverse=reverse)
    if reverse:
        return [hand for hand in sorted_hands if (hand.hero_bb_won or 0) > 0][:3]
    return [hand for hand in sorted_hands if (hand.hero_bb_won or 0) < 0][:3]
