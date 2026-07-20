from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from poker_tracker.db import PokerDatabase
from poker_tracker.models import Hand
from poker_tracker.study_math import mean_confidence_interval


AGGRESSIVE_ACTIONS = {"bet", "raise", "all-in"}
PASSIVE_ACTIONS = {"check", "call"}


@dataclass(frozen=True)
class SessionStats:
    """Basic manual stats, not HUD-grade poker statistics."""

    hand_count: int
    hands_with_result: int
    total_hero_bb: float
    average_hero_bb: float
    bb_per_100: float
    # 95% confidence interval on bb/100 (None with fewer than 2 recorded hands).
    # Session samples are small, so this is a reminder of variance, not proof of a winrate.
    bb_per_100_ci: tuple[float, float] | None = None
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
    # Only hands with a recorded result count toward result stats; treating a
    # missing result as 0 BB would deflate the averages.
    recorded = [hand.hero_bb_won for hand in hands if hand.hero_bb_won is not None]
    total = sum(recorded)
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

    if len(recorded) >= 2:
        _, ci_low, ci_high = mean_confidence_interval(recorded)
        bb_per_100_ci = (100 * ci_low, 100 * ci_high)
    else:
        bb_per_100_ci = None

    return SessionStats(
        hand_count=hand_count,
        hands_with_result=len(recorded),
        total_hero_bb=total,
        average_hero_bb=total / len(recorded) if recorded else 0,
        bb_per_100=100 * total / len(recorded) if recorded else 0,
        bb_per_100_ci=bb_per_100_ci,
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
