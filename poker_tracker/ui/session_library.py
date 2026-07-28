"""Pure helpers for memorable session naming and fast library filtering."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from poker_tracker.persistence.models import Hand, Session


def date_session_name(
    played_on: date,
    existing_sessions: Iterable[Session] = (),
) -> str:
    """Return a memorable, collision-safe default name anchored to the played date."""

    base = f"{played_on.strftime('%A, %B')} {played_on.day}, {played_on.year}"
    names = {session.name.casefold() for session in existing_sessions}
    if base.casefold() not in names:
        return base

    suffix = 2
    while f"{base} · Session {suffix}".casefold() in names:
        suffix += 1
    return f"{base} · Session {suffix}"


def session_search_text(session: Session) -> str:
    """Build the user-visible fields searched by the session library."""

    return " ".join(
        (
            session.name,
            session.date_played.isoformat(),
            session.date_played.strftime("%A %B"),
            session.platform,
            session.stakes,
            session.notes,
        )
    ).casefold()


def filter_sessions(sessions: Iterable[Session], query: str) -> list[Session]:
    """Filter sessions using names, dates, platform, stakes, and notes."""

    terms = query.casefold().split()
    if not terms:
        return list(sessions)
    return [
        session
        for session in sessions
        if all(term in session_search_text(session) for term in terms)
    ]


def hand_search_text(hand: Hand, session: Session) -> str:
    """Build a broad text index for fast hand-library search."""

    return " ".join(
        (
            session.name,
            session.date_played.isoformat(),
            session.platform,
            session.stakes,
            str(hand.hand_number),
            hand.hero_cards,
            hand.board_cards,
            hand.hero_position,
            hand.review_status,
            hand.source_type,
            " ".join(hand.tags),
            hand.notes,
        )
    ).casefold()


def filter_hands(
    hands: Iterable[Hand],
    sessions_by_id: dict[int, Session],
    *,
    query: str = "",
    review_status: str = "all",
    result_filter: str = "all",
) -> list[Hand]:
    """Apply the library's search and one-tap filters without UI coupling."""

    terms = query.casefold().split()
    matches: list[Hand] = []
    for hand in hands:
        session = sessions_by_id.get(hand.session_id)
        if session is None:
            continue
        if review_status != "all" and hand.review_status != review_status:
            continue
        if result_filter == "wins" and not (hand.hero_bb_won is not None and hand.hero_bb_won > 0):
            continue
        if result_filter == "losses" and not (
            hand.hero_bb_won is not None and hand.hero_bb_won < 0
        ):
            continue
        if result_filter == "unknown" and hand.hero_bb_won is not None:
            continue
        haystack = hand_search_text(hand, session)
        if terms and not all(term in haystack for term in terms):
            continue
        matches.append(hand)
    return matches
