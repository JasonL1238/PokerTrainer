"""Pure helpers for memorable session naming and fast library filtering."""

from __future__ import annotations

from collections.abc import Container, Iterable, Mapping
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


def sessions_on_date(sessions: Iterable[Session], played_on: date) -> list[Session]:
    """Return sessions whose date_played matches the selected calendar day."""

    return [session for session in sessions if session.date_played == played_on]


def session_dates(sessions: Iterable[Session]) -> set[date]:
    """Return the distinct played dates present in a session list."""

    return {session.date_played for session in sessions}


# The evidence flags a hand can be filtered on. They are not mutually exclusive
# facts about a hand -- a hand can carry an open issue AND stale analysis -- so
# this is a single-choice narrowing control, not a partition, and "clean" means
# "neither of the two above" rather than "verified".
HAND_FLAG_FILTERS = ("all", "open_issue", "stale", "clean")


def hand_search_text(
    hand: Hand,
    session: Session,
    *,
    issue_text: str = "",
    has_open_issue: bool = False,
    has_stale_analysis: bool = False,
) -> str:
    """Build a broad text index for fast hand-library search.

    The three keyword arguments carry what is true ABOUT a hand rather than what
    is stored ON it. An open debugging issue lives in its own table, so a hand
    the operator flagged could not be found by searching for what they wrote on
    the flag -- the one search most likely to be run against an inbox. The
    literal words below are indexed too so "stale" and "issue" find the rows the
    badges show those words on; a search that cannot find what the list displays
    is the same defect in a smaller font.
    """

    return " ".join(
        (
            session.name,
            session.date_played.isoformat(),
            # So "august" and "saturday" find a session the way the calendar
            # browser names it, matching `session_search_text`.
            session.date_played.strftime("%A %B"),
            session.platform,
            session.stakes,
            str(hand.hand_number),
            hand.hero_cards,
            hand.board_cards,
            hand.hero_position,
            hand.review_status,
            hand.source_type,
            hand.completion_status,
            hand.study_inclusion,
            " ".join(hand.tags),
            hand.notes,
            issue_text,
            "open issue" if has_open_issue else "",
            "stale analysis" if has_stale_analysis else "",
        )
    ).casefold()


def filter_hands(
    hands: Iterable[Hand],
    sessions_by_id: dict[int, Session],
    *,
    query: str = "",
    review_status: str = "all",
    result_filter: str = "all",
    source_filter: str = "all",
    flag_filter: str = "all",
    issue_text_by_hand: Mapping[int, str] | None = None,
    issue_hand_ids: Container[int] = frozenset(),
    stale_hand_ids: Container[int] = frozenset(),
) -> list[Hand]:
    """Apply the library's search and one-tap filters without UI coupling.

    Issue and staleness membership are passed in rather than queried, so the
    caller pays one bounded lookup for the whole list instead of two per row and
    this stays a pure function.
    """

    terms = query.casefold().split()
    matches: list[Hand] = []
    for hand in hands:
        session = sessions_by_id.get(hand.session_id)
        if session is None:
            continue
        if review_status != "all" and hand.review_status != review_status:
            continue
        if source_filter != "all" and hand.source_type != source_filter:
            continue
        if result_filter == "wins" and not (hand.hero_bb_won is not None and hand.hero_bb_won > 0):
            continue
        if result_filter == "losses" and not (
            hand.hero_bb_won is not None and hand.hero_bb_won < 0
        ):
            continue
        if result_filter == "unknown" and hand.hero_bb_won is not None:
            continue
        has_issue = hand.id is not None and hand.id in issue_hand_ids
        is_stale = hand.id is not None and hand.id in stale_hand_ids
        if flag_filter == "open_issue" and not has_issue:
            continue
        if flag_filter == "stale" and not is_stale:
            continue
        if flag_filter == "clean" and (has_issue or is_stale):
            continue
        if terms:
            haystack = hand_search_text(
                hand,
                session,
                issue_text=(
                    ""
                    if issue_text_by_hand is None or hand.id is None
                    else issue_text_by_hand.get(hand.id, "")
                ),
                has_open_issue=has_issue,
                has_stale_analysis=is_stale,
            )
            if not all(term in haystack for term in terms):
                continue
        matches.append(hand)
    return matches
