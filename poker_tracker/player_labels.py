"""Consistent player and position labels for completed-hand displays."""

from __future__ import annotations


def labels_match(left: str | None, right: str | None) -> bool:
    """Return whether two non-empty display labels are equivalent."""
    if not left or not right:
        return False
    return _normalize(left) == _normalize(right)


def actor_label(
    player_name: str | None,
    position: str | None,
    *,
    position_first: bool = False,
) -> str:
    """Combine a player name and position without repeating equivalent labels."""
    name = _clean(player_name)
    seat = _clean(position)
    if labels_match(name, seat):
        return name or seat
    parts = (seat, name) if position_first else (name, seat)
    return " ".join(part for part in parts if part)


def distinct_position(player_name: str | None, position: str | None) -> str:
    """Return a position only when it adds information beyond the player name."""
    seat = _clean(position)
    return "" if labels_match(player_name, seat) else seat


def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


def _normalize(value: str) -> str:
    return _clean(value).casefold()
