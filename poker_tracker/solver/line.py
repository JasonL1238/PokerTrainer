"""Whether a recorded heads-up postflop line is a betting sequence at all.

Everything downstream of ``prepare_solver_spot`` treats ``SolverSpot.recorded_line``
as a faithful transcript of one heads-up subtree: ``texassolver.parse_strategy_result``
walks it branch by branch and reads Hero's frequencies out of whatever node the
walk lands on. That walk is only sound if the line really is the sequence it
looks like -- the right player acting at each turn, every betting round finishing
before the next street starts, and the whole thing ending somewhere a hand can
actually end.

Nothing checked any of that. The chip ledger, which is the only gate the solver
had, is a *summing* engine: it verifies that money in equals money out and that
no individual action is illegal in isolation. It has no model of a betting round,
so it settles a line that stops on an unanswered bet (by refunding the bet) and a
line with an action missing from the middle (by summing what is there), reporting
both as ``is_settled / is_balanced / is_legal`` with no warnings. Two distinct
failures followed:

* A line whose last action is an unanswered wager is, by definition, a hand still
  in progress. Offering to analyse it is offering analysis of a decision nobody
  has made yet, which this product does not do.
* A line with a gap in it silently relocates the solve. With Hero in position and
  Hero's flop response missing, the walk maps Villain's *turn* bet onto Hero's
  *flop* node, descends a bet branch Hero never took, and reports the node where
  *Villain* acts as Hero's frequencies -- correct-looking numbers about a decision
  no one in the hand ever faced, with no warning beyond a bet-size approximation.

Both are the same missing predicate, so it lives here once and is applied at both
ends: ``eligibility`` refuses to build a spot from a line that fails it, and the
walk refuses to read a node it reached by crossing a street boundary.

The rules are expressed over ``RecordedSolverAction`` alone -- ``player_key``,
``street``, ``action_type`` and the incremental ``amount`` the ledger normalised
-- so a spot deserialised from a stored ``solver_runs.spot`` payload is judged by
exactly the same predicate as a freshly prepared one.
"""

from __future__ import annotations

from poker_tracker.solver.models import RecordedSolverAction

STREET_ORDER = ("flop", "turn", "river")
# Postflop decisions the solve tree has branches for. `show` and `win` are valid
# `ActionType` values that reach postflop street rows, and they are not decisions:
# mapping one would either refuse loudly or, when it is Hero's, read a node out
# under a label that is not a choice.
_DECISIONS = frozenset({"fold", "check", "call", "bet", "raise", "all-in"})
# Chip comparisons here come from ledger snapshot amounts that have already been
# round-tripped through float. This is representation noise only.
_EPSILON = 1e-9


def _label(action: RecordedSolverAction) -> str:
    if action.amount is None:
        return action.action_type
    return f"{action.action_type} {action.amount:g} BB"


class _Round:
    """One street's heads-up betting round, as far as the line describes it."""

    def __init__(self, street: str, oop_key: str, ip_key: str) -> None:
        self.street = street
        self.contrib = {oop_key: 0.0, ip_key: 0.0}
        self.to_act = oop_key
        self.checked_by: set[str] = set()
        self.closed = False
        self.folded = False
        self.all_in = False
        self.last: RecordedSolverAction | None = None


def recorded_line_defect(
    recorded_line: list[RecordedSolverAction],
    *,
    oop_key: str,
    ip_key: str,
    start_street: str,
) -> str | None:
    """Name the first way this line is not a playable heads-up sequence.

    Returns ``None`` when the line is one: each street's round is legal and
    finished, the streets it covers are consecutive from ``start_street``, and it
    ends where a hand can end (a fold, an all-in that closed the betting, or a
    completed final round). Returns a single operator-facing sentence otherwise.

    The check is deliberately about the SHAPE of the sequence and not about chip
    legality -- minimum raise sizes, stack coverage and pot arithmetic all belong
    to the ledger, which already does them. What the ledger cannot do, and this
    can, is notice that nobody answered the last bet.
    """

    if start_street not in STREET_ORDER:
        return f"'{start_street}' is not a postflop street."
    if not recorded_line:
        return "The heads-up subtree records no postflop action."

    order = list(STREET_ORDER[STREET_ORDER.index(start_street) :])
    current = _Round(order[0], oop_key, ip_key)
    for action in recorded_line:
        kind = action.action_type.strip().lower()
        if kind not in _DECISIONS:
            return (
                f"{action.player_name}'s recorded '{action.action_type}' on the "
                f"{action.street} is not a postflop betting decision."
            )
        if action.player_key not in current.contrib:
            return (
                f"{action.player_name} is not one of the two players in the "
                "heads-up subtree."
            )
        if action.street != current.street:
            defect = _street_change(current, action, order)
            if defect is not None:
                return defect
            current = _Round(action.street, oop_key, ip_key)
        elif current.closed:
            return (
                f"{action.player_name}'s {_label(action)} follows a {current.street} "
                "betting round that had already closed."
            )
        if action.player_key != current.to_act:
            return (
                f"{action.player_name} acted out of turn on the {current.street}: "
                "the other player was still to act."
            )
        defect = _apply(current, action, kind)
        if defect is not None:
            return defect
        current.last = action

    if not current.closed:
        last = current.last
        detail = f"the last recorded action is {_label(last)}" if last else "it is empty"
        return (
            f"The {current.street} betting round never finished -- {detail}, and "
            "nobody answered it. A hand that is still in progress is not a "
            "completed spot this product analyses."
        )
    return None


def unplayed_streets(
    recorded_line: list[RecordedSolverAction], board_cards: str
) -> tuple[str, ...]:
    """Streets the saved board reached that the recorded line never plays.

    A line can be perfectly well formed and still stop early: a flop that checks
    through on a board saved with five cards means the turn and river are simply
    absent from the record. The node being solved is still a decision Hero made,
    so this is a coverage limitation rather than a defect -- but it is one an
    operator has to be told about, because the spot otherwise presents as a
    complete hand.

    A fold or an all-in ends the betting legitimately, so neither leaves any
    street unplayed.
    """

    if not recorded_line:
        return ()
    if any(
        action.action_type.strip().lower() in {"fold", "all-in"}
        for action in recorded_line
    ):
        return ()
    board_size = len(board_cards.split())
    if board_size < 3:
        return ()
    dealt = STREET_ORDER[: min(board_size, 5) - 2]
    last_played = recorded_line[-1].street
    if last_played not in dealt:
        return ()
    return tuple(dealt[dealt.index(last_played) + 1 :])


def _street_change(
    current: _Round, action: RecordedSolverAction, order: list[str]
) -> str | None:
    if not current.closed:
        return (
            f"The {current.street} betting round never closed, yet the line "
            f"continues on the {action.street}. An action is missing between them."
        )
    if current.folded:
        return (
            f"The line continues on the {action.street} after a fold ended the "
            f"hand on the {current.street}."
        )
    if current.all_in:
        return (
            f"The line records {action.street} betting after a player was already "
            f"all-in on the {current.street}."
        )
    if action.street not in order:
        return (
            f"The line records a {action.street} action, which does not follow the "
            f"{current.street} in this subtree."
        )
    expected = order[order.index(current.street) + 1 :]
    if not expected or action.street != expected[0]:
        return (
            f"The line jumps from the {current.street} to the {action.street}; the "
            f"{expected[0] if expected else 'following'} betting round is missing."
        )
    return None


def _apply(current: _Round, action: RecordedSolverAction, kind: str) -> str | None:
    actor = action.player_key
    other = next(key for key in current.contrib if key != actor)
    amount = float(action.amount or 0.0)
    high = max(current.contrib.values())
    to_call = high - current.contrib[actor]

    if kind == "check":
        if to_call > _EPSILON:
            return (
                f"{action.player_name} is recorded as checking the {current.street} "
                f"while facing {to_call:g} BB."
            )
        current.checked_by.add(actor)
        if other in current.checked_by:
            current.closed = True
        else:
            current.to_act = other
        return None
    if kind == "fold":
        current.closed = True
        current.folded = True
        return None
    if kind == "call":
        if to_call <= _EPSILON:
            return (
                f"{action.player_name} is recorded as calling on the "
                f"{current.street} with no wager to call."
            )
        current.contrib[actor] += amount
        current.closed = True
        return None
    if kind == "bet":
        if to_call > _EPSILON:
            return (
                f"{action.player_name} is recorded as betting the {current.street} "
                f"while facing {to_call:g} BB; that is a raise, not a bet."
            )
        if amount <= _EPSILON:
            return (
                f"{action.player_name}'s recorded {current.street} bet carries no "
                "amount, so the line cannot be followed."
            )
        current.contrib[actor] += amount
        current.checked_by.clear()
        current.to_act = other
        return None
    if kind == "raise":
        if to_call <= _EPSILON:
            return (
                f"{action.player_name} is recorded as raising on the "
                f"{current.street} with no wager to raise."
            )
        if amount <= to_call + _EPSILON:
            return (
                f"{action.player_name}'s recorded {current.street} raise of "
                f"{amount:g} BB does not exceed the {to_call:g} BB it faces."
            )
        current.contrib[actor] += amount
        current.to_act = other
        return None
    # all-in. Whether it reopens the action or merely settles it is decided by
    # the chips it puts across the line, not by its name: an all-in for less than
    # the wager it faces closes the round exactly like a call.
    current.contrib[actor] += amount
    current.all_in = True
    if current.contrib[actor] > high + _EPSILON:
        current.checked_by.clear()
        current.to_act = other
        return None
    current.closed = True
    return None
