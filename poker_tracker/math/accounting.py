"""Authoritative chip accounting for completed poker hands.

The reducer in this module deliberately works with normalized, incremental
commitments.  A raise of 8 BB means eight additional chips entered the pot;
importers that receive "raise to" amounts must normalize them first.

The ledger does not evaluate cards or choose winners.  Settlement is explicit:
callers provide the ordered winner(s) for each generated pot layer.  That keeps
chip conservation independently testable and prevents incomplete histories
from silently inventing outcomes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Literal, Protocol

LedgerStreet = Literal["preflop", "flop", "turn", "river", "showdown"]
LedgerActionKind = Literal[
    "ante",
    "post_blind",
    "bet",
    "call",
    "raise",
    "all-in",
    "fold",
    "check",
    "show",
    "win",
]

_COMMITMENT_KINDS = {"ante", "post_blind", "bet", "call", "raise", "all-in"}
_BETTING_COMMITMENT_KINDS = _COMMITMENT_KINDS - {"ante"}
_NON_COMMITMENT_KINDS = {"fold", "check", "show", "win"}
_FLOP_STREETS = {"flop", "turn", "river", "showdown"}
_ZERO = Decimal("0")
# The coarsest denomination a chopped pot may be divided at. A whole chip is the
# indivisible unit a real table deals in; anything above it is a claim about the
# room that the hand's own action line cannot demonstrate, and it was exactly
# that claim -- taken verbatim from a declared field -- that let "Chip unit"
# redirect a chop. See _split_granularity.
_MAX_SPLIT_QUANTUM = Decimal("1")


class LedgerError(ValueError):
    """Raised when a completed-hand ledger is structurally impossible."""


class PlayerRecord(Protocol):
    player_key: str
    player_name: str
    starting_stack: float | None
    seat_index: int | None


class ActionRecord(Protocol):
    player_key: str | None
    player_name: str
    street: str
    action_type: str
    amount: float | None
    amount_semantics: str
    is_live_post: bool | None
    pot_before: float | None


@dataclass(frozen=True)
class LedgerPlayer:
    name: str
    starting_stack: float
    seat: int | None = None
    key: str | None = None


@dataclass(frozen=True)
class LedgerAction:
    player: str
    street: LedgerStreet
    kind: LedgerActionKind
    amount: float = 0
    is_live_post: bool = True


@dataclass(frozen=True)
class RakePolicy:
    """Rake assumptions in the same chip unit as the actions.

    Rake is rounded down to ``rounding_unit``.  Rooms differ in their exact
    rounding and eligibility rules, so callers must store the policy used with
    any derived result instead of treating this default as universal.
    """

    rate: float = 0
    cap: float | None = None
    rounding_unit: float = 0.01
    no_flop_no_drop: bool = False


@dataclass(frozen=True)
class ActionSnapshot:
    index: int
    player: str
    street: LedgerStreet
    kind: LedgerActionKind
    amount: float
    pot_before: float
    pot_after: float
    stack_before: float
    stack_after: float
    to_call_before: float
    call_increment: float
    street_contribution_after: float
    hand_contribution_after: float
    effective_stack_before: float
    effective_stack_range_before: tuple[float, float]
    spr_before: float | None
    spr_range_before: tuple[float, float] | None
    active_players: tuple[str, ...]


@dataclass(frozen=True)
class PotLayer:
    index: int
    amount: float
    net_amount: float
    rake: float
    contributors: tuple[str, ...]
    eligible_players: tuple[str, ...]
    winners: tuple[str, ...] = ()


@dataclass(frozen=True)
class HandLedger:
    contributions: dict[str, float]
    refunds: dict[str, float]
    payouts: dict[str, float]
    net_results: dict[str, float]
    gross_pot: float
    rake: float
    net_pot: float
    pots: tuple[PotLayer, ...]
    snapshots: tuple[ActionSnapshot, ...]
    folded_players: tuple[str, ...]
    warnings: tuple[str, ...]
    legality_issues: tuple[str, ...]
    is_settled: bool
    is_balanced: bool
    is_legal: bool


def build_hand_ledger(
    players: Sequence[LedgerPlayer],
    actions: Sequence[LedgerAction],
    winners: Mapping[int, Sequence[str]] | None = None,
    *,
    rake: RakePolicy | None = None,
    odd_chip_order: Sequence[str] = (),
    dead_money: float = 0,
    flop_seen: bool | None = None,
) -> HandLedger:
    """Reduce normalized completed-hand actions into pots and player results.

    ``winners`` maps each generated pot index to one or more ordered winners.
    Pot 0 is the main pot; later indexes are side pots.  Omitting winners
    returns a useful but explicitly unsettled ledger. ``flop_seen`` is an
    optional completed-hand fact for histories where a board ran out without
    any postflop action (for example, a preflop all-in). When omitted, the
    ledger preserves the historic behavior of inferring it from action streets.
    """

    player_order, starting = _validate_players(players)
    dead = _decimal(dead_money, "dead_money")
    if dead < 0:
        raise LedgerError("dead_money must not be negative.")
    policy = rake or RakePolicy()
    rate, cap, unit = _validate_rake(policy)
    winner_map = {index: tuple(names) for index, names in (winners or {}).items()}
    odd_order = _validate_odd_chip_order(odd_chip_order, starting)

    contributions = {name: _ZERO for name in player_order}
    # Split the same chips two ways. ``live_contributions`` buys a place in a pot
    # layer and is what an uncalled bet is measured against; ``dead_contributions``
    # (antes, dead blinds) is owed to the table and joins the main pot whole.
    # Measuring refunds against the total instead hands back an unmatched ante as
    # though it were an overbet, and removes it from the pot entirely.
    live_contributions = {name: _ZERO for name in player_order}
    dead_contributions = {name: _ZERO for name in player_order}
    street_contributions = {name: _ZERO for name in player_order}
    current_street: LedgerStreet | None = None
    folded: set[str] = set()
    all_in: set[str] = set()
    snapshots: list[ActionSnapshot] = []
    warnings: list[str] = []
    legality_issues: list[str] = []
    street_order = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}
    last_street_rank = -1
    last_full_raise = _ZERO
    last_wager_faced: dict[str, Decimal] = {}

    for index, action in enumerate(actions):
        if action.player not in starting:
            raise LedgerError(f"Action {index + 1} references unknown player {action.player!r}.")
        if action.player in folded:
            raise LedgerError(f"Player {action.player!r} acts after folding.")
        if action.player in all_in and action.kind not in {"show", "win"}:
            legality_issues.append(
                f"Action {index + 1}: player {action.player!r} acts after being all-in."
            )
        if action.kind not in _COMMITMENT_KINDS | _NON_COMMITMENT_KINDS:
            raise LedgerError(f"Unsupported action kind {action.kind!r}.")
        if action.street not in street_order:
            raise LedgerError(f"Action {index + 1} has unsupported street {action.street!r}.")
        amount = _decimal(action.amount, f"action {index + 1} amount")
        if amount < 0:
            raise LedgerError("Action amounts must not be negative.")
        if action.kind in _COMMITMENT_KINDS and amount <= 0:
            raise LedgerError(f"{action.kind} requires a positive incremental amount.")
        if action.kind in _NON_COMMITMENT_KINDS and amount != 0:
            raise LedgerError(f"{action.kind} cannot commit chips.")

        if action.street != current_street:
            rank = street_order[action.street]
            if rank < last_street_rank:
                legality_issues.append(
                    f"Action {index + 1}: street order moves backward to {action.street}."
                )
            last_street_rank = max(last_street_rank, rank)
            current_street = action.street
            street_contributions = {name: _ZERO for name in player_order}
            last_full_raise = _ZERO
            last_wager_faced = {}

        committed_before = contributions[action.player]
        stack_before = starting[action.player] - committed_before
        if amount > stack_before:
            raise LedgerError(
                f"Player {action.player!r} commits {amount} with only {stack_before} remaining."
            )

        active_before = tuple(name for name in player_order if name not in folded)
        actionable_before = tuple(
            name for name in active_before if name not in all_in
        )
        opponent_effective = [
            min(stack_before, starting[name] - contributions[name])
            for name in actionable_before
            if name != action.player
        ]
        if opponent_effective:
            effective_low = min(opponent_effective)
            effective_high = max(opponent_effective)
        else:
            effective_low = effective_high = stack_before
        effective_before = effective_low
        betting_max = max(street_contributions.values(), default=_ZERO)
        player_street_before = street_contributions[action.player]
        to_call = max(_ZERO, betting_max - player_street_before)
        pot_before = dead + sum(contributions.values(), _ZERO)
        new_total = player_street_before + amount
        aggressive_increment = (
            max(_ZERO, new_total - betting_max)
            if action.kind in {"raise", "all-in"}
            else _ZERO
        )

        action_label = f"Action {index + 1}"
        if action.kind == "check" and to_call > 0:
            legality_issues.append(f"{action_label}: check while facing {to_call}.")
        if action.kind == "call":
            if to_call <= 0:
                legality_issues.append(f"{action_label}: call with nothing to call.")
            elif amount != to_call and not (amount == stack_before and amount < to_call):
                legality_issues.append(
                    f"{action_label}: call commits {amount}, but the amount to call is {to_call}."
                )
        if action.kind == "bet" and betting_max > 0:
            legality_issues.append(f"{action_label}: bet used while facing an existing wager.")
        if action.kind == "raise":
            raise_size = new_total - betting_max
            if to_call <= 0 or raise_size <= 0:
                legality_issues.append(f"{action_label}: raise does not increase the wager.")
            elif last_full_raise > 0 and raise_size < last_full_raise:
                legality_issues.append(
                    f"{action_label}: raise size {raise_size} is below the minimum "
                    f"full raise {last_full_raise}."
                )
        if (
            aggressive_increment > 0
            and action.player in last_wager_faced
            and last_full_raise > 0
            and betting_max - last_wager_faced[action.player] < last_full_raise
        ):
            legality_issues.append(
                f"{action_label}: betting was not reopened after a short all-in raise."
            )
        if action.kind == "all-in" and amount != stack_before:
            legality_issues.append(
                f"{action_label}: all-in commits {amount}, but {stack_before} remains."
            )

        if action.kind in _COMMITMENT_KINDS:
            contributions[action.player] += amount
            is_live_bet = action.kind in _BETTING_COMMITMENT_KINDS and not (
                action.kind == "post_blind" and not action.is_live_post
            )
            # Live money buys a place in a pot layer and can come back as an
            # uncalled bet. Dead money -- antes and dead blinds -- is owed to the
            # table: it joins the main pot whole and is never returnable.
            if is_live_bet:
                live_contributions[action.player] += amount
            else:
                dead_contributions[action.player] += amount
            if is_live_bet:
                street_contributions[action.player] += amount
                new_max = max(street_contributions.values(), default=_ZERO)
                if action.kind == "post_blind" and new_max > betting_max:
                    last_full_raise = max(last_full_raise, new_max)
                elif action.kind == "bet" and new_max > betting_max:
                    last_full_raise = new_max
                elif action.kind == "raise" and new_max > betting_max:
                    raise_size = new_max - betting_max
                    if raise_size >= last_full_raise:
                        last_full_raise = raise_size
                elif action.kind == "all-in" and new_max > betting_max:
                    raise_size = new_max - betting_max
                    if last_full_raise == 0 or raise_size >= last_full_raise:
                        last_full_raise = raise_size
        if action.kind not in {"ante", "post_blind", "show", "win"}:
            last_wager_faced[action.player] = max(
                street_contributions.values(), default=_ZERO
            )
        if action.kind == "fold":
            folded.add(action.player)
        if contributions[action.player] == starting[action.player]:
            all_in.add(action.player)

        pot_after = dead + sum(contributions.values(), _ZERO)
        call_increment = min(amount, to_call) if action.kind in {"call", "all-in"} else _ZERO
        snapshots.append(
            ActionSnapshot(
                index=index,
                player=action.player,
                street=action.street,
                kind=action.kind,
                amount=_float(amount),
                pot_before=_float(pot_before),
                pot_after=_float(pot_after),
                stack_before=_float(stack_before),
                stack_after=_float(starting[action.player] - contributions[action.player]),
                to_call_before=_float(to_call),
                call_increment=_float(call_increment),
                street_contribution_after=_float(street_contributions[action.player]),
                hand_contribution_after=_float(contributions[action.player]),
                effective_stack_before=_float(effective_before),
                effective_stack_range_before=(
                    _float(effective_low),
                    _float(effective_high),
                ),
                spr_before=(
                    _float(effective_before / pot_before) if pot_before > 0 else None
                ),
                spr_range_before=(
                    (
                        _float(effective_low / pot_before),
                        _float(effective_high / pot_before),
                    )
                    if pot_before > 0
                    else None
                ),
                active_players=active_before,
            )
        )

    # Only live money can be uncalled. An ante nobody matched is not an overbet.
    refunds = _uncalled_refunds(player_order, live_contributions)
    settled_contributions = {
        name: live_contributions[name] - refunds[name] for name in player_order
    }
    # Dead money pools into the main pot rather than forming a layer only its
    # poster is eligible for: a lone button ante belongs to whoever wins the hand.
    total_dead = dead + sum(dead_contributions.values(), _ZERO)
    raw_pots = _build_pots(
        player_order,
        settled_contributions,
        folded,
        total_dead,
        dead_contributions,
    )
    _validate_winners(winner_map, raw_pots, starting, folded)

    gross = sum((pot["amount"] for pot in raw_pots), _ZERO)
    observed_flop = (
        any(action.street in _FLOP_STREETS for action in actions)
        if flop_seen is None
        else flop_seen
    )
    rake_total = _compute_rake(
        gross,
        rate=rate,
        cap=cap,
        unit=unit,
        waived=policy.no_flop_no_drop and not observed_flop,
    )
    rake_by_pot = _allocate_rake(raw_pots, rake_total, unit)
    payouts = {name: _ZERO for name in player_order}
    rendered_pots: list[PotLayer] = []
    # Derived from the observed action line only. `unit` -- the declared "Chip
    # unit" -- is deliberately NOT passed and neither is anything computed from
    # it: it rounds the rake and nothing else, so the granularity a chopped pot
    # is divided at is the same at every declared value, at every rake rate.
    # Each dead contribution individually: summing four 0.25 antes into 1.00
    # destroys the hundredths the hand demonstrably deals in, and coarsening
    # the quantum turns an exactly-even chop into an odd one.
    split_unit = _split_granularity(
        [*settled_contributions.values(), *dead_contributions.values()], dead
    )

    for index, pot in enumerate(raw_pots):
        pot_winners = winner_map.get(index, ())
        pot_rake = rake_by_pot[index]
        net_amount = pot["amount"] - pot_rake
        if pot_winners:
            _split_pot(
                net_amount,
                pot_winners,
                payouts,
                unit=split_unit,
                odd_order=odd_order,
            )
        rendered_pots.append(
            PotLayer(
                index=index,
                amount=_float(pot["amount"]),
                net_amount=_float(net_amount),
                rake=_float(pot_rake),
                contributors=pot["contributors"],
                eligible_players=pot["eligible_players"],
                winners=pot_winners,
            )
        )

    is_settled = bool(raw_pots) and len(winner_map) == len(raw_pots)
    if raw_pots and not is_settled:
        warnings.append("Ledger is unsettled because one or more pots have no declared winner.")
    if not raw_pots:
        warnings.append("Ledger has no contestable pot.")

    net_results = {
        name: payouts[name] + refunds[name] - contributions[name] for name in player_order
    }
    paid = sum(payouts.values(), _ZERO)
    is_balanced = is_settled and paid + rake_total == gross

    return HandLedger(
        contributions=_float_map(contributions),
        refunds=_float_map(refunds),
        payouts=_float_map(payouts),
        net_results=_float_map(net_results),
        gross_pot=_float(gross),
        rake=_float(rake_total),
        net_pot=_float(gross - rake_total),
        pots=tuple(rendered_pots),
        snapshots=tuple(snapshots),
        folded_players=tuple(name for name in player_order if name in folded),
        warnings=tuple(warnings),
        legality_issues=tuple(dict.fromkeys(legality_issues)),
        is_settled=is_settled,
        is_balanced=is_balanced,
        is_legal=not legality_issues,
    )


def build_ledger_from_records(
    players: Sequence[PlayerRecord],
    actions: Sequence[ActionRecord],
    *,
    dead_money: float = 0,
    winners: Mapping[int, Sequence[str]] | None = None,
    rake: RakePolicy | None = None,
    odd_chip_order: Sequence[str] = (),
    flop_seen: bool | None = None,
) -> HandLedger:
    """Adapt persisted application records to the normalized ledger contract.

    New records explicitly declare whether an amount is incremental or a
    betting-round total ("raise to"). Historic records marked ``unknown`` are
    rejected for monetary actions rather than being silently reinterpreted.
    """

    normalized_players: list[LedgerPlayer] = []
    identities_by_name: dict[str, list[str]] = {}
    for player in players:
        if player.starting_stack is None:
            raise LedgerError(
                f"Player {player.player_name!r} needs a starting stack for derived accounting."
            )
        identity = getattr(player, "player_key", None) or player.player_name
        identities_by_name.setdefault(player.player_name, []).append(identity)
        normalized_players.append(
            LedgerPlayer(
                name=player.player_name,
                starting_stack=player.starting_stack,
                seat=getattr(player, "seat_index", None),
                key=identity,
            )
        )

    normalized_actions: list[LedgerAction] = []
    street: str | None = None
    street_contributions: dict[str, Decimal] = {
        player.key or player.name: _ZERO for player in normalized_players
    }
    for index, action in enumerate(actions):
        kind = action.action_type
        if kind == "all_in":
            kind = "all-in"
        if kind not in _COMMITMENT_KINDS | _NON_COMMITMENT_KINDS:
            raise LedgerError(f"Action {index + 1} has unsupported kind {kind!r}.")
        identity = getattr(action, "player_key", None)
        if identity is None:
            candidates = identities_by_name.get(action.player_name, [])
            if len(candidates) != 1:
                raise LedgerError(
                    f"Action {index + 1} cannot resolve player {action.player_name!r} "
                    "to one stable identity."
                )
            identity = candidates[0]
        if identity not in street_contributions:
            raise LedgerError(
                f"Action {index + 1} references unknown player identity {identity!r}."
            )
        if action.street != street:
            street = action.street
            street_contributions = {
                player.key or player.name: _ZERO for player in normalized_players
            }
        if kind in _COMMITMENT_KINDS:
            if action.amount is None:
                raise LedgerError(f"Action {index + 1} needs an incremental amount.")
            semantics = getattr(action, "amount_semantics", "unknown")
            if semantics == "unknown":
                raise LedgerError(
                    f"Action {index + 1} has unknown amount semantics and needs correction."
                )
            raw_amount = _decimal(action.amount, f"action {index + 1} amount")
            if semantics == "incremental":
                amount = raw_amount
            elif semantics == "raise_to":
                if kind == "ante":
                    raise LedgerError("Ante amounts cannot use raise-to semantics.")
                amount = raw_amount - street_contributions[identity]
                if amount <= 0:
                    raise LedgerError(
                        f"Action {index + 1} raise-to amount does not exceed the "
                        "player's current street contribution."
                    )
            else:
                raise LedgerError(
                    f"Action {index + 1} has unsupported amount semantics {semantics!r}."
                )
            if kind != "ante" and not (
                kind == "post_blind"
                and getattr(action, "is_live_post", None) is False
            ):
                street_contributions[identity] += amount
        else:
            # Historical win rows sometimes store the reported result in
            # ``amount``.  It is evidence, not a chip commitment or pot award.
            amount = 0
        normalized_actions.append(
            LedgerAction(
                player=identity,
                street=action.street,
                kind=kind,
                amount=_float(amount),
                is_live_post=(
                    True
                    if getattr(action, "is_live_post", None) is None
                    else bool(action.is_live_post)
                ),
            )
        )

    return build_hand_ledger(
        normalized_players,
        normalized_actions,
        winners=winners,
        rake=rake,
        odd_chip_order=odd_chip_order,
        dead_money=dead_money,
        flop_seen=flop_seen,
    )


def _validate_players(
    players: Sequence[LedgerPlayer],
) -> tuple[tuple[str, ...], dict[str, Decimal]]:
    if not players:
        raise LedgerError("At least one player is required.")
    order: list[str] = []
    starting: dict[str, Decimal] = {}
    seats: set[int] = set()
    for player in players:
        name = player.name.strip()
        if not name:
            raise LedgerError("Player names must not be blank.")
        identity = (player.key or name).strip()
        if not identity:
            raise LedgerError("Player keys must not be blank.")
        if identity in starting:
            raise LedgerError(f"Duplicate player identity {identity!r}.")
        stack = _decimal(player.starting_stack, f"starting stack for {name}")
        if stack < 0:
            raise LedgerError("Starting stacks must not be negative.")
        if player.seat is not None:
            if player.seat in seats:
                raise LedgerError(f"Duplicate seat {player.seat}.")
            seats.add(player.seat)
        order.append(identity)
        starting[identity] = stack
    return tuple(order), starting


def _validate_rake(policy: RakePolicy) -> tuple[Decimal, Decimal | None, Decimal]:
    rate = _decimal(policy.rate, "rake rate")
    cap = None if policy.cap is None else _decimal(policy.cap, "rake cap")
    unit = _decimal(policy.rounding_unit, "rake rounding unit")
    if not _ZERO <= rate <= Decimal("1"):
        raise LedgerError("Rake rate must be between zero and one.")
    if cap is not None and cap < 0:
        raise LedgerError("Rake cap must not be negative.")
    if unit <= 0:
        raise LedgerError("Rake rounding unit must be positive.")
    return rate, cap, unit


def _validate_odd_chip_order(
    order: Sequence[str], starting: Mapping[str, Decimal]
) -> tuple[str, ...]:
    if len(set(order)) != len(order):
        raise LedgerError("odd_chip_order contains duplicate players.")
    unknown = [name for name in order if name not in starting]
    if unknown:
        raise LedgerError(f"odd_chip_order references unknown players: {', '.join(unknown)}.")
    return tuple(order)


def _uncalled_refunds(
    order: Sequence[str], contributions: Mapping[str, Decimal]
) -> dict[str, Decimal]:
    refunds = {name: _ZERO for name in order}
    ranked = sorted(((amount, name) for name, amount in contributions.items()), reverse=True)
    if not ranked or ranked[0][0] <= 0:
        return refunds
    highest, player = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else _ZERO
    if highest > second and sum(1 for amount, _ in ranked if amount == highest) == 1:
        refunds[player] = highest - second
    return refunds


def _build_pots(
    order: Sequence[str],
    contributions: Mapping[str, Decimal],
    folded: set[str],
    dead_money: Decimal,
    dead_contributions: Mapping[str, Decimal] | None = None,
) -> list[dict]:
    dead_contributions = dead_contributions or {}
    levels = sorted({amount for amount in contributions.values() if amount > 0})
    pots: list[dict] = []
    previous = _ZERO
    for level in levels:
        contributors = tuple(name for name in order if contributions[name] >= level)
        amount = (level - previous) * len(contributors)
        eligible = tuple(name for name in contributors if name not in folded)
        if not pots:
            amount += dead_money
            # Dead money joins this layer, so everyone still in the hand can win
            # it -- including a player whose ENTIRE commitment was a forced post.
            # Deriving eligibility from live contributions alone left a player
            # all-in for their ante eligible for no pot at all, while their
            # chips sat in one they could not be declared the winner of. That
            # made a hand the short stack won unrecordable, which is routine
            # once a stack is at or below the ante.
            eligible = tuple(
                name
                for name in order
                if name not in folded
                and (name in contributors or dead_contributions.get(name, _ZERO) > 0)
            )
        if amount > 0:
            if not eligible:
                raise LedgerError("A pot has no eligible player.")
            pots.append(
                {
                    "amount": amount,
                    "contributors": contributors,
                    "eligible_players": eligible,
                }
            )
        previous = level
    if not pots and dead_money > 0:
        eligible = tuple(name for name in order if name not in folded)
        if not eligible:
            raise LedgerError("Dead money has no eligible player.")
        pots.append(
            {
                "amount": dead_money,
                "contributors": (),
                "eligible_players": eligible,
            }
        )
    return pots


def _validate_winners(
    winners: Mapping[int, tuple[str, ...]],
    pots: Sequence[dict],
    starting: Mapping[str, Decimal],
    folded: set[str],
) -> None:
    for index, names in winners.items():
        if not isinstance(index, int) or index < 0 or index >= len(pots):
            raise LedgerError(f"Winner declaration references missing pot {index}.")
        if not names:
            raise LedgerError(f"Pot {index} must have at least one winner.")
        if len(set(names)) != len(names):
            raise LedgerError(f"Pot {index} repeats a winner.")
        for name in names:
            if name not in starting:
                raise LedgerError(f"Pot {index} references unknown winner {name!r}.")
            if name in folded:
                raise LedgerError(f"Folded player {name!r} cannot win pot {index}.")
            if name not in pots[index]["eligible_players"]:
                raise LedgerError(f"Player {name!r} is not eligible for pot {index}.")


def _compute_rake(
    gross: Decimal,
    *,
    rate: Decimal,
    cap: Decimal | None,
    unit: Decimal,
    waived: bool,
) -> Decimal:
    if waived or rate == 0 or gross == 0:
        return _ZERO
    raw = gross * rate
    if cap is not None:
        raw = min(raw, cap)
    return _round_down(raw, unit)


def _allocate_rake(pots: Sequence[dict], total: Decimal, unit: Decimal) -> list[Decimal]:
    """Spread one rake total across the pot layers, never past a layer's own size.

    Every non-final share used to be rounded DOWN to the declared unit and the
    whole leftover charged to the LAST pot with no cap at that pot's amount, so
    an ordinary hand under an ordinary policy could rake a pot beyond what it
    contained: a 149.25 main pot and a 0.50 side pot at 5% capped at 5 with a
    whole-chip drop took 4 from the main pot and charged the leftover 1 to a pot
    of 0.50, giving that layer ``net_amount = -0.50`` and paying its winner minus
    half a chip. ``is_balanced`` stayed True because the negative preserved
    ``paid + rake == gross``, so a hand whose side-pot winner also won the main
    pot certified as authoritative and study-ready, and a hand whose pots had
    different winners became permanently unreconcilable: the derived payout was
    negative, ``SettlementEntry.amount`` is ``ge=0`` so the operator could not
    declare it, and ACCOUNTING_NOT_AUTHORITATIVE named a save that could never
    clear it.

    Each share is therefore capped at its own pot, and the rounding leftover is
    offered to the layers in order, each taking only what it still has room for.
    ``_validate_rake`` bounds the rate at one and ``_compute_rake`` caps the
    total at the gross, so the leftover always finds room and the loop always
    settles at zero.
    """
    if not pots:
        return []
    if total <= 0:
        return [_ZERO for _ in pots]
    gross = sum((pot["amount"] for pot in pots), _ZERO)
    allocated: list[Decimal] = []
    remaining = total
    for pot in pots:
        share = _round_down(total * pot["amount"] / gross, unit)
        share = min(share, remaining, pot["amount"])
        allocated.append(share)
        remaining -= share
    for index, pot in enumerate(pots):
        if remaining <= 0:
            break
        take = min(pot["amount"] - allocated[index], remaining)
        if take > 0:
            allocated[index] += take
            remaining -= take
    return allocated


def _split_granularity(
    contributions: Iterable[Decimal], dead_money: Decimal
) -> Decimal:
    """The finest denomination the hand's own numbers are written in.

    ``rake_rounding_unit`` is a DECLARED operator field ("Chip unit" in the
    settlement editor, and a verbatim value in an import payload) with no upper
    bound. Its documented job is rounding the rake, which is a real room rule --
    a house that drops whole dollars against a 0.50 blind is ordinary. Its
    undocumented second job was to be the granularity a CHOPPED pot was divided
    at: ``_split_pot`` rounds each winner's share DOWN to the unit and gives
    every leftover chip to the front of ``odd_chip_order``, so with the rake rate
    at zero -- where neither declared-chips disclosure fires and no correction is
    recorded -- one number in the settlement editor moved the derived hero result
    and made a fabricated ``hands.hero_bb_won`` reconcile exactly.

    Round 8 bounded the declared unit by the greatest common divisor of the
    observed contributions and claimed that removed the dial. It did not. Every
    DIVISOR of that gcd was still reachable, and divisors do not agree once the
    pot is anything but two equal halves: on three seats contributing 8 each and
    a two-way chop of 24, units 0.01/1/2/4 pay 12/12 while 3, 5 and 8 pay 16/8.
    The bound was also the wrong direction. The gcd is an UPPER bound on the
    denomination in play -- every amount is a whole multiple of the real chip, so
    the real chip DIVIDES the gcd and may be far finer -- and splitting at an
    upper bound maximises the redistribution, because rounding the base share
    down sheds up to a whole unit from every winner before the round-robin hands
    those chips back from the front of the order.

    So the split granularity is now derived from evidence alone -- the settled
    contributions and the declared dead money, never the declared unit and never
    anything computed from it -- and it is the FINEST signal those amounts carry,
    not the coarsest. Concretely it is the smallest decimal place any observed
    amount is written in, capped at one whole chip. That is deliberately the
    conservative direction. A denomination cannot be established from above: three
    seats each committing 8 share a factor of 8 and demonstrate nothing about
    8-chips, whereas an amount of 49.75 does demonstrate that hundredths exist.
    Choosing the finest evidence keeps every derived chop as close to an even one
    as the hand allows, so the most any granularity can move a seat is under one
    chip -- against half the pot, which is what the declared unit could move.

    The odd chip survives, because it is real: chips are indivisible, so a
    21-chip pot chopped two ways is genuinely pushed 11/10, and deriving
    10.5/10.5 would raise a false blocker against an honest declared award. What
    it can no longer be is a dial -- the deviation from an even chop is now under
    one chip on every hand, and pinned by the hand's own amounts rather than
    chosen by an operator. ``_split_pot`` always distributes the full amount, so
    no granularity can change the gross, the rake, or chip conservation; only who
    receives an odd chip, and that is decided by the audited ``odd_chip_order``.
    """
    quantum = _MAX_SPLIT_QUANTUM
    for amount in (*contributions, dead_money):
        if amount <= 0:
            continue
        exponent = amount.normalize().as_tuple().exponent
        if not isinstance(exponent, int):
            # 'n', 'N' or 'F' -- a non-finite Decimal. `_decimal` already refuses
            # those on the way in, so this is unreachable defence that keeps the
            # helper total rather than raising from inside the split.
            continue
        # normalize() first, so an amount that arrived as "5.0" and one that
        # arrived as "5" describe the same denomination. Without it the split of
        # one hand depended on whether its amounts reached the ledger as ints or
        # as floats.
        candidate = Decimal(1).scaleb(exponent)
        if candidate < quantum:
            quantum = candidate
    return quantum


def _split_pot(
    amount: Decimal,
    winners: Sequence[str],
    payouts: dict[str, Decimal],
    *,
    unit: Decimal,
    odd_order: Sequence[str],
) -> None:
    base = _round_down(amount / len(winners), unit)
    for winner in winners:
        payouts[winner] += base
    remainder = amount - base * len(winners)
    priority = [name for name in odd_order if name in winners]
    priority.extend(name for name in winners if name not in priority)
    cursor = 0
    while remainder >= unit:
        payouts[priority[cursor % len(priority)]] += unit
        remainder -= unit
        cursor += 1
    if remainder:
        payouts[priority[cursor % len(priority)]] += remainder


def _round_down(value: Decimal, unit: Decimal) -> Decimal:
    return (value / unit).to_integral_value(rounding=ROUND_DOWN) * unit


def _decimal(value: float | int | Decimal, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise LedgerError(f"{label} must be numeric.") from exc
    if not result.is_finite():
        raise LedgerError(f"{label} must be finite.")
    return result


def _float(value: Decimal) -> float:
    return float(value)


def _float_map(values: Mapping[str, Decimal]) -> dict[str, float]:
    return {name: _float(value) for name, value in values.items()}
