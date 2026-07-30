"""Build a solver-ready manual hand from a compact heads-up spot form."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from poker_tracker.math.accounting import build_ledger_from_records
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    SettlementEntry,
)
from poker_tracker.services.hand_accounting import (
    AccountingReconciliation,
    persist_reconciliation,
)

PotType = Literal["single_raised", "three_bet"]
Actor = Literal["hero", "villain"]

_POSITION_ORDER = {
    "SB": 0,
    "BB": 1,
    "UTG": 2,
    "UTG+1": 3,
    "LJ": 4,
    "HJ": 5,
    "CO": 6,
    "BTN": 7,
}
_BLIND_AMOUNTS = {"SB": 0.5, "BB": 1.0}
_POSTFLOP_ACTIONS = {"fold", "check", "call", "bet", "raise", "all-in"}
_POSTFLOP_STREETS = ("flop", "turn", "river")


@dataclass(frozen=True)
class PostflopActionInput:
    street: str
    actor: Actor
    action_type: str
    amount: float | None = None


@dataclass(frozen=True)
class ManualSpotInput:
    """Minimal fields needed to feed TexasSolver from a completed HU spot."""

    hand_number: int
    hero_cards: str
    board_cards: str
    hero_position: str
    villain_position: str
    table_size: int = 6
    starting_stack: float = 100.0
    pot_type: PotType = "single_raised"
    opener: Actor = "villain"
    three_bettor: Actor = "hero"
    open_to: float = 2.5
    three_bet_to: float = 9.0
    postflop_actions: tuple[PostflopActionInput, ...] = ()
    winner: Actor = "hero"
    notes: str = ""
    blinds_antes: str = "1/2 NL"


@dataclass
class BuiltManualSpot:
    hand: Hand
    players: list[HandPlayer]
    actions: list[Action]
    winner_key: str
    warnings: list[str] = field(default_factory=list)


def validate_manual_spot(spot: ManualSpotInput) -> list[str]:
    """Return human-readable validation errors; empty means the spot is saveable."""

    errors: list[str] = []
    hero_cards = spot.hero_cards.split()
    board_cards = spot.board_cards.split()
    if len(hero_cards) != 2:
        errors.append("Enter both Hero hole cards, like Ah Qs.")
    if len(board_cards) < 3:
        errors.append("Enter at least a flop board, like Qd 7s 2c.")
    if len(board_cards) > 5:
        errors.append("Board can have at most five cards.")
    if spot.hero_position not in _POSITION_ORDER:
        errors.append("Choose a Hero position.")
    if spot.villain_position not in _POSITION_ORDER:
        errors.append("Choose a Villain position.")
    if spot.hero_position == spot.villain_position:
        errors.append("Hero and Villain need different positions.")
    if not 5 <= spot.table_size <= 8:
        errors.append("Table size must be 5 through 8 for the solver.")
    if spot.starting_stack <= 0:
        errors.append("Starting stack must be positive.")
    if spot.open_to <= 1:
        errors.append("Open size must be greater than 1 BB.")
    if spot.pot_type == "three_bet":
        if spot.three_bettor == spot.opener:
            errors.append("The 3-bettor must be the other player.")
        if spot.three_bet_to <= spot.open_to:
            errors.append("3-bet size must be larger than the open.")
    if not spot.postflop_actions:
        errors.append("Add at least one postflop action that includes Hero.")
    else:
        errors.extend(_validate_postflop_actions(spot))
    return errors


def build_manual_spot(spot: ManualSpotInput) -> BuiltManualSpot:
    """Expand a compact spot into Hand / players / actions ready to persist."""

    errors = validate_manual_spot(spot)
    if errors:
        raise ValueError(errors[0])

    hero = HandPlayer(
        hand_id=0,
        player_key="hero",
        player_name="Hero",
        position=spot.hero_position,
        starting_stack=spot.starting_stack,
        is_hero=True,
    )
    villain = HandPlayer(
        hand_id=0,
        player_key="villain",
        player_name="Villain",
        position=spot.villain_position,
        starting_stack=spot.starting_stack,
        is_hero=False,
    )
    players = [hero, villain]
    by_position = {
        spot.hero_position: hero,
        spot.villain_position: villain,
    }
    for blind_position in ("SB", "BB"):
        if blind_position not in by_position:
            filler = HandPlayer(
                hand_id=0,
                player_key=f"{blind_position.lower()}_fold",
                player_name=blind_position,
                position=blind_position,
                starting_stack=spot.starting_stack,
                is_hero=False,
            )
            players.append(filler)
            by_position[blind_position] = filler

    actions: list[Action] = []
    for blind_position, amount in _BLIND_AMOUNTS.items():
        player = by_position[blind_position]
        actions.append(
            _action(
                player,
                street="preflop",
                action_type="post_blind",
                amount=amount,
                forced_bet_type="small_blind" if blind_position == "SB" else "big_blind",
                is_live_post=True,
            )
        )

    for blind_position in ("SB", "BB"):
        player = by_position[blind_position]
        if player.player_key in {"hero", "villain"}:
            continue
        actions.append(_action(player, street="preflop", action_type="fold"))

    actor_map = {"hero": hero, "villain": villain}
    opener = actor_map[spot.opener]
    defender = villain if spot.opener == "hero" else hero
    actions.append(
        _action(
            opener,
            street="preflop",
            action_type="raise",
            amount=_incremental_to(spot.open_to, opener.position),
        )
    )

    if spot.pot_type == "single_raised":
        actions.append(
            _action(
                defender,
                street="preflop",
                action_type="call",
                amount=_incremental_to(spot.open_to, defender.position),
            )
        )
    else:
        three_bettor = actor_map[spot.three_bettor]
        caller = opener
        actions.append(
            _action(
                three_bettor,
                street="preflop",
                action_type="raise",
                amount=_incremental_to(spot.three_bet_to, three_bettor.position),
            )
        )
        actions.append(
            _action(
                caller,
                street="preflop",
                action_type="call",
                amount=_incremental_to(spot.three_bet_to, caller.position)
                - _incremental_to(spot.open_to, caller.position),
            )
        )

    for row in spot.postflop_actions:
        player = actor_map[row.actor]
        actions.append(
            _action(
                player,
                street=row.street,
                action_type=row.action_type,
                amount=row.amount,
            )
        )

    oop = _oop_actor(spot.hero_position, spot.villain_position)
    warnings: list[str] = []
    first_postflop = next(
        (row for row in spot.postflop_actions if row.street in _POSTFLOP_STREETS),
        None,
    )
    if first_postflop is not None and first_postflop.actor != oop:
        warnings.append(
            f"First postflop actor is usually {_label(oop)} (OOP). "
            "Check action order if solver eligibility fails."
        )

    hand = Hand(
        session_id=0,
        hand_number=spot.hand_number,
        game_type="NLHE cash",
        blinds_antes=spot.blinds_antes,
        table_size=spot.table_size,
        effective_stack=spot.starting_stack,
        hero_position=spot.hero_position,
        hero_cards=" ".join(spot.hero_cards.split()),
        board_cards=" ".join(spot.board_cards.split()),
        review_status="unreviewed",
        source_type="manual",
        completion_status="not_applicable",
        notes=spot.notes.strip(),
    )
    winner = actor_map[spot.winner]
    return BuiltManualSpot(
        hand=hand,
        players=players,
        actions=actions,
        winner_key=winner.player_key,
        warnings=warnings,
    )


def save_manual_spot(
    db: PokerDatabase,
    session_id: int,
    spot: ManualSpotInput,
) -> tuple[Hand, AccountingReconciliation, list[str]]:
    """Persist a compact spot and reconcile settlement so Study/solver can use it."""

    built = build_manual_spot(spot)
    with db.transaction():
        saved_hand = db.create_hand(
            built.hand.model_copy(update={"session_id": session_id})
        )
        assert saved_hand.id is not None
        saved_players: list[HandPlayer] = []
        for player in built.players:
            saved_players.append(
                db.create_hand_player(player.model_copy(update={"hand_id": saved_hand.id}))
            )
        for action in built.actions:
            db.create_action(action.model_copy(update={"hand_id": saved_hand.id}))
        winner = next(
            player for player in saved_players if player.player_key == built.winner_key
        )
        # Folded blind fillers can create side pots; award every layer to the winner.
        draft_ledger = build_ledger_from_records(saved_players, built.actions)
        pot_indexes = [pot.index for pot in draft_ledger.pots] or [0]
        db.upsert_hand_settlement(
            HandSettlement(hand_id=saved_hand.id, status="settled")
        )
        db.replace_settlement_entries(
            saved_hand.id,
            [
                SettlementEntry(
                    hand_id=saved_hand.id,
                    entry_type="award",
                    pot_index=pot_index,
                    player_key=winner.player_key,
                    player_name=winner.player_name,
                    amount=None,
                    entry_order=order,
                )
                for order, pot_index in enumerate(pot_indexes, start=1)
            ],
        )
    accounting = persist_reconciliation(db, saved_hand.id)
    return saved_hand, accounting, built.warnings


def _validate_postflop_actions(spot: ManualSpotInput) -> list[str]:
    errors: list[str] = []
    seen_hero = False
    previous_street_index = -1
    for index, row in enumerate(spot.postflop_actions, start=1):
        label = f"Postflop action {index}"
        if row.street not in _POSTFLOP_STREETS:
            errors.append(f"{label}: street must be flop, turn, or river.")
            continue
        street_index = _POSTFLOP_STREETS.index(row.street)
        if street_index < previous_street_index:
            errors.append(f"{label}: streets must stay in order.")
        previous_street_index = max(previous_street_index, street_index)
        if row.action_type not in _POSTFLOP_ACTIONS:
            errors.append(f"{label}: unsupported action {row.action_type!r}.")
        if row.action_type in {"bet", "raise", "call", "all-in"} and (
            row.amount is None or row.amount < 0
        ):
            errors.append(f"{label}: enter the chips committed for {row.action_type}.")
        if row.action_type in {"check", "fold"} and row.amount not in (None, 0):
            errors.append(f"{label}: {row.action_type} should not commit chips.")
        if row.actor == "hero":
            seen_hero = True
    if not seen_hero:
        errors.append("Include at least one Hero decision in the postflop line.")
    return errors


def _incremental_to(raise_to: float, position: str) -> float:
    already = _BLIND_AMOUNTS.get(position, 0.0)
    amount = raise_to - already
    if amount <= 0:
        raise ValueError(
            f"Raise-to {raise_to:g} BB is not larger than the {position} blind."
        )
    return amount


def _oop_actor(hero_position: str, villain_position: str) -> Actor:
    return (
        "hero"
        if _POSITION_ORDER[hero_position] < _POSITION_ORDER[villain_position]
        else "villain"
    )


def _label(actor: Actor) -> str:
    return "Hero" if actor == "hero" else "Villain"


def _action(
    player: HandPlayer,
    *,
    street: str,
    action_type: str,
    amount: float | None = None,
    forced_bet_type: str | None = None,
    is_live_post: bool | None = None,
) -> Action:
    return Action(
        hand_id=0,
        player_key=player.player_key,
        player_name=player.player_name,
        position=player.position,
        street=street,  # type: ignore[arg-type]
        action_type=action_type,  # type: ignore[arg-type]
        amount=amount,
        amount_semantics="incremental",
        forced_bet_type=forced_bet_type,  # type: ignore[arg-type]
        is_live_post=is_live_post,
    )
