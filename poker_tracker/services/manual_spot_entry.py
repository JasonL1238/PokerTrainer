"""Build a solver-ready manual hand from a compact heads-up spot form."""

from __future__ import annotations

import re
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
_STREET_ALIASES = {
    "f": "flop",
    "flop": "flop",
    "t": "turn",
    "turn": "turn",
    "r": "river",
    "river": "river",
}
_ACTION_TOKEN_RE = re.compile(
    r"^(?:(?P<actor>[hvHV])[:\-]?)?(?P<code>x|f|c|b|r|ai|allin|all-in|check|fold|call|bet|raise)"
    r"(?P<amount>\d+(?:\.\d+)?)?$",
    re.IGNORECASE,
)
_POSITION_VS_RE = re.compile(
    r"^(?P<hero>UTG\+1|UTG|LJ|HJ|CO|BTN|SB|BB)\s+vs\.?\s+"
    r"(?P<villain>UTG\+1|UTG|LJ|HJ|CO|BTN|SB|BB)$",
    re.IGNORECASE,
)
_CARD_TOKEN_RE = re.compile(r"^[2-9TJQKA][shdc]$", re.IGNORECASE)
_CARD_RUN_RE = re.compile(r"[2-9TJQKA][shdc]", re.IGNORECASE)
_WINNER_RE = re.compile(r"^(?:w(?:inner)?[:\s]*)?(hero|villain|h|v)$", re.IGNORECASE)
_POT_TYPE_RE = re.compile(r"^(srp|single[_ -]?raised|3bet|three[_ -]?bet)$", re.IGNORECASE)
_SIZE_RE = re.compile(
    r"^(?:(?P<kind>open|3b(?:et)?|three[_ -]?bet)[:\s]*)?(?P<amount>\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


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


@dataclass(frozen=True)
class ManualSpotDefaults:
    """Sticky structure fields applied to single-line and multi-hand paste."""

    hero_position: str = "BB"
    villain_position: str = "BTN"
    table_size: int = 6
    starting_stack: float = 100.0
    pot_type: PotType = "single_raised"
    opener: Actor = "villain"
    three_bettor: Actor = "hero"
    open_to: float = 2.5
    three_bet_to: float = 9.0
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


@dataclass(frozen=True)
class ParsedSpotBatch:
    spots: tuple[ManualSpotInput, ...]
    errors: tuple[str, ...] = ()


def parse_postflop_line(
    text: str,
    *,
    hero_position: str,
    villain_position: str,
) -> tuple[tuple[PostflopActionInput, ...], list[str]]:
    """Parse compact HU notation like ``x/b3.5/c | x/b8/f`` into postflop actions.

    Streets are separated by ``|`` or ``;``. Actions within a street are separated
    by ``/`` or whitespace. Optional street labels (``F:``, ``turn``) and actor
    prefixes (``h``, ``v``) are supported. Bare ``c`` uses the current to-call.
    """

    raw = text.strip()
    if not raw:
        return (), ["Add a postflop line, like x/b3.5/c | x/b8/f."]
    if hero_position not in _POSITION_ORDER or villain_position not in _POSITION_ORDER:
        return (), ["Choose Hero and Villain positions before parsing the line."]
    if hero_position == villain_position:
        return (), ["Hero and Villain need different positions."]

    oop = _oop_actor(hero_position, villain_position)
    street_chunks = _split_street_chunks(raw)
    if not street_chunks:
        return (), ["Could not read any streets in the postflop line."]

    actions: list[PostflopActionInput] = []
    errors: list[str] = []
    used_streets: set[str] = set()
    auto_street_index = 0

    for chunk_index, chunk in enumerate(street_chunks, start=1):
        street, action_text = _split_street_label(chunk)
        if street is None:
            if auto_street_index >= len(_POSTFLOP_STREETS):
                errors.append(
                    f"Street {chunk_index}: only flop, turn, and river are supported."
                )
                continue
            street = _POSTFLOP_STREETS[auto_street_index]
            auto_street_index += 1
        elif street in used_streets:
            errors.append(f"Street {chunk_index}: {street} appears more than once.")
            continue
        else:
            # Keep auto-index ahead of any explicit later streets.
            auto_street_index = max(auto_street_index, _POSTFLOP_STREETS.index(street) + 1)
        used_streets.add(street)

        tokens = _tokenize_action_tokens(action_text)
        if not tokens:
            errors.append(f"{street}: add at least one action.")
            continue

        committed = {"hero": 0.0, "villain": 0.0}
        next_actor: Actor = oop
        for token_index, token in enumerate(tokens, start=1):
            parsed = _parse_action_token(token)
            if parsed is None:
                errors.append(f"{street} action {token_index}: could not parse {token!r}.")
                continue
            actor_hint, action_type, explicit_amount = parsed
            actor = actor_hint or next_actor
            amount = explicit_amount
            to_call = max(committed.values(), default=0.0) - committed[actor]
            if action_type == "check":
                if to_call > 1e-9:
                    errors.append(
                        f"{street} action {token_index}: cannot check facing a bet."
                    )
                    continue
                amount = None
            elif action_type == "fold":
                amount = None
            elif action_type == "call":
                if amount is None:
                    amount = to_call
                if amount is None or amount < 0:
                    errors.append(f"{street} action {token_index}: call amount is invalid.")
                    continue
                if amount == 0 and to_call <= 1e-9:
                    errors.append(
                        f"{street} action {token_index}: nothing to call; use check (x)."
                    )
                    continue
                committed[actor] += float(amount)
            elif action_type in {"bet", "raise", "all-in"}:
                if amount is None:
                    if action_type == "all-in":
                        errors.append(
                            f"{street} action {token_index}: enter all-in chips, like ai50."
                        )
                    else:
                        errors.append(
                            f"{street} action {token_index}: enter an amount, like "
                            f"{'b3.5' if action_type == 'bet' else 'r10'}."
                        )
                    continue
                if action_type == "bet" and to_call > 1e-9:
                    errors.append(
                        f"{street} action {token_index}: facing a bet; use raise (r) or call (c)."
                    )
                    continue
                if action_type == "raise" and to_call <= 1e-9:
                    errors.append(
                        f"{street} action {token_index}: nothing to raise; use bet (b)."
                    )
                    continue
                committed[actor] += float(amount)
            actions.append(
                PostflopActionInput(
                    street=street,
                    actor=actor,
                    action_type=action_type,
                    amount=amount,
                )
            )
            next_actor = "villain" if actor == "hero" else "hero"

    return tuple(actions), errors


def parse_manual_spot_lines(
    text: str,
    defaults: ManualSpotDefaults,
    *,
    starting_hand_number: int,
) -> ParsedSpotBatch:
    """Parse one or more compact hand lines into saveable spots.

    Each non-empty, non-comment line is one hand. Pipe-separated fields:

    ``AhQs | Qd7s2c | x/b3.5/c | hero``

    Optional fields: ``BB vs BTN``, ``SRP`` / ``3bet``, ``open2.5``, ``3b9``,
    ``W:villain``. Structure fields fall back to *defaults*.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return ParsedSpotBatch((), ("Paste at least one hand line.",))

    spots: list[ManualSpotInput] = []
    errors: list[str] = []
    hand_number = starting_hand_number
    for line_index, line in enumerate(lines, start=1):
        spot, line_errors = _parse_manual_spot_line(
            line,
            defaults,
            hand_number=hand_number,
            line_label=f"Line {line_index}",
        )
        if line_errors:
            errors.extend(line_errors)
            continue
        assert spot is not None
        spots.append(spot)
        hand_number += 1
    return ParsedSpotBatch(tuple(spots), tuple(errors))


def format_postflop_line(actions: tuple[PostflopActionInput, ...] | list[PostflopActionInput]) -> str:
    """Render postflop actions back to compact notation for display/editing."""

    by_street: dict[str, list[str]] = {}
    for row in actions:
        token = _format_action_token(row)
        by_street.setdefault(row.street, []).append(token)
    parts: list[str] = []
    for street in _POSTFLOP_STREETS:
        tokens = by_street.get(street)
        if tokens:
            parts.append("/".join(tokens))
    return " | ".join(parts)


def validate_manual_spot(spot: ManualSpotInput) -> list[str]:
    """Return human-readable validation errors; empty means the spot is saveable."""

    errors: list[str] = []
    hero_cards = (_normalize_card_field(spot.hero_cards) or "").split()
    board_cards = (_normalize_card_field(spot.board_cards) or "").split()
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
        shape_errors = _validate_postflop_actions(spot)
        errors.extend(shape_errors)
        if not shape_errors:
            # Only once the line is structurally readable: closure is a statement
            # about amounts and actors, and there is nothing to say about either
            # while the tokens themselves are still wrong.
            errors.extend(_validate_line_is_settleable(spot))
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

    hero_cards = _normalize_card_field(spot.hero_cards) or spot.hero_cards
    board_cards = _normalize_card_field(spot.board_cards) or spot.board_cards
    hand = Hand(
        session_id=0,
        hand_number=spot.hand_number,
        game_type="NLHE cash",
        blinds_antes=spot.blinds_antes,
        table_size=spot.table_size,
        effective_stack=spot.starting_stack,
        hero_position=spot.hero_position,
        hero_cards=hero_cards,
        board_cards=board_cards,
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
        # Folded blind fillers split the pot into layers -- dead-money layers,
        # not side pots -- so award every layer to the winner.
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


def save_manual_spots(
    db: PokerDatabase,
    session_id: int,
    spots: tuple[ManualSpotInput, ...] | list[ManualSpotInput],
) -> list[tuple[Hand, AccountingReconciliation, list[str]]]:
    """Validate all spots, then persist each. Raises ValueError on the first invalid spot."""

    if not spots:
        raise ValueError("No hands to save.")
    for spot in spots:
        errors = validate_manual_spot(spot)
        if errors:
            raise ValueError(f"Hand #{spot.hand_number}: {errors[0]}")
    return [save_manual_spot(db, session_id, spot) for spot in spots]


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


def _validate_line_is_settleable(spot: ManualSpotInput) -> list[str]:
    """Refuse a spot whose action line cannot produce the settlement it declares.

    ``_validate_postflop_actions`` checks that the tokens are legal one at a
    time -- street order, a known verb, an amount where one belongs. Nothing
    checked that the SEQUENCE ends anywhere, and a compact-notation form invites
    exactly that truncation: an operator recording the decision point they want
    to study types ``x/b3.5`` (Hero checks, Villain bets 3.5, Hero has not acted)
    and stops, because that is the spot.

    Saving it was not a coverage limitation, it was a fabrication. The ledger
    resolves Villain's unmatched 3.5 as an UNCALLED bet and refunds it -- which
    is the arithmetic of a hand Hero folded -- while ``save_manual_spot`` pushes
    every pot layer to the declared winner. The persisted settlement therefore
    asserts both "Villain's last wager went unanswered" and "Hero won the pot",
    which describes a hand that cannot happen at any table, and it asserted it
    silently: no validation error, no save warning, no ledger warning, no
    readiness blocker, settlement status ``reconciled``, and a Study headline
    reading "This hand passed every trust check". Hero's real result on that line
    is either -2.5 (he folded) or +6.5 (he called and won); the +3.0 that was
    published is a number no completion of the hand produces.

    Four impossibilities are refused here, and they are the ones the compact form
    can express:

    * a betting round nobody closed -- a player still in the hand, with chips
      behind, left facing a wager they never answered;
    * a declared winner who folded in the line above the declaration;
    * a player who keeps acting after folding;
    * a player committing more chips than their stack holds.

    The last three are the ledger's own rules, and the ledger already enforces
    them -- by raising ``LedgerError`` from inside ``save_manual_spot``, after
    the hand, its players and its actions have been committed in that function's
    own transaction. Enforcing them HERE is what makes this function's contract
    ("empty means the spot is saveable") true: a form that reports no errors and
    then throws is a form whose validation does not describe what saving will do.

    Deliberately NOT refused: a hand that is closed but short. ``x/b3.5/c`` on a
    three-card board stops before the turn, and ``x/x`` stops with nobody facing
    anything. Those are ordinary study spots -- the operator watched the hand and
    is recording the part worth reviewing -- and the winner they declare is an
    observation the accounting layer already measures as an assumption. What
    cannot be an observation is a result derived from a wager the record shows
    was never answered.
    """

    errors: list[str] = []
    # Both seats reach the flop having committed the same "to" amount: the
    # builder pairs an open with a call, or a 3-bet with a call.
    preflop_to = spot.three_bet_to if spot.pot_type == "three_bet" else spot.open_to
    committed = {"hero": float(preflop_to), "villain": float(preflop_to)}
    folded: dict[str, str] = {}

    for street in _POSTFLOP_STREETS:
        rows = [row for row in spot.postflop_actions if row.street == street]
        if not rows:
            continue
        street_committed = {"hero": 0.0, "villain": 0.0}
        for row in rows:
            if row.actor in folded:
                errors.append(
                    f"{street}: {_label(row.actor)} already folded on the "
                    f"{folded[row.actor]} and cannot act again."
                )
                continue
            if row.action_type in {"bet", "raise", "call", "all-in"} and row.amount:
                street_committed[row.actor] += float(row.amount)
                committed[row.actor] += float(row.amount)
                if committed[row.actor] - float(spot.starting_stack) > 1e-9:
                    errors.append(
                        f"{street}: {_label(row.actor)} commits "
                        f"{committed[row.actor]:g} BB in total from a "
                        f"{spot.starting_stack:g} BB stack."
                    )
            elif row.action_type == "fold":
                folded.setdefault(row.actor, street)
        wager = max(street_committed.values())
        for actor in ("hero", "villain"):
            if actor in folded:
                continue
            to_call = wager - street_committed[actor]
            behind = float(spot.starting_stack) - committed[actor]
            if to_call > 1e-9 and behind > 1e-9:
                errors.append(
                    f"{street}: {_label(actor)} is left facing {to_call:g} BB that "
                    "the line never answers. Add the call, fold, raise, or all-in "
                    "that closed the betting -- an unanswered wager is not a "
                    "completed hand, and no result can be derived from one."
                )

    if spot.winner in folded:
        errors.append(
            f"{_label(spot.winner)} folded on the {folded[spot.winner]} and cannot "
            "be the declared winner. Correct the winner or the postflop line."
        )
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


def _split_street_chunks(text: str) -> list[str]:
    # Prefer explicit street separators; otherwise treat the whole line as one street.
    if "|" in text or ";" in text:
        return [part.strip() for part in re.split(r"[|;]", text) if part.strip()]
    return [text.strip()]


def _split_street_label(chunk: str) -> tuple[str | None, str]:
    labeled = re.match(
        r"^(?P<label>flop|turn|river|f|t|r)\s*[:\-]\s*(?P<body>.+)$",
        chunk.strip(),
        re.IGNORECASE,
    )
    if labeled:
        return _STREET_ALIASES[labeled.group("label").lower()], labeled.group("body").strip()
    return None, chunk.strip()


def _tokenize_action_tokens(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if "/" in cleaned:
        return [part.strip() for part in cleaned.split("/") if part.strip()]
    return [part for part in re.split(r"\s+", cleaned) if part]


def _parse_action_token(
    token: str,
) -> tuple[Actor | None, str, float | None] | None:
    match = _ACTION_TOKEN_RE.match(token.strip())
    if match is None:
        return None
    actor_raw = match.group("actor")
    actor: Actor | None = None
    if actor_raw:
        actor = "hero" if actor_raw.lower() == "h" else "villain"
    code = match.group("code").lower()
    action_map = {
        "x": "check",
        "check": "check",
        "f": "fold",
        "fold": "fold",
        "c": "call",
        "call": "call",
        "b": "bet",
        "bet": "bet",
        "r": "raise",
        "raise": "raise",
        "ai": "all-in",
        "allin": "all-in",
        "all-in": "all-in",
    }
    action_type = action_map[code]
    amount_raw = match.group("amount")
    amount = float(amount_raw) if amount_raw is not None else None
    return actor, action_type, amount


def _format_action_token(row: PostflopActionInput) -> str:
    short = {
        "check": "x",
        "fold": "f",
        "call": "c",
        "bet": "b",
        "raise": "r",
        "all-in": "ai",
    }[row.action_type]
    prefix = "h" if row.actor == "hero" else "v"
    if row.amount is None or row.action_type in {"check", "fold"}:
        return f"{prefix}{short}"
    amount = f"{row.amount:g}"
    return f"{prefix}{short}{amount}"


def _parse_manual_spot_line(
    line: str,
    defaults: ManualSpotDefaults,
    *,
    hand_number: int,
    line_label: str,
) -> tuple[ManualSpotInput | None, list[str]]:
    fields = [part.strip() for part in line.split("|")]
    if len(fields) == 1:
        # Allow whitespace form: AhQs Qd7s2c x/b3.5/c hero
        fields = _split_whitespace_hand_fields(line)
    if len(fields) < 3:
        return None, [
            f"{line_label}: need hero cards, board, and line "
            "(e.g. AhQs | Qd7s2c | x/b3.5/c | hero)."
        ]

    hero_cards = _normalize_card_field(fields[0])
    board_cards = _normalize_card_field(fields[1])
    if hero_cards is None:
        return None, [f"{line_label}: enter Hero hole cards, like Ah Qs."]
    if board_cards is None:
        return None, [f"{line_label}: enter a board, like Qd 7s 2c."]
    rest = fields[2:]
    hero_position = defaults.hero_position
    villain_position = defaults.villain_position
    pot_type = defaults.pot_type
    open_to = defaults.open_to
    three_bet_to = defaults.three_bet_to
    winner = defaults.winner
    action_line: str | None = None
    errors: list[str] = []

    for part in rest:
        position_match = _POSITION_VS_RE.match(part)
        if position_match:
            hero_position = _normalize_position(position_match.group("hero"))
            villain_position = _normalize_position(position_match.group("villain"))
            continue
        pot_match = _POT_TYPE_RE.match(part)
        if pot_match:
            token = pot_match.group(1).lower()
            pot_type = (
                "three_bet" if token.startswith("3") or "three" in token else "single_raised"
            )
            continue
        winner_match = _WINNER_RE.match(part)
        if winner_match:
            token = winner_match.group(1).lower()
            winner = "hero" if token in {"hero", "h"} else "villain"
            continue
        size_match = _SIZE_RE.match(part)
        if size_match and _looks_like_size_field(part):
            kind = (size_match.group("kind") or "").lower()
            amount = float(size_match.group("amount"))
            if kind.startswith("3") or "three" in kind:
                three_bet_to = amount
            else:
                open_to = amount
            continue
        if action_line is None and _looks_like_action_line(part):
            action_line = part
            continue
        errors.append(f"{line_label}: unrecognized field {part!r}.")

    if action_line is None:
        errors.append(f"{line_label}: missing postflop line (e.g. x/b3.5/c).")
        return None, errors

    actions, action_errors = parse_postflop_line(
        action_line,
        hero_position=hero_position,
        villain_position=villain_position,
    )
    errors.extend(f"{line_label}: {message}" for message in action_errors)
    if errors:
        return None, errors

    spot = ManualSpotInput(
        hand_number=hand_number,
        hero_cards=hero_cards,
        board_cards=board_cards,
        hero_position=hero_position,
        villain_position=villain_position,
        table_size=defaults.table_size,
        starting_stack=defaults.starting_stack,
        pot_type=pot_type,
        opener=defaults.opener,
        three_bettor=defaults.three_bettor,
        open_to=open_to,
        three_bet_to=three_bet_to,
        postflop_actions=actions,
        winner=winner,
        notes=defaults.notes,
        blinds_antes=defaults.blinds_antes,
    )
    spot_errors = validate_manual_spot(spot)
    if spot_errors:
        return None, [f"{line_label}: {message}" for message in spot_errors]
    return spot, []


def _split_whitespace_hand_fields(line: str) -> list[str]:
    """Best-effort split when the user omits pipes."""

    tokens = line.split()
    if len(tokens) < 3:
        return tokens
    # Hero cards: 2 tokens or 1 glued pair AhQs
    index = 0
    if (
        len(tokens) >= 2
        and _CARD_TOKEN_RE.match(tokens[0])
        and _CARD_TOKEN_RE.match(tokens[1])
    ):
        hero = f"{tokens[0]} {tokens[1]}"
        index = 2
    elif re.fullmatch(r"(?:[2-9TJQKA][shdc]){2}", tokens[0], re.IGNORECASE):
        hero = f"{tokens[0][:2]} {tokens[0][2:]}"
        index = 1
    else:
        return [line]

    board_cards: list[str] = []
    while index < len(tokens) and _CARD_TOKEN_RE.match(tokens[index]):
        board_cards.append(tokens[index])
        index += 1
    if len(board_cards) < 3:
        return [line]
    rest = " ".join(tokens[index:]).strip()
    fields = [hero, " ".join(board_cards)]
    if rest:
        # Re-run pipe-style parsing on the remainder by inventing pipes around known bits
        fields.extend(_extract_trailing_fields(rest))
    return fields


def _extract_trailing_fields(rest: str) -> list[str]:
    """Pull optional metadata and the action line out of a whitespace remainder."""

    fields: list[str] = []
    remainder = rest
    position_match = re.search(
        r"\b(UTG\+1|UTG|LJ|HJ|CO|BTN|SB|BB)\s+vs\.?\s+(UTG\+1|UTG|LJ|HJ|CO|BTN|SB|BB)\b",
        remainder,
        re.IGNORECASE,
    )
    if position_match:
        fields.append(position_match.group(0))
        remainder = (
            remainder[: position_match.start()] + remainder[position_match.end() :]
        ).strip()

    pot_match = re.search(
        r"\b(?:srp|single[_ -]?raised|3bet|three[_ -]?bet)\b",
        remainder,
        re.IGNORECASE,
    )
    if pot_match:
        fields.append(pot_match.group(0))
        remainder = (remainder[: pot_match.start()] + remainder[pot_match.end() :]).strip()

    winner_match = re.search(
        r"\b(?:w(?:inner)?[:\s]*)?(hero|villain)\b\s*$",
        remainder,
        re.IGNORECASE,
    )
    winner_field = None
    action_part = remainder
    if winner_match:
        winner_field = winner_match.group(0)
        action_part = remainder[: winner_match.start()].strip()
    if action_part:
        fields.append(action_part)
    if winner_field:
        fields.append(winner_field)
    return fields


def _normalize_position(value: str) -> str:
    raw = value.strip().upper()
    if raw == "UTG+1":
        return "UTG+1"
    aliases = {position.upper(): position for position in _POSITION_ORDER}
    return aliases.get(raw, value.strip())


def _normalize_card_field(value: str) -> str | None:
    """Accept spaced or glued cards (``Ah Qs`` / ``AhQs`` / ``Qd7s2c``)."""

    cards = [
        f"{match.group(0)[0].upper()}{match.group(0)[1].lower()}"
        for match in _CARD_RUN_RE.finditer(value)
    ]
    if not cards:
        return None
    return " ".join(cards)


def _looks_like_action_line(field: str) -> bool:
    if "/" in field or "|" in field or ";" in field:
        return True
    tokens = _tokenize_action_tokens(field)
    return bool(tokens) and all(_parse_action_token(token) is not None for token in tokens)


def _looks_like_size_field(field: str) -> bool:
    lowered = field.strip().lower()
    if lowered.startswith(("open", "3b", "three")):
        return True
    # Bare numbers are treated as open size only when prefixed; otherwise they
    # collide with action amounts. Require a keyword.
    return False
