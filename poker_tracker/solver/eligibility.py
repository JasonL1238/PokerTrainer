from __future__ import annotations

import re
from dataclasses import dataclass

from poker_tracker.persistence.models import Action, Hand, HandPlayer
from poker_tracker.services.hand_accounting import AccountingReconciliation
from poker_tracker.services.study_readiness import accounting_is_established
from poker_tracker.solver.models import (
    EligibilityResult,
    RecordedSolverAction,
    SolverPlayer,
    SolverSpot,
)

_STREET_ORDER = ("flop", "turn", "river")


class SolverEligibilityError(ValueError):
    """Raised when a saved hand cannot produce an honest TexasSolver spot."""


@dataclass(frozen=True)
class PreparedSpot:
    eligibility: EligibilityResult
    spot: SolverSpot | None


def prepare_solver_spot(
    hand: Hand,
    players: list[HandPlayer],
    actions: list[Action],
    accounting: AccountingReconciliation | None,
) -> PreparedSpot:
    reasons: list[str] = []
    warnings: list[str] = []
    if hand.id is None:
        reasons.append("The hand must be saved before solver analysis.")
    # `accounting_is_established`, not `is_authoritative`. The spot's pot sizes
    # are built from this ledger's snapshots, and declared dead money is added
    # straight into pot 0, so an unanswered declaration inflates the geometry the
    # solve is run on -- and the saved artefact is then retained as evidence about
    # the hand. The coaching button beside this one was already gated on the
    # established predicate, so leaving the raw flag here left a second, unguarded
    # route to a saved `coaching_responses` row for the same hand.
    if not accounting_is_established(hand, accounting):
        reasons.append(
            "Reconcile a legal, balanced completed-hand ledger first, and confirm "
            "any declared settlement assumption it rests on."
        )
    game = hand.game_type.strip().lower()
    if not game:
        reasons.append("Record the game type as no-limit Hold'em cash.")
    elif not _is_nlhe(game):
        reasons.append("TexasSolver coaching currently supports no-limit Hold'em only.")
    elif not _is_cash_game(game):
        reasons.append("TexasSolver coaching requires an explicit cash or ring-game label.")
    if _is_tournament(game):
        reasons.append("Tournament and ICM spots are outside the cash-game solver scope.")
    if hand.table_size is None or not 5 <= hand.table_size <= 8:
        reasons.append("Record a table size from 5 through 8.")
    if len(hand.hero_cards.split()) != 2:
        reasons.append("Record both Hero hole cards.")
    if len(hand.board_cards.split()) < 3:
        reasons.append("A postflop board is required.")
    hero_players = [player for player in players if player.is_hero]
    if len(hero_players) != 1:
        reasons.append("Exactly one saved player must be marked as Hero.")

    preflop_aggression = [
        action
        for action in actions
        if action.street == "preflop" and action.action_type in {"raise", "all-in"}
    ]
    if len(preflop_aggression) == 1:
        pot_type = "single_raised"
    elif len(preflop_aggression) == 2:
        pot_type = "three_bet"
    elif len(preflop_aggression) == 0:
        reasons.append("Limped or missing-preflop-action pots are not supported.")
        pot_type = "single_raised"
    else:
        reasons.append("Four-bet and larger preflop structures are not supported.")
        pot_type = "three_bet"

    if reasons or accounting is None or hand.id is None or hand.table_size is None:
        return PreparedSpot(
            eligibility=EligibilityResult(eligible=False, reasons=reasons, warnings=warnings),
            spot=None,
        )

    snapshots = list(accounting.ledger.snapshots)
    if len(snapshots) != len(actions):
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=["The reconciled action ledger cannot be aligned to the saved actions."],
                warnings=warnings,
            ),
            spot=None,
        )

    selected_index: int | None = None
    selected_street: str | None = None
    for street in _STREET_ORDER:
        indexes = [index for index, action in enumerate(actions) if action.street == street]
        if not indexes:
            continue
        first_index = indexes[0]
        first_snapshot = snapshots[first_index]
        if len(first_snapshot.active_players) == 2 and first_snapshot.to_call_before == 0:
            selected_index = first_index
            selected_street = street
            break
    if selected_index is None or selected_street is None:
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=[
                    "No street begins with exactly two active players and a clean betting state."
                ],
                warnings=warnings,
            ),
            spot=None,
        )

    first_snapshot = snapshots[selected_index]
    active_keys = tuple(first_snapshot.active_players)
    players_by_key = {player.player_key: player for player in players}
    if any(key not in players_by_key for key in active_keys):
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=["The heads-up players cannot be resolved from saved identities."],
                warnings=warnings,
            ),
            spot=None,
        )
    prior_contributions = {key: 0.0 for key in active_keys}
    for snapshot in snapshots[:selected_index]:
        if snapshot.player in prior_contributions:
            prior_contributions[snapshot.player] = snapshot.hand_contribution_after
    non_actionable = [
        players_by_key[key].player_name
        for key in active_keys
        if players_by_key[key].starting_stack is None
        or players_by_key[key].starting_stack - prior_contributions[key] <= 1e-9
    ]
    if non_actionable:
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=[
                    "Both heads-up players must still have chips available at the "
                    f"solver node; not actionable: {', '.join(non_actionable)}."
                ],
                warnings=warnings,
            ),
            spot=None,
        )
    hero = hero_players[0]
    if hero.player_key not in active_keys:
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=["Hero is not one of the two players in the usable postflop spot."],
                warnings=warnings,
            ),
            spot=None,
        )

    oop_key = actions[selected_index].player_key
    if oop_key not in active_keys:
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=["The first postflop actor cannot be resolved as the out-of-position player."],
                warnings=warnings,
            ),
            spot=None,
        )
    expected_oop = _expected_oop(active_keys, players_by_key)
    if expected_oop is None:
        warnings.append(
            "IP/OOP could not be verified from saved positions and was inferred "
            "from the first postflop actor."
        )
    if expected_oop is not None and oop_key != expected_oop:
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=[
                    "The first postflop actor conflicts with the saved positions. "
                    "Correct missing or out-of-order actions before solver analysis."
                ],
                warnings=warnings,
            ),
            spot=None,
        )
    ip_key = next(key for key in active_keys if key != oop_key)
    oop_record = players_by_key[oop_key]
    ip_record = players_by_key[ip_key]
    board_count = {"flop": 3, "turn": 4, "river": 5}[selected_street]
    board = " ".join(hand.board_cards.split()[:board_count])
    if len(board.split()) != board_count:
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=[f"The {selected_street} board is incomplete."],
                warnings=warnings,
            ),
            spot=None,
        )
    if first_snapshot.pot_before <= 0 or first_snapshot.effective_stack_before <= 0:
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=["The starting pot and effective stack must both be positive."],
                warnings=warnings,
            ),
            spot=None,
        )

    prior_multiway = any(
        snapshot.street in _STREET_ORDER
        and snapshot.index < first_snapshot.index
        and len(snapshot.active_players) > 2
        for snapshot in snapshots
    )
    if prior_multiway:
        warnings.append(
            "The hand became heads-up after earlier multiway play; starting ranges are approximate."
        )
    if accounting.ledger.rake > 0:
        warnings.append(
            f"Recorded rake was {accounting.ledger.rake:g} BB; TexasSolver is running a "
            "no-rake equilibrium approximation."
        )

    recorded_line: list[RecordedSolverAction] = []
    for action, snapshot in zip(actions[selected_index:], snapshots[selected_index:], strict=True):
        if action.street not in _STREET_ORDER or action.player_key not in active_keys:
            continue
        recorded_line.append(
            RecordedSolverAction(
                player_key=action.player_key,
                player_name=action.player_name,
                street=action.street,
                action_type=action.action_type,
                # TexasSolver action amounts are incremental chips committed by
                # this action. The ledger has already normalized raise-to input.
                amount=(snapshot.amount if action.amount is not None else None),
                pot_before=snapshot.pot_before,
            )
        )
    if not any(action.player_key == hero.player_key for action in recorded_line):
        return PreparedSpot(
            eligibility=EligibilityResult(
                eligible=False,
                reasons=["The usable heads-up subtree contains no recorded Hero decision."],
                warnings=warnings,
            ),
            spot=None,
        )

    spot = SolverSpot(
        hand_id=hand.id,
        table_size=hand.table_size,
        street=selected_street,
        board=board,
        pot=first_snapshot.pot_before,
        effective_stack=first_snapshot.effective_stack_before,
        pot_type=pot_type,
        preflop_aggressor_key=preflop_aggression[-1].player_key or "",
        oop=_solver_player(oop_record, "oop"),
        ip=_solver_player(ip_record, "ip"),
        hero_cards=hand.hero_cards,
        recorded_line=recorded_line,
        prior_multiway=prior_multiway,
        rake_amount=accounting.ledger.rake,
    )
    return PreparedSpot(
        eligibility=EligibilityResult(eligible=True, warnings=warnings),
        spot=spot,
    )


def _solver_player(player: HandPlayer, role: str) -> SolverPlayer:
    return SolverPlayer(
        player_key=player.player_key,
        player_name=player.player_name,
        position=player.position,
        role=role,
        is_hero=player.is_hero,
    )


def _is_nlhe(game: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", game.lower()).strip()
    normalized = normalized.replace("hold em", "holdem")
    tokens = set(normalized.split())
    if {"plo", "omaha"} & tokens or "short deck" in normalized or "shortdeck" in tokens:
        return False
    if "fixed limit" in normalized or "pot limit" in normalized:
        return False
    if "limit" in tokens and "no limit" not in normalized and not normalized.startswith("nl "):
        return False
    return (
        "nlhe" in tokens
        or ("holdem" in tokens and "no limit" in normalized)
        or ("holdem" in tokens and normalized.startswith("nl "))
    )


def _is_tournament(game: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", game.lower()).strip()
    tokens = set(normalized.split())
    return bool(
        {
            "tournament",
            "tourney",
            "tourn",
            "mtt",
            "sng",
            "satellite",
            "freeroll",
            "freezeout",
            "bounty",
            "pko",
        }
        & tokens
        or "sit and go" in normalized
        or "sit n go" in normalized
    )


def _is_cash_game(game: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", game.lower()).strip()
    tokens = set(normalized.split())
    return "cash" in tokens or "ring game" in normalized or "ringgame" in tokens


def _expected_oop(
    active_keys: tuple[str, ...],
    players_by_key: dict[str, HandPlayer],
) -> str | None:
    order = {
        "SB": 0,
        "BB": 1,
        "UTG": 2,
        "UTG+1": 3,
        "LJ": 4,
        "HJ": 5,
        "CO": 6,
        "BTN": 7,
    }
    aliases = {"BUTTON": "BTN", "BU": "BTN"}
    ranked: list[tuple[int, str]] = []
    for key in active_keys:
        position = players_by_key[key].position.strip().upper()
        position = aliases.get(position, position)
        if position not in order:
            return None
        ranked.append((order[position], key))
    if ranked[0][0] == ranked[1][0]:
        return None
    return min(ranked)[1]
