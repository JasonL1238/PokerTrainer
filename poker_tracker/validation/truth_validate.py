"""Structural validation for Phase 2 answer keys.

Arithmetic is a cross-check only: when actions are incomplete or marked
uncertain/unobservable, pot conservation is skipped rather than forced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from poker_tracker.math.cards import RANKS, SUITS, parse_card
from poker_tracker.validation.schemas import (
    AMOUNT_SEMANTICS,
    ANSWER_KEY_SCHEMA_VERSION,
    SCORABLE_HAND_FACTS,
    VALID_ACTIONS,
    VALID_COMPLETION_CLASSES,
    VALID_STREETS,
    VALID_TERMINAL_EVENTS,
)


@dataclass(frozen=True)
class TruthIssue:
    path: str
    message: str


@dataclass
class TruthValidationResult:
    ok: bool
    issues: list[TruthIssue] = field(default_factory=list)
    completed_hands: int = 0
    partial_hands: int = 0
    uncertain_hands: int = 0

    def add(self, path: str, message: str) -> None:
        self.issues.append(TruthIssue(path=path, message=message))
        self.ok = False


def _is_card_label(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 2:
        return False
    return value[0] in RANKS and value[1] in SUITS


def _card_uniqueness(cards: list[str], *, path: str, result: TruthValidationResult) -> None:
    seen: set[str] = set()
    for index, card in enumerate(cards):
        if card in seen:
            result.add(f"{path}[{index}]", f"duplicate card {card}")
        seen.add(card)


def _validate_action(
    action: dict[str, Any],
    *,
    path: str,
    dealt_in: set[int],
    result: TruthValidationResult,
) -> None:
    street = action.get("street")
    if street not in VALID_STREETS:
        result.add(f"{path}.street", f"invalid street {street!r}")
    order = action.get("order")
    if not isinstance(order, int) or order < 1:
        result.add(f"{path}.order", "order must be a positive integer")
    seat = action.get("seat")
    if not isinstance(seat, int):
        result.add(f"{path}.seat", "seat must be an integer")
    elif dealt_in and seat not in dealt_in:
        result.add(f"{path}.seat", f"seat {seat} is not in dealt_in_seats")
    kind = action.get("action")
    if kind not in VALID_ACTIONS:
        result.add(f"{path}.action", f"invalid action {kind!r}")
    if "certain" not in action or not isinstance(action.get("certain"), bool):
        result.add(f"{path}.certain", "certain must be a boolean")
    if "observable" in action and not isinstance(action.get("observable"), bool):
        result.add(f"{path}.observable", "observable must be a boolean when present")
    semantics = action.get("amount_semantics", "incremental")
    if semantics not in AMOUNT_SEMANTICS:
        result.add(f"{path}.amount_semantics", f"invalid amount_semantics {semantics!r}")
    amount = action.get("amount")
    if amount is not None and not isinstance(amount, (int, float)):
        result.add(f"{path}.amount", "amount must be a number or null")
    if kind in {"bet", "raise", "call", "all-in", "post_blind", "ante"}:
        if action.get("certain") is True and action.get("observable", True) is True:
            if amount is None and semantics != "unknown":
                result.add(
                    f"{path}.amount",
                    "certain observable money action requires amount or amount_semantics=unknown",
                )
    if kind in {"fold", "check"} and amount not in (None, 0, 0.0):
        result.add(f"{path}.amount", f"{kind} actions must not carry a positive amount")


def _action_legality(
    actions: list[dict[str, Any]],
    *,
    path: str,
    result: TruthValidationResult,
) -> None:
    """Lightweight legality: order uniqueness per street and no act-after-fold."""
    folded: set[int] = set()
    orders_by_street: dict[str, set[int]] = {}
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        street = str(action.get("street") or "")
        order = action.get("order")
        if isinstance(order, int):
            seen = orders_by_street.setdefault(street, set())
            if order in seen:
                result.add(f"{path}[{index}].order", f"duplicate order {order} on {street}")
            seen.add(order)
        seat = action.get("seat")
        kind = action.get("action")
        if isinstance(seat, int) and seat in folded and kind not in {"show", "win"}:
            result.add(f"{path}[{index}]", f"seat {seat} acts after folding")
        if kind == "fold" and isinstance(seat, int):
            folded.add(seat)
        if kind == "check" and any(
            isinstance(prior.get("action"), str)
            and prior.get("action") in {"bet", "raise", "all-in"}
            and prior.get("street") == street
            and prior.get("certain") is True
            for prior in actions[:index]
            if isinstance(prior, dict)
        ):
            # Facing a certain bet/raise on this street, a check is illegal.
            # Calls/folds are fine; this is a structural annotation guard only.
            facing = True
            # Allow check only if that bet was from the same seat somehow — no.
            if facing and action.get("certain") is True:
                # Only flag when a different seat already put in a certain bet.
                if any(
                    isinstance(prior.get("seat"), int)
                    and prior.get("seat") != seat
                    and prior.get("action") in {"bet", "raise", "all-in"}
                    and prior.get("street") == street
                    and prior.get("certain") is True
                    for prior in actions[:index]
                    if isinstance(prior, dict)
                ):
                    result.add(
                        f"{path}[{index}]",
                        "certain check after a certain bet/raise on the same street",
                    )


def _arithmetic_cross_check(
    hand: dict[str, Any],
    *,
    path: str,
    result: TruthValidationResult,
) -> None:
    """When the action line is complete, certain, and incremental, contributed
    chips should land near final_pot (rake/uncalled slack allowed).

    ``amount_semantics: "to"`` lines are excluded from the sum: converting
    to-amounts into incremental contributions needs street state the answer key
    does not yet carry. Mixing ``to`` into an incremental sum false-fails legal
    keys, so those hands skip this particular cross-check rather than invent
    a conversion.
    """
    if hand.get("actions_complete") is not True:
        return
    if hand.get("completion_class") != "complete":
        return
    final_pot = hand.get("final_pot")
    if not isinstance(final_pot, (int, float)):
        return
    if float(final_pot) < 0:
        result.add(f"{path}.final_pot", "final_pot must not be negative")
        return
    actions = hand.get("actions") or []
    if not isinstance(actions, list):
        return
    contributed = 0.0
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("certain") is not True:
            return
        if action.get("observable", True) is not True:
            return
        semantics = action.get("amount_semantics", "incremental")
        if semantics == "to":
            return
        if semantics == "unknown":
            return
        kind = action.get("action")
        amount = action.get("amount")
        if kind in {"bet", "raise", "call", "all-in", "post_blind", "ante"}:
            if not isinstance(amount, (int, float)):
                return
            if float(amount) < 0:
                result.add(f"{path}.actions", "money action amount must not be negative")
                return
            contributed += float(amount)
    slack = 3.0
    if contributed > float(final_pot) + slack:
        result.add(
            f"{path}.final_pot",
            f"certain complete action contributions {contributed:.2f} exceed "
            f"final_pot {float(final_pot):.2f} by more than slack",
        )
    elif float(final_pot) > contributed + slack:
        result.add(
            f"{path}.final_pot",
            f"final_pot {float(final_pot):.2f} exceeds certain incremental "
            f"contributions {contributed:.2f} by more than slack",
        )


def _street_order_check(
    actions: list[dict[str, Any]],
    *,
    path: str,
    result: TruthValidationResult,
) -> None:
    street_rank = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}
    last_rank = -1
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        street = action.get("street")
        rank = street_rank.get(str(street), -1)
        if rank < 0:
            continue
        if rank < last_rank:
            result.add(
                f"{path}[{index}].street",
                f"street {street!r} appears after a later street",
            )
        last_rank = max(last_rank, rank)


def _fact_is_supplied(hand: dict[str, Any], fact: str) -> bool:
    """Whether the answer key actually annotated this fact.

    ``final_board: []`` is a real annotation (the board never came), so
    emptiness is not absence — only ``None``/missing is.
    """
    return hand.get(fact) is not None


def _observability_check(
    hand: dict[str, Any],
    *,
    path: str,
    result: TruthValidationResult,
) -> None:
    """Every scored fact is annotated or explicitly named unobservable.

    Schema v1 let a ``null`` critical fact mean "skip this check", so a hand
    nobody finished annotating scored exactly like a hand where the fact was
    genuinely off-camera. v2 makes the annotator say which one it is: silence is
    a corpus defect, not a free pass.
    """
    declared = hand.get("unobservable", [])
    if not isinstance(declared, list):
        result.add(f"{path}.unobservable", "unobservable must be a list of fact names")
        return
    seen: set[str] = set()
    for index, fact in enumerate(declared):
        if not isinstance(fact, str):
            result.add(f"{path}.unobservable[{index}]", "fact name must be a string")
            continue
        if fact not in SCORABLE_HAND_FACTS:
            result.add(
                f"{path}.unobservable[{index}]",
                f"unknown scored fact {fact!r}; expected one of {sorted(SCORABLE_HAND_FACTS)}",
            )
            continue
        if fact in seen:
            result.add(f"{path}.unobservable[{index}]", f"duplicate fact {fact!r}")
        seen.add(fact)
        if _fact_is_supplied(hand, fact):
            result.add(
                f"{path}.unobservable[{index}]",
                f"{fact} is declared unobservable but carries a value",
            )
    for fact in SCORABLE_HAND_FACTS:
        if fact in seen or _fact_is_supplied(hand, fact):
            continue
        result.add(
            f"{path}.{fact}",
            f"{fact} has no value and is not listed in unobservable; "
            "annotate it or declare it explicitly",
        )
    if isinstance(hand.get("result"), str) and not hand["result"].strip():
        result.add(f"{path}.result", "result must be a non-empty string when observable")


def _validate_hand(
    hand: dict[str, Any],
    *,
    path: str,
    result: TruthValidationResult,
) -> None:
    hand_id = hand.get("hand_id")
    if not isinstance(hand_id, str) or not hand_id.strip():
        result.add(f"{path}.hand_id", "hand_id is required")

    for key in ("t_first", "t_last"):
        value = hand.get(key)
        if not isinstance(value, (int, float)):
            result.add(f"{path}.{key}", f"{key} must be a number")
    if (
        isinstance(hand.get("t_first"), (int, float))
        and isinstance(hand.get("t_last"), (int, float))
        and float(hand["t_last"]) < float(hand["t_first"])
    ):
        result.add(f"{path}.t_last", "t_last must be >= t_first")

    for key in ("partial_start", "partial_end", "actions_complete"):
        if key in hand and not isinstance(hand.get(key), bool):
            result.add(f"{path}.{key}", f"{key} must be a boolean")

    completion = hand.get("completion_class")
    if completion not in VALID_COMPLETION_CLASSES:
        result.add(f"{path}.completion_class", f"invalid completion_class {completion!r}")
    else:
        if completion == "complete":
            if hand.get("actions_complete") is not True:
                result.add(
                    f"{path}.actions_complete",
                    "completion_class=complete requires actions_complete=true",
                )
            else:
                result.completed_hands += 1
        elif completion == "partial":
            result.partial_hands += 1
        else:
            result.uncertain_hands += 1

    terminal = hand.get("terminal_event")
    if terminal not in VALID_TERMINAL_EVENTS:
        result.add(f"{path}.terminal_event", f"invalid terminal_event {terminal!r}")

    if hand.get("partial_start") or hand.get("partial_end"):
        if completion == "complete":
            result.add(
                f"{path}.completion_class",
                "partial_start/partial_end hands cannot be completion_class=complete",
            )

    dealt_raw = hand.get("dealt_in_seats")
    dealt_in: set[int] = set()
    if not isinstance(dealt_raw, list) or not dealt_raw:
        result.add(f"{path}.dealt_in_seats", "dealt_in_seats must be a non-empty list")
    else:
        for index, seat in enumerate(dealt_raw):
            if not isinstance(seat, int) or seat < 0:
                result.add(f"{path}.dealt_in_seats[{index}]", "seat must be a non-negative int")
            else:
                dealt_in.add(seat)

    dealer = hand.get("dealer_seat")
    if dealer is not None and not isinstance(dealer, int):
        result.add(f"{path}.dealer_seat", "dealer_seat must be an int or null")
    elif isinstance(dealer, int) and dealt_in and dealer not in dealt_in:
        result.add(f"{path}.dealer_seat", "dealer_seat must be one of dealt_in_seats")

    hero = hand.get("hero_cards")
    visible: list[str] = []
    if hero is not None:
        if not isinstance(hero, list) or len(hero) != 2:
            result.add(f"{path}.hero_cards", "hero_cards must be null or two cards")
        else:
            for index, card in enumerate(hero):
                if not _is_card_label(card):
                    result.add(f"{path}.hero_cards[{index}]", f"invalid card {card!r}")
                else:
                    try:
                        parse_card(card)
                    except Exception as exc:  # noqa: BLE001 - surface as truth issue
                        result.add(f"{path}.hero_cards[{index}]", str(exc))
                    visible.append(str(card))

    board = hand.get("final_board")
    declared_unobservable = hand.get("unobservable")
    board_declared = (
        isinstance(declared_unobservable, list) and "final_board" in declared_unobservable
    )
    if board is None:
        if not board_declared:
            # ``_observability_check`` owns the undeclared case; keep the older,
            # more specific hint for the common "board never came" mistake.
            result.add(
                f"{path}.final_board",
                "final_board is required (use [] when no board came, or declare it unobservable)",
            )
    elif not isinstance(board, list):
        result.add(f"{path}.final_board", "final_board must be a list")
    elif len(board) not in {0, 3, 4, 5}:
        result.add(f"{path}.final_board", "final_board length must be 0, 3, 4, or 5")
    else:
        for index, card in enumerate(board):
            if not _is_card_label(card):
                result.add(f"{path}.final_board[{index}]", f"invalid card {card!r}")
            else:
                visible.append(str(card))
    if visible:
        _card_uniqueness(visible, path=f"{path}.visible_cards", result=result)

    stacks = hand.get("starting_stacks")
    if stacks is not None:
        if not isinstance(stacks, dict):
            result.add(f"{path}.starting_stacks", "starting_stacks must be an object")
        else:
            for seat_key, amount in stacks.items():
                if not str(seat_key).isdigit():
                    result.add(f"{path}.starting_stacks", f"invalid seat key {seat_key!r}")
                elif amount is not None and not isinstance(amount, (int, float)):
                    result.add(
                        f"{path}.starting_stacks.{seat_key}",
                        "stack must be a number or null",
                    )

    actions = hand.get("actions")
    if not isinstance(actions, list):
        result.add(f"{path}.actions", "actions must be a list")
    else:
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                result.add(f"{path}.actions[{index}]", "action must be an object")
                continue
            _validate_action(
                action,
                path=f"{path}.actions[{index}]",
                dealt_in=dealt_in,
                result=result,
            )
        _action_legality(actions, path=f"{path}.actions", result=result)
        _street_order_check(actions, path=f"{path}.actions", result=result)

    for key in ("final_pot", "hero_net"):
        value = hand.get(key)
        if value is not None and not isinstance(value, (int, float)):
            result.add(f"{path}.{key}", f"{key} must be a number or null")
    winner = hand.get("winner_seat")
    if winner is not None and not isinstance(winner, int):
        result.add(f"{path}.winner_seat", "winner_seat must be an int or null")

    evidence = hand.get("evidence")
    if evidence is None:
        if completion == "complete":
            result.add(
                f"{path}.evidence",
                "complete hands require an evidence object with critical-fact timestamps",
            )
    elif not isinstance(evidence, dict):
        result.add(f"{path}.evidence", "evidence must be an object when present")
    elif completion == "complete":
        for key in ("hero_cards_t", "final_board_t", "final_pot_t"):
            if key not in evidence:
                result.add(f"{path}.evidence.{key}", f"complete hands require evidence.{key}")

    _observability_check(hand, path=path, result=result)
    _arithmetic_cross_check(hand, path=path, result=result)


def validate_answer_key(document: dict[str, Any]) -> TruthValidationResult:
    result = TruthValidationResult(ok=True)
    version = document.get("schema_version")
    if version != ANSWER_KEY_SCHEMA_VERSION:
        result.add(
            "schema_version",
            f"expected {ANSWER_KEY_SCHEMA_VERSION}, got {version!r}",
        )
    case_id = document.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        result.add("case_id", "case_id is required")

    annotation = document.get("annotation")
    if not isinstance(annotation, dict):
        result.add("annotation", "annotation object is required")
    else:
        passes = annotation.get("passes")
        if not isinstance(passes, int) or passes < 1:
            result.add("annotation.passes", "passes must be an integer >= 1")
        if not isinstance(annotation.get("adjudicated"), bool):
            result.add("annotation.adjudicated", "adjudicated must be a boolean")

    hands = document.get("hands")
    if not isinstance(hands, list) or not hands:
        result.add("hands", "hands must be a non-empty list")
        return result

    seen_ids: set[str] = set()
    for index, hand in enumerate(hands):
        path = f"hands[{index}]"
        if not isinstance(hand, dict):
            result.add(path, "hand must be an object")
            continue
        hand_id = hand.get("hand_id")
        if isinstance(hand_id, str):
            if hand_id in seen_ids:
                result.add(f"{path}.hand_id", f"duplicate hand_id {hand_id}")
            seen_ids.add(hand_id)
        _validate_hand(hand, path=path, result=result)
    return result
