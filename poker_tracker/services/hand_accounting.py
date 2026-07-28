"""Persistence-aware completed-hand accounting and reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from poker_tracker.math.accounting import (
    HandLedger,
    LedgerError,
    RakePolicy,
    build_ledger_from_records,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import HandSettlement, SettlementEntry, utc_now


@dataclass(frozen=True)
class AccountingReconciliation:
    ledger: HandLedger
    settlement: HandSettlement | None
    entries: tuple[SettlementEntry, ...]
    issues: tuple[str, ...]
    is_authoritative: bool


def reconcile_persisted_hand(
    db: PokerDatabase,
    hand_id: int,
) -> AccountingReconciliation:
    """Build and cross-check one completed hand from durable records."""

    hand = db.fetch_hand(hand_id)
    if hand is None:
        raise LedgerError(f"Hand {hand_id} does not exist.")
    players = db.fetch_players_by_hand(hand_id)
    actions = db.fetch_actions_by_hand(hand_id)
    settlement = db.fetch_hand_settlement(hand_id)
    entries = tuple(db.fetch_settlement_entries(hand_id))

    awards = [entry for entry in entries if entry.entry_type == "award"]
    refunds = [entry for entry in entries if entry.entry_type == "refund"]
    winners: dict[int, list[str]] | None = None
    odd_chip_order: list[str] = []
    if awards:
        winners = {}
        for entry in sorted(awards, key=lambda row: (row.pot_index or 0, row.entry_order)):
            identity = _entry_identity(entry, players)
            winners.setdefault(entry.pot_index or 0, []).append(identity)
            if identity not in odd_chip_order:
                odd_chip_order.append(identity)

    rake = RakePolicy()
    dead_money = 0.0
    if settlement is not None:
        dead_money = settlement.dead_money
        rake = RakePolicy(
            rate=settlement.rake_rate,
            cap=settlement.rake_cap,
            rounding_unit=settlement.rake_rounding_unit,
            no_flop_no_drop=settlement.no_flop_no_drop,
        )

    ledger = build_ledger_from_records(
        players,
        actions,
        dead_money=dead_money,
        winners=winners,
        rake=rake,
        odd_chip_order=odd_chip_order,
        flop_seen=bool(hand.board_cards)
        or any(action.street in {"flop", "turn", "river", "showdown"} for action in actions),
    )
    issues = [*ledger.warnings, *ledger.legality_issues]
    tolerance = max((settlement.rake_rounding_unit if settlement else 0.01) / 2, 1e-9)

    if settlement is None:
        issues.append("No persisted settlement assumptions or awards.")
    else:
        _compare_optional(
            issues, "gross pot", settlement.gross_pot, ledger.gross_pot, tolerance
        )
        _compare_optional(
            issues, "rake", settlement.rake_amount, ledger.rake, tolerance
        )
        _compare_optional(
            issues, "net pot", settlement.net_pot, ledger.net_pot, tolerance
        )
        if settlement.status == "reconciled" and not ledger.is_settled:
            issues.append("Settlement is marked reconciled but not every pot has a winner.")
        if settlement.is_balanced and not ledger.is_balanced:
            issues.append("Settlement is marked balanced but chip conservation fails.")

    if hand.pot_size is not None and not isclose(
        hand.pot_size, ledger.gross_pot, abs_tol=tolerance
    ):
        issues.append("Observed final pot does not match the derived gross pot.")
    hero_players = [player for player in players if player.is_hero]
    if len(hero_players) > 1:
        issues.append("More than one player is marked as Hero.")
    elif len(hero_players) == 1 and hand.hero_bb_won is not None and ledger.is_settled:
        hero_result = ledger.net_results.get(hero_players[0].player_key)
        if hero_result is None or not isclose(
            hand.hero_bb_won, hero_result, abs_tol=tolerance
        ):
            issues.append("Observed Hero result does not match the derived ledger result.")

    declared_refunds: dict[str, float] = {}
    for entry in refunds:
        identity = _entry_identity(entry, players)
        declared_refunds[identity] = declared_refunds.get(identity, 0) + (entry.amount or 0)
    if refunds:
        for identity in set(declared_refunds) | set(ledger.refunds):
            if not isclose(
                declared_refunds.get(identity, 0),
                ledger.refunds.get(identity, 0),
                abs_tol=tolerance,
            ):
                issues.append(
                    f"Declared refund for {identity!r} does not match the derived refund."
                )

    declared_awards: dict[str, float] = {}
    if awards and all(entry.amount is not None for entry in awards):
        for entry in awards:
            identity = _entry_identity(entry, players)
            declared_awards[identity] = declared_awards.get(identity, 0) + (
                entry.amount or 0
            )
        for identity in set(declared_awards) | set(ledger.payouts):
            if not isclose(
                declared_awards.get(identity, 0),
                ledger.payouts.get(identity, 0),
                abs_tol=tolerance,
            ):
                issues.append(
                    f"Declared awards for {identity!r} do not match derived payouts."
                )

    is_authoritative = (
        settlement is not None
        and settlement.status == "reconciled"
        and ledger.is_settled
        and ledger.is_balanced
        and ledger.is_legal
        and not issues
    )
    return AccountingReconciliation(
        ledger=ledger,
        settlement=settlement,
        entries=entries,
        issues=tuple(dict.fromkeys(issues)),
        is_authoritative=is_authoritative,
    )


def persist_reconciliation(
    db: PokerDatabase,
    hand_id: int,
    *,
    status_when_valid: str = "reconciled",
) -> AccountingReconciliation:
    """Recompute summaries and persist a truthful reconciliation status."""

    result = reconcile_persisted_hand(db, hand_id)
    if not any(entry.entry_type == "refund" for entry in result.entries):
        players_by_key = {
            player.player_key: player for player in db.fetch_players_by_hand(hand_id)
        }
        derived_refunds = [
            SettlementEntry(
                hand_id=hand_id,
                entry_type="refund",
                player_key=identity,
                player_name=players_by_key[identity].player_name,
                amount=amount,
                entry_order=index,
            )
            for index, (identity, amount) in enumerate(
                (
                    (identity, amount)
                    for identity, amount in result.ledger.refunds.items()
                    if amount > 0
                ),
                start=1,
            )
            if identity in players_by_key
        ]
        if derived_refunds:
            db.replace_settlement_entries(
                hand_id, [*result.entries, *derived_refunds]
            )
            result = reconcile_persisted_hand(db, hand_id)
    existing = result.settlement or HandSettlement(hand_id=hand_id)
    valid = (
        result.ledger.is_settled
        and result.ledger.is_balanced
        and result.ledger.is_legal
        and not result.issues
    )
    status = status_when_valid if valid else "needs_correction"
    updated = existing.model_copy(
        update={
            "status": status,
            "gross_pot": result.ledger.gross_pot,
            "rake_amount": result.ledger.rake,
            "net_pot": result.ledger.net_pot,
            "is_balanced": result.ledger.is_balanced,
            "warnings": list(result.issues),
            "updated_at": utc_now(),
        }
    )
    db.upsert_hand_settlement(updated)
    return reconcile_persisted_hand(db, hand_id)


def _entry_identity(entry: SettlementEntry, players: list) -> str:
    if entry.player_key is not None:
        if not any(player.player_key == entry.player_key for player in players):
            raise LedgerError(
                f"Settlement entry references unknown player key {entry.player_key!r}."
            )
        return entry.player_key
    matches = [player.player_key for player in players if player.player_name == entry.player_name]
    if len(matches) != 1:
        raise LedgerError(
            f"Settlement entry cannot resolve player {entry.player_name!r} "
            "to one stable identity."
        )
    return matches[0]


def _compare_optional(
    issues: list[str],
    label: str,
    recorded: float | None,
    derived: float,
    tolerance: float,
) -> None:
    if recorded is not None and not isclose(recorded, derived, abs_tol=tolerance):
        issues.append(f"Recorded {label} does not match the derived ledger.")
