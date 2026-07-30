"""The one writer that may replace a recorded figure with a derived one.

``hands.pot_size`` and ``hands.hero_bb_won`` are OBSERVED-fact columns. The
accounting cross-check compares both against the derived ledger EXACTLY,
precisely because they are meant to be independent evidence of what the hero won,
and PLAN.md:679 ("Distinguish observed facts from user-entered settlement
assumptions") is the constraint the whole dependence rule serves.

The settlement editor offers "Replace observed final pot/result with the derived
ledger values" for a real workflow: on a freshly reconstructed hand both columns
are null, the operator has verified the action line and the winners, and the
derived ledger is the best record of the summary. That is legitimate exactly when
the derivation is established, and the gate for "established" is the same one
every other consumer of a derived figure reads --
``study_readiness.accounting_is_established``.

The editor used ``reconciled.ledger.is_settled`` instead, which is strictly
weaker, and the two diverge on exactly the hands the dependence rule exists for:

* a reconstructed hand recording nothing, with a declared 25% rake, wrote the
  declaration-derived +20 into ``hands.hero_bb_won`` when the honest,
  declaration-free result its action line supports is +40. Study correctly refused
  the hand (ACCOUNTING_ASSUMPTION_DEPENDENT, ``accounting_is_established`` False),
  so ``math.analytics`` declined to substitute the derived result -- and fell back
  to ``hand.hero_bb_won``, which this path had just overwritten WITH that derived
  result. The figure was then published as an OBSERVED result, and counted in
  ``observed_result_count``, on a hand Study refused for the reason that made the
  figure unproven;
* a hand carrying GENUINE observations -- ``pot_size`` 80, ``hero_bb_won`` +40 --
  with 75 chips of declared dead money had both replaced by 155 and +115: two real
  measurements overwritten with numbers containing 75 chips nothing observed, from
  a declaration the operator could not have attested to, because the measurement
  does not exist until the same save creates it.

The contamination survives export (the columns are exported as observed facts)
while the attestation is reset on import, so it is portable as well as permanent.
``_hands_with_accounting_results`` marks its display copies
``derived_result_substituted`` so ``db._refuse_display_copy`` can reject them; this
path took the raw floats and bypassed that layer entirely.

Living here rather than in ``app.py`` is the point. ``hand_accounting`` cannot
import ``study_readiness`` (the dependence is measured in the former and answered
against the hand's evidence in the latter), and the previous nine rounds each
repaired a UI call site and found the next one, so the decision is taken once, in
a service, and the UI has no arguments to get wrong.
"""

from __future__ import annotations

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.services.hand_accounting import (
    AccountingReconciliation,
    persist_reconciliation,
)
from poker_tracker.services.study_readiness import (
    accounting_is_established,
    unattested_assumption_dependence,
)

REFUSED_UNESTABLISHED = (
    "The reconciled ledger is not established by this hand's records, so its "
    "figures cannot replace the recorded pot and hero result."
)


class SettlementSyncRefused(ValueError):
    """The derived ledger may not be recorded as this hand's observed summary."""


def sync_recorded_figures_from_ledger(
    db: PokerDatabase, hand_id: int
) -> AccountingReconciliation:
    """Record the derived gross pot and hero result as the hand's observed summary.

    Refuses, rather than writing a weaker figure, whenever the reconciliation is
    not established: an unbalanced or illegal ledger, or a measured settlement
    dependence this hand still owes an attestation for. The refusal names the
    assumption, because the operator's next action is either to confirm it or to
    correct the declaration -- both in the panel they are already looking at.

    Returns the re-derived reconciliation, so the caller reports what the save
    achieved from the same records the write produced.
    """
    reconciled = persist_reconciliation(db, hand_id)
    hand = db.fetch_hand(hand_id)
    if hand is None:
        raise ValueError("Hand not found.")
    if not accounting_is_established(hand, reconciled):
        pending = unattested_assumption_dependence(hand, reconciled)
        if pending:
            names = ", ".join(dependence.input_name for dependence in pending)
            raise SettlementSyncRefused(
                f"{REFUSED_UNESTABLISHED} It reconciles only under settlement "
                f"inputs you declared ({names}). Press 'Confirm this assumption' "
                "beside each one, or correct the declaration, and save again."
            )
        raise SettlementSyncRefused(
            f"{REFUSED_UNESTABLISHED} Reconcile a legal, balanced ledger first."
        )
    players = db.fetch_players_by_hand(hand_id)
    hero = next((player for player in players if player.is_hero), None)
    db.update_hand_accounting_evidence(
        hand_id,
        pot_size=reconciled.ledger.gross_pot,
        hero_bb_won=(
            None if hero is None else reconciled.ledger.net_results.get(hero.player_key)
        ),
    )
    return persist_reconciliation(db, hand_id)
