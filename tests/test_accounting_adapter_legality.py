from __future__ import annotations

import pytest

from poker_tracker.math.accounting import LedgerError, build_ledger_from_records
from poker_tracker.persistence.models import Action, HandPlayer


def _player(
    key: str,
    *,
    name: str | None = None,
    stack: float = 100,
    seat: int | None = None,
) -> HandPlayer:
    return HandPlayer(
        hand_id=1,
        player_key=key,
        seat_index=seat,
        player_name=name or key,
        starting_stack=stack,
    )


def _action(
    key: str | None,
    kind: str,
    amount: float | None = None,
    *,
    name: str | None = None,
    street: str = "preflop",
    semantics: str = "incremental",
    is_live_post: bool | None = None,
) -> Action:
    return Action(
        hand_id=1,
        player_key=key,
        player_name=name or key or "Alex",
        street=street,
        action_type=kind,
        amount=amount,
        amount_semantics=semantics,
        is_live_post=is_live_post,
    )


def _assert_balanced_but_illegal(ledger, issue_text: str) -> None:
    assert ledger.is_settled is True
    assert ledger.is_balanced is True
    assert ledger.is_legal is False
    assert any(issue_text in issue.lower() for issue in ledger.legality_issues)
    assert sum(ledger.net_results.values()) + ledger.rake == pytest.approx(0)


def test_duplicate_display_names_are_resolved_by_stable_player_key() -> None:
    players = [
        _player("seat-2", name="Alex", seat=2),
        _player("seat-7", name="Alex", seat=7),
    ]
    ledger = build_ledger_from_records(
        players,
        [
            _action("seat-2", "bet", 10, name="Alex", street="river"),
            _action("seat-7", "call", 10, name="Alex", street="river"),
        ],
        winners={0: ("seat-2",)},
    )

    assert ledger.contributions == pytest.approx({"seat-2": 10, "seat-7": 10})
    assert ledger.payouts == pytest.approx({"seat-2": 20, "seat-7": 0})
    assert ledger.is_balanced is True
    assert ledger.is_legal is True


def test_duplicate_display_name_without_player_key_is_rejected() -> None:
    players = [
        _player("seat-2", name="Alex", seat=2),
        _player("seat-7", name="Alex", seat=7),
    ]

    with pytest.raises(LedgerError, match="stable identity"):
        build_ledger_from_records(
            players,
            [_action(None, "check", name="Alex", street="flop")],
        )


@pytest.mark.parametrize(
    ("semantics", "raise_amount", "call_amount"),
    [
        ("incremental", 3, 2),
        ("raise_to", 3, 3),
    ],
)
def test_incremental_and_raise_to_records_normalize_to_same_commitments(
    semantics: str,
    raise_amount: float,
    call_amount: float,
) -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B")],
        [
            _action("A", "post_blind", 1),
            _action("B", "raise", raise_amount, semantics=semantics),
            _action("A", "call", call_amount, semantics=semantics),
        ],
        winners={0: ("A",)},
    )

    assert [snapshot.amount for snapshot in ledger.snapshots] == pytest.approx([1, 3, 2])
    assert ledger.contributions == pytest.approx({"A": 3, "B": 3})
    assert ledger.net_results == pytest.approx({"A": 3, "B": -3})
    assert ledger.is_balanced is True
    assert ledger.is_legal is True


def test_unknown_monetary_amount_semantics_are_rejected() -> None:
    with pytest.raises(LedgerError, match="unknown amount semantics"):
        build_ledger_from_records(
            [_player("A"), _player("B")],
            [_action("A", "bet", 10, street="river", semantics="unknown")],
        )


def test_unknown_semantics_on_nonmonetary_legacy_evidence_are_ignored() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B")],
        [
            _action("A", "check", street="river", semantics="unknown"),
            # Historical win rows may contain a reported result, but that
            # number is not treated as a contribution or settlement award.
            _action("A", "win", 12, street="showdown", semantics="unknown"),
        ],
    )

    assert ledger.contributions == pytest.approx({"A": 0, "B": 0})
    assert ledger.payouts == pytest.approx({"A": 0, "B": 0})
    assert ledger.is_legal is True


def test_dead_blind_adds_to_the_pot_without_changing_amount_to_call() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B"), _player("C")],
        [
            _action("A", "post_blind", 1, is_live_post=True),
            _action("B", "post_blind", 2, is_live_post=True),
            _action("C", "post_blind", 5, is_live_post=False),
            _action("A", "call", 1),
        ],
    )

    dead_post = ledger.snapshots[2]
    call = ledger.snapshots[3]
    assert dead_post.pot_after == pytest.approx(8)
    assert dead_post.street_contribution_after == pytest.approx(0)
    assert dead_post.hand_contribution_after == pytest.approx(5)
    assert call.to_call_before == pytest.approx(1)
    assert call.call_increment == pytest.approx(1)


def test_dead_blind_does_not_reduce_a_later_raise_to_amount() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B")],
        [
            _action("A", "post_blind", 5, is_live_post=False),
            _action("B", "post_blind", 2, is_live_post=True),
            _action("A", "raise", 10, semantics="raise_to"),
            _action("B", "call", 8),
        ],
        winners={0: ("A",)},
    )

    assert [snapshot.amount for snapshot in ledger.snapshots] == pytest.approx(
        [5, 2, 10, 8]
    )
    assert ledger.contributions == pytest.approx({"A": 15, "B": 10})
    assert ledger.gross_pot == pytest.approx(20)
    assert ledger.refunds == pytest.approx({"A": 5, "B": 0})
    assert ledger.is_legal is True


def test_live_blind_changes_amount_to_call() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B"), _player("C")],
        [
            _action("A", "post_blind", 1, is_live_post=True),
            _action("B", "post_blind", 2, is_live_post=True),
            _action("C", "post_blind", 5, is_live_post=True),
            _action("A", "call", 4),
        ],
    )

    live_post = ledger.snapshots[2]
    call = ledger.snapshots[3]
    assert live_post.street_contribution_after == pytest.approx(5)
    assert call.to_call_before == pytest.approx(4)
    assert call.call_increment == pytest.approx(4)


def test_check_while_facing_a_bet_is_illegal_but_chips_still_balance() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B")],
        [
            _action("A", "bet", 10, street="flop"),
            _action("B", "check", street="flop"),
            _action("B", "call", 10, street="flop"),
        ],
        winners={0: ("A",)},
    )

    _assert_balanced_but_illegal(ledger, "check while facing")


def test_mismatched_call_is_illegal_but_chips_still_balance() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B")],
        [
            _action("A", "bet", 10, street="river"),
            _action("B", "call", 8, street="river"),
        ],
        winners={0: ("A",)},
    )

    assert ledger.refunds == pytest.approx({"A": 2, "B": 0})
    _assert_balanced_but_illegal(ledger, "amount to call")


def test_short_all_in_call_is_legal() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B", stack=8)],
        [
            _action("A", "bet", 10, street="river"),
            _action("B", "call", 8, street="river"),
        ],
        winners={0: ("A",)},
    )

    assert ledger.refunds == pytest.approx({"A": 2, "B": 0})
    assert ledger.snapshots[1].to_call_before == pytest.approx(10)
    assert ledger.snapshots[1].stack_after == pytest.approx(0)
    assert ledger.is_balanced is True
    assert ledger.is_legal is True
    assert ledger.legality_issues == ()


def test_below_minimum_regular_raise_is_illegal() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B")],
        [
            _action("A", "post_blind", 1),
            _action("B", "post_blind", 2),
            _action("A", "raise", 5),
            _action("B", "raise", 7),
            _action("A", "call", 3),
        ],
        winners={0: ("A",)},
    )

    _assert_balanced_but_illegal(ledger, "below the minimum full raise")


def test_short_all_in_raise_is_legal_but_does_not_reopen_full_raise_size() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B", stack=9)],
        [
            _action("A", "post_blind", 1),
            _action("B", "post_blind", 2),
            _action("A", "raise", 5),
            _action("B", "all-in", 7),
            _action("A", "call", 3),
        ],
        winners={0: ("A",)},
    )

    assert ledger.contributions == pytest.approx({"A": 9, "B": 9})
    assert ledger.snapshots[3].stack_after == pytest.approx(0)
    assert ledger.is_balanced is True
    assert ledger.is_legal is True
    assert ledger.legality_issues == ()


def test_short_all_in_raise_does_not_reopen_betting_to_a_prior_raiser() -> None:
    ledger = build_ledger_from_records(
        [_player("A", stack=100), _player("B", stack=7)],
        [
            _action("B", "post_blind", 2),
            _action("A", "raise", 6),
            _action("B", "all-in", 5),
            _action("A", "raise", 5),
        ],
        winners={0: ("B",)},
    )

    assert ledger.refunds == pytest.approx({"A": 4, "B": 0})
    _assert_balanced_but_illegal(ledger, "not reopened after a short all-in")


def test_cumulative_short_all_ins_can_reopen_betting_by_a_full_raise() -> None:
    ledger = build_ledger_from_records(
        [
            _player("A", stack=100),
            _player("B", stack=15),
            _player("C", stack=20),
        ],
        [
            _action("A", "bet", 10, street="flop"),
            _action("B", "all-in", 15, street="flop"),
            _action("C", "all-in", 20, street="flop"),
            _action("A", "raise", 20, street="flop"),
        ],
        winners={0: ("B",), 1: ("C",)},
    )

    assert ledger.refunds == pytest.approx({"A": 10, "B": 0, "C": 0})
    assert ledger.is_legal is True
    assert ledger.legality_issues == ()


def test_action_after_all_in_is_illegal_but_chips_still_balance() -> None:
    ledger = build_ledger_from_records(
        [_player("A", stack=10), _player("B", stack=10)],
        [
            _action("A", "all-in", 10, street="river"),
            _action("B", "call", 10, street="river"),
            _action("A", "check", street="river"),
        ],
        winners={0: ("B",)},
    )

    _assert_balanced_but_illegal(ledger, "acts after being all-in")


def test_backward_street_is_illegal_but_chips_still_balance() -> None:
    ledger = build_ledger_from_records(
        [_player("A"), _player("B")],
        [
            _action("A", "bet", 5, street="turn"),
            _action("B", "call", 5, street="turn"),
            _action("A", "check", street="flop"),
        ],
        winners={0: ("A",)},
    )

    _assert_balanced_but_illegal(ledger, "street order moves backward")
