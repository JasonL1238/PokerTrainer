"""Regressions for the hand-history renderer's handling of UNKNOWN money fields.

The Option-A reader contract makes None a routine value for every money field
(10 of 31 current development hands carry a None starting_stack), and the
renderer predated that contract on every field except action amounts: a None
starting stack raised TypeError at the seat line, and main()'s single join then
aborted rendering of EVERY hand in the file, including the renderable ones.
"""
from cv_lab.scripts.pipeline.generate_hand_history import render_hand


def _hand(**overrides):
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "dealer_seat": 4,
        "hero": ["As", "Kd"],
        "board": ["2c", "7d", "9h"],
        "hero_folded": False,
        "players": [
            {"seat": 0, "position": "SB", "player_name": "Hero",
             "starting_stack": 100.0, "is_hero": True},
            {"seat": 4, "position": "BB", "player_name": "Seat4",
             "starting_stack": 100.0, "is_hero": False},
        ],
        "actions": [
            {"street": "preflop", "seat": 0, "action_type": "call", "amount": 0.5},
            {"street": "flop", "seat": 0, "action_type": "bet", "amount": None},
        ],
        "pot": 20.0,
        "winner_seat": 0,
    }
    hand.update(overrides)
    return hand


def test_an_unknown_starting_stack_renders_as_unknown_not_a_crash():
    """THE ROUND-2 B5 REGRESSION: `_fmt(p['starting_stack'])` raised TypeError
    on the None the contract now routinely produces."""
    hand = _hand()
    hand["players"][1]["starting_stack"] = None
    out = render_hand(hand, "v00")
    assert "Seat 4: Seat4 (stack unread)" in out
    # An unknown live stack makes the effective stack unknown: no SPR may be
    # computed from the known subset.
    assert "SPR" not in out
    # ... while the action-amount branches keep their existing behaviour.
    assert "bets (amount unread)" in out


def test_known_stacks_still_render_with_spr():
    out = render_hand(_hand(), "v00")
    assert "Seat 4: Seat4 (100bb in chips)" in out
    assert "SPR" in out


def test_a_winner_with_an_unknown_pot_renders_as_unknown():
    hand = _hand(pot=None)
    out = render_hand(hand, "v00")
    assert "collected (pot unread)" in out
    assert "Total pot" not in out
