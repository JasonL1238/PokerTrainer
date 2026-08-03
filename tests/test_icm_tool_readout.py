"""The ICM screen may not state one premium as if it were the only answer.

``icm_risk_premium`` used to delete the risked chips from the table instead of
transferring them, which returned a figure below every outcome the tournament
can produce. That is fixed in the module, but the screen still rendered a single
number under help text reading "Prize equity lost if Hero loses this many
chips" -- and the figure it now renders is the WORST single-winner case, so the
prose asserts as universal a number that is true only against one opponent.
This pins the screen to the span the module can actually defend, and to the
admission that a multiway split is not bounded by it.
"""

from __future__ import annotations

import itertools
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

# The tool's own default screen.
DEFAULT_STACKS = [5000.0, 3000.0, 2000.0]
DEFAULT_PAYOUTS = [50.0, 30.0, 20.0]


def _equity_by_enumeration(stacks: list[float], payouts: list[float], hero: int) -> float:
    """Malmuth-Harville by brute-force permutation, independent of the module."""

    n = len(stacks)
    prizes = list(payouts) + [0.0] * (n - len(payouts))
    total = 0.0
    for order in itertools.permutations(range(n)):
        probability = 1.0
        alive = set(range(n))
        for player in order:
            chips = sum(stacks[i] for i in alive)
            probability *= stacks[player] / chips
            alive.discard(player)
        total += probability * prizes[order.index(hero)]
    return total


def _true_single_winner_premiums(
    stacks: list[float], payouts: list[float], hero: int, risk: float
) -> dict[int, float]:
    before = _equity_by_enumeration(stacks, payouts, hero)
    premiums = {}
    for opponent in range(len(stacks)):
        if opponent == hero:
            continue
        after = list(stacks)
        after[hero] -= risk
        after[opponent] += risk
        premiums[opponent] = before - _equity_by_enumeration(after, payouts, hero)
    return premiums


def _readout(stacks, payouts, hero_index, risk):
    import app as app_module

    return app_module._icm_risk_premium_readout(stacks, payouts, hero_index, risk)


def test_the_readout_is_the_span_over_who_wins_the_chips() -> None:
    truth = _true_single_winner_premiums(DEFAULT_STACKS, DEFAULT_PAYOUTS, 0, 1000.0)
    value, _help = _readout(DEFAULT_STACKS, DEFAULT_PAYOUTS, 0, 1000.0)

    low, high = min(truth.values()), max(truth.values())
    assert low != high, "this fixture is only meaningful when the answer varies"
    assert value == f"{low:.2f} – {high:.2f}"
    # And that span is the real one, not a restatement of whatever the module says.
    assert value == "2.73 – 2.96"


def test_the_help_text_does_not_assert_one_figure_as_the_answer() -> None:
    _value, help_text = _readout(DEFAULT_STACKS, DEFAULT_PAYOUTS, 0, 1000.0)

    assert "by which opponent wins the chips" in help_text
    # The prose that made a single opponent's cost read as universal.
    assert "Prize equity lost if Hero loses this many chips" not in help_text
    # The span bounds single winners only; a split can land above it.
    assert "split between several opponents can cost more" in help_text


def test_the_span_names_the_seat_at_each_end() -> None:
    truth = _true_single_winner_premiums(DEFAULT_STACKS, DEFAULT_PAYOUTS, 0, 1000.0)
    cheapest = min(truth, key=lambda seat: truth[seat])
    dearest = max(truth, key=lambda seat: truth[seat])

    _value, help_text = _readout(DEFAULT_STACKS, DEFAULT_PAYOUTS, 0, 1000.0)

    assert f"Cheapest if player {cheapest + 1} wins the chips" in help_text
    assert f"dearest if player {dearest + 1} does" in help_text


def test_equal_stacks_collapse_to_one_number_rather_than_a_fake_range() -> None:
    stacks = [100.0, 100.0, 100.0, 100.0]
    value, help_text = _readout(stacks, DEFAULT_PAYOUTS, 0, 50.0)

    assert value == "9.50"
    assert "–" not in value
    assert "Every opponent costs Hero the same here." in help_text


def test_the_screen_renders_the_span_not_a_single_figure() -> None:
    """The readout has to reach the metric, not just exist as a helper."""

    app = AppTest.from_string(
        "import app as app_module\napp_module.show_icm_tool()\n",
        default_timeout=60,
    )
    app.run()

    metrics = [m for m in app.metric if m.label == "ICM cost of losing those chips"]
    assert len(metrics) == 1
    # Hero defaults to player 1 with 1000 at risk on the tool's own default stacks.
    assert metrics[0].value == "2.73 – 2.96"
