"""Property and fuzz tests for timeline state ordering (PLAN Phase 14).

``validate_timeline`` is the ONLY evidence the export gate reads
(``export_yolo_card_hands_for_app._split_source_codes`` consumes its warning
codes and nothing else), so an ordering rule it fails to state is an ordering
rule the product does not have.

The invariants generated against here:

  * **Time is the ordering.** The report is a function of WHAT was observed and
    WHEN, never of the order the producer happened to list the observations in.
    Permuting a hand's states without touching a timestamp cannot change which
    board/street warnings the hand carries.
  * **The board is append-only within a hand.** Cards do not leave the felt
    until the terminal sweep, so any shrink or substitution in time order is a
    ``board_regression``, from wherever in the list it was read.
  * **The street index never goes backwards or skips.** 0 -> 3 -> 4 -> 5 cards,
    in that order, with no rung missed.
  * **A state never precedes its own cause.** An action attributed to a street
    cannot be listed after an action on a later one, and a per-street pot cannot
    shrink.
  * **A monotone hand is never flagged.** The rules have to be silent on a
    clean reconstruction or they are not usable as a gate.

The permutation invariant is the one that was missing, and it was missing in
the direction that ships a wrong hand: see
``test_permuting_the_state_list_cannot_launder_a_board_regression``.
"""

from __future__ import annotations

import itertools

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from cv_lab.scripts.eval.validate_yolo_card_timeline import (
    ACTION_STREET_ORDER,
    MalformedTimeline,
    validate_timeline,
)

SETTINGS = settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much],
)

RANKS = "23456789TJQKA"
SUITS = "CDHS"
DECK = [rank + suit for rank in RANKS for suit in SUITS]
STREET_NAMES = ("preflop", "flop", "turn", "river", "showdown")


HERO = ["AS", "KD"]
# The board is dealt from what Hero does not hold; a hero/board collision is a
# duplicate_visible_cards finding about the CARD READER, and generating one
# would make every ordering assertion below fail for an unrelated reason.
BOARD_DECK = [card for card in DECK if card not in HERO]


def _state(time_s: float, image: str, board: list[str], hero: list[str] | None = None) -> dict:
    return {
        "time_s": time_s,
        "image": image,
        "hero_cards": list(HERO) if hero is None else hero,
        "board_cards": list(board),
        "other_cards": [],
        "missing": None,
    }


def _timeline(states: list[dict], **hand_overrides) -> dict:
    hand = {
        "hand_number": 1,
        "t_start": min(s["time_s"] for s in states),
        "t_end": max(s["time_s"] for s in states),
        "hero": list(HERO),
        "board": max((s["board_cards"] for s in states), key=len),
        "streets": [],
        "source_images": [s["image"] for s in states],
    }
    hand.update(hand_overrides)
    return {"states": states, "hands": [hand]}


def _codes(report: dict) -> list[str]:
    return sorted(w["code"] for w in report["hands"][0]["warnings"])


@st.composite
def monotone_hand(draw) -> list[dict]:
    """A board that only ever grows, sampled at strictly increasing times.

    This is what a correctly reconstructed hand looks like: 0 cards, then a
    flop, then a turn, then a river, each held for one or more samples, with the
    pot sweep clearing the felt at the end.
    """
    cards = draw(st.permutations(BOARD_DECK).map(lambda deck: list(deck[:5])))
    reached = draw(st.sampled_from([0, 3, 4, 5]))
    counts = [0] + [n for n in (3, 4, 5) if n <= reached]
    states: list[dict] = []
    time_s = 0.0
    for index, count in enumerate(counts):
        for _ in range(draw(st.integers(min_value=1, max_value=3))):
            time_s += draw(st.floats(min_value=0.5, max_value=4.0))
            states.append(_state(time_s, f"f{len(states):03d}.jpg", cards[:count]))
        del index
    if draw(st.booleans()) and states[-1]["board_cards"]:
        # The terminal pot sweep clears the board; it is settlement, not a
        # regression, and the validator has to keep treating it that way.
        states.append(_state(time_s + 1.0, f"f{len(states):03d}.jpg", []))
    return states


@given(states=monotone_hand())
@SETTINGS
def test_a_monotone_hand_carries_no_ordering_warning(states: list[dict]) -> None:
    """The negative half of every rule below. A gate that fires on a clean
    reconstruction is not a gate, it is noise, and the export path would then be
    rejecting good hands on ordering grounds."""
    report = validate_timeline(_timeline(states))
    codes = _codes(report)
    for code in ("board_regression", "street_order_issue", "state_time_order"):
        assert code not in codes, f"clean hand flagged {code}: {codes}"
    assert report["hands"][0]["confidence_score"] == 1.0


@given(states=monotone_hand(), data=st.data())
@SETTINGS
def test_the_report_is_invariant_to_the_order_the_states_are_listed_in(
    states: list[dict], data
) -> None:
    """THE property. Two producers that observed the same thing must be told the
    same thing, whatever order they wrote their list in.

    Only ``state_time_order`` may differ, and only in the direction of appearing
    -- it is the report SAYING the list disagreed with the clock, which is a
    fact about the input rather than about the hand.
    """
    assume(len(states) > 1)
    permuted = data.draw(st.permutations(states))
    assume([s["image"] for s in permuted] != [s["image"] for s in states])

    baseline = _codes(validate_timeline(_timeline(states)))
    shuffled = _codes(validate_timeline(_timeline(list(permuted))))
    assert [c for c in shuffled if c != "state_time_order"] == baseline


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_permuting_the_state_list_cannot_launder_a_board_regression(order) -> None:
    """THE REGRESSION. Three observations of one hand: an empty board at t=0, a
    four-card board at t=5, a three-card board at t=10. In time order that is a
    board_regression plus two street_order_issues and a confidence of 0.1.

    Listed 0, 10, 5 -- the same three states, the same three timestamps, nothing
    edited -- every sequence rule walked the list instead of the clock, read
    0 -> 3 -> 4 cards, and returned a CLEAN hand at confidence 1.0. Nothing
    downstream re-derives this: the export gate reads this report and nothing
    else, so a hand whose board was destroyed shipped as study-ready because of
    the order its states happened to sit in.
    """
    states = [
        _state(0.0, "a.jpg", []),
        _state(5.0, "b.jpg", ["2C", "3D", "4H", "5S"]),
        _state(10.0, "c.jpg", ["2C", "3D", "4H"]),
    ]
    report = validate_timeline(
        _timeline([states[i] for i in order], board=["2C", "3D", "4H"])
    )
    codes = _codes(report)
    assert "board_regression" in codes, f"order {order} hid the regression: {codes}"
    assert "street_order_issue" in codes
    assert report["hands"][0]["confidence_score"] < 0.8
    if list(order) != [0, 1, 2]:
        assert "state_time_order" in codes


@given(states=monotone_hand(), data=st.data())
@SETTINGS
def test_removing_a_board_card_mid_hand_is_always_flagged(
    states: list[dict], data
) -> None:
    """Board append-only, generated rather than exampled: drop one card from one
    mid-hand observation and the hand must be flagged, wherever that observation
    sits in the list."""
    indexed = [i for i, s in enumerate(states) if len(s["board_cards"]) >= 3]
    assume(indexed)
    # Not the last state: an empty or shrunken FINAL board is the terminal sweep,
    # which is settlement and deliberately exempt.
    victim = data.draw(st.sampled_from(indexed))
    assume(victim < len(states) - 1)
    # The cut has to make the board SHRINK against what was already on the felt.
    # Taking a card off the first turn observation only postpones the turn, and
    # 0 -> 3 -> 3 -> 4 is a legal hand, not a regression.
    assume(victim > 0)
    assume(len(states[victim - 1]["board_cards"]) >= len(states[victim]["board_cards"]))

    damaged = [dict(s) for s in states]
    damaged[victim] = dict(states[victim])
    damaged[victim]["board_cards"] = states[victim]["board_cards"][:-1]

    order = data.draw(st.permutations(range(len(damaged))))
    report = validate_timeline(_timeline([damaged[i] for i in order]))
    codes = _codes(report)
    # Any of the three is a correct refusal: the shrink itself, the street index
    # it moves backwards, or -- when the cut leaves 2 or 4 cards where a flop was
    # -- a board count no street can produce. What may not happen is silence.
    assert {"board_regression", "street_order_issue", "invalid_board_count"} & set(codes), codes
    assert report["hands"][0]["confidence_score"] < 1.0


@given(states=monotone_hand(), data=st.data())
@SETTINGS
def test_substituting_a_board_card_mid_hand_is_always_flagged(
    states: list[dict], data
) -> None:
    """The other half of append-only: the count can hold while the CARDS change.
    A board that keeps its length but swaps a card is not a longer board, it is a
    different one, and the earlier reading was wrong."""
    indexed = [i for i, s in enumerate(states) if len(s["board_cards"]) >= 3]
    assume(len(indexed) >= 2)
    victim = data.draw(st.sampled_from(indexed[1:]))
    replacement = data.draw(
        st.sampled_from([c for c in BOARD_DECK if c not in states[victim]["board_cards"]])
    )
    damaged = [dict(s) for s in states]
    damaged[victim] = dict(states[victim])
    board = list(states[victim]["board_cards"])
    board[0] = replacement
    damaged[victim]["board_cards"] = board
    assume(any(damaged[i]["board_cards"] != damaged[victim]["board_cards"]
               for i in indexed if i < victim))

    order = data.draw(st.permutations(range(len(damaged))))
    report = validate_timeline(_timeline([damaged[i] for i in order]))
    assert "board_regression" in _codes(report), _codes(report)


@given(
    cards=st.permutations(BOARD_DECK).map(lambda deck: list(deck[:5])),
    skipped=st.sampled_from([(0, 4), (0, 5), (3, 5)]),
)
@SETTINGS
def test_a_skipped_street_is_flagged_however_the_states_are_listed(
    cards: list[str], skipped: tuple[int, int]
) -> None:
    """A street index that jumps a rung means the reconstruction never saw the
    street in between, so any action it attributes there was invented."""
    low, high = skipped
    states = [
        _state(0.0, "a.jpg", cards[:low]),
        _state(5.0, "b.jpg", cards[:high]),
        _state(10.0, "c.jpg", cards[:high]),
    ]
    for order in itertools.permutations(range(3)):
        report = validate_timeline(_timeline([states[i] for i in order], board=cards[:high]))
        assert "street_order_issue" in _codes(report), (order, _codes(report))


@given(
    times=st.lists(
        st.floats(min_value=0.0, max_value=600.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
        unique=True,
    )
)
@SETTINGS
def test_a_state_list_that_disagrees_with_its_own_clock_is_reported(
    times: list[float],
) -> None:
    """Time monotonicity, stated as its own rule rather than inferred from the
    board. It is a fact about the PRODUCER: the reconstruction spine consumes the
    same list in the same list order (build_states, the run-length debounces,
    fold detection, action attribution), so an out-of-order list means its
    players, actions, pot and winner describe a sequence the recording never
    had. Nothing anywhere read ``time_s`` before this."""
    states = [_state(t, f"f{i}.jpg", []) for i, t in enumerate(times)]
    ascending = sorted(times)

    ordered = validate_timeline(_timeline([_state(t, f"f{i}.jpg", []) for i, t in enumerate(ascending)]))
    assert "state_time_order" not in _codes(ordered)

    if times != ascending:
        assert "state_time_order" in _codes(validate_timeline(_timeline(states)))


@given(
    streets=st.lists(st.sampled_from(STREET_NAMES[:4]), min_size=2, max_size=8),
)
@SETTINGS
def test_actions_never_precede_their_own_street(streets: list[str]) -> None:
    """A state never precedes its cause. Preflop action cannot be recorded after
    flop action; there is no way back up the street ladder inside one hand."""
    actions = [
        {"seat": i % 3, "street": street, "action_type": "check", "action_index": i}
        for i, street in enumerate(streets)
    ]
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 1.0,
        "hero": ["AS", "KD"],
        "board": ["2C", "3D", "4H"],
        "streets": [{"street": "flop", "time_s": 1.0, "board": ["2C", "3D", "4H"]}],
        "source_images": ["a.jpg"],
        "players": [{"seat": i, "position": p, "is_hero": i == 0}
                    for i, p in enumerate(("BTN", "SB", "BB"))],
        "actions": actions,
    }
    timeline = {"states": [_state(0.0, "a.jpg", ["2C", "3D", "4H"])], "hands": [hand]}
    codes = _codes(validate_timeline(timeline))

    indexes = [ACTION_STREET_ORDER[s] for s in streets]
    went_backwards = any(b < a for a, b in zip(indexes, indexes[1:], strict=False))
    assert ("action_street_order" in codes) == went_backwards, (streets, codes)


@given(
    pots=st.lists(
        st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=4,
    )
)
@SETTINGS
def test_a_per_street_pot_never_shrinks(pots: list[float]) -> None:
    """The pot is a running total, so a later street holding less than an earlier
    one is not a smaller pot -- it is a misread, and the hand's whole money
    ledger is derived from it."""
    names = STREET_NAMES[1:1 + len(pots)]
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 1.0,
        "hero": ["AS", "KD"],
        "board": ["2C", "3D", "4H", "5S", "9C"][: 3 + len(pots) - 1],
        "streets": [
            {"street": name, "time_s": float(i), "board": ["2C", "3D", "4H"], "pot": pot}
            for i, (name, pot) in enumerate(zip(names, pots, strict=True))
        ],
        "source_images": ["a.jpg"],
        "actions": [],
    }
    timeline = {"states": [_state(0.0, "a.jpg", ["2C", "3D", "4H"])], "hands": [hand]}
    codes = _codes(validate_timeline(timeline))
    shrank = any(b < a - 1e-6 for a, b in zip(pots, pots[1:], strict=False))
    assert ("pot_regression" in codes) == shrank, (pots, codes)


@given(
    payload=st.recursive(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(),
            st.floats(allow_nan=True, allow_infinity=True),
            st.text(max_size=8),
        ),
        lambda children: st.one_of(
            st.lists(children, max_size=4),
            st.dictionaries(st.text(max_size=6), children, max_size=4),
        ),
        max_leaves=25,
    )
)
@SETTINGS
def test_an_arbitrary_document_is_either_validated_or_named_malformed(payload) -> None:
    """Totality. The validator is handed timelines written by other builds, by
    the manual-correction path, and by hand; it may return a report or raise
    ``MalformedTimeline``, and nothing else. A ``TypeError`` or ``KeyError`` out
    of here reaches the operator as a stack trace with no verdict, which the
    export gate cannot fail closed on."""
    try:
        report = validate_timeline(payload)
    except MalformedTimeline:
        return
    assert set(report) == {"summary", "hands"}
    assert report["summary"]["malformed"] is False
    assert 0.0 <= report["summary"]["confidence_score"] <= 1.0
    for hand in report["hands"]:
        assert hand["warning_count"] == len(hand["warnings"])
        assert all("code" in w and "message" in w for w in hand["warnings"])
