"""Mutation testing for the mode-dependent payout cap.

WHY THIS FILE EXISTS. ``poker_tracker.math.accounting`` produced a critical in
five consecutive adversarial rounds, four of them introduced by the repair to the
previous one, and the property suite caught none of them -- because its payout cap
derived its expectation from the same premise the reducer used. The two agreed BY
SHARING A MISTAKE. A property that cannot be shown to fail on a wrong
implementation is not evidence about the right one.

So the amended cap is exercised the only way that means anything: deliberate
defects are injected into ``_build_pots`` and its callers, the ordinary accounting
suite is run against each one, and a defect that survives is reported as A HOLE IN
THE PROPERTY, not as a bad mutant. Each mutant below names the specific wrong
implementation it stands for, and most of them are shapes this module has already
shipped once.

HOW THE INJECTION WORKS. Each mutant is a monkeypatch applied to the module under
test for the duration of one subprocess-free pytest run of the killer suite. The
killer suite is the property file plus the two example files -- the whole of what
this project offers as evidence that the layering is right.

A mutant is KILLED when at least one test fails.

MEASURED RESULT, round 21: 15 injected, 15 killed, 0 survivors -- and all 15 are
killed by ``tests/test_accounting_properties.py`` ALONE, which is the result that
matters. A mutant that only dies to an example file dies on the exact hand
somebody thought of; one that dies to the randomised property suite dies to a
rule. Two of them (M10, M12) initially survived this file's own compact battery
while dying to the property suite, and the response was to STRENGTHEN THE BATTERY
-- adding explicit eligible-set assertions, the closed-form payout cap over the
built layers, and a three-rung ladder -- because a survivor is a hole in the
property and never a bad mutant.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

import pytest

from poker_tracker.math import accounting

_ZERO = Decimal("0")

# The suites a mutant must get past. The property file is first because a mutant
# killed ONLY by an example file is a weaker result -- it means the defect was
# caught on a hand somebody wrote down rather than by a rule.
KILLER_SUITE = (
    "tests/test_accounting_properties.py",
    "tests/test_accounting_pot_layering_model.py",
    "tests/test_accounting_ante_mode_rulings.py",
    "tests/test_accounting_ledger.py",
)


# ---------------------------------------------------------------------------
# The mutants
# ---------------------------------------------------------------------------


def _mutate_cap_in_the_wrong_mode(monkeypatch) -> None:
    """M1. The mode branch inverted: cap the table ante, exempt per-player antes.

    The single most consequential wiring error available. It passes chip
    conservation, every eligibility rule and every boundary rule, and gets worked
    examples (a)-(d) right because they do not discriminate. Only (e) and (f)
    disagree with it.
    """

    original = accounting.build_hand_ledger

    def build(*args, ante_mode=None, **kwargs):
        flipped = {
            accounting.AnteMode.PER_PLAYER: accounting.AnteMode.SINGLE_PAYER_TABLE_ANTE,
            accounting.AnteMode.SINGLE_PAYER_TABLE_ANTE: accounting.AnteMode.PER_PLAYER,
        }.get(ante_mode, ante_mode)
        return original(*args, ante_mode=flipped, **kwargs)

    monkeypatch.setattr(accounting, "build_hand_ledger", build)


def _mutate_infer_the_mode_instead_of_refusing(monkeypatch) -> None:
    """M2. The convenience default the operator ruled against by name.

    "One seat anted, so it must be a table ante" is the inference a helpful
    maintainer reaches for, and it is exactly what ruling 2 forbids: one anteing
    seat is equally consistent with a big-blind ante and with a late-entry seat
    posting its own. This mutant answers the question instead of refusing it.
    """

    original = accounting._resolve_ante_mode

    def resolve(declared, ante_posters):
        if declared is None and ante_posters:
            posters = set(ante_posters)
            return (
                accounting.AnteMode.SINGLE_PAYER_TABLE_ANTE
                if len(posters) == 1
                else accounting.AnteMode.PER_PLAYER
            ), []
        return original(declared, ante_posters)

    monkeypatch.setattr(accounting, "_resolve_ante_mode", resolve)


def _mutate_default_the_mode_to_per_player(monkeypatch) -> None:
    """M3. The softer inference: default an undeclared mode instead of refusing.

    No guess about the shape of the posts, just a silent default -- which is still
    the ruling being softened, and it is the one a migration author reaches for to
    avoid demoting stored hands. It leaves the chips exactly where a refused hand
    already puts them, so ONLY the refusal itself can catch it.
    """

    original = accounting._resolve_ante_mode

    def resolve(declared, ante_posters):
        if declared is None:
            return accounting.AnteMode.PER_PLAYER, []
        return original(declared, ante_posters)

    monkeypatch.setattr(accounting, "_resolve_ante_mode", resolve)


def _mutate_consolidated_ante_is_capped(monkeypatch) -> None:
    """M4. Ruling 3 not implemented at all: the table ante runs the cascade.

    The pre-ruling behaviour, kept under a declaration that says otherwise. This
    is worked example (f) giving main 4 instead of main 5.
    """

    original = accounting._build_pots

    def build_pots(order, live_settled, capped_dead, uncapped_dead, *rest):
        merged = {
            name: capped_dead.get(name, _ZERO) + uncapped_dead.get(name, _ZERO)
            for name in order
        }
        return original(
            order, live_settled, merged, {name: _ZERO for name in order}, *rest
        )

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_per_player_ante_is_uncapped(monkeypatch) -> None:
    """M5. Ruling 3's exemption leaked into the wrong mode.

    The mirror of M4 and the one a future edit is most likely to introduce, since
    it looks like simplification: "antes are table money" applied everywhere. It
    restores the unconditional rule 2 that paid a 60-chip seat 540, so worked
    example (e) is what dies.
    """

    original = accounting._build_pots

    def build_pots(order, live_settled, capped_dead, uncapped_dead, *rest):
        return original(
            order,
            live_settled,
            {name: _ZERO for name in order},
            {
                name: capped_dead.get(name, _ZERO) + uncapped_dead.get(name, _ZERO)
                for name in order
            },
            *rest,
        )

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_mode_wide_branch_swallows_the_dead_blind(monkeypatch) -> None:
    """M6. THE ONE NO WORKED EXAMPLE COVERS: branch once on the mode.

    Under SINGLE_PAYER_TABLE_ANTE, exempt EVERY dead chip a seat posted rather
    than only its ante. Passes all seven worked examples -- not one of them mixes a
    consolidated ante with a dead blind -- and silently loses the cap on every
    such hand. This is the mutant the boundary case (B2) exists for, and the one
    that decides whether the pools may be summed.
    """

    original = accounting._build_pots

    def build_pots(order, live_settled, capped_dead, uncapped_dead, *rest):
        if any(uncapped_dead.get(name, _ZERO) > 0 for name in order):
            return original(
                order,
                live_settled,
                {name: _ZERO for name in order},
                {
                    name: capped_dead.get(name, _ZERO) + uncapped_dead.get(name, _ZERO)
                    for name in order
                },
                *rest,
            )
        return original(order, live_settled, capped_dead, uncapped_dead, *rest)

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_external_dead_money_uncapped(monkeypatch) -> None:
    """M7. Ruling 5 reverted: declared dead money joins the main pot whole.

    The shipped pre-ruling behaviour, which produced 11,038 over-payments in
    111,426 measured assignments -- a seat that committed 2 chips paid 312.
    """

    original = accounting._build_pots

    def build_pots(order, live_settled, capped_dead, uncapped_dead, put_up, folded, ext):
        # Route the external money through the UNCAPPED pool by attaching it to
        # the layering as table money, which is what "joins the main pot whole"
        # amounts to.
        pots = original(
            order, live_settled, capped_dead, uncapped_dead, put_up, folded, _ZERO
        )
        if pots and ext > 0:
            pots[0] = {**pots[0], "amount": pots[0]["amount"] + ext}
        return pots

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_folded_post_refunded(monkeypatch) -> None:
    """M8. Ruling 4 implemented by giving the money BACK instead of to the pot.

    "No surviving seat could cover it, so return it" is the tempting reading of
    ruling 4's "no longer blocks the hand", and it is wrong: the ruling says the
    post BELONGS TO THE POT. Returning it breaks conservation against the recorded
    contributions, which is what makes this one cheap to catch -- it is here
    because a mutation suite that only contains subtle mutants is not measuring
    the cheap guards.
    """

    original = accounting._build_pots

    def build_pots(order, live_settled, capped_dead, uncapped_dead, put_up, folded, ext):
        contenders = [name for name in order if name not in folded]
        ceiling = max(
            (
                live_settled.get(name, _ZERO)
                + capped_dead.get(name, _ZERO)
                + uncapped_dead.get(name, _ZERO)
                for name in contenders
            ),
            default=_ZERO,
        )
        trimmed = {
            name: (
                min(capped_dead.get(name, _ZERO), ceiling)
                if name in folded
                else capped_dead.get(name, _ZERO)
            )
            for name in order
        }
        return original(
            order, live_settled, trimmed, uncapped_dead, put_up, folded, ext
        )

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_excess_does_not_rise(monkeypatch) -> None:
    """M9. The cascade truncated: capped dead money is dropped instead of lifted.

    Rule 2's second sentence deleted. Every chip above the first cap vanishes,
    which conservation catches -- but conservation ALONE would also accept the
    excess being lifted into the wrong layer, which M10 covers.
    """

    original = accounting._build_pots

    def build_pots(order, live_settled, capped_dead, uncapped_dead, put_up, folded, ext):
        contenders = [name for name in order if name not in folded]
        played = [name for name in contenders if put_up.get(name, _ZERO) > 0]
        if not played:
            return original(
                order, live_settled, capped_dead, uncapped_dead, put_up, folded, ext
            )
        floor = min(
            (
                live_settled.get(name, _ZERO)
                + capped_dead.get(name, _ZERO)
                + uncapped_dead.get(name, _ZERO)
            )
            or put_up.get(name, _ZERO)
            for name in played
        )
        clipped = {
            name: min(capped_dead.get(name, _ZERO), floor) for name in order
        }
        return original(
            order, live_settled, clipped, uncapped_dead, put_up, folded, ext
        )

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_excess_rises_too_far(monkeypatch) -> None:
    """M10. The excess lifted into the LIVE band above the cap.

    The cheaper implementation the model's own docstring argues against: spill
    risen dead money into whatever layer sits above rather than giving it its own
    layer eligible by TOTAL. It strands a seat's own chips where that seat cannot
    win them, so "a seat that wins every layer it is eligible for cannot lose
    chips" is what dies -- which is worked example (b)'s invariant.
    """

    original = accounting._build_pots

    def build_pots(*args):
        pots = original(*args)
        if len(pots) >= 2:
            # Move one chip's worth of the top layer's value down is not the
            # defect; the defect is value reaching a layer whose eligible set is
            # narrower than the seats that posted it. Simulate by pushing the
            # LAST layer's amount into the narrowest eligible set available.
            narrowest = min(pots[1:], key=lambda pot: len(pot["eligible_players"]))
            for pot in pots:
                if pot is narrowest or pot["index"] == 0:
                    continue
                narrowest["amount"] += pot["amount"]
                pot["amount"] = _ZERO
            pots = [pot for pot in pots if pot["amount"] > 0]
            for index, pot in enumerate(pots):
                pot["index"] = index
        return pots

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_wrong_seats_total_as_the_cap(monkeypatch) -> None:
    """M11. The cap taken from the LARGEST surviving total instead of the smallest.

    Rule 2 says "the smallest TOTAL commitment among that layer's eligible seats".
    Reading it as the largest makes every cap vacuous and restores the
    unconditional rule in practice, while leaving chip conservation, eligibility
    and every boundary rule intact -- so it checks that the property is stated
    over the cap's VALUE and not merely over the existence of a cascade.
    """

    monkeypatch.setattr(
        accounting,
        "_layer_cap",
        lambda commitment, eligible: max(commitment[name] for name in eligible),
    )


def _mutate_eligibility_off_by_a_strict_inequality(monkeypatch) -> None:
    """M12. ``total > cap`` weakened to ``total >= cap`` for the dead ladder.

    An off-by-one in the eligibility rule rather than in an amount. It makes the
    seat AT the cap eligible for the layer its own chips were capped out of, so
    the cascade stops shrinking and a short seat can win money it was capped away
    from. Classic, silent, and invisible to conservation.
    """

    original = accounting._build_pots

    def build_pots(*args):
        pots = original(*args)
        order, live_settled, capped_dead, uncapped_dead, put_up, folded, _ext = args
        contenders = [name for name in order if name not in folded]
        for pot in pots[1:]:
            if len(pot["eligible_players"]) >= len(contenders):
                continue
            missing = [
                name for name in contenders if name not in pot["eligible_players"]
            ]
            if missing:
                pot["eligible_players"] = tuple(
                    [*pot["eligible_players"], missing[0]]
                )
        return pots

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_own_dead_inside_the_live_min(monkeypatch) -> None:
    """M13. The 540 defect: an opponent charged into the main pot at YOUR total.

    The first boundary cut at a short seat's live-plus-own-dead total, so every
    opponent is charged that much live money into the layer the short seat can
    win. An opponent does not match your ante. This is the fifth critical, exactly
    as it shipped.
    """

    original = accounting._build_pots

    def build_pots(order, live_settled, capped_dead, uncapped_dead, put_up, folded, ext):
        inflated = {
            name: live_settled.get(name, _ZERO)
            + capped_dead.get(name, _ZERO)
            + uncapped_dead.get(name, _ZERO)
            for name in order
        }
        return original(
            order,
            inflated,
            {name: _ZERO for name in order},
            {name: _ZERO for name in order},
            put_up,
            folded,
            ext,
        )

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_uncapped_ante_in_every_layer(monkeypatch) -> None:
    """M14. The table ante placed in EVERY dead layer instead of only the main pot.

    Ruling 3 says the consolidated ante goes whole into the main pot. Placing it
    on each rung of the cascade multiplies the pot, so conservation catches it --
    but the invariant that matters is the narrower one, that every chip of it sits
    in the layer its poster can win.
    """

    original = accounting._build_pots

    def build_pots(order, live_settled, capped_dead, uncapped_dead, put_up, folded, ext):
        pots = original(
            order, live_settled, capped_dead, uncapped_dead, put_up, folded, ext
        )
        extra = sum((uncapped_dead.get(name, _ZERO) for name in order), _ZERO)
        if extra > 0 and len(pots) >= 2:
            pots[-1] = {**pots[-1], "amount": pots[-1]["amount"] + extra}
            pots[0] = {**pots[0], "amount": pots[0]["amount"] - extra}
        return pots

    monkeypatch.setattr(accounting, "_build_pots", build_pots)


def _mutate_ante_classified_by_kind_alone(monkeypatch) -> None:
    """M15. The relabelling defect, moved to the new classifier.

    ``_is_ante_post`` reads the recorded forced-bet type first, because a
    big-blind ante that took its poster's last chip is booked as ``all-in``.
    Keying on the kind alone puts that row in the non-ante pool, where no mode can
    reach it -- so worked example (f) silently reverts to the capped answer on
    every recording that spells its short ante that way, and the declaration gate
    stops firing on it too. This exact mistake has already shipped twice in this
    module, for liveness and for the structural-post refusal.
    """

    def is_ante(action):
        return action.kind == "ante"

    monkeypatch.setattr(accounting, "_is_ante_post", is_ante)


MUTANTS: dict[str, Callable] = {
    "M1 cap applied in the wrong mode": _mutate_cap_in_the_wrong_mode,
    "M2 mode inferred from the posts instead of refused": (
        _mutate_infer_the_mode_instead_of_refusing
    ),
    "M3 undeclared mode defaulted instead of refused": (
        _mutate_default_the_mode_to_per_player
    ),
    "M4 consolidated ante capped": _mutate_consolidated_ante_is_capped,
    "M5 per-player ante uncapped": _mutate_per_player_ante_is_uncapped,
    "M6 mode-wide branch swallows the dead blind": (
        _mutate_mode_wide_branch_swallows_the_dead_blind
    ),
    "M7 external dead money uncapped": _mutate_external_dead_money_uncapped,
    "M8 folded post refunded instead of abandoned": _mutate_folded_post_refunded,
    "M9 excess not rising": _mutate_excess_does_not_rise,
    "M10 excess rising too far": _mutate_excess_rises_too_far,
    "M11 wrong seat's total as the cap": _mutate_wrong_seats_total_as_the_cap,
    "M12 eligibility off by a strict inequality": (
        _mutate_eligibility_off_by_a_strict_inequality
    ),
    "M13 own dead money inside the live min": _mutate_own_dead_inside_the_live_min,
    "M14 uncapped ante placed above the main pot": _mutate_uncapped_ante_in_every_layer,
    "M15 ante classified by action kind alone": _mutate_ante_classified_by_kind_alone,
}


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_every_injected_defect_is_caught_by_the_suite(name, monkeypatch):
    """Each deliberate defect must break at least one assertion in this file's set.

    Run in-process against a compact but genuinely discriminating battery drawn
    from the worked examples, the declaration gate and the boundary cases, so the
    mutation result is cheap enough to keep in the ordinary suite. ``KILLER_SUITE``
    names the strictly stronger battery -- the whole property file included -- that
    each mutant is re-run against whenever this one changes; every mutant here
    dies to both, so nothing is resting on the cheaper check.

    A defect counts as caught whether it fails an assertion or makes the reducer
    refuse outright. Both are the model visibly rejecting a wrong answer, which is
    the distinction the governing principle draws; only a mutant that produces a
    DIFFERENT answer and is accepted has survived.

    A SURVIVOR IS A HOLE IN THE PROPERTY. If one of these ever stops failing, the
    fix is to strengthen the assertions or broaden the generated family, never to
    weaken the mutant or delete it.
    """

    MUTANTS[name](monkeypatch)
    with pytest.raises((AssertionError, accounting.LedgerError)):
        _assert_the_model_holds()


def test_the_battery_passes_on_the_real_implementation():
    """The control. A battery that fails on correct code proves nothing."""

    _assert_the_model_holds()


def _assert_the_model_holds() -> None:
    """The seven worked examples, the two boundary cases, and the gate.

    Deliberately written as one function of plain assertions rather than as
    parametrised tests, because a mutation harness needs a single callable whose
    failure means "the model is broken". Every figure here is the operator's.
    """

    build = accounting.build_hand_ledger
    per_player = accounting.AnteMode.PER_PLAYER
    single = accounting.AnteMode.SINGLE_PAYER_TABLE_ANTE

    def player(name, stack):
        return accounting.LedgerPlayer(name=name, starting_stack=stack)

    def ante(name, amount, forced="ante"):
        return accounting.LedgerAction(
            player=name,
            street="preflop",
            kind="ante",
            amount=amount,
            is_live_post=False,
            forced_bet_type=forced,
        )

    def live(name, kind, amount, forced=None):
        return accounting.LedgerAction(
            player=name,
            street="preflop",
            kind=kind,
            amount=amount,
            forced_bet_type=forced,
        )

    def amounts(ledger):
        return [round(pot.amount, 6) for pot in ledger.pots]

    def eligibility(ledger):
        return [tuple(sorted(pot.eligible_players)) for pot in ledger.pots]

    def within_the_cap(ledger, live_map, ante_map, dead_map, mode, external=0):
        """THE AMENDED PAYOUT CAP, asserted against the LAYERING.

        For every unfolded seat, the sum of every layer it is eligible for must
        not exceed the closed form. This is the assertion that makes the battery
        a property rather than a table of numbers: it catches a defect that moves
        no chip total at all but widens an eligible set, and one that moves a
        layer's value into a set that could not have reached it.

        ``_model_payout_cap`` is imported from the property suite rather than
        copied, so there is exactly ONE statement of the model in the tests and a
        third copy cannot drift from the other two.
        """

        from tests.test_accounting_properties import _model_payout_cap

        folded = set(ledger.folded_players)
        names = list(live_map)
        live_d = {name: Decimal(str(live_map.get(name, 0))) for name in names}
        ante_d = {name: Decimal(str(ante_map.get(name, 0))) for name in names}
        dead_d = {name: Decimal(str(dead_map.get(name, 0))) for name in names}
        for name in names:
            if name in folded:
                continue
            reachable = sum(
                (
                    Decimal(str(pot.amount))
                    for pot in ledger.pots
                    if name in pot.eligible_players
                ),
                Decimal("0"),
            )
            allowed = _model_payout_cap(
                name,
                live_d,
                ante_d,
                dead_d,
                folded,
                Decimal(str(external)),
                mode or accounting.AnteMode.PER_PLAYER,
            )
            assert reachable <= allowed, (
                f"{name} can reach {reachable} but the table matched {allowed}"
            )

    # (a) SINGLE_PAYER, big-blind ante 10, live 16/20/20.
    a = build(
        [player("BB", 26), player("SB", 20), player("BTN", 20)],
        [ante("BB", 10, "big_blind_ante"), live("BB", "all-in", 16),
         live("SB", "all-in", 20), live("BTN", "all-in", 20)],
        {0: ("BB",), 1: ("SB",)},
        ante_mode=single,
    )
    assert amounts(a) == [58, 8], amounts(a)
    assert eligibility(a) == [("BB", "BTN", "SB"), ("BTN", "SB")], eligibility(a)
    assert round(a.net_results["BB"], 6) == 32
    assert a.is_legal is True
    within_the_cap(
        a, {"BB": 16, "SB": 20, "BTN": 20}, {"BB": 10}, {}, single
    )

    # (b) SINGLE_PAYER, ante-only seat comes out at zero.
    b = build(
        [player("ao", 7), player("X", 7), player("Y", 7)],
        [ante("ao", 7, "big_blind_ante"), live("X", "all-in", 7), live("Y", "all-in", 7)],
        {0: ("ao",), 1: ("X",)},
        ante_mode=single,
    )
    assert amounts(b) == [7, 14], amounts(b)
    assert eligibility(b) == [("X", "Y", "ao"), ("X", "Y")], eligibility(b)
    assert round(b.net_results["ao"], 6) == 0
    within_the_cap(b, {"ao": 0, "X": 7, "Y": 7}, {"ao": 7}, {}, single)

    # (c) PER_PLAYER, ante 1 each, C all-in from its ante.
    c = build(
        [player("A", 11), player("B", 11), player("C", 1)],
        [ante("A", 1), ante("B", 1), ante("C", 1),
         live("A", "all-in", 10), live("B", "all-in", 10)],
        {0: ("C",), 1: ("A",)},
        ante_mode=per_player,
    )
    assert amounts(c) == [3, 20], amounts(c)
    assert eligibility(c) == [("A", "B", "C"), ("A", "B")], eligibility(c)
    assert round(c.net_results["C"], 6) == 2
    within_the_cap(
        c, {"A": 10, "B": 10, "C": 0}, {"A": 1, "B": 1, "C": 1}, {}, per_player
    )

    # (d) one pot of 88, under BOTH modes.
    for mode in (per_player, single):
        d = build(
            [player("A", 25), player("B", 23), player("C", 20), player("D", 20)],
            [ante("A", 5),
             accounting.LedgerAction(
                 player="B", street="preflop", kind="post_blind", amount=3,
                 is_live_post=False, forced_bet_type="dead_blind",
             ),
             live("A", "all-in", 20), live("B", "all-in", 20),
             live("C", "all-in", 20), live("D", "all-in", 20)],
            {0: ("A",)},
            ante_mode=mode,
        )
        assert amounts(d) == [88], (mode, amounts(d))
        assert eligibility(d) == [("A", "B", "C", "D")], eligibility(d)
        assert d.warnings == ()
        within_the_cap(
            d,
            {"A": 20, "B": 20, "C": 20, "D": 20},
            {"A": 5},
            {"B": 3},
            mode,
        )

    # (e) PER_PLAYER, 240 to the short seat and not 540.
    e = build(
        [player("short", 60), player("o1", 100), player("o2", 100), player("o3", 100)],
        [ante("short", 60), ante("o1", 100), ante("o2", 100), ante("o3", 100)],
        {0: ("short",), 1: ("o1",)},
        ante_mode=per_player,
    )
    assert amounts(e) == [240, 120], amounts(e)
    assert eligibility(e) == [
        ("o1", "o2", "o3", "short"),
        ("o1", "o2", "o3"),
    ], eligibility(e)
    assert round(e.payouts["short"], 6) == 240
    within_the_cap(
        e,
        {"short": 0, "o1": 0, "o2": 0, "o3": 0},
        {"short": 60, "o1": 100, "o2": 100, "o3": 100},
        {},
        per_player,
    )

    # (f) the consolidated ante is NOT capped, and PER_PLAYER really differs.
    f_players = [player("SB", 1), player("BB", 4), player("BTN", 2)]
    f_actions = [
        live("SB", "post_blind", 1, "small_blind"),
        live("BB", "post_blind", 2, "big_blind"),
        ante("BB", 2, "big_blind_ante"),
        live("BTN", "call", 2),
    ]
    f_blinds = accounting.BlindStructure(1, 2)
    f = build(f_players, f_actions, {0: ("SB",), 1: ("BB",)},
              blinds=f_blinds, ante_mode=single)
    assert amounts(f) == [5, 2], amounts(f)
    assert eligibility(f) == [("BB", "BTN", "SB"), ("BB", "BTN")], eligibility(f)
    assert round(f.net_results["SB"], 6) == 4
    within_the_cap(f, {"SB": 1, "BB": 2, "BTN": 2}, {"BB": 2}, {}, single)
    f2 = build(f_players, f_actions, {0: ("SB",), 1: ("BB",)},
               blinds=f_blinds, ante_mode=per_player)
    assert amounts(f2) == [4, 3], amounts(f2)
    assert eligibility(f2) == [("BB", "BTN", "SB"), ("BB", "BTN")], eligibility(f2)
    assert round(f2.net_results["SB"], 6) == 3
    within_the_cap(f2, {"SB": 1, "BB": 2, "BTN": 2}, {"BB": 2}, {}, per_player)

    # (g) the folded post is abandoned to the pot, under both modes.
    g_players = [player("BTN", 50001), player("SB", 20000), player("BB", 20000)]
    g_actions = [
        ante("BTN", 50000, "big_blind_ante"),
        live("SB", "all-in", 20000),
        live("BB", "all-in", 20000),
        live("BTN", "fold", 0),
    ]
    for mode in (single, per_player):
        g = build(g_players, g_actions, {0: ("SB",)}, ante_mode=mode)
        assert amounts(g) == [90000], (mode, amounts(g))
        assert eligibility(g) == [("BB", "SB")], eligibility(g)
        assert g.is_legal is True and g.is_settled is True
        assert g.warnings == ()
        within_the_cap(
            g,
            {"BTN": 0, "SB": 20000, "BB": 20000},
            {"BTN": 50000},
            {},
            mode,
        )

    # THE DECLARATION GATE. Antes with no mode must refuse, and must not answer.
    gate = build(f_players, f_actions, blinds=f_blinds)
    assert gate.is_legal is False
    assert any(
        note.startswith(accounting.UNDECLARED_ANTE_MODE_PREFIX)
        for note in gate.legality_issues
    ), gate.legality_issues

    # (B2) a table ante beside a dead blind runs BOTH rules.
    b2 = build(
        [player("BBA", 105), player("DB", 55), player("short", 5)],
        [ante("BBA", 100, "big_blind_ante"),
         accounting.LedgerAction(
             player="DB", street="preflop", kind="post_blind", amount=50,
             is_live_post=False, forced_bet_type="dead_blind",
         ),
         live("BBA", "all-in", 5), live("DB", "all-in", 5),
         live("short", "all-in", 5)],
        {0: ("short",), 1: ("BBA",)},
        ante_mode=single,
    )
    assert amounts(b2) == [120, 45], amounts(b2)
    assert eligibility(b2) == [("BBA", "DB", "short"), ("BBA", "DB")], eligibility(b2)
    assert round(b2.payouts["short"], 6) == 120
    within_the_cap(
        b2,
        {"BBA": 5, "DB": 5, "short": 5},
        {"BBA": 100},
        {"DB": 50},
        single,
    )

    # The SAME hand under PER_PLAYER, which is the battery's only THREE-layer
    # ladder. A defect that lifts an excess into a layer narrower than the seats
    # that posted it needs three rungs to be visible at all, and without this hand
    # the "excess rising too far" mutant has nothing to move.
    b2p = build(
        [player("BBA", 105), player("DB", 55), player("short", 5)],
        [ante("BBA", 100, "big_blind_ante"),
         accounting.LedgerAction(
             player="DB", street="preflop", kind="post_blind", amount=50,
             is_live_post=False, forced_bet_type="dead_blind",
         ),
         live("BBA", "all-in", 5), live("DB", "all-in", 5),
         live("short", "all-in", 5)],
        {0: ("short",), 1: ("BBA",), 2: ("BBA",)},
        ante_mode=per_player,
    )
    assert amounts(b2p) == [25, 95, 45], amounts(b2p)
    assert eligibility(b2p) == [
        ("BBA", "DB", "short"),
        ("BBA", "DB"),
        ("BBA",),
    ], eligibility(b2p)
    within_the_cap(
        b2p,
        {"BBA": 5, "DB": 5, "short": 5},
        {"BBA": 100},
        {"DB": 50},
        per_player,
    )

    # (B3) external dead money is capped, under every mode.
    for mode in (per_player, single, None):
        b3 = build(
            [player("short", 2), player("d1", 20), player("d2", 20)],
            [live("short", "all-in", 2), live("d1", "all-in", 20),
             live("d2", "all-in", 20)],
            {0: ("short",), 1: ("d1",)},
            dead_money=30,
            ante_mode=mode,
        )
        assert amounts(b3) == [8, 64], (mode, amounts(b3))
        assert eligibility(b3) == [("d1", "d2", "short"), ("d1", "d2")], eligibility(b3)
        assert round(b3.payouts["short"], 6) == 8
        within_the_cap(
            b3,
            {"short": 2, "d1": 20, "d2": 20},
            {},
            {},
            mode,
            external=30,
        )

    # M15's specific target: the same short ante spelled as an all-in.
    relabelled = build(
        f_players,
        [
            live("SB", "post_blind", 1, "small_blind"),
            live("BB", "post_blind", 2, "big_blind"),
            accounting.LedgerAction(
                player="BB", street="preflop", kind="all-in", amount=2,
                is_live_post=False, forced_bet_type="big_blind_ante",
            ),
            live("BTN", "call", 2),
        ],
        {0: ("SB",), 1: ("BB",)},
        blinds=f_blinds,
        ante_mode=single,
    )
    assert amounts(relabelled) == [5, 2], amounts(relabelled)

    # Conservation, on every hand above that settled.
    for ledger in (a, b, c, e, f, f2, b2, b2p):
        paid = sum(ledger.payouts.values()) + sum(ledger.refunds.values())
        assert round(paid + ledger.rake, 6) == round(
            sum(ledger.contributions.values()), 6
        ), "chips appeared or vanished"
