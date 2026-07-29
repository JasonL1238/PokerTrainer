"""Regressions for the two families round 11 found around the dependence rule.

FAMILY 1 -- the neutralisation set was a field list of two.

The rule replaced eight rounds of per-field disclosure conditions with a
measurement, and then measured exactly two declared inputs: the rake policy and
the dead money. The declared POT AWARDS were held constant across both passes,
documented as "the hand, not the policy". On a reconstructed hand that is not
true: the CV exporter emits no ``settlement`` key at all, so every award row was
typed into the same panel, by the same operator, in the same save as the rake --
and it is the single input the derived payouts and therefore the reported hero
result are computed from. One dropdown moved the recorded hero result by the
whole pot in either direction with no measurement, no disclosure, no correction
record and an EMPTY blocker tuple, while a declared rake of the same chips was
named, measured and blocked.

The repair is not "add awards to the list". ``_Declaration`` is now the complete
set of inputs a cross-check pass takes from what somebody declared rather than
from what was recorded, and
``test_a_neutral_declaration_derives_a_ledger_from_the_recording_alone`` asserts
that completeness as a property, so an input added to the derived side later
without being added to the declaration fails a test instead of opening a hole.

FAMILY 2 -- "which predicate does this consumer read?" was an enumerated list.

``_accounting_is_established`` was introduced to stop non-Study surfaces printing
an assumption-dependent reconciliation as fact, and was applied to the surfaces
one adversary demonstrated. Six consumers were left reading ``is_authoritative``,
including the session win rate, the hero-result column of every list view and
solver eligibility, so a 72-chip fabrication was published as a reconciled result
on a hand Study refused. It also never consulted the attestation, so confirming
an assumption cleared the Study blocker and re-enabled nothing, and a manual hand
carrying an ordinary room rake had coaching disabled with no control anywhere
that could ever enable it.

The repair is one definition in one place plus
``test_no_consumer_decides_on_is_authoritative_alone``, which walks the source of
every module and fails on a new reader of the raw flag.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

import app as app_module
from poker_tracker.math.accounting import RakePolicy
from poker_tracker.math.analytics import compute_session_stats
from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    CompletionEvidence,
    dump_completion_evidence,
    parse_completion_evidence,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import import_session
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    HandSettlement,
    Session,
    SettlementEntry,
)
from poker_tracker.services import hand_accounting
from poker_tracker.services.hand_accounting import (
    POT_AWARD_INPUT,
    RAKE_POLICY_INPUT,
    _cross_check,
    _Declaration,
    _declaration_fingerprint,
    _format_chips,
    _load_hand_records,
    attest_assumption,
    persist_reconciliation,
    reconcile_persisted_hand,
)
from poker_tracker.services.study_readiness import (
    accounting_is_established,
    evaluate_study_readiness,
)
from poker_tracker.solver.eligibility import prepare_solver_spot
from tests.conftest import attest_declared_assumptions

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSUMPTION_BLOCKER = "ACCOUNTING_ASSUMPTION_DEPENDENT"


def _clean_evidence(**overrides: object) -> dict[str, object]:
    payload = dump_completion_evidence(
        CompletionEvidence(
            evidence_version=EVIDENCE_SCHEMA_VERSION,
            partial_start=False,
            partial_end=False,
            terminal_event="showdown",
            boundary_confidence=0.92,
            layout_supported=True,
            table_size=6,
        )
    )
    payload.update(overrides)
    return payload


def _open_db(tmp_path: Path, name: str) -> PokerDatabase:
    db = PokerDatabase(str(tmp_path / name))
    db.init_db()
    return db


def _seed(
    db: PokerDatabase,
    *,
    seats: int = 2,
    bet: float = 40.0,
    hero_bb_won: float | None = None,
    pot_size: float | None = None,
    source_type: str = "cv_import",
    completion_status: str = "complete",
    evidence: dict[str, object] | None = None,
    board_cards: str = "Qd 7s 2c",
    stacks: tuple[float, ...] | None = None,
) -> Hand:
    """``seats`` seats each commit ``bet``; nothing is declared about the settlement."""
    session = db.create_session(Session(name="Declared", date_played=date(2026, 1, 1)))
    assert session.id is not None
    keys = ["hero", "villain", "third", "fourth"][:seats]
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            table_size=6,
            hero_cards="Ah Qs",
            board_cards=board_cards,
            pot_size=pot_size,
            hero_bb_won=hero_bb_won,
            source_type=source_type,  # type: ignore[arg-type]
            completion_status=completion_status,  # type: ignore[arg-type]
            completion_evidence=_clean_evidence() if evidence is None else evidence,
        )
    )
    assert hand.id is not None
    for index, key in enumerate(keys):
        db.create_hand_player(
            HandPlayer(
                hand_id=hand.id,
                player_key=key,
                player_name=key.capitalize(),
                is_hero=key == "hero",
                starting_stack=1000 if stacks is None else stacks[index],
            )
        )
    for index, key in enumerate(keys, start=1):
        db.create_action(
            Action(
                hand_id=hand.id,
                player_key=key,
                street="river",
                action_index=index,
                player_name=key.capitalize(),
                action_type="bet" if index == 1 else "call",
                amount=bet if stacks is None else min(bet, stacks[index - 1]),
            )
        )
    return hand


def _declare(
    db: PokerDatabase,
    hand: Hand,
    awards: tuple[tuple[str, int, float | None], ...],
    **settlement: object,
) -> None:
    assert hand.id is not None
    db.replace_settlement_entries(
        hand.id,
        [
            SettlementEntry(
                hand_id=hand.id,
                entry_type="award",
                pot_index=pot_index,
                player_key=key,
                player_name=key.capitalize(),
                amount=amount,
                entry_order=order,
            )
            for order, (key, pot_index, amount) in enumerate(awards, start=1)
        ],
    )
    db.upsert_hand_settlement(HandSettlement(hand_id=hand.id, **settlement))  # type: ignore[arg-type]


def _readiness(db: PokerDatabase, hand_id: int, *, user_confirmed: bool = True):
    hand = db.fetch_hand(hand_id)
    assert hand is not None
    accounting = reconcile_persisted_hand(db, hand_id)
    return evaluate_study_readiness(
        hand,
        accounting=accounting,
        hand_issues=db.fetch_hand_issues(hand_id=hand_id),
        coaching_reviews=db.fetch_coaching_reviews_by_hand(hand_id),
        solver_runs=db.fetch_solver_runs_by_hand(hand_id),
        user_confirmed=user_confirmed,
    )


def _named(result: object, input_name: str):
    (found,) = [
        item
        for item in result.assumption_dependence  # type: ignore[attr-defined]
        if item.input_name == input_name
    ]
    return found


# ---------------------------------------------------------------------------
# Family 1. The declared pot awards are a declared settlement input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("winner", ["hero", "villain"])
def test_a_first_declared_winner_is_measured_disclosed_and_blocking(
    tmp_path: Path, winner: str
) -> None:
    """The demonstrated instance: one dropdown, the whole pot, in either direction.

    A freshly imported hand records no ``pot_size`` and no ``hero_bb_won`` -- the
    exporter emits neither and ``import_session`` never reconciles -- so nothing
    contradicts whatever winner is typed in, and the hero result the product
    reports swings by the entire 80-chip pot between the two saves below.

    BEFORE this change: both saves land ``is_ready=True`` with an EMPTY blocker
    tuple, zero measured dependence, zero corrections and zero warning codes.
    """
    db = _open_db(tmp_path, f"winner_{winner}.db")
    hand = _seed(db)
    assert hand.id is not None

    before = _readiness(db, hand.id)
    assert before.codes() == ("ACCOUNTING_NOT_AUTHORITATIVE",)

    _declare(db, hand, ((winner, 0, None),))
    result = persist_reconciliation(db, hand.id)
    assert result.is_authoritative is True

    dependence = _named(result, POT_AWARD_INPUT)
    assert winner in dependence.declared
    assert dependence.neutral == "no declared winner"
    assert dict(dependence.deltas)["payout"] == pytest.approx(80.0)
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True

    # And the control the blocker names clears it, for this declaration only.
    assert dependence.code in attest_declared_assumptions(db, hand.id)
    assert _readiness(db, hand.id).is_ready is True

    # Re-declaring the other winner lapses the attestation without any other
    # field moving: the code carries who was paid.
    other = "villain" if winner == "hero" else "hero"
    _declare(db, hand, ((other, 0, None),))
    persist_reconciliation(db, hand.id)
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True
    db.close()


def test_a_neutral_declaration_derives_a_ledger_from_the_recording_alone(
    tmp_path: Path,
) -> None:
    """The completeness property, which is what replaces the field list.

    A declared input the neutral pass still reads is an input the rule cannot
    measure, and that is exactly how the declared awards escaped for eleven
    rounds. So rather than listing which inputs exist, this asserts the property
    the list was supposed to have: with the declaration withdrawn, the derived
    ledger is a function of the players and the action line and of nothing in
    ``hand_settlements`` or ``settlement_entries`` at all.

    It sweeps wildly different settlement rows and award sets over one unchanged
    recording. Every fully-neutral ledger must be identical.
    """
    db = _open_db(tmp_path, "neutral.db")
    hand = _seed(db, seats=3, hero_bb_won=None, pot_size=None)
    assert hand.id is not None

    def neutral_ledger() -> tuple[object, ...]:
        records = _load_hand_records(db, hand.id)  # type: ignore[arg-type]
        ledger = _cross_check(records, _Declaration.neutral()).ledger
        return (
            ledger.gross_pot,
            ledger.rake,
            ledger.net_pot,
            tuple(sorted(ledger.payouts.items())),
            tuple(sorted(ledger.net_results.items())),
            tuple(sorted(ledger.refunds.items())),
            ledger.is_settled,
            ledger.is_balanced,
            ledger.is_legal,
        )

    baseline = neutral_ledger()
    declarations: list[tuple[tuple[tuple[str, int, float | None], ...], dict[str, object]]] = [
        ((), {}),
        ((("hero", 0, None),), {}),
        ((("hero", 0, 120.0),), {"rake_rate": 0.9, "rake_cap": 3.0}),
        (
            (("villain", 0, None), ("hero", 0, None)),
            {"rake_rate": 0.5, "rake_rounding_unit": 5.0, "dead_money": 250.0},
        ),
        (
            (("third", 0, 7.0), ("hero", 0, 7.0), ("villain", 0, 7.0)),
            {"rake_rate": 1.0, "no_flop_no_drop": True, "dead_money": 0.25},
        ),
        ((("hero", 0, None), ("villain", 1, None)), {"rake_rate": 0.03, "rake_cap": 1.0}),
    ]
    for awards, settlement in declarations:
        _declare(db, hand, awards, **settlement)
        assert neutral_ledger() == baseline, (awards, settlement)
    db.close()


def test_a_re_ordered_chop_is_a_different_declaration(tmp_path: Path) -> None:
    """Family instance 2: the odd chip, which no settlement column records.

    Two winners share 81 chips. Which of them receives the odd chip is decided by
    the order of the award rows and by nothing else, so re-ordering them moves a
    chip between two seats with every settlement field identical. The measured
    code must not be inherited across that.
    """
    db = _open_db(tmp_path, "chop.db")
    hand = _seed(db, seats=3, bet=27.0)
    assert hand.id is not None

    _declare(db, hand, (("hero", 0, None), ("villain", 0, None)))
    first = persist_reconciliation(db, hand.id)
    assert first.ledger.payouts["hero"] == pytest.approx(41.0)
    assert first.ledger.payouts["villain"] == pytest.approx(40.0)
    attest_declared_assumptions(db, hand.id)
    assert _readiness(db, hand.id).is_ready is True

    _declare(db, hand, (("villain", 0, None), ("hero", 0, None)))
    second = persist_reconciliation(db, hand.id)
    assert second.ledger.payouts["hero"] == pytest.approx(40.0)
    assert _named(second, POT_AWARD_INPUT).code != _named(first, POT_AWARD_INPUT).code
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True
    db.close()


def test_a_side_pot_winner_is_measured_on_the_pot_it_was_declared_for(
    tmp_path: Path,
) -> None:
    """Family instance 3: a side pot, where the award set is two rows on two pots."""
    db = _open_db(tmp_path, "side.db")
    hand = _seed(db, seats=3, bet=10.0, stacks=(10.0, 1000.0, 1000.0))
    assert hand.id is not None
    for action in db.fetch_actions_by_hand(hand.id):
        assert action.id is not None
        if action.player_key == "hero":
            continue
        db.update_action(
            action.model_copy(
                update={
                    "amount": 40.0,
                    "action_type": "raise" if action.player_key == "villain" else "call",
                }
            ),
            correction_notes="Two seats continue past the short stack's all-in.",
        )

    _declare(db, hand, (("hero", 0, None), ("villain", 1, None)))
    result = persist_reconciliation(db, hand.id)
    assert result.is_authoritative is True
    dependence = _named(result, POT_AWARD_INPUT)
    assert "pot 0 -> hero" in dependence.declared
    assert "pot 1 -> villain" in dependence.declared
    assert dict(dependence.deltas)["payout"] > 0
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True

    # Handing the side pot to the third seat instead is a different declaration
    # moving different chips, and the earlier attestation does not cover it.
    attest_declared_assumptions(db, hand.id)
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is False
    _declare(db, hand, (("hero", 0, None), ("third", 1, None)))
    persist_reconciliation(db, hand.id)
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True
    db.close()


def test_an_imported_payload_cannot_land_its_own_declared_winner_unmeasured(
    tmp_path: Path,
) -> None:
    """Family instance 4: the declaration arrives in a payload rather than a click.

    ``import_session`` stamps every hand it lands, so the importing operator owes
    an attestation for a winner they never entered -- the same rule the rake goes
    through, through the same predicate.
    """
    db = _open_db(tmp_path, "payload.db")
    payload = {
        "export_version": 5,
        "session": {"name": "Imported", "date_played": "2026-01-01"},
        "hands": [
            {
                "hand": {
                    "hand_number": 1,
                    "table_size": 6,
                    "hero_cards": "Ah Qs",
                    "board_cards": "Qd 7s 2c",
                    "source_type": "cv_import",
                    "completion_status": "complete",
                    "completion_evidence": _clean_evidence(),
                    "tags": [],
                },
                "players": [
                    {
                        "player_key": "hero",
                        "player_name": "Hero",
                        "is_hero": True,
                        "starting_stack": 1000,
                    },
                    {
                        "player_key": "villain",
                        "player_name": "Villain",
                        "is_hero": False,
                        "starting_stack": 1000,
                    },
                ],
                "actions": [
                    {
                        "player_key": "hero",
                        "player_name": "Hero",
                        "street": "river",
                        "action_index": 1,
                        "action_type": "bet",
                        "amount": 40.0,
                        "amount_semantics": "incremental",
                    },
                    {
                        "player_key": "villain",
                        "player_name": "Villain",
                        "street": "river",
                        "action_index": 2,
                        "action_type": "call",
                        "amount": 40.0,
                        "amount_semantics": "incremental",
                    },
                ],
                "settlement": {"status": "reconciled"},
                "settlement_entries": [
                    {
                        "entry_type": "award",
                        "pot_index": 0,
                        "player_key": "hero",
                        "player_name": "Hero",
                        "entry_order": 1,
                    }
                ],
            }
        ],
    }
    import_session(db, payload)
    (landed,) = db.fetch_all_hands()
    assert landed.id is not None
    result = reconcile_persisted_hand(db, landed.id)
    assert result.is_authoritative is True
    assert _named(result, POT_AWARD_INPUT).declared == "pot 0 -> hero"
    assert _readiness(db, landed.id).has(ASSUMPTION_BLOCKER) is True
    db.close()


def test_withdrawing_the_awards_is_a_dependence_the_figures_cannot_show(
    tmp_path: Path,
) -> None:
    """The VERDICT half, on a real hand rather than an injected pass.

    Round 11 reported the verdict half as dead: over 27,000 shapes it was never
    the sole reason for a dependence, because every figure the cross-check
    compares was already in ``_ledger_deltas``. That is no longer true. Declare a
    policy that takes the whole pot and every payout is zero already, so
    withdrawing the winners leaves gross, rake, net, payout and hero all standing
    still while the ledger goes unsettled -- the dependence exists and not one
    chip moved.

    Deleting ``if not neutralised.reconciles: return True`` from ``_is_dependent``
    makes this test fail.
    """
    db = _open_db(tmp_path, "verdict.db")
    hand = _seed(db, hero_bb_won=-40.0, pot_size=80.0)
    assert hand.id is not None
    _declare(db, hand, (("hero", 0, 0.0),), rake_rate=1.0)
    result = persist_reconciliation(db, hand.id)

    assert result.is_authoritative is True
    assert result.ledger.rake == pytest.approx(80.0)
    assert result.ledger.payouts["hero"] == pytest.approx(0.0)

    dependence = _named(result, POT_AWARD_INPUT)
    assert dependence.deltas == ()
    assert dependence.code.endswith(":verdict-only")
    assert "without moving any reported figure" in dependence.describe()
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True

    records = _load_hand_records(db, hand.id)
    baseline = _cross_check(records, records.declaration)
    without = _cross_check(records, records.declaration.without_awards())
    assert hand_accounting._ledger_deltas(
        baseline.ledger, without.ledger, records.hero_key
    ) == ()
    assert without.reconciles is False
    db.close()


# ---------------------------------------------------------------------------
# The measurement itself: the code has to identify what it measured
# ---------------------------------------------------------------------------


def test_the_measured_movement_string_round_trips_and_never_collides() -> None:
    """An attestation to a quantity has to name the quantity uniquely.

    ``f"{value:+.6f}"`` rendered every movement in (1e-9, 5e-7) as "+0", so
    distinct measured quantities above the dependence tolerance shared one
    acknowledgement string while the module's stated invariant was that the
    measurement is in the code.
    """
    values = [
        1e-8,
        4.9e-7,
        -1e-8,
        -4.9e-7,
        0.005,
        1 / 3,
        40.0,
        -80.0,
        1e12,
        1e12 + 1e-8,
        80.01,
        0.01,
    ]
    rendered = [_format_chips(value) for value in values]
    assert len(set(rendered)) == len(set(values))
    for value, text in zip(values, rendered, strict=True):
        assert float(text) == value
        assert text[0] in "+-"


def test_the_fingerprint_separates_every_context_term_it_digests(
    tmp_path: Path,
) -> None:
    """Each term of the attestation fingerprint is load-bearing, and pinned as such.

    Mutation testing found the GROSS POT term survivable: no test anywhere failed
    when it was dropped. It is not dead weight -- with a binding rake cap the
    measured movement is byte-identical across a corrected action line, because
    the cap pins the rake either way, so the gross term is the only thing telling
    the two declarations apart. The end-to-end half below is that exact shape;
    the unit half pins the other three terms the same way.
    """
    db = _open_db(tmp_path, "fingerprint.db")
    hand = _seed(db, hero_bb_won=None, pot_size=None)
    assert hand.id is not None
    _declare(db, hand, (("hero", 0, None),), rake_rate=0.5, rake_cap=10.0)
    small = persist_reconciliation(db, hand.id)
    assert small.ledger.gross_pot == pytest.approx(80.0)
    small_rake = _named(small, RAKE_POLICY_INPUT)
    assert small_rake.code.endswith("rake+10.0|net-10.0|payout+10.0|hero-10.0")
    attest_declared_assumptions(db, hand.id)
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is False

    for action in db.fetch_actions_by_hand(hand.id):
        assert action.id is not None
        db.update_action(
            action.model_copy(update={"amount": 100.0}),
            correction_notes="Corrected the river amounts.",
        )
    grown = persist_reconciliation(db, hand.id)
    assert grown.ledger.gross_pot == pytest.approx(200.0)
    grown_rake = _named(grown, RAKE_POLICY_INPUT)

    # The measurement is identical; only the pot it was taken on differs.
    assert grown_rake.code.rsplit(":", 1)[1] == small_rake.code.rsplit(":", 1)[1]
    assert grown_rake.code != small_rake.code
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True

    # Unit half: each context term on its own changes the digest.
    records = _load_hand_records(db, hand.id)
    baseline = _cross_check(records, records.declaration)
    reference = _declaration_fingerprint(records, baseline, "declared", "neutral")
    assert _declaration_fingerprint(records, baseline, "declared!", "neutral") != reference
    assert _declaration_fingerprint(records, baseline, "declared", "neutral!") != reference

    other_gross = _cross_check(
        records,
        _Declaration(
            rake=RakePolicy(),
            dead_money=17.0,
            awards=records.awards,
        ),
    )
    assert (
        _declaration_fingerprint(records, other_gross, "declared", "neutral") != reference
    )
    assert records.settlement is not None
    db.upsert_hand_settlement(
        records.settlement.model_copy(update={"dead_money": 3.0})
    )
    dead_records = _load_hand_records(db, hand.id)
    assert (
        _declaration_fingerprint(dead_records, baseline, "declared", "neutral")
        != reference
    )
    db.close()


# ---------------------------------------------------------------------------
# Family 2. One predicate, and every consumer reads it
# ---------------------------------------------------------------------------


# Where reading the raw ledger verdict is the right question, with the reason.
_RAW_VERDICT_READERS = {
    # ACCOUNTING_NOT_AUTHORITATIVE is the raw-verdict blocker; the assumption
    # blocker beside it is what adds the declaration question, and
    # `accounting_is_established` is the composition of the two.
    ("poker_tracker/services/study_readiness.py", "_accounting_blockers"),
    ("poker_tracker/services/study_readiness.py", "accounting_is_established"),
    # The Study accounting panel explains the raw verdict; it is the one surface
    # where the difference between the two questions is rendered side by side.
    ("app.py", "_render_accounting_status"),
    # Confirmation-checkbox key material: the raw verdict is part of what
    # identifies the facts a tick attests to, not a decision about publishing.
    ("app.py", "_study_evidence_digest"),
    # The settlement editor's own save flash, reporting what the save achieved.
    ("app.py", "show_accounting_editor"),
    # Coach Review's fallback branch, reached only once nothing is unattested,
    # to tell an unreconciled hand apart from an unanswered one.
    ("app.py", "show_hand_coach_review"),
}


class _AttributeReadWalker(ast.NodeVisitor):
    """Every read of ``.is_authoritative``, with the function it happens in."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.found: set[tuple[str, str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    # An `async def` reader would otherwise be attributed to the enclosing scope
    # or to `<module>`, which would let it slip past the allow-list under a name
    # that is already on it.
    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "is_authoritative":
            self.found.add(
                (self.module, self.scope[-1] if self.scope else "<module>")
            )
        self.generic_visit(node)


def _is_authoritative_readers() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in [
        REPO_ROOT / "app.py",
        *sorted((REPO_ROOT / "poker_tracker").rglob("*.py")),
    ]:
        walker = _AttributeReadWalker(str(path.relative_to(REPO_ROOT)))
        walker.visit(ast.parse(path.read_text(encoding="utf-8")))
        found |= walker.found
    return found


def test_no_consumer_decides_on_is_authoritative_alone() -> None:
    """The family regression: a new consumer cannot quietly rejoin the old predicate.

    ``_accounting_is_established`` was added and then applied to the surfaces one
    adversary demonstrated. Six consumers stayed on ``is_authoritative`` --
    ``math.analytics.compute_session_stats``, ``_hands_with_accounting_results``,
    the Overview featured pot, ``_hero_ledger_result``, the Math Review defaults
    and ``solver.eligibility.prepare_solver_spot`` -- and PLAN.md recorded the
    repair as complete anyway. An enumerated repair to an enumerated list is what
    the previous ten rounds kept re-finding, so the list is now enforced rather
    than written down.

    Adding a reader means adding it here WITH ITS REASON, which is the point: the
    raw ledger verdict is the right question in six places and the wrong one
    everywhere else. (It said "seven" for two rounds while the enforced set held
    six; the count is derived here so the sentence cannot drift from the set.)
    """
    assert _is_authoritative_readers() == _RAW_VERDICT_READERS
    assert len(_RAW_VERDICT_READERS) == 6


def test_every_derived_figure_surface_refuses_an_unattested_reconciliation(
    tmp_path: Path,
) -> None:
    """One hand, every surface that publishes a derived figure, in one pass.

    The hand records nothing -- a null ``pot_size`` and a null ``hero_bb_won``,
    which is the ordinary state of a freshly imported hand -- and a declared 90%
    rake turns the +40 its action line supports into -32. Study refuses it. Every
    other surface used to publish the -32 as a reconciled fact.
    """
    db = _open_db(tmp_path, "consumers.db")
    hand = _seed(db)
    assert hand.id is not None and hand.session_id is not None
    _declare(db, hand, (("hero", 0, None),), rake_rate=0.9)
    accounting = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None

    assert accounting.is_authoritative is True
    assert accounting.ledger.net_results["hero"] == pytest.approx(-32.0)
    assert _readiness(db, hand.id).has(ASSUMPTION_BLOCKER) is True

    stats = compute_session_stats(db, hand.session_id)
    assert stats.total_hero_bb == 0.0
    assert stats.reconciled_result_count == 0
    assert stats.observed_result_count == 0

    (listed,) = app_module._hands_with_accounting_results(db, [stored])
    assert listed.hero_bb_won is None
    assert listed.derived_result_substituted is False

    players = db.fetch_players_by_hand(hand.id)
    assert app_module._hero_ledger_result(stored, accounting, players, None) is None
    assert accounting_is_established(stored, accounting) is False
    assert app_module._accounting_prompt_math_facts(stored, accounting) == {}

    spot = prepare_solver_spot(
        stored, players, db.fetch_actions_by_hand(hand.id), accounting
    )
    assert any("settlement assumption" in reason for reason in spot.eligibility.reasons)

    # Attesting to the declaration is what publishes it, and it publishes
    # everywhere at once.
    attest_declared_assumptions(db, hand.id)
    answered = db.fetch_hand(hand.id)
    assert answered is not None
    assert accounting_is_established(answered, accounting) is True
    assert compute_session_stats(db, hand.session_id).reconciled_result_count == 1
    (published,) = app_module._hands_with_accounting_results(db, [answered])
    assert published.hero_bb_won == pytest.approx(-32.0)
    assert app_module._accounting_prompt_math_facts(answered, accounting)
    later = prepare_solver_spot(
        answered, players, db.fetch_actions_by_hand(hand.id), accounting
    )
    assert not any("settlement assumption" in r for r in later.eligibility.reasons)
    db.close()


def test_a_manual_hand_with_a_room_rake_is_established_with_no_control_to_press(
    tmp_path: Path,
) -> None:
    """The dead end that had the operator's own truthful entry on the wrong side.

    A hand typed in here, with the room's ordinary 10% rake, is study-ready and
    exempt from the assumption blocker by design -- and no 'Confirm this
    assumption' control is drawn for it, because there is nothing for the
    operator to attest to that they did not already state. The predicate every
    coaching and solver gate reads must therefore treat it as established.

    BEFORE this change: ``_accounting_is_established`` returned False for any
    hand whose rake actually takes chips, so on this hand coaching was disabled,
    the provider's math facts were empty, and the message above the disabled
    button named an action the product does not offer here.
    """
    db = _open_db(tmp_path, "manual.db")
    hand = _seed(
        db,
        bet=10.0,
        hero_bb_won=8.0,
        pot_size=20.0,
        source_type="manual",
        completion_status="not_applicable",
        evidence={},
    )
    assert hand.id is not None
    _declare(db, hand, (("hero", 0, 18.0),), rake_rate=0.1)
    accounting = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None

    assert accounting.is_authoritative is True
    assert accounting.assumption_dependence, "still measured, and still displayed"
    assert _readiness(db, hand.id, user_confirmed=False).is_ready is True
    assert accounting_is_established(stored, accounting) is True
    assert app_module._accounting_prompt_math_facts(stored, accounting)
    assert app_module._hero_ledger_result(
        stored, accounting, db.fetch_players_by_hand(hand.id), None
    ) == pytest.approx(8.0)
    db.close()


def test_confirming_an_assumption_re_enables_everything_it_disabled(
    tmp_path: Path,
) -> None:
    """A blocker's clearing action has to clear the thing it was blocking.

    The attestation cleared the Study blocker and nothing else: the measurement
    is re-derived from the chips on every read, so a predicate keyed on its mere
    existence stayed False forever. An operator who did exactly what the product
    asked held a study-ready hand that could never be coached.
    """
    db = _open_db(tmp_path, "reenable.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0)
    assert hand.id is not None
    _declare(db, hand, (("hero", 0, 40.0),), rake_rate=0.5)
    accounting = persist_reconciliation(db, hand.id)
    stored = db.fetch_hand(hand.id)
    assert stored is not None
    assert accounting_is_established(stored, accounting) is False

    attest_declared_assumptions(db, hand.id)
    attested = db.fetch_hand(hand.id)
    assert attested is not None
    later = reconcile_persisted_hand(db, hand.id)
    assert later.assumption_dependence, "the measurement is not erased by the answer"
    assert accounting_is_established(attested, later) is True
    assert app_module._accounting_prompt_math_facts(attested, later)
    assert app_module.unattested_assumption_dependence(attested, later) == ()

    # The hand history stops contradicting the panel beside it.
    issues = app_module._accounting_prompt_issues(later, None)
    assert any("declared settlement assumption" in issue for issue in issues)
    db.close()


def test_the_attestation_lapses_when_the_declaration_changes(tmp_path: Path) -> None:
    """The other half of the previous test: answered is not answered forever."""
    db = _open_db(tmp_path, "lapse.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0)
    assert hand.id is not None
    _declare(db, hand, (("hero", 0, 40.0),), rake_rate=0.5)
    persist_reconciliation(db, hand.id)
    attest_declared_assumptions(db, hand.id)
    attested = db.fetch_hand(hand.id)
    assert attested is not None
    assert accounting_is_established(
        attested, reconcile_persisted_hand(db, hand.id)
    ) is True

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    db.update_hand_facts(
        stored.model_copy(update={"hero_bb_won": 20.0}),
        correction_notes="Corrected the recorded result.",
    )
    _declare(db, hand, (("hero", 0, 60.0),), rake_rate=0.25)
    relapsed = persist_reconciliation(db, hand.id)
    reread = db.fetch_hand(hand.id)
    assert reread is not None
    assert relapsed.is_authoritative is True
    assert accounting_is_established(reread, relapsed) is False
    assert parse_completion_evidence(reread.completion_evidence).confirmed_assumption_codes
    db.close()


# ---------------------------------------------------------------------------
# Family 3. One door to the attestation writer, and it measures before it writes
# ---------------------------------------------------------------------------


# The persistence writer can only validate the SHAPE of a dependence code -- a
# prefix test -- because measuring one needs the ledger and `hand_accounting`
# imports `db` rather than the other way round. `attest_assumption` is where the
# measurement happens, so it is the only place allowed to call the writer.
_ATTESTATION_WRITER_CALLERS = {
    ("poker_tracker/services/hand_accounting.py", "attest_assumption"),
}


class _AttestationCallWalker(ast.NodeVisitor):
    """Every call of ``acknowledge_accounting_assumption``, with its function."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.scope: list[str] = []
        self.found: set[tuple[str, str]] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        if name == "acknowledge_accounting_assumption":
            self.found.add((self.module, self.scope[-1] if self.scope else "<module>"))
        self.generic_visit(node)


def test_the_attestation_writer_has_exactly_one_caller() -> None:
    """A writer that cannot check its own input must not be callable directly.

    `db.acknowledge_accounting_assumption` accepted any well-formed code, filed a
    `hand_corrections` row attesting to it, and -- because at most one
    attestation survives per declared input -- evicted the operator's GENUINE
    attestation for that input on the way past. The check that closes it needs a
    ledger, which that layer cannot have, so the check lives one layer up and this
    pins that no consumer can go round it.
    """
    found: set[tuple[str, str]] = set()
    for path in [
        REPO_ROOT / "app.py",
        *sorted((REPO_ROOT / "poker_tracker").rglob("*.py")),
    ]:
        walker = _AttestationCallWalker(str(path.relative_to(REPO_ROOT)))
        walker.visit(ast.parse(path.read_text(encoding="utf-8")))
        found |= walker.found
    assert found == _ATTESTATION_WRITER_CALLERS


def test_a_code_naming_no_current_dependence_is_refused_and_evicts_nothing(
    tmp_path: Path,
) -> None:
    """The behaviour behind the pin above, on the shape that was demonstrated."""
    db = _open_db(tmp_path, "fabricated.db")
    hand = _seed(db, hero_bb_won=0.0, pot_size=80.0)
    assert hand.id is not None
    _declare(db, hand, (("hero", 0, 40.0),), rake_rate=0.5)
    persist_reconciliation(db, hand.id)
    genuine = attest_declared_assumptions(db, hand.id, only="rake_policy")
    assert genuine

    fabricated = "declared_settlement_dependence:rake_policy:0000000000:rake+0"
    assert attest_assumption(db, hand.id, fabricated) is False

    stored = db.fetch_hand(hand.id)
    assert stored is not None
    confirmed = parse_completion_evidence(stored.completion_evidence).confirmed_assumption_codes
    assert fabricated not in confirmed
    assert set(genuine) <= set(confirmed)
    assert not [
        correction
        for correction in db.fetch_hand_corrections(hand.id)
        if fabricated in str(correction.notes)
    ]
    db.close()
