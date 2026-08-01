"""What a solver run must admit about itself, and what must outlive its files.

Every label in this module is produced today. None of it was asserted anywhere,
so the whole set could be deleted in a refactor with a green suite: the rake
approximation, the disabled suit isomorphism, the built-in ranges being
estimates rather than solved preflop GTO, and the separation between a range the
operator supplied and one the product guessed. These tests assert the producer,
not a hand-built object that happens to contain the sentence.

The retention half is the same argument about files: a run whose artifacts are
gone still shows its frequencies, and until the settings behind them lived on the
row there was nothing left to say what had been solved.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from poker_tracker.coaching.coaching_prompts import build_hand_review_prompt
from poker_tracker.coaching.solver_grounding import validate_solver_coaching_response
from poker_tracker.math.accounting import ActionSnapshot, HandLedger
from poker_tracker.persistence import db as db_module
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    Hand,
    HandPlayer,
    Session,
    SolverRangeProfile,
    SolverRun,
)
from poker_tracker.services.hand_accounting import AccountingReconciliation
from poker_tracker.solver import jobs as solver_jobs
from poker_tracker.solver import storage as solver_storage
from poker_tracker.solver.eligibility import prepare_solver_spot
from poker_tracker.solver.jobs import start_solver_job
from poker_tracker.solver.models import (
    ActionFrequency,
    SolverEvidence,
    SolverRunParameters,
    current_run_parameters,
)
from poker_tracker.solver.ranges import (
    BUILTIN_RANGE_PROFILES,
    resolve_custom_range,
    resolve_profile,
    resolve_selected_profile,
)
from poker_tracker.solver.run_job import run_solver_job
from poker_tracker.solver.storage import (
    backend_identity_assumption,
    missing_run_artifacts,
    remove_solver_run_artifacts,
    resolved_backend_identity,
)
from poker_tracker.solver.texassolver import PINNED_CONSOLE_COMMIT

RAKE_BB = 0.5
HERO_RANGE = "AhQs,AsQh"
VILLAIN_RANGE = "22+,A2s+"
# A minimal dump for rows that are seeded directly rather than solved.
STUB_DUMP = {
    "node_type": "action_node",
    "actions": ["CHECK", "BET 3.75"],
    "strategy": {
        "actions": ["CHECK", "BET 3.75"],
        "strategy": {"AhQs": [0.7, 0.3], "AsQh": [0.6, 0.4]},
    },
}


def _snapshot(
    index: int,
    player: str,
    street: str,
    kind: str,
    *,
    pot_before: float,
    pot_after: float,
    stack_before: float,
    stack_after: float,
    amount: float = 0,
) -> ActionSnapshot:
    return ActionSnapshot(
        index=index,
        player=player,
        street=street,
        kind=kind,
        amount=amount,
        pot_before=pot_before,
        pot_after=pot_after,
        stack_before=stack_before,
        stack_after=stack_after,
        to_call_before=0,
        call_increment=0,
        street_contribution_after=amount,
        hand_contribution_after=amount,
        effective_stack_before=min(stack_before, 97.5),
        effective_stack_range_before=(min(stack_before, 97.5), min(stack_before, 97.5)),
        spr_before=(min(stack_before, 97.5) / pot_before if pot_before else None),
        spr_range_before=None,
        active_players=("hero", "villain"),
    )


def _hand_facts(hand_id: int = 1, *, rake: float = RAKE_BB):
    """One raked, reconciled, heads-up flop spot: BB checks to the BTN raiser."""

    hand = Hand(
        id=hand_id,
        session_id=1,
        hand_number=1,
        game_type="NLHE cash",
        table_size=6,
        effective_stack=100,
        hero_position="BB",
        hero_cards="Ah Qs",
        board_cards="Qd 7s 2c",
    )
    players = [
        HandPlayer(
            hand_id=hand_id,
            player_key="hero",
            player_name="Hero",
            position="BB",
            starting_stack=100,
            is_hero=True,
        ),
        HandPlayer(
            hand_id=hand_id,
            player_key="villain",
            player_name="Villain",
            position="BTN",
            starting_stack=100,
        ),
    ]
    actions = [
        Action(
            hand_id=hand_id,
            player_key="hero",
            player_name="Hero",
            position="BB",
            street="preflop",
            action_index=1,
            action_type="post_blind",
            amount=1,
        ),
        Action(
            hand_id=hand_id,
            player_key="villain",
            player_name="Villain",
            position="BTN",
            street="preflop",
            action_index=2,
            action_type="raise",
            amount=2.5,
        ),
        Action(
            hand_id=hand_id,
            player_key="hero",
            player_name="Hero",
            position="BB",
            street="preflop",
            action_index=3,
            action_type="call",
            amount=1.5,
        ),
        Action(
            hand_id=hand_id,
            player_key="hero",
            player_name="Hero",
            position="BB",
            street="flop",
            action_index=1,
            action_type="check",
        ),
        Action(
            hand_id=hand_id,
            player_key="villain",
            player_name="Villain",
            position="BTN",
            street="flop",
            action_index=2,
            action_type="bet",
            amount=3.75,
        ),
        Action(
            hand_id=hand_id,
            player_key="hero",
            player_name="Hero",
            position="BB",
            street="flop",
            action_index=3,
            action_type="call",
            amount=3.75,
        ),
    ]
    snapshots = (
        _snapshot(0, "hero", "preflop", "post_blind", pot_before=0, pot_after=1,
                  stack_before=100, stack_after=99, amount=1),
        _snapshot(1, "villain", "preflop", "raise", pot_before=1, pot_after=3.5,
                  stack_before=100, stack_after=97.5, amount=2.5),
        _snapshot(2, "hero", "preflop", "call", pot_before=3.5, pot_after=5,
                  stack_before=99, stack_after=97.5, amount=1.5),
        _snapshot(3, "hero", "flop", "check", pot_before=5, pot_after=5,
                  stack_before=97.5, stack_after=97.5),
        _snapshot(4, "villain", "flop", "bet", pot_before=5, pot_after=8.75,
                  stack_before=97.5, stack_after=93.75, amount=3.75),
        _snapshot(5, "hero", "flop", "call", pot_before=8.75, pot_after=12.5,
                  stack_before=97.5, stack_after=93.75, amount=3.75),
    )
    ledger = HandLedger(
        contributions={"hero": 6.25, "villain": 6.25},
        refunds={},
        payouts={"hero": 12.5},
        net_results={"hero": 6.25, "villain": -6.25},
        gross_pot=12.5,
        rake=rake,
        net_pot=12.5 - rake,
        pots=(),
        snapshots=snapshots,
        folded_players=(),
        warnings=(),
        legality_issues=(),
        is_settled=True,
        is_balanced=True,
        is_legal=True,
    )
    accounting = AccountingReconciliation(
        ledger=ledger,
        settlement=None,
        entries=(),
        issues=(),
        is_authoritative=True,
    )
    prepared = prepare_solver_spot(hand, players, actions, accounting)
    assert prepared.eligibility.eligible, prepared.eligibility.reasons
    assert prepared.spot is not None
    return hand, players, actions, accounting, prepared


_STUB_SOLVER = '''#!/usr/bin/env python3
"""Answers whatever range it is handed, so range coverage is never the variable."""
import json
import pathlib
import sys

command = pathlib.Path(sys.argv[2]).read_text().splitlines()
output = next(
    line.split(" ", 1)[1] for line in command if line.startswith("dump_result ")
)
combos = []
for line in command:
    if line.startswith(("set_range_ip ", "set_range_oop ")):
        for token in line.split(" ", 1)[1].split(","):
            combo = token.split(":", 1)[0].strip()
            if len(combo) == 4:
                combos.append(combo)
actions = ["CHECK", "BET 3.75"]
pathlib.Path(output).write_text(
    json.dumps(
        {
            "node_type": "action_node",
            "actions": actions,
            "strategy": {
                "actions": actions,
                "strategy": {combo: [0.7, 0.3] for combo in combos},
            },
        }
    )
)
print("Total exploitability 0.33 percent")
'''


def _install_stub_solver(tmp_path: Path, monkeypatch) -> Path:
    """A binary that answers the submitted range, so the worker path runs for real."""

    binary = tmp_path / "console_solver"
    binary.write_text(_STUB_SOLVER, encoding="utf-8")
    binary.chmod(0o755)
    evaluator = tmp_path / "resources" / "compairer" / "card5_dic_sorted.txt"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("test", encoding="utf-8")
    monkeypatch.setenv("TEXAS_SOLVER_PATH", str(binary))
    monkeypatch.delenv("TEXAS_SOLVER_RESOURCE_DIR", raising=False)
    monkeypatch.delenv("POKERTRAINER_SOLVER_MEMORY_GB", raising=False)
    monkeypatch.delenv("POKERTRAINER_SOLVER_THREADS", raising=False)
    return binary


def _isolate_run_directories(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "solver_runs"
    monkeypatch.setattr(solver_storage, "SOLVER_RUNS_DIR", root)

    def directory(run_id: int) -> Path:
        path = root / f"run_{run_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(solver_jobs, "solver_run_directory", directory)
    return root


class _DummyProcess:
    """Stands in for the detached worker so the test drives it in-process."""

    pid = 987654


def _block_worker_launch(monkeypatch) -> None:
    """Keep start_solver_job from detaching a worker, without disarming Popen.

    ``jobs.subprocess`` is the stdlib module itself, so patching its ``Popen``
    attribute would also disarm the worker's own launch of the solver, which
    these tests need to run for real.
    """

    class _NoLaunch:
        DEVNULL = subprocess.DEVNULL

        @staticmethod
        def Popen(*args, **kwargs):
            return _DummyProcess()

    monkeypatch.setattr(solver_jobs, "subprocess", _NoLaunch)


def _seeded_db(tmp_path: Path, name: str = "honesty.db"):
    db = PokerDatabase(tmp_path / name)
    db.init_db()
    session = db.create_session(Session(name="Solver honesty"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="NLHE cash", table_size=6)
    )
    return db, session, hand


def _started_run(
    db,
    spot,
    *,
    hero_source: str = "default",
    villain_range=None,
    extra_assumptions: list[str] | None = None,
):
    hero = resolve_profile(spot, spot.oop, BUILTIN_RANGE_PROFILES, source=hero_source)
    villain = villain_range or resolve_profile(
        spot, spot.ip, BUILTIN_RANGE_PROFILES, source="default"
    )
    return start_solver_job(
        db, spot, villain, hero, assumptions=extra_assumptions or []
    ), hero, villain


# ---------------------------------------------------------------------------
# Honesty: the producer, not a hand-built evidence object
# ---------------------------------------------------------------------------


def test_started_run_retains_every_assumption_the_phase_requires(tmp_path, monkeypatch):
    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    _, _, _, _, prepared = _hand_facts()
    db, _, hand = _seeded_db(tmp_path)
    spot = prepared.spot.model_copy(update={"hand_id": hand.id})

    run, _, _ = _started_run(
        db, spot, extra_assumptions=list(prepared.eligibility.warnings)
    )
    retained = "\n".join(run.assumptions)

    assert f"Recorded rake was {RAKE_BB:g} BB" in retained
    assert "no-rake equilibrium approximation" in retained
    assert "No-rake equilibrium approximation." in run.assumptions
    assert "Suit isomorphism is disabled" in retained
    assert "exact suit-specific combinations" in retained
    assert retained.count("estimated study input, not solved preflop GTO") == 2
    assert "IP range" in retained and "OOP range" in retained
    db.close()


def test_the_rake_label_tracks_the_recorded_rake(tmp_path, monkeypatch):
    """Negative control: the sentence is produced from the ledger, not always."""

    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    _, _, _, _, prepared = _hand_facts(rake=0)
    db, _, hand = _seeded_db(tmp_path)
    spot = prepared.spot.model_copy(update={"hand_id": hand.id})

    run, _, _ = _started_run(
        db, spot, extra_assumptions=list(prepared.eligibility.warnings)
    )
    retained = "\n".join(run.assumptions)

    assert "Recorded rake was" not in retained
    # The flat approximation is a property of the backend, not of this hand.
    assert "No-rake equilibrium approximation." in run.assumptions
    db.close()


def test_saved_user_range_is_not_labeled_an_estimate(tmp_path, monkeypatch):
    """The saved-range path end to end: a supplied range must read as supplied."""

    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    _, _, _, _, prepared = _hand_facts()
    db, _, hand = _seeded_db(tmp_path)
    spot = prepared.spot.model_copy(update={"hand_id": hand.id})
    db.create_solver_range_profile(
        SolverRangeProfile(
            name="BTN open, my own study range",
            notation="22+,A2s+,AQo+",
            table_size=6,
            position="BTN",
            scenario="rfi",
            pot_type="single_raised",
            stack_bb=100,
        )
    )
    saved = db.fetch_solver_range_profiles()[0]
    assert saved.id is not None
    # The same classification the Premade selector performs.
    source = "user" if saved.id is not None else "builtin"
    villain = resolve_selected_profile(spot, spot.ip, saved, source=source)
    assert villain.source == "user"
    assert villain.combo_count > 0

    run, _, _ = _started_run(db, spot, villain_range=villain)
    retained = "\n".join(run.assumptions)

    assert "'BTN open, my own study range' was explicitly supplied by the user" in retained
    assert "BTN open, my own study range' is an estimated study input" not in retained
    assert retained.count("estimated study input, not solved preflop GTO") == 1
    db.close()


def test_every_builtin_profile_is_labeled_an_estimate():
    """Guards the label against a profile added later without it."""

    assert BUILTIN_RANGE_PROFILES
    unlabeled = [
        profile.name
        for profile in BUILTIN_RANGE_PROFILES
        if not profile.name.endswith("· estimated")
        or "not solved preflop GTO" not in profile.description
    ]
    assert unlabeled == []


def test_assumptions_survive_the_worker_into_evidence_and_the_prompt(
    tmp_path, monkeypatch
):
    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    hand_facts, players, actions, _, prepared = _hand_facts()
    db, session, hand = _seeded_db(tmp_path)
    spot = prepared.spot.model_copy(update={"hand_id": hand.id})
    run, _, _ = _started_run(
        db, spot, extra_assumptions=list(prepared.eligibility.warnings)
    )

    worker_db = PokerDatabase(tmp_path / "honesty.db")
    worker_db.init_db()
    run_solver_job(worker_db, run.id, timeout_seconds=60)

    saved = db.fetch_solver_run(run.id)
    assert saved.status == "completed", saved.error_message
    evidence = SolverEvidence.model_validate(saved.evidence)
    assert set(run.assumptions) <= set(evidence.assumptions)

    prompt = build_hand_review_prompt(
        Session(id=session.id, name="Solver honesty"),
        hand_facts,
        actions,
        players,
        solver_evidence=evidence,
    )
    assert "no-rake equilibrium approximation" in prompt
    assert "Suit isomorphism is disabled" in prompt
    assert "estimated study input, not solved preflop GTO" in prompt
    assert "action_ev_and_bb_loss: unavailable" in prompt
    db.close()


# ---------------------------------------------------------------------------
# Honesty: the assumptions are shown beside the explanation, not merely stored
# ---------------------------------------------------------------------------

_SOLVER_PANEL_SCRIPT = """
import app as app_module
from poker_tracker.persistence.db import PokerDatabase

db = PokerDatabase({path!r})
db.init_db()
session = db.fetch_sessions()[0]
hand = db.fetch_hands_by_session(session.id)[0]
players = db.fetch_players_by_hand(hand.id)
actions = db.fetch_actions_by_hand(hand.id)
runs = db.fetch_solver_runs_by_hand(hand.id)
readiness = app_module.hand_study_readiness(db, hand, None, None)
app_module._show_solver_runs(
    db, session, hand, players, actions, None, None, runs, readiness
)
"""


def _seed_completed_run(
    tmp_path: Path,
    *,
    assumptions: list[str],
    artifacts: bool,
    frequencies: list[ActionFrequency] | None = None,
):
    db, _, hand = _seeded_db(tmp_path, "panel.db")
    run_dir = tmp_path / "solver_runs" / "run_1"
    run_dir.mkdir(parents=True)
    command_path = run_dir / "input.txt"
    result_path = run_dir / "result.json"
    log_path = run_dir / "solver.log"
    command_path.write_text("set_use_isomorphism 0\n", encoding="utf-8")
    result_path.write_text(json.dumps(STUB_DUMP), encoding="utf-8")
    log_path.write_text("Total exploitability 0.33 percent\n", encoding="utf-8")
    evidence = SolverEvidence(
        backend_version=PINNED_CONSOLE_COMMIT,
        street="flop",
        board="Qd 7s 2c",
        pot=5,
        effective_stack=97.5,
        hero_player="Hero",
        hero_combo="Ah Qs",
        recorded_action="check",
        mapped_action="CHECK",
        action_frequencies=(
            [
                ActionFrequency(action="CHECK", frequency=0.7),
                ActionFrequency(action="BET 3.75", frequency=0.3),
            ]
            if frequencies is None
            else frequencies
        ),
        range_ip_name="IP estimate",
        range_oop_name="OOP estimate",
        assumptions=assumptions,
    )
    db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            status="completed",
            input_hash="c" * 64,
            backend_version=PINNED_CONSOLE_COMMIT,
            assumptions=assumptions,
            evidence=evidence.model_dump(mode="json"),
            command_path=str(command_path),
            result_path=str(result_path),
            log_path=str(log_path),
        )
    )
    db.close()
    if not artifacts:
        for path in (command_path, result_path, log_path):
            path.unlink()
    return tmp_path / "panel.db"


def _run_panel(path: Path, monkeypatch) -> AppTest:
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("POKERTRAINER_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()
    app = AppTest.from_string(
        _SOLVER_PANEL_SCRIPT.format(path=str(path)), default_timeout=60
    ).run()
    assert not list(app.exception)
    return app


def test_solver_panel_shows_the_assumptions_beside_the_explain_control(
    tmp_path, monkeypatch
):
    assumptions = [
        "No-rake equilibrium approximation.",
        f"Recorded rake was {RAKE_BB:g} BB; TexasSolver is running a "
        "no-rake equilibrium approximation.",
        "Suit isomorphism is disabled because board/Hero blocker filtering "
        "uses exact suit-specific combinations.",
        "OOP range 'BB call open · estimated' is an estimated study input, "
        "not solved preflop GTO.",
    ]
    path = _seed_completed_run(tmp_path, assumptions=assumptions, artifacts=True)

    app = _run_panel(path, monkeypatch)

    captions = [item.value for item in app.caption]
    for assumption in assumptions:
        assert any(assumption in caption for caption in captions), assumption
    assert any("Solved tree ·" in caption for caption in captions)
    assert any("Convergence target ·" in caption for caption in captions)
    assert any(
        button.label == "Explain solver result with AI" for button in app.button
    )
    assert not any(
        "can no longer be reproduced" in warning.value for warning in app.warning
    )


def test_solver_panel_says_a_run_with_no_artifacts_cannot_be_reproduced(
    tmp_path, monkeypatch
):
    path = _seed_completed_run(
        tmp_path, assumptions=["No-rake equilibrium approximation."], artifacts=False
    )

    app = _run_panel(path, monkeypatch)

    warnings = "\n".join(item.value for item in app.warning)
    assert "no longer be reproduced or audited" in warnings
    assert "input.txt" in warnings and "result.json" in warnings


_COACH_PANEL_SCRIPT = """
import app as app_module
from poker_tracker.persistence.db import PokerDatabase

db = PokerDatabase({path!r})
db.init_db()
session = db.fetch_sessions()[0]
hand = db.fetch_hands_by_session(session.id)[0]
players = db.fetch_players_by_hand(hand.id)
actions = db.fetch_actions_by_hand(hand.id)
readiness = app_module.hand_study_readiness(db, hand, None, None)
app_module.show_study_coach_review(
    db, session, hand, actions, players, None, None, [], readiness
)
"""


def test_coaching_shows_the_same_assumptions_beside_the_attached_evidence(
    tmp_path, monkeypatch
):
    """Every surface that explains a solver result carries its conditions."""

    assumptions = ["No-rake equilibrium approximation."]
    path = _seed_completed_run(tmp_path, assumptions=assumptions, artifacts=True)

    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()
    app = AppTest.from_string(
        _COACH_PANEL_SCRIPT.format(path=str(path)), default_timeout=60
    ).run()

    assert not list(app.exception)
    captions = [item.value for item in app.caption]
    assert any("Solver evidence attached" in caption for caption in captions)
    assert any(assumptions[0] in caption for caption in captions)
    assert any("Solved tree ·" in caption for caption in captions)


def test_coaching_refuses_to_attach_a_run_that_saved_no_frequencies(
    tmp_path, monkeypatch
):
    path = _seed_completed_run(
        tmp_path,
        assumptions=["No-rake equilibrium approximation."],
        artifacts=True,
        frequencies=[],
    )

    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.setenv("POKER_DB_PATH", str(path))
    monkeypatch.setattr(db_module, "DEFAULT_DB_PATH", str(path))
    st.cache_resource.clear()
    app = AppTest.from_string(
        _COACH_PANEL_SCRIPT.format(path=str(path)), default_timeout=60
    ).run()

    assert not list(app.exception)
    warnings = "\n".join(item.value for item in app.warning)
    assert "saved no action frequencies" in warnings
    assert not any(
        "Solver evidence attached" in item.value for item in app.caption
    )
    assert "Solver evidence:\n- none provided" in "\n".join(
        item.value for item in app.code
    )


def test_a_saved_explanation_still_carries_the_assumptions_it_was_written_under(
    tmp_path, monkeypatch
):
    """The pairing has to survive the session that generated it."""

    import app as app_module

    evidence = SolverEvidence(
        street="flop",
        board="Qd 7s 2c",
        pot=5,
        effective_stack=97.5,
        hero_player="Hero",
        hero_combo="Ah Qs",
        action_frequencies=[ActionFrequency(action="CHECK", frequency=1.0)],
        range_ip_name="IP estimate",
        range_oop_name="OOP estimate",
        assumptions=[
            "No-rake equilibrium approximation.",
            "Suit isomorphism is disabled because board/Hero blocker filtering "
            "uses exact suit-specific combinations.",
        ],
    )
    hand, players, actions, _, _ = _hand_facts()
    prompt = build_hand_review_prompt(
        Session(id=1, name="Solver honesty"),
        hand,
        actions,
        players,
        solver_evidence=evidence,
    )

    recovered = app_module.retained_solver_assumptions(prompt)

    assert any("No-rake equilibrium approximation." in item for item in recovered)
    assert any("Suit isomorphism is disabled" in item for item in recovered)
    assert all(item.startswith("Solver ") for item in recovered)
    assert app_module.retained_solver_assumptions("no solver block here") == []


def test_saved_review_expander_renders_the_retained_assumptions(tmp_path, monkeypatch):
    script = """
import app as app_module
from poker_tracker.persistence.models import CoachingResponse

app_module.show_saved_provider_reviews([
    CoachingResponse(
        provider_name="fixture",
        model_name="deterministic",
        raw_prompt="Solver evidence:\\n- assumptions: No-rake equilibrium approximation.\\n",
        raw_response="Theory Coach: CHECK 100%.",
        review_type="hand",
    )
])
"""
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    st.cache_resource.clear()
    app = AppTest.from_string(script, default_timeout=60).run()

    assert not list(app.exception)
    assert any(
        "No-rake equilibrium approximation." in item.value for item in app.caption
    )


# ---------------------------------------------------------------------------
# Honesty: quantities the retained dump does not expose
# ---------------------------------------------------------------------------


def _grounded_evidence() -> SolverEvidence:
    return SolverEvidence(
        street="flop",
        board="Qd 7s 2c",
        pot=5,
        effective_stack=97.5,
        hero_player="Hero",
        hero_combo="Ah Qs",
        action_frequencies=[
            ActionFrequency(action="CHECK", frequency=0.6),
            ActionFrequency(action="BET 3.75", frequency=0.4),
        ],
        range_ip_name="IP estimate",
        range_oop_name="OOP estimate",
    )


@pytest.mark.parametrize(
    "claim",
    [
        "The solver's regret for checking was large.",
        "Checking has high counterfactual regret.",
        "CFR shows this line is fine.",
        "You will regret checking here.",
        "The accumulated regrets favour betting.",
    ],
)
def test_regret_claims_are_refused(claim: str):
    with pytest.raises(ValueError, match="action EV or BB loss"):
        validate_solver_coaching_response(
            f"CHECK 60% and BET 3.75 40%. {claim}", _grounded_evidence()
        )


@pytest.mark.parametrize(
    "claim",
    [
        "The expected value of checking is higher.",
        "Checking has better EV.",
        "The EV of betting justifies it.",
        "Checking's expected value dominates.",
    ],
)
def test_unnumbered_ev_claims_are_refused(claim: str):
    with pytest.raises(ValueError, match="action EV or BB loss"):
        validate_solver_coaching_response(
            f"CHECK 60% and BET 3.75 40%. {claim}", _grounded_evidence()
        )


def test_ordinary_explanations_still_pass():
    """The refusals above must not swallow well-formed coaching."""

    validate_solver_coaching_response(
        "Hand Summary: Hero checked the flop.\n"
        "EV / Math Notes: Recorded facts only.\n"
        "Theory Coach: CHECK 60% and BET 3.75 40% form a genuine mix.\n",
        _grounded_evidence(),
    )


def test_an_empty_result_cannot_be_explained_at_all():
    """The strongest check used to degrade exactly when evidence was worthless."""

    empty = _grounded_evidence().model_copy(
        update={"action_frequencies": [], "range_action_frequencies": []}
    )
    with pytest.raises(ValueError, match="established no action frequencies"):
        validate_solver_coaching_response("Theory Coach: CHECK is fine.", empty)


# ---------------------------------------------------------------------------
# Retention: artifacts on disk, and settings that outlive them
# ---------------------------------------------------------------------------


def test_a_completed_run_leaves_readable_artifacts_its_row_points_at(
    tmp_path, monkeypatch
):
    _install_stub_solver(tmp_path, monkeypatch)
    root = _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    _, _, _, _, prepared = _hand_facts()
    db, _, hand = _seeded_db(tmp_path)
    spot = prepared.spot.model_copy(update={"hand_id": hand.id})
    hero = resolve_custom_range(spot, spot.oop, HERO_RANGE, name="Hero exact")
    villain = resolve_custom_range(spot, spot.ip, VILLAIN_RANGE, name="Villain")
    run = start_solver_job(db, spot, villain, hero)

    worker_db = PokerDatabase(tmp_path / "honesty.db")
    worker_db.init_db()
    run_solver_job(worker_db, run.id, timeout_seconds=60)

    saved = db.fetch_solver_run(run.id)
    assert saved.status == "completed", saved.error_message
    run_dir = root / f"run_{run.id}"
    for attribute, name in (
        ("command_path", "input.txt"),
        ("result_path", "result.json"),
        ("log_path", "solver.log"),
    ):
        path = Path(getattr(saved, attribute))
        assert path == run_dir / name
        assert path.is_file()
        assert path.stat().st_size > 0
    command = (run_dir / "input.txt").read_text(encoding="utf-8")
    assert "set_use_isomorphism 0" in command
    assert "set_bet_sizes oop,flop,bet,33,75" in command
    assert "set_accuracy 0.5" in command
    assert missing_run_artifacts(saved) == []
    db.close()


def test_deleting_the_artifacts_leaves_the_settings_and_a_visible_gap(
    tmp_path, monkeypatch
):
    """The point of the column: what was solved must outlive the run directory."""

    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    _, _, _, _, prepared = _hand_facts()
    db, _, hand = _seeded_db(tmp_path)
    spot = prepared.spot.model_copy(update={"hand_id": hand.id})
    hero = resolve_custom_range(spot, spot.oop, HERO_RANGE, name="Hero exact")
    villain = resolve_custom_range(spot, spot.ip, VILLAIN_RANGE, name="Villain")
    run = start_solver_job(db, spot, villain, hero)

    worker_db = PokerDatabase(tmp_path / "honesty.db")
    worker_db.init_db()
    run_solver_job(worker_db, run.id, timeout_seconds=60)

    saved = db.fetch_solver_run(run.id)
    assert remove_solver_run_artifacts(saved) is True

    reread = db.fetch_solver_run(run.id)
    parameters = SolverRunParameters.model_validate(reread.run_parameters)
    assert parameters.is_retained
    assert parameters.tree["isomorphism"] is False
    assert parameters.tree["flop_bets"] == [33, 75]
    assert parameters.accuracy_target_pct == 0.5
    assert parameters.max_iterations == 200
    summary = "\n".join(parameters.summary_lines())
    assert "flop_bets 33,75" in summary
    assert "0.5% of the starting pot" in summary
    assert "within 200 iterations" in summary
    assert len(missing_run_artifacts(reread)) == 3
    db.close()


def test_a_pre_column_run_reports_its_settings_as_unretained(tmp_path):
    """A migrated row must stay honestly blank rather than claim today's tree."""

    path = tmp_path / "legacy.db"
    db, _, hand = _seeded_db(tmp_path, "legacy.db")
    db.create_solver_run(SolverRun(hand_id=hand.id, input_hash="d" * 64))
    db.close()

    with sqlite3.connect(path) as raw:
        raw.execute("ALTER TABLE solver_runs DROP COLUMN run_parameters")
        raw.execute(
            "UPDATE schema_metadata SET value = '17' WHERE key = 'schema_version'"
        )

    migrated = PokerDatabase(path)
    migrated.init_db()
    assert migrated.schema_version() == db_module.SCHEMA_VERSION
    legacy = migrated.fetch_solver_runs_by_hand(hand.id)[0]
    assert legacy.run_parameters == {}
    assert not SolverRunParameters.model_validate(legacy.run_parameters).is_retained

    fresh = migrated.create_solver_run(SolverRun(hand_id=hand.id, input_hash="e" * 64))
    assert SolverRunParameters.model_validate(
        migrated.fetch_solver_run(fresh.id).run_parameters
    ).is_retained
    migrated.close()


def test_run_parameters_track_the_backend_constants():
    """The snapshot must follow the adapter, not a second copy of its numbers."""

    from poker_tracker.solver import texassolver

    parameters = current_run_parameters()
    assert parameters.backend == texassolver.BACKEND_NAME
    assert parameters.accuracy_target_pct == texassolver.DEFAULT_ACCURACY_PCT
    assert parameters.max_iterations == texassolver.DEFAULT_MAX_ITERATIONS
    assert parameters.tree == texassolver.TREE_SPEC
    parameters.tree["flop_bets"] = ["mutated"]
    assert texassolver.TREE_SPEC["flop_bets"] == [33, 75]


# ---------------------------------------------------------------------------
# Retention: which binary actually produced the result
# ---------------------------------------------------------------------------


def test_backend_identity_distinguishes_two_builds(tmp_path):
    first = tmp_path / "console_solver_a"
    second = tmp_path / "console_solver_b"
    first.write_bytes(b"build one")
    second.write_bytes(b"build two")

    identity = resolved_backend_identity(first)

    assert identity.startswith("sha256:")
    assert identity == resolved_backend_identity(first)
    assert identity != resolved_backend_identity(second)
    assert resolved_backend_identity(tmp_path / "absent") == ""


def test_backend_identity_assumption_never_presents_the_pin_as_verified(tmp_path):
    binary = tmp_path / "console_solver"
    binary.write_bytes(b"build one")

    stated = backend_identity_assumption(binary, PINNED_CONSOLE_COMMIT)

    assert resolved_backend_identity(binary) in stated
    assert PINNED_CONSOLE_COMMIT in stated
    assert "was not verified" in stated

    unknown = backend_identity_assumption(None, PINNED_CONSOLE_COMMIT)
    assert "could not be identified" in unknown
    assert PINNED_CONSOLE_COMMIT in unknown


def test_a_run_started_from_the_app_retains_which_binary_produced_it(
    tmp_path, monkeypatch
):
    binary = _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    _, _, _, _, prepared = _hand_facts()
    db, _, hand = _seeded_db(tmp_path)
    spot = prepared.spot.model_copy(update={"hand_id": hand.id})

    run, _, _ = _started_run(
        db,
        spot,
        extra_assumptions=[
            backend_identity_assumption(binary, PINNED_CONSOLE_COMMIT)
        ],
    )
    retained = "\n".join(run.assumptions)

    assert resolved_backend_identity(binary) in retained
    assert "not verified against this binary" in retained
    # The column still records the pin this build asserts, which is exactly why
    # the retained identity has to sit beside it.
    assert run.backend_version == PINNED_CONSOLE_COMMIT
    db.close()
