"""Reconciled hand -> solver spot -> retained result -> grounded explanation.

The three halves of this chain were each tested in isolation and never joined.
The spot was always built from a hand-assembled ``AccountingReconciliation``
rather than one ``reconcile_persisted_hand`` derived from rows; the retained
result was asserted as a saved row and then dropped; and the grounding rules
were exercised against a ``SolverEvidence`` object constructed inline in a test.
Nothing read a solver run back out of SQLite and grounded an explanation in it.

That join is the whole point. An operator opening a hand a week later has no
object in memory from the solve -- they have a row. A field dropped, a float
rounded, or an action label renamed on the round trip would pass every existing
solver test while the product explained frequencies nobody solved. So every
test here reconciles a hand from durable records, runs the real worker, and
then closes the writing connection and grounds from what a fresh reader finds.

The hostile cases are the ones an operator actually meets: a run whose
directory has been pruned, a run that predates the hand's last correction, and
a recorded action the tree could not place.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from poker_tracker.coaching.coaching_prompts import build_hand_review_prompt
from poker_tracker.coaching.solver_grounding import validate_solver_coaching_response
from poker_tracker.perf.probes import SOLVER_MEDIAN, SOLVER_RUNS, ProbeContext, probe_solver
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Hand, Session, SolverRun
from poker_tracker.services.hand_accounting import reconcile_persisted_hand
from poker_tracker.services.manual_spot_entry import (
    ManualSpotInput,
    PostflopActionInput,
    save_manual_spot,
)
from poker_tracker.services.study_readiness import evaluate_study_readiness
from poker_tracker.solver import jobs as solver_jobs
from poker_tracker.solver import storage as solver_storage
from poker_tracker.solver.eligibility import prepare_solver_spot
from poker_tracker.solver.jobs import start_solver_job
from poker_tracker.solver.models import (
    ActionFrequency,
    SolverEvidence,
    SolverRunParameters,
)
from poker_tracker.solver.ranges import resolve_custom_range
from poker_tracker.solver.run_job import run_solver_job
from poker_tracker.solver.storage import (
    missing_run_artifacts,
    remove_solver_run_artifacts,
)
from poker_tracker.solver.texassolver import (
    PINNED_CONSOLE_COMMIT,
    SolverResultUnusableError,
    input_hash,
    parse_strategy_result,
)

HERO_RANGE = "AhQs,AsQh"
VILLAIN_RANGE = "22+,A2s+"

# The spot every test in this module solves: 6-max NLHE cash, Hero opens the
# BTN to 2.5, the BB calls, and the flop comes Qd 7s 2c heads-up. The dead small
# blind is why the solver node's pot is 5.5 rather than 5.
_FLOP_POT_BB = 5.5
_EFFECTIVE_STACK_BB = 97.5

# A dump shaped like a real one: a root node for the out-of-position player and
# a distinct child under each of its branches, so the parser has to walk the
# recorded line rather than read the root and stop.
_STUB_SOLVER = '''#!/usr/bin/env python3
"""Answers whatever ranges the command file submits, with a two-level tree."""
import json
import pathlib
import sys

command = pathlib.Path(sys.argv[2]).read_text().splitlines()
output = next(
    line.split(" ", 1)[1] for line in command if line.startswith("dump_result ")
)
combos = {"ip": [], "oop": []}
for line in command:
    for role in ("ip", "oop"):
        if line.startswith(f"set_range_{role} "):
            for token in line.split(" ", 1)[1].split(","):
                combo = token.split(":", 1)[0].strip()
                if len(combo) == 4:
                    combos[role].append(combo)


def node(actions, hand_combos, vector):
    return {
        "node_type": "action_node",
        "actions": list(actions),
        "strategy": {
            "actions": list(actions),
            "strategy": {combo: list(vector) for combo in hand_combos},
        },
    }


root = node(["CHECK", "BET 3.75"], combos["oop"], [0.7, 0.3])
root["childrens"] = {
    "CHECK": node(["CHECK", "BET 3.75"], combos["ip"], [0.55, 0.45]),
    "BET 3.75": node(["FOLD", "CALL", "RAISE 12"], combos["ip"], [0.2, 0.5, 0.3]),
}
pathlib.Path(output).write_text(json.dumps(root))
print("Total exploitability 0.33 percent")
'''

# What the stub reports at the node Hero actually reaches in the default line.
_HERO_NODE_STRATEGY = (("CHECK", 0.55), ("BET 3.75", 0.45))
_GROUNDED_RESPONSE = (
    "Hand Summary\n"
    "Villain checked the flop to Hero.\n"
    "Theory Coach\n"
    "CHECK 55% and BET 3.75 45% are the equilibrium mix here.\n"
)


class _DummyProcess:
    """Stands in for the detached worker so the test drives it in-process."""

    pid = 987654


def _install_stub_solver(tmp_path: Path, monkeypatch) -> Path:
    binary = tmp_path / "console_solver"
    binary.write_text(_STUB_SOLVER, encoding="utf-8")
    binary.chmod(0o755)
    evaluator = tmp_path / "resources" / "compairer" / "card5_dic_sorted.txt"
    evaluator.parent.mkdir(parents=True, exist_ok=True)
    evaluator.write_text("test", encoding="utf-8")
    monkeypatch.setenv("TEXAS_SOLVER_PATH", str(binary))
    monkeypatch.delenv("TEXAS_SOLVER_RESOURCE_DIR", raising=False)
    monkeypatch.delenv("POKERTRAINER_SOLVER_MEMORY_GB", raising=False)
    monkeypatch.delenv("POKERTRAINER_SOLVER_THREADS", raising=False)
    return binary


def _isolate_run_directories(tmp_path: Path, monkeypatch) -> Path:
    """Keep every artifact under tmp_path, including the deletion path's root."""

    root = tmp_path / "solver_runs"
    monkeypatch.setattr(solver_storage, "SOLVER_RUNS_DIR", root)

    def directory(run_id: int) -> Path:
        path = root / f"run_{run_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(solver_jobs, "solver_run_directory", directory)
    return root


def _block_worker_launch(monkeypatch) -> None:
    """Stop start_solver_job detaching a worker without disarming its own Popen.

    ``jobs.subprocess`` is the stdlib module, so patching ``Popen`` on it would
    also stop the worker launching the solver, which these tests need to run for
    real.
    """

    class _NoLaunch:
        DEVNULL = subprocess.DEVNULL

        @staticmethod
        def Popen(*args, **kwargs):
            return _DummyProcess()

    monkeypatch.setattr(solver_jobs, "subprocess", _NoLaunch)


def _spot_input(
    hand_number: int,
    postflop: tuple[PostflopActionInput, ...],
) -> ManualSpotInput:
    return ManualSpotInput(
        hand_number=hand_number,
        hero_cards="Ah Qs",
        board_cards="Qd 7s 2c",
        hero_position="BTN",
        villain_position="BB",
        table_size=6,
        starting_stack=100.0,
        pot_type="single_raised",
        opener="hero",
        open_to=2.5,
        postflop_actions=postflop,
        winner="hero",
    )


_DEFAULT_LINE = (
    PostflopActionInput("flop", "villain", "check"),
    PostflopActionInput("flop", "hero", "bet", 3.75),
    PostflopActionInput("flop", "villain", "call", 3.75),
)


def _reconciled_hand(db: PokerDatabase, session_id: int, **kwargs):
    """A hand that exists as rows and reconciles from them, not from a fixture."""

    hand, accounting, warnings = save_manual_spot(
        db,
        session_id,
        _spot_input(kwargs.get("hand_number", 1), kwargs.get("postflop", _DEFAULT_LINE)),
    )
    assert warnings == []
    assert accounting.is_authoritative
    return hand


def _prepare(db: PokerDatabase, hand_id: int):
    """Re-derive the spot the way the product does: from durable records only."""

    accounting = reconcile_persisted_hand(db, hand_id)
    hand = db.fetch_hand(hand_id)
    players = db.fetch_players_by_hand(hand_id)
    actions = db.fetch_actions_by_hand(hand_id)
    prepared = prepare_solver_spot(hand, players, actions, accounting)
    assert prepared.eligibility.eligible, prepared.eligibility.reasons
    assert prepared.spot is not None
    return accounting, hand, players, actions, prepared


def _solve(db: PokerDatabase, db_path: Path, hand_id: int):
    """Run the whole job path for one hand and return the run plus its inputs."""

    _, _, _, _, prepared = _prepare(db, hand_id)
    spot = prepared.spot
    hero_range = resolve_custom_range(spot, spot.ip, HERO_RANGE, name="Hero exact")
    villain_range = resolve_custom_range(spot, spot.oop, VILLAIN_RANGE, name="Villain")
    run = start_solver_job(
        db,
        spot,
        hero_range,
        villain_range,
        assumptions=list(prepared.eligibility.warnings),
    )
    worker_db = PokerDatabase(db_path)
    worker_db.init_db()
    run_solver_job(worker_db, run.id, timeout_seconds=60)
    return run, spot, hero_range, villain_range


def _seeded_db(tmp_path: Path, name: str = "chain.db"):
    path = tmp_path / name
    db = PokerDatabase(path)
    db.init_db()
    session = db.create_session(Session(name="Solver chain"))
    return db, path, session


def _reread(db_path: Path, hand_id: int):
    """What a reader who was not present for the solve finds on disk."""

    reader = PokerDatabase(db_path)
    reader.init_db()
    try:
        return reader.fetch_solver_runs_by_hand(hand_id)
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# The chain, seam by seam
# ---------------------------------------------------------------------------


def test_a_reconciled_hand_grounds_an_explanation_read_back_out_of_sqlite(
    tmp_path, monkeypatch
) -> None:
    """Every seam carries its data, and the last one reads from the store."""

    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    db, db_path, session = _seeded_db(tmp_path)
    hand = _reconciled_hand(db, session.id)

    # Seam 1: reconciled hand -> solver spot. The pot and effective stack are
    # the ledger's, measured at the first flop action, not the hand's summary
    # columns -- 5.5 rather than 5 because the folded small blind is dead money.
    accounting, saved_hand, players, actions, prepared = _prepare(db, hand.id)
    spot = prepared.spot
    flop_snapshot = next(
        snapshot
        for snapshot in accounting.ledger.snapshots
        if snapshot.street == "flop"
    )
    assert spot.pot == pytest.approx(flop_snapshot.pot_before) == _FLOP_POT_BB
    assert spot.effective_stack == pytest.approx(_EFFECTIVE_STACK_BB)
    assert spot.board == "Qd 7s 2c"
    assert spot.hero_cards == saved_hand.hero_cards == "Ah Qs"
    assert spot.hand_id == hand.id
    assert [
        (recorded.player_key, recorded.action_type, recorded.amount)
        for recorded in spot.recorded_line
    ] == [
        ("villain", "check", None),
        ("hero", "bet", 3.75),
        ("villain", "call", 3.75),
    ]

    run, _, hero_range, villain_range = _solve(db, db_path, hand.id)
    db.close()

    # Seam 2: solver spot -> retained row, read by a connection that was not
    # open while the solve ran.
    retained = _reread(db_path, hand.id)
    assert len(retained) == 1
    saved = retained[0]
    assert saved.id == run.id
    assert saved.status == "completed", saved.error_message
    assert saved.hand_id == hand.id
    assert saved.input_hash == input_hash(spot, hero_range, villain_range)
    assert saved.backend_version == PINNED_CONSOLE_COMMIT
    assert saved.completed_at is not None

    # The spot column is what makes the row self-describing; a reader must be
    # able to recover the question that was asked, not only the answer.
    assert saved.spot["board"] == spot.board
    assert saved.spot["pot"] == pytest.approx(spot.pot)
    assert saved.spot["hero_cards"] == spot.hero_cards
    assert len(saved.spot["recorded_line"]) == len(spot.recorded_line)

    # Seam 3: the store round-trip loses nothing. Comparing against the parse of
    # the artifact the solver wrote, rather than against literals, means a field
    # dropped or a float rounded on the way through SQLite fails here.
    produced = parse_strategy_result(
        saved.result_path,
        spot=spot,
        range_ip=hero_range,
        range_oop=villain_range,
        backend_version=saved.backend_version,
        exploitability_pct=saved.exploitability_pct,
        runtime_seconds=saved.runtime_seconds,
        assumptions=saved.assumptions,
    )
    evidence = SolverEvidence.model_validate(saved.evidence)
    assert evidence == produced
    assert evidence.board == spot.board
    assert evidence.pot == pytest.approx(spot.pot)
    assert evidence.effective_stack == pytest.approx(spot.effective_stack)
    assert evidence.hero_player == "Hero"
    assert evidence.hero_combo == "Ah Qs"
    assert evidence.recorded_action == "bet 3.75 BB"
    assert evidence.mapped_action == "BET 3.75"
    assert [
        (item.action, item.frequency) for item in evidence.action_frequencies
    ] == list(_HERO_NODE_STRATEGY)
    assert evidence.exploitability_pct == 0.33

    # Seam 4: retained result -> grounded explanation. The prompt carries every
    # retained frequency and both halves of the mapping, so the model is told
    # what was recorded and what branch it was solved as.
    prompt = build_hand_review_prompt(
        session,
        saved_hand,
        actions,
        players,
        ledger=accounting.ledger,
        accounting_authoritative=True,
        solver_evidence=evidence,
    )
    assert "CHECK 55.0%" in prompt
    assert "BET 3.75 45.0%" in prompt
    assert "recorded_action: bet 3.75 BB" in prompt
    assert "mapped_solver_action: BET 3.75" in prompt
    assert "action_ev_and_bb_loss: unavailable" in prompt

    validate_solver_coaching_response(_GROUNDED_RESPONSE, evidence)
    with pytest.raises(ValueError, match="changed the saved solver frequency"):
        validate_solver_coaching_response(
            "Theory Coach\nCHECK 65% and BET 3.75 35% are the mix.\n", evidence
        )
    with pytest.raises(ValueError, match="omitted"):
        validate_solver_coaching_response("Theory Coach\nCHECK 55%.\n", evidence)


def test_the_explanation_is_grounded_in_the_stored_row_not_the_solve(
    tmp_path, monkeypatch
) -> None:
    """A frequency altered in the store must change what the coach may say.

    This is the difference the join exists for. If grounding read the object the
    solve returned, an explanation would still validate after the retained
    frequencies had changed underneath it -- which is exactly what a
    serialization defect looks like from the operator's side.
    """

    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    db, db_path, session = _seeded_db(tmp_path)
    hand = _reconciled_hand(db, session.id)
    run, _, _, _ = _solve(db, db_path, hand.id)
    db.close()

    validate_solver_coaching_response(
        _GROUNDED_RESPONSE,
        SolverEvidence.model_validate(_reread(db_path, hand.id)[0].evidence),
    )

    # Rewrite one frequency in the column, the way a bad round trip would.
    raw = sqlite3.connect(db_path)
    stored = json.loads(
        raw.execute(
            "SELECT evidence FROM solver_runs WHERE id = ?", (run.id,)
        ).fetchone()[0]
    )
    stored["action_frequencies"][0]["frequency"] = 0.65
    stored["action_frequencies"][1]["frequency"] = 0.35
    raw.execute(
        "UPDATE solver_runs SET evidence = ? WHERE id = ?",
        (json.dumps(stored), run.id),
    )
    raw.commit()
    raw.close()

    altered = SolverEvidence.model_validate(_reread(db_path, hand.id)[0].evidence)
    assert [item.frequency for item in altered.action_frequencies] == [0.65, 0.35]
    with pytest.raises(ValueError, match="changed the saved solver frequency"):
        validate_solver_coaching_response(_GROUNDED_RESPONSE, altered)
    validate_solver_coaching_response(
        "Theory Coach\nCHECK 65% and BET 3.75 35% are the equilibrium mix here.\n",
        altered,
    )


def test_a_retained_result_whose_artifacts_are_gone_degrades_honestly(
    tmp_path, monkeypatch
) -> None:
    """The row must still say what was solved, and admit it cannot be rechecked."""

    _install_stub_solver(tmp_path, monkeypatch)
    root = _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    db, db_path, session = _seeded_db(tmp_path)
    hand = _reconciled_hand(db, session.id)
    run, _, _, _ = _solve(db, db_path, hand.id)
    _, saved_hand, players, actions, _ = _prepare(db, hand.id)
    db.close()

    before = _reread(db_path, hand.id)[0]
    assert missing_run_artifacts(before) == []
    assert remove_solver_run_artifacts(before) is True
    assert not (root / f"run_{run.id}").exists()

    after = _reread(db_path, hand.id)[0]
    # What was solved outlives the files it was solved into.
    parameters = SolverRunParameters.model_validate(after.run_parameters)
    assert parameters.is_retained
    assert parameters.tree["flop_bets"] == [33, 75]
    assert parameters.max_iterations == 200
    summary = "\n".join(parameters.summary_lines())
    assert "flop_bets 33,75" in summary
    assert "0.5% of the starting pot" in summary

    # And the gap is named rather than papered over: the row still points at
    # every artifact, and every one of them is reported missing by its path.
    missing = missing_run_artifacts(after)
    assert len(missing) == 3
    assert after.result_path and not Path(after.result_path).exists()
    for path in (after.command_path, after.result_path, after.log_path):
        assert any(path in item for item in missing)

    # The frequencies themselves are untouched, so an explanation of them is
    # still grounded -- losing the artifacts costs reproducibility, not the
    # result. Asserted so that a future change that quietly empties the evidence
    # along with the directory is caught here.
    evidence = SolverEvidence.model_validate(after.evidence)
    assert [
        (item.action, item.frequency) for item in evidence.action_frequencies
    ] == list(_HERO_NODE_STRATEGY)
    prompt = build_hand_review_prompt(
        session, saved_hand, actions, players, solver_evidence=evidence
    )
    assert "CHECK 55.0%" in prompt
    validate_solver_coaching_response(_GROUNDED_RESPONSE, evidence)


def test_a_run_that_predates_the_last_correction_cannot_ground_an_explanation(
    tmp_path, monkeypatch
) -> None:
    """A corrected hand must not be explained by the solve of its old facts."""

    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    db, db_path, session = _seeded_db(tmp_path)
    hand = _reconciled_hand(db, session.id)
    _solve(db, db_path, hand.id)
    assert _reread(db_path, hand.id)[0].status == "completed"

    corrected = db.fetch_hand(hand.id).model_copy(
        update={"board_cards": "Qd 7s 2c 9h", "hero_cards": "Ad Qc"}
    )
    db.update_hand_facts(corrected, correction_notes="Re-read the board and Hero cards.")
    accounting = reconcile_persisted_hand(db, hand.id)
    runs = db.fetch_solver_runs_by_hand(hand.id)
    readiness = evaluate_study_readiness(
        db.fetch_hand(hand.id), accounting=accounting, solver_runs=runs
    )
    db.close()

    stale = _reread(db_path, hand.id)[0]
    assert stale.status == "stale"
    assert "rerun solver analysis" in stale.error_message

    # The selection rule the product uses for attaching evidence to a coaching
    # prompt. It yields nothing, which is the whole guard.
    assert [run for run in runs if run.status == "completed" and run.evidence] == []
    assert readiness.has("STALE_SOLVER_EVIDENCE")

    # The row keeps its evidence as retained history, and that evidence is now
    # provably about a different hand -- which is why status, not presence, has
    # to be what decides whether it may be explained.
    assert stale.evidence
    retained = SolverEvidence.model_validate(stale.evidence)
    assert retained.board == "Qd 7s 2c"
    assert retained.hero_combo == "Ah Qs"
    assert retained.hero_combo != corrected.hero_cards
    assert retained.board != corrected.board_cards


def test_a_recorded_action_the_tree_cannot_place_produces_no_explanation(
    tmp_path, monkeypatch
) -> None:
    """A refused mapping ends the chain; it does not end it with an answer."""

    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    db, db_path, session = _seeded_db(tmp_path)
    session_id = session.id
    hand = _reconciled_hand(
        db,
        session_id,
        postflop=(
            PostflopActionInput("flop", "villain", "bet", 25.0),
            PostflopActionInput("flop", "hero", "call", 25.0),
        ),
    )
    with pytest.raises(SolverResultUnusableError, match="substitution limit"):
        _solve(db, db_path, hand.id)
    runs = db.fetch_solver_runs_by_hand(hand.id)
    db.close()

    refused = _reread(db_path, hand.id)[0]
    assert refused.status == "failed"
    assert "cannot be placed in the solver tree" in refused.error_message
    assert "BET 3.75" in refused.error_message

    # Nothing was retained that an explanation could be built from, and the
    # coaching path's own selection rule skips the row.
    assert refused.evidence == {}
    assert [run for run in runs if run.status == "completed" and run.evidence] == []
    with pytest.raises(ValueError):
        SolverEvidence.model_validate(refused.evidence)

    # Even handed an evidence object for this spot, the grounding gate refuses
    # rather than waving through an unfalsifiable explanation.
    empty = SolverEvidence(
        street="flop",
        board="Qd 7s 2c",
        pot=_FLOP_POT_BB,
        effective_stack=_EFFECTIVE_STACK_BB,
        hero_player="Hero",
        hero_combo="Ah Qs",
        range_ip_name="Hero exact",
        range_oop_name="Villain",
    )
    with pytest.raises(ValueError, match="cannot be grounded"):
        validate_solver_coaching_response("Theory Coach\nCalling is fine.\n", empty)

    # Negative control: the refusal is about this recorded line, not about the
    # stub. The same tree and ranges on a placeable line retains a result.
    again = PokerDatabase(db_path)
    again.init_db()
    placeable = _reconciled_hand(again, session_id, hand_number=2)
    _solve(again, db_path, placeable.id)
    again.close()
    assert _reread(db_path, placeable.id)[0].status == "completed"


# ---------------------------------------------------------------------------
# What else reads the retained store
# ---------------------------------------------------------------------------


def test_the_perf_probe_counts_the_solver_runs_this_installation_retained(
    tmp_path, monkeypatch
) -> None:
    """The runtime summary must be taken from the runs that actually exist.

    ``probe_solver`` queried ``status = 'succeeded'``, which ``SolverRunStatus``
    has never contained, so every installation reported zero recorded solver
    runs and withheld both percentiles with "this installation has recorded no
    completed solver run" -- on a library full of completed ones. A measured
    zero is a claim, and this one was false.
    """

    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    db, db_path, session = _seeded_db(tmp_path)
    hand = _reconciled_hand(db, session.id)
    _solve(db, db_path, hand.id)
    db.close()

    saved = _reread(db_path, hand.id)[0]
    assert saved.status == "completed"
    assert saved.runtime_seconds is not None

    context = ProbeContext(
        repo_root=Path(__file__).resolve().parents[1],
        workspace=tmp_path / "probe_workspace",
        db_path=db_path,
        data_root=tmp_path / "probe_data",
        manifest_path=tmp_path / "probe_manifest.json",
    )
    context.prepare()
    measured = {item.spec.name: item for item in probe_solver(context)}

    assert measured[SOLVER_RUNS.name].value == 1
    assert measured[SOLVER_MEDIAN.name].taken
    # The harness rounds a reported second to three places; the point of the
    # assertion is that it summarised THIS run rather than nothing.
    assert measured[SOLVER_MEDIAN.name].value == pytest.approx(
        saved.runtime_seconds, abs=0.001
    )
    assert "completed" in measured[SOLVER_RUNS.name].conditions["source"]


def test_the_perf_probe_ignores_runs_that_retained_nothing(tmp_path) -> None:
    """Negative control: only a completed run is a recorded solver runtime."""

    db, db_path, session = _seeded_db(tmp_path, "statuses.db")
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="NLHE cash",
            table_size=6,
        )
    )
    for index, status in enumerate(
        ("completed", "failed", "cancelled", "stale", "queued"), start=1
    ):
        run = db.create_solver_run(
            SolverRun(hand_id=hand.id, input_hash=f"{index:064d}")
        )
        db.update_solver_run(run.id, status=status, runtime_seconds=10.0 * index)
    db.close()

    context = ProbeContext(
        repo_root=Path(__file__).resolve().parents[1],
        workspace=tmp_path / "workspace",
        db_path=db_path,
        data_root=tmp_path / "data",
        manifest_path=tmp_path / "manifest.json",
    )
    context.prepare()
    measured = {item.spec.name: item for item in probe_solver(context)}

    assert measured[SOLVER_RUNS.name].value == 1
    assert measured[SOLVER_MEDIAN.name].value == pytest.approx(10.0)


def test_action_frequency_labels_survive_the_store_unchanged(
    tmp_path, monkeypatch
) -> None:
    """A renamed action label would break grounding without breaking parsing.

    ``_validate_frequency_claims`` keys on the first token of the label and, for
    BET and RAISE, on the size that follows it. A label that arrived back as
    "BET 3.750" or "Bet 3.75" would still validate as a model and still render
    into a prompt, while every claim about it became unmatchable.
    """

    _install_stub_solver(tmp_path, monkeypatch)
    _isolate_run_directories(tmp_path, monkeypatch)
    _block_worker_launch(monkeypatch)
    db, db_path, session = _seeded_db(tmp_path)
    hand = _reconciled_hand(db, session.id)
    run, _, _, _ = _solve(db, db_path, hand.id)
    db.close()

    raw = sqlite3.connect(db_path)
    stored = json.loads(
        raw.execute(
            "SELECT evidence FROM solver_runs WHERE id = ?", (run.id,)
        ).fetchone()[0]
    )
    raw.close()
    assert [item["action"] for item in stored["action_frequencies"]] == [
        "CHECK",
        "BET 3.75",
    ]

    evidence = SolverEvidence.model_validate(_reread(db_path, hand.id)[0].evidence)
    assert evidence.action_frequencies == [
        ActionFrequency(action=action, frequency=frequency)
        for action, frequency in _HERO_NODE_STRATEGY
    ]
    # The size in the label is what a size claim is checked against, so a claim
    # naming a size the run did not solve has to be refused.
    with pytest.raises(ValueError, match="invented a solver size"):
        validate_solver_coaching_response(
            "Theory Coach\nCHECK 55% and BET 5 45% are the mix.\n", evidence
        )
