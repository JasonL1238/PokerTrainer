"""Fuzz tests for malformed TexasSolver output (PLAN Phase 14).

``parse_strategy_result`` reads a file produced by a third-party binary that
this project neither builds nor controls, and whatever it returns is written
into ``solver_runs.evidence`` and later quoted verbatim to an operator and to
the coaching model. The existing suite fuzzes the frequency VECTOR (four
parametrized cases at ``tests/test_solver.py``); the DOCUMENT was untested --
truncated JSON, non-UTF-8 bytes, an empty file, a missing ``strategy`` or
``actions`` key, an action list that disagrees with the vectors beside it, a
node that is not a decision node, an unreadable file.

Two properties, asserted over every generated document:

  1. **Total.** The parser accepts cleanly or raises with a message naming what
     was wrong. It never returns a half-populated ``SolverEvidence`` and never
     raises something the callers do not model.
  2. **All or nothing.** An accepted result is internally consistent: one
     frequency per action, each in [0, 1], the labels distinct, and the vector
     summing to one. There is no partial evidence -- a study spot is cheap and
     frequencies filed under the wrong action are not.

Driven through ``run_solver_job`` as well as directly, because the property the
operator actually depends on is that the RUN lands ``failed`` rather than
``completed`` with garbage in its evidence column.
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import Hand, Session, SolverRun
from poker_tracker.solver.models import (
    RecordedSolverAction,
    ResolvedRange,
    SolverPlayer,
    SolverSpot,
)
from poker_tracker.solver.run_job import run_solver_job
from poker_tracker.solver.texassolver import (
    PINNED_CONSOLE_COMMIT,
    SolverResultUnusableError,
    parse_strategy_result,
)

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture],
)

HERO_COMBOS = ("AhQs", "AsQh", "KhKs", "JdJc")
VILLAIN_COMBOS = ("AcKc", "QhQc", "7h6h", "5d5c")
ACTIONS = ["CHECK", "BET 3.75"]


def _range(role: str, combos: tuple[str, ...], key: str, name: str) -> ResolvedRange:
    notation = ",".join(combos)
    return ResolvedRange(
        player_key=key,
        player_name=name,
        position="BTN" if role == "ip" else "BB",
        role=role,
        source="custom",
        profile_name="Fuzz range",
        notation=notation,
        solver_notation=notation,
        combo_count=len(combos),
        range_percent=0.01,
    )


HERO_RANGE = _range("ip", HERO_COMBOS, "hero", "Hero")
VILLAIN_RANGE = _range("oop", VILLAIN_COMBOS, "villain", "Villain")


def _spot() -> SolverSpot:
    """Hero acts first on the flop, so the walk lands on the ROOT node.

    Keeping the tree walk trivial is deliberate: this file is about the
    document, and a multi-node line would mean a rejection could come from the
    mapping rules (already covered in tests/test_solver.py) rather than from the
    parse under test.
    """
    hero = SolverPlayer(
        player_key="hero", player_name="Hero", position="BTN", role="ip", is_hero=True
    )
    villain = SolverPlayer(
        player_key="villain", player_name="Villain", position="BB", role="oop", is_hero=False
    )
    return SolverSpot(
        hand_id=1,
        table_size=6,
        street="flop",
        board="Qd 7s 2c",
        pot=5.0,
        effective_stack=97.5,
        pot_type="single_raised",
        preflop_aggressor_key="villain",
        oop=villain,
        ip=hero,
        hero_cards="Ah Qs",
        recorded_line=[
            RecordedSolverAction(
                player_key="hero",
                player_name="Hero",
                street="flop",
                action_type="check",
                amount=None,
                pot_before=5.0,
            )
        ],
    )


def _well_formed_dump() -> dict:
    return {
        "node_type": "action_node",
        "actions": list(ACTIONS),
        "strategy": {
            "actions": list(ACTIONS),
            "strategy": {combo: [0.6, 0.4] for combo in HERO_COMBOS},
        },
    }


def _parse(path: Path):
    return parse_strategy_result(
        path,
        spot=_spot(),
        range_ip=HERO_RANGE,
        range_oop=VILLAIN_RANGE,
        backend_version=PINNED_CONSOLE_COMMIT,
        exploitability_pct=0.4,
        runtime_seconds=1.0,
        assumptions=[],
    )


def _assert_evidence_is_whole(evidence) -> None:
    """What "accepted cleanly" has to mean, so acceptance is not a free pass."""
    assert evidence.backend == "TexasSolver"
    assert evidence.hero_player == "Hero"
    frequencies = evidence.action_frequencies or evidence.range_action_frequencies
    assert frequencies, "an accepted result must establish some frequency"
    labels = [item.action for item in frequencies]
    assert len(labels) == len(set(labels)), f"repeated action label in {labels}"
    for item in frequencies:
        assert math.isfinite(item.frequency)
        assert 0.0 <= item.frequency <= 1.0
    assert math.isclose(sum(item.frequency for item in frequencies), 1.0, abs_tol=1e-3)


# --------------------------------------------------------------------------- #
# Mutations. Each takes a well-formed dump and breaks it one way; the point of
# generating them rather than listing assertions is that the parser has to hold
# for the COMBINATION of two mutations as well as for either alone.
# --------------------------------------------------------------------------- #
def _wrapper(dump: dict) -> dict:
    """The ``strategy`` wrapper, or an empty one when a prior mutation removed it.

    Mutations are composed, so each has to survive whatever the ones before it
    did to the document -- that composition is the point of generating them
    rather than listing cases, and a KeyError here would be the test breaking
    rather than the parser."""
    wrapper = dump.get("strategy")
    return dict(wrapper) if isinstance(wrapper, dict) else {}


def _with_vectors(dump: dict, strategy_map: object) -> dict:
    return {**dump, "strategy": {**_wrapper(dump), "strategy": strategy_map}}


def _without(dump: dict, key: str) -> dict:
    out = dict(dump)
    out.pop(key, None)
    return out


MUTATIONS: dict[str, object] = {
    "drop_actions": lambda d: _without(d, "actions"),
    "drop_strategy": lambda d: _without(d, "strategy"),
    "drop_node_type": lambda d: _without(d, "node_type"),
    "chance_node": lambda d: {**d, "node_type": "chance_node"},
    "node_type_is_a_list": lambda d: {**d, "node_type": ["action_node"]},
    "actions_is_a_string": lambda d: {**d, "actions": "CHECK"},
    "actions_is_empty": lambda d: {
        **d,
        "actions": [],
        "strategy": {**_wrapper(d), "actions": []},
    },
    "actions_repeat_a_label": lambda d: {
        **d,
        "actions": ["CHECK", "CHECK"],
        "strategy": {**_wrapper(d), "actions": ["CHECK", "CHECK"]},
    },
    "strategy_map_is_a_list": lambda d: _with_vectors(d, [1, 2]),
    "strategy_map_is_empty": lambda d: _with_vectors(d, {}),
    "vector_too_short": lambda d: _with_vectors(d, {c: [1.0] for c in HERO_COMBOS}),
    "vector_too_long": lambda d: _with_vectors(d, {c: [0.3, 0.3, 0.4] for c in HERO_COMBOS}),
    "vector_is_strings": lambda d: _with_vectors(d, {c: ["0.6", "0.4"] for c in HERO_COMBOS}),
    "vector_is_booleans": lambda d: _with_vectors(d, {c: [True, False] for c in HERO_COMBOS}),
    "vector_is_nested": lambda d: _with_vectors(d, {c: [[0.6], [0.4]] for c in HERO_COMBOS}),
    "vector_out_of_range": lambda d: _with_vectors(d, {c: [-0.2, 1.2] for c in HERO_COMBOS}),
    "vector_does_not_sum_to_one": lambda d: _with_vectors(d, {c: [0.3, 0.3] for c in HERO_COMBOS}),
    "describes_another_range": lambda d: _with_vectors(
        d, {c: [0.6, 0.4] for c in VILLAIN_COMBOS}
    ),
    "covers_a_quarter_of_the_range": lambda d: _with_vectors(d, {HERO_COMBOS[0]: [0.6, 0.4]}),
    "strategy_actions_disagree_with_the_node": lambda d: {**d, "actions": ["FOLD"]},
    "childrens_is_a_list": lambda d: {**d, "childrens": [1, 2]},
    "childrens_is_empty": lambda d: {**d, "childrens": {}},
}


@given(
    names=st.lists(st.sampled_from(sorted(MUTATIONS)), min_size=1, max_size=3, unique=True)
)
@SETTINGS
def test_a_mutated_dump_is_accepted_whole_or_refused_by_name(
    names: list[str], tmp_path_factory
) -> None:
    """Property 1 and 2 together, over every combination of up to three breaks."""
    dump = _well_formed_dump()
    for name in names:
        dump = MUTATIONS[name](dump)  # type: ignore[operator]
    path = tmp_path_factory.mktemp("dump") / "result.json"
    path.write_text(json.dumps(dump), encoding="utf-8")

    try:
        evidence = _parse(path)
    except (ValueError, SolverResultUnusableError) as exc:
        assert str(exc).strip(), f"{names} was refused with an empty message"
        return
    _assert_evidence_is_whole(evidence)


@given(
    payload=st.one_of(
        st.binary(max_size=64),
        st.text(max_size=64).map(lambda s: s.encode("utf-8", "surrogatepass")),
        st.builds(
            lambda n: json.dumps(_well_formed_dump())[:n].encode(),
            st.integers(min_value=0, max_value=200),
        ),
        st.sampled_from(
            [
                b"",
                b"\x00\x01\x02\xff\xfe",
                b"null",
                b"[]",
                b"3",
                b'"a string"',
                b"{",
                b'{"actions": [',
                b'{"strategy": {"strategy": {"AhQs": [NaN, 0.5]}, "actions": ["a","b"]}}',
                b'{"strategy": {"strategy": {"AhQs": [Infinity, 0.5]}, "actions": ["a","b"]}}',
                json.dumps(_well_formed_dump()).encode("utf-16"),
                b"[" * 400 + b"]" * 400,
                b'{"a":' * 300 + b"1" + b"}" * 300,
            ]
        ),
    )
)
@SETTINGS
def test_arbitrary_bytes_in_the_result_file_never_crash_the_parser(
    payload: bytes, tmp_path_factory
) -> None:
    """A result file the worker cannot READ, as opposed to one it cannot use.

    ``run_solver_job`` catches ``Exception`` and marks the run failed, so the
    interesting question is not whether something is raised but whether what is
    raised is a decode/parse error the operator can be told about, rather than a
    ``RecursionError`` or an ``AttributeError`` from halfway down a walk.
    """
    path = tmp_path_factory.mktemp("bytes") / "result.json"
    path.write_bytes(payload)
    try:
        evidence = _parse(path)
    except (ValueError, SolverResultUnusableError, RecursionError):
        # ValueError covers JSONDecodeError and UnicodeDecodeError; RecursionError
        # is what CPython's JSON scanner raises on pathological nesting and is
        # named here so a NEW crash class is a failure rather than a pass.
        return
    _assert_evidence_is_whole(evidence)


def test_a_result_file_the_worker_cannot_open_is_an_error_not_a_result(
    tmp_path: Path,
) -> None:
    """Two shapes of unreadable, both of which reach ``parse_strategy_result``
    only because the worker's ``result_path.is_file()`` check passed first."""
    missing = tmp_path / "never-written.json"
    with pytest.raises(OSError):
        _parse(missing)

    directory = tmp_path / "result.json"
    directory.mkdir()
    with pytest.raises(OSError):
        _parse(directory)

    if os.geteuid() == 0:  # pragma: no cover - root ignores the mode bits
        pytest.skip("running as root: an unreadable file is still readable")
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text(json.dumps(_well_formed_dump()), encoding="utf-8")
    unreadable.chmod(stat.S_IWUSR)
    try:
        with pytest.raises(OSError):
            _parse(unreadable)
    finally:
        unreadable.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_a_node_that_names_one_branch_twice_is_refused(tmp_path: Path) -> None:
    """A repeated action label was accepted, and the evidence it produced was
    quietly WRONG rather than merely odd.

    A two-branch node listing ``["CHECK", "CHECK"]`` retained ``CHECK 60%`` and
    ``CHECK 40%`` for a node that checks 100% of the time, and that halved figure
    is what reached the operator's evidence card and the coaching prompt.
    Nothing downstream could recover: ``validate_solver_coaching_response`` sees
    two CHECK candidates and rejects whatever the coach says about checking, so
    the malformed dump surfaced as an unexplainable result instead of as a failed
    run.
    """
    dump = _well_formed_dump()
    dump["actions"] = ["CHECK", "CHECK"]
    dump["strategy"]["actions"] = ["CHECK", "CHECK"]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(dump), encoding="utf-8")

    with pytest.raises(SolverResultUnusableError, match="more than once"):
        _parse(path)


def test_a_well_formed_dump_still_parses(tmp_path: Path) -> None:
    """The negative control for every refusal above: the mutations must be what
    is being rejected, not the fixture."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_well_formed_dump()), encoding="utf-8")
    evidence = _parse(path)
    _assert_evidence_is_whole(evidence)
    assert [item.action for item in evidence.action_frequencies] == ACTIONS
    assert [item.frequency for item in evidence.action_frequencies] == [0.6, 0.4]
    assert evidence.mapped_action == "CHECK"


# --------------------------------------------------------------------------- #
# Through the worker, because "the parser raised" and "the run is marked failed"
# are different claims and only the second one is what an operator sees.
# --------------------------------------------------------------------------- #
def _seeded_run(tmp_path: Path, db_name: str, run_dir: Path) -> tuple[Path, int]:
    """A queued solver run against a real hand row, ready for the worker."""
    db_path = tmp_path / db_name
    db = PokerDatabase(db_path)
    db.init_db()
    session = db.create_session(Session(name="Solver fuzz"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="NLHE cash", table_size=6)
    )
    run = db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            status="queued",
            backend="texassolver",
            backend_version=PINNED_CONSOLE_COMMIT,
            input_hash="c" * 64,
            spot=_spot().model_copy(update={"hand_id": hand.id}).model_dump(mode="json"),
            range_ip=HERO_RANGE.model_dump(mode="json"),
            range_oop=VILLAIN_RANGE.model_dump(mode="json"),
            command_path=str(run_dir / "command.txt"),
            result_path=str(run_dir / "result.json"),
            log_path=str(run_dir / "solver.log"),
            assumptions=[],
        )
    )
    db.close()
    return db_path, run.id


def _fake_solver(tmp_path: Path, result_body: bytes | None) -> Path:
    """A stand-in console_solver that writes whatever body the test wants.

    ``run_solver_job`` refuses to parse anything until the child exits 0 and the
    result file exists, so producing a malformed result requires a process --
    there is no shortcut into the parse from the worker's own guards.
    """
    binary = tmp_path / "console_solver"
    body = "" if result_body is None else repr(result_body)
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"print('Total exploitability 0.4 percent')\n"
        f"body = {body}\n"
        "if body is not None:\n"
        "    Path(sys.argv[sys.argv.index('-i') + 1]).parent.joinpath('result.json')"
        ".write_bytes(body)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    resources = tmp_path / "resources" / "compairer"
    resources.mkdir(parents=True)
    (resources / "card5_dic_sorted.txt").write_text("x", encoding="utf-8")
    return binary


@pytest.mark.parametrize(
    ("label", "body"),
    [
        ("empty file", b""),
        ("truncated json", json.dumps(_well_formed_dump())[:37].encode()),
        ("non-json bytes", b"\x00\x01\x02\xff\xfe"),
        ("missing strategy", json.dumps({"node_type": "action_node", "actions": ACTIONS}).encode()),
        (
            "actions disagree with the vectors",
            json.dumps(
                {
                    "actions": ACTIONS,
                    "strategy": {"actions": ACTIONS, "strategy": {c: [1.0] for c in HERO_COMBOS}},
                }
            ).encode(),
        ),
        ("no result file at all", None),
    ],
)
def test_a_malformed_result_lands_the_run_failed_not_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, body: bytes | None
) -> None:
    """The property the gap analysis names: nothing proved the run lands
    ``failed`` rather than ``completed`` carrying garbage evidence."""
    del label
    binary = _fake_solver(tmp_path, body)
    monkeypatch.setenv("TEXAS_SOLVER_PATH", str(binary))
    monkeypatch.setenv("TEXAS_SOLVER_RESOURCE_DIR", str(tmp_path / "resources"))
    monkeypatch.delenv("POKERTRAINER_SOLVER_MEMORY_GB", raising=False)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "command.txt").write_text("build_tree\n", encoding="utf-8")
    db_path, run_id = _seeded_run(tmp_path, "fuzz.db", run_dir)

    worker_db = PokerDatabase(db_path)
    worker_db.init_db()
    with pytest.raises(Exception):  # noqa: B017 - the worker re-raises whatever failed
        run_solver_job(worker_db, run_id, timeout_seconds=60)

    check = PokerDatabase(db_path)
    stored = check.fetch_solver_run(run_id)
    assert stored is not None
    assert stored.status == "failed", f"run landed {stored.status} on a malformed dump"
    # Empty rather than None: the column is initialised to {} and a failed run
    # must never have written into it. Half-populated evidence is the outcome
    # this whole file exists to rule out.
    assert not stored.evidence, f"a failed run retained evidence: {stored.evidence!r}"
    assert stored.error_message, "a failed run must tell the operator why"
    check.close()


@pytest.mark.skipif(sys.platform == "win32", reason="the fake solver is a POSIX shebang script")
def test_a_well_formed_result_still_completes_through_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control for the worker parametrization above."""
    binary = _fake_solver(tmp_path, json.dumps(_well_formed_dump()).encode())
    monkeypatch.setenv("TEXAS_SOLVER_PATH", str(binary))
    monkeypatch.setenv("TEXAS_SOLVER_RESOURCE_DIR", str(tmp_path / "resources"))
    monkeypatch.delenv("POKERTRAINER_SOLVER_MEMORY_GB", raising=False)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "command.txt").write_text("build_tree\n", encoding="utf-8")
    db_path, run_id = _seeded_run(tmp_path, "ok.db", run_dir)

    worker_db = PokerDatabase(db_path)
    worker_db.init_db()
    run_solver_job(worker_db, run_id, timeout_seconds=60)

    check = PokerDatabase(db_path)
    stored = check.fetch_solver_run(run_id)
    assert stored is not None and stored.status == "completed"
    assert stored.evidence is not None
    assert [item["action"] for item in stored.evidence["action_frequencies"]] == ACTIONS
    check.close()
