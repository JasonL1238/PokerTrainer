from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from poker_tracker.coaching.coaching_prompts import build_hand_review_prompt
from poker_tracker.coaching.solver_grounding import (
    validate_solver_coaching_response,
)
from poker_tracker.math.accounting import ActionSnapshot, HandLedger
from poker_tracker.persistence.db import SCHEMA_VERSION, PokerDatabase
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
from poker_tracker.solver.eligibility import prepare_solver_spot
from poker_tracker.solver.jobs import (
    SolverJobAlreadyRunningError,
    reconcile_stale_solver_runs,
    start_solver_job,
)
from poker_tracker.solver.models import (
    ActionFrequency,
    SolverEvidence,
)
from poker_tracker.solver.profile_io import export_range_profiles, import_range_profiles
from poker_tracker.solver.ranges import (
    BUILTIN_RANGE_PROFILES,
    normalize_weighted_notation,
    parse_range,
    resolve_custom_range,
    resolve_profile,
    resolve_selected_profile,
)
from poker_tracker.solver.run_job import run_solver_job
from poker_tracker.solver.texassolver import (
    PINNED_CONSOLE_COMMIT,
    build_command_file,
    configured_resource_dir,
    input_hash,
    parse_final_exploitability,
    parse_strategy_result,
)


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


def _completed_cash_spot(table_size: int = 6):
    hand = Hand(
        id=1,
        session_id=1,
        hand_number=1,
        game_type="NLHE cash",
        table_size=table_size,
        effective_stack=100,
        hero_position="BB",
        hero_cards="Ah Qs",
        board_cards="Qd 7s 2c",
    )
    players = [
        HandPlayer(
            id=1,
            hand_id=1,
            player_key="hero",
            player_name="Hero",
            position="BB",
            starting_stack=100,
            is_hero=True,
        ),
        HandPlayer(
            id=2,
            hand_id=1,
            player_key="villain",
            player_name="Villain",
            position="BTN",
            starting_stack=100,
        ),
    ]
    actions = [
        Action(
            hand_id=1,
            player_key="hero",
            player_name="Hero",
            position="BB",
            street="preflop",
            action_index=1,
            action_type="post_blind",
            amount=1,
        ),
        Action(
            hand_id=1,
            player_key="villain",
            player_name="Villain",
            position="BTN",
            street="preflop",
            action_index=2,
            action_type="raise",
            amount=2.5,
        ),
        Action(
            hand_id=1,
            player_key="hero",
            player_name="Hero",
            position="BB",
            street="preflop",
            action_index=3,
            action_type="call",
            amount=1.5,
        ),
        Action(
            hand_id=1,
            player_key="hero",
            player_name="Hero",
            position="BB",
            street="flop",
            action_index=1,
            action_type="check",
        ),
        Action(
            hand_id=1,
            player_key="villain",
            player_name="Villain",
            position="BTN",
            street="flop",
            action_index=2,
            action_type="bet",
            amount=3.75,
        ),
        Action(
            hand_id=1,
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
        _snapshot(0, "hero", "preflop", "post_blind", pot_before=0, pot_after=1, stack_before=100, stack_after=99, amount=1),
        _snapshot(1, "villain", "preflop", "raise", pot_before=1, pot_after=3.5, stack_before=100, stack_after=97.5, amount=2.5),
        _snapshot(2, "hero", "preflop", "call", pot_before=3.5, pot_after=5, stack_before=99, stack_after=97.5, amount=1.5),
        _snapshot(3, "hero", "flop", "check", pot_before=5, pot_after=5, stack_before=97.5, stack_after=97.5),
        _snapshot(4, "villain", "flop", "bet", pot_before=5, pot_after=8.75, stack_before=97.5, stack_after=93.75, amount=3.75),
        _snapshot(5, "hero", "flop", "call", pot_before=8.75, pot_after=12.5, stack_before=97.5, stack_after=93.75, amount=3.75),
    )
    ledger = HandLedger(
        contributions={"hero": 6.25, "villain": 6.25},
        refunds={},
        payouts={"hero": 12.5},
        net_results={"hero": 6.25, "villain": -6.25},
        gross_pot=12.5,
        rake=0.5,
        net_pot=12,
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
    assert prepared.eligibility.eligible
    assert prepared.spot is not None
    return hand, players, actions, accounting, prepared.spot


def test_weighted_range_accepts_texassolver_colon_syntax() -> None:
    assert normalize_weighted_notation("TT+,AJs+:0.5,AQo+") == "TT+,50%(AJs+),AQo+"
    parsed = parse_range("TT+,AJs+:0.5,AQo+", dead_cards="Qd 7s 2c")
    assert parsed.combo_count > 0
    assert "AhQs" in parsed.solver_notation or "QsAh" in parsed.solver_notation
    with pytest.raises(ValueError, match="Invalid range"):
        parse_range("not-a-range")
    for notation in ("AA:0.005,KK", "AhQs:0.0050001"):
        with pytest.raises(ValueError, match="silently drops"):
            parse_range(notation)
    for notation in ("150%(AA)", "101%(AA)", "AA,150%(KK)"):
        with pytest.raises(ValueError, match="cannot exceed"):
            parse_range(notation)
    deduplicated = parse_range("AA,AA")
    assert deduplicated.combo_count == 6


@pytest.mark.parametrize("table_size", [5, 6, 7, 8])
def test_default_ranges_resolve_for_supported_table_sizes(table_size: int) -> None:
    _, _, _, _, spot = _completed_cash_spot(table_size)
    hero_range = resolve_profile(
        spot, spot.oop, BUILTIN_RANGE_PROFILES, source="default"
    )
    villain_range = resolve_profile(
        spot, spot.ip, BUILTIN_RANGE_PROFILES, source="default"
    )
    assert hero_range.combo_count > 0
    assert villain_range.combo_count > 0
    assert not any("Table size substituted" in item for item in hero_range.mismatches)


def test_custom_hero_range_must_contain_recorded_combo() -> None:
    _, _, _, _, spot = _completed_cash_spot()
    with pytest.raises(ValueError, match="does not contain"):
        resolve_custom_range(spot, spot.oop, "AA,KK")
    resolved = resolve_custom_range(spot, spot.oop, "AQo+,AA")
    assert resolved.source == "custom"
    villain = resolve_custom_range(spot, spot.ip, "AA,AQo+")
    exact_combos = [token.split(":", 1)[0] for token in villain.solver_notation.split(",")]
    assert all("Ah" not in combo and "Qs" not in combo for combo in exact_combos)


def test_premade_range_cannot_cross_pot_structures() -> None:
    _, _, _, _, spot = _completed_cash_spot()
    three_bet_spot = spot.model_copy(update={"pot_type": "three_bet"})
    single_raised = next(
        profile
        for profile in BUILTIN_RANGE_PROFILES
        if profile.pot_type == "single_raised"
    )
    with pytest.raises(ValueError, match="not three bet"):
        resolve_selected_profile(
            three_bet_spot,
            three_bet_spot.ip,
            single_raised,
            source="builtin",
        )


def test_command_file_and_parser_preserve_combo_frequencies(tmp_path) -> None:
    _, _, _, _, spot = _completed_cash_spot()
    oop = resolve_custom_range(
        spot, spot.oop, "AhQs:0.25,AsQh", name="Hero custom"
    )
    ip = resolve_custom_range(spot, spot.ip, "22+,A2s+", name="Villain custom")
    result_path = tmp_path / "result.json"
    command = build_command_file(
        spot,
        ip,
        oop,
        output_path=result_path,
        thread_count=2,
    )
    assert "set_bet_sizes oop,flop,bet,33,75" in command
    assert "set_accuracy 0.5" in command
    assert "set_use_isomorphism 0" in command
    assert "dump_result result.json" in command

    result_path.write_text(
        json.dumps(
            {
                "node_type": "action_node",
                "actions": ["CHECK", "BET 3.75"],
                "childrens": {},
                "strategy": {
                    "actions": ["CHECK", "BET 3.75"],
                    "strategy": {
                        "AhQs": [0.6, 0.4],
                        "AsQh": [0.2, 0.8],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    evidence = parse_strategy_result(
        result_path,
        spot=spot,
        range_ip=ip,
        range_oop=oop,
        backend_version=PINNED_CONSOLE_COMMIT,
        exploitability_pct=0.42,
        runtime_seconds=2.5,
        assumptions=["test assumption"],
    )
    assert evidence.action_frequencies == [
        ActionFrequency(action="CHECK", frequency=0.6),
        ActionFrequency(action="BET 3.75", frequency=0.4),
    ]
    assert [item.action for item in evidence.range_action_frequencies] == [
        "CHECK",
        "BET 3.75",
    ]
    assert [item.frequency for item in evidence.range_action_frequencies] == pytest.approx(
        [0.28, 0.72]
    )
    assert evidence.mapped_action == "CHECK"
    assert parse_final_exploitability(
        "Total exploitability 1.2 precent\nTotal exploitability 0.42 percent"
    ) == 0.42
    unconverged = parse_strategy_result(
        result_path,
        spot=spot,
        range_ip=ip,
        range_oop=oop,
        backend_version=PINNED_CONSOLE_COMMIT,
        exploitability_pct=1.2,
        runtime_seconds=2.5,
        assumptions=[],
    )
    assert any("exceeded the 0.5% target" in item for item in unconverged.warnings)


@pytest.mark.parametrize(
    "frequencies",
    ([0.8, 0.8], [-0.2, 1.2], [float("nan"), 0.5], ["wrong", 1.0]),
)
def test_solver_parser_rejects_invalid_strategy_vectors(
    tmp_path, frequencies
) -> None:
    _, _, _, _, spot = _completed_cash_spot()
    oop = resolve_custom_range(spot, spot.oop, "AQo+,AA")
    ip = resolve_custom_range(spot, spot.ip, "22+,A2s+")
    result_path = tmp_path / "invalid-result.json"
    result_path.write_text(
        json.dumps(
            {
                "actions": ["CHECK", "BET 3.75"],
                "strategy": {
                    "actions": ["CHECK", "BET 3.75"],
                    "strategy": {"AhQs": frequencies},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strategy"):
        parse_strategy_result(
            result_path,
            spot=spot,
            range_ip=ip,
            range_oop=oop,
            backend_version=PINNED_CONSOLE_COMMIT,
            exploitability_pct=0.4,
            runtime_seconds=1,
            assumptions=[],
        )


def test_solver_persistence_and_range_profile_round_trip(tmp_path) -> None:
    db = PokerDatabase(tmp_path / "solver.db")
    db.init_db()
    assert db.schema_version() == SCHEMA_VERSION
    session = db.create_session(Session(name="Solver test"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="NLHE cash",
            table_size=6,
        )
    )
    profile = db.create_solver_range_profile(
        SolverRangeProfile(name="My BTN range", notation="22+,A2s+")
    )
    assert db.fetch_solver_range_profiles()[0].id == profile.id
    run = db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            input_hash="a" * 64,
            spot={"street": "flop"},
            assumptions=["no rake"],
        )
    )
    db.update_solver_run(
        run.id,
        status="completed",
        evidence={"backend": "TexasSolver"},
        completed_at=datetime.now(UTC),
    )
    saved = db.fetch_cached_solver_run("a" * 64)
    assert saved is not None
    assert saved.assumptions == ["no rake"]
    exported = export_range_profiles(db.fetch_solver_range_profiles())
    imported = import_range_profiles(exported)
    assert imported[0].name == "My BTN range"
    db.delete_hand(hand.id)
    assert db.fetch_solver_run(run.id) is None
    db.close()


def test_solver_evidence_is_grounded_in_coaching_prompt() -> None:
    hand, players, actions, _, spot = _completed_cash_spot()
    evidence = SolverEvidence(
        backend_version=PINNED_CONSOLE_COMMIT,
        street="flop",
        board=spot.board,
        pot=spot.pot,
        effective_stack=spot.effective_stack,
        hero_player="Hero",
        hero_combo="Ah Qs",
        recorded_action="check",
        mapped_action="CHECK",
        action_frequencies=[
            ActionFrequency(action="CHECK", frequency=0.6),
            ActionFrequency(action="BET 3.75", frequency=0.4),
        ],
        range_action_frequencies=[
            ActionFrequency(action="CHECK", frequency=0.3),
            ActionFrequency(action="BET 3.75", frequency=0.7),
        ],
        range_ip_name="IP test",
        range_oop_name="OOP test",
        assumptions=["No-rake equilibrium approximation."],
    )
    prompt = build_hand_review_prompt(
        Session(id=1, name="Solver session"),
        hand,
        actions,
        players,
        solver_evidence=evidence,
    )
    assert "CHECK 60.0%" in prompt
    assert "BET 3.75 40.0%" in prompt
    assert "action_ev_and_bb_loss: unavailable" in prompt
    assert "do not invent either" in prompt
    assert "input_weighted_range_strategy: omitted because" in prompt
    assert "CHECK 30.0%" not in prompt

    validate_solver_coaching_response(
        "Theory Coach\nCHECK 60% and BET 3.75 40.0% form a genuine mix.",
        evidence,
    )
    with pytest.raises(ValueError, match="saved solver frequency|did not preserve"):
        validate_solver_coaching_response(
            "Theory Coach\nCHECK 50% and BET 3.75 50%.",
            evidence,
        )
    with pytest.raises(ValueError, match="action EV or BB loss"):
        validate_solver_coaching_response(
            "CHECK 60% and BET 3.75 40%. Checking loses exactly 4.2 BB.",
            evidence,
        )
    for unsupported in (
        "This line burns 4.2 BB.",
        "The solver says checking is -4.2 BB.",
        "You give up exactly 4.2 big blinds.",
        "The EV difference is 4.2 BB.",
        "Checking forfeits 4.2 BB.",
        "The EV loss from checking equals 4.2 BB.",
        "Checking has 4.2 BB lower EV.",
        "The difference between checking and betting is 4.2 BB.",
        "Checking surrenders 4.2 big blinds.",
        "Checking loses 4.2 blinds.",
        "Checking underperforms by 4.2 BB.",
        "Checking is 4.2 BB inferior.",
        "You give away 4.2 BB by checking.",
        "The check is worth 4.2 BB less.",
        "Betting earns an extra 4.2 BB.",
        "The bet adds 4.2 BB to your expectation.",
        "Checking sacrifices four big blinds.",
        "Checking costs 420 chips.",
        "Checking burns the whole 5 BB pot.",
        "Checking costs the entire 5 BB pot.",
    ):
        with pytest.raises(ValueError, match="action EV or BB loss"):
            validate_solver_coaching_response(
                f"CHECK 60% and BET 3.75 40%. {unsupported}",
                evidence,
            )
    with pytest.raises(ValueError, match="omitted"):
        validate_solver_coaching_response(
            "CHECK and BET 3.75 are options. Pot odds are 60% and equity is 40%.",
            evidence,
        )
    for invented in ("FOLD 20%.", "CALL 12%.", "Raising is used 50%."):
        with pytest.raises(ValueError, match="invented"):
            validate_solver_coaching_response(
                f"CHECK 60% and BET 3.75 40%. {invented}",
                evidence,
            )
    for invented in ("20% FOLD.", "JAM 20%.", "SHOVE 20%."):
        with pytest.raises(ValueError, match="invented"):
            validate_solver_coaching_response(
                f"CHECK 60% and BET 3.75 40%. {invented}",
                evidence,
            )
    validate_solver_coaching_response(
        "60% CHECK and 40% BET 3.75.",
        evidence,
    )
    for truthful in (
        "CHECK (60%) and BET 3.75 (40%).",
        "CHECK is used 60% and BET 3.75 BB at 40%.",
        "CHECK, 60%; BET 3.75, 40%.",
        "CHECK frequency: 60%; BET 3.75 frequency: 40%.",
        "In a 5 BB pot, checking 60% controls the pot; BET 3.75 40%.",
        "At a 97.5 BB effective stack, checking 60% is mixed with BET 3.75 40%.",
        "Checking 60% and BET 3.75 40% are the mix in this 5 BB pot.",
    ):
        validate_solver_coaching_response(truthful, evidence)


@pytest.mark.parametrize(
    "game_type",
    [
        "Limit Hold'em cash",
        "Stud cash",
        "banana cash",
        "PLO cash",
        "PLO NLHE cash",
        "Omaha / NLHE cash",
        "short deck NLHE cash",
        "NL Holdem fixed limit",
        "NL Holdem pot limit",
        "NLHE Sit and Go",
        "NLHE sit-and-go",
        "NLHE tourney",
        "NLHE Tourn",
        "NLHE satellite",
        "NLHE freeroll",
        "NLHE freezeout",
        "NLHE bounty event",
        "NLHE event",
        "NLHE turbo",
        "NLHE bracelet",
        "NLHE championship",
        "NLHE WSOP",
    ],
)
def test_solver_rejects_non_nlhe_game_labels(game_type: str) -> None:
    hand, players, actions, accounting, _ = _completed_cash_spot()
    prepared = prepare_solver_spot(
        hand.model_copy(update={"game_type": game_type}),
        players,
        actions,
        accounting,
    )
    assert not prepared.eligibility.eligible
    assert prepared.spot is None
    assert any(
        "no-limit Hold'em only" in reason
        or "Tournament and ICM" in reason
        or "explicit cash" in reason
        for reason in prepared.eligibility.reasons
    )


def test_solver_rejects_blank_game_type() -> None:
    hand, players, actions, accounting, _ = _completed_cash_spot()
    prepared = prepare_solver_spot(
        hand.model_copy(update={"game_type": ""}),
        players,
        actions,
        accounting,
    )
    assert not prepared.eligibility.eligible
    assert any("Record the game type" in item for item in prepared.eligibility.reasons)


def test_solver_rejects_first_actor_that_conflicts_with_positions() -> None:
    hand, players, actions, accounting, _ = _completed_cash_spot()
    actions[3] = actions[3].model_copy(
        update={
            "player_key": "villain",
            "player_name": "Villain",
            "position": "BTN",
            "action_type": "bet",
            "amount": 1.0,
        }
    )
    prepared = prepare_solver_spot(hand, players, actions, accounting)
    assert not prepared.eligibility.eligible
    assert any("conflicts with the saved positions" in item for item in prepared.eligibility.reasons)


def test_solver_rejects_postflop_node_with_preflop_all_in_player() -> None:
    hand, players, actions, accounting, _ = _completed_cash_spot()
    players[1] = players[1].model_copy(update={"starting_stack": 2.5})
    prepared = prepare_solver_spot(hand, players, actions, accounting)
    assert not prepared.eligibility.eligible
    assert any("must still have chips" in item for item in prepared.eligibility.reasons)


def test_solver_worker_runs_configured_cli_and_saves_evidence(
    tmp_path, monkeypatch
) -> None:
    fake_solver = tmp_path / "console_solver"
    fake_solver.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

command = pathlib.Path(sys.argv[2]).read_text()
resources = pathlib.Path(sys.argv[sys.argv.index("-r") + 1])
assert (resources / "compairer" / "card5_dic_sorted.txt").is_file()
output = next(line.split(" ", 1)[1] for line in command.splitlines() if line.startswith("dump_result "))
pathlib.Path(output).write_text(json.dumps({
    "node_type": "action_node",
    "actions": ["CHECK", "BET 3.75"],
    "childrens": {},
    "strategy": {
        "actions": ["CHECK", "BET 3.75"],
        "strategy": {"AhQs": [0.7, 0.3]}
    }
}))
print("Total exploitability 0.33 percent")
""",
        encoding="utf-8",
    )
    fake_solver.chmod(0o755)
    evaluator = tmp_path / "resources" / "compairer" / "card5_dic_sorted.txt"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("test", encoding="utf-8")
    monkeypatch.setenv("TEXAS_SOLVER_PATH", str(fake_solver))

    _, _, _, _, spot = _completed_cash_spot()
    oop = resolve_custom_range(spot, spot.oop, "AQo+,AA", name="Hero custom")
    ip = resolve_custom_range(spot, spot.ip, "22+,A2s+", name="Villain custom")
    db_path = tmp_path / "worker.db"
    db = PokerDatabase(db_path)
    db.init_db()
    session = db.create_session(Session(name="Worker test"))
    hand = db.create_hand(
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="NLHE cash",
            table_size=6,
        )
    )
    spot = spot.model_copy(update={"hand_id": hand.id})
    result_path = tmp_path / "result.json"
    command_path = tmp_path / "input.txt"
    log_path = tmp_path / "solver.log"
    command_path.write_text(
        build_command_file(
            spot,
            ip,
            oop,
            output_path=result_path,
            thread_count=1,
        ),
        encoding="utf-8",
    )
    run = db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            input_hash="b" * 64,
            backend_version=PINNED_CONSOLE_COMMIT,
            spot=spot.model_dump(mode="json"),
            range_ip=ip.model_dump(mode="json"),
            range_oop=oop.model_dump(mode="json"),
            command_path=str(command_path),
            result_path=str(result_path),
            log_path=str(log_path),
        )
    )
    run_solver_job(db, run.id, timeout_seconds=30)

    check = PokerDatabase(db_path)
    saved = check.fetch_solver_run(run.id)
    assert saved is not None
    assert saved.status == "completed"
    assert saved.exploitability_pct == 0.33
    evidence = SolverEvidence.model_validate(saved.evidence)
    assert evidence.action_frequencies[0].frequency == 0.7
    check.close()


def test_resource_directory_can_be_explicit_and_contain_spaces(
    tmp_path, monkeypatch
) -> None:
    resources = tmp_path / "solver resources"
    evaluator = resources / "compairer" / "card5_dic_sorted.txt"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("test", encoding="utf-8")
    monkeypatch.setenv("TEXAS_SOLVER_RESOURCE_DIR", str(resources))
    assert configured_resource_dir(tmp_path / "console_solver") == resources.resolve()


def test_recorded_line_uses_ledger_normalized_incremental_amount() -> None:
    hand, players, actions, accounting, _ = _completed_cash_spot()
    actions[-1] = actions[-1].model_copy(
        update={
            "action_type": "raise",
            "amount": 10.0,
            "amount_semantics": "raise_to",
        }
    )
    snapshots = list(accounting.ledger.snapshots)
    snapshots[-1] = replace(snapshots[-1], kind="raise", amount=6.25)
    normalized_accounting = replace(
        accounting,
        ledger=replace(accounting.ledger, snapshots=tuple(snapshots)),
    )
    prepared = prepare_solver_spot(hand, players, actions, normalized_accounting)
    assert prepared.spot is not None
    assert prepared.spot.recorded_line[-1].amount == 6.25


def test_solver_cache_key_excludes_hand_mapping_but_includes_tree_inputs() -> None:
    _, _, _, _, spot = _completed_cash_spot()
    oop = resolve_custom_range(spot, spot.oop, "AQo+,AA")
    ip = resolve_custom_range(spot, spot.ip, "22+,A2s+")
    remapped = spot.model_copy(
        update={
            "hand_id": 999,
            "recorded_line": [
                spot.recorded_line[0].model_copy(update={"player_name": "Renamed Hero"})
            ],
        }
    )
    assert input_hash(spot, ip, oop) == input_hash(remapped, ip, oop)


def test_cache_hit_creates_current_hand_run_with_independent_artifacts(
    tmp_path, monkeypatch
) -> None:
    fake_solver, _ = _configure_fake_solver_install(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "poker_tracker.solver.jobs.solver_run_directory",
        lambda run_id: _run_directory(tmp_path, run_id),
    )
    monkeypatch.setattr(
        "poker_tracker.solver.jobs.subprocess.Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cache hit must not launch a solver")
        ),
    )
    _, _, _, _, spot = _completed_cash_spot()
    oop = resolve_custom_range(spot, spot.oop, "AQo+,AA")
    ip = resolve_custom_range(spot, spot.ip, "22+,A2s+")
    db = PokerDatabase(tmp_path / "cache.db")
    db.init_db()
    session = db.create_session(Session(name="Cache"))
    first = db.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="NLHE cash", table_size=6)
    )
    second = db.create_hand(
        Hand(session_id=session.id, hand_number=2, game_type="NLHE cash", table_size=6)
    )
    source_dir = _run_directory(tmp_path, 900)
    source_result = source_dir / "result.json"
    source_log = source_dir / "solver.log"
    source_result.write_text(
        json.dumps(
            {
                "node_type": "action_node",
                "actions": ["CHECK", "BET 3.75"],
                "strategy": {
                    "actions": ["CHECK", "BET 3.75"],
                    "strategy": {"AhQs": [0.7, 0.3]},
                },
            }
        ),
        encoding="utf-8",
    )
    source_log.write_text("Total exploitability 0.4 percent", encoding="utf-8")
    source_spot = spot.model_copy(update={"hand_id": first.id})
    digest = input_hash(source_spot, ip, oop)
    source = db.create_solver_run(
        SolverRun(
            hand_id=first.id,
            status="completed",
            input_hash=digest,
            backend_version=PINNED_CONSOLE_COMMIT,
            spot=source_spot.model_dump(mode="json"),
            range_ip=ip.model_dump(mode="json"),
            range_oop=oop.model_dump(mode="json"),
            result_path=str(source_result),
            log_path=str(source_log),
            exploitability_pct=0.4,
            runtime_seconds=12,
            completed_at=datetime.now(UTC),
        )
    )
    cached = start_solver_job(
        db,
        spot.model_copy(update={"hand_id": second.id}),
        ip,
        oop,
    )
    assert fake_solver.is_file()
    assert cached.id != source.id
    assert cached.hand_id == second.id
    assert cached.status == "completed"
    assert cached.result_path != source.result_path
    assert Path(cached.result_path).is_file()
    assert any("Reused strategy artifacts" in item for item in cached.assumptions)
    db.close()


def test_solver_job_reservation_is_atomic_across_connections(
    tmp_path, monkeypatch
) -> None:
    _configure_fake_solver_install(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "poker_tracker.solver.jobs.solver_run_directory",
        lambda run_id: _run_directory(tmp_path, run_id),
    )

    class DummyProcess:
        pid = 43210

    monkeypatch.setattr(
        "poker_tracker.solver.jobs.subprocess.Popen",
        lambda *args, **kwargs: DummyProcess(),
    )
    db_path = tmp_path / "race.db"
    owner = PokerDatabase(db_path)
    owner.init_db()
    session = owner.create_session(Session(name="Race"))
    hand = owner.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="NLHE cash", table_size=6)
    )
    _, _, _, _, original = _completed_cash_spot()
    spot = original.model_copy(update={"hand_id": hand.id})
    oop = resolve_custom_range(spot, spot.oop, "AQo+,AA")
    ip = resolve_custom_range(spot, spot.ip, "22+,A2s+")
    connections = [PokerDatabase(db_path), PokerDatabase(db_path)]
    barrier = Barrier(2)

    def attempt(db: PokerDatabase):
        barrier.wait()
        try:
            return start_solver_job(db, spot, ip, oop)
        except SolverJobAlreadyRunningError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, connections))
    assert sum(isinstance(result, SolverRun) for result in results) == 1
    assert sum(isinstance(result, SolverJobAlreadyRunningError) for result in results) == 1
    assert len(owner.fetch_active_solver_runs()) == 1
    for connection in connections:
        connection.close()
    owner.close()


def test_invalid_thread_configuration_does_not_leave_queued_run(
    tmp_path, monkeypatch
) -> None:
    _configure_fake_solver_install(tmp_path, monkeypatch)
    monkeypatch.setenv("POKERTRAINER_SOLVER_THREADS", "abc")
    db = PokerDatabase(tmp_path / "threads.db")
    db.init_db()
    session = db.create_session(Session(name="Threads"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="NLHE cash", table_size=6)
    )
    _, _, _, _, original = _completed_cash_spot()
    spot = original.model_copy(update={"hand_id": hand.id})
    oop = resolve_custom_range(spot, spot.oop, "AQo+,AA")
    ip = resolve_custom_range(spot, spot.ip, "22+,A2s+")
    with pytest.raises(ValueError, match="must be an integer"):
        start_solver_job(db, spot, ip, oop)
    assert db.fetch_active_solver_runs() == []
    db.close()


def test_stale_reconciliation_preserves_fresh_queue_and_terminates_live_stale(
    tmp_path, monkeypatch
) -> None:
    db = PokerDatabase(tmp_path / "stale.db")
    db.init_db()
    session = db.create_session(Session(name="Stale"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="NLHE cash", table_size=6)
    )
    now = datetime.now(UTC)
    fresh_queue = db.create_solver_run(
        SolverRun(hand_id=hand.id, input_hash="fresh", created_at=now)
    )
    stale_live = db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            status="running",
            input_hash="live",
            pid=111,
            heartbeat_at=now - timedelta(minutes=10),
        )
    )
    fresh_dead = db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            status="running",
            input_hash="dead",
            pid=222,
            heartbeat_at=now,
        )
    )
    monkeypatch.setattr(
        solver_jobs,
        "_pid_is_alive",
        lambda pid: pid == 111,
    )
    terminated: list[int | None] = []
    monkeypatch.setattr(
        solver_jobs,
        "_terminate_solver_group",
        lambda pid: terminated.append(pid) or True,
    )
    failed = reconcile_stale_solver_runs(db, now=now)
    assert set(failed) == {stale_live.id, fresh_dead.id}
    assert terminated == [111]
    assert db.fetch_solver_run(fresh_queue.id).status == "queued"
    db.close()


def test_pid_permission_error_is_treated_as_alive(monkeypatch) -> None:
    def denied(_pid, _signal_number):
        raise PermissionError

    monkeypatch.setattr(solver_jobs.os, "kill", denied)
    assert solver_jobs._pid_is_alive(123)


def test_hand_invalidation_cancels_active_solve_and_blocks_late_completion(
    tmp_path,
) -> None:
    db = PokerDatabase(tmp_path / "invalidate.db")
    db.init_db()
    session = db.create_session(Session(name="Invalidate"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="NLHE cash", table_size=6)
    )
    run = db.create_solver_run(
        SolverRun(
            hand_id=hand.id,
            status="running",
            input_hash="invalidate",
            pid=123,
        )
    )
    db._invalidate_hand_derivatives(hand.id)
    db._commit()
    invalidated = db.fetch_solver_run(run.id)
    assert invalidated.status == "cancelling"
    late = db.update_solver_run(
        run.id,
        expected_statuses=("running",),
        status="completed",
        evidence={"wrong": "stale"},
    )
    assert late.status == "cancelling"
    assert late.evidence == {}
    db.close()


def test_hand_edit_during_worker_launch_cannot_resurrect_run(
    tmp_path, monkeypatch
) -> None:
    _configure_fake_solver_install(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "poker_tracker.solver.jobs.solver_run_directory",
        lambda run_id: _run_directory(tmp_path, run_id),
    )
    db = PokerDatabase(tmp_path / "launch-edit.db")
    db.init_db()
    session = db.create_session(Session(name="Launch edit"))
    hand = db.create_hand(
        Hand(session_id=session.id, hand_number=1, game_type="NLHE cash", table_size=6)
    )
    _, _, _, _, original = _completed_cash_spot()
    spot = original.model_copy(update={"hand_id": hand.id})
    oop = resolve_custom_range(spot, spot.oop, "AQo+,AA")
    ip = resolve_custom_range(spot, spot.ip, "22+,A2s+")

    class DummyProcess:
        pid = 54321

    def invalidate_during_launch(*args, **kwargs):
        db._invalidate_hand_derivatives(hand.id)
        db._commit()
        return DummyProcess()

    monkeypatch.setattr(
        "poker_tracker.solver.jobs.subprocess.Popen",
        invalidate_during_launch,
    )
    terminated: list[int | None] = []
    monkeypatch.setattr(
        solver_jobs,
        "_terminate_solver_group",
        lambda pid: terminated.append(pid) or True,
    )
    with pytest.raises(ValueError, match="hand changed"):
        start_solver_job(db, spot, ip, oop)
    runs = db.fetch_solver_runs_by_hand(hand.id)
    assert runs[0].status == "stale"
    assert terminated == [54321]
    assert db.fetch_active_solver_runs() == []
    db.close()


def _configure_fake_solver_install(tmp_path, monkeypatch):
    install = tmp_path / "fake install"
    binary = install / "console_solver"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    evaluator = install / "resources" / "compairer" / "card5_dic_sorted.txt"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("test", encoding="utf-8")
    monkeypatch.setenv("TEXAS_SOLVER_PATH", str(binary))
    return binary, evaluator


def _run_directory(tmp_path, run_id: int) -> Path:
    path = tmp_path / "solver runs" / f"run_{run_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path
