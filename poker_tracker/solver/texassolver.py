from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path

from poker_tracker.solver.models import (
    ActionFrequency,
    RecordedSolverAction,
    ResolvedRange,
    SolverEvidence,
    SolverSpot,
)

BACKEND_NAME = "texassolver"
PINNED_CONSOLE_COMMIT = "42313c9cce96130d2341a8fc265160f580956054"
DEFAULT_ACCURACY_PCT = 0.5
DEFAULT_MAX_ITERATIONS = 200
DEFAULT_TIMEOUT_SECONDS = 30 * 60
TEXASSOLVER_MIN_WEIGHT = 0.005
TREE_SPEC = {
    "flop_bets": [33, 75],
    "turn_river_bets": [50, 100],
    "raises": [50, 100],
    "turn_river_oop_donk": [50],
    "allin": True,
    "allin_threshold": 0.67,
    "isomorphism": False,
    "dump_rounds": 4,
    "print_interval": 10,
}
_EXPLOITABILITY = re.compile(
    r"Total exploitability\s+([0-9]+(?:\.[0-9]+)?)\s+(?:precent|percent)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def configured_binary() -> Path:
    configured = os.environ.get("TEXAS_SOLVER_PATH", "").strip()
    if not configured:
        raise FileNotFoundError(
            "TexasSolver is not configured. Set TEXAS_SOLVER_PATH to the console_solver binary."
        )
    binary = Path(configured).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"TexasSolver binary was not found: {binary}")
    if not os.access(binary, os.X_OK):
        raise PermissionError(f"TexasSolver binary is not executable: {binary}")
    return binary


def configured_resource_dir(binary: Path | None = None) -> Path:
    configured = os.environ.get("TEXAS_SOLVER_RESOURCE_DIR", "").strip()
    resource_dir = (
        Path(configured).expanduser().resolve()
        if configured
        else (binary or configured_binary()).parent / "resources"
    )
    evaluator = resource_dir / "compairer" / "card5_dic_sorted.txt"
    if not evaluator.is_file():
        raise FileNotFoundError(
            "TexasSolver evaluator resources were not found. Keep the installed "
            f"'resources' directory beside console_solver or set "
            f"TEXAS_SOLVER_RESOURCE_DIR (looked for {evaluator})."
        )
    return resource_dir.resolve()


def build_command_file(
    spot: SolverSpot,
    range_ip: ResolvedRange,
    range_oop: ResolvedRange,
    *,
    output_path: str | Path,
    thread_count: int,
    accuracy_pct: float = DEFAULT_ACCURACY_PCT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> str:
    lines = [
        f"set_pot {spot.pot:g}",
        f"set_effective_stack {spot.effective_stack:g}",
        f"set_board {','.join(spot.board.split())}",
        f"set_range_ip {range_ip.solver_notation}",
        f"set_range_oop {range_oop.solver_notation}",
    ]
    streets = ("flop", "turn", "river")
    start = streets.index(spot.street)
    for street in streets[start:]:
        sizes = ",".join(
            str(size)
            for size in (
                TREE_SPEC["flop_bets"]
                if street == "flop"
                else TREE_SPEC["turn_river_bets"]
            )
        )
        for role in ("oop", "ip"):
            lines.extend(
                [
                    f"set_bet_sizes {role},{street},bet,{sizes}",
                    "set_bet_sizes "
                    f"{role},{street},raise,"
                    + ",".join(str(size) for size in TREE_SPEC["raises"]),
                ]
            )
            if TREE_SPEC["allin"]:
                lines.append(f"set_bet_sizes {role},{street},allin")
        if street in {"turn", "river"}:
            lines.append(
                f"set_bet_sizes oop,{street},donk,"
                + ",".join(str(size) for size in TREE_SPEC["turn_river_oop_donk"])
            )
    lines.extend(
        [
            f"set_allin_threshold {TREE_SPEC['allin_threshold']}",
            "build_tree",
            f"set_thread_num {max(1, thread_count)}",
            f"set_accuracy {accuracy_pct:g}",
            f"set_max_iteration {max_iterations}",
            f"set_print_interval {TREE_SPEC['print_interval']}",
            f"set_use_isomorphism {int(bool(TREE_SPEC['isomorphism']))}",
            "start_solve",
            f"set_dump_rounds {TREE_SPEC['dump_rounds']}",
            # The console command parser rejects parameters containing spaces. The
            # adapter executes from the run directory, so a basename is both safe
            # and sufficient even when POKER_DATA_DIR contains spaces.
            f"dump_result {Path(output_path).name}",
        ]
    )
    return "\n".join(lines) + "\n"


def input_hash(
    spot: SolverSpot,
    range_ip: ResolvedRange,
    range_oop: ResolvedRange,
    *,
    accuracy_pct: float = DEFAULT_ACCURACY_PCT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> str:
    payload = {
        "backend": BACKEND_NAME,
        "backend_version": PINNED_CONSOLE_COMMIT,
        "spot": {
            "street": spot.street,
            "board": spot.board,
            "pot": spot.pot,
            "effective_stack": spot.effective_stack,
        },
        "range_ip": range_ip.solver_notation,
        "range_oop": range_oop.solver_notation,
        "accuracy_pct": accuracy_pct,
        "max_iterations": max_iterations,
        "tree": TREE_SPEC,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_final_exploitability(log_text: str) -> float | None:
    values = [float(match.group(1)) for match in _EXPLOITABILITY.finditer(log_text)]
    return values[-1] if values else None


def parse_strategy_result(
    result_path: str | Path,
    *,
    spot: SolverSpot,
    range_ip: ResolvedRange,
    range_oop: ResolvedRange,
    backend_version: str,
    exploitability_pct: float | None,
    runtime_seconds: float | None,
    assumptions: list[str],
) -> SolverEvidence:
    with Path(result_path).open(encoding="utf-8") as source:
        root = json.load(source)
    if not isinstance(root, dict):
        raise ValueError("TexasSolver output root must be a JSON object.")

    hero = spot.oop if spot.oop.is_hero else spot.ip
    node: Mapping[str, object] = root
    recorded_hero_action = ""
    mapped_hero_action = ""
    warnings: list[str] = []
    if exploitability_pct is None:
        warnings.append(
            "Final exploitability was unavailable; convergence to the 0.5% target "
            "could not be verified."
        )
    elif exploitability_pct > DEFAULT_ACCURACY_PCT:
        warnings.append(
            f"Final exploitability {exploitability_pct:g}% exceeded the "
            f"{DEFAULT_ACCURACY_PCT:g}% target after at most "
            f"{DEFAULT_MAX_ITERATIONS} iterations."
        )
    for recorded in spot.recorded_line:
        if recorded.player_key == hero.player_key:
            recorded_hero_action = _action_label(recorded)
            mapped_hero_action = map_recorded_action(node, recorded) or ""
            break
        selected = map_recorded_action(node, recorded)
        if selected is None:
            warnings.append(
                f"Could not map {recorded.player_name}'s recorded {_action_label(recorded)} "
                "to the solver tree."
            )
            break
        children = node.get("childrens")
        if not isinstance(children, dict) or selected not in children:
            warnings.append(f"Solver output did not include the recorded branch {selected}.")
            break
        child = children[selected]
        if not isinstance(child, dict):
            break
        node = child

    hero_range = range_oop if hero.role == "oop" else range_ip
    actions, combo_frequencies, range_frequencies = _node_strategy(
        node, spot.hero_cards, hero_range.solver_notation
    )
    if not combo_frequencies:
        warnings.append(
            "The recorded Hero combo was not present at the mapped decision node; "
            "only range-level frequencies are available."
        )
    if recorded_hero_action and not mapped_hero_action:
        warnings.append("The recorded Hero action could not be mapped to an allowed solver size.")
    if actions and mapped_hero_action and mapped_hero_action not in actions:
        warnings.append("The nearest recorded action is not available at the mapped strategy node.")

    return SolverEvidence(
        backend="TexasSolver",
        backend_version=backend_version,
        street=spot.street,
        board=spot.board,
        pot=spot.pot,
        effective_stack=spot.effective_stack,
        hero_player=hero.player_name,
        hero_combo=spot.hero_cards,
        recorded_action=recorded_hero_action,
        mapped_action=mapped_hero_action,
        action_frequencies=combo_frequencies,
        range_action_frequencies=range_frequencies,
        exploitability_pct=exploitability_pct,
        runtime_seconds=runtime_seconds,
        range_ip_name=range_ip.profile_name,
        range_oop_name=range_oop.profile_name,
        assumptions=assumptions,
        warnings=warnings,
    )


def map_recorded_action(
    node: Mapping[str, object], action: RecordedSolverAction
) -> str | None:
    children = node.get("childrens")
    available = list(children) if isinstance(children, dict) else []
    if not available:
        raw_actions = node.get("actions")
        available = [str(item) for item in raw_actions] if isinstance(raw_actions, list) else []
    desired = action.action_type.lower()
    exact = {
        "check": "CHECK",
        "call": "CALL",
        "fold": "FOLD",
    }.get(desired)
    if exact and exact in available:
        return exact
    prefix = "RAISE" if desired == "raise" else "BET"
    if desired == "all-in":
        all_in = next((item for item in available if "ALLIN" in item.replace("-", "")), None)
        if all_in:
            return all_in
        candidates = [
            item for item in available if item.startswith("BET") or item.startswith("RAISE")
        ]
        return max(candidates, key=_action_number, default=None)
    candidates = [item for item in available if item.startswith(prefix)]
    if not candidates:
        return None
    target = float(action.amount or 0)
    return min(candidates, key=lambda item: abs(_action_number(item) - target))


def _node_strategy(
    node: Mapping[str, object], hero_cards: str, range_notation: str
) -> tuple[list[str], list[ActionFrequency], list[ActionFrequency]]:
    strategy_wrapper = node.get("strategy")
    actions: list[str] = []
    strategy_map: Mapping[str, object] = {}
    if isinstance(strategy_wrapper, dict):
        raw_actions = strategy_wrapper.get("actions")
        if isinstance(raw_actions, list):
            actions = [str(item) for item in raw_actions]
        raw_strategy = strategy_wrapper.get("strategy")
        if isinstance(raw_strategy, dict):
            strategy_map = raw_strategy
    if not actions:
        raw_actions = node.get("actions")
        if isinstance(raw_actions, list):
            actions = [str(item) for item in raw_actions]
    if not actions or not strategy_map:
        return actions, [], []

    hero_tokens = hero_cards.split()
    hero_keys = set()
    if len(hero_tokens) == 2:
        hero_keys = {
            "".join(hero_tokens),
            "".join(reversed(hero_tokens)),
        }
    combo_values: list[float] | None = None
    weights = _exact_combo_weights(range_notation)
    totals = [0.0] * len(actions)
    total_weight = 0.0
    for combo, raw_values in strategy_map.items():
        if not isinstance(raw_values, list) or len(raw_values) != len(actions):
            raise ValueError(
                f"TexasSolver strategy for {combo} does not match its action vector."
            )
        if any(isinstance(value, bool) for value in raw_values):
            raise ValueError(f"TexasSolver strategy for {combo} contains a nonnumeric value.")
        try:
            values = [float(value) for value in raw_values]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TexasSolver strategy for {combo} contains a nonnumeric value."
            ) from exc
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError(
                f"TexasSolver strategy for {combo} contains an invalid frequency."
            )
        if not math.isclose(sum(values), 1.0, abs_tol=1e-3):
            raise ValueError(
                f"TexasSolver strategy frequencies for {combo} do not sum to 1."
            )
        combo_text = str(combo)
        combo_weight = weights.get(combo_text, weights.get(_reverse_combo(combo_text), 0.0))
        if combo_weight > 0:
            totals = [
                total + (value * combo_weight)
                for total, value in zip(totals, values, strict=True)
            ]
            total_weight += combo_weight
        if combo_text in hero_keys:
            combo_values = values
    combo_frequencies = (
        [
            ActionFrequency(action=action, frequency=value)
            for action, value in zip(actions, combo_values, strict=True)
        ]
        if combo_values is not None
        else []
    )
    range_frequencies = (
        [
            ActionFrequency(action=action, frequency=total / total_weight)
            for action, total in zip(actions, totals, strict=True)
        ]
        if total_weight
        else []
    )
    return actions, combo_frequencies, range_frequencies


def _action_label(action: RecordedSolverAction) -> str:
    return (
        action.action_type
        if action.amount is None
        else f"{action.action_type} {action.amount:g} BB"
    )


def _action_number(action: str) -> float:
    match = _NUMBER.search(action)
    return float(match.group(1)) if match else 0.0


def _exact_combo_weights(notation: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for raw_token in notation.split(","):
        token = raw_token.strip()
        if not token:
            continue
        combo, separator, raw_weight = token.partition(":")
        if len(combo) != 4:
            continue
        weight = float(raw_weight) if separator else 1.0
        weights[combo] = weight
    return weights


def _reverse_combo(combo: str) -> str:
    return combo[2:] + combo[:2] if len(combo) == 4 else combo
