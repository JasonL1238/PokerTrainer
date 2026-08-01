from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

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
# How far a recorded bet may sit from the nearest size the tree offers, measured
# against the pot it was made into. Chips are the wrong unit: two blinds of
# substitution into a five blind pot is a different decision, while the same two
# blinds into a two hundred blind pot is rounding.
#
# The flop abstraction offers 33% and 75% pot, so the widest gap between
# neighbouring sizes is 42% of pot and any bet strictly between them lands at
# most 21% of pot from one of them; turn and river offer 50% and 100%, where the
# same figure is 25%. A quarter-pot ceiling therefore admits every size the
# abstraction was built to approximate -- including a pot-sized flop bet, which
# sits exactly on the bound -- and refuses everything the abstraction was never
# meant to stand in for. Below 2% of pot the difference is unit conversion
# rather than a decision, so it is not worth an operator's attention.
ACTION_MAPPING_EXACT_POT_FRACTION = 0.02
ACTION_MAPPING_MAX_POT_FRACTION = 0.25
# The share of the submitted range, by the operator's own weights, that the
# dumped strategy map must describe before the result counts as a solve of the
# range that was asked about. The only legitimate way for combinations to go
# missing is TexasSolver's internal minimum-weight pruning, and parse_range
# already refuses to submit anything at or below that cutoff, so a correct dump
# covers essentially the whole range. Half is a wide margin below any plausible
# legitimate loss: under it, the map describes a different node's range or the
# file is incomplete, and neither is evidence about this hand.
MIN_STRATEGY_RANGE_COVERAGE = 0.5
FULL_STRATEGY_RANGE_COVERAGE = 0.999
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


class SolverResultUnusableError(ValueError):
    """A dump that cannot be read as evidence about the recorded hand.

    Subclasses ValueError because the worker and the cache materializer already
    turn a parse failure into a failed run with a message the operator sees.
    Refusing costs one study spot; accepting would file frequencies computed for
    some other decision under this hand's number, where nothing downstream can
    tell them apart from a real result.
    """


@dataclass(frozen=True)
class ActionMapping:
    """How a recorded action lines up with the branches the solve tree offers.

    ``label`` is set only when the branch may actually be used, so a caller that
    ignores ``quality`` still cannot descend into a branch that answers a
    different question. ``detail`` is written to be shown to an operator and
    handed to the coaching model verbatim.
    """

    label: str | None
    quality: Literal["exact", "approximate", "unusable", "absent"]
    detail: str
    pot_fraction_error: float | None = None

    @property
    def is_usable(self) -> bool:
        return self.quality in {"exact", "approximate"}


class NodeStrategy(NamedTuple):
    actions: list[str]
    combo_frequencies: list[ActionFrequency]
    range_frequencies: list[ActionFrequency]
    range_coverage: float
    strategy_combos: int


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
    recorded_hero: RecordedSolverAction | None = None
    hero_mapping = ActionMapping(None, "absent", "no Hero decision was reached")
    for recorded in spot.recorded_line:
        mapping = map_recorded_action(
            node, recorded, pot_reference=_pot_reference(recorded, spot)
        )
        if recorded.player_key == hero.player_key:
            recorded_hero = recorded
            hero_mapping = mapping
            break
        # Everything below this point is read out of the node the walk lands on,
        # so a villain action that has no honest counterpart in the tree cannot
        # be approximated away: descending anyway would swap the decision the
        # evidence is about while leaving the recorded line beside it unchanged.
        if not mapping.is_usable:
            raise SolverResultUnusableError(
                f"{recorded.player_name}'s recorded {_action_label(recorded)} cannot be "
                f"placed in the solver tree: {mapping.detail}. Strategy read past that "
                "point would describe a different decision."
            )
        children = node.get("childrens")
        if not isinstance(children, dict) or mapping.label not in children:
            raise SolverResultUnusableError(
                f"The solver output has no {mapping.label} branch for "
                f"{recorded.player_name}'s recorded {_action_label(recorded)}."
            )
        child = children[mapping.label]
        if not isinstance(child, dict):
            raise SolverResultUnusableError(
                f"The solver output's {mapping.label} branch is not a decision node."
            )
        if mapping.quality == "approximate":
            warnings.append(
                f"{recorded.player_name}'s recorded {_action_label(recorded)} was solved "
                f"as {mapping.label}: {mapping.detail}."
            )
        node = child
    if recorded_hero is None:
        raise SolverResultUnusableError(
            "The recorded line reaches no Hero decision, so this solve is not evidence "
            "about a choice Hero made."
        )

    recorded_hero_action = _action_label(recorded_hero)
    mapped_hero_action = hero_mapping.label or ""
    if not hero_mapping.is_usable:
        warnings.append(
            f"Hero's recorded {recorded_hero_action} has no counterpart in the solve "
            f"tree: {hero_mapping.detail}. The frequencies below are the equilibrium "
            "for this decision, but they cannot be compared with what Hero did."
        )
    elif hero_mapping.quality == "approximate":
        warnings.append(
            f"Hero's recorded {recorded_hero_action} was matched to "
            f"{mapped_hero_action}: {hero_mapping.detail}."
        )

    hero_range = range_oop if hero.role == "oop" else range_ip
    strategy = _node_strategy(node, spot.hero_cards, hero_range.solver_notation)
    _require_usable_strategy(node, strategy, hero_range)
    if not strategy.combo_frequencies:
        warnings.append(
            "The recorded Hero combo was not present at the mapped decision node; "
            "only range-level frequencies are available."
        )
    if strategy.range_coverage < FULL_STRATEGY_RANGE_COVERAGE:
        warnings.append(
            f"The dumped strategy covered {strategy.range_coverage * 100:.1f}% of the "
            f"submitted {hero_range.role.upper()} range by weight; range frequencies "
            "are averaged over that subset only."
        )
    if mapped_hero_action and mapped_hero_action not in strategy.actions:
        warnings.append(
            f"The mapped Hero branch {mapped_hero_action} is not one of the actions "
            "the strategy node reports."
        )

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
        action_frequencies=strategy.combo_frequencies,
        range_action_frequencies=strategy.range_frequencies,
        exploitability_pct=exploitability_pct,
        runtime_seconds=runtime_seconds,
        range_ip_name=range_ip.profile_name,
        range_oop_name=range_oop.profile_name,
        assumptions=assumptions,
        warnings=warnings,
    )


def map_recorded_action(
    node: Mapping[str, object],
    action: RecordedSolverAction,
    *,
    pot_reference: float,
) -> ActionMapping:
    """Line a recorded action up with a branch of the solve tree.

    The tree offers a handful of discrete sizes, so a recorded bet rarely equals
    one of them and some substitution is unavoidable. What is not tolerable is
    an unbounded one: the substitution is measured against the pot the action
    was made into and refused past ACTION_MAPPING_MAX_POT_FRACTION, because past
    that the branch is a different decision no matter how near it is in chips.

    Actions that carry no size are matched by name or not at all. A check is not
    a small bet and a raise is not a call, so when the tree omits the recorded
    action there is no nearer branch, only a different one.
    """

    available = _available_actions(node)
    if not available:
        return ActionMapping(None, "absent", "the node offers no actions")
    offered = ", ".join(available)
    desired = action.action_type.strip().lower()
    unsized = {"check": "CHECK", "call": "CALL", "fold": "FOLD"}.get(desired)
    if unsized is not None:
        if unsized in available:
            return ActionMapping(unsized, "exact", f"{unsized} is offered at this node")
        return ActionMapping(
            None, "absent", f"{unsized} is not offered at this node (offered: {offered})"
        )
    if desired == "all-in":
        all_in = next(
            (item for item in available if "ALLIN" in re.sub(r"[-_\s]", "", item).upper()),
            None,
        )
        if all_in is not None:
            return ActionMapping(all_in, "exact", "the tree's all-in branch")
        sized = _sized_actions(available, ("BET", "RAISE"))
        if not sized:
            return ActionMapping(
                None,
                "absent",
                f"the tree offers neither an all-in nor a sized bet (offered: {offered})",
            )
        if action.amount is None:
            return ActionMapping(
                None,
                "unusable",
                "the recorded all-in carries no amount, so a sized branch cannot "
                "stand in for it",
            )
        label, amount = max(sized, key=lambda pair: pair[1])
        return _bounded_mapping(
            label, amount, action, pot_reference, "the largest sized branch"
        )
    if desired not in {"bet", "raise"}:
        return ActionMapping(
            None, "absent", f"'{action.action_type}' is not a postflop tree action"
        )
    prefix = "RAISE" if desired == "raise" else "BET"
    sized = _sized_actions(available, (prefix,))
    if not sized:
        return ActionMapping(
            None, "absent", f"the node offers no {prefix} branch (offered: {offered})"
        )
    if action.amount is None:
        return ActionMapping(
            None,
            "unusable",
            f"the recorded {desired} carries no amount, so it cannot be sized "
            "against the tree",
        )
    target = float(action.amount)
    label, amount = min(sized, key=lambda pair: abs(pair[1] - target))
    return _bounded_mapping(label, amount, action, pot_reference, "the nearest branch")


def _require_usable_strategy(
    node: Mapping[str, object], strategy: NodeStrategy, hero_range: ResolvedRange
) -> None:
    """State what a dump must contain before it is retained as study evidence.

    The conditions are checked one at a time so the failure names the one that
    was not met. An operator whose dump reached a chance node, one whose dump
    lists no actions, and one whose dump describes a range they did not submit
    have three different problems, and a single "malformed output" would hide
    which of them they have.
    """

    node_type = node.get("node_type")
    if isinstance(node_type, str) and node_type.strip() and "action" not in node_type.lower():
        raise SolverResultUnusableError(
            f"The mapped solver node is a {node_type}, not a node where a player acts."
        )
    if not strategy.actions:
        raise SolverResultUnusableError(
            "The mapped solver node lists no actions to choose between."
        )
    if not strategy.strategy_combos:
        raise SolverResultUnusableError(
            "The mapped solver node carries no per-combination strategy."
        )
    if not _exact_combo_weights(hero_range.solver_notation):
        raise SolverResultUnusableError(
            f"The submitted {hero_range.role.upper()} range names no exact "
            "combinations to check the dumped strategy against."
        )
    if strategy.range_coverage < MIN_STRATEGY_RANGE_COVERAGE:
        raise SolverResultUnusableError(
            f"The dumped strategy covers {strategy.range_coverage * 100:.1f}% of the "
            f"submitted {hero_range.role.upper()} range by weight, short of the "
            f"{MIN_STRATEGY_RANGE_COVERAGE * 100:.0f}% a usable result must reach; it "
            "describes a different range or the file is incomplete."
        )


def _available_actions(node: Mapping[str, object]) -> list[str]:
    children = node.get("childrens")
    if isinstance(children, dict) and children:
        return [str(item) for item in children]
    raw_actions = node.get("actions")
    return [str(item) for item in raw_actions] if isinstance(raw_actions, list) else []


def _sized_actions(
    available: list[str], prefixes: tuple[str, ...]
) -> list[tuple[str, float]]:
    """Branches of the given kind that carry a size the walk can measure.

    A label with no number in it -- an all-in branch, say -- cannot be compared
    with a recorded amount, and including it would let a nearest-match resolve
    to a size nobody can check.
    """

    sized: list[tuple[str, float]] = []
    for item in available:
        if not item.startswith(prefixes):
            continue
        amount = _action_number(item)
        if amount is not None:
            sized.append((item, amount))
    return sized


def _bounded_mapping(
    label: str,
    tree_amount: float,
    action: RecordedSolverAction,
    pot_reference: float,
    context: str,
) -> ActionMapping:
    recorded = float(action.amount or 0)
    if pot_reference <= 0:
        return ActionMapping(
            None,
            "unusable",
            "the pot the action was made into is unknown, so the substitution "
            "cannot be bounded",
        )
    error = abs(tree_amount - recorded) / pot_reference
    if error <= ACTION_MAPPING_EXACT_POT_FRACTION:
        return ActionMapping(label, "exact", f"{label} matches the recorded size", error)
    if error <= ACTION_MAPPING_MAX_POT_FRACTION:
        return ActionMapping(
            label,
            "approximate",
            f"{context} {label} stands in for {recorded:g} BB into a "
            f"{pot_reference:g} BB pot, {error * 100:.0f}% of that pot away",
            error,
        )
    return ActionMapping(
        None,
        "unusable",
        f"{context} {label} is {error * 100:.0f}% of the {pot_reference:g} BB pot "
        f"away from the recorded {recorded:g} BB, past the "
        f"{ACTION_MAPPING_MAX_POT_FRACTION * 100:.0f}% substitution limit",
        error,
    )


def _pot_reference(action: RecordedSolverAction, spot: SolverSpot) -> float:
    """The pot the substitution is proportional to.

    Each recorded action carries the pot it faced, which is what a size means
    something relative to. The spot's starting pot is the fallback for a line
    recorded before that snapshot existed; it is the same number for the first
    action of the street and an underestimate afterwards, which makes the bound
    stricter rather than looser.
    """

    if action.pot_before is not None and action.pot_before > 0:
        return action.pot_before
    return spot.pot


def _node_strategy(
    node: Mapping[str, object], hero_cards: str, range_notation: str
) -> NodeStrategy:
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
        return NodeStrategy(actions, [], [], 0.0, len(strategy_map))

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
    # Coverage is weighted rather than counted so that a dump missing only
    # combinations the operator had already discounted costs proportionally
    # less than one missing the body of the range.
    submitted_weight = sum(weights.values())
    coverage = total_weight / submitted_weight if submitted_weight else 0.0
    return NodeStrategy(
        actions,
        combo_frequencies,
        range_frequencies,
        min(1.0, coverage),
        len(strategy_map),
    )


def _action_label(action: RecordedSolverAction) -> str:
    return (
        action.action_type
        if action.amount is None
        else f"{action.action_type} {action.amount:g} BB"
    )


def _action_number(action: str) -> float | None:
    match = _NUMBER.search(action)
    return float(match.group(1)) if match else None


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
