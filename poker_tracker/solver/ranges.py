from __future__ import annotations

import re
from dataclasses import dataclass

import eval7
from eval7.rangestring import RangeStringError

from poker_tracker.math.preflop_ranges import resolve_preflop_range
from poker_tracker.persistence.models import SolverRangeProfile
from poker_tracker.solver.models import ResolvedRange, SolverPlayer, SolverSpot
from poker_tracker.solver.texassolver import TEXASSOLVER_MIN_WEIGHT

TOTAL_COMBOS = 1326
# TexasSolver reads six decimals of weight out of the command file, so that is
# the precision a combo is actually submitted at whatever the operator wrote.
SOLVER_WEIGHT_DECIMALS = 6
# At or above this the combo is written bare instead of with an explicit weight:
# the remaining gap to 1.0 is finer than the emitted precision can express.
_FULL_WEIGHT = 0.999999
_COLON_WEIGHT = re.compile(
    r"^(?P<token>[^:]+):(?P<weight>(?:0(?:\.\d+)?|1(?:\.0+)?))$"
)

_CALL_RANGES = {
    "BB": "22+,A2s+,K2s+,Q5s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s,A5o+,K9o+,Q9o+,J9o+,T9o",
    "SB": "22-TT,A2s+,K8s+,Q9s+,J9s+,T8s+,98s,87s,ATo+,KJo+,QJo",
    "default": "22-TT,A8s+,KTs+,QTs+,JTs,T9s,98s,87s,AJo+,KQo",
}
_THREE_BET_RANGE = "99+,AJs+,A5s-A4s,KQs,AQo+"
_CALL_THREE_BET_RANGE = "88-QQ,ATs+,KQs,AQo+"
_OPENING_POSITION_MAP = {
    5: {"UTG": "HJ", "UTG+1": "HJ", "LJ": "HJ"},
    6: {"UTG": "HJ", "UTG+1": "HJ", "LJ": "HJ"},
    7: {"UTG": "LJ", "UTG+1": "HJ"},
    8: {"UTG": "UTG+1"},
}


@dataclass(frozen=True)
class ParsedRange:
    notation: str
    solver_notation: str
    combo_count: int
    range_percent: float
    combo_keys: frozenset[str]


def _submitted_weight(weight: float) -> float:
    """The weight the solver actually receives for a combo written as ``weight``.

    This is a numeric step, not a formatting one, and that distinction is the
    whole point. The unsupported-weight guard used to be spelled
    ``float(f"{weight:.6f}")``, which made it a test of the printed form: a
    weight under 5e-7 printed as ``0.000000``, compared equal to zero, and so
    read as "this combo carries no weight" rather than as "the solver would drop
    this combo". It went on to be emitted as an ``AdAc:0`` token -- the silent
    drop the guard exists to prevent. Quantizing here and validating the result
    keeps the guard and the emitted token talking about the same number, and
    survives any later change to how that number is spelled.
    """

    return round(weight, SOLVER_WEIGHT_DECIMALS)


def _unsupported_weight_error() -> ValueError:
    return ValueError(
        "TexasSolver silently drops weights at or below "
        f"{TEXASSOLVER_MIN_WEIGHT:g}; increase or remove those weights."
    )


def _weight_token(combo: str, weight: float) -> str:
    """Render one already-validated submitted weight for the command file."""

    if weight >= _FULL_WEIGHT:
        return combo
    return f"{combo}:{weight:.{SOLVER_WEIGHT_DECIMALS}f}".rstrip("0").rstrip(".")


def normalize_weighted_notation(notation: str) -> str:
    """Accept TexasSolver ``HAND:weight`` tokens and return eval7 notation."""

    converted: list[str] = []
    for raw_token in notation.split(","):
        token = raw_token.strip()
        if not token:
            continue
        match = _COLON_WEIGHT.match(token)
        if match:
            weight = float(match.group("weight"))
            # Judged on the number the operator wrote, before it becomes the
            # percent string eval7 takes. A weight small enough to render as
            # ``0%`` reaches eval7 as a token it rejects as ungrammatical, which
            # blames the operator's spelling for a value this function threw
            # away; one merely small reaches eval7 intact and is dropped later
            # by the solver instead. Both are the same fact, so both are said
            # here in the same words.
            if _submitted_weight(weight) <= TEXASSOLVER_MIN_WEIGHT:
                raise _unsupported_weight_error()
            percent_text = f"{weight * 100:.6f}".rstrip("0").rstrip(".")
            converted.append(f"{percent_text}%({match.group('token').strip()})")
        else:
            converted.append(token)
    return ",".join(converted)


def parse_range(notation: str, *, dead_cards: str = "") -> ParsedRange:
    normalized = normalize_weighted_notation(notation)
    if not normalized:
        raise ValueError("Range notation is required.")
    try:
        hands = eval7.HandRange(normalized).hands
    except (RangeStringError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid range notation: {notation}") from exc

    dead = {str(eval7.Card(card)) for card in dead_cards.split() if card}
    combos: dict[str, tuple[str, float]] = {}
    for (first, second), weight in hands:
        first_text, second_text = str(first), str(second)
        if first_text in dead or second_text in dead or weight <= 0:
            continue
        combo = f"{first_text}{second_text}"
        key = _combo_key(first_text, second_text)
        prior = combos.get(key)
        if prior is None or float(weight) > prior[1]:
            combos[key] = (combo, float(weight))
    # Quantized once, here, so that everything below -- the guard, the emitted
    # tokens and the reported width -- is talking about the weight the solver
    # will receive rather than three separate readings of it.
    submitted = [
        (combo, _submitted_weight(weight)) for combo, weight in combos.values()
    ]
    unsupported = [
        combo for combo, weight in submitted if weight <= TEXASSOLVER_MIN_WEIGHT
    ]
    if unsupported:
        raise _unsupported_weight_error()
    overweight = [combo for combo, weight in combos.values() if weight > 1.0000001]
    if overweight:
        raise ValueError("Range weights cannot exceed 1.0 (100%).")
    solver_tokens = [_weight_token(combo, weight) for combo, weight in submitted]
    weighted_combo_total = sum(weight for _, weight in submitted)
    if not solver_tokens:
        raise ValueError("Range has no available combinations after removing visible cards.")
    return ParsedRange(
        notation=normalized,
        solver_notation=",".join(solver_tokens),
        combo_count=len(solver_tokens),
        range_percent=min(1.0, weighted_combo_total / TOTAL_COMBOS),
        combo_keys=frozenset(combos),
    )


def builtin_range_profiles(table_sizes: range = range(5, 9)) -> list[SolverRangeProfile]:
    profiles: list[SolverRangeProfile] = []
    positions = ("UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB")
    for table_size in table_sizes:
        for position in positions:
            chart_position = _OPENING_POSITION_MAP.get(table_size, {}).get(
                position, position
            )
            opening = resolve_preflop_range(chart_position, "rfi")
            if opening is not None:
                profiles.append(
                    SolverRangeProfile(
                        name=f"{table_size}-max {position} open · estimated",
                        notation=opening.notation,
                        table_size=table_size,
                        position=position,
                        scenario="rfi",
                        pot_type="single_raised",
                        stack_bb=100,
                        description=(
                            "Estimated post-session starting range derived from the built-in "
                            f"100 BB {chart_position} study chart; not solved preflop GTO."
                        ),
                        source="builtin",
                    )
                )
            profiles.append(
                SolverRangeProfile(
                    name=f"{table_size}-max {position} call open · estimated",
                    notation=_CALL_RANGES.get(position, _CALL_RANGES["default"]),
                    table_size=table_size,
                    position=position,
                    scenario="call_open",
                    pot_type="single_raised",
                    stack_bb=100,
                    description="Estimated call-versus-open study range; not solved preflop GTO.",
                    source="builtin",
                )
            )
            profiles.extend(
                [
                    SolverRangeProfile(
                        name=f"{table_size}-max {position} 3-bet · estimated",
                        notation=_THREE_BET_RANGE,
                        table_size=table_size,
                        position=position,
                        scenario="three_bet",
                        pot_type="three_bet",
                        stack_bb=100,
                        description="Estimated three-bet study range; not solved preflop GTO.",
                        source="builtin",
                    ),
                    SolverRangeProfile(
                        name=f"{table_size}-max {position} call 3-bet · estimated",
                        notation=_CALL_THREE_BET_RANGE,
                        table_size=table_size,
                        position=position,
                        scenario="call_three_bet",
                        pot_type="three_bet",
                        stack_bb=100,
                        description="Estimated call-versus-three-bet range; not solved preflop GTO.",
                        source="builtin",
                    ),
                ]
            )
    return profiles


BUILTIN_RANGE_PROFILES = builtin_range_profiles()


# The two sides of the attribution a run makes about a range. "default" and
# "builtin" both mean the product's own study estimate; "user" and "custom" mean
# the operator supplied it. start_solver_job writes a different assumption onto
# the run for each side, so which side a resolution falls on is a claim about
# whose range was solved.
_ESTIMATE_SOURCES = frozenset({"default", "builtin"})


def _check_profile_source(profile: SolverRangeProfile, source: str) -> None:
    """Refuse to present a profile as something its own record contradicts.

    A run built from a built-in profile carries "estimated study input, not
    solved preflop GTO"; one built from a saved profile carries "explicitly
    supplied by the user". Which of the two a profile is, the profile already
    records, so a caller asking for the other one is asking the run either to
    attribute the product's estimate to the operator or to file the operator's
    own range as a study default. Both are wrong attributions presented as fact,
    and neither is visible once the run is written.
    """

    if (profile.source in _ESTIMATE_SOURCES) == (source in _ESTIMATE_SOURCES):
        return
    raise ValueError(
        f"Range profile '{profile.name}' is recorded as {profile.source}, so it "
        f"cannot be resolved as {source}."
    )


def default_scenario(spot: SolverSpot, player: SolverPlayer) -> str:
    aggressor = _last_aggressor_key(spot)
    if spot.pot_type == "single_raised":
        return "rfi" if player.player_key == aggressor else "call_open"
    return "three_bet" if player.player_key == aggressor else "call_three_bet"


def resolve_profile(
    spot: SolverSpot,
    player: SolverPlayer,
    profiles: list[SolverRangeProfile],
    *,
    source: str,
    requested_name: str = "",
) -> ResolvedRange:
    if not player.position.strip():
        raise ValueError(
            f"Default range selection for {player.player_name} requires a saved "
            "position. Record the position or choose Premade/Custom explicitly."
        )
    scenario = default_scenario(spot, player)
    compatible = [
        profile
        for profile in profiles
        if profile.pot_type == spot.pot_type and profile.scenario == scenario
    ]
    if not compatible:
        raise ValueError(
            f"No compatible {spot.pot_type.replace('_', ' ')} range for "
            f"{player.player_name} ({scenario})."
        )

    def score(profile: SolverRangeProfile) -> tuple[int, int, float, str]:
        position_penalty = 0 if profile.position == player.position else 1
        table_penalty = abs((profile.table_size or spot.table_size) - spot.table_size)
        stack_penalty = abs((profile.stack_bb or 100) - spot.effective_stack)
        return position_penalty, table_penalty, stack_penalty, profile.name

    chosen = min(compatible, key=score)
    _check_profile_source(chosen, source)
    mismatches: list[str] = []
    if chosen.position and chosen.position != player.position:
        mismatches.append(
            f"Position substituted: requested {player.position or 'unknown'}, "
            f"used {chosen.position}."
        )
    if chosen.table_size is not None and chosen.table_size != spot.table_size:
        mismatches.append(
            f"Table size substituted: requested {spot.table_size}-max, "
            f"used {chosen.table_size}-max."
        )
    if chosen.stack_bb is not None and abs(chosen.stack_bb - spot.effective_stack) > 0.01:
        mismatches.append(
            f"Range stack substituted: spot {spot.effective_stack:g} BB, "
            f"profile {chosen.stack_bb:g} BB."
        )
    parsed = parse_range(chosen.notation, dead_cards=_dead_cards(spot, player))
    if player.is_hero:
        hero_cards = spot.hero_cards.split()
        if len(hero_cards) != 2 or _combo_key(*hero_cards) not in parse_range(
            chosen.notation, dead_cards=spot.board
        ).combo_keys:
            raise ValueError(
                f"{player.player_name}'s selected range does not contain the recorded Hero combo."
            )
    return ResolvedRange(
        player_key=player.player_key,
        player_name=player.player_name,
        position=player.position,
        role=player.role,
        source=source,
        profile_name=chosen.name,
        notation=parsed.notation,
        solver_notation=parsed.solver_notation,
        combo_count=parsed.combo_count,
        range_percent=parsed.range_percent,
        requested_profile=requested_name,
        mismatches=mismatches,
    )


def resolve_custom_range(
    spot: SolverSpot,
    player: SolverPlayer,
    notation: str,
    *,
    name: str = "Custom range",
) -> ResolvedRange:
    parsed_without_hero = parse_range(notation, dead_cards=spot.board)
    if player.is_hero:
        hero_cards = spot.hero_cards.split()
        if len(hero_cards) != 2 or _combo_key(*hero_cards) not in parsed_without_hero.combo_keys:
            raise ValueError(
                f"{player.player_name}'s custom range does not contain the recorded Hero combo."
            )
    parsed = parse_range(notation, dead_cards=_dead_cards(spot, player))
    return ResolvedRange(
        player_key=player.player_key,
        player_name=player.player_name,
        position=player.position,
        role=player.role,
        source="custom",
        profile_name=name.strip() or "Custom range",
        notation=parsed.notation,
        solver_notation=parsed.solver_notation,
        combo_count=parsed.combo_count,
        range_percent=parsed.range_percent,
    )


def resolve_selected_profile(
    spot: SolverSpot,
    player: SolverPlayer,
    profile: SolverRangeProfile,
    *,
    source: str,
) -> ResolvedRange:
    scenario = default_scenario(spot, player)
    _check_profile_source(profile, source)
    if profile.pot_type and profile.pot_type != spot.pot_type:
        raise ValueError(
            f"Premade range '{profile.name}' is for "
            f"{profile.pot_type.replace('_', ' ')}, not "
            f"{spot.pot_type.replace('_', ' ')}."
        )
    if profile.scenario and profile.scenario != scenario:
        raise ValueError(
            f"Premade range '{profile.name}' is for {profile.scenario}, "
            f"not {scenario}."
        )
    parsed_without_hero = parse_range(profile.notation, dead_cards=spot.board)
    if player.is_hero:
        hero_cards = spot.hero_cards.split()
        if len(hero_cards) != 2 or _combo_key(*hero_cards) not in parsed_without_hero.combo_keys:
            raise ValueError(
                f"{player.player_name}'s selected range does not contain the recorded Hero combo."
            )
    parsed = parse_range(profile.notation, dead_cards=_dead_cards(spot, player))
    mismatches: list[str] = []
    if profile.position and profile.position != player.position:
        mismatches.append(
            f"Explicit profile is for {profile.position}, while this player is "
            f"{player.position or 'position unknown'}."
        )
    return ResolvedRange(
        player_key=player.player_key,
        player_name=player.player_name,
        position=player.position,
        role=player.role,
        source=source,
        profile_name=profile.name,
        notation=parsed.notation,
        solver_notation=parsed.solver_notation,
        combo_count=parsed.combo_count,
        range_percent=parsed.range_percent,
        requested_profile=profile.name,
        mismatches=mismatches,
    )


def _last_aggressor_key(spot: SolverSpot) -> str:
    return spot.preflop_aggressor_key


def _combo_key(first: str, second: str) -> str:
    return "|".join(sorted((first, second)))


def _dead_cards(spot: SolverSpot, player: SolverPlayer) -> str:
    if player.is_hero:
        return spot.board
    return f"{spot.board} {spot.hero_cards}".strip()
