from __future__ import annotations

from datetime import UTC, datetime

from poker_tracker.persistence.models import SolverRangeProfile
from poker_tracker.solver.ranges import parse_range

PROFILE_EXPORT_VERSION = 1


def export_range_profiles(profiles: list[SolverRangeProfile]) -> dict[str, object]:
    return {
        "format": "pokertrainer_solver_ranges",
        "version": PROFILE_EXPORT_VERSION,
        "profiles": [
            profile.model_dump(
                mode="json",
                exclude={"id", "created_at", "updated_at", "source"},
            )
            for profile in profiles
        ],
    }


def import_range_profiles(payload: object) -> list[SolverRangeProfile]:
    if not isinstance(payload, dict):
        raise ValueError("Range-profile import must be a JSON object.")
    if payload.get("format") != "pokertrainer_solver_ranges":
        raise ValueError("Unrecognized range-profile import format.")
    if payload.get("version") != PROFILE_EXPORT_VERSION:
        raise ValueError("Unsupported range-profile import version.")
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("Range-profile import must contain a profiles list.")
    if len(raw_profiles) > 500:
        raise ValueError("Range-profile import is limited to 500 profiles.")
    now = datetime.now(UTC)
    profiles: list[SolverRangeProfile] = []
    names: set[str] = set()
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            raise ValueError("Every imported range profile must be an object.")
        profile = SolverRangeProfile.model_validate(
            {
                **raw,
                "source": "user",
                "created_at": now,
                "updated_at": now,
            }
        )
        parse_range(profile.notation)
        normalized_name = profile.name.strip().casefold()
        if normalized_name in names:
            raise ValueError(f"Duplicate imported range name: {profile.name}")
        names.add(normalized_name)
        profiles.append(profile)
    return profiles
