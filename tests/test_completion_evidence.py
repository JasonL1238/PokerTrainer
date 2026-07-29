from __future__ import annotations

import json

import pytest

from poker_tracker.persistence.completion import (
    EVIDENCE_SCHEMA_VERSION,
    BoundaryEvidence,
    CompletionEvidence,
    acknowledge_codes,
    derive_completion_status,
    dump_completion_evidence,
    parse_completion_evidence,
)

_PRIMITIVES = (str, int, float, bool, type(None))


def _complete_evidence(**overrides: object) -> CompletionEvidence:
    base: dict[str, object] = {
        "evidence_version": EVIDENCE_SCHEMA_VERSION,
        "partial_start": False,
        "partial_end": False,
        "terminal_event": "showdown",
        "boundary_confidence": 0.94,
    }
    base.update(overrides)
    return CompletionEvidence(**base)  # type: ignore[arg-type]


def _walk(value: object) -> list[object]:
    if isinstance(value, dict):
        leaves: list[object] = []
        for key, item in value.items():
            assert isinstance(key, str)
            leaves.extend(_walk(item))
        return leaves
    if isinstance(value, list):
        leaves = []
        for item in value:
            leaves.extend(_walk(item))
        return leaves
    return [value]


def test_evidence_round_trips_through_json() -> None:
    evidence = CompletionEvidence(
        evidence_version=EVIDENCE_SCHEMA_VERSION,
        partial_start=False,
        partial_end=True,
        terminal_event="unobserved",
        first_source_timestamp_s=12.5,
        last_source_timestamp_s=48.25,
        preceding_boundary=BoundaryEvidence(
            kind="hand_start", timestamp_s=12.0, frame_ref="frames/a.jpg", confidence=0.9
        ),
        following_boundary=BoundaryEvidence(kind="recording_end", codes=("truncated",)),
        boundary_confidence=0.7,
        source_frames=("frames/a.jpg", "frames/b.jpg"),
        warning_codes=("pot_not_reconciled",),
        rejection_codes=("hero_seat_mismatch",),
        acknowledged_codes=("pot_not_reconciled",),
        layout_profile="6-max",
        layout_supported=True,
        table_size=6,
        pipeline_version="two-model-v7",
        model_versions={"detector": "v7", "cards": "v3"},
        extra={"future_key": {"nested": [1, 2]}},
    )

    restored = parse_completion_evidence(json.dumps(dump_completion_evidence(evidence)))

    assert restored == evidence


def test_dump_contains_only_json_primitives() -> None:
    payload = dump_completion_evidence(
        _complete_evidence(source_frames=("a.jpg",), model_versions={"m": "1"})
    )

    assert all(isinstance(leaf, _PRIMITIVES) for leaf in _walk(payload))
    json.dumps(payload)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_parse_returns_unknown_for_none_and_empty_string(value: object) -> None:
    assert parse_completion_evidence(value) == CompletionEvidence()


@pytest.mark.parametrize("value", [{}, "{}"])
def test_parse_returns_unknown_for_empty_object(value: object) -> None:
    parsed = parse_completion_evidence(value)

    assert parsed == CompletionEvidence()
    assert parsed.is_known is False


def test_parse_returns_unknown_for_corrupt_json() -> None:
    assert parse_completion_evidence('{"evidence_version": ') == CompletionEvidence()
    assert parse_completion_evidence("not json at all") == CompletionEvidence()


@pytest.mark.parametrize("value", ["[1, 2, 3]", "7", '"text"', "null"])
def test_parse_returns_unknown_for_non_object_json(value: str) -> None:
    assert parse_completion_evidence(value) == CompletionEvidence()


@pytest.mark.parametrize("value", [7, 1.5, [1, 2], object(), True])
def test_parse_returns_unknown_for_non_string_non_dict_input(value: object) -> None:
    assert parse_completion_evidence(value) == CompletionEvidence()


def test_parse_tolerates_wrong_field_types() -> None:
    parsed = parse_completion_evidence(
        {
            "evidence_version": "one",
            "partial_start": "yes",
            "partial_end": 1,
            "terminal_event": ["showdown"],
            "first_source_timestamp_s": "12.5",
            "last_source_timestamp_s": {},
            "preceding_boundary": "hand_start",
            "following_boundary": 4,
            "boundary_confidence": "high",
            "source_frames": 7,
            "warning_codes": "one_code",
            "rejection_codes": {"a": 1},
            "acknowledged_codes": [1, 2],
            "layout_profile": 6,
            "layout_supported": "true",
            "table_size": "six",
            "pipeline_version": None,
            "model_versions": ["detector"],
        }
    )

    assert parsed == CompletionEvidence()


def test_parse_preserves_unknown_keys_in_extra() -> None:
    parsed = parse_completion_evidence(
        {"evidence_version": 1, "future_signal": {"depth": 2}, "another": [1]}
    )

    assert parsed.extra == {"future_signal": {"depth": 2}, "another": [1]}
    assert dump_completion_evidence(parsed)["future_signal"] == {"depth": 2}


def test_parse_marks_newer_evidence_version_as_unknown() -> None:
    parsed = parse_completion_evidence(
        {"evidence_version": EVIDENCE_SCHEMA_VERSION + 1, "partial_start": False}
    )

    assert parsed.evidence_version == EVIDENCE_SCHEMA_VERSION + 1
    assert parsed.is_known is False
    assert parsed.partial_start is False


def test_unresolved_codes_excludes_acknowledged_codes() -> None:
    evidence = CompletionEvidence(
        warning_codes=("a", "b"), acknowledged_codes=("a",)
    )

    assert evidence.unresolved_codes == ("b",)


def test_unresolved_codes_includes_rejection_codes() -> None:
    evidence = CompletionEvidence(
        warning_codes=("a",), rejection_codes=("r",), acknowledged_codes=("a",)
    )

    assert evidence.unresolved_codes == ("r",)


@pytest.mark.parametrize(
    ("evidence", "source_type", "expected"),
    [
        (_complete_evidence(), "manual", "not_applicable"),
        (CompletionEvidence(), "cv_import", "uncertain"),
        (_complete_evidence(partial_start=True), "cv_import", "partial"),
        (_complete_evidence(partial_end=True), "cv_import", "partial"),
        (_complete_evidence(partial_start=None), "cv_import", "uncertain"),
        (_complete_evidence(terminal_event=""), "cv_import", "uncertain"),
        (_complete_evidence(terminal_event="unobserved"), "cv_import", "uncertain"),
        (_complete_evidence(boundary_confidence=None), "cv_import", "uncertain"),
        (_complete_evidence(warning_codes=("x",)), "cv_import", "uncertain"),
        (_complete_evidence(), "cv_import", "complete"),
        (_complete_evidence(), "corrected_cv", "complete"),
    ],
)
def test_derive_completion_status_truth_table(
    evidence: CompletionEvidence, source_type: str, expected: str
) -> None:
    assert derive_completion_status(evidence, source_type=source_type) == expected


def test_derive_completion_status_never_promotes_partial() -> None:
    truncated = _complete_evidence(
        partial_end=True, warning_codes=("x",), acknowledged_codes=("x",)
    )

    assert derive_completion_status(truncated, source_type="cv_import") == "partial"


def test_acknowledging_every_code_promotes_uncertain_to_complete() -> None:
    evidence = _complete_evidence(warning_codes=("pot_not_reconciled",))
    assert derive_completion_status(evidence, source_type="cv_import") == "uncertain"

    accepted = acknowledge_codes(evidence, ["pot_not_reconciled", "pot_not_reconciled"])

    assert accepted.acknowledged_codes == ("pot_not_reconciled",)
    assert derive_completion_status(accepted, source_type="cv_import") == "complete"
