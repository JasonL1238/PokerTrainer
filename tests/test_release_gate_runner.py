"""Phase 3 runner: mode wiring, model identity, aggregates, and exit codes.

The distinction these tests defend is between *setup invalid* (exit 2 — the run
could not be performed) and *gate failed* (exit 1 — it was performed and the
product was wrong). Collapsing the two is how a release with no models installed
reads as a release that scored badly, or worse, the reverse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_tracker.release_gate.evaluate import FACT_SEVERITY
from poker_tracker.release_gate.models import (
    allowlist_violations,
    resolve_models,
    unpinned_cases,
)
from poker_tracker.release_gate.resources import directory_bytes, peak_memory_bytes
from poker_tracker.release_gate.runner import run_release_gate
from poker_tracker.validation.corpus import EXIT_GATE_FAILED, EXIT_SETUP_INVALID
from poker_tracker.validation.hashing import sha256_canonical_json
from poker_tracker.validation.schemas import (
    REQUIRED_COVERAGE_TAGS,
    SCORABLE_HAND_FACTS,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "validation" / "clubwpt_v1.json"


def test_every_schema_fact_has_a_scoring_severity():
    """A fact the schema demands but the evaluator ignores is worse than absent."""
    assert set(FACT_SEVERITY) == set(SCORABLE_HAND_FACTS)


# --- Corpus that is complete enough to reach the mode stage -----------------


HANDS_PER_CASE = 10  # 10 cases x 10 = the 100 completed hands the floor demands
SPLIT_PLAN = (
    ["development"] * 5 + ["validation"] * 2 + ["locked_test"] * 3
)


def _complete_hand(case_id: str, index: int) -> dict:
    start = index * 20.0
    return {
        "hand_id": f"{case_id}-h{index}",
        "t_first": start,
        "t_last": start + 10.0,
        "partial_start": False,
        "partial_end": False,
        "completion_class": "complete",
        "terminal_event": "showdown",
        "hero_cards": ["As", "Kd"],
        "final_board": ["2c", "7d", "9h", "Ts", "Jc"],
        "dealer_seat": 1,
        "winner_seat": 0,
        "result": "Hero wins",
        "final_pot": 0.0,
        "hero_net": 0.0,
        "dealt_in_seats": [0, 1],
        "starting_stacks": {"0": 100.0, "1": 100.0},
        "actions": [],
        "actions_complete": True,
        "unobservable": [],
        "evidence": {
            "hero_cards_t": start,
            "final_board_t": start + 8.0,
            "final_pot_t": start + 10.0,
        },
    }


def _partial_hand(case_id: str) -> dict:
    """The recording's trailing hand, cut off by the end of the video."""
    start = HANDS_PER_CASE * 20.0
    return {
        "hand_id": f"{case_id}-partial",
        "t_first": start,
        "t_last": start + 10.0,
        "partial_start": False,
        "partial_end": True,
        "completion_class": "partial",
        "terminal_event": "recording_end",
        "hero_cards": ["7h", "2d"],
        "final_board": [],
        "dealer_seat": 0,
        "dealt_in_seats": [0, 1],
        "starting_stacks": {"0": 100.0, "1": 100.0},
        "actions": [],
        "actions_complete": False,
        "hero_net": 0.0,
        "unobservable": ["winner_seat", "result", "final_pot"],
        "evidence": {"partial_end_t": start + 10.0},
    }


def _truth_for(case_id: str) -> dict:
    return {
        "schema_version": 2,
        "case_id": case_id,
        "annotation": {"passes": 2, "adjudicated": True},
        "hands": [
            *(_complete_hand(case_id, i) for i in range(HANDS_PER_CASE)),
            _partial_hand(case_id),
        ],
    }


def _prediction_for(truth: dict, *, matching: bool) -> dict:
    hands = []
    for number, hand in enumerate(truth["hands"], start=1):
        complete = hand["completion_class"] == "complete"
        hands.append(
            {
                "hand_number": number,
                "t_start": hand["t_first"],
                "t_end": hand["t_last"],
                "complete": complete,
                "partial_end": hand.get("partial_end", False),
                "terminal_event": hand["terminal_event"],
                "hero": (
                    hand["hero_cards"]
                    if matching or not complete
                    else ["Qs", "Jd"]
                ),
                "board": hand["final_board"],
                "dealer_seat": hand["dealer_seat"],
                "winner_seat": hand.get("winner_seat"),
                "result": hand.get("result"),
                "pot": hand.get("final_pot"),
                "hero_bb_won": hand.get("hero_net"),
                "actions": [],
            }
        )
    return {"hands": hands}


@pytest.fixture
def release_corpus(tmp_path: Path):
    """A synthetic corpus that genuinely clears the Phase 2 floors.

    The floors are code-enforced and a manifest may not lower them, so reaching
    the mode stage at all requires ten hash-identified sessions, a hundred
    completed hands, ten partials, and every required coverage tag. Building
    that honestly is the only way to test the stages behind the corpus gate.
    """

    def build(*, matching_prediction: bool = True, pin_models: bool = False) -> Path:
        truth_dir = tmp_path / "truth"
        truth_dir.mkdir(exist_ok=True)
        tags = list(REQUIRED_COVERAGE_TAGS)
        cases = []
        for index, split in enumerate(SPLIT_PLAN):
            case_id = f"case_{index:02d}"
            truth = _truth_for(case_id)
            (truth_dir / f"{case_id}.json").write_text(
                json.dumps(truth), encoding="utf-8"
            )
            (truth_dir / f"{case_id}.prediction.json").write_text(
                json.dumps(_prediction_for(truth, matching=matching_prediction)),
                encoding="utf-8",
            )
            # Spread the required tags so the union covers every one of them.
            case_tags = tags[index :: len(SPLIT_PLAN)] or [tags[0]]
            cases.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "used_for_tuning": False,
                    "layout_profile": "clubwpt",
                    "runtime_class": "video",
                    "counts_toward_release": True,
                    "coverage_tags": case_tags,
                    "recording": {
                        "logical_name": f"{case_id}.mp4",
                        "sha256": None,
                        "bytes": None,
                        "duration_s": 10.0,
                        "fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                    },
                    "truth_relpath": f"truth/{case_id}.json",
                    "truth_sha256": sha256_canonical_json(truth),
                    "model_allowlist": (
                        {"region_detector": {"sha256": "deadbeef"}}
                        if pin_models
                        else {}
                    ),
                }
            )
        manifest = {
            "schema_version": 1,
            "corpus_id": "synthetic_release",
            "cases": cases,
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    return build


# --- Mode wiring ------------------------------------------------------------


def test_full_mode_without_a_vault_is_setup_invalid_not_a_bad_score(
    release_corpus, tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("POKER_VALIDATION_ROOT", raising=False)
    result = run_release_gate(
        manifest_path=release_corpus(),
        mode="full",
        report_dir=tmp_path / "reports",
    )
    assert result.exit_code == EXIT_SETUP_INVALID
    stage = next(s for s in result.report["stages"] if s["name"] == "full_video")
    assert stage["detail"]["setup_invalid"] is True


def test_container_mode_without_docker_is_setup_invalid(
    release_corpus, tmp_path: Path, monkeypatch
):
    # An empty PATH guarantees the docker probe fails the same way everywhere.
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    result = run_release_gate(
        manifest_path=release_corpus(),
        mode="container",
        report_dir=tmp_path / "reports",
        container_image="does-not-exist:latest",
    )
    assert result.exit_code == EXIT_SETUP_INVALID
    stage = next(s for s in result.report["stages"] if s["name"] == "container")
    assert stage["detail"]["setup_invalid"] is True


def test_fixture_mode_scores_a_matching_prediction(release_corpus, tmp_path: Path):
    result = run_release_gate(
        manifest_path=release_corpus(matching_prediction=True),
        mode="fixture",
        report_dir=tmp_path / "reports",
    )
    stage = next(s for s in result.report["stages"] if s["name"] == "fixture_eval")
    assert stage["ok"] is True
    assert result.report["aggregate"]["hands_scored"] == 110
    assert result.report["aggregate"]["completion"]["precision"] == 1.0
    assert result.report["aggregate"]["completion"]["recall"] == 1.0


def test_wrong_prediction_is_a_gate_failure_not_a_setup_failure(
    release_corpus, tmp_path: Path
):
    result = run_release_gate(
        manifest_path=release_corpus(matching_prediction=False),
        mode="fixture",
        report_dir=tmp_path / "reports",
    )
    assert result.exit_code == EXIT_GATE_FAILED
    assert result.report["aggregate"]["critical_errors"] >= 1


# --- Model identity ---------------------------------------------------------


def test_allowlist_mismatch_fails_the_gate(release_corpus, tmp_path: Path):
    """Running weights the case did not pin invalidates its verdict."""
    result = run_release_gate(
        manifest_path=release_corpus(pin_models=True),
        mode="fixture",
        report_dir=tmp_path / "reports",
    )
    stage = next(s for s in result.report["stages"] if s["name"] == "models")
    assert stage["ok"] is False
    assert "case_01" in stage["detail"]["allowlist_violations"]
    assert result.exit_code == EXIT_SETUP_INVALID


def test_unpinned_case_is_reported_in_fixture_mode(release_corpus, tmp_path: Path):
    result = run_release_gate(
        manifest_path=release_corpus(pin_models=False),
        mode="fixture",
        report_dir=tmp_path / "reports",
    )
    stage = next(s for s in result.report["stages"] if s["name"] == "models")
    assert stage["detail"]["unpinned_cases"] == [f"case_{i:02d}" for i in range(10)]


def test_empty_allowlist_is_unpinned_not_agreement():
    """``{}`` means nobody pinned anything; it must not read as a passing check."""
    case = {"case_id": "c", "model_allowlist": {}}
    assert allowlist_violations(case, resolve_models(REPO)) == []
    assert unpinned_cases([case]) == ["c"]


def test_absent_weights_are_reported_as_absent(tmp_path: Path):
    resolved = resolve_models(tmp_path)
    assert all(entry["present"] is False for entry in resolved.values())
    assert all(entry["sha256"] is None for entry in resolved.values())


# --- Report contents --------------------------------------------------------


def test_report_records_resources_environment_and_models(
    release_corpus, tmp_path: Path
):
    result = run_release_gate(
        manifest_path=release_corpus(),
        mode="fixture",
        report_dir=tmp_path / "reports",
    )
    report = result.report
    assert report["resources"]["peak_memory_bytes"] > 0
    assert "dependencies" in report["environment"]
    assert "cpu_count" in report["environment"]
    assert set(report["models"]) == {"region_detector", "card_classifier"}
    assert report["manifest_sha256"]

    # The durable artifact is deterministic apart from the timing block.
    written = json.loads((tmp_path / "reports" / "release_gate_report.json").read_text())
    assert "timing" in written
    assert all("elapsed_s" not in stage for stage in written["stages"])


def test_peak_memory_is_plausible_not_a_unit_error():
    """macOS reports bytes and Linux kilobytes; a 1024x slip is silent."""
    peak = peak_memory_bytes()
    assert 4 * 1024 * 1024 < peak < 64 * 1024 * 1024 * 1024


def test_directory_bytes_does_not_follow_symlinks(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "payload.bin").write_bytes(b"x" * 4096)
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "link").symlink_to(real / "payload.bin")
    # Counting the link's own size, not its 4 KiB target.
    assert directory_bytes(linked) < 4096
