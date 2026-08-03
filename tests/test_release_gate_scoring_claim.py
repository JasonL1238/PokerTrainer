"""The gate may not claim it scored timelines when it scored no hand.

``_run_fixture_predictions`` excluded only the case whose artifacts would not
read. A case whose answer key holds no hand to score against -- ``empty_truth``,
or ``zero_scored_hands`` -- came back with ``hands_scored: 0`` and still
incremented ``cases_scored`` and stamped ``SCORE_TIMELINES`` into the
certification summary, three keys above the ``measured: false`` the same report
prints because nothing was measured. The verdict was already a failure; the
false statement was the claim to have performed the act.
"""

from __future__ import annotations

import json
from pathlib import Path

from poker_tracker.release_gate import certification as cert
from poker_tracker.release_gate.runner import (
    _aggregate_metrics,
    _run_fixture_predictions,
)


def _write_case(
    tmp_path: Path, case_id: str, truth: dict, prediction: dict
) -> dict:
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir(exist_ok=True)
    (truth_dir / f"{case_id}.json").write_text(json.dumps(truth), encoding="utf-8")
    (truth_dir / f"{case_id}.prediction.json").write_text(
        json.dumps(prediction), encoding="utf-8"
    )
    return {
        "case_id": case_id,
        "split": "development",
        "counts_toward_release": True,
        "truth_relpath": f"truth/{case_id}.json",
    }


def _manifest(tmp_path: Path, cases: list[dict]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "corpus_id": "probe", "cases": cases}),
        encoding="utf-8",
    )
    return path


def _empty_truth_case(tmp_path: Path) -> dict:
    return _write_case(
        tmp_path,
        "nothing_to_score",
        {"schema_version": 2, "case_id": "nothing_to_score", "hands": []},
        {"hands": []},
    )


def test_a_case_that_scored_no_hand_is_not_counted_as_scored(tmp_path: Path) -> None:
    detail = _run_fixture_predictions(_manifest(tmp_path, [_empty_truth_case(tmp_path)]))

    assert detail["cases_scored"] == 0
    assert detail["cases_failed"] == 1
    assert detail["ok"] is False
    # It is a gate failure, not a setup failure: the artifacts were there and
    # the run reached them.
    assert detail.get("setup_invalid") is not True


def test_score_timelines_is_not_stamped_for_a_case_that_scored_no_hand(
    tmp_path: Path,
) -> None:
    detail = _run_fixture_predictions(_manifest(tmp_path, [_empty_truth_case(tmp_path)]))

    assert detail["executed"] == []
    assert cert.SCORE_TIMELINES not in detail["executed"]


def test_the_summary_does_not_claim_an_act_the_same_report_says_was_unmeasured(
    tmp_path: Path,
) -> None:
    """The two statements that disagreed sit in one report; pin them together."""

    detail = _run_fixture_predictions(_manifest(tmp_path, [_empty_truth_case(tmp_path)]))
    aggregate = _aggregate_metrics(detail)

    assert aggregate["measured"] is False
    assert aggregate["hands_scored"] == 0
    # The claim has to move with the measurement, not against it.
    assert aggregate["cases_scored"] == 0
    assert detail["executed"] == []


def test_an_unreadable_prediction_is_still_excluded(tmp_path: Path) -> None:
    """The narrower rule this replaces must not have been dropped."""

    truth_dir = tmp_path / "truth"
    truth_dir.mkdir(exist_ok=True)
    (truth_dir / "broken.json").write_text("{not json", encoding="utf-8")
    (truth_dir / "broken.prediction.json").write_text("{}", encoding="utf-8")
    case = {
        "case_id": "broken",
        "split": "development",
        "counts_toward_release": True,
        "truth_relpath": "truth/broken.json",
    }

    detail = _run_fixture_predictions(_manifest(tmp_path, [case]))

    assert detail["cases_scored"] == 0
    assert detail["cases_failed"] == 1
    assert detail["executed"] == []
    assert str(detail["cases"][0]["fail_closed"]).startswith("unreadable_artifacts")
