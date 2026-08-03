"""Phase 2 corpus / answer-key validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from poker_tracker.validation.corpus import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_SETUP_INVALID,
    check_corpus,
)
from poker_tracker.validation.hashing import sha256_canonical_json
from poker_tracker.validation.truth_validate import validate_answer_key

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "validation" / "clubwpt_v1.json"
TRUTH = REPO / "validation" / "truth" / "fixture_synthetic_hu_01.json"


def _write_corpus(tmp_path: Path, *, truth: dict, case_extra: dict | None = None) -> Path:
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    case = {
        "case_id": truth["case_id"],
        "split": "development",
        "used_for_tuning": False,
        "layout_profile": "synthetic_fixture",
        "runtime_class": "full_video",
        "counts_toward_release": True,
        "coverage_tags": ["showdown"],
        "recording": {"logical_name": "x.mp4", "sha256": None},
        "truth_relpath": "truth.json",
        "truth_sha256": sha256_canonical_json(truth),
        "model_allowlist": {},
    }
    if case_extra:
        case.update(case_extra)
    manifest = {
        "schema_version": 1,
        "corpus_id": "test_corpus",
        "cases": [case],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_multi_case_corpus(tmp_path: Path, specs: list[dict]) -> Path:
    """Write a manifest with one frozen answer key per spec.

    Each spec overrides a valid default case, so the only thing a test's manifest
    can fail on is the property that test is about.
    """
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    cases = []
    for spec in specs:
        case_id = spec["case_id"]
        document = dict(truth)
        document["case_id"] = case_id
        (tmp_path / f"{case_id}.json").write_text(json.dumps(document), encoding="utf-8")
        case = {
            "case_id": case_id,
            "split": "development",
            "used_for_tuning": False,
            "layout_profile": "clubwpt",
            "runtime_class": "full_video",
            "counts_toward_release": True,
            "coverage_tags": ["showdown"],
            "recording": {"logical_name": f"{case_id}.mp4", "sha256": None},
            "truth_relpath": f"{case_id}.json",
            "truth_sha256": sha256_canonical_json(document),
            "model_allowlist": {},
        }
        case.update({key: value for key, value in spec.items() if key != "case_id"})
        cases.append(case)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "corpus_id": "multi_case", "cases": cases}),
        encoding="utf-8",
    )
    return manifest_path


def test_committed_synthetic_truth_validates():
    document = json.loads(TRUTH.read_text(encoding="utf-8"))
    result = validate_answer_key(document)
    assert result.ok, [f"{i.path}: {i.message}" for i in result.issues]
    assert result.completed_hands == 1
    assert result.partial_hands == 1


def test_committed_manifest_truth_hash_matches():
    document = json.loads(TRUTH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["cases"][0]["truth_sha256"] == sha256_canonical_json(document)


def test_release_minimums_fail_closed_on_checked_in_manifest():
    result = check_corpus(MANIFEST, require_release_minimums=True)
    assert result.ok is False
    assert result.exit_code == EXIT_SETUP_INVALID
    assert any(i.path.startswith("minimums.") for i in result.issues)
    # Fixture cases do not count toward release floors.
    assert result.stats["sessions"] == 0
    assert result.stats["completed_hands"] == 0


def test_skip_release_minimums_accepts_schema_plumbing_corpus():
    result = check_corpus(MANIFEST, require_release_minimums=False)
    assert result.ok is True
    assert result.exit_code == EXIT_OK
    assert result.stats["sessions"] == 0
    assert result.stats["listed_cases"] == 1


def test_manifest_cannot_lower_phase2_floors(tmp_path: Path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    manifest_path = _write_corpus(tmp_path, truth=truth)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["minimums"] = {"sessions": 1, "completed_hands": 1, "partial_hands": 1,
                            "development_sessions": 1, "validation_sessions": 0,
                            "locked_test_sessions": 0}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = check_corpus(manifest_path, require_release_minimums=True)
    assert result.ok is False
    assert result.exit_code == EXIT_SETUP_INVALID
    assert any("may not lower Phase 2 floor" in i.message for i in result.issues)


def test_manifest_cannot_drop_required_coverage_tags(tmp_path: Path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    manifest_path = _write_corpus(tmp_path, truth=truth)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["required_coverage_tags"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert any("may not drop Phase 2 coverage tags" in i.message for i in result.issues)


def test_truth_hash_mismatch_is_a_gate_failure(tmp_path: Path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    manifest_path = _write_corpus(tmp_path, truth=truth)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["truth_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert result.exit_code == EXIT_GATE_FAILED
    assert any("hash mismatch" in i.message for i in result.issues)


def test_duplicate_cards_rejected():
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    truth["hands"][0]["final_board"] = ["As", "2c", "7d"]
    result = validate_answer_key(truth)
    assert result.ok is False
    assert any("duplicate card" in i.message for i in result.issues)


def test_act_after_fold_rejected():
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    truth["hands"][0]["actions"] = [
        {
            "street": "preflop",
            "order": 1,
            "seat": 0,
            "action": "fold",
            "amount": None,
            "certain": True,
            "observable": True,
        },
        {
            "street": "preflop",
            "order": 2,
            "seat": 0,
            "action": "raise",
            "amount": 3.0,
            "certain": True,
            "observable": True,
        },
    ]
    result = validate_answer_key(truth)
    assert result.ok is False
    assert any("acts after folding" in i.message for i in result.issues)


def test_complete_requires_actions_complete():
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    truth["hands"][0]["actions_complete"] = False
    result = validate_answer_key(truth)
    assert result.ok is False
    assert result.completed_hands == 0
    assert any("actions_complete=true" in i.message for i in result.issues)


def test_inflated_final_pot_rejected_for_incremental_line():
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    truth["hands"][0]["final_pot"] = 500.0
    result = validate_answer_key(truth)
    assert result.ok is False
    assert any("exceeds certain incremental" in i.message for i in result.issues)


def test_locked_test_used_for_tuning_fails(tmp_path: Path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    manifest_path = _write_corpus(
        tmp_path,
        truth=truth,
        case_extra={"split": "locked_test", "used_for_tuning": True},
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert any("used_for_tuning to false" in i.message for i in result.issues)


def test_locked_test_truthy_non_bool_tuning_flag_fails(tmp_path: Path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    manifest_path = _write_corpus(
        tmp_path,
        truth=truth,
        case_extra={"split": "locked_test", "used_for_tuning": 1},
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert any("used_for_tuning to false" in i.message for i in result.issues)


def test_truth_path_escape_rejected(tmp_path: Path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    manifest_path = _write_corpus(tmp_path, truth=truth)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["truth_relpath"] = "../outside.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert result.exit_code == EXIT_SETUP_INVALID
    assert any("escapes" in i.message for i in result.issues)


def test_recording_path_escape_rejected(tmp_path: Path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    manifest_path = _write_corpus(
        tmp_path,
        truth=truth,
        case_extra={"recording": {"logical_name": "../secret.mp4", "sha256": None}},
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert any("no absolute paths or '..'" in i.message for i in result.issues)


def test_recording_path_alias_cannot_bypass_split_integrity(tmp_path: Path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    truth_a = dict(truth)
    truth_a["case_id"] = "dev_case"
    truth_b = dict(truth)
    truth_b["case_id"] = "lock_case"
    (tmp_path / "truth_a.json").write_text(json.dumps(truth_a), encoding="utf-8")
    (tmp_path / "truth_b.json").write_text(json.dumps(truth_b), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "corpus_id": "alias_atk",
        "cases": [
            {
                "case_id": "dev_case",
                "split": "development",
                "used_for_tuning": False,
                "layout_profile": "x",
                "runtime_class": "full_video",
                "counts_toward_release": True,
                "coverage_tags": ["showdown"],
                "recording": {"logical_name": "x.mp4", "sha256": None},
                "truth_relpath": "truth_a.json",
                "truth_sha256": sha256_canonical_json(truth_a),
                "model_allowlist": {},
            },
            {
                "case_id": "lock_case",
                "split": "locked_test",
                "used_for_tuning": False,
                "layout_profile": "x",
                "runtime_class": "full_video",
                "counts_toward_release": True,
                "coverage_tags": ["showdown"],
                "recording": {"logical_name": "./x.mp4", "sha256": None},
                "truth_relpath": "truth_b.json",
                "truth_sha256": sha256_canonical_json(truth_b),
                "model_allowlist": {},
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert any(i.path == "split_integrity" for i in result.issues)


def test_require_recordings_rejects_null_sha(tmp_path: Path, monkeypatch):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "x.mp4").write_bytes(b"video")
    monkeypatch.setenv("POKER_VALIDATION_ROOT", str(vault))
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest_path = _write_corpus(corpus_dir, truth=truth)
    result = check_corpus(
        manifest_path,
        require_release_minimums=False,
        require_recording_files=True,
    )
    assert result.ok is False
    assert any("recording.sha256 is required" in i.message for i in result.issues)


def test_fixture_cannot_opt_into_release_counts(tmp_path: Path):
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    manifest_path = _write_corpus(
        tmp_path,
        truth=truth,
        case_extra={
            "runtime_class": "fixture",
            "counts_toward_release": True,
            "coverage_tags": ["showdown"],
        },
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert any("cannot count toward release" in i.message for i in result.issues)
    assert result.stats["sessions"] == 0


# --- The locked-test seal is content, not path ------------------------------


def test_locked_recording_copied_into_development_breaks_the_seal(
    tmp_path: Path, monkeypatch
):
    """A byte-identical copy under another path is the same recording.

    Comparing only logical names let the sealed split be tuned against by
    ``cp locked/sealed.mov dev/copy.mov``, and the report said the seal held.
    """
    vault = tmp_path / "vault"
    (vault / "locked").mkdir(parents=True)
    (vault / "dev").mkdir(parents=True)
    payload = b"sealed recording bytes"
    (vault / "locked" / "sealed.mov").write_bytes(payload)
    (vault / "dev" / "copy.mov").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setenv("POKER_VALIDATION_ROOT", str(vault))
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest_path = _write_multi_case_corpus(
        corpus_dir,
        [
            {
                "case_id": "lock_case",
                "split": "locked_test",
                "recording": {"logical_name": "locked/sealed.mov", "sha256": digest},
            },
            {
                "case_id": "dev_case",
                "used_for_tuning": True,
                "recording": {"logical_name": "dev/copy.mov", "sha256": digest},
            },
        ],
    )
    result = check_corpus(
        manifest_path,
        require_release_minimums=False,
        require_recording_files=True,
    )
    assert result.ok is False
    assert any(
        i.path == "split_integrity" and "sha256" in i.message for i in result.issues
    )
    assert result.stats["locked_used_for_tuning"] is True
    assert result.stats["locked_seal"]["intact"] is False


def test_case_only_rename_cannot_alias_a_locked_recording(tmp_path: Path):
    """``Locked/Sealed.mov`` is the same file as ``locked/sealed.mov`` on macOS."""
    manifest_path = _write_multi_case_corpus(
        tmp_path,
        [
            {
                "case_id": "lock_case",
                "split": "locked_test",
                "recording": {"logical_name": "Locked/Sealed.mov", "sha256": None},
            },
            {
                "case_id": "dev_case",
                "used_for_tuning": True,
                "recording": {"logical_name": "locked/sealed.mov", "sha256": None},
            },
        ],
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert any(
        i.path == "split_integrity" and "logical_name" in i.message
        for i in result.issues
    )


def test_declared_digests_alone_catch_the_leak_without_reading_files(tmp_path: Path):
    """Public CI never opens the vault; the manifest's own digests still bind.

    Both cases deny tuning here, so the seal is broken without
    ``locked_used_for_tuning`` becoming true: the flag stays a claim about what
    the manifest declared, and ``locked_seal.intact`` carries the breach.
    """
    digest = "b" * 64
    manifest_path = _write_multi_case_corpus(
        tmp_path,
        [
            {
                "case_id": "lock_case",
                "split": "locked_test",
                "recording": {"logical_name": "locked/a.mov", "sha256": digest},
            },
            {
                "case_id": "val_case",
                "split": "validation",
                "recording": {"logical_name": "val/b.mov", "sha256": digest},
            },
        ],
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert any(i.path == "split_integrity" for i in result.issues)
    assert result.stats["locked_seal"]["intact"] is False
    assert result.stats["locked_used_for_tuning"] is False


def test_report_carries_the_digests_the_seal_was_decided_from(tmp_path: Path):
    """A reader must be able to redo the intersection, not trust the verdict."""
    digest_a = "a" * 64
    digest_b = "c" * 64
    manifest_path = _write_multi_case_corpus(
        tmp_path,
        [
            {
                "case_id": "lock_case",
                "split": "locked_test",
                "recording": {"logical_name": "locked/a.mov", "sha256": digest_a},
            },
            {
                "case_id": "dev_case",
                "recording": {"logical_name": "dev/b.mov", "sha256": digest_b},
            },
        ],
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is True
    by_case = {entry["case_id"]: entry for entry in result.stats["recordings"]}
    assert by_case["lock_case"]["sha256"] == digest_a
    assert by_case["lock_case"]["split"] == "locked_test"
    assert by_case["dev_case"]["sha256"] == digest_b
    # These digests were never checked against bytes, and each entry says so.
    assert by_case["lock_case"]["digest_verified"] is False
    assert by_case["dev_case"]["digest_verified"] is False


def test_seal_report_admits_when_no_content_check_was_possible(tmp_path: Path):
    """A locked case with no declared digest can only be compared by path."""
    manifest_path = _write_multi_case_corpus(
        tmp_path,
        [
            {
                "case_id": "lock_case",
                "split": "locked_test",
                "recording": {"logical_name": "locked/a.mov", "sha256": None},
            },
            {"case_id": "dev_case"},
        ],
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is True
    seal = result.stats["locked_seal"]
    assert seal["checked_by_content"] is False
    assert seal["digests_verified_against_files"] is False
    # Unverifiable, so withheld: a seal nobody could check must not read as one
    # that was checked and held.
    assert seal["intact"] is None
    assert seal["locked_cases_without_digest"] == ["lock_case"]
    # The other side of the comparison is named too: dev_case declared no digest,
    # so nothing in this corpus could have been content-matched against it.
    assert seal["compared_cases_without_digest"] == ["dev_case"]
    assert seal["does_not_cover"]


def test_seal_report_separates_verified_digests_from_asserted_ones(
    tmp_path: Path, monkeypatch
):
    """``digests_verified_against_files`` must be earned by hashing the bytes."""
    vault = tmp_path / "vault"
    vault.mkdir()
    payload = b"locked bytes"
    (vault / "locked.mov").write_bytes(payload)
    monkeypatch.setenv("POKER_VALIDATION_ROOT", str(vault))
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest_path = _write_multi_case_corpus(
        corpus_dir,
        [
            {
                "case_id": "lock_case",
                "split": "locked_test",
                "recording": {
                    "logical_name": "locked.mov",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
            }
        ],
    )
    result = check_corpus(
        manifest_path,
        require_release_minimums=False,
        require_recording_files=True,
    )
    assert result.ok is True
    seal = result.stats["locked_seal"]
    assert seal["checked_by_content"] is True
    assert seal["digests_verified_against_files"] is True
    assert seal["intact"] is True
    entry = next(
        e for e in result.stats["recordings"] if e["case_id"] == "lock_case"
    )
    assert entry["digest_verified"] is True


def test_require_recordings_demands_a_digest_for_unscored_locked_cases(
    tmp_path: Path, monkeypatch
):
    """A locked case exempt from release counts is not exempt from the seal."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("POKER_VALIDATION_ROOT", str(vault))
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest_path = _write_multi_case_corpus(
        corpus_dir,
        [
            {
                "case_id": "lock_fixture",
                "split": "locked_test",
                "runtime_class": "fixture",
                "counts_toward_release": False,
                "recording": {"logical_name": "locked.mov", "sha256": None},
            }
        ],
    )
    result = check_corpus(
        manifest_path,
        require_release_minimums=False,
        require_recording_files=True,
    )
    assert result.ok is False
    assert any("checked by content" in i.message for i in result.issues)


def test_recording_shared_between_development_and_validation_is_rejected(
    tmp_path: Path,
):
    """PLAN.md splits by whole recording; no pair of splits may share one."""
    digest = "d" * 64
    manifest_path = _write_multi_case_corpus(
        tmp_path,
        [
            {
                "case_id": "dev_case",
                "recording": {"logical_name": "dev/a.mov", "sha256": digest},
            },
            {
                "case_id": "val_case",
                "split": "validation",
                "recording": {"logical_name": "val/b.mov", "sha256": digest},
            },
        ],
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert any(
        i.path == "split_integrity" and "locked_test" not in i.message
        for i in result.issues
    )


def test_one_recording_cannot_be_counted_as_two_scored_sessions(tmp_path: Path):
    """Session floors are counted in cases; two cases over one file inflate them."""
    digest = "e" * 64
    manifest_path = _write_multi_case_corpus(
        tmp_path,
        [
            {
                "case_id": "dev_one",
                "recording": {"logical_name": "dev/a.mov", "sha256": digest},
            },
            {
                "case_id": "dev_two",
                "recording": {"logical_name": "dev/copy.mov", "sha256": digest},
            },
        ],
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert result.exit_code == EXIT_SETUP_INVALID
    assert any(i.path == "recording_uniqueness" for i in result.issues)


def test_corpus_with_no_locked_cases_does_not_claim_an_intact_seal():
    """The checked-in manifest has nothing sealed yet; the report must say so."""
    result = check_corpus(MANIFEST, require_release_minimums=False)
    assert result.ok is True
    seal = result.stats["locked_seal"]
    assert seal["locked_cases"] == 0
    assert seal["intact"] is None
    assert seal["checked_by_content"] is False


def test_unhashable_split_is_reported_not_raised(tmp_path: Path):
    """A malformed manifest is a setup verdict; a traceback is no verdict at all."""
    manifest_path = _write_multi_case_corpus(
        tmp_path, [{"case_id": "bad_case", "split": ["development"]}]
    )
    result = check_corpus(manifest_path, require_release_minimums=False)
    assert result.ok is False
    assert result.exit_code == EXIT_SETUP_INVALID
    assert any(i.path.endswith(".split") for i in result.issues)


def test_require_recordings_skips_fixture_cases(tmp_path: Path, monkeypatch):
    """Documented release check must not demand vault files for plumbing fixtures."""
    monkeypatch.setenv("POKER_VALIDATION_ROOT", str(tmp_path / "empty_vault"))
    (tmp_path / "empty_vault").mkdir()
    result = check_corpus(
        MANIFEST,
        require_release_minimums=False,
        require_recording_files=True,
    )
    assert result.ok is True
    assert result.stats["sessions"] == 0
