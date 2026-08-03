"""Fail-closed private-beta corpus manifest checks (Phase 2 exit gate)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poker_tracker.validation.hashing import sha256_canonical_json, sha256_file
from poker_tracker.validation.schemas import (
    DEFAULT_MINIMUMS,
    MANIFEST_SCHEMA_VERSION,
    REQUIRED_COVERAGE_TAGS,
    VALID_SPLITS,
    load_json,
    require_mapping,
)
from poker_tracker.validation.truth_validate import TruthIssue, validate_answer_key

# Exit codes aligned with the planned release_gate contract.
EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_SETUP_INVALID = 2


@dataclass
class CorpusCheckResult:
    ok: bool
    exit_code: int
    issues: list[TruthIssue] = field(default_factory=list)
    warnings: list[TruthIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def fail(self, path: str, message: str, *, setup: bool = False) -> None:
        self.issues.append(TruthIssue(path=path, message=message))
        self.ok = False
        if setup:
            self.exit_code = max(self.exit_code, EXIT_SETUP_INVALID)
        elif self.exit_code < EXIT_SETUP_INVALID:
            self.exit_code = EXIT_GATE_FAILED

    def warn(self, path: str, message: str) -> None:
        self.warnings.append(TruthIssue(path=path, message=message))


def _validation_root() -> Path | None:
    raw = os.environ.get("POKER_VALIDATION_ROOT", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _normalize_logical_name(logical_name: str) -> str:
    """Comparison key for a recording path: ``.``/redundant separators collapsed,
    then case folded because the vault may sit on a case-insensitive filesystem
    where ``Locked/Sealed.mov`` and ``locked/sealed.mov`` are one file.

    Path text is only half of a recording's identity. Two unrelated paths can
    hold the same bytes, so split integrity compares declared digests as well;
    this key alone cannot carry the seal.
    """
    normalized = Path(os.path.normpath(logical_name))
    return normalized.as_posix().casefold()


def _group_by_identity(
    recordings: list[dict[str, Any]], field: str
) -> dict[str, list[dict[str, Any]]]:
    """Group recording entries by one identity field, ignoring absent values.

    A null digest is the absence of an identity claim, not an identity shared
    with every other null, so those entries are left out of the grouping.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in recordings:
        value = entry.get(field)
        if isinstance(value, str) and value:
            groups.setdefault(value, []).append(entry)
    return groups


# Content equality is what a digest can answer. These are the ways one recording
# can reach two splits without the digests colliding, named in the report so a
# reader treats a clean seal as the narrow finding it is.
SEAL_DOES_NOT_COVER: tuple[str, ...] = (
    "re-encoded, trimmed, or otherwise re-rendered copies of a locked recording",
    "near-duplicate or adjacent segments captured from the same source session",
)


def check_corpus(
    manifest_path: Path,
    *,
    require_release_minimums: bool = True,
    require_recording_files: bool = False,
) -> CorpusCheckResult:
    """Validate a corpus manifest and every referenced answer key.

    ``require_release_minimums`` enforces the Phase 2 session/hand counts and
    required coverage tags. Keep it true for release; tests may disable it when
    checking schema plumbing on a tiny fixture corpus.

    ``require_recording_files`` verifies SHA-256 of the release-scored raw videos
    under ``POKER_VALIDATION_ROOT`` (plumbing fixtures are skipped) and demands a
    declared digest for every locked_test case, since the seal is checked by
    content. Public CI leaves this false; local release runs set it true. With it
    false the split checks compare what the manifest declares, and
    ``stats["locked_seal"]`` says so rather than implying the bytes were read.
    """
    result = CorpusCheckResult(ok=True, exit_code=EXIT_OK)
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        result.fail("manifest", f"manifest not found: {manifest_path}", setup=True)
        return result

    try:
        document = require_mapping(load_json(manifest_path), label="manifest")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result.fail("manifest", f"cannot read manifest: {exc}", setup=True)
        return result

    version = document.get("schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        result.fail(
            "schema_version",
            f"expected {MANIFEST_SCHEMA_VERSION}, got {version!r}",
            setup=True,
        )

    corpus_id = document.get("corpus_id")
    if not isinstance(corpus_id, str) or not corpus_id.strip():
        result.fail("corpus_id", "corpus_id is required", setup=True)

    # Phase 2 floors are code-enforced. A manifest may raise a bar, never lower it.
    minimums = dict(DEFAULT_MINIMUMS)
    raw_minimums = document.get("minimums") or {}
    if raw_minimums and not isinstance(raw_minimums, dict):
        result.fail("minimums", "minimums must be an object", setup=True)
    elif isinstance(raw_minimums, dict):
        for key, value in raw_minimums.items():
            if key not in DEFAULT_MINIMUMS:
                result.fail(f"minimums.{key}", f"unknown minimum key {key!r}", setup=True)
            elif not isinstance(value, int) or value < 0:
                result.fail(f"minimums.{key}", "minimum must be a non-negative int", setup=True)
            elif value < DEFAULT_MINIMUMS[key]:
                result.fail(
                    f"minimums.{key}",
                    f"manifest may not lower Phase 2 floor "
                    f"({value} < {DEFAULT_MINIMUMS[key]})",
                    setup=True,
                )
            else:
                minimums[key] = value

    # Coverage tags likewise: REQUIRED_COVERAGE_TAGS is a hard floor; extras ok.
    required_tags = list(REQUIRED_COVERAGE_TAGS)
    raw_tags = document.get("required_coverage_tags")
    if raw_tags is not None:
        if not isinstance(raw_tags, list) or not all(isinstance(t, str) for t in raw_tags):
            result.fail(
                "required_coverage_tags",
                "required_coverage_tags must be a list of strings",
                setup=True,
            )
        else:
            missing_floor = [tag for tag in REQUIRED_COVERAGE_TAGS if tag not in raw_tags]
            if missing_floor:
                result.fail(
                    "required_coverage_tags",
                    "manifest may not drop Phase 2 coverage tags: "
                    + ", ".join(missing_floor),
                    setup=True,
                )
            required_tags = list(dict.fromkeys([*REQUIRED_COVERAGE_TAGS, *raw_tags]))

    cases = document.get("cases")
    if not isinstance(cases, list):
        result.fail("cases", "cases must be a list", setup=True)
        return result

    vault = _validation_root()
    if require_recording_files and vault is None:
        result.fail(
            "POKER_VALIDATION_ROOT",
            "recording verification requested but POKER_VALIDATION_ROOT is unset",
            setup=True,
        )

    split_counts = {"development": 0, "validation": 0, "locked_test": 0}
    scored_sessions = 0
    completed_hands = 0
    partial_hands = 0
    uncertain_hands = 0
    covered_tags: set[str] = set()
    seen_case_ids: set[str] = set()
    locked_used_for_tuning = False
    # One entry per case that declared a usable recording identity. Split
    # integrity is decided from this list rather than from per-split name sets,
    # because a recording is identified by its content and its path alike.
    recordings: list[dict[str, Any]] = []
    file_verified_case_ids: set[str] = set()

    for index, case in enumerate(cases):
        path = f"cases[{index}]"
        if not isinstance(case, dict):
            result.fail(path, "case must be an object", setup=True)
            continue

        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            result.fail(f"{path}.case_id", "case_id is required", setup=True)
            case_id = f"<invalid:{index}>"
        elif case_id in seen_case_ids:
            result.fail(f"{path}.case_id", f"duplicate case_id {case_id}", setup=True)
        seen_case_ids.add(str(case_id))

        runtime_class = case.get("runtime_class")
        counts_toward_release = case.get("counts_toward_release")
        if runtime_class == "fixture":
            # Fixtures are schema plumbing only. They cannot opt into Phase 2
            # release floors/coverage by setting counts_toward_release=true.
            if counts_toward_release is True:
                result.fail(
                    f"{path}.counts_toward_release",
                    "runtime_class=fixture cases cannot count toward release",
                    setup=True,
                )
            counts_toward_release = False
        elif counts_toward_release is None:
            counts_toward_release = True
        elif not isinstance(counts_toward_release, bool):
            result.fail(
                f"{path}.counts_toward_release",
                "counts_toward_release must be a boolean when present",
                setup=True,
            )
            counts_toward_release = False

        split = case.get("split")
        # The isinstance guard is load-bearing: an unhashable value here (a list,
        # say) makes the set test raise, and the gate would die with a traceback
        # instead of reporting the malformed manifest as setup-invalid.
        if not isinstance(split, str) or split not in VALID_SPLITS:
            result.fail(f"{path}.split", f"invalid split {split!r}", setup=True)
        elif counts_toward_release:
            split_counts[str(split)] += 1
            scored_sessions += 1

        tuning_flag = case.get("used_for_tuning")
        if split == "locked_test":
            if tuning_flag is not False:
                locked_used_for_tuning = True
                result.fail(
                    f"{path}.used_for_tuning",
                    "locked_test cases must set used_for_tuning to false "
                    f"(got {tuning_flag!r})",
                )
        elif tuning_flag is not None and not isinstance(tuning_flag, bool):
            result.fail(
                f"{path}.used_for_tuning",
                "used_for_tuning must be a boolean when present",
                setup=True,
            )

        tags = case.get("coverage_tags") or []
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            result.fail(
                f"{path}.coverage_tags",
                "coverage_tags must be a string list",
                setup=True,
            )
        elif counts_toward_release:
            covered_tags.update(tags)

        layout = case.get("layout_profile")
        if not isinstance(layout, str) or not layout.strip():
            result.fail(f"{path}.layout_profile", "layout_profile is required", setup=True)

        recording = case.get("recording")
        if not isinstance(recording, dict):
            result.fail(f"{path}.recording", "recording object is required", setup=True)
            recording = {}
        logical_name = recording.get("logical_name")
        normalized_name: str | None = None
        if not isinstance(logical_name, str) or not logical_name.strip():
            result.fail(
                f"{path}.recording.logical_name",
                "logical_name is required",
                setup=True,
            )
        elif Path(logical_name).is_absolute() or ".." in Path(logical_name).parts:
            result.fail(
                f"{path}.recording.logical_name",
                "logical_name must be a relative path under POKER_VALIDATION_ROOT "
                "(no absolute paths or '..' segments)",
                setup=True,
            )
        else:
            normalized_name = _normalize_logical_name(logical_name)
            if normalized_name.startswith("../") or normalized_name == "..":
                result.fail(
                    f"{path}.recording.logical_name",
                    "logical_name must be a relative path under POKER_VALIDATION_ROOT "
                    "(no absolute paths or '..' segments)",
                    setup=True,
                )
                normalized_name = None

        expected_sha = recording.get("sha256")
        declared_digest: str | None = None
        if expected_sha is not None and (
            not isinstance(expected_sha, str) or len(expected_sha) != 64
        ):
            result.fail(
                f"{path}.recording.sha256",
                "sha256 must be a 64-char hex digest or null",
                setup=True,
            )
        elif isinstance(expected_sha, str):
            declared_digest = expected_sha.lower()
        # A locked_test case needs a digest even when it is not release-scored:
        # the seal is checked by content, and a locked recording with no declared
        # content can only ever be compared by path.
        digest_required = require_recording_files and (
            counts_toward_release or split == "locked_test"
        )
        if digest_required and declared_digest is None:
            seal_note = (
                "; the locked_test seal is checked by content"
                if split == "locked_test"
                else ""
            )
            result.fail(
                f"{path}.recording.sha256",
                "recording.sha256 is required when --require-recordings is set"
                + seal_note,
                setup=True,
            )

        has_identity = bool(normalized_name or declared_digest)
        if has_identity and isinstance(split, str) and split in VALID_SPLITS:
            recordings.append(
                {
                    "case_id": str(case_id),
                    "split": split,
                    "logical_name": (
                        logical_name if isinstance(logical_name, str) else None
                    ),
                    "name_key": normalized_name,
                    "sha256": declared_digest,
                    "counts_toward_release": bool(counts_toward_release),
                    "used_for_tuning": tuning_flag,
                }
            )

        if (
            require_recording_files
            and counts_toward_release
            and vault is not None
            and isinstance(logical_name, str)
            and not Path(logical_name).is_absolute()
            and ".." not in Path(logical_name).parts
        ):
            video_path = (vault / logical_name).resolve()
            try:
                video_path.relative_to(vault)
            except ValueError:
                result.fail(
                    f"{path}.recording.logical_name",
                    "recording path escapes POKER_VALIDATION_ROOT",
                    setup=True,
                )
                video_path = None
            if video_path is not None and not video_path.is_file():
                result.fail(
                    f"{path}.recording",
                    f"recording file missing under validation root: {logical_name}",
                    setup=True,
                )
            elif video_path is not None and declared_digest is not None:
                actual = sha256_file(video_path)
                if actual != declared_digest:
                    result.fail(
                        f"{path}.recording.sha256",
                        f"recording hash mismatch for {logical_name}",
                    )
                else:
                    # Only these cases have had their declared digest confirmed
                    # against bytes; the seal report distinguishes them from
                    # cases whose digest is merely asserted by the manifest.
                    file_verified_case_ids.add(str(case_id))

        truth_relpath = case.get("truth_relpath")
        if not isinstance(truth_relpath, str) or not truth_relpath.strip():
            result.fail(f"{path}.truth_relpath", "truth_relpath is required", setup=True)
            continue
        truth_path = (manifest_path.parent / truth_relpath).resolve()
        manifest_root = manifest_path.parent.resolve()
        try:
            truth_path.relative_to(manifest_root)
        except ValueError:
            result.fail(
                f"{path}.truth_relpath",
                "truth_relpath escapes the manifest directory",
                setup=True,
            )
            continue
        if not truth_path.is_file():
            result.fail(
                f"{path}.truth_relpath",
                f"answer key not found: {truth_relpath}",
                setup=True,
            )
            continue

        try:
            truth_doc = require_mapping(load_json(truth_path), label="answer_key")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result.fail(
                f"{path}.truth_relpath",
                f"cannot read answer key: {exc}",
                setup=True,
            )
            continue

        expected_truth_sha = case.get("truth_sha256")
        actual_truth_sha = sha256_canonical_json(truth_doc)
        if expected_truth_sha is None:
            result.fail(
                f"{path}.truth_sha256",
                "truth_sha256 is required (freeze answer keys before scoring)",
                setup=True,
            )
        elif not isinstance(expected_truth_sha, str) or len(expected_truth_sha) != 64:
            result.fail(
                f"{path}.truth_sha256",
                "truth_sha256 must be a 64-char hex digest",
                setup=True,
            )
        elif expected_truth_sha.lower() != actual_truth_sha:
            result.fail(
                f"{path}.truth_sha256",
                "answer-key hash mismatch; re-freeze after intentional edits",
            )

        if truth_doc.get("case_id") != case_id:
            result.fail(
                f"{path}.truth_relpath",
                f"answer key case_id {truth_doc.get('case_id')!r} != manifest {case_id!r}",
            )

        truth_result = validate_answer_key(truth_doc)
        for issue in truth_result.issues:
            result.fail(f"{path}.truth.{issue.path}", issue.message)
        if counts_toward_release:
            completed_hands += truth_result.completed_hands
            partial_hands += truth_result.partial_hands
            uncertain_hands += truth_result.uncertain_hands

        annotation = truth_doc.get("annotation")
        if isinstance(annotation, dict):
            if annotation.get("adjudicated") is not True:
                if require_release_minimums and counts_toward_release:
                    result.fail(
                        f"{path}.annotation.adjudicated",
                        "release corpus requires adjudicated answer keys",
                        setup=True,
                    )
                else:
                    result.warn(
                        f"{path}.annotation.adjudicated",
                        "answer key is not adjudicated; release scoring must not treat it as frozen truth",
                    )
            passes = annotation.get("passes")
            if isinstance(passes, int) and passes < 2:
                if require_release_minimums and counts_toward_release:
                    result.fail(
                        f"{path}.annotation.passes",
                        "release corpus requires at least two independent annotation passes",
                        setup=True,
                    )
                else:
                    result.warn(
                        f"{path}.annotation.passes",
                        "Phase 2 requires two independent annotation passes before release scoring",
                    )
        elif require_release_minimums and counts_toward_release:
            result.fail(
                f"{path}.annotation",
                "release corpus requires an annotation object",
                setup=True,
            )

    # PLAN.md splits by whole recording, so one recording reaching two splits is
    # a breach whichever splits they are; when locked_test is one of them it is
    # the seal itself. Path and digest are checked separately because either can
    # be the only identity a manifest offers.
    collisions: list[dict[str, Any]] = []
    for kind in ("name_key", "sha256"):
        for value, entries in sorted(_group_by_identity(recordings, kind).items()):
            splits = sorted({entry["split"] for entry in entries})
            case_ids = sorted({entry["case_id"] for entry in entries})
            label = "logical_name" if kind == "name_key" else "sha256"
            if len(splits) > 1:
                collisions.append(
                    {
                        "identity": label,
                        "value": value,
                        "splits": splits,
                        "case_ids": case_ids,
                        "locked": "locked_test" in splits,
                    }
                )
                if "locked_test" in splits:
                    others = [s for s in splits if s != "locked_test"]
                    result.fail(
                        "split_integrity",
                        f"locked_test seal broken: {label} shared with "
                        + "/".join(others)
                        + " (cases "
                        + ", ".join(case_ids)
                        + ")",
                    )
                    # development and validation ARE the tuning pool. A locked
                    # recording found there was tuned against unless every case
                    # holding it denies it outright; an undeclared flag is not a
                    # denial, so it counts against the seal.
                    if any(
                        entry["used_for_tuning"] is not False
                        for entry in entries
                        if entry["split"] != "locked_test"
                    ):
                        locked_used_for_tuning = True
                else:
                    result.fail(
                        "split_integrity",
                        f"{label} shared across "
                        + "/".join(splits)
                        + " (cases "
                        + ", ".join(case_ids)
                        + "); split by whole recording",
                    )
            elif (
                len(case_ids) > 1
                and sum(1 for entry in entries if entry["counts_toward_release"]) > 1
            ):
                # Two scored cases over one recording inflate the session count,
                # which is the number the release floors are checked against.
                result.fail(
                    "recording_uniqueness",
                    f"{label} claimed by more than one release-scored case (cases "
                    + ", ".join(case_ids)
                    + "); each scored session must be a distinct recording",
                    setup=True,
                )

    locked_entries = [e for e in recordings if e["split"] == "locked_test"]
    locked_without_digest = sorted(
        e["case_id"] for e in locked_entries if e["sha256"] is None
    )
    breached = locked_used_for_tuning or any(c["locked"] for c in collisions)
    # False with zero locked cases too: a check with no subject certifies
    # nothing, and must not read as one that ran and found nothing wrong.
    checked_by_content = bool(locked_entries) and not locked_without_digest
    locked_seal = {
        "locked_cases": len(locked_entries),
        "checked_by_content": checked_by_content,
        "digests_verified_against_files": bool(locked_entries)
        and all(e["case_id"] in file_verified_case_ids for e in locked_entries),
        "locked_cases_without_digest": locked_without_digest,
        # A locked digest can only collide with a digest something else declared.
        # Cases that declare none were compared by path alone, and the seal is
        # exactly that much weaker for them.
        "compared_cases_without_digest": sorted(
            e["case_id"]
            for e in recordings
            if e["split"] != "locked_test" and e["sha256"] is None
        ),
        "cross_split_collisions": collisions,
        # Tri-state on purpose. A corpus with no locked cases, or with locked
        # cases nobody gave a digest, has a seal that could not be checked by
        # content; ``true`` there would be a pass nobody earned. Only a breach
        # actually found reads as false.
        "intact": False if breached else (True if checked_by_content else None),
        "does_not_cover": list(SEAL_DOES_NOT_COVER),
    }

    missing_tags = [tag for tag in required_tags if tag not in covered_tags]
    result.stats = {
        "corpus_id": corpus_id,
        "sessions": scored_sessions,
        "listed_cases": len(cases),
        "split_counts": split_counts,
        "completed_hands": completed_hands,
        "partial_hands": partial_hands,
        "uncertain_hands": uncertain_hands,
        "covered_tags": sorted(covered_tags),
        "missing_coverage_tags": missing_tags,
        "minimums": minimums,
        "locked_used_for_tuning": locked_used_for_tuning,
        # The digests the split checks ran on, carried into the report so a later
        # reader can redo the intersection instead of trusting the flag above.
        "recordings": [
            {
                "case_id": entry["case_id"],
                "split": entry["split"],
                "logical_name": entry["logical_name"],
                "sha256": entry["sha256"],
                "digest_verified": entry["case_id"] in file_verified_case_ids,
                "counts_toward_release": entry["counts_toward_release"],
            }
            for entry in sorted(recordings, key=lambda e: (e["split"], e["case_id"]))
        ],
        "locked_seal": locked_seal,
    }

    if require_release_minimums:
        # Incomplete corpus is a setup failure (exit 2), not a product accuracy miss.
        if scored_sessions < minimums["sessions"]:
            result.fail(
                "minimums.sessions",
                f"need {minimums['sessions']} scored sessions, found {scored_sessions}",
                setup=True,
            )
        if split_counts["development"] < minimums["development_sessions"]:
            result.fail(
                "minimums.development_sessions",
                f"need {minimums['development_sessions']} development sessions, "
                f"found {split_counts['development']}",
                setup=True,
            )
        if split_counts["validation"] < minimums["validation_sessions"]:
            result.fail(
                "minimums.validation_sessions",
                f"need {minimums['validation_sessions']} validation sessions, "
                f"found {split_counts['validation']}",
                setup=True,
            )
        if split_counts["locked_test"] < minimums["locked_test_sessions"]:
            result.fail(
                "minimums.locked_test_sessions",
                f"need {minimums['locked_test_sessions']} locked_test sessions, "
                f"found {split_counts['locked_test']}",
                setup=True,
            )
        if completed_hands < minimums["completed_hands"]:
            result.fail(
                "minimums.completed_hands",
                f"need {minimums['completed_hands']} completed hands, found {completed_hands}",
                setup=True,
            )
        if partial_hands < minimums["partial_hands"]:
            result.fail(
                "minimums.partial_hands",
                f"need {minimums['partial_hands']} partial hands, found {partial_hands}",
                setup=True,
            )
        if missing_tags:
            result.fail(
                "required_coverage_tags",
                "missing coverage tags: " + ", ".join(missing_tags),
                setup=True,
            )

    return result
