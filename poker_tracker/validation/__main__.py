"""CLI: ``python -m poker_tracker.validation --manifest validation/clubwpt_v1.json``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from poker_tracker.validation.corpus import check_corpus
from poker_tracker.validation.hashing import sha256_canonical_json
from poker_tracker.validation.schemas import load_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed Phase 2 corpus / answer-key checks.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("validation/clubwpt_v1.json"),
        help="Path to the corpus manifest JSON",
    )
    parser.add_argument(
        "--skip-release-minimums",
        action="store_true",
        help="Validate schema/hashes only; do not enforce 10/100/10 release counts",
    )
    parser.add_argument(
        "--require-recordings",
        action="store_true",
        help="Verify raw videos under POKER_VALIDATION_ROOT match recording.sha256",
    )
    parser.add_argument(
        "--hash-truth",
        type=Path,
        default=None,
        help="Print canonical SHA-256 of an answer-key JSON and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.hash_truth is not None:
        document = load_json(args.hash_truth)
        print(sha256_canonical_json(document))
        return 0

    result = check_corpus(
        args.manifest,
        require_release_minimums=not args.skip_release_minimums,
        require_recording_files=args.require_recordings,
    )
    payload = {
        "ok": result.ok,
        "exit_code": result.exit_code,
        "stats": result.stats,
        "issues": [{"path": i.path, "message": i.message} for i in result.issues],
        "warnings": [{"path": w.path, "message": w.message} for w in result.warnings],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
