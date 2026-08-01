"""Build a self-contained bundle for one saved issue.

The saved evidence snapshot freezes what the hand looked like when it was
flagged. That is enough to see the symptom and not enough to reproduce it: it
says nothing about which recording the hand came from, which timeline the
reconstruction wrote, or which model weights produced it. Six months later,
with the models retrained, "the flop card was wrong" is unreproducible.

A bundle answers "what exactly would I have to put back to see this again":
the frozen evidence, the identity (path, size, hash) of the source recording
and timeline, the model hashes, and the redacted runtime configuration. It
carries identifiers and hashes, never the recording itself -- a bundle is
something an operator attaches to a report, and PLAN.md is explicit that videos
stay out of SQL and out of anything routinely copied around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.release_gate.environment import collect_environment
from poker_tracker.release_gate.models import resolve_models
from poker_tracker.safety.redaction import redact_structure
from poker_tracker.validation.hashing import sha256_file

BUNDLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactIdentity:
    """What a file was, recorded so its absence is distinguishable from drift."""

    path: str
    present: bool
    bytes: int | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "present": self.present,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


def identify_artifact(path: str | Path | None, *, hash_limit_bytes: int) -> ArtifactIdentity | None:
    """Describe a file without copying it.

    ``hash_limit_bytes`` bounds hashing so a bundle for a hand from a 12 GB
    recording does not spend minutes re-reading it. An unhashed large file still
    records its size, which catches replacement and truncation.
    """
    if path is None or not str(path).strip():
        return None
    candidate = Path(str(path))
    if not candidate.is_file():
        return ArtifactIdentity(path=str(candidate), present=False)
    size = candidate.stat().st_size
    digest = sha256_file(candidate) if size <= hash_limit_bytes else None
    return ArtifactIdentity(
        path=str(candidate), present=True, bytes=size, sha256=digest
    )


# 512 MiB: comfortably covers timelines, exports and frames while skipping raw
# recordings, which are identified by size alone.
DEFAULT_HASH_LIMIT_BYTES = 512 * 1024 * 1024


def build_issue_bundle(
    db: PokerDatabase,
    issue_id: int,
    *,
    repo_root: Path | None = None,
    hash_limit_bytes: int = DEFAULT_HASH_LIMIT_BYTES,
) -> dict[str, Any]:
    """Assemble everything needed to reproduce one issue later."""

    issue = db.fetch_hand_issue(issue_id)
    if issue is None:
        raise ValueError(f"Issue {issue_id} was not found.")
    hand = db.fetch_hand(issue.hand_id)
    if hand is None:
        raise ValueError(f"Issue {issue_id} points at a hand that no longer exists.")

    session = db.fetch_session(hand.session_id)
    videos = db.fetch_videos(hand.session_id) if hand.session_id else []
    root = repo_root or Path(__file__).resolve().parents[2]

    recordings = [
        {
            "video_id": video.id,
            "original_filename": video.original_filename,
            "duration_seconds": video.duration_seconds,
            "fps": video.fps,
            "width": video.width,
            "height": video.height,
            "frame_count": video.frame_count,
            "recorded_sha256": video.content_sha256 or None,
            "file": (
                identity.to_dict()
                if (
                    identity := identify_artifact(
                        video.stored_path, hash_limit_bytes=hash_limit_bytes
                    )
                )
                else None
            ),
        }
        for video in videos
    ]

    frames = sorted(
        {
            str(action.source_image)
            for action in db.fetch_actions_by_hand(issue.hand_id)
            if getattr(action, "source_image", None)
        }
    )

    return {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "issue": issue.model_dump(mode="json"),
        # The frozen snapshot is reproduced verbatim; it is the record of what
        # the operator was looking at, and re-deriving it would defeat the point.
        "hand": hand.model_dump(mode="json"),
        "session": None if session is None else session.model_dump(mode="json"),
        "corrections": [
            correction.model_dump(mode="json")
            for correction in db.fetch_hand_corrections(issue.hand_id)
        ],
        "source_recordings": recordings,
        "source_frames": [
            identity.to_dict()
            for path in frames
            if (identity := identify_artifact(path, hash_limit_bytes=hash_limit_bytes))
        ],
        "models": resolve_models(root),
        # Redacted twice over: collect_environment scrubs by key name and value
        # shape, and the bundle is a document an operator may hand to someone else.
        "environment": collect_environment(root),
        "notes": (
            "Identifiers and hashes only. The recording itself is deliberately "
            "not included; restore it from the path and hash above."
        ),
    }


def serialize_issue_bundle(bundle: dict[str, Any]) -> str:
    """Deterministic JSON, with a final redaction pass over the whole document.

    The per-field scrubbing upstream covers the fields we know about. This
    catches a credential that reached the bundle through free text -- an issue
    description, a correction note, a resolution comment -- which is exactly
    where an operator pastes one without thinking.
    """
    # Scrub first, serialize second. json.dumps escapes the quotes inside every
    # string field, and an escaped key stops matching the assignment pattern --
    # so a credential pasted in as JSON survived a pass run over the output.
    scrubbed = redact_structure(bundle)
    return json.dumps(scrubbed, indent=2, sort_keys=True, ensure_ascii=False)
