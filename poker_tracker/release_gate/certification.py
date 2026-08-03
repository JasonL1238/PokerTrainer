"""What a release run actually performed, and what its verdict may claim.

A coverage statement computed from the requested mode describes what the run was
supposed to do. The two diverge whenever a mode degrades: container mode executes
the fixture gate inside the image, and full mode returns before decoding anything
when the vault, FFmpeg or the weights are absent. A mode-derived claim survives
both, so the report goes on advertising video decoding and end-to-end
reconstruction that no code performed.

So coverage is computed from a record of acts, and each act is appended by the
code that carries it out, on evidence that it succeeded. A mode that silently
does less than its name says cannot inflate its own claim, because the claim is
assembled from what came back rather than from what was asked for.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Acts a run can perform. Recorded by the code that performs them.
SCORE_TIMELINES = "score_timelines"
DECODE_VIDEO = "decode_video"
LOAD_WEIGHTS = "load_pinned_weights"
RECONSTRUCT = "reconstruct_recordings"
RUN_IN_IMAGE = "run_gate_in_container_image"

# Canonical order, so two runs of one verdict serialize byte-identically.
ACT_ORDER: tuple[str, ...] = (
    DECODE_VIDEO,
    LOAD_WEIGHTS,
    RECONSTRUCT,
    SCORE_TIMELINES,
    RUN_IN_IMAGE,
)

_COVERS: dict[str, str] = {
    DECODE_VIDEO: "video decoding and sampling of the recordings this run processed",
    LOAD_WEIGHTS: "execution of the pinned model weights",
    RECONSTRUCT: "end-to-end reconstruction of the recordings this run processed",
    SCORE_TIMELINES: "hand-level scoring of the timelines this run was given",
    RUN_IN_IMAGE: (
        "reproducibility of the gate inside the pinned container image: the "
        "image ran the gate and reached the same verdict"
    ),
}

# What the absence of an act leaves unexamined. ``SCORE_TIMELINES`` is absent
# here because the aggregate's own ``measured`` flag reports that shortfall in
# the numbers a reader would quote, and saying it twice invites skimming past it.
_DOES_NOT_COVER: dict[str, str] = {
    DECODE_VIDEO: "video decoding, sampling, anchoring, and hand-boundary detection",
    LOAD_WEIGHTS: "model execution: this run loaded no model weights",
    RECONSTRUCT: "end-to-end reconstruction of the corpus recordings",
}

_SHORTFALL: dict[str, str] = {
    DECODE_VIDEO: "decode video",
    LOAD_WEIGHTS: "load the pinned weights",
    RECONSTRUCT: "reconstruct the corpus recordings",
    SCORE_TIMELINES: "score any timeline",
}

# A release verdict claims the product did its whole job on real input, so every
# act below has to have happened for the run to certify one.
CERTIFYING_ACTS: frozenset[str] = frozenset(
    {DECODE_VIDEO, LOAD_WEIGHTS, RECONSTRUCT, SCORE_TIMELINES}
)

# Scoring a timeline this run did not itself produce says nothing about where
# that timeline came from.
_UNATTRIBUTED_TIMELINES = (
    "provenance of the scored timelines: inputs this run did not reconstruct "
    "are trusted as given and are not attributed to a pipeline run"
)


def normalize_acts(acts: Any) -> list[str]:
    """The recognized acts in ``acts``, in canonical order, deduplicated.

    Unknown entries are dropped rather than passed through: an act only means
    something if this module can say what it covers, and an unrecognized string
    reaching the report would be a coverage claim nobody defined.
    """
    if isinstance(acts, str) or not isinstance(acts, Iterable):
        return []
    present = {act for act in acts if isinstance(act, str)}
    return [act for act in ACT_ORDER if act in present]


def certification(
    *,
    mode: str,
    executed: Any,
    measured: bool,
) -> dict[str, Any]:
    """The coverage statement implied by what this run actually did.

    ``measured`` comes from the aggregate rather than from the act record: a run
    can score cases and still have scored zero hands, and the certification must
    not imply an accuracy result the numbers do not contain.
    """
    acts = normalize_acts(executed)
    performed = set(acts)
    missing = [
        act for act in ACT_ORDER if act in CERTIFYING_ACTS and act not in performed
    ]
    certifying = not missing

    # Schema integrity is checked on every path the report is written from, so
    # it is the one claim that does not depend on the act record.
    covers = ["answer-key and manifest schema integrity"]
    covers.extend(_COVERS[act] for act in acts if act in _COVERS)

    does_not_cover = [
        _DOES_NOT_COVER[act]
        for act in ACT_ORDER
        if act in _DOES_NOT_COVER and act not in performed
    ]
    if SCORE_TIMELINES in performed and RECONSTRUCT not in performed:
        does_not_cover.append(_UNATTRIBUTED_TIMELINES)
    if not measured:
        does_not_cover.append("accuracy: no hand was scored in this run")

    if certifying:
        summary = (
            "Release-certifying run: the pinned weights reconstructed the corpus "
            "recordings and the result was scored."
        )
    else:
        shortfall = _join(_SHORTFALL[act] for act in missing)
        summary = (
            "NOT a release certification: this run did not "
            f"{shortfall}. Coverage below is what it performed, not what "
            f"{mode!r} mode is named for."
        )
    return {
        "release_certifying": certifying,
        "mode": mode,
        # The record the rest of this block is computed from, so a reader can
        # check the claim against the acts instead of trusting the prose.
        "executed": acts,
        "missing_for_certification": missing,
        "summary": summary,
        "covers": covers,
        "does_not_cover": does_not_cover,
    }


def _join(parts: Iterable[str]) -> str:
    items = list(parts)
    if not items:
        return "nothing"
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])}, or {items[-1]}"
