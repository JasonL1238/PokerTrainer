"""Versioned reconstruction-completion evidence for completed hands.

This module is deliberately a leaf: it imports only the standard library and the
``CompletionStatus`` literal, so the CV producer, the persistence layer, and the
study-readiness service can all depend on it without an import cycle.

Nothing in this module raises. A missing, legacy, or corrupt evidence blob parses
to an all-unknown envelope, which *blocks* study readiness instead of failing a
read or surfacing a ValidationError in the UI. The envelope lives here rather
than in ``models.py`` precisely because ``models.py`` is the validating boundary
and this parser must never validate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from typing import Any

from poker_tracker.persistence.models import CompletionStatus

EVIDENCE_SCHEMA_VERSION = 1

# Where the reader records a card column it could not read back. It lives in the
# evidence's open ``extra`` mapping rather than in a new Hand column: no schema
# change, it survives export/import, and it carries no ``evidence_version``, so a
# manual hand annotated this way is still not "carrying reconstruction evidence".
UNREADABLE_CARDS_KEY = "unreadable_card_columns"

# Where the reader records every OTHER hand column it could not read back — a
# review status this build has no vocabulary for, a negative pot, a table size
# of 99. Same channel and same lifecycle as ``UNREADABLE_CARDS_KEY``: injected
# at read time by ``db._hand_from_row``'s degradation, reported as a
# study-readiness blocker with the offending text, and stripped by every writer
# (``strip_derived_evidence_markers``) so it stays purely derived — it appears
# while the row is unreadable and disappears the moment it is corrected.
UNREADABLE_HAND_COLUMNS_KEY = "unreadable_hand_columns"

# Every read-time annotation, so ``strip_derived_evidence_markers`` cannot fall
# behind when the next one is added. Public, because three separate rules are
# keyed on "does this hand carry a read-time degradation marker?" rather than on
# which one: no writer may persist any of them
# (``strip_derived_evidence_markers``), any of them demotes ``review_status``
# (``db._demote_degraded_hand``), and import restores the raw columns any of them
# names (``import_export._recorded_unreadable_columns``).
DERIVED_EVIDENCE_KEYS = frozenset({UNREADABLE_CARDS_KEY, UNREADABLE_HAND_COLUMNS_KEY})

# The closed vocabulary of terminal events the pipeline can OBSERVE. This is an
# allow-list on purpose: ``derive_completion_status`` used to deny-list two
# spellings ("" and "unobserved") while the CV exporter allow-listed these
# three, so any unrecognised value — a grown pipeline vocabulary, a hand-edited
# timeline, a payload's "chop" or "PREFLOP" — was treated as an observed
# terminal event by the one function the completion invariant hangs on, while
# the producer classified the same hand as terminal-not-observed. An
# unrecognised terminal event is not an observed terminal event; it fails
# closed to ``uncertain`` exactly as an unreadable ``evidence_version`` does.
OBSERVED_TERMINAL_EVENTS = frozenset({"showdown", "fold_win", "hero_fold"})

# One completed hand's reconciliation may rest on settlement inputs only the
# operator can assert -- a rake policy, external dead money. When it does, the
# dependence is measured (see ``services.hand_accounting``) and named by a code
# of the form:
#
#     declared_settlement_dependence:<input>:<declaration fingerprint>:<movement>
#
# Both the declaration and the measured movement are part of the code on
# purpose. An attestation is stored as the code string, so one earned against
# 0.01 chips of rake is a DIFFERENT string from one covering 80.01 chips, and one
# earned against a 50% rake is a different string from one covering a 25% rake
# that happens to remove the same number of chips off a corrected action line.
# There is no field list here and none anywhere else: the code exists iff
# neutralising the declared inputs changes what this hand reports.
#
# The vocabulary lives in this leaf module so the persistence writer, the
# accounting service, and the readiness service can all name the same codes
# without an import cycle.
ASSUMPTION_DEPENDENCE_PREFIX = "declared_settlement_dependence"


def is_assumption_dependence_code(code: object) -> bool:
    """True for a code produced by the assumption-dependence measurement."""
    return isinstance(code, str) and code.startswith(f"{ASSUMPTION_DEPENDENCE_PREFIX}:")


# The writer-side audit record that a settlement input the OPERATOR declared
# actually moves chips on this hand (`db._declared_chips_taken`). It is a
# statement about a declaration, not a pipeline finding, and it has its own
# channel for the same reason the attestation does.
#
# It used to be written into ``warning_codes``. That is the pipeline's channel:
# ``derive_completion_status`` demotes on an unresolved entry in it, so declaring
# a rake on a hand whose reconstruction evidence was complete and clean turned
# the hand ``uncertain`` and told the operator "The pipeline could not prove this
# hand was fully reconstructed" and "The pipeline flagged 1 unresolved source
# warning(s)" -- about a figure the pipeline never claimed and never observed --
# and sent them to Correct hand facts, a form with no rake field in it, to fix a
# value that exists only in the Accounting reconciliation panel.
#
# Nothing in readiness reads this channel. The gate on a declared settlement
# input is ``ACCOUNTING_ASSUMPTION_DEPENDENT``, measured per read from the chips,
# and the answer to it is an attestation to that measurement -- so there is
# nothing here to acknowledge, and no control offers to.
DECLARED_SETTLEMENT_PREFIX = "declared_unobserved_"


def is_declared_settlement_code(code: object) -> bool:
    """True for a writer-side disclosure of an operator's own settlement declaration."""
    return isinstance(code, str) and code.startswith(DECLARED_SETTLEMENT_PREFIX)


def assumption_dependence_input(code: str) -> str:
    """The declared input one dependence code names, or '' when it names none.

    Used to keep at most one recorded disclosure per input: a re-measured
    quantity replaces the previous measurement of the SAME input rather than
    piling a second, contradictory line into the evidence.
    """
    if not is_assumption_dependence_code(code):
        return ""
    parts = code.split(":")
    return parts[1] if len(parts) > 2 else ""


# Where a hand records that it arrived in an import payload rather than being
# entered here. It lives in the open ``extra`` mapping for the same reasons
# ``UNREADABLE_CARDS_KEY`` does: no schema change, and it carries no
# ``evidence_version``, so a manual hand annotated this way is still not
# "carrying reconstruction evidence" and import's manual-payload guard is
# unaffected.
#
# It exists because the manual-hand exemption is an argument about WHO entered
# the hand, not about a string in a column. "You entered this hand yourself, so a
# declared ante or rake is your own observation" is true of a hand typed into
# this database and false of every hand that arrived as user-supplied JSON --
# which is exactly why import already refuses to land ``reviewed`` and already
# resets acknowledged codes. A payload that declares ``source_type: manual`` with
# no evidence is byte-identical to a genuine manual export, so no guard can tell
# them apart; what closes the hole is that neither of them may claim an exemption
# that rests on being entered here.
IMPORTED_HAND_KEY = "imported_from_payload"


def is_reconstructed(source_type: object, completion_status: object) -> bool:
    """The one predicate deciding which blockers, controls, and writers apply.

    Both halves are required, and it lives in this leaf module so the readiness
    service, the persistence writers, and the UI cannot drift to different
    versions of it. They did: ``acknowledge_accounting_assumption`` refused every
    ``source_type == 'manual'`` row while readiness emitted the blocker it clears
    for a manual row whose completion status was anything but
    ``not_applicable``, so the control reported success for a write it had
    discarded and the blocker could never clear.
    """
    return not (source_type == "manual" and completion_status == "not_applicable")


@dataclass(frozen=True)
class BoundaryEvidence:
    """What was observed immediately before or after this hand."""

    kind: str = ""  # "" unknown | hand_start | hand_end
    # | recording_start | recording_end | sampling_gap
    timestamp_s: float | None = None  # seconds into the source recording
    frame_ref: str = ""  # data-dir-relative image path
    confidence: float | None = None  # 0.0-1.0
    codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompletionEvidence:
    """Everything the pipeline can prove about one reconstructed hand's boundaries."""

    evidence_version: int = 0  # 0 == unknown/legacy; >=1 == readable
    partial_start: bool | None = None
    partial_end: bool | None = None
    # "" | showdown | fold_win | hero_fold | unobserved. Only the values in
    # OBSERVED_TERMINAL_EVENTS count as observed; everything else fails closed.
    terminal_event: str = ""
    first_source_timestamp_s: float | None = None
    last_source_timestamp_s: float | None = None
    preceding_boundary: BoundaryEvidence = field(default_factory=BoundaryEvidence)
    following_boundary: BoundaryEvidence = field(default_factory=BoundaryEvidence)
    boundary_confidence: float | None = None
    source_frames: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()
    rejection_codes: tuple[str, ...] = ()
    acknowledged_codes: tuple[str, ...] = ()
    # A settlement-assumption attestation is NOT a pipeline acknowledgement, and
    # it has its own channel so that no control which clears one can ever clear
    # the other. It used to share ``warning_codes`` / ``acknowledged_codes``:
    # every generic "Acknowledge" button in the Source warnings panel therefore
    # offered it, one press cleared ACCOUNTING_ASSUMPTION_DEPENDENT, and an
    # ordinary export/import round trip was enough to put a legitimately attested
    # code back into the acknowledgeable set. ``parse_completion_evidence``
    # enforces the separation in both directions on every read, so no writer, no
    # payload, and no hand-edited row can mix them.
    confirmed_assumption_codes: tuple[str, ...] = ()
    # The operator's own settlement declarations that move chips on this hand.
    # A third channel for a third kind of claim: ``warning_codes`` is what the
    # PIPELINE could not prove, ``confirmed_assumption_codes`` is what the
    # OPERATOR attested to, and this is what the operator DECLARED. Sharing the
    # first of those made a rake declaration demote the reconstruction's own
    # completion status and be reported as a pipeline finding.
    declared_settlement_codes: tuple[str, ...] = ()
    layout_profile: str = ""
    layout_supported: bool | None = None
    table_size: int | None = None
    pipeline_version: str = ""
    model_versions: dict[str, str] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def is_known(self) -> bool:
        """True only for an evidence version this build can read end to end."""
        return 1 <= self.evidence_version <= EVIDENCE_SCHEMA_VERSION

    @property
    def claims_reconstruction(self) -> bool:
        """True when this blob declares ANY pipeline evidence version, readable or not.

        The question "does this hand carry reconstruction evidence?" is a
        question about the CLAIM, not about whether this build can read it. It
        used to be asked as ``is_known`` (1..1), so a manual-declaring payload
        that bumped ``evidence_version`` to 2 defeated import's manual-payload
        refusal while KEEPING the pipeline's rejection codes in the stored row
        -- the hand landed exempt from every reconstructed-hand blocker with
        ``board_unreadable`` sitting in its own completion evidence. A nonzero
        version is a reconstruction claim whatever its number; only a hand the
        pipeline never touched carries none.
        """
        return self.evidence_version != 0

    @property
    def is_imported(self) -> bool:
        """True when this hand arrived in an import payload rather than being entered here."""
        return self.extra.get(IMPORTED_HAND_KEY) is True

    @property
    def unresolved_codes(self) -> tuple[str, ...]:
        """Warning and rejection codes the user has not explicitly accepted.

        Every rejection code is permanently unresolved: ``acknowledge_codes``
        refuses to accept one, so a rejection can only disappear by the pipeline
        producing new evidence without it.

        This mixes the two kinds, and a consumer that tells the operator what to DO
        about a code must not: an Acknowledge button clears a warning and is neither
        drawn nor honoured for a rejection. Say what a code IS with
        ``unresolved_warning_codes`` / ``unresolved_rejection_codes``; use this one
        only to count or to ask "is anything outstanding?".
        ``test_no_consumer_prescribes_an_action_from_unresolved_codes`` enforces
        that split.
        """
        acknowledged = set(self.acknowledged_codes)
        return tuple(
            code
            for code in (*self.warning_codes, *self.rejection_codes)
            if code not in acknowledged
        )

    @property
    def unresolved_rejection_codes(self) -> tuple[str, ...]:
        """Outstanding codes that are pipeline REFUSALS, which nothing can accept.

        A rejection cannot be acknowledged (``acknowledge_codes`` drops it) and
        ``app.show_source_warning_controls`` draws no Acknowledge button for one, so
        any blocker that names Acknowledge for a rejection is naming an action the
        product cannot perform.
        """
        rejected = set(self.rejection_codes)
        return tuple(code for code in self.unresolved_codes if code in rejected)

    @property
    def unresolved_warning_codes(self) -> tuple[str, ...]:
        """Outstanding codes the operator CAN accept in the Source warnings panel."""
        rejected = set(self.rejection_codes)
        return tuple(code for code in self.unresolved_codes if code not in rejected)


_RECOGNIZED_KEYS = frozenset(
    name for name in (item.name for item in fields(CompletionEvidence)) if name != "extra"
)
_BOUNDARY_KEYS = frozenset(item.name for item in fields(BoundaryEvidence))


def dump_completion_evidence(evidence: CompletionEvidence) -> dict[str, object]:
    """Return a JSON-primitive-only mapping suitable for json.dumps and _dump_model.

    Only ``dict``, ``list``, ``str``, ``int``, ``float``, ``bool`` and ``None``
    may appear. A nested ``datetime`` would only fail at download time inside the
    UI, because ``import_export._dump_model`` isoformats top-level values only.
    """
    payload: dict[str, object] = dict(_as_str_keyed_map(evidence.extra))
    payload.update(
        {
            "evidence_version": int(evidence.evidence_version),
            "partial_start": evidence.partial_start,
            "partial_end": evidence.partial_end,
            "terminal_event": evidence.terminal_event,
            "first_source_timestamp_s": evidence.first_source_timestamp_s,
            "last_source_timestamp_s": evidence.last_source_timestamp_s,
            "preceding_boundary": _dump_boundary(evidence.preceding_boundary),
            "following_boundary": _dump_boundary(evidence.following_boundary),
            "boundary_confidence": evidence.boundary_confidence,
            "source_frames": list(evidence.source_frames),
            "warning_codes": list(evidence.warning_codes),
            "rejection_codes": list(evidence.rejection_codes),
            "acknowledged_codes": list(evidence.acknowledged_codes),
            "confirmed_assumption_codes": list(evidence.confirmed_assumption_codes),
            "declared_settlement_codes": list(evidence.declared_settlement_codes),
            "layout_profile": evidence.layout_profile,
            "layout_supported": evidence.layout_supported,
            "table_size": evidence.table_size,
            "pipeline_version": evidence.pipeline_version,
            "model_versions": {
                str(key): str(value) for key, value in evidence.model_versions.items()
            },
        }
    )
    return payload


def strip_derived_evidence_markers(value: object) -> dict[str, object]:
    """Drop the read-time annotations no writer may persist.

    ``UNREADABLE_CARDS_KEY`` is injected by ``db._degrade_unreadable_cards`` when
    a hand-edited card column cannot be read back. It describes the CURRENT
    contents of a column, so it is a derivation, not evidence — but every writer
    that round-tripped a fetched hand's evidence stored it: the Acknowledge
    button, and ``import_session``. Once stored it was permanent, because
    ``_unreadable_card_columns`` is consulted before the live column and no writer
    removed it, so the hand carried INVALID_HERO_OR_BOARD_CARDS forever while its
    board was valid — and the blocker's stated clearing action did nothing.

    Stripping it on write keeps it purely derived: it appears the moment the
    column is unreadable and disappears the moment it is corrected.
    """
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if str(key) not in DERIVED_EVIDENCE_KEYS
    }


def parse_completion_evidence(value: object) -> CompletionEvidence:
    """Never raises. Missing, legacy, or corrupt input yields all-unknown evidence."""
    data = _as_object(value)
    if data is None:
        return CompletionEvidence()
    extra = {
        str(key): item for key, item in data.items() if str(key) not in _RECOGNIZED_KEYS
    }
    return CompletionEvidence(
        evidence_version=_as_int_or_none(data.get("evidence_version")) or 0,
        partial_start=_as_bool_or_none(data.get("partial_start")),
        partial_end=_as_bool_or_none(data.get("partial_end")),
        terminal_event=_as_str(data.get("terminal_event")),
        first_source_timestamp_s=_as_float_or_none(data.get("first_source_timestamp_s")),
        last_source_timestamp_s=_as_float_or_none(data.get("last_source_timestamp_s")),
        preceding_boundary=_as_boundary(data.get("preceding_boundary")),
        following_boundary=_as_boundary(data.get("following_boundary")),
        boundary_confidence=_as_float_or_none(data.get("boundary_confidence")),
        source_frames=_as_str_tuple(data.get("source_frames")),
        # The channel separation is enforced HERE, on every read, rather than in
        # each writer. A pipeline code channel carries neither a measured
        # settlement-assumption dependence nor an operator's own settlement
        # declaration, and neither of the other two channels carries anything
        # else, whatever an import payload, a legacy row, or a hand-edited
        # database put there. Nothing is lost by dropping a forged dependence
        # code: the dependence itself is re-measured from the chips on every read
        # (``hand_accounting._derive_assumption_dependence``), so a string in a
        # column cannot create one and cannot answer one. A misfiled DECLARED
        # code is relocated rather than dropped -- it is an audit record with
        # nothing resting on it, and every database written before this channel
        # existed holds its declarations in ``warning_codes``.
        warning_codes=_pipeline_codes(data.get("warning_codes")),
        rejection_codes=_pipeline_codes(data.get("rejection_codes")),
        acknowledged_codes=_pipeline_codes(data.get("acknowledged_codes")),
        confirmed_assumption_codes=_only_assumption_codes(
            data.get("confirmed_assumption_codes")
        ),
        declared_settlement_codes=_declared_settlement_codes(data),
        layout_profile=_as_str(data.get("layout_profile")),
        layout_supported=_as_bool_or_none(data.get("layout_supported")),
        table_size=_as_int_or_none(data.get("table_size")),
        pipeline_version=_as_str(data.get("pipeline_version")),
        model_versions=_as_str_map(data.get("model_versions")),
        extra=extra,
    )


def _pipeline_codes(value: object) -> tuple[str, ...]:
    """Only what the PIPELINE said: no attestation, no operator declaration."""
    return tuple(
        code
        for code in _as_str_tuple(value)
        if not is_assumption_dependence_code(code)
        and not is_declared_settlement_code(code)
    )


def _declared_settlement_codes(data: dict[str, Any]) -> tuple[str, ...]:
    """The operator's declarations, gathered from wherever a writer filed them."""
    seen: list[str] = []
    for key in (
        "declared_settlement_codes",
        "warning_codes",
        "acknowledged_codes",
        "rejection_codes",
    ):
        for code in _as_str_tuple(data.get(key)):
            if is_declared_settlement_code(code) and code not in seen:
                seen.append(code)
    return tuple(seen)


def _only_assumption_codes(value: object) -> tuple[str, ...]:
    return tuple(
        code for code in _as_str_tuple(value) if is_assumption_dependence_code(code)
    )


def derive_completion_status(
    evidence: CompletionEvidence, *, source_type: str
) -> CompletionStatus:
    """The only promotion path to ``complete``; it can never un-truncate a hand."""
    if source_type == "manual":
        return "not_applicable"
    if not evidence.is_known:
        return "uncertain"
    if evidence.partial_start is True or evidence.partial_end is True:
        return "partial"  # sticky: no correction can undo truncation
    if evidence.partial_start is None or evidence.partial_end is None:
        return "uncertain"
    if evidence.terminal_event not in OBSERVED_TERMINAL_EVENTS:
        # Allow-list, never deny-list: an unrecognised terminal event is not an
        # observed one, whatever produced it. See OBSERVED_TERMINAL_EVENTS.
        return "uncertain"
    if (
        evidence.boundary_confidence is None
        or not 0.0 < evidence.boundary_confidence <= 1.0
    ):
        # A presence check alone accepted 0.0, -3.0 and 42.0 as proof that the
        # boundaries were observed. The field is documented as 0.0-1.0; a value
        # outside that range can only come from a producer bug, a hand-edited
        # row, or a foreign payload, and a confidence of exactly 0.0 is the
        # pipeline stating it has NO confidence in the boundary at all. None of
        # those are a measurement that satisfies this gate, so they fail closed
        # the same way NaN (unreadable, parsed to None) already did.
        return "uncertain"
    if evidence.rejection_codes:
        # A rejection code is the pipeline refusing the hand, not a note the user
        # may accept. It is checked before unresolved_codes so that a hand-edited
        # acknowledged_codes list cannot launder one into a promotion.
        return "uncertain"
    if evidence.unresolved_codes:
        return "uncertain"
    return "complete"


def acknowledge_codes(
    evidence: CompletionEvidence, codes: object
) -> CompletionEvidence:
    """Return a copy with warning ``codes`` added to the accepted set, order preserved.

    Only ``warning_codes`` are acknowledgeable. A rejection code means the
    pipeline refused the hand, so accepting it would promote a hand no fact was
    corrected on; it clears only by re-running the reconstruction. A measured
    settlement-assumption dependence is not acknowledgeable here either, and
    cannot reach ``warning_codes`` in the first place: it is answered by
    ``db.acknowledge_accounting_assumption``, whose control states the chip
    movement being attested to.
    """
    acknowledgeable = {
        code for code in evidence.warning_codes if not is_assumption_dependence_code(code)
    }
    accepted = list(evidence.acknowledged_codes)
    for code in _as_str_tuple(codes):
        if code in acknowledgeable and code not in accepted:
            accepted.append(code)
    return _replace_acknowledged(evidence, tuple(accepted))


def requires_assumption_attestation(
    *, source_type: object, completion_status: object, evidence: CompletionEvidence
) -> bool:
    """Must THIS operator attest to the declared settlement assumptions of this hand?

    Yes for every reconstructed hand: the recording is the claim, and a
    declaration the recording does not support outranks it only if someone says
    so. Yes for every imported hand, whatever it declares its ``source_type`` to
    be: the manual exemption is the argument "you entered this hand yourself",
    which is false for a hand that arrived as user-supplied JSON. No for a hand
    typed into this database by its own operator, where a declared ante, dead
    blind, or rake is that operator's own observation.

    One predicate, consulted by the readiness blocker, by the control that clears
    it, and by the writer behind that control, so none of them can be scoped
    differently from the others.
    """
    return is_reconstructed(source_type, completion_status) or evidence.is_imported


def requires_user_confirmation(
    *, source_type: object, completion_status: object, evidence: CompletionEvidence
) -> bool:
    """Must THIS operator confirm the whole hand before it can be study-ready?

    The same argument, and deliberately the same predicate, as
    ``requires_assumption_attestation``: a hand entered in this database was
    already vouched for by the person who typed it, and a hand that was not --
    every reconstructed hand, and every hand that arrived as user-supplied JSON,
    whatever ``source_type`` it declares -- owes this operator's explicit
    confirmation. It used to be scoped on ``is_reconstructed`` alone, so an
    import payload relabelled ``manual`` with blank evidence bypassed
    USER_CONFIRMATION_MISSING together with every other reconstructed-hand
    blocker and landed study-ready straight from a JSON file.
    """
    return requires_assumption_attestation(
        source_type=source_type,
        completion_status=completion_status,
        evidence=evidence,
    )


def confirm_assumption(
    evidence: CompletionEvidence, code: str
) -> CompletionEvidence | None:
    """Record one measured settlement-assumption attestation; None when nothing changes.

    At most one attestation survives per declared input: a fresh measurement
    REPLACES the previous measurement of the same input instead of leaving two
    contradictory chip figures in the evidence, and a stale attestation must not
    stay behind to re-clear the blocker if the hand is later corrected back.
    """
    if not is_assumption_dependence_code(code):
        raise ValueError(
            "Only a measured settlement-assumption dependence code can be "
            f"confirmed; got {code!r}."
        )
    if code in evidence.confirmed_assumption_codes:
        return None
    replaced = assumption_dependence_input(code)
    kept = tuple(
        item
        for item in evidence.confirmed_assumption_codes
        if assumption_dependence_input(item) != replaced
    )
    values = {item.name: getattr(evidence, item.name) for item in fields(CompletionEvidence)}
    values["confirmed_assumption_codes"] = (*kept, code)
    return CompletionEvidence(**values)


def set_declared_settlement_code(
    evidence: CompletionEvidence, code: str, *, declared: bool
) -> CompletionEvidence | None:
    """Record or withdraw one declared-settlement disclosure; None when unchanged.

    Idempotent by construction: the record says "this hand declares a settlement
    input that moves chips", and how MANY chips lives in the measured dependence
    code, which is the thing an attestation is bound to. Nothing here is
    acknowledgeable and nothing here moves ``completion_status``.
    """
    if not is_declared_settlement_code(code):
        raise ValueError(
            "Only a declared-settlement disclosure code can be recorded here; "
            f"got {code!r}."
        )
    present = code in evidence.declared_settlement_codes
    if present == declared:
        return None
    codes = (
        (*evidence.declared_settlement_codes, code)
        if declared
        else tuple(item for item in evidence.declared_settlement_codes if item != code)
    )
    values = {item.name: getattr(evidence, item.name) for item in fields(CompletionEvidence)}
    values["declared_settlement_codes"] = codes
    return CompletionEvidence(**values)


def _replace_acknowledged(
    evidence: CompletionEvidence, acknowledged: tuple[str, ...]
) -> CompletionEvidence:
    values = {item.name: getattr(evidence, item.name) for item in fields(CompletionEvidence)}
    values["acknowledged_codes"] = acknowledged
    return CompletionEvidence(**values)


def _dump_boundary(boundary: BoundaryEvidence) -> dict[str, object]:
    return {
        "kind": boundary.kind,
        "timestamp_s": boundary.timestamp_s,
        "frame_ref": boundary.frame_ref,
        "confidence": boundary.confidence,
        "codes": list(boundary.codes),
    }


def _as_object(value: object) -> dict[str, Any] | None:
    """Coerce a stored blob to a JSON object, or None when it is unreadable."""
    if isinstance(value, bytes | bytearray | memoryview):
        # A BLOB written into the TEXT column by a hand-edited database. Decoding
        # it would raise UnicodeDecodeError (a ValueError, but raised before
        # json.loads sees it), so it is rejected here as simply unreadable.
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            # RecursionError inherits from RuntimeError, not ValueError, so a
            # deeply nested blob written by a hand-edited database used to escape
            # this parser -- the same threat model the BLOB branch above exists
            # for -- and take the whole session's hand list down with it.
            return None
    if isinstance(value, dict) and value:
        return {str(key): item for key, item in value.items()}
    return None


def _as_boundary(value: object) -> BoundaryEvidence:
    if not isinstance(value, dict):
        return BoundaryEvidence()
    data = {str(key): item for key, item in value.items() if str(key) in _BOUNDARY_KEYS}
    return BoundaryEvidence(
        kind=_as_str(data.get("kind")),
        timestamp_s=_as_float_or_none(data.get("timestamp_s")),
        frame_ref=_as_str(data.get("frame_ref")),
        confidence=_as_float_or_none(data.get("confidence")),
        codes=_as_str_tuple(data.get("codes")),
    )


def _as_int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, float) and not value.is_integer():
        # Truncating 1.9 to 1 would let a version this build never wrote pass the
        # is_known gate. A non-integral value is unreadable, not "close enough".
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        # NaN and +/-Infinity are not measurements. They also cannot be written to
        # RFC 8259 JSON, so a blob carrying one produced an export that Python's
        # lenient reader accepted and no other consumer could parse. Treated the
        # same way a non-integral evidence_version is: unreadable, not
        # "close enough" -- which makes derive_completion_status return
        # 'uncertain' rather than promoting on a confidence of NaN.
        return None
    return number


def _as_bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _as_str_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item for key, item in value.items() if isinstance(item, str)
    }


def _as_str_keyed_map(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}
