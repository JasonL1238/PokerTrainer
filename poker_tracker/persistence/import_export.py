from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from poker_tracker.persistence.completion import (
    DERIVED_EVIDENCE_KEYS,
    IMPORTED_HAND_KEY,
    OPERATOR_MANUAL_COMPLETION_KEY,
    derive_completion_status,
    parse_completion_evidence,
    strip_derived_evidence_markers,
)
from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import (
    Action,
    CoachingResponse,
    Hand,
    HandCorrection,
    HandIssue,
    HandPlayer,
    HandReview,
    HandSettlement,
    Session,
    SettlementEntry,
)

EXPORT_VERSION = 6
# The literal 4 must stay listed: bumping EXPORT_VERSION alone would silently
# drop support for payloads written by the previous release.
SUPPORTED_IMPORT_VERSIONS = {1, 2, 3, 4, 5, EXPORT_VERSION}


def export_hand(db: PokerDatabase, hand_id: int) -> dict[str, Any]:
    """Export one hand and its related rows as JSON-compatible data."""
    hand = db.fetch_hand(hand_id)
    if hand is None:
        raise ValueError(f"Hand not found: {hand_id}")
    settlement = db.fetch_hand_settlement(hand_id)
    return {
        "export_version": EXPORT_VERSION,
        "hand": _dump_model(hand),
        "players": [_dump_model(player) for player in db.fetch_players_by_hand(hand_id)],
        "actions": [_dump_model(action) for action in db.fetch_actions_by_hand(hand_id)],
        "settlement": None if settlement is None else _dump_model(settlement),
        "settlement_entries": [
            _dump_model(entry) for entry in db.fetch_settlement_entries(hand_id)
        ],
        "reviews": [_dump_model(review) for review in db.fetch_reviews_by_hand(hand_id)],
        "coaching_reviews": [
            _dump_model(review) for review in db.fetch_coaching_reviews_by_hand(hand_id)
        ],
        "corrections": [
            _dump_model(correction) for correction in db.fetch_hand_corrections(hand_id)
        ],
        "issues": [
            _dump_model(issue) for issue in db.fetch_hand_issues(hand_id=hand_id)
        ],
    }


def export_session(db: PokerDatabase, session_id: int) -> dict[str, Any]:
    """Export one full session with hands, players, actions, and reviews."""
    session = db.fetch_session(session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")
    return {
        "export_version": EXPORT_VERSION,
        "session": _dump_model(session),
        "hands": [
            export_hand(db, hand.id)
            for hand in db.fetch_hands_by_session(session_id)
            if hand.id is not None
        ],
        "coaching_reviews": [
            _dump_model(review)
            for review in db.fetch_coaching_reviews_by_session(session_id)
        ],
    }


def export_session_json(db: PokerDatabase, session_id: int, path: str | Path) -> None:
    Path(path).write_text(json.dumps(export_session(db, session_id), indent=2), encoding="utf-8")


def import_session(db: PokerDatabase, payload: dict[str, Any]) -> Session:
    """Import a previously exported session into the current database."""
    version = payload.get("export_version", 1)
    if version not in SUPPORTED_IMPORT_VERSIONS:
        raise ValueError(
            f"Unsupported export_version {version}; this app understands "
            f"{sorted(SUPPORTED_IMPORT_VERSIONS)}."
        )
    session_data = dict(payload["session"])
    session_data.pop("id", None)
    session_model = Session(**session_data)
    validated_hands: list[
        tuple[
            Hand,
            dict[str, object],
            list[HandPlayer],
            list[Action],
            HandSettlement | None,
            list[SettlementEntry],
            list[HandReview],
            list[CoachingResponse],
            list[HandCorrection],
            list[HandIssue],
        ]
    ] = []

    for hand_payload in payload.get("hands", []):
        hand_data = dict(hand_payload["hand"])
        hand_data.pop("id", None)
        hand_data["session_id"] = 0
        # Captured BEFORE the markers are stripped. They are the only surviving
        # record of a column the exporting database could not read, and without
        # them INVALID_HERO_OR_BOARD_CARDS and UNREADABLE_HAND_COLUMNS are
        # consumers whose producer -- the corrupt column, which the exporter has
        # already degraded to a fallback -- disappears across the round trip.
        unreadable = _recorded_unreadable_columns(hand_data.get("completion_evidence"))
        _apply_completion_import_defaults(hand_data)
        hand = Hand(**hand_data)

        players: list[HandPlayer] = []
        for player_data in hand_payload.get("players", []):
            imported = dict(player_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            players.append(HandPlayer(**imported))

        actions: list[Action] = []
        for action_data in hand_payload.get("actions", []):
            imported = dict(action_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            if "amount_semantics" not in imported:
                imported["amount_semantics"] = "unknown"
            actions.append(Action(**imported))
        _link_actions_to_players(actions, players)
        _normalize_duplicate_action_indexes(actions)

        settlement_data = hand_payload.get("settlement")
        settlement: HandSettlement | None = None
        if settlement_data is not None:
            imported_settlement = dict(settlement_data)
            imported_settlement["hand_id"] = 0
            settlement = HandSettlement(**imported_settlement)

        settlement_entries: list[SettlementEntry] = []
        for entry_data in hand_payload.get("settlement_entries", []):
            imported = dict(entry_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            settlement_entries.append(SettlementEntry(**imported))

        reviews: list[HandReview] = []
        for review_data in hand_payload.get("reviews", []):
            imported = dict(review_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            _mark_imported_analysis_stale(imported)
            reviews.append(HandReview(**imported))

        coaching_reviews: list[CoachingResponse] = []
        for review_data in hand_payload.get("coaching_reviews", []):
            imported = dict(review_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            imported["session_id"] = 0
            _mark_imported_analysis_stale(imported)
            coaching_reviews.append(CoachingResponse(**imported))

        corrections: list[HandCorrection] = []
        for correction_data in hand_payload.get("corrections", []):
            imported = dict(correction_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            corrections.append(HandCorrection(**imported))

        issues: list[HandIssue] = []
        for issue_data in hand_payload.get("issues", []):
            imported = dict(issue_data)
            imported.pop("id", None)
            imported["hand_id"] = 0
            _mark_imported_issue_unresolved(imported)
            issues.append(HandIssue(**imported))

        validated_hands.append(
            (
                hand,
                unreadable,
                players,
                actions,
                settlement,
                settlement_entries,
                reviews,
                coaching_reviews,
                corrections,
                issues,
            )
        )

    session_coaching_reviews: list[CoachingResponse] = []
    for review_data in payload.get("coaching_reviews", []):
        imported = dict(review_data)
        imported.pop("id", None)
        imported["hand_id"] = None
        imported["session_id"] = 0
        _mark_imported_analysis_stale(imported)
        session_coaching_reviews.append(CoachingResponse(**imported))

    with db.transaction():
        session = db.create_session(session_model)
        if session.id is None:
            raise RuntimeError("Imported session did not receive an id.")
        for (
            hand,
            unreadable,
            players,
            actions,
            settlement,
            settlement_entries,
            reviews,
            coaching_reviews,
            corrections,
            issues,
        ) in validated_hands:
            saved_hand = db.create_hand(hand.model_copy(update={"session_id": session.id}))
            if saved_hand.id is None:
                raise RuntimeError("Imported hand did not receive an id.")
            if unreadable:
                db.restore_unreadable_columns(saved_hand.id, unreadable)
            for player in players:
                db.create_hand_player(player.model_copy(update={"hand_id": saved_hand.id}))
            for action in actions:
                db.create_action(action.model_copy(update={"hand_id": saved_hand.id}))
            if settlement is not None:
                db.upsert_hand_settlement(settlement.model_copy(update={"hand_id": saved_hand.id}))
            if settlement_entries:
                # Never nested under `settlement is not None`: a settlement row is
                # not required to declare a winner (`create_settlement_entry` needs
                # none, and the exporter emits the entries regardless), so nesting
                # silently dropped every award on a settled, balanced hand and left
                # it unsettled with no error and no reported count.
                db.replace_settlement_entries(
                    saved_hand.id,
                    [
                        entry.model_copy(update={"hand_id": saved_hand.id})
                        for entry in settlement_entries
                    ],
                )
            for review in reviews:
                db.create_hand_review(review.model_copy(update={"hand_id": saved_hand.id}))
            for review in coaching_reviews:
                db.create_coaching_response(
                    review.model_copy(
                        update={"hand_id": saved_hand.id, "session_id": session.id}
                    )
                )
            for correction in corrections:
                db.create_hand_correction(
                    correction.model_copy(update={"hand_id": saved_hand.id})
                )
            for issue in issues:
                db.create_hand_issue(
                    issue.model_copy(update={"hand_id": saved_hand.id}),
                    apply_workflow=False,
                )
            _enforce_review_status_floor(db, saved_hand.id, hand.review_status)
        for review in session_coaching_reviews:
            db.create_coaching_response(
                review.model_copy(update={"session_id": session.id})
            )

    return session


def import_hands_into_session(
    db: PokerDatabase,
    payload: dict[str, Any],
    session_id: int,
) -> Session:
    """Append an imported payload's hands to an existing session.

    Imported hand numbers are preserved when available. Collisions are assigned
    the next number in the target session, keeping every hand addressable.
    """

    target = db.fetch_session(session_id)
    if target is None:
        raise ValueError(f"Session not found: {session_id}")

    with db.transaction():
        temporary = import_session(db, payload)
        if temporary.id is None:
            raise RuntimeError("Imported session was not persisted.")
        for hand in db.fetch_hands_by_session(temporary.id):
            if hand.id is not None:
                db.move_hand_to_session(hand.id, session_id)
        # move_hand_to_session re-parents review_type='hand' rows only. Without
        # this, coaching_reviews.session_id ON DELETE CASCADE silently deleted
        # every session-level coaching review the payload carried, with no error
        # and no reported count.
        db.move_session_coaching_reviews(temporary.id, session_id)
        db.delete_session(temporary.id)

    refreshed = db.fetch_session(session_id)
    if refreshed is None:
        raise RuntimeError("Target session could not be reloaded after import.")
    return refreshed


IMPORTED_ANALYSIS_STALE_REASON = (
    "Imported from another database; rerun coaching against these records."
)


def _mark_imported_analysis_stale(review_data: dict[str, Any]) -> None:
    """Land every imported coaching row stale, whatever the payload declared.

    ``is_stale`` is a blocker input -- ``STALE_COACHING_EVIDENCE`` exists to stop
    a review written against superseded facts being presented as CURRENT -- and
    ``HandReview(**imported)`` / ``CoachingResponse(**imported)`` accepted it
    verbatim, so editing one boolean in an exported JSON file cleared the
    blocker and republished stale coaching as current.

    Nothing in the importing database can verify the claim: a retained review
    describes the hand, ledger and winners of the database that produced it, and
    what arrives here are different rows with different ids. This is the same
    rule that already applies to an acknowledgement and to a ``reviewed``
    promotion -- an assertion about evidence cannot travel in the payload that
    carries the evidence. The text itself is preserved, so nothing is lost: it is
    re-run in the importing database.
    """
    review_data["is_stale"] = True
    if not str(review_data.get("stale_reason") or "").strip():
        review_data["stale_reason"] = IMPORTED_ANALYSIS_STALE_REASON


IMPORTED_ISSUE_REOPEN_NOTE = (
    "Reopened on import; the resolution was recorded in another database. "
    "Previously recorded resolution: "
)


def _mark_imported_issue_unresolved(issue_data: dict[str, Any]) -> None:
    """Land every imported debugging issue OPEN, whatever the payload declared.

    Exactly the rule ``_mark_imported_analysis_stale`` applies to coaching, that
    ``_enforce_review_status_floor`` applies to ``reviewed``, and that
    ``_apply_completion_import_defaults`` applies to ``acknowledged_codes``,
    ``confirmed_assumption_codes`` and the declared ``completion_status``: an
    assertion about evidence cannot travel in the payload that carries the
    evidence. A resolution says "somebody looked at this hand and fixed the thing"
    -- ``resolve_hand_issue`` demands notes for exactly that reason -- and the
    importing operator has looked at nothing.

    Changing one JSON field from ``"open"`` to ``"resolved"`` cleared
    OPEN_DEBUGGING_ISSUE, landed the hand study-ready with an empty blocker tuple,
    and let ``db.update_hand_status`` promote it to ``reviewed`` -- in a state
    ``resolve_hand_issue`` refuses to create, because the same edit left
    ``resolution_notes`` empty and ``resolved_at`` null. Requiring those two
    fields instead would only have moved the forgery two fields along; the
    resolution is not verifiable here at any level of detail.

    Nothing is discarded. The description, the issue types and the immutable
    evidence snapshot travel unchanged, and any resolution notes the exporting
    database recorded are carried into the reopened issue's description, so the
    importing operator can re-resolve it knowing what was done before.
    """
    if issue_data.get("status") == "open":
        issue_data["resolution_notes"] = ""
        issue_data["resolved_at"] = None
        return
    issue_data["status"] = "open"
    issue_data["resolved_at"] = None
    notes = str(issue_data.pop("resolution_notes", "") or "").strip()
    issue_data["resolution_notes"] = ""
    if notes:
        description = str(issue_data.get("description") or "").strip()
        issue_data["description"] = (
            f"{description}\n\n{IMPORTED_ISSUE_REOPEN_NOTE}{notes}".strip()
        )


def _enforce_review_status_floor(
    db: PokerDatabase, hand_id: int, declared_status: str
) -> None:
    """No hand lands 'reviewed' from a payload, whatever source_type it declares.

    ``create_hand`` writes the declared review_status verbatim, so this runs
    after the hand's rows exist and demotes what a payload may not claim.
    ``reviewed`` is this database operator's attestation: readiness requires
    explicit user confirmation, which is derived per render and deliberately
    never persisted, so it cannot travel in a payload -- the importing operator
    has not seen this hand. That argument does not depend on ``source_type``,
    and scoping the floor on it was the hole: a payload relabelled
    ``source_type: manual`` was re-promoted straight to ``reviewed`` by this
    very function, on the strength of two strings the same payload wrote. The
    same rule already applies to every other travelling assertion -- imported
    coaching lands stale, acknowledgements and attestations are reset.

    A genuine manual export loses its label too, because it is byte-identical to
    the forgery and the label is one tick and one save away for the operator who
    now vouches for it. (The v13 MIGRATION keeps manual review statuses -- a
    migrated database is the same operator's own data, not somebody's JSON.)
    """

    if declared_status != "reviewed":
        return
    db.update_hand_status(hand_id, "needs_correction")


# How restrictive each reconstructed-hand completion status is. Import may move a
# hand up this ordering (weaken its claim) and never down it. `not_applicable` is
# deliberately absent: it is the manual-hand exemption, is refused outright on a
# reconstructed payload above, and is not comparable with the rest.
_IMPORT_COMPLETION_RESTRICTION = {"complete": 0, "uncertain": 1, "partial": 2}


def _apply_completion_import_defaults(hand_data: dict[str, Any]) -> None:
    """Re-derive one imported hand's completion status from its own evidence.

    Runs before any application-data write so a rejected payload leaves no
    partial session behind. Version-independent by design: a v5 payload's
    declared completion_status is exactly as untrusted as a pre-v5 payload's
    absent one, because both arrive as user-supplied JSON.
    """

    source = hand_data.get("source_type") or "manual"
    if source != "manual" and hand_data.get("completion_status") == "not_applicable":
        raise ValueError(
            "Imported hand declares a reconstructed source_type with "
            "completion_status 'not_applicable'."
        )
    # source_type is the one field the whole completion invariant hangs on: the
    # 'manual' + 'not_applicable' pair exempts a hand from every completion,
    # layout, source-warning, and confirmation blocker. A payload that declares
    # 'manual' while carrying reconstruction evidence is claiming the exemption
    # for a hand the pipeline built, so it is refused rather than trusted.
    # `claims_reconstruction` (any nonzero evidence_version), NOT `is_known`:
    # gating the refusal on readability meant a payload that bumped
    # evidence_version past what this build reads defeated it while KEEPING the
    # pipeline's rejection codes in the stored row.
    if source == "manual" and parse_completion_evidence(
        hand_data.get("completion_evidence")
    ).claims_reconstruction:
        raise ValueError(
            "Imported hand declares source_type 'manual' but carries "
            "reconstruction completion evidence."
        )
    evidence = hand_data.get("completion_evidence")
    evidence = strip_derived_evidence_markers(evidence)
    # An acknowledgement is an operator attesting to a code in THIS database, and
    # it is an input to `derive_completion_status` through `unresolved_codes`. A
    # payload that declared a warning AND acknowledged it was internally
    # consistent, so the ceiling below could not see it: the hand derived
    # 'complete' with an empty blocker tuple, having been attested to by nobody,
    # and a genuine `hero_seat_mismatch` arriving pre-acknowledged silenced
    # UNSUPPORTED_TABLE_LAYOUT while app.py drew no Acknowledge control for it.
    # Stripped for exactly the reason `reviewed` is never landed on a reconstructed
    # hand: the importing operator has not seen this hand's evidence. The codes
    # themselves are preserved, so nothing is lost -- they are re-acknowledged in
    # the importing database. Reset rather than removed, so the stored blob keeps
    # the shape a v5 export writes.
    if "acknowledged_codes" in evidence:
        evidence["acknowledged_codes"] = []
    # A settlement-assumption attestation is the same kind of statement, made
    # about chips instead of pipeline codes: it says THIS operator asserts these
    # unobserved chips were really taken or added. The importing operator has
    # asserted nothing, so it is reset here too, and the dependence -- which is
    # re-measured from the chips on every read -- simply reappears until they do.
    if "confirmed_assumption_codes" in evidence:
        evidence["confirmed_assumption_codes"] = []
    # Operator finalize attestation is the same class of statement: it says THIS
    # operator completed a truncated draft. A payload must not carry that claim
    # into a foreign database and launder sticky partial into complete.
    evidence.pop(OPERATOR_MANUAL_COMPLETION_KEY, None)
    evidence.pop("operator_terminal_event", None)
    # Study inclusion is a local operator preference, not a transferable fact.
    # Reset to auto so a forged "skip"/"study" is not attributed to the importer.
    hand_data["study_inclusion"] = "auto"
    # Stamped on every imported hand, whatever it declares its source_type to be.
    # The manual exemption is the argument "you entered this hand yourself, so a
    # declared ante or rake is your own observation", and that argument is false
    # for a hand that arrived as user-supplied JSON. A payload declaring
    # `source_type: manual` with no evidence is byte-identical to a genuine
    # manual export, so no guard can disprove the claim -- but it does not have
    # to: what the payload cannot manufacture is having been entered here. This
    # marker carries no `evidence_version`, so it does not make a manual hand
    # "carry reconstruction evidence" and the refusal above is unaffected.
    evidence[IMPORTED_HAND_KEY] = True
    hand_data["completion_evidence"] = evidence
    # The declared completion_status is never trusted, at any export version: a
    # payload is user-supplied JSON and reaches create_hand, which has no guard.
    # derive_completion_status is the only promotion path to 'complete', and a
    # pre-v5 payload carries no evidence, so it derives 'uncertain' for a
    # reconstructed hand and 'not_applicable' for a manual one -- the same
    # conservative defaults the previous version-keyed branch applied.
    derived = derive_completion_status(
        parse_completion_evidence(hand_data["completion_evidence"]),
        source_type=source,
    )
    # ...and it is never trusted upward either. Both the declared status and the
    # evidence it is re-derived from arrive in the same user-supplied JSON, so a
    # forger who writes a CONSISTENT evidence blob would otherwise win. Import
    # applies safe defaults: it may only ever weaken what the payload claimed.
    #
    # The comparison runs over the whole ordering, not just the `complete` step.
    # Keying it on `derived == "complete"` silently swapped a declared, permanent
    # `partial` for the re-derived, acknowledgeable `uncertain` -- which is what
    # every stripped, corrupt, pre-v5 or future-`evidence_version` payload derives
    # -- and one Acknowledge click then walked the hand to `complete`.
    declared = hand_data.get("completion_status")
    if (
        declared in _IMPORT_COMPLETION_RESTRICTION
        and derived in _IMPORT_COMPLETION_RESTRICTION
        and _IMPORT_COMPLETION_RESTRICTION[declared]
        > _IMPORT_COMPLETION_RESTRICTION[derived]
    ):
        derived = declared
    hand_data["completion_status"] = derived
    # Redundant in outcome, and deliberately kept as a local invariant rather than
    # advertised as the guard. `_enforce_review_status_floor` is where the
    # protection actually lives: it unconditionally writes 'needs_correction' for
    # every declared 'reviewed' a few lines later, inside the same transaction,
    # whatever source_type the payload claims, and removing IT is killed by
    # tests. Removing this line changes no observable outcome; it states here,
    # at the point the status is decided, that an unproven hand may not declare
    # itself reviewed in a JSON file.
    if (
        hand_data.get("completion_status") not in {"complete", "not_applicable"}
        and hand_data.get("review_status") == "reviewed"
    ):
        hand_data["review_status"] = "needs_correction"


def _recorded_unreadable_columns(evidence: object) -> dict[str, object]:
    """Every column the exporting database recorded as unreadable, from EVERY marker.

    Keyed on ``DERIVED_EVIDENCE_KEYS`` rather than on ``UNREADABLE_CARDS_KEY``
    alone. The card marker was restored from round 5 and the hand-column marker,
    added in rounds 12-13, was not, so an ordinary export/import round trip
    stripped UNREADABLE_HAND_COLUMNS, destroyed the recorded values, recorded no
    correction, and landed the hand study-ready with an empty blocker tuple --
    while the card half of the same mechanism survived intact. Reading the marker
    SET means a marker added later is restored without anyone editing this
    function.
    """
    if not isinstance(evidence, dict):
        return {}
    columns: dict[str, object] = {}
    for key in sorted(DERIVED_EVIDENCE_KEYS):
        recorded = evidence.get(key)
        if not isinstance(recorded, dict):
            continue
        for column, value in recorded.items():
            columns.setdefault(str(column), value)
    return columns


def _dump_model(model: Any) -> dict[str, Any]:
    data = model.model_dump()
    for key, value in list(data.items()):
        if isinstance(value, (date, datetime)):
            data[key] = value.isoformat()
    return data


def _link_actions_to_players(actions: list[Action], players: list[HandPlayer]) -> None:
    """Attach imported legacy actions only when identity resolution is unambiguous."""

    for index, action in enumerate(actions):
        if action.player_key is not None:
            continue
        candidates = [
            player
            for player in players
            if player.player_name == action.player_name
            and (not action.position or player.position == action.position)
        ]
        if len(candidates) != 1:
            candidates = [player for player in players if player.player_name == action.player_name]
        if len(candidates) == 1:
            actions[index] = action.model_copy(update={"player_key": candidates[0].player_key})


def _normalize_duplicate_action_indexes(actions: list[Action]) -> None:
    """Resolve ambiguous legacy/import order using stable payload order."""
    counts: dict[tuple[str, int], int] = {}
    duplicate_streets: set[str] = set()
    for action in actions:
        if action.action_index is None:
            continue
        key = (action.street, action.action_index)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > 1:
            duplicate_streets.add(action.street)
    if not duplicate_streets:
        return

    next_index = {street: 1 for street in duplicate_streets}
    for index, action in enumerate(actions):
        if action.street not in duplicate_streets:
            continue
        actions[index] = action.model_copy(update={"action_index": next_index[action.street]})
        next_index[action.street] += 1
