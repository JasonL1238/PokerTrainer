"""Session export and import.

Duplicate import is an APPEND, never an overwrite. ``import_session`` drops the
payload's session id and always calls ``db.create_session``: importing the same
file twice leaves two independent sessions, and no payload can address, replace
or edit a session that already exists. Appending to a session the operator
picked is the separate, explicit ``import_hands_into_session``. There is no
update path in this module, which is why silent overwrite is not merely
unlikely but unreachable.

Every claim a payload makes about its own trustworthiness is re-derived here
rather than believed: coaching lands stale, issues land open, ``reviewed`` lands
demoted, and completion status is recomputed from the evidence. See the
individual docstrings for why each one cannot travel in the file that carries
the evidence for it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

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
    CompletionStatus,
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

# Ceilings on the payload itself, enforced by import_session rather than by its
# callers. The equivalent 10 MB limit used to live only at the three Streamlit
# upload widgets, so the CV pipeline exporter and any future upload surface were
# bounded by nothing at all.
MAX_IMPORT_PAYLOAD_TEXT_CHARS = 10 * 1024 * 1024
MAX_IMPORT_PAYLOAD_VALUES = 2_000_000
MAX_IMPORT_PAYLOAD_DEPTH = 64

ModelT = TypeVar("ModelT", bound=BaseModel)


class ImportValidationError(ValueError):
    """A refused import payload, named down to the record that broke it.

    Subclasses ``ValueError`` because every import surface already funnels
    payload problems through ``except ValueError``; a new exception type would
    have turned refusals into unhandled tracebacks at call sites this module
    does not own. ``location`` is the path to the offending record --
    ``hands[2].players[1]`` -- so the message says WHICH record is wrong in a
    file the operator may have spent an hour producing, rather than that
    something somewhere in it was.
    """

    def __init__(self, location: str, problem: str) -> None:
        self.location = location
        self.problem = problem
        super().__init__(f"{location}: {problem}" if location else problem)


@dataclass(frozen=True)
class ValidatedHand:
    """One hand's records, already constructed and cross-checked, ready to write."""

    hand: Hand
    unreadable_columns: dict[str, object]
    players: list[HandPlayer]
    actions: list[Action]
    settlement: HandSettlement | None
    settlement_entries: list[SettlementEntry]
    reviews: list[HandReview]
    coaching_reviews: list[CoachingResponse]
    corrections: list[HandCorrection]
    issues: list[HandIssue]


@dataclass(frozen=True)
class ValidatedImport:
    """A whole payload, accepted. Everything the write pass is allowed to read."""

    export_version: int
    session: Session
    hands: list[ValidatedHand] = field(default_factory=list)
    session_coaching_reviews: list[CoachingResponse] = field(default_factory=list)


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


def validate_import_payload(payload: object) -> ValidatedImport:
    """Turn a payload into records, or refuse it, without touching any database.

    Every check the import performs lives here: ceilings, shape, field types,
    duplicate identifiers, and the relationships between a hand's records.
    ``import_session`` calls this once and its write block then reads nothing
    but the returned object, so "import validation completes before any
    application-data write" is a property of the call graph rather than a claim
    about statement order that the next edit can quietly invalidate.

    It used to be a claim, and it was false: uniqueness and relational problems
    were left to SQLite's constraints and to the writers' own guards, so a
    payload with two players sharing a ``player_key`` inserted the session and
    the hand and then raised ``sqlite3.IntegrityError`` -- an exception no
    import surface caught -- from the middle of the write pass. The transaction
    rolled it all back, so nothing was corrupted, but the operator got a
    traceback naming an index instead of a message naming their record.
    """

    _assert_within_ceilings(payload)
    root = _require_mapping(payload, "payload")
    version = root.get("export_version", 1)
    # The type test comes first because `version` is raw payload data: a hand
    # edit to `"export_version": []` made the set membership below raise
    # `TypeError: unhashable type: 'list'`. `bool` is excluded because it is an
    # `int` subclass, so `True` would otherwise read as version 1.
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in SUPPORTED_IMPORT_VERSIONS
    ):
        raise ImportValidationError(
            "",
            # Repr, so a payload declaring the string "6" is distinguishable from
            # one declaring the integer 6 in the message the operator reads.
            f"Unsupported export_version {version!r}; this app understands "
            f"{sorted(SUPPORTED_IMPORT_VERSIONS)}.",
        )
    if "session" not in root:
        raise ImportValidationError(
            "payload", "no 'session' object; this is not an exported session file."
        )
    session_data = dict(_require_mapping(root["session"], "session"))
    # The payload's own id is dropped, not honoured: an import creates a session,
    # it never addresses one. See the module docstring.
    session_data.pop("id", None)
    session_model = _build(Session, session_data, "session")

    validated_hands: list[ValidatedHand] = []
    hand_numbers: dict[int, int] = {}
    for index, hand_payload in enumerate(_require_sequence(root, "hands", "payload")):
        location = f"hands[{index}]"
        validated = _validate_hand(_require_mapping(hand_payload, location), location)
        previous = hand_numbers.setdefault(validated.hand.hand_number, index)
        if previous != index:
            raise ImportValidationError(
                location,
                f"hand_number {validated.hand.hand_number} is already used by "
                f"hands[{previous}]; a hand number addresses a hand within its "
                "session and cannot be shared.",
            )
        validated_hands.append(validated)

    session_coaching_reviews: list[CoachingResponse] = []
    for index, review_data in enumerate(
        _require_sequence(root, "coaching_reviews", "payload")
    ):
        location = f"coaching_reviews[{index}]"
        imported = dict(_require_mapping(review_data, location))
        imported.pop("id", None)
        imported["hand_id"] = None
        imported["session_id"] = 0
        _mark_imported_analysis_stale(imported)
        session_coaching_reviews.append(_build(CoachingResponse, imported, location))

    return ValidatedImport(
        export_version=int(version),
        session=session_model,
        hands=validated_hands,
        session_coaching_reviews=session_coaching_reviews,
    )


def import_session(db: PokerDatabase, payload: dict[str, Any]) -> Session:
    """Import a previously exported session as a NEW session.

    Never updates or replaces an existing session, whatever ids the payload
    carries; re-importing the same file appends a second copy. See the module
    docstring for why that is the whole of the duplicate-import contract.
    """
    validated = validate_import_payload(payload)

    with db.transaction():
        session = db.create_session(validated.session)
        if session.id is None:
            raise RuntimeError("Imported session did not receive an id.")
        for entry in validated.hands:
            saved_hand = db.create_hand(
                entry.hand.model_copy(update={"session_id": session.id})
            )
            if saved_hand.id is None:
                raise RuntimeError("Imported hand did not receive an id.")
            if entry.unreadable_columns:
                db.restore_unreadable_columns(saved_hand.id, entry.unreadable_columns)
            for player in entry.players:
                db.create_hand_player(player.model_copy(update={"hand_id": saved_hand.id}))
            for action in entry.actions:
                db.create_action(action.model_copy(update={"hand_id": saved_hand.id}))
            if entry.settlement is not None:
                db.upsert_hand_settlement(
                    entry.settlement.model_copy(update={"hand_id": saved_hand.id})
                )
            if entry.settlement_entries:
                # Never nested under `settlement is not None`: a settlement row is
                # not required to declare a winner (`create_settlement_entry` needs
                # none, and the exporter emits the entries regardless), so nesting
                # silently dropped every award on a settled, balanced hand and left
                # it unsettled with no error and no reported count.
                db.replace_settlement_entries(
                    saved_hand.id,
                    [
                        settlement_entry.model_copy(update={"hand_id": saved_hand.id})
                        for settlement_entry in entry.settlement_entries
                    ],
                )
            for review in entry.reviews:
                db.create_hand_review(review.model_copy(update={"hand_id": saved_hand.id}))
            for review in entry.coaching_reviews:
                db.create_coaching_response(
                    review.model_copy(
                        update={"hand_id": saved_hand.id, "session_id": session.id}
                    )
                )
            for correction in entry.corrections:
                db.create_hand_correction(
                    correction.model_copy(update={"hand_id": saved_hand.id})
                )
            for issue in entry.issues:
                db.create_hand_issue(
                    issue.model_copy(update={"hand_id": saved_hand.id}),
                    apply_workflow=False,
                )
            _enforce_review_status_floor(db, saved_hand.id, entry.hand.review_status)
        for review in validated.session_coaching_reviews:
            db.create_coaching_response(
                review.model_copy(update={"session_id": session.id})
            )

    return session


def _validate_hand(hand_payload: Mapping[str, Any], location: str) -> ValidatedHand:
    """Build and cross-check one hand entry's records. No database access."""

    if "hand" not in hand_payload:
        raise ImportValidationError(
            location, "no 'hand' object; every entry in 'hands' must carry one."
        )
    hand_data = dict(_require_mapping(hand_payload["hand"], f"{location}.hand"))
    hand_data.pop("id", None)
    hand_data["session_id"] = 0
    # Captured BEFORE the markers are stripped. They are the only surviving
    # record of a column the exporting database could not read, and without
    # them INVALID_HERO_OR_BOARD_CARDS and UNREADABLE_HAND_COLUMNS are
    # consumers whose producer -- the corrupt column, which the exporter has
    # already degraded to a fallback -- disappears across the round trip.
    unreadable = _recorded_unreadable_columns(hand_data.get("completion_evidence"))
    try:
        _apply_completion_import_defaults(hand_data)
    except (ValueError, TypeError) as exc:
        # TypeError as well as ValueError: the completion defaults run on raw
        # payload data, ahead of the model that would reject a field's type, so a
        # hand-edited value of the wrong shape reaches comparisons and lookups
        # written for strings. Located and refused rather than propagated, so no
        # import surface has to grow a new except clause to stay whole.
        raise ImportValidationError(f"{location}.hand", str(exc)) from exc
    hand = _build(Hand, hand_data, f"{location}.hand")

    players = [
        _build(HandPlayer, imported, item_location)
        for imported, item_location in _records(hand_payload, "players", location)
    ]
    actions: list[Action] = []
    for imported, item_location in _records(hand_payload, "actions", location):
        if "amount_semantics" not in imported:
            imported["amount_semantics"] = "unknown"
        actions.append(_build(Action, imported, item_location))
    _link_actions_to_players(actions, players)
    _normalize_duplicate_action_indexes(actions)
    _assign_missing_action_indexes(actions)

    settlement_data = hand_payload.get("settlement")
    settlement: HandSettlement | None = None
    if settlement_data is not None:
        imported_settlement = dict(
            _require_mapping(settlement_data, f"{location}.settlement")
        )
        imported_settlement["hand_id"] = 0
        settlement = _build(
            HandSettlement, imported_settlement, f"{location}.settlement"
        )

    settlement_entries = [
        _build(SettlementEntry, imported, item_location)
        for imported, item_location in _records(
            hand_payload, "settlement_entries", location
        )
    ]

    reviews: list[HandReview] = []
    for imported, item_location in _records(hand_payload, "reviews", location):
        _mark_imported_analysis_stale(imported)
        reviews.append(_build(HandReview, imported, item_location))

    coaching_reviews: list[CoachingResponse] = []
    for imported, item_location in _records(hand_payload, "coaching_reviews", location):
        imported["session_id"] = 0
        _mark_imported_analysis_stale(imported)
        coaching_reviews.append(_build(CoachingResponse, imported, item_location))

    corrections = [
        _build(HandCorrection, imported, item_location)
        for imported, item_location in _records(hand_payload, "corrections", location)
    ]

    issues: list[HandIssue] = []
    for imported, item_location in _records(hand_payload, "issues", location):
        _mark_imported_issue_unresolved(imported)
        issues.append(_build(HandIssue, imported, item_location))

    _validate_hand_relationships(
        location,
        players=players,
        actions=actions,
        settlement_entries=settlement_entries,
    )

    return ValidatedHand(
        hand=hand,
        unreadable_columns=unreadable,
        players=players,
        actions=actions,
        settlement=settlement,
        settlement_entries=settlement_entries,
        reviews=reviews,
        coaching_reviews=coaching_reviews,
        corrections=corrections,
        issues=issues,
    )


def _validate_hand_relationships(
    location: str,
    *,
    players: list[HandPlayer],
    actions: list[Action],
    settlement_entries: list[SettlementEntry],
) -> None:
    """Refuse duplicate player identifiers and references to players who are absent.

    Each of these was previously enforced somewhere downstream, if at all: two by
    a UNIQUE index, so they surfaced as ``sqlite3.IntegrityError`` from the
    middle of the write pass; one by ``create_hand_player``; the dangling keys by
    the writers' own reference checks; and the unresolvable NAME by nothing until
    the ledger refused the hand, inside an accounting panel, long after it had
    been stored.

    The unattributed-action rule is deliberately conditional on the hand
    declaring a roster at all. A hand with players and an action naming somebody
    outside them has a broken relationship in a file that claims to be complete.
    A hand with NO players is a different, older shape -- a legacy payload, or a
    hand entered as prose -- where there is no roster to contradict; the ledger
    already refuses to derive accounting for it, so it arrives visibly blocked
    rather than silently wrong, and rejecting the whole file would make a
    coverage limitation unimportable instead.
    """

    seen_keys: dict[str, int] = {}
    seen_seats: dict[int, int] = {}
    hero_index: int | None = None
    for index, player in enumerate(players):
        first = seen_keys.setdefault(player.player_key, index)
        if first != index:
            raise ImportValidationError(
                f"{location}.players[{index}]",
                f"player_key {player.player_key!r} is already used by "
                f"players[{first}]; a player key identifies one player in a hand.",
            )
        if player.seat_index is not None:
            seated = seen_seats.setdefault(player.seat_index, index)
            if seated != index:
                raise ImportValidationError(
                    f"{location}.players[{index}]",
                    f"seat_index {player.seat_index} is already used by "
                    f"players[{seated}]; two players cannot occupy one seat.",
                )
        if player.is_hero:
            if hero_index is not None:
                raise ImportValidationError(
                    f"{location}.players[{index}]",
                    f"a second player is flagged is_hero; players[{hero_index}] "
                    "already is, and a hand can have only one Hero player.",
                )
            hero_index = index

    names: dict[str, int] = {}
    for player in players:
        names[player.player_name] = names.get(player.player_name, 0) + 1

    for index, action in enumerate(actions):
        item_location = f"{location}.actions[{index}]"
        if action.player_key is not None:
            if action.player_key not in seen_keys:
                raise ImportValidationError(
                    item_location,
                    f"player_key {action.player_key!r} matches no player in this "
                    "hand; the action cannot be attributed.",
                )
            continue
        if not players:
            continue
        matches = names.get(action.player_name, 0)
        if matches == 0:
            raise ImportValidationError(
                item_location,
                f"player {action.player_name!r} is not among this hand's "
                f"{len(players)} player(s); the action cannot be attributed.",
            )
        if matches > 1:
            raise ImportValidationError(
                item_location,
                f"player {action.player_name!r} is shared by {matches} players in "
                "this hand and the action carries no player_key, so it cannot be "
                "attributed to one of them.",
            )

    for index, entry in enumerate(settlement_entries):
        item_location = f"{location}.settlement_entries[{index}]"
        if entry.player_key is not None:
            if entry.player_key not in seen_keys:
                raise ImportValidationError(
                    item_location,
                    f"player_key {entry.player_key!r} matches no player in this "
                    "hand; the award cannot be attributed.",
                )
            continue
        if not players:
            continue
        matches = names.get(entry.player_name, 0)
        if matches == 0:
            raise ImportValidationError(
                item_location,
                f"player {entry.player_name!r} is not among this hand's "
                f"{len(players)} player(s); the award cannot be attributed.",
            )
        if matches > 1:
            raise ImportValidationError(
                item_location,
                f"player {entry.player_name!r} is shared by {matches} players in "
                "this hand and the award carries no player_key, so it cannot be "
                "attributed to one of them.",
            )


def _assert_within_ceilings(payload: object) -> None:
    """Refuse a payload too large to be a session before anything else walks it.

    The equivalent ceiling used to exist only at the Streamlit upload widgets, so
    ``import_session`` itself accepted an arbitrarily large or deeply nested
    dict from any programmatic caller. Measured by traversal rather than by
    re-serialising, because this function is also handed dicts that were never
    JSON text and because ``json.dumps`` of a hostile payload is itself the
    allocation being guarded against. Strings dominate an export's size, so their
    combined length stands in for the byte ceiling; the value and depth limits
    bound the shapes that are cheap on disk and expensive in memory. The walk
    keeps its own stack so a deeply nested payload cannot exhaust Python's.
    """

    values = 0
    text = 0
    stack: list[tuple[object, int]] = [(payload, 1)]
    while stack:
        value, depth = stack.pop()
        values += 1
        if values > MAX_IMPORT_PAYLOAD_VALUES:
            raise ImportValidationError(
                "",
                f"Import payload holds more than {MAX_IMPORT_PAYLOAD_VALUES} "
                "values, which is far larger than any exported session.",
            )
        if depth > MAX_IMPORT_PAYLOAD_DEPTH:
            raise ImportValidationError(
                "",
                f"Import payload nests deeper than {MAX_IMPORT_PAYLOAD_DEPTH} "
                "levels; an exported session is a handful deep.",
            )
        if isinstance(value, str):
            text += len(value)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                text += len(str(key))
                stack.append((item, depth + 1))
        elif isinstance(value, (list, tuple)):
            for item in value:
                stack.append((item, depth + 1))
        if text > MAX_IMPORT_PAYLOAD_TEXT_CHARS:
            raise ImportValidationError(
                "",
                f"Import payload holds more than "
                f"{MAX_IMPORT_PAYLOAD_TEXT_CHARS // (1024 * 1024)} MB of text.",
            )


def _require_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ImportValidationError(
            location, f"expected an object, found {type(value).__name__}."
        )
    return value


def _require_sequence(
    container: Mapping[str, Any], key: str, location: str
) -> Sequence[Any]:
    value = container.get(key)
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ImportValidationError(
            f"{location}.{key}", f"expected a list, found {type(value).__name__}."
        )
    return value


def _records(
    hand_payload: Mapping[str, Any], key: str, location: str
) -> list[tuple[dict[str, Any], str]]:
    """Each record under `key`, copied, re-parented, and paired with its location."""

    prepared: list[tuple[dict[str, Any], str]] = []
    for index, item in enumerate(_require_sequence(hand_payload, key, location)):
        item_location = f"{location}.{key}[{index}]"
        imported = dict(_require_mapping(item, item_location))
        imported.pop("id", None)
        imported["hand_id"] = 0
        prepared.append((imported, item_location))
    return prepared


def _read_time_only_fields(model: type[BaseModel]) -> frozenset[str]:
    """Fields the models mark ``exclude=True``: derivations about the current row.

    ``Hand.derived_result_substituted`` and ``PersistedModel.unreadable_columns``
    both document that no export, payload or database row can carry them -- and
    both were true only of the EXPORT half. Nothing dropped them on the way IN,
    so a hand-edited ``"derived_result_substituted": true`` was accepted by the
    model and then refused by ``create_hand`` from the middle of the write pass.
    Derived from the model rather than listed here, so a field added with
    ``exclude=True`` later is covered without anyone remembering this exists.
    """
    return frozenset(
        name for name, spec in model.model_fields.items() if spec.exclude
    )


def _build(model: type[ModelT], data: Mapping[str, Any], location: str) -> ModelT:
    """Construct one record, reporting a field failure against its own location.

    Pydantic already names the field and the value it refused; what it cannot
    know is which of a session's several hundred records it was looking at, and
    that is the part an operator needs to fix the file.
    """

    read_time_only = _read_time_only_fields(model)
    if read_time_only & data.keys():
        data = {key: value for key, value in data.items() if key not in read_time_only}
    try:
        return model(**data)
    except ValidationError as exc:
        raise ImportValidationError(location, str(exc)) from exc
    except TypeError as exc:
        raise ImportValidationError(
            location, f"cannot be read as a {model.__name__} record. {exc}"
        ) from exc


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
    # `isinstance` first: this runs on raw payload data, before `Hand` has had a
    # chance to reject the field, so `declared` can be any JSON value. A hand
    # edited to `"completion_status": []` made this membership test raise
    # `TypeError: unhashable type: 'list'` out of the middle of the import.
    if (
        isinstance(declared, str)
        and declared in _IMPORT_COMPLETION_RESTRICTION
        and derived in _IMPORT_COMPLETION_RESTRICTION
        and _IMPORT_COMPLETION_RESTRICTION[declared]
        > _IMPORT_COMPLETION_RESTRICTION[derived]
    ):
        # The membership test above already restricts `declared` to the three
        # CompletionStatus values, which mypy cannot infer through a str-keyed map.
        derived = cast(CompletionStatus, declared)
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


def _assign_missing_action_indexes(actions: list[Action]) -> None:
    """Give every action its street order before the write, not during it.

    ``create_action`` fills a NULL ``action_index`` with ``MAX + 1`` for the
    street and then asserts the slot is free, so a payload mixing an explicit
    index with a NULL one on the same street raised out of the middle of the
    write pass rather than out of validation. Reproducing the writer's own
    ``MAX + 1`` rule here -- rather than packing the indexes densely -- keeps the
    stored order identical to what the writer would have chosen, and leaves it
    nothing left to discover.
    """
    highest: dict[str, int] = {}
    for action in actions:
        if action.action_index is not None:
            highest[action.street] = max(
                highest.get(action.street, 0), action.action_index
            )
    for index, action in enumerate(actions):
        if action.action_index is not None:
            continue
        assigned = highest.get(action.street, 0) + 1
        highest[action.street] = assigned
        actions[index] = action.model_copy(update={"action_index": assigned})
