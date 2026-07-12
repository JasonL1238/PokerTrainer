"""Export YOLO card timelines as PokerTrainer draft session imports.

This is an offline, post-session bridge. It does not write to the app database
unless --import-db is explicitly provided, and by default it only exports hands
whose card fields satisfy the app's Hand validation rules.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker_tracker.db import PokerDatabase
from poker_tracker.import_export import EXPORT_VERSION, import_session
from poker_tracker.models import Action, Hand, HandPlayer, Session

from cv_lab.scripts.validate_yolo_card_timeline import validate_timeline


DEFAULT_TIMELINE = "cv_lab/results/yolo_card_timeline_card_changes_v1.json"
DEFAULT_OUT = "cv_lab/results/yolo_card_hands_draft_session.json"
VALID_BOARD_COUNTS = {0, 3, 4, 5}
HAND_CORRECTION_FIELDS = ["hand_number", "hero_cards", "board_cards", "action", "notes"]


def _card_to_app(token: str) -> str:
    """Convert detector labels like AS/10H/td to app cards like As/Th/Td."""
    text = token.strip()
    if not text:
        raise ValueError("empty card token")
    rank_text = text[:-1].upper()
    suit = text[-1].lower()
    rank = "T" if rank_text in {"10", "T"} else rank_text
    return f"{rank}{suit}"


def _cards_to_app(cards: list[str] | None) -> str:
    return " ".join(_card_to_app(card) for card in (cards or []))


def _split_card_text(value: str) -> list[str]:
    text = value.strip().replace(",", " ").replace("/", " ").replace("-", " ")
    if not text:
        return []
    tokens = text.split()
    if len(tokens) == 1 and len(tokens[0]) > 2:
        compact = tokens[0]
        if len(compact) % 2 != 0:
            raise ValueError(f"invalid compact card text: {value}")
        return [compact[index:index + 2] for index in range(0, len(compact), 2)]
    return tokens


def load_hand_corrections(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    corrections: dict[int, dict[str, str]] = {}
    for row in rows:
        hand_number = row.get("hand_number", "").strip()
        if not hand_number:
            continue
        corrections[int(hand_number)] = row
    return corrections


def apply_hand_corrections(timeline: dict[str, Any], corrections: dict[int, dict[str, str]]) -> dict[str, Any]:
    if not corrections:
        return timeline
    updated = json.loads(json.dumps(timeline))
    for hand in updated.get("hands", []):
        hand_number = hand.get("hand_number")
        correction = corrections.get(hand_number)
        if not correction:
            continue
        action = correction.get("action", "").strip().lower()
        hand["manual_correction"] = {
            "action": action or "keep",
            "notes": correction.get("notes", ""),
        }
        if action in {"drop", "delete", "remove", "skip"}:
            hand["drop_from_export"] = True
            continue
        if correction.get("hero_cards", "").strip():
            hand["hero"] = _split_card_text(correction["hero_cards"])
        if correction.get("board_cards", "").strip():
            hand["board"] = _split_card_text(correction["board_cards"])
        cards = (hand.get("hero") or []) + (hand.get("board") or [])
        hand["complete_cards"] = (
            len(hand.get("hero") or []) == 2
            and len(hand.get("board") or []) in VALID_BOARD_COUNTS
            and len(cards) == len(set(cards))
        )
        hand["warnings"] = ["manual_hand_correction"]
    return updated


def _confidence_for_hand(hand: dict[str, Any]) -> float:
    confidence = 0.95
    confidence -= 0.15 * len(hand.get("warnings", []))
    if not hand.get("complete_cards"):
        confidence -= 0.25
    if not hand.get("hero"):
        confidence -= 0.20
    board_count = len(hand.get("board") or [])
    if board_count not in VALID_BOARD_COUNTS:
        confidence -= 0.20
    return max(0.0, min(1.0, round(confidence, 3)))


def _hand_notes(hand: dict[str, Any], *, timeline_path: Path) -> str:
    warnings = ", ".join(hand.get("warnings", [])) or "none"
    sources = ", ".join(hand.get("source_images", [])[:6])
    if len(hand.get("source_images", [])) > 6:
        sources += ", ..."
    notes = (
        "CV draft from YOLO card timeline. "
        f"timeline={timeline_path}; t={hand.get('t_start')}..{hand.get('t_end')}; "
        f"warnings={warnings}; source_images={sources}"
    )
    manual = hand.get("manual_correction")
    if manual:
        notes += f"; manual_correction={manual.get('action', 'keep')}; manual_notes={manual.get('notes', '')}"
    return notes


def _build_players(hand: dict[str, Any]) -> list[HandPlayer]:
    """Build HandPlayer rows from a reconstruction-spine hand (empty for card-only)."""
    players: list[HandPlayer] = []
    for row in hand.get("players", []) or []:
        name = row.get("player_name") or (
            "Hero" if row.get("is_hero") else (row.get("position") or f"Seat{row.get('seat')}")
        )
        players.append(HandPlayer(
            hand_id=0,
            player_name=name,
            position=row.get("position", "") or "",
            starting_stack=row.get("starting_stack"),
            is_hero=bool(row.get("is_hero")),
        ))
    return players


def _build_actions(hand: dict[str, Any]) -> list[Action]:
    """Build Action rows from a reconstruction-spine hand (empty for card-only)."""
    actions: list[Action] = []
    for row in hand.get("actions", []) or []:
        actions.append(Action(
            hand_id=0,
            street=row["street"],
            action_index=row.get("action_index"),
            player_name=row.get("player_name") or f"Seat{row.get('seat')}",
            position=row.get("position", "") or "",
            action_type=row["action_type"],
            amount=row.get("amount"),
            pot_before=row.get("pot_before"),
            stack_before=row.get("stack_before"),
        ))
    return actions


def hand_to_import_payload(
    hand: dict[str, Any],
    *,
    output_hand_number: int,
    timeline_path: Path,
    include_incomplete: bool = False,
) -> dict[str, Any] | None:
    hero_cards = _cards_to_app(hand.get("hero"))
    board_cards = _cards_to_app(hand.get("board"))
    board_count = len(hand.get("board") or [])

    if hand.get("drop_from_export"):
        return None

    if not include_incomplete:
        if not hand.get("complete_cards"):
            return None
        if len(hand.get("hero") or []) != 2 or board_count not in VALID_BOARD_COUNTS:
            return None

    confidence = _confidence_for_hand(hand)

    # Extended fields present only on reconstruction-spine hands; card-only timelines
    # (the current 53-class YOLO path) leave these empty and unchanged.
    players = _build_players(hand)
    actions = _build_actions(hand)
    hero_player = next((p for p in players if p.is_hero), None)
    table_size = len(players) if 2 <= len(players) <= 10 else None

    hand_model = Hand(
        session_id=0,
        hand_number=output_hand_number,
        game_type="",
        table_size=table_size,
        hero_position=hero_player.position if hero_player else "",
        hero_cards=hero_cards,
        board_cards=board_cards,
        pot_size=hand.get("pot"),
        result=hand.get("result", "") or "",
        hero_bb_won=hand.get("hero_bb_won"),
        review_status="needs_correction",
        confidence_score=confidence,
        source_type="cv_import",
        tags=["LOW_CONFIDENCE"] if confidence < 0.8 else [],
        notes=_hand_notes(hand, timeline_path=timeline_path),
    )
    return {
        "hand": hand_model.model_dump(mode="json"),
        "players": [player.model_dump(mode="json") for player in players],
        "actions": [action.model_dump(mode="json") for action in actions],
        "reviews": [],
        "cv_source": {
            "timeline_hand_number": hand.get("hand_number"),
            "t_start": hand.get("t_start"),
            "t_end": hand.get("t_end"),
            "source_images": hand.get("source_images", []),
            "warnings": hand.get("warnings", []),
            "winner_seat": hand.get("winner_seat"),
            "reconciled": hand.get("reconciled"),
        },
    }


def timeline_to_session_payload(
    timeline: dict[str, Any],
    *,
    timeline_path: Path,
    session_name: str,
    include_incomplete: bool = False,
    allow_validation_warnings: bool = False,
) -> dict[str, Any]:
    hands = []
    skipped = []
    validation_report = validate_timeline(timeline) if isinstance(timeline.get("states"), list) else None
    validation_by_hand = {
        report.get("hand_number"): report
        for report in (validation_report or {}).get("hands", [])
    }
    for hand in timeline.get("hands", []):
        manual_corrected = bool(hand.get("manual_correction"))
        validation = validation_by_hand.get(hand.get("hand_number"))
        if (
            validation
            and validation.get("warning_count", 0) > 0
            and not allow_validation_warnings
            and not manual_corrected
        ):
            skipped.append({
                "timeline_hand_number": hand.get("hand_number"),
                "reason": "validation_warnings",
                "detail": f"{validation['warning_count']} validation warnings; use --allow-validation-warnings to export anyway.",
            })
            continue
        try:
            payload = hand_to_import_payload(
                hand,
                output_hand_number=len(hands) + 1,
                timeline_path=timeline_path,
                include_incomplete=include_incomplete,
            )
        except (ValueError, ValidationError) as exc:
            skipped.append({
                "timeline_hand_number": hand.get("hand_number"),
                "reason": type(exc).__name__,
                "detail": str(exc),
            })
            continue
        if payload is None:
            skipped.append({
                "timeline_hand_number": hand.get("hand_number"),
                "reason": "incomplete_or_invalid_cards",
                "detail": "Use --include-incomplete to export needs-correction drafts when model validation allows it.",
            })
            continue
        hands.append(payload)

    session = Session(
        name=session_name,
        date_played=date.today(),
        platform="ClubWPT Gold",
        notes=(
            "Imported from offline YOLO card timeline. "
            "Actions, stacks, pot, and winners still require OCR/state reconstruction."
        ),
    )
    return {
        "export_version": EXPORT_VERSION,
        "session": session.model_dump(mode="json"),
        "hands": hands,
        "cv_import_summary": {
            "timeline": str(timeline_path),
            "timeline_hands": len(timeline.get("hands", [])),
            "exported_hands": len(hands),
            "skipped_hands": len(skipped),
            "skipped": skipped,
            "validation_summary": (validation_report or {}).get("summary"),
        },
    }


def export_timeline(
    timeline_path: Path,
    out_path: Path,
    *,
    session_name: str,
    include_incomplete: bool = False,
    allow_validation_warnings: bool = False,
    hand_corrections_path: Path | None = None,
) -> dict[str, Any]:
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if hand_corrections_path is not None:
        timeline = apply_hand_corrections(timeline, load_hand_corrections(hand_corrections_path))
    payload = timeline_to_session_payload(
        timeline,
        timeline_path=timeline_path,
        session_name=session_name,
        include_incomplete=include_incomplete,
        allow_validation_warnings=allow_validation_warnings,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", default=DEFAULT_TIMELINE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--session-name", default="YOLO card draft session")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--allow-validation-warnings", action="store_true")
    parser.add_argument("--hand-corrections", help="Optional hand_corrections.csv from the hand review UI")
    parser.add_argument("--import-db", help="Optional SQLite DB path to import into after writing JSON.")
    args = parser.parse_args()

    timeline_path = Path(args.timeline)
    out_path = Path(args.out)
    payload = export_timeline(
        timeline_path,
        out_path,
        session_name=args.session_name,
        include_incomplete=args.include_incomplete,
        allow_validation_warnings=args.allow_validation_warnings,
        hand_corrections_path=Path(args.hand_corrections) if args.hand_corrections else None,
    )

    print(f"timeline={timeline_path}")
    print(f"out={out_path}")
    print(f"exported_hands={payload['cv_import_summary']['exported_hands']}")
    print(f"skipped_hands={payload['cv_import_summary']['skipped_hands']}")

    if args.import_db:
        db = PokerDatabase(args.import_db)
        db.init_db()
        session = import_session(db, payload)
        db.close()
        print(f"imported_session_id={session.id}")


if __name__ == "__main__":
    main()
