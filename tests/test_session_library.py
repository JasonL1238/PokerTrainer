from __future__ import annotations

from datetime import date

import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.import_export import (
    export_session,
    import_hands_into_session,
)
from poker_tracker.persistence.models import Hand, Session, VideoRecord
from poker_tracker.ui.session_library import (
    date_session_name,
    filter_hands,
    filter_sessions,
    session_dates,
    sessions_on_date,
)


def make_db() -> PokerDatabase:
    db = PokerDatabase(":memory:")
    db.init_db()
    return db


def test_date_session_names_are_memorable_and_collision_safe() -> None:
    played_on = date(2026, 7, 27)
    first = Session(name="Monday, July 27, 2026", date_played=played_on)
    second = Session(name="Monday, July 27, 2026 · Session 2", date_played=played_on)

    assert date_session_name(played_on) == "Monday, July 27, 2026"
    assert date_session_name(played_on, [first]) == "Monday, July 27, 2026 · Session 2"
    assert date_session_name(played_on, [first, second]) == "Monday, July 27, 2026 · Session 3"


def test_session_and_hand_search_include_dates_and_study_fields() -> None:
    session = Session(
        id=4,
        name="Monday night",
        date_played=date(2026, 7, 27),
        platform="ClubWPT",
        stakes="1/2",
        notes="deep table",
    )
    hand = Hand(
        id=8,
        session_id=4,
        hand_number=12,
        hero_cards="Ah Qs",
        hero_position="BTN",
        hero_bb_won=-15,
        tags=["RIVER_DECISION"],
    )

    assert filter_sessions([session], "july clubwpt") == [session]
    assert filter_hands([hand], {4: session}, query="ah river") == [hand]
    assert filter_hands([hand], {4: session}, result_filter="losses") == [hand]
    assert filter_hands([hand], {4: session}, result_filter="wins") == []


def test_sessions_on_date_and_session_dates_help_calendar_browse() -> None:
    monday = Session(id=1, name="Monday", date_played=date(2026, 7, 27))
    tuesday = Session(id=2, name="Tuesday", date_played=date(2026, 7, 28))
    monday_two = Session(id=3, name="Monday late", date_played=date(2026, 7, 27))
    sessions = [monday, tuesday, monday_two]

    assert sessions_on_date(sessions, date(2026, 7, 27)) == [monday, monday_two]
    assert sessions_on_date(sessions, date(2026, 7, 29)) == []
    assert session_dates(sessions) == {date(2026, 7, 27), date(2026, 7, 28)}


def test_update_session_can_change_date_and_details() -> None:
    db = make_db()
    session = db.create_session(
        Session(
            name="Original",
            date_played=date(2026, 7, 27),
            platform="Manual",
            stakes="1/2",
            notes="first note",
        )
    )

    updated = db.update_session(
        session.model_copy(
            update={
                "name": "Renamed night",
                "date_played": date(2026, 7, 28),
                "platform": "ClubWPT",
                "stakes": "2/5",
                "notes": "corrected date",
            }
        )
    )
    fetched = db.fetch_session(session.id)

    assert updated.name == "Renamed night"
    assert updated.date_played == date(2026, 7, 28)
    assert updated.platform == "ClubWPT"
    assert updated.stakes == "2/5"
    assert updated.notes == "corrected date"
    assert fetched == updated
    db.close()


def test_update_session_rejects_missing_or_empty_name() -> None:
    db = make_db()
    session = db.create_session(Session(name="Keep", date_played=date(2026, 7, 27)))

    with pytest.raises(ValueError, match="without an id"):
        db.update_session(Session(name="Nope", date_played=date(2026, 7, 27)))
    with pytest.raises(ValueError, match="not found"):
        db.update_session(session.model_copy(update={"id": 999}))
    with pytest.raises(ValueError, match="empty"):
        db.update_session(session.model_copy(update={"name": "   "}))
    db.close()


def test_hands_and_videos_can_be_reorganized_without_schema_changes() -> None:
    db = make_db()
    source = db.create_session(Session(name="Source"))
    target = db.create_session(Session(name="Target"))
    db.create_hand(Hand(session_id=target.id, hand_number=1))
    movable = db.create_hand(Hand(session_id=source.id, hand_number=1, hero_cards="As Kh"))
    video = db.create_video(
        VideoRecord(
            session_id=source.id,
            original_filename="part-two.mp4",
            stored_path="/tmp/part-two.mp4",
            file_size_bytes=10,
        )
    )

    moved = db.move_hand_to_session(movable.id, target.id)
    attached = db.update_video_session(video.id, target.id)

    assert moved.session_id == target.id
    assert moved.hand_number == 2
    assert attached.session_id == target.id
    db.close()


def test_reconstructed_payload_can_append_to_an_existing_session() -> None:
    db = make_db()
    source = db.create_session(Session(name="Video two"))
    db.create_hand(Hand(session_id=source.id, hand_number=1, hero_cards="Ah Qs"))
    payload = export_session(db, source.id)
    target = db.create_session(Session(name="Monday, July 27, 2026"))
    db.create_hand(Hand(session_id=target.id, hand_number=1, hero_cards="Kd Kc"))

    imported = import_hands_into_session(db, payload, target.id)
    target_hands = db.fetch_hands_by_session(target.id)

    assert imported.id == target.id
    assert [(hand.hand_number, hand.hero_cards) for hand in target_hands] == [
        (1, "Kd Kc"),
        (2, "Ah Qs"),
    ]
    db.close()
