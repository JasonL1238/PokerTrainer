from __future__ import annotations

from datetime import date

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
