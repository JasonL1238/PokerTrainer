from __future__ import annotations

from poker_tracker.db import PokerDatabase
from poker_tracker.models import Action, Hand, HandPlayer, Session
from poker_tracker.review import generate_mock_review


def create_sample_data(db: PokerDatabase) -> Session:
    """Create a representative manual-review sample session."""
    session = db.create_session(
        Session(
            name="Sample post-session review",
            platform="ClubWPT Gold",
            stakes="1/2 NL",
            notes="Seed data for testing the manual review workflow.",
        )
    )

    _create_hand(
        db,
        session.id,
        Hand(
            session_id=session.id,
            hand_number=1,
            game_type="No-limit Hold'em",
            blinds_antes="1/2 NL",
            table_size=6,
            effective_stack=100,
            hero_position="BTN",
            hero_cards="Ah Qs",
            board_cards="Qd 7s 2c 9h 3s",
            hero_bb_won=12.5,
            result="Hero wins with top pair",
            review_status="reviewed",
            tags=["BIG_POT"],
            notes="Simple top-pair value hand.",
        ),
        [
            ("Hero", "BTN", 200, True, ""),
            ("HJ", "HJ", 180, False, ""),
            ("BB", "BB", 210, False, ""),
        ],
        [
            ("preflop", "HJ", "HJ", "raise", 2.5),
            ("preflop", "Hero", "BTN", "call", 2.5),
            ("preflop", "BB", "BB", "call", 1.5),
            ("flop", "BB", "BB", "check", None),
            ("flop", "HJ", "HJ", "bet", 4),
            ("flop", "Hero", "BTN", "call", 4),
            ("turn", "HJ", "HJ", "check", None),
            ("turn", "Hero", "BTN", "bet", 9),
            ("river", "HJ", "HJ", "check", None),
            ("river", "Hero", "BTN", "bet", 16),
        ],
    )

    _create_hand(
        db,
        session.id,
        Hand(
            session_id=session.id,
            hand_number=2,
            game_type="No-limit Hold'em",
            blinds_antes="1/2 NL",
            table_size=6,
            hero_position="CO",
            hero_cards="Kc Qc",
            board_cards="Kh 8d 4s 2c 2h",
            hero_bb_won=4,
            result="Hero wins after checking river",
            tags=["MISSED_VALUE", "RIVER_DECISION"],
            notes="Missed river thin value candidate.",
        ),
        [("Hero", "CO", 160, True, ""), ("BB", "BB", 155, False, "stationy")],
        [
            ("preflop", "Hero", "CO", "raise", 2.5),
            ("preflop", "BB", "BB", "call", 1.5),
            ("flop", "BB", "BB", "check", None),
            ("flop", "Hero", "CO", "bet", 3),
            ("flop", "BB", "BB", "call", 3),
            ("turn", "BB", "BB", "check", None),
            ("turn", "Hero", "CO", "check", None),
            ("river", "BB", "BB", "check", None),
            ("river", "Hero", "CO", "check", None),
        ],
    )

    _create_hand(
        db,
        session.id,
        Hand(
            session_id=session.id,
            hand_number=3,
            game_type="No-limit Hold'em",
            blinds_antes="1/2 NL",
            table_size=6,
            hero_position="BB",
            hero_cards="Js Ts",
            board_cards="Td 8s 5h 2d Ac",
            hero_bb_won=-8,
            result="Hero loses multiway pot",
            tags=["MULTIWAY"],
        ),
        [
            ("Hero", "BB", 190, True, ""),
            ("UTG", "UTG", 120, False, ""),
            ("CO", "CO", 175, False, ""),
            ("BTN", "BTN", 205, False, ""),
        ],
        [
            ("preflop", "UTG", "UTG", "raise", 2.5),
            ("preflop", "CO", "CO", "call", 2.5),
            ("preflop", "BTN", "BTN", "call", 2.5),
            ("preflop", "Hero", "BB", "call", 1.5),
            ("flop", "Hero", "BB", "check", None),
            ("flop", "UTG", "UTG", "bet", 5),
            ("flop", "CO", "CO", "call", 5),
            ("flop", "Hero", "BB", "call", 5),
        ],
    )

    _create_hand(
        db,
        session.id,
        Hand(
            session_id=session.id,
            hand_number=4,
            game_type="No-limit Hold'em",
            blinds_antes="1/2 NL",
            hero_position="SB",
            hero_cards="Ad Kd",
            board_cards="Kc 9c 6d 4c 2s",
            hero_bb_won=-65,
            result="Hero loses large river call",
            tags=["BIG_POT", "RIVER_DECISION"],
            review_status="needs_correction",
        ),
        [("Hero", "SB", 140, True, ""), ("BTN", "BTN", 160, False, "aggressive")],
        [
            ("preflop", "BTN", "BTN", "raise", 2.5),
            ("preflop", "Hero", "SB", "raise", 9),
            ("preflop", "BTN", "BTN", "call", 6.5),
            ("flop", "Hero", "SB", "bet", 8),
            ("flop", "BTN", "BTN", "call", 8),
            ("turn", "Hero", "SB", "bet", 20),
            ("turn", "BTN", "BTN", "call", 20),
            ("river", "Hero", "SB", "check", None),
            ("river", "BTN", "BTN", "bet", 55),
            ("river", "Hero", "SB", "call", 55),
        ],
    )

    _create_hand(
        db,
        session.id,
        Hand(
            session_id=session.id,
            hand_number=5,
            game_type="No-limit Hold'em",
            blinds_antes="1/2 NL",
            hero_position="CO",
            hero_cards="9h 9s",
            hero_bb_won=3.5,
            result="Hero wins preflop after 3-bet",
            tags=["PREFLOP_3BET_SPOT"],
        ),
        [("Hero", "CO", 210, True, ""), ("HJ", "HJ", 180, False, "")],
        [
            ("preflop", "HJ", "HJ", "raise", 2.5),
            ("preflop", "Hero", "CO", "raise", 8),
            ("preflop", "HJ", "HJ", "fold", None),
            ("showdown", "Hero", "CO", "win", 3.5),
        ],
    )

    return session


def _create_hand(
    db: PokerDatabase,
    session_id: int,
    hand: Hand,
    players: list[tuple[str, str, float, bool, str]],
    actions: list[tuple[str, str, str, str, float | None]],
) -> Hand:
    saved_hand = db.create_hand(hand.model_copy(update={"session_id": session_id}))
    for name, position, stack, is_hero, notes in players:
        db.create_hand_player(
            HandPlayer(
                hand_id=saved_hand.id,
                player_name=name,
                position=position,
                starting_stack=stack,
                is_hero=is_hero,
                notes=notes,
            )
        )
    for street, player_name, position, action_type, amount in actions:
        db.create_action(
            Action(
                hand_id=saved_hand.id,
                street=street,
                player_name=player_name,
                position=position,
                action_type=action_type,
                amount=amount,
            )
        )
    if saved_hand.review_status == "reviewed":
        db.create_hand_review(
            generate_mock_review(
                saved_hand,
                db.fetch_actions_by_hand(saved_hand.id),
                db.fetch_players_by_hand(saved_hand.id),
            )
        )
    return saved_hand


# TODO: Future CV/OCR sample fixtures should import through the same DB methods.
