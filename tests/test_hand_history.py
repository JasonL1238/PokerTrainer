from poker_tracker.hand_history import format_hand_history
from poker_tracker.models import Action, Hand, HandPlayer, Session


def test_hand_history_formatting() -> None:
    session = Session(name="Friday", platform="ClubWPT Gold")
    hand = Hand(
        session_id=1,
        hand_number=3,
        hero_position="BTN",
        hero_cards="AhQs",
        board_cards="Qd 7s 2c 9h 3s",
        pot_size=42.5,
        result="Hero wins",
        hero_bb_won=12.5,
        review_status="reviewed",
        tags=["MISSED_VALUE", "RIVER_DECISION"],
    )
    actions = [
        Action(hand_id=1, street="preflop", action_index=1, player_name="HJ", position="HJ", action_type="raise", amount=2.5),
        Action(hand_id=1, street="preflop", action_index=2, player_name="Hero", position="BTN", action_type="call", amount=2.5),
        Action(hand_id=1, street="river", action_index=1, player_name="Hero", position="BTN", action_type="check"),
    ]
    players = [HandPlayer(hand_id=1, player_name="Hero", position="BTN", is_hero=True)]

    history = format_hand_history(session, hand, actions, players)

    assert "Session:" in history
    assert "Hand #3" in history
    assert "Hero: BTN, Ah Qs" in history
    assert "Final pot: 42.5" in history
    assert "Outcome: Hero wins" in history
    assert "Result: +12.5 BB" in history
    assert "Preflop:" in history
    assert "HJ HJ raise 2.5" in history
    assert "Review status: reviewed" in history


def test_hand_history_omits_pot_and_outcome_when_absent() -> None:
    session = Session(name="Friday", platform="ClubWPT Gold")
    hand = Hand(session_id=1, hand_number=1)

    history = format_hand_history(session, hand, [], [])

    assert "Final pot:" not in history
    assert "Outcome:" not in history
