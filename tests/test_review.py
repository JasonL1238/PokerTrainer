from poker_tracker.models import Action, Hand, HandPlayer
from poker_tracker.review import generate_mock_review


def test_generate_mock_review_uses_tags_actions_and_result() -> None:
    hand = Hand(
        id=1,
        session_id=1,
        hand_number=7,
        hero_position="BTN",
        hero_cards="As Qh",
        board_cards="Qd 7s 2c 9h 3s",
        result="Hero wins",
        hero_bb_won=18,
        tags=["MISSED_VALUE", "RIVER_DECISION"],
    )
    actions = [
        Action(
            hand_id=1,
            street="preflop",
            action_index=1,
            player_name="Hero",
            position="BTN",
            action_type="raise",
            amount=2.5,
        ),
        Action(
            hand_id=1,
            street="river",
            action_index=1,
            player_name="Hero",
            position="BTN",
            action_type="check",
        ),
    ]
    players = [
        HandPlayer(hand_id=1, player_name="Hero", position="BTN", starting_stack=200, is_hero=True),
        HandPlayer(hand_id=1, player_name="Villain", position="BB", starting_stack=190),
    ]

    review = generate_mock_review(hand, actions, players)

    assert review.hand_id == 1
    assert "Hand #7" in review.hand_summary
    assert "MISSED_VALUE" in review.hand_summary
    assert "thin value" in review.theory_coach
    assert "Aggressive actions" in review.ev_math_notes
    assert "worse hands" in review.next_review_question
