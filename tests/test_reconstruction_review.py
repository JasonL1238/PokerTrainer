from poker_tracker.ui.reconstruction_review import (
    history_impacts,
    observed_facts,
    states_for_hand,
)


def _fixture():
    states = [
        {
            "state_index": 0,
            "time_s": 0.0,
            "image": "/tmp/a.jpg",
            "stage": "preflop",
            "hero_cards": ["As", "Kd"],
            "board_cards": [],
            "pot": 1.5,
            "dealt_in": [0, 4],
            "stacks": {0: 100, 4: 100},
            "bets": {},
            "pills": {},
            "active_seat": 4,
        },
        {
            "state_index": 1,
            "time_s": 1.0,
            "image": "/tmp/b.jpg",
            "stage": "preflop",
            "hero_cards": ["As", "Kd"],
            "board_cards": [],
            "pot": 3.0,
            "dealt_in": [0, 4],
            "stacks": {0: 100, 4: 97},
            "bets": {4: 3},
            "pills": {4: "raise"},
            "active_seat": 0,
        },
    ]
    hand = {
        "hand_number": 1,
        "t_start": 0.0,
        "t_end": 1.0,
        "hero": ["As", "Kd"],
        "source_images": ["/tmp/a.jpg", "/tmp/b.jpg"],
        "players": [
            {"seat": 0, "player_name": "Hero", "position": "SB"},
            {"seat": 4, "player_name": "Seat4", "position": "BTN"},
        ],
        "actions": [
            {
                "street": "preflop",
                "seat": 4,
                "position": "BTN",
                "player_name": "Seat4",
                "action_type": "raise",
                "amount": 3.0,
                "source_image": "/tmp/b.jpg",
                "derivation": "stack_delta",
            }
        ],
        "result": "Hero folds",
        "pot": 3.0,
        "reconciled": True,
    }
    return {"states": states, "hands": [hand]}, hand


def test_frame_review_pairs_observations_with_history_impacts() -> None:
    timeline, hand = _fixture()
    states = states_for_hand(timeline, hand)

    assert [state["image"] for state in states] == ["/tmp/a.jpg", "/tmp/b.jpg"]
    assert ("Hero cards", "As Kd") in observed_facts(states[0])
    impacts = history_impacts(hand, states, 1)
    assert any("BTN Seat4 raise 3 BB" in impact["text"] for impact in impacts)
    assert any(impact["source"] == "stack delta" for impact in impacts)
    assert any(impact["kind"] == "Settlement" for impact in impacts)
