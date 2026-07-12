from cv_lab.scripts.build_yolo_card_timeline import _stage, _summarize_hands, _zone_for_box


def test_zone_for_box_assigns_hero_board_other():
    assert _zone_for_box(0.48, 0.70) == "hero"
    assert _zone_for_box(0.45, 0.44) == "board"
    assert _zone_for_box(0.90, 0.20) == "other"


def test_stage_from_board_count():
    assert _stage(0) == "preflop"
    assert _stage(2) == "partial_board"
    assert _stage(3) == "flop"
    assert _stage(4) == "turn"
    assert _stage(5) == "river"


def test_summarize_hands_splits_on_new_hero_after_board():
    states = [
        {"time_s": 0.0, "hero_cards": ["AS", "KD"], "board_cards": [], "image": "a.jpg"},
        {"time_s": 5.0, "hero_cards": ["AS", "KD"], "board_cards": ["2C", "3D", "4H"], "image": "b.jpg"},
        {"time_s": 10.0, "hero_cards": ["7S", "7D"], "board_cards": [], "image": "c.jpg"},
        {"time_s": 15.0, "hero_cards": ["7S", "7D"], "board_cards": ["AH", "QC", "9S"], "image": "d.jpg"},
    ]

    hands = _summarize_hands(states)

    assert len(hands) == 2
    assert hands[0]["hero"] == ["AS", "KD"]
    assert hands[0]["board"] == ["2C", "3D", "4H"]
    assert hands[0]["complete_cards"] is True
    assert hands[1]["hero"] == ["7S", "7D"]
    assert hands[1]["board"] == ["AH", "QC", "9S"]
    assert hands[1]["complete_cards"] is True
