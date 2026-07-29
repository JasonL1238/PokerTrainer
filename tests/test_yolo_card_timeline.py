from cv_lab.scripts.pipeline.build_yolo_card_timeline import _stage, _summarize_hands, _zone_for_box


def test_zone_for_box_assigns_hero_board_other():
    assert _zone_for_box(0.48, 0.70) == "hero"
    assert _zone_for_box(0.45, 0.44) == "board"
    assert _zone_for_box(0.90, 0.20) == "other"


def test_legacy_zone_for_box_is_unanchored_and_documented():
    """Pin the KNOWN failure of the legacy unanchored rectangles so nobody
    re-wires them into the reconstruction spine.

    On the 06-21 recording (2062x1178, AR 1.750) the real community row sits at
    raw normalized cy 0.335-0.338. The legacy board window starts at cy 0.36, so
    every board card there is zoned "other" -- 435 detections, silently re-routed
    to villain seats. The spine anchors instead (landmark_anchor.zone_for_ref).
    """
    assert _zone_for_box(0.45, 0.337) == "other"
    assert "LEGACY" in (_zone_for_box.__doc__ or "")


def test_hard_example_miner_imports_the_single_zone_definition():
    from cv_lab.scripts.training import mine_yolo_card_hard_examples as miner

    assert miner._zone_for_box is _zone_for_box


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
