from poker_tracker.player_labels import actor_label, distinct_position, labels_match


def test_matching_labels_are_case_and_whitespace_insensitive() -> None:
    assert labels_match(" HJ ", "hj")
    assert actor_label("HJ", "hj", position_first=True) == "HJ"
    assert distinct_position("BB", " bb ") == ""


def test_distinct_name_and_position_are_both_preserved() -> None:
    assert actor_label("Hero", "BTN", position_first=True) == "BTN Hero"
    assert actor_label("Hero", "BTN") == "Hero BTN"
    assert distinct_position("Hero", "BTN") == "BTN"
