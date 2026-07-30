from poker_tracker.persistence.models import Action, HandPlayer
from poker_tracker.ui.poker_visuals import (
    _POKER_CSS,
    action_replay_state,
    action_timeline_html,
    equity_meter_html,
    playing_card_html,
    poker_table_html,
    range_cells_from_notation,
    range_matrix_html,
)


def test_poker_visuals_include_narrow_phone_layout() -> None:
    assert "@media (max-width: 420px)" in _POKER_CSS
    assert ".pt-table-shell { min-height: 232px" in _POKER_CSS


def test_poker_table_responds_to_its_column_width() -> None:
    assert ".pt-poker-stage { margin: 0; min-width: 0; container-type: inline-size; }" in _POKER_CSS
    assert "@container (max-width: 560px)" in _POKER_CSS
    assert "@container (max-width: 390px)" in _POKER_CSS


def test_playing_card_renders_rank_suit_and_escapes_unknown_input() -> None:
    ace = playing_card_html("Ah")
    unknown = playing_card_html("<bad>")

    assert "A of hearts" in ace
    assert "♥" in ace
    assert "&lt;bad&gt;" in unknown
    assert "<bad>" not in unknown


def test_poker_table_uses_completed_hand_values() -> None:
    players = [
        HandPlayer(
            hand_id=1,
            player_name="Hero",
            position="BTN",
            starting_stack=100,
            is_hero=True,
        ),
        HandPlayer(hand_id=1, player_name="Villain", position="BB", starting_stack=98),
    ]

    html = poker_table_html(
        hero_cards="As Kh",
        board_cards="Qs 7d 2c",
        pot_size=18.5,
        players=players,
        result_bb=9.25,
    )

    assert "18.5 BB" in html
    assert "+9.25 BB" in html
    assert "pt-seat-hero" in html
    assert "Product preview" not in html


def test_poker_table_does_not_repeat_position_as_player_name() -> None:
    html = poker_table_html(
        hero_cards="As Kh",
        board_cards="",
        pot_size=3,
        players=[HandPlayer(hand_id=1, player_name="HJ", position="HJ", starting_stack=100)],
    )

    assert html.count(">HJ<") == 1


def test_action_timeline_maps_semantic_action_tones() -> None:
    actions = [
        Action(
            hand_id=1,
            street="preflop",
            player_name="Hero",
            action_type="raise",
            amount=3,
        ),
        Action(
            hand_id=1,
            street="preflop",
            player_name="Villain",
            action_type="call",
            amount=3,
        ),
    ]

    html = action_timeline_html(actions)

    assert "pt-action-raise" in html
    assert "pt-action-call" in html
    assert "pt-history-panel" in html
    assert "Completed hand decision history" in html
    assert "All 2 saved actions" in html
    assert '<span class="pt-history-decision">Raise</span>' in html
    assert '<span class="pt-history-size">3 BB</span>' in html
    assert "No saved actions are hidden" in html
    assert "Reconstructed table after action" not in html


def test_action_timeline_does_not_repeat_matching_position() -> None:
    html = action_timeline_html(
        [
            Action(
                hand_id=1,
                street="preflop",
                player_name="HJ",
                position="HJ",
                action_type="raise",
                amount=2.5,
            )
        ]
    )

    assert html.count(">HJ<") == 1


def test_decision_history_uses_scoped_consistent_rows() -> None:
    assert ".stMarkdown ol.pt-decision-history" in _POKER_CSS
    assert "display: block !important" in _POKER_CSS
    assert "min-height: 76px" in _POKER_CSS
    assert ".pt-history-note { width: 100%; white-space: normal" in _POKER_CSS


def test_decision_history_includes_reconstructed_pot_stack_spr_and_notes() -> None:
    actions = [
        Action(
            hand_id=1,
            street="flop",
            player_name="Hero",
            position="BTN",
            action_type="bet",
            amount=6,
            pot_before=12,
            stack_before=84,
            notes="Detected from bet pill",
        )
    ]
    players = [
        HandPlayer(
            hand_id=1,
            player_name="Hero",
            position="BTN",
            starting_stack=84,
            is_hero=True,
        ),
        HandPlayer(hand_id=1, player_name="BB", position="BB", starting_stack=72),
    ]

    html = action_timeline_html(actions, players=players)

    assert "12 → 18" in html
    assert "72 BB" in html
    assert "<b>SPR</b> 6.0" in html
    assert "Detected from bet pill" in html


def test_action_replay_reconstructs_the_selected_board_and_player_state() -> None:
    players = [
        HandPlayer(
            hand_id=1,
            player_name="Hero",
            position="BTN",
            starting_stack=100,
            is_hero=True,
        ),
        HandPlayer(hand_id=1, player_name="Villain", position="BB", starting_stack=100),
    ]
    actions = [
        Action(
            hand_id=1,
            street="preflop",
            player_name="Hero",
            position="BTN",
            action_type="raise",
            amount=3,
            stack_before=100,
            pot_before=1.5,
        ),
        Action(
            hand_id=1,
            street="flop",
            player_name="Villain",
            position="BB",
            action_type="fold",
            pot_before=7.5,
            stack_before=97,
        ),
    ]

    preflop = action_replay_state(
        actions,
        0,
        players=players,
        board_cards="Qs 7d 2c 9h 3s",
    )
    flop = action_replay_state(
        actions,
        1,
        players=players,
        board_cards="Qs 7d 2c 9h 3s",
    )

    assert preflop.board_cards == ""
    assert preflop.pot_size == 4.5
    assert preflop.players[0].starting_stack == 97
    assert preflop.folded_player_keys == frozenset()
    assert flop.board_cards == "Qs 7d 2c"
    assert flop.pot_size == 7.5
    assert players[1].player_key in flop.folded_player_keys

    html = poker_table_html(
        hero_cards="As Kh",
        board_cards=flop.board_cards,
        pot_size=flop.pot_size,
        players=flop.players,
        actor_player_key=flop.actor_player_key,
        folded_player_keys=flop.folded_player_keys,
    )
    assert "Q of spades" in html
    assert "9 of hearts" not in html
    assert "pt-seat-acting" in html
    assert "pt-seat-folded" in html


def test_cards_layer_behind_hero_position_and_pot_clears_board() -> None:
    assert ".pt-seat-hero { z-index: 5" in _POKER_CSS
    assert ".pt-table-center { position: absolute; z-index: 2" in _POKER_CSS
    assert "bottom: calc(24px - 2%)" in _POKER_CSS
    assert ".pt-hero-cards { position: relative; z-index: 2" in _POKER_CSS
    assert ".pt-pot { position: relative; z-index: 3" in _POKER_CSS
    assert html_order(_table_markup()) == ["pt-board", "pt-pot", "pt-hero-cards"]


def _table_markup() -> str:
    players = [
        HandPlayer(
            hand_id=1,
            player_name="Hero",
            position="BTN",
            starting_stack=100,
            is_hero=True,
        ),
        HandPlayer(hand_id=1, player_name="Villain", position="BB", starting_stack=100),
    ]
    return poker_table_html(
        hero_cards="As Kh",
        board_cards="Qs 7d 2c",
        pot_size=18.5,
        players=players,
    )


def html_order(markup: str) -> list[str]:
    classes = ["pt-board", "pt-pot", "pt-hero-cards"]
    return sorted(classes, key=markup.index)


def test_equity_meter_bounds_values_and_marks_threshold() -> None:
    html = equity_meter_html(1.4, threshold=0.33)

    assert 'aria-valuenow="100.0"' in html
    assert "left:33.00%" in html


def test_range_matrix_is_canonical_13_by_13() -> None:
    cells = range_cells_from_notation("TT+,AJs+,AQo+,KQs")
    html = range_matrix_html(cells)

    assert len(html.split('class="pt-range-cell')) - 1 == 169
    assert cells["AA"] == "value"
    assert cells["AKs"] == "value"
    assert "72o" in html
