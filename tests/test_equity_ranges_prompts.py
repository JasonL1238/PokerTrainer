from poker_tracker.analytics import compute_session_stats
from poker_tracker.coaching_prompts import (
    POST_SESSION_SAFETY,
    REQUIRED_REVIEW_SECTIONS,
    SESSION_REVIEW_SECTIONS,
    build_hand_review_prompt,
    build_session_review_prompt,
)
from poker_tracker.db import PokerDatabase
from poker_tracker.equity import PlaceholderEquityCalculator
from poker_tracker.hand_history import format_hand_history
from poker_tracker.models import Action, Hand, HandPlayer, Session
from poker_tracker.pot_odds import required_equity_to_call
from poker_tracker.preflop_ranges import get_preflop_range
from poker_tracker.ranges import estimate_villain_range_label, get_range_description


def test_equity_abstraction_returns_valid_result() -> None:
    result = PlaceholderEquityCalculator().calculate_equity("Ah Qs", "Qd 7s 2c", "loose")

    assert result.hero_hand == "AhQs"
    assert result.board == "Qd 7s 2c"
    assert result.villain_range_label == "loose"
    assert result.method == "placeholder"
    assert result.confidence < 0.5
    assert "Not a real equity calculation" in result.notes
    assert result.equity is not None


def test_range_label_mapping() -> None:
    assert estimate_villain_range_label(["PREFLOP_3BET_SPOT"], "3-bet only") == "premium"
    assert estimate_villain_range_label([], "nit tight") == "tight"
    assert estimate_villain_range_label([], "loose passive station") == "loose"
    assert estimate_villain_range_label([], "splashy whale") == "very_loose"
    assert estimate_villain_range_label([], "") == "unknown"
    assert get_range_description("bad-label").label == "unknown"


def test_hand_prompt_contains_required_sections_and_safety() -> None:
    session, hand, actions, players = _sample_hand()
    equity = PlaceholderEquityCalculator().calculate_equity(
        hand.hero_cards, hand.board_cards, "standard"
    )
    prompt = build_hand_review_prompt(
        session,
        hand,
        actions,
        players,
        pot_odds_facts={"required_equity_to_call": required_equity_to_call(25, 75)},
        equity_result=equity,
        villain_range_label="standard",
    )

    assert POST_SESSION_SAFETY in prompt
    assert "Do not invent equities" in prompt
    assert "Hand history:" in prompt
    assert "required_equity_to_call" in prompt
    assert "placeholder" in prompt
    for section in REQUIRED_REVIEW_SECTIONS:
        assert section in prompt


def test_hand_prompt_includes_hero_range_for_valid_position() -> None:
    session, hand, actions, players = _sample_hand()  # hero_position == "BTN"
    prompt = build_hand_review_prompt(
        session, hand, actions, players, villain_range_label="standard"
    )

    btn_rfi = get_preflop_range("BTN", "rfi")
    assert "Hero preflop range" in prompt
    assert "study baseline" in prompt
    assert "position: BTN" in prompt
    assert btn_rfi.notation in prompt
    assert "percent of hands:" in prompt
    # Existing villain behavior remains intact.
    assert "Villain range label:" in prompt


def test_hand_prompt_hero_position_alias_and_override() -> None:
    session, hand, actions, players = _sample_hand()
    hand.hero_position = "button"  # alias should normalize to BTN
    prompt = build_hand_review_prompt(session, hand, actions, players)
    assert "position: BTN" in prompt

    # Explicit override wins over hand.hero_position.
    override = build_hand_review_prompt(
        session, hand, actions, players, hero_position="CO"
    )
    assert "position: CO" in override
    assert get_preflop_range("CO", "rfi").notation in override


def test_hand_prompt_omits_hero_range_for_unknown_position() -> None:
    session, hand, actions, players = _sample_hand()
    for bad in ("", "STRADDLE"):
        hand.hero_position = bad
        prompt = build_hand_review_prompt(session, hand, actions, players)
        assert "Hero preflop range" not in prompt
        # Rest of the prompt still builds correctly.
        assert POST_SESSION_SAFETY in prompt
        for section in REQUIRED_REVIEW_SECTIONS:
            assert section in prompt

    # BB has no RFI chart, so the default scenario must omit the block too.
    hand.hero_position = "BB"
    assert "Hero preflop range" not in build_hand_review_prompt(
        session, hand, actions, players
    )


def test_session_prompt_contains_required_sections_and_safety() -> None:
    db = PokerDatabase(":memory:")
    db.init_db()
    session = db.create_session(Session(name="Prompt session", platform="Manual"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1, hero_cards="Ah Qs"))
    db.create_action(Action(hand_id=hand.id, street="preflop", player_name="Hero", action_type="raise"))
    histories = [
        format_hand_history(session, hand, db.fetch_actions_by_hand(hand.id), [])
    ]

    prompt = build_session_review_prompt(session, compute_session_stats(db, session.id), histories)

    assert POST_SESSION_SAFETY in prompt
    assert "basic/manual review stats" in prompt
    for section in SESSION_REVIEW_SECTIONS:
        assert section in prompt

    db.close()


def _sample_hand() -> tuple[Session, Hand, list[Action], list[HandPlayer]]:
    session = Session(name="Review", platform="ClubWPT Gold")
    hand = Hand(
        id=1,
        session_id=1,
        hand_number=3,
        hero_position="BTN",
        hero_cards="Ah Qs",
        board_cards="Qd 7s 2c",
        hero_bb_won=-30,
        tags=["RIVER_DECISION"],
    )
    actions = [
        Action(hand_id=1, street="preflop", player_name="Hero", position="BTN", action_type="raise", amount=2.5),
        Action(hand_id=1, street="flop", player_name="Hero", position="BTN", action_type="call", amount=8),
    ]
    players = [HandPlayer(hand_id=1, player_name="Hero", position="BTN", is_hero=True)]
    return session, hand, actions, players
