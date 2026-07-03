from poker_tracker.coaching_prompts import (
    build_hand_review_prompt,
    build_session_review_prompt,
)
from poker_tracker.analytics import compute_session_stats
from poker_tracker.db import PokerDatabase
from poker_tracker.hand_history import format_hand_history
from poker_tracker.llm_providers import (
    MockLLMProvider,
    build_coaching_response,
    get_provider_from_env,
    parse_sections,
    provider_config_from_env,
)
from poker_tracker.models import Action, Hand, Session
from poker_tracker.safety import validate_post_session_prompt


def test_mock_provider_hand_review_generation() -> None:
    prompt = _hand_prompt()
    response = MockLLMProvider().generate_hand_review(prompt)

    assert "Hand Summary:" in response
    assert "EV / Math Notes:" in response
    assert "Do not invent equities" in response


def test_mock_provider_session_review_generation() -> None:
    prompt = _session_prompt()
    response = MockLLMProvider().generate_session_review(prompt)

    assert "Session Summary:" in response
    assert "Biggest Leaks:" in response
    assert "Next Study Plan:" in response


def test_provider_fallback_when_cloud_config_missing(monkeypatch) -> None:
    monkeypatch.setenv("POKER_TRACKER_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = provider_config_from_env()
    provider = get_provider_from_env("cloud")

    assert config.provider_name == "mock"
    assert config.has_api_key is False
    assert isinstance(provider, MockLLMProvider)


def test_prompt_safety_validation() -> None:
    safe = validate_post_session_prompt(_hand_prompt())
    unsafe = validate_post_session_prompt("Give current-hand recommendation from live table capture.")

    assert safe.is_safe is True
    assert unsafe.is_safe is False


def test_prompt_contains_post_session_constraints_and_no_invention_rule() -> None:
    prompt = _hand_prompt()

    assert "post-session" in prompt
    assert "completed hands" in prompt
    assert "Do not invent equities" in prompt
    assert "solver outputs" in prompt


def test_hand_review_can_be_stored_and_fetched() -> None:
    db = PokerDatabase(":memory:")
    db.init_db()
    session = db.create_session(Session(name="LLM test"))
    hand = db.create_hand(Hand(session_id=session.id, hand_number=1, hero_cards="Ah Qs"))
    prompt = build_hand_review_prompt(session, hand, [], [])
    provider = MockLLMProvider()
    raw = provider.generate_hand_review(prompt)

    saved = db.create_coaching_response(
        build_coaching_response(
            provider=provider,
            prompt=prompt,
            raw_response=raw,
            review_type="hand",
            hand_id=hand.id,
            session_id=session.id,
        )
    )
    fetched = db.fetch_coaching_reviews_by_hand(hand.id)

    assert saved.id is not None
    assert fetched[0].provider_name == "mock"
    assert fetched[0].parsed_sections["Hand Summary"]
    assert "OPENAI_API_KEY" not in fetched[0].raw_prompt

    db.close()


def test_session_review_can_be_stored_and_fetched() -> None:
    db = PokerDatabase(":memory:")
    db.init_db()
    session = db.create_session(Session(name="Session review"))
    prompt = build_session_review_prompt(session, compute_session_stats(db, session.id), [])
    provider = MockLLMProvider()
    raw = provider.generate_session_review(prompt)

    db.create_coaching_response(
        build_coaching_response(
            provider=provider,
            prompt=prompt,
            raw_response=raw,
            review_type="session",
            session_id=session.id,
        )
    )

    fetched = db.fetch_coaching_reviews_by_session(session.id)
    assert fetched[0].review_type == "session"
    assert fetched[0].parsed_sections["Session Summary"]

    db.close()


def test_parse_sections() -> None:
    parsed = parse_sections("Hand Summary:\nA\n\nTheory Coach:\nB")
    assert parsed == {"Hand Summary": "A", "Theory Coach": "B"}


def _hand_prompt() -> str:
    session = Session(name="Prompt", platform="Manual")
    hand = Hand(
        id=1,
        session_id=1,
        hand_number=1,
        hero_cards="Ah Qs",
        board_cards="Qd 7s 2c",
        tags=["RIVER_DECISION"],
    )
    action = Action(hand_id=1, street="preflop", player_name="Hero", action_type="raise")
    return build_hand_review_prompt(session, hand, [action], [])


def _session_prompt() -> str:
    session = Session(name="Prompt", platform="Manual")
    hand = Hand(id=1, session_id=1, hand_number=1, hero_cards="Ah Qs")
    history = format_hand_history(session, hand, [], [])
    db = PokerDatabase(":memory:")
    db.init_db()
    saved_session = db.create_session(session)
    prompt = build_session_review_prompt(
        saved_session,
        compute_session_stats(db, saved_session.id),
        [history],
    )
    db.close()
    return prompt
