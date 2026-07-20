from poker_tracker.coaching_eval import (
    UNSAFE_PHRASES,
    CoachingEvalReport,
    run_coaching_eval,
)
from poker_tracker.coaching_prompts import REQUIRED_REVIEW_SECTIONS
from poker_tracker.llm_providers import (
    ANTHROPIC_DEFAULT_MODEL,
    AnthropicLLMProvider,
    MockLLMProvider,
    get_provider_from_env,
    provider_config_from_env,
)


def test_mock_provider_passes_all_golden_hands() -> None:
    report = run_coaching_eval(MockLLMProvider())

    assert isinstance(report, CoachingEvalReport)
    assert report.total == 5
    assert report.all_passed, [
        (case.name, case.failures) for case in report.cases if not case.passed
    ]
    assert report.passed_count == report.total
    for case in report.cases:
        assert case.failures == []


def test_mock_report_has_all_required_sections() -> None:
    report = run_coaching_eval(MockLLMProvider())

    # The mock returns every required section, so no case should report a
    # missing-section failure.
    for case in report.cases:
        for section in REQUIRED_REVIEW_SECTIONS:
            assert f"Missing required section: {section}" not in case.failures


def test_anthropic_provider_falls_back_to_mock_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    provider = get_provider_from_env("anthropic")

    assert isinstance(provider, MockLLMProvider)


def test_claude_alias_also_falls_back_to_mock_without_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert isinstance(get_provider_from_env("claude"), MockLLMProvider)


def test_provider_config_returns_anthropic_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("POKER_TRACKER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("POKER_TRACKER_LLM_MODEL", raising=False)

    config = provider_config_from_env()

    assert config.provider_name == "anthropic"
    assert config.model_name == ANTHROPIC_DEFAULT_MODEL
    assert config.has_api_key is True


def test_provider_config_anthropic_honors_model_override(monkeypatch) -> None:
    monkeypatch.setenv("POKER_TRACKER_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("POKER_TRACKER_LLM_MODEL", "claude-custom")

    config = provider_config_from_env()

    assert config.provider_name == "anthropic"
    assert config.model_name == "claude-custom"


def test_provider_config_anthropic_without_key_is_mock(monkeypatch) -> None:
    monkeypatch.setenv("POKER_TRACKER_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = provider_config_from_env()

    assert config.provider_name == "mock"
    assert config.has_api_key is False


def test_anthropic_provider_selected_when_key_present(monkeypatch) -> None:
    # Construction only — this must NOT make any network call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("POKER_TRACKER_LLM_MODEL", "claude-opus-4-8")

    provider = get_provider_from_env("anthropic")

    assert isinstance(provider, AnthropicLLMProvider)
    assert provider.provider_name == "anthropic"
    assert provider.model_name == "claude-opus-4-8"


def test_anthropic_provider_public_surface() -> None:
    provider = AnthropicLLMProvider(api_key="test-key")

    assert provider.provider_name == "anthropic"
    assert provider.model_name == ANTHROPIC_DEFAULT_MODEL
    assert hasattr(provider, "generate_hand_review")
    assert hasattr(provider, "generate_session_review")


def test_eval_uses_a_broad_set_of_safety_phrases() -> None:
    # Guardrail so the safety keyword list is not accidentally emptied.
    assert "current hand" in UNSAFE_PHRASES
    assert "in real time" in UNSAFE_PHRASES
    assert len(UNSAFE_PHRASES) >= 5
