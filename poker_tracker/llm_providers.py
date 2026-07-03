from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from poker_tracker.models import CoachingResponse, LLMProviderConfig
from poker_tracker.safety import ensure_post_session_prompt


DEFAULT_MODEL = "gpt-4o-mini"


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot generate a coaching response."""


class LLMProvider(Protocol):
    provider_name: str
    model_name: str

    def generate_hand_review(self, prompt: str) -> str:
        """Generate a hand-level post-session coaching review."""

    def generate_session_review(self, prompt: str) -> str:
        """Generate a session-level post-session coaching review."""


class MockLLMProvider:
    provider_name = "mock"
    model_name = "mock-local"

    def generate_hand_review(self, prompt: str) -> str:
        """Return deterministic hand-review sections for offline use and tests."""
        ensure_post_session_prompt(prompt)
        return _structured_response(
            {
                "Hand Summary": "Mock post-session summary based only on the supplied hand history.",
                "Theory Coach": "Review range, position, pot odds, and sizing without inventing solver output.",
                "Exploit Coach": "Use only provided player notes and stored actions; do not infer live tendencies.",
                "EV / Math Notes": "Use supplied math facts only. Do not invent equities or exact EV.",
                "Mistake Severity": "Medium unless the stored result/tags indicate a larger issue.",
                "Best Alternative Line": "Compare one lower-variance line and one value-maximizing line.",
                "Study Lesson": "Write down the key decision point and the assumption behind it.",
                "Next Review Question": "What information in the stored hand most changes the best line?",
            }
        )

    def generate_session_review(self, prompt: str) -> str:
        """Return deterministic session-review sections for offline use and tests."""
        ensure_post_session_prompt(prompt)
        return _structured_response(
            {
                "Session Summary": "Mock post-session session summary from the provided manual stats.",
                "Biggest Leaks": "Prioritize repeated tags and large losing hands.",
                "Best Played Spots": "Review the biggest wins for value maximization.",
                "Theory Study Priorities": "Study range construction, pot odds, and street planning.",
                "Exploit Study Priorities": "Use only provided notes and observed stored hands.",
                "Hands To Review Again": "Revisit unreviewed, big-pot, and river-decision hands.",
                "Next Study Plan": "Pick three hands, write alternatives, then compare the math facts.",
            }
        )


class CloudLLMProvider:
    """OpenAI-compatible cloud provider configured by environment variables.

    API keys are read from the environment and are never stored in prompts or DB rows.
    """

    provider_name = "cloud"

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key
        self.model_name = model_name

    def generate_hand_review(self, prompt: str) -> str:
        ensure_post_session_prompt(prompt)
        return self._complete(prompt)

    def generate_session_review(self, prompt: str) -> str:
        ensure_post_session_prompt(prompt)
        return self._complete(prompt)

    def _complete(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a post-session poker study coach. Never provide "
                        "real-time poker assistance or invent unavailable math."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMProviderError(f"Cloud LLM request failed: {exc}") from exc

        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("Cloud LLM response did not contain message content.") from exc


def provider_config_from_env() -> LLMProviderConfig:
    """Read provider config from environment without exposing secrets."""
    provider_name = os.getenv("POKER_TRACKER_LLM_PROVIDER", "mock").strip().lower() or "mock"
    model_name = os.getenv("POKER_TRACKER_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if provider_name in {"openai", "cloud"} and has_key:
        return LLMProviderConfig(provider_name="cloud", model_name=model_name, has_api_key=True)
    if provider_name in {"openai", "cloud"}:
        return LLMProviderConfig(provider_name="mock", model_name="mock-local", has_api_key=False)
    return LLMProviderConfig(provider_name="mock", model_name="mock-local", has_api_key=has_key)


def get_provider_from_env(preferred_provider: str | None = None) -> LLMProvider:
    """Return a configured provider, falling back to mock when cloud config is missing."""
    requested = (preferred_provider or os.getenv("POKER_TRACKER_LLM_PROVIDER", "mock")).lower()
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("POKER_TRACKER_LLM_MODEL", DEFAULT_MODEL)
    if requested in {"openai", "cloud"} and api_key:
        return CloudLLMProvider(api_key=api_key, model_name=model_name)
    return MockLLMProvider()


def parse_sections(raw_response: str) -> dict[str, str]:
    """Parse simple `Section: content` or markdown heading responses."""
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in raw_response.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        heading = _section_heading(stripped)
        if heading:
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current = heading
            buffer = []
            remainder = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if remainder:
                buffer.append(remainder)
        elif current is not None:
            buffer.append(stripped)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()
    return sections


def build_coaching_response(
    *,
    provider: LLMProvider,
    prompt: str,
    raw_response: str,
    review_type: str,
    hand_id: int | None = None,
    session_id: int | None = None,
) -> CoachingResponse:
    """Create a persistable coaching response model from provider output."""
    return CoachingResponse(
        provider_name=provider.provider_name,
        model_name=provider.model_name,
        raw_prompt=prompt,
        raw_response=raw_response,
        review_type=review_type,
        hand_id=hand_id,
        session_id=session_id,
        parsed_sections=parse_sections(raw_response),
    )


def _structured_response(sections: dict[str, str]) -> str:
    return "\n\n".join(f"{heading}:\n{body}" for heading, body in sections.items())


def _section_heading(line: str) -> str | None:
    cleaned = line.strip("# ").strip()
    if ":" in cleaned:
        candidate = cleaned.split(":", 1)[0].strip()
    else:
        candidate = cleaned
    expected = {
        "Hand Summary",
        "Theory Coach",
        "Exploit Coach",
        "EV / Math Notes",
        "Mistake Severity",
        "Best Alternative Line",
        "Study Lesson",
        "Next Review Question",
        "Session Summary",
        "Biggest Leaks",
        "Best Played Spots",
        "Theory Study Priorities",
        "Exploit Study Priorities",
        "Hands To Review Again",
        "Next Study Plan",
    }
    return candidate if candidate in expected else None
