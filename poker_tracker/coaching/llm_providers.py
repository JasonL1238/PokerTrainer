from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from poker_tracker.coaching.coaching_prompts import retained_solver_evidence
from poker_tracker.coaching.grounding import check_grounding, grounding_stale_reason
from poker_tracker.coaching.safety import ensure_post_session_prompt
from poker_tracker.persistence.models import CoachingResponse, LLMProviderConfig

DEFAULT_MODEL = "gpt-4o-mini"
ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-8"

# Shared coach system prompt reused by every cloud provider (OpenAI + Anthropic)
# so the post-session safety framing is identical regardless of backend.
COACH_SYSTEM_PROMPT = (
    "You are a post-session poker study coach. Never provide "
    "real-time poker assistance or invent unavailable math."
)


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot generate a coaching response."""


class LLMProvider(Protocol):
    provider_name: str
    model_name: str

    def generate_hand_review(self, prompt: str) -> str:
        """Generate a hand-level post-session coaching review."""

    def generate_session_review(self, prompt: str) -> str:
        """Generate a session-level post-session coaching review."""


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
                    "content": COACH_SYSTEM_PROMPT,
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


class AnthropicLLMProvider:
    """Anthropic Claude provider using the official `anthropic` Python SDK.

    API keys are read from the environment and are never stored in prompts or DB
    rows. The `anthropic` import is intentionally lazy (done inside `_complete`)
    so this module imports cleanly even when the SDK is not installed.
    """

    provider_name = "anthropic"

    def __init__(self, api_key: str, model_name: str = ANTHROPIC_DEFAULT_MODEL) -> None:
        self.api_key = api_key
        self.model_name = model_name

    def generate_hand_review(self, prompt: str) -> str:
        ensure_post_session_prompt(prompt)
        return self._complete(prompt)

    def generate_session_review(self, prompt: str) -> str:
        ensure_post_session_prompt(prompt)
        return self._complete(prompt)

    def _complete(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:  # SDK not installed in this environment
            raise LLMProviderError(
                "The 'anthropic' package is required for AnthropicLLMProvider. "
                "Install it with: pip install anthropic"
            ) from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        try:
            with client.messages.stream(
                model=self.model_name,
                max_tokens=4096,
                system=COACH_SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
        except (anthropic.APIConnectionError, anthropic.APIError) as exc:
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - surface any SDK error uniformly
            raise LLMProviderError(f"Anthropic request failed: {exc}") from exc

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise LLMProviderError("Anthropic response did not contain any text content.")
        return text


def provider_config_from_env() -> LLMProviderConfig:
    """Read provider config from environment without exposing secrets."""
    provider_name = (
        os.getenv("POKER_TRACKER_LLM_PROVIDER", "openai").strip().lower() or "openai"
    )
    requested_model = os.getenv("POKER_TRACKER_LLM_MODEL", "").strip()
    model_name = requested_model or DEFAULT_MODEL
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if provider_name in {"anthropic", "claude"}:
        return LLMProviderConfig(
            provider_name="anthropic",
            model_name=requested_model or ANTHROPIC_DEFAULT_MODEL,
            has_api_key=has_anthropic_key,
        )
    if provider_name in {"openai", "cloud"}:
        return LLMProviderConfig(
            provider_name="cloud",
            model_name=model_name,
            has_api_key=has_openai_key,
        )
    return LLMProviderConfig(
        provider_name=provider_name,
        model_name=requested_model,
        has_api_key=False,
    )


def get_provider_from_env(preferred_provider: str | None = None) -> LLMProvider:
    """Return the requested real provider or fail clearly when it is not configured."""
    requested = (
        preferred_provider or os.getenv("POKER_TRACKER_LLM_PROVIDER", "openai")
    ).strip().lower()
    requested_model = os.getenv("POKER_TRACKER_LLM_MODEL", "").strip()
    if requested in {"anthropic", "claude"}:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            return AnthropicLLMProvider(
                api_key=anthropic_key,
                model_name=requested_model or ANTHROPIC_DEFAULT_MODEL,
            )
        raise LLMProviderError(
            "Anthropic coaching is unavailable until ANTHROPIC_API_KEY is configured."
        )
    if requested in {"openai", "cloud"}:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return CloudLLMProvider(
                api_key=api_key,
                model_name=requested_model or DEFAULT_MODEL,
            )
        raise LLMProviderError(
            "OpenAI coaching is unavailable until OPENAI_API_KEY is configured."
        )
    raise LLMProviderError(f"Unsupported coaching provider: {requested or '(empty)'}.")


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
    """Create a persistable coaching response model from provider output.

    The grounding check runs here rather than at each call site because this is
    the one constructor standing between a provider's text and the database. A
    check a caller has to remember is a check some future caller forgets, and the
    forgetting is silent: the row still stores, still renders, and still reads as
    reviewed analysis. There is nothing to forget if there is nowhere to put it.

    A response that fails is still built and still persisted. The operator paid
    for it and has to be able to read what the provider actually said, and a
    rejection nobody can see is indistinguishable from no check at all. It is
    built stale instead, which is this schema's existing word for "retained, not
    current analysis": stale rows are filtered out of the current-review list,
    are rendered with their reason, and raise STALE_COACHING_EVIDENCE so the hand
    cannot be called studied on the strength of them.
    """
    report = check_grounding(
        prompt,
        raw_response,
        solver_evidence=retained_solver_evidence(prompt),
    )
    return CoachingResponse(
        provider_name=provider.provider_name,
        model_name=provider.model_name,
        raw_prompt=prompt,
        raw_response=raw_response,
        review_type=review_type,
        hand_id=hand_id,
        session_id=session_id,
        parsed_sections=parse_sections(raw_response),
        is_stale=not report.ok,
        stale_reason=grounding_stale_reason(report),
    )


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
