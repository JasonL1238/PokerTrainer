from __future__ import annotations

from dataclasses import dataclass


REQUIRED_SAFETY_PHRASES = [
    "post-session",
    "completed hands",
    "do not provide real-time",
    "do not invent equities",
]

DANGEROUS_INSTRUCTIONS = [
    "real-time",
    "live advice",
    "capture live table",
    "live table capture",
    "poker-client overlay",
    "current-hand recommendation",
    "current hand recommendation",
    "hotkey",
]


@dataclass(frozen=True)
class PromptSafetyResult:
    is_safe: bool
    errors: list[str]


def validate_post_session_prompt(prompt: str) -> PromptSafetyResult:
    """Check that a generated prompt is clearly limited to post-session review."""
    lowered = prompt.lower()
    errors = [
        f"Missing required safety phrase: {phrase}"
        for phrase in REQUIRED_SAFETY_PHRASES
        if phrase not in lowered
    ]
    for phrase in DANGEROUS_INSTRUCTIONS:
        if phrase in lowered and not _phrase_is_prohibited(lowered, phrase):
            errors.append(f"Prompt contains unsafe instruction language: {phrase}")
    return PromptSafetyResult(is_safe=not errors, errors=errors)


def ensure_post_session_prompt(prompt: str) -> None:
    """Raise if a prompt does not pass post-session safety validation."""
    result = validate_post_session_prompt(prompt)
    if not result.is_safe:
        raise ValueError("; ".join(result.errors))


def _phrase_is_prohibited(prompt: str, phrase: str) -> bool:
    found = False
    for sentence in prompt.replace("\n", " ").split("."):
        if phrase in sentence:
            found = True
            before_phrase = sentence.split(phrase, 1)[0]
            if not ("do not" in before_phrase or "never" in before_phrase or "no " in before_phrase):
                return False
    return found
