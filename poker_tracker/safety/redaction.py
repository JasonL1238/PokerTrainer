"""Scrub credentials out of text that gets persisted or displayed.

Job error messages, diagnostics, and release reports are all written somewhere
durable and later read by a human — or attached to an issue bundle and sent
somewhere else. Any of them can end up carrying a provider key, because the
exception that produced them was raised by a client library that helpfully
included the request it was making.

Truncation is not redaction. Bounding a message to 500 characters removes a
secret only when the secret happens to sit past character 500; a key at the
front is stored in full. This module removes secrets by shape, then the caller
bounds what is left.
"""

from __future__ import annotations

import os
import re

REDACTED = "<redacted>"

# Token shapes worth matching on sight, independent of any surrounding key name.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Anthropic / OpenAI style prefixed keys.
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}", re.IGNORECASE),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{8,}", re.IGNORECASE),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # JSON Web Tokens.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    # Bearer/authorization headers.
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}"),
    # user:password@host in a URL.
    re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/@:]+:[^\s/@]+@"),
)

# ``key = value`` / ``"key": "value"`` / ``key => value`` where the key names a
# credential. ``=>`` is listed before ``[:=]`` so the alternation cannot match a
# bare ``=`` and leave a stray ``>`` in front of the value it was meant to hide.
_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    \b
    (?P<key>[A-Za-z0-9_.\-]*
        (?:password|passwd|secret|token|api[_\-]?key|apikey|credential|
           auth|access[_\-]?key|private[_\-]?key|session[_\-]?id)
        [A-Za-z0-9_.\-]*)
    (?P<keyquote>["']?)
    (?P<sep>\s*(?:=>|[:=])\s*)
    (?P<quote>["']?)
    (?P<value>[^\s"',;}\)]+)
    (?P=quote)
    """
)

# Environment variables whose literal values must never appear in output, even
# when they carry no recognizable shape. A password of "hunter2" matches no
# pattern above, but it is still the operator's password.
_SENSITIVE_ENV_NAMES = re.compile(
    r"(?i)(password|secret|token|api[_-]?key|credential|auth)"
)
# Below this length, a value is too short to redact by literal match without
# scrambling ordinary prose (an env var set to "1" or "on" would blank out every
# digit in the message).
_MIN_LITERAL_LENGTH = 6


def _environment_literals() -> list[str]:
    literals = []
    for name, value in os.environ.items():
        if not _SENSITIVE_ENV_NAMES.search(name):
            continue
        cleaned = (value or "").strip()
        if len(cleaned) >= _MIN_LITERAL_LENGTH:
            literals.append(cleaned)
    # Longest first, so a secret containing another secret redacts completely.
    return sorted(literals, key=len, reverse=True)


def redact_text(text: str, *, include_environment: bool = True) -> str:
    """Remove credential-shaped substrings and configured secret values."""
    if not text:
        return text
    scrubbed = text
    if include_environment:
        for literal in _environment_literals():
            scrubbed = scrubbed.replace(literal, REDACTED)
    # Token shapes first. Running the assignment rule first would consume
    # "Authorization: Bearer" as a key/value pair, redact the word "Bearer", and
    # leave the actual token standing in the message.
    for pattern in _TOKEN_PATTERNS:
        scrubbed = pattern.sub(_replace_token, scrubbed)
    scrubbed = _ASSIGNMENT_PATTERN.sub(
        lambda m: (
            f"{m.group('key')}{m.group('keyquote')}{m.group('sep')}{REDACTED}"
        ),
        scrubbed,
    )
    return scrubbed


def _replace_token(match: re.Match[str]) -> str:
    # DSN credentials keep the scheme so the message still identifies the host.
    if match.lastindex and match.group(1) and match.group(1).endswith("://"):
        return f"{match.group(1)}{REDACTED}@"
    return REDACTED


def safe_error_message(exc: BaseException, *, limit: int = 500) -> str:
    """A single-line, secret-free, bounded description of a failure.

    Order matters: redact first, then bound. Bounding first would leave a secret
    intact whenever it appeared before the cutoff, which is exactly where an
    exception from a client library tends to put it.
    """
    raw = str(exc)
    scrubbed = redact_text(raw).replace("\n", " ").replace("\r", " ")
    collapsed = re.sub(r"\s{2,}", " ", scrubbed).strip()
    if not collapsed:
        return type(exc).__name__
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"
