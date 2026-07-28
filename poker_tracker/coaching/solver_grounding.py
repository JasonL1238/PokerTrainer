from __future__ import annotations

import re

from poker_tracker.solver.models import ActionFrequency, SolverEvidence

_PERCENT = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%")
_BB_UNIT = r"(?:bb|big\s+blinds?)"
_ACTION_WORD = re.compile(
    r"(?i)\b(?:check(?:ing)?|bet(?:ting)?|call(?:ing)?|rais(?:e|ing)|fold(?:ing)?|"
    r"solver-action)\b"
)
_CLAIM_ACTION = re.compile(
    r"(?ix)\b"
    r"(?P<kind>check(?:ing)?|bet(?:ting)?|call(?:ing)?|rais(?:e|ing)|"
    r"fold(?:ing)?|jam(?:ming)?|shov(?:e|ing)|all[\s-]?in)"
    r"\b"
)
_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"
)
_POKER_UNIT_AMOUNT = re.compile(
    rf"(?ix)(?:[-+]?\d+(?:\.\d+)?|{_NUMBER_WORD}(?:[-\s]+{_NUMBER_WORD})*)"
    r"\s*(?:bb|blinds?|big\s+blinds?|chips?)\b"
)
_LOSS_CONTEXT = re.compile(
    r"(?ix)\b(?:"
    r"los(?:e|es|ing|t)|loss|costs?|burns?|saves?|gains?|earns?|adds?|"
    r"sacrifices?|forfeits?|surrenders?|gives?\s+(?:up|away)|"
    r"underperforms?|inferior|worth|less|lower|higher|worse|better|"
    r"mistake|difference|expectation|ev"
    r")\b"
)
_UNSUPPORTED_BB_LOSS = re.compile(
    r"(?ix)"
    r"(?:"
    r"(?:los(?:e|es|ing)|costs?|saves?|gains?|burns?|sacrifices?|"
    r"forfeits?|surrenders?|gives?\s+up)\s+(?:exactly\s+)?"
    r"(?:the\s+)?[-+]?\d+(?:\.\d+)?\s*" + _BB_UNIT
    + r"|[-+]?\d+(?:\.\d+)?\s*" + _BB_UNIT
    + r"\s+(?:of\s+)?(?:loss|cost|ev|worse|better)"
    r"|(?:action\s+)?ev\s*(?:difference|gap|edge)?\s*(?:is|=|of)?\s*"
    r"[-+]?\d+(?:\.\d+)?(?:\s*" + _BB_UNIT + r")?"
    r"|(?:mistake|action|line)\s+(?:costs?|loses?|burns?)\s+"
    r"[-+]?\d+(?:\.\d+)?\s*" + _BB_UNIT
    + r"|\bev\b[^.!?\n]{0,60}[-+]?\d+(?:\.\d+)?\s*" + _BB_UNIT
    + r"|[-+]?\d+(?:\.\d+)?\s*" + _BB_UNIT
    + r"[^.!?\n]{0,30}(?:lower|higher)\s+\bev\b"
    r"|difference\s+between[^.!?\n]{0,80}[-+]?\d+(?:\.\d+)?\s*" + _BB_UNIT
    + r"|(?:check(?:ing)?|bet(?:ting)?|call(?:ing)?|rais(?:e|ing)|fold(?:ing)?)"
    r"[^.!?\n]{0,40}\s-\d+(?:\.\d+)?\s*" + _BB_UNIT
    + r")"
)


def validate_solver_coaching_response(
    response: str,
    evidence: SolverEvidence,
) -> None:
    """Reject explanations that alter or exceed the saved solver evidence."""

    if _contains_action_amount_claim(response, evidence) or _UNSUPPORTED_BB_LOSS.search(
        response
    ):
        raise ValueError(
            "The explanation claimed action EV or BB loss that TexasSolver's "
            "saved strategy output does not establish."
        )

    frequencies = (
        evidence.action_frequencies
        if evidence.action_frequencies
        else evidence.range_action_frequencies
    )
    _validate_frequency_claims(response, frequencies)


def _contains_action_amount_claim(
    response: str,
    evidence: SolverEvidence,
) -> bool:
    sanitized = _mask_spot_facts(response, evidence)
    for item in [
        *evidence.action_frequencies,
        *evidence.range_action_frequencies,
    ]:
        if " " not in item.action:
            continue
        action = _action_expression(item.action)
        sanitized = re.sub(
            rf"(?i)(?<!\w){action}(?!\w)",
            "solver-action",
            sanitized,
        )
    for sentence in re.split(r"(?:[!?]|\.(?!\d)|\n)+", sanitized):
        if _ACTION_WORD.search(sentence) and _POKER_UNIT_AMOUNT.search(sentence):
            return True
    return False


def _validate_frequency_claims(
    response: str,
    frequencies: list[ActionFrequency],
) -> None:
    by_kind: dict[str, list[ActionFrequency]] = {}
    for item in frequencies:
        kind = item.action.split(" ", 1)[0].upper()
        by_kind.setdefault(kind, []).append(item)
    aliases = {
        "check": "CHECK",
        "checking": "CHECK",
        "bet": "BET",
        "betting": "BET",
        "call": "CALL",
        "calling": "CALL",
        "raise": "RAISE",
        "raising": "RAISE",
        "fold": "FOLD",
        "folding": "FOLD",
        "jam": "ALLIN",
        "jamming": "ALLIN",
        "shove": "ALLIN",
        "shoving": "ALLIN",
        "all-in": "ALLIN",
        "all in": "ALLIN",
    }
    matched_actions: set[str] = set()
    for clause in re.split(
        r"(?:[;!?]|\.(?!\d)|\n|\band\b)+",
        response,
        flags=re.IGNORECASE,
    ):
        percentages = [
            (match.start(), match.end(), float(match.group(1)))
            for match in _PERCENT.finditer(clause)
        ]
        if not percentages:
            continue
        for match in _CLAIM_ACTION.finditer(clause):
            raw_kind = re.sub(r"\s+", " ", match.group("kind").lower())
            kind = aliases[raw_kind]
            candidates = by_kind.get(kind, [])
            if not candidates:
                raise ValueError(
                    f"The explanation invented a solver frequency for unsupported action {kind}."
                )
            amount_info = _following_action_amount(clause, match, kind)
            if amount_info is not None:
                claimed_amount, amount_end = amount_info
                candidates = [
                    item
                    for item in candidates
                    if " " in item.action
                    and abs(float(item.action.split(" ", 1)[1]) - claimed_amount)
                    <= 1e-6
                ]
                if not candidates:
                    raise ValueError(
                        f"The explanation invented a solver size for action {kind}."
                    )
            elif len(candidates) > 1:
                raise ValueError(
                    f"The explanation gave an ambiguous solver frequency for action {kind}."
                )
            action_end = amount_end if amount_info is not None else match.end()
            action_center = (match.start() + action_end) / 2
            _, _, claimed_frequency = min(
                percentages,
                key=lambda value: abs(((value[0] + value[1]) / 2) - action_center),
            )
            matched = [
                item
                for item in candidates
                if abs((item.frequency * 100) - claimed_frequency) <= 0.051
            ]
            if not matched:
                raise ValueError(
                    f"The explanation changed the saved solver frequency for action {kind}."
                )
            matched_actions.update(item.action for item in matched)
    missing = [item.action for item in frequencies if item.action not in matched_actions]
    if missing:
        raise ValueError(
            "The explanation omitted the saved solver frequency for "
            + ", ".join(missing)
            + "."
        )


def _following_action_amount(
    clause: str,
    match: re.Match[str],
    kind: str,
) -> tuple[float, int] | None:
    if kind not in {"BET", "RAISE"}:
        return None
    tail = clause[match.end() : match.end() + 24]
    amount_match = re.match(
        r"\s+(?:to\s+)?([-+]?\d+(?:\.\d+)?)(?!\s*%)",
        tail,
        flags=re.IGNORECASE,
    )
    if amount_match is None:
        return None
    return float(amount_match.group(1)), match.end() + amount_match.end()


def _mask_spot_facts(response: str, evidence: SolverEvidence) -> str:
    sanitized = response
    pot = _number_expression(evidence.pot)
    stack = _number_expression(evidence.effective_stack)
    patterns = (
        rf"{pot}\s*BB\s+pot",
        rf"\bpot\s*(?:is|of|:)?\s*{pot}\s*BB",
        rf"{stack}\s*BB\s+effective\s+stack",
        rf"\beffective\s+stack\s*(?:is|of|:)?\s*{stack}\s*BB",
    )
    for pattern in patterns:
        compiled = re.compile(pattern, flags=re.IGNORECASE)

        def replace_if_factual(
            match: re.Match[str],
            current_text: str = sanitized,
        ) -> str:
            start = max(
                current_text.rfind("\n", 0, match.start()),
                current_text.rfind(";", 0, match.start()),
                current_text.rfind("!", 0, match.start()),
                current_text.rfind("?", 0, match.start()),
            )
            endings = [
                index
                for delimiter in ("\n", ";", "!", "?")
                if (index := current_text.find(delimiter, match.end())) >= 0
            ]
            end = min(endings, default=len(current_text))
            context = current_text[start + 1 : end]
            if "%" in context and not _LOSS_CONTEXT.search(context):
                return "spot-fact"
            return match.group(0)

        sanitized = compiled.sub(replace_if_factual, sanitized)
    return sanitized


def _number_expression(value: float) -> str:
    rendered = f"{value:g}"
    if "." in rendered:
        return re.escape(rendered) + r"0*"
    return re.escape(rendered) + r"(?:\.0+)?"


def _action_expression(action: str) -> str:
    kind, _, raw_amount = action.partition(" ")
    aliases = {
        "CHECK": r"check(?:ing)?",
        "BET": r"bet(?:ting)?",
        "CALL": r"call(?:ing)?",
        "RAISE": r"rais(?:e|ing)",
        "FOLD": r"fold(?:ing)?",
    }
    expression = aliases.get(kind.upper(), re.escape(kind))
    if raw_amount:
        amount = f"{float(raw_amount):g}"
        if "." in amount:
            amount_expression = re.escape(amount) + r"0*"
        else:
            amount_expression = re.escape(amount) + r"(?:\.0+)?"
        expression += rf"\s+(?:to\s+)?{amount_expression}(?:\s*BB)?"
    return expression
