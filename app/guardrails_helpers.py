"""Output-moderation checks used by moderate_output."""

import re
from typing import Tuple

# 13–16 digits only. Grouped cards are 4-4-4-4, or 15-digit Amex as 4-6-5.
# Neighbor checks keep a match from being a slice of a longer digit run.
_CARD_CANDIDATE = re.compile(
    r"(?:"
    r"(?<!\d)(?<!\d[\s-])"
    r"(?:\d{4}[\s-]\d{6}[\s-]\d{5}|\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4})"
    r"(?!\d)(?![\s-]\d)"
    r"|"
    r"(?<!\d)\d{13,16}(?!\d)"
    r")"
)

_SYSTEM_LEAKS = (
    "system prompt",
    "my instructions",
    "i am programmed",
    "as an ai model",
    "my training",
)

_HARMFUL_PATTERNS = (
    r"\b(password|credit card|ssn)\b.*[:=]",
    r"\d{3}-\d{2}-\d{4}",  # SSN pattern
)


def _is_luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def contains_system_leak(text: str) -> Tuple[bool, str]:
    """Return (is_safe, reason) for leaked system-prompt phrasing."""
    text_lower = text.lower()
    for leak in _SYSTEM_LEAKS:
        if leak in text_lower:
            return False, "Output contains system information leak"
    return True, ""


def contains_credit_card(text: str) -> Tuple[bool, str]:
    """Return (is_safe, reason) when the text contains a Luhn-valid card number."""
    for match in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if _is_luhn_valid(digits):
            return False, "Output contains credit card number"
    return True, ""


def contains_harmful_pattern(text: str) -> Tuple[bool, str]:
    """Return (is_safe, reason) for labeled secrets or an SSN-shaped number."""
    text_lower = text.lower()
    for pattern in _HARMFUL_PATTERNS:
        if re.search(pattern, text_lower):
            return False, "Output contains sensitive information"
    return True, ""
