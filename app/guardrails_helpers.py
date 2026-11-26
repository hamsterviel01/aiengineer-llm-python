"""Output-moderation checks used by moderate_output."""

import functools
import logging
import os
import re
from typing import Tuple

import httpx
import numpy as np

from app.config import FACT_CHECK_MODEL, OLLAMA_BASE_URL
from app.guardrails_semantic import get_embedding

logger = logging.getLogger(__name__)

# Minimum cosine similarity for a response sentence to be checked against a policy paragraph.
RELATED_THRESHOLD = float(os.getenv("RELATED_THRESHOLD", "0.5"))

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
}
_WORD_NUMBER = re.compile(r"\b(" + "|".join(_WORD_NUMBERS) + r")\b")

_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
# A number, optionally a range ("5 to 7", "10-14"), then a time unit: "2 business days", "1-year".
_DURATION = re.compile(
    r"(\d+)(?:\s*(?:to|-)\s*(\d+))?[\s-]*(?:business\s+)?(hour|day|week|month|year)s?\b"
)
# Hours stay hours; longer units are converted to days so "two weeks" equals "14 days".
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

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


def _extract_facts(text: str) -> set[tuple[str, float]]:
    """Return normalized (unit, value) pairs for prices, percentages, and durations."""
    text = _WORD_NUMBER.sub(lambda m: str(_WORD_NUMBERS[m.group(1)]), text.lower())
    facts = {("$", float(v.replace(",", ""))) for v in _MONEY.findall(text)}
    facts |= {("%", float(v)) for v in _PERCENT.findall(text)}
    for start, end, unit in _DURATION.findall(text):
        for value in filter(None, (start, end)):
            if unit == "hour":
                facts.add(("hour", float(value)))
            else:
                facts.add(("day", float(value) * _UNIT_DAYS[unit]))
    return facts


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@functools.lru_cache(maxsize=8)
def _context_chunks(context: str) -> tuple[tuple[str, np.ndarray], ...]:
    """Embed each non-heading paragraph of the context once per distinct context."""
    paragraphs = [p.strip() for p in context.split("\n\n")]
    paragraphs = [p for p in paragraphs if p and not p.startswith("#")]
    return tuple((p, get_embedding(p)) for p in paragraphs)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def is_supported(document: str, claim: str) -> bool:
    """Ask the MiniCheck model whether the document supports the claim.

    bespoke-minicheck expects exactly this "Document: ... Claim: ..." prompt and
    answers "Yes" or "No".
    """
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": FACT_CHECK_MODEL,
            "prompt": f"Document: {document}\nClaim: {claim}",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2},
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip().lower().startswith("yes")


def _closest_chunk(sentence: str, chunks: tuple[tuple[str, np.ndarray], ...]) -> str | None:
    """Return the most similar context paragraph, or None if none reaches RELATED_THRESHOLD."""
    if not chunks:
        return None
    embedding = get_embedding(sentence)
    chunk, similarity = max(
        ((text, _cosine_similarity(embedding, vector)) for text, vector in chunks),
        key=lambda pair: pair[1],
    )
    return chunk if similarity >= RELATED_THRESHOLD else None


def _has_unsupported_sentence(response: str, context: str) -> bool:
    """True if a policy-related sentence is not supported by its closest paragraph."""
    chunks = _context_chunks(context)
    for sentence in _split_sentences(response):
        chunk = _closest_chunk(sentence, chunks)
        if chunk is not None and not is_supported(chunk, sentence):
            return True
    return False


def contains_hallucination(response: str, context: str) -> Tuple[bool, str]:
    """Return (is_safe, reason) for claims not grounded in the context.

    Hard facts (prices, percentages, durations) must appear in the context. Other
    sentences are matched to their most similar context paragraph and flagged when
    the MiniCheck model says that paragraph does not support them. Sentences unrelated
    to every paragraph are skipped. If Ollama is unavailable, the semantic step is
    skipped (fails open).
    """
    if not response.strip() or not context.strip():
        return True, ""
    if not _extract_facts(response) <= _extract_facts(context):
        return False, "Output contains hallucination"
    try:
        if _has_unsupported_sentence(response, context):
            return False, "Output contains hallucination"
    except Exception as e:
        logger.warning(f"Semantic hallucination check skipped: {e}")
    return True, ""
