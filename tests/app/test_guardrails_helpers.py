from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.guardrails_helpers import _context_chunks, contains_hallucination

POLICY = (Path(__file__).parents[2] / "app" / "store_policy.md").read_text()

RELATED = np.array([1.0, 0.0], dtype=float)


@pytest.fixture
def grounded_semantics():
    """Default: every sentence is related to the policy and supported by it."""
    _context_chunks.cache_clear()
    with (
        patch("app.guardrails_helpers.get_embedding", return_value=RELATED) as embed,
        patch("app.guardrails_helpers.is_supported", return_value=True) as fact_check,
    ):
        yield embed, fact_check
    _context_chunks.cache_clear()


def test_input_fact_matching_policy_expect_safe(grounded_semantics):
    assert contains_hallucination("Standard shipping is free on orders over $50.", POLICY) == (True, "")


def test_input_paraphrased_number_expect_safe(grounded_semantics):
    """Spelled-out numbers and units normalize to the policy's values (30 days, 14 days)."""
    response = "You have thirty days to send it back, and price drops are honored for two weeks."
    assert contains_hallucination(response, POLICY) == (True, "")


def test_input_wrong_day_count_expect_hallucination(grounded_semantics):
    is_safe, reason = contains_hallucination("Items can be returned within 60 days of delivery.", POLICY)
    assert is_safe is False
    assert reason == "Output contains hallucination"


def test_input_wrong_price_expect_hallucination(grounded_semantics):
    is_safe, _ = contains_hallucination("Express shipping costs $9.99.", POLICY)
    assert is_safe is False


def test_input_wrong_percentage_expect_hallucination(grounded_semantics):
    is_safe, _ = contains_hallucination("Opened electronics have a 20% restocking fee.", POLICY)
    assert is_safe is False


def test_input_wrong_fact_expect_no_semantic_calls(grounded_semantics):
    """A wrong hard fact is decided without calling the embedding or fact-check model."""
    embed, fact_check = grounded_semantics
    contains_hallucination("Express shipping costs $9.99.", POLICY)
    embed.assert_not_called()
    fact_check.assert_not_called()


def test_input_contradiction_without_numbers_expect_hallucination(grounded_semantics):
    """Flipped meaning with no numbers is caught by the fact-check model."""
    _, fact_check = grounded_semantics
    fact_check.return_value = False
    is_safe, _ = contains_hallucination("Gift cards can be returned for a full refund.", POLICY)
    assert is_safe is False


def test_input_sentence_unrelated_to_policy_expect_not_fact_checked(grounded_semantics):
    """Small talk below RELATED_THRESHOLD is skipped instead of being sent to the fact-check model."""
    embed, fact_check = grounded_semantics
    unrelated = np.array([0.0, 1.0], dtype=float)
    embed.side_effect = lambda text: unrelated if text == "Happy to help!" else RELATED
    assert contains_hallucination("Happy to help!", POLICY) == (True, "")
    fact_check.assert_not_called()


def test_input_related_sentence_expect_checked_against_closest_paragraph(grounded_semantics):
    embed, fact_check = grounded_semantics
    context = "Para A about returns.\n\nPara B about shipping."
    sentence = "Shipping is fast."
    vectors = {"Para A about returns.": np.array([0.0, 1.0]), "Para B about shipping.": RELATED}
    embed.side_effect = lambda text: vectors.get(text, RELATED)
    contains_hallucination(sentence, context)
    fact_check.assert_called_once_with("Para B about shipping.", sentence)


def test_input_one_bad_sentence_among_good_expect_hallucination(grounded_semantics):
    response = "Standard shipping takes 5 to 7 business days. Overnight shipping costs $19.99."
    is_safe, _ = contains_hallucination(response, POLICY)
    assert is_safe is False


def test_input_empty_response_expect_safe(grounded_semantics):
    assert contains_hallucination("", POLICY) == (True, "")


def test_input_empty_context_expect_safe(grounded_semantics):
    """Without context there is nothing to compare against, so the check does not block."""
    assert contains_hallucination("Express shipping costs $9.99.", "") == (True, "")


def test_input_embedding_service_down_expect_fact_check_still_applies():
    """If Ollama is unavailable, the hard-fact check still blocks wrong numbers."""
    with patch("app.guardrails_helpers.get_embedding", side_effect=RuntimeError("down")):
        is_safe, _ = contains_hallucination("Express shipping costs $9.99.", POLICY)
    assert is_safe is False
