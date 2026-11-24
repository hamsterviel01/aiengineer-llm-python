"""Guardrail utilities for input filtering and safety checks."""

import re
from typing import Tuple

from app.guardrails_models import GuardVerdict

OFF_TOPIC_KEYWORDS = [
    "weather", "sports", "politics", "religion",
    "movie", "recipe", "travel destination", "homework"
]


def filter_input(text: str) -> Tuple[bool, str]:
    """
    Filter and sanitize user input.
    
    Returns:
        Tuple of (is_valid, filtered_text)
    """
    if not text or len(text.strip()) == 0:
        return False, "Input cannot be empty"
    
    if len(text) > 1000:
        return False, "Input is too long (max 1000 characters)"
    
    filtered = text.strip()
    return True, filtered


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    """
    Detect potential prompt injection attempts.
    
    Returns:
        Tuple of (is_injection_detected, reason)
    """
    text_lower = text.lower()
    
    # Common prompt injection patterns
    injection_patterns = [
        r"ignore (?:\w+\s+)?(instructions|rules|prompts)",
        r"ignore all (previous|above)",
        r"disregard (previous|above|all)",
        r"forget (your|previous|all) (instructions|rules|training)",
        r"you are now",
        r"new (role|instructions|system)",
        r"system\s*:+",
        r"<\s*prompt\s*>",
        r"act as a"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return True, "Prompt injection attempt detected"
    
    return False, ""


def detect_policy_violation(text: str) -> Tuple[bool, str]:
    """
    Detect policy-violating content.
    
    Policy violations include:
    - Harmful/illegal content
    - Personal data requests
    - Off-topic queries
    
    Returns:
        Tuple of (is_violation_detected, reason)
    """
    text_lower = text.lower()
    
    # Harmful/illegal content patterns
    harmful_keywords = [
        "hack", "crack", "steal", "illegal", "drug", "weapon",
        "violence", "harm", "kill", "murder", "suicide"
    ]
    
    # Personal data request patterns
    personal_data_keywords = [
        "social security", "ssn", "credit card", "password",
        "personal information", "private data", "confidential"
    ]
    
    checks = [
        (harmful_keywords, "Policy violation: Harmful/illegal content"),
        (personal_data_keywords, "Policy violation: Personal data request"),
    ]
    for keywords, reason in checks:
        if _matches_any_word(text_lower, keywords):
            return True, reason

    return detect_off_topic(text)


def detect_off_topic(text: str) -> Tuple[bool, str]:
    """
    Detect queries unrelated to e-commerce customer service.

    Returns:
        Tuple of (is_off_topic, reason)
    """
    if _matches_any_word(text.lower(), OFF_TOPIC_KEYWORDS):
        return True, "Policy violation: Off-topic query"
    return False, ""


def _matches_any_word(text: str, keywords: list[str]) -> bool:
    """Match whole words plus common endings: "harm" hits "harmful", not "pharmacy"."""
    words = "|".join(re.escape(k) for k in keywords)
    return re.search(rf"\b(?:{words})(?:s|es|ed|ing|ful|er|ers)?\b", text) is not None


def moderate_output(text: str) -> Tuple[bool, str]:
    """
    Moderate LLM output for safety issues.
    
    Returns:
        Tuple of (is_safe, reason_if_unsafe)
    """
    text_lower = text.lower()
    
    # Check for leaked system information
    system_leaks = [
        "system prompt", "my instructions", "i am programmed",
        "as an ai model", "my training"
    ]
    
    for leak in system_leaks:
        if leak in text_lower:
            return False, "Output contains system information leak"
    
    # Check for potentially harmful output
    harmful_patterns = [
        r"\b(password|credit card|ssn)\b.*[:=]",
        r"\d{3}-\d{2}-\d{4}",  # SSN pattern
        r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",  # Credit card pattern
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, text_lower):
            return False, "Output contains sensitive information"
    
    return True, ""


def classify_input(user_message: str) -> GuardVerdict:
    """
    Keyword safety guard for customer messages (same interface as guardrails_classifier_model).

    Returns:
        GuardVerdict(is_safe, reason_if_unsafe, violated_category_codes); keyword checks have
        no category codes, so the list is always empty.
    """
    is_violation, reason = detect_policy_violation(user_message)
    return GuardVerdict(not is_violation, reason, [])


def classify_output(user_message: str, agent_response: str) -> GuardVerdict:
    """
    Keyword safety guard for LLM responses (same interface as guardrails_classifier_model).

    Always passes: keyword output checks already run in moderate_output.

    Returns:
        GuardVerdict(is_safe, reason_if_unsafe, violated_category_codes)
    """
    return GuardVerdict(True, "", [])

