import pytest
import numpy as np

from evaluation.evaluate import calculate_metrics, cosine_similarity


def test_cosine_similarity_values():
    assert cosine_similarity(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_cosine_similarity_zero_vector_returns_zero():
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0


def test_safe_answer_similarity_ignores_blocked_rows():
    results = [
        {"category": "safe", "blocked": False, "similarity": 0.6, "correct_blocking": True},
        {"category": "safe", "blocked": True, "similarity": 0.0, "correct_blocking": False},
        {"category": "injection", "blocked": True, "similarity": 1.0, "correct_blocking": True},
    ]
    metrics = calculate_metrics(results)
    assert metrics["safe_answer_similarity"] == 0.6
    assert metrics["safe_answered_count"] == 1
