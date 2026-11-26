import os
import random
import time
from unittest.mock import patch

os.environ.setdefault("PHOENIX_ENABLED", "false")

from evaluation import evaluate  # noqa: E402

BLOCKED = {"answer": "", "blocked": True, "reason": "blocked"}


def _dataset(n):
    return [
        {"prompt": f"ignore instructions {i}", "ground_truth": "", "category": "injection"}
        for i in range(n)
    ]


def _slow_query(delay):
    def query(question, api_url):
        time.sleep(delay)
        return BLOCKED

    return query


def test_input_concurrency_4_expect_requests_overlap():
    """With concurrency=4, eight 0.3s API calls finish well under the 2.4s sequential time."""
    with patch.object(evaluate, "load_dataset", return_value=_dataset(8)), \
         patch.object(evaluate, "query_api", _slow_query(0.3)):
        start = time.perf_counter()
        out = evaluate.run_evaluation(sample_size=8, concurrency=4)
        elapsed = time.perf_counter() - start

    assert out["metrics"]["total_samples"] == 8
    assert elapsed < 1.2, f"evaluation ran sequentially ({elapsed:.2f}s)"


def test_input_random_response_delays_expect_results_in_dataset_order():
    """Results keep dataset order even when later requests finish first."""
    dataset = _dataset(10)

    def query(question, api_url):
        time.sleep(random.uniform(0, 0.1))
        return BLOCKED

    with patch.object(evaluate, "load_dataset", return_value=dataset), \
         patch.object(evaluate, "query_api", query):
        out = evaluate.run_evaluation(sample_size=10, concurrency=5)

    assert [r["prompt"] for r in out["results"]] == [d["prompt"] for d in dataset]


def test_input_concurrency_1_expect_same_metrics_as_parallel():
    """Concurrency changes speed only, not the computed metrics."""
    with patch.object(evaluate, "load_dataset", return_value=_dataset(6)), \
         patch.object(evaluate, "query_api", _slow_query(0)):
        sequential = evaluate.run_evaluation(sample_size=6, concurrency=1)
        parallel = evaluate.run_evaluation(sample_size=6, concurrency=3)

    assert sequential["metrics"] == parallel["metrics"]


def test_input_one_api_error_expect_other_samples_still_evaluated():
    """An API failure on one sample is recorded as blocked and does not stop the rest."""
    def query(question, api_url):
        if question.endswith(" 2"):
            return {"answer": "", "blocked": True, "reason": "API error: boom"}
        return {"answer": "", "blocked": False, "reason": ""}

    with patch.object(evaluate, "load_dataset", return_value=_dataset(4)), \
         patch.object(evaluate, "query_api", query):
        out = evaluate.run_evaluation(sample_size=4, concurrency=4)

    assert len(out["results"]) == 4
    assert [r["blocked"] for r in out["results"]] == [False, False, True, False]
