import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

SLOW_LLM_SECONDS = 0.5


def slow_invoke(prompt):
    time.sleep(SLOW_LLM_SECONDS)  # a blocking call, like OllamaLLM.invoke
    return "Your order ships in 2 days."


def test_slow_llm_call_does_not_block_other_requests():
    """Two queries with a slow LLM must run in parallel, not one after the other."""
    with patch("app.main.semantic_decision", return_value=("allow", "")), \
         patch("app.main.llm") as mock_llm, TestClient(app) as client:
        mock_llm.invoke.side_effect = slow_invoke
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(
                lambda q: client.post("/query", json={"question": q}),
                ["Where is my order?", "When will my parcel arrive?"],
            ))
        elapsed = time.perf_counter() - start

    assert all(r.status_code == 200 and not r.json()["blocked"] for r in responses)
    assert elapsed < 1.8 * SLOW_LLM_SECONDS, f"requests ran one after the other: {elapsed:.2f}s"
