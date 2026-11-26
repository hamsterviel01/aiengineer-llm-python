"""Print attack recall and safe block rate per margin on the semantic dev set.

Usage: python evaluation/tune_semantic_margin.py   (needs Ollama with nomic-embed-text)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.semantic_guard import ATTACK_EXAMPLES, SAFE_EXAMPLES, semantic_score  # noqa: E402


def main():
    dev = json.loads((ROOT / "evaluation" / "semantic_dev_set.json").read_text())
    eval_prompts = {r["prompt"].lower() for r in json.loads((ROOT / "data" / "qa_dataset.json").read_text())}
    overlap = [t for t in ATTACK_EXAMPLES + SAFE_EXAMPLES + dev["attack"] + dev["safe"] if t.lower() in eval_prompts]
    assert not overlap, f"Examples leak from the eval set: {overlap}"

    attack = sorted(semantic_score(t)[0] for t in dev["attack"])
    safe = sorted(semantic_score(t)[0] for t in dev["safe"])
    print(f"attack scores: min {attack[0]:.3f} max {attack[-1]:.3f} (n={len(attack)})")
    print(f"safe scores:   min {safe[0]:.3f} max {safe[-1]:.3f} (n={len(safe)})")
    for m in [x / 100 for x in range(-10, 16, 2)]:
        recall = sum(s > m for s in attack) / len(attack)
        false_block = sum(s > m for s in safe) / len(safe)
        print(f"margin {m:+.2f}: attack recall {recall:.2f}  safe blocked {false_block:.2f}")


if __name__ == "__main__":
    main()
