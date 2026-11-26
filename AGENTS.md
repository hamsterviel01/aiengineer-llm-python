# AGENTS.md

Instructions for AI coding agents working in this repository (`customer-service-ai`: FastAPI + LangChain/Ollama customer service app with guardrails).

## Project map

| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI entry point |
| `app/guardrails.py` | Input/output guardrails |
| `app/guardrails_classifier_model.py` | Guard classifier model |
| `app/guardrails_models.py` | `GuardVerdict` result model shared by the safety guards |
| `app/guardrails_config.py` | Selects the safety guard from `SAFETY_GUARD` |
| `app/config.py` | Configuration |
| `evaluation/evaluate.py` | Evaluation harness |
| `tests/` | Pytest suite (`test_app.py`, `test_evaluate.py`, `test_guardrails_classifier_model.py`) |
| `data/` | Datasets (do not modify unless the task requires it) |
| `README.md` | User-facing docs |

## Required workflow: the change loop

Every change — feature, fix, refactor — MUST go through the six-step loop below.
**Run the full loop 2 times per change**:

- **Pass 1 — Make it work.** Deliver the smallest correct change and prove it with tests.
- **Pass 2 — Make it right.** Re-read your own diff with fresh eyes, then repeat every step: fix gaps, edge cases, naming, error handling, missing tests and stale docs found in pass 1.

Do not skip steps, do not merge passes, and do not move to the next change until both passes are complete and green.

```
for change in changes:
    for pass in [1, 2]:
        understand -> plan -> implement -> test -> update docs
        (use tools at every step)
```

### 1. Understand
- Restate the goal and acceptance criteria in one or two sentences.
- Read the relevant code, tests and docs before editing. Use search tools (`grep`/`rg`, file reads) — never guess at structure.
- Identify callers, dependencies and side effects of what you will touch.
- Pass 2: re-read the pass-1 diff (`git diff`) and list what is weak, missing or untested.

### 2. Plan
- **Propose options and let the user pick.** Before implementing, present 2–4 distinct approaches,
 each with a one-line summary and its main trade-offs (complexity, risk, scope). Recommend one, then
 **stop and wait for the user's choice** — don't start implementing until they pick. Skip this only
 for trivial changes with one obvious solution.
- Write a short numbered plan: files to change, functions to add/modify, tests to add.
- Note risks and how you will verify each step.
- Keep the change small and focused; split large work into separate changes, each with its own 2-pass loop.
- Pass 2: plan only the improvements identified in Understand.

### 3. Implement
- Make the planned edits with file-editing tools; follow existing style.
- Python ≥ 3.11, type hints on public functions, line length 100 (ruff config in `pyproject.toml`).
- No secrets or credentials in code (repo uses Talisman — see `.talismanrc`).
- Do not change unrelated code.
- **Document tricky lines.** For any non-obvious line or block (regex, async/concurrency, LangChain/Ollama calls, Pydantic validators, FastAPI dependencies, OpenTelemetry/Phoenix setup, numeric tricks, workarounds), add a short comment that says *why*, not just what, with a URL to the official docs it relies on:

  ```python
  # Re-raise as HTTPException so FastAPI returns a 422 instead of a 500.
  # Ref: https://fastapi.tiangolo.com/tutorial/handling-errors/
  raise HTTPException(status_code=422, detail=str(exc)) from exc
  ```

  - Keep it to 1–2 lines of description plus a `Ref:` line.
  - Link the primary source: official library docs, the Python docs (https://docs.python.org/3/), a PEP, an RFC or the upstream GitHub issue/PR. Don't use blogs or Stack Overflow unless nothing official exists.
  - Look the link up and open it to check it's live and actually explains the line. Never invent or guess URLs.
  - Leave obvious code uncommented; this is only for lines a reviewer would stop at.

### 4. Test
Run and read the output of every command; fix failures before moving on.

```bash
uv sync --extra dev                  # or: pip install -e ".[dev]"
uv run ruff check .                  # lint
uv run ruff format --check .         # formatting
uv run mypy app evaluation           # types
uv run pytest                        # full test suite
uv run pytest tests/test_app.py -v   # targeted run while iterating
PHOENIX_ENABLED=true uv run evaluation/evaluation.py
```

- Add or update tests for every behaviour change (happy path + at least one edge/failure case).
- Tests that need a live Ollama model should be clearly marked/skipped when unavailable; don't make the suite depend on network access.
- For guardrail or model changes, also run the evaluation in `evaluation/evaluate.py` and compare results to before.
- Pass 2: the full suite must pass, not just targeted tests.

### 5. Update docs
- Update `README.md` for any change to setup, commands, API endpoints, configuration or behaviour.
- Update docstrings and inline comments for changed functions.
- Update this `AGENTS.md` if the workflow, commands or project map change.
- Pass 2: verify docs match the final code (commands actually run, examples are accurate), every tricky line added in this change has a description and a working `Ref:` URL, and the `Ref:` URLs still point to the right page.

### 6. Use tools (throughout)
Tools are not a separate phase — use them at every step:
- **Search/read** to understand (`rg`, `grep`, file reads, `git log`/`git blame`).
- **Edit** with file tools rather than rewriting whole files.
- **Run** linters, type checker, tests and the app (`uv run uvicorn app.main:app --reload`) to verify — never claim something works without running it.
- **Diff** (`git diff`, `git status`) before finishing each pass.

## Definition of done (per change)
- [ ] Pass 1 completed: understand → plan → implement → test → docs
- [ ] Pass 2 completed: understand → plan → implement → test → docs
- [ ] `ruff check`, `ruff format --check`, `mypy` and `pytest` all pass
- [ ] Tests cover the new/changed behaviour
- [ ] README / docstrings / AGENTS.md updated as needed
- [ ] Tricky lines have a short comment with a `Ref:` URL to official docs
- [ ] Short summary written: what changed, why, how it was verified, and what pass 2 improved

## Don'ts
- Don't commit or push unless asked.
- Don't edit `.venv/`, `uv.lock` (except via `uv`), `*.egg-info/` or `data/` without reason.
- Don't disable or delete tests to make the suite pass.
