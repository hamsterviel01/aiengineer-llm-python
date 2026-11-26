# Agent Instructions

## Workflow

Follow these steps in order for every non-trivial task. Do not skip a step that needs the user's input.

1. **Understand** – Read the relevant code (`app/`, `tests/`, `evaluation/`) and restate the problem in a few sentences. Ask if anything is unclear.
2. **Propose action plans** – Give the user 2–3 different plans, each with a short description, the files it touches, and its trade-offs. **Wait for the user to choose one** before writing code.
3. **Write tests first** – Add test cases in `tests/` for the chosen plan.
   - Name tests `test_input_<something>_expect_<what>`, e.g. `test_input_prompt_injection_expect_blocked`, `test_input_empty_message_expect_400`.
   - Cover the main case, edge cases, and failure cases.
4. **Ask the user to verify the tests** – Show the test names and what each one checks. **Wait for approval** before writing the implementation.
5. **Implement** the chosen plan.
6. **Run tests** – `pytest tests/test_app.py -v`. Fix failures before moving on.
<!-- 7. **Run evaluation** – `python evaluation/evaluate.py --sample-size 20`. Report the key metrics and compare them with the previous results if available. -->
8. **Review the code** – Check the changes for:
   - correctness (logic errors, edge cases, error handling)
   - redundant code (duplicate logic, unused imports/variables, dead code)
   Fix the issues and report what changed.

## Second-pass review with a smaller model

After step 8, repeat **review → run tests → run evaluation** using a different, smaller model.

- **Ask the user to choose** which smaller model to use (list the available options). Do not pick one yourself.
- Report anything the second pass found that the first pass missed.

## Documentation

- Add short docstrings or comments only where the code is hard to understand (non-obvious logic, regexes, thresholds, guardrail rules, prompt construction).
- Don't comment code that is already easy to read.
- Document especially regex, with 1 sentence explanation and examples of match, not match strings.

## Keep changes small

- Change as little existing code as possible. Prefer small, targeted edits.
- Don't rename, reformat, or restructure code you didn't need to touch.
- Only do large refactors when the user explicitly asks for one.
