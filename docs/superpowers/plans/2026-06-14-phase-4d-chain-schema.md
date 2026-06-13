# Phase 4D Chain Schema Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional schema-first validation for synthetic two-hop QA generation.

**Architecture:** Extend `synthesis.py` with chain schema parsing and strict checks behind `require_chain_schema`. Thread that option through file validation, validated synthesis, and CLI flags while keeping old data valid by default.

**Tech Stack:** Python 3.10+, pytest, existing LightningSearch-RL CLI.

---

## Chunk 1: Chain Schema Validator

### Task 1: Add strict validator tests

**Files:**
- Modify: `tests/test_synthesis.py`
- Modify: `src/lightningsearch_rl/synthesis.py`

- [ ] Write tests for strict acceptance and rejection cases.
- [ ] Run the new tests and confirm they fail for missing functionality.
- [ ] Implement chain schema parsing and validation.
- [ ] Run the tests and confirm they pass.

### Task 2: Thread strict mode through file and synthesis APIs

**Files:**
- Modify: `tests/test_synthesis.py`
- Modify: `src/lightningsearch_rl/synthesis.py`
- Modify: `src/lightningsearch_rl/cli.py`
- Modify: `tests/test_cli.py`

- [ ] Add failing tests for `require_chain_schema` file validation, validated synthesis, and CLI flags.
- [ ] Run the tests and confirm expected failures.
- [ ] Add function parameters and CLI flags.
- [ ] Run targeted tests and full pytest.

### Task 3: Verify and prepare remote smoke

**Files:**
- Modify: `docs/experiments/*` after remote smoke only.

- [ ] Run full local pytest.
- [ ] Merge the feature branch after verification.
- [ ] Push main.
- [ ] Narrow sync or git pull remote.
- [ ] Run remote pytest and mock strict synthesis smoke.
- [ ] Report launch payload for a 50-valid DeepSeek schema pilot before starting it.
