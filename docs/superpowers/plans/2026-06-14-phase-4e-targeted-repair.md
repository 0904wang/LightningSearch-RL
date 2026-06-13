# Phase 4E Targeted Repair Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional deterministic repair for common strict `chain_schema` failures.

**Architecture:** Keep strict validation unchanged. Add a small repair helper in `synthesis.py` and call it only from validated synthesis when `repair_chain_schema=True`.

**Tech Stack:** Python 3.10+, pytest, existing LightningSearch-RL CLI.

---

## Chunk 1: Deterministic Repair

### Task 1: Unit tests

**Files:**
- Modify: `tests/test_synthesis.py`
- Modify: `src/lightningsearch_rl/synthesis.py`

- [ ] Add failing tests for repairable and unrecoverable rows.
- [ ] Run targeted tests and confirm they fail.
- [ ] Implement the minimal repair helper.
- [ ] Run targeted tests and confirm they pass.

### Task 2: Validated synthesis integration

**Files:**
- Modify: `tests/test_synthesis.py`
- Modify: `src/lightningsearch_rl/synthesis.py`
- Modify: `tests/test_cli.py`
- Modify: `src/lightningsearch_rl/cli.py`

- [ ] Add failing tests for summary repair counters and CLI flag.
- [ ] Run targeted tests and confirm they fail.
- [ ] Thread `repair_chain_schema` through the API and CLI.
- [ ] Run targeted tests and full pytest.

### Task 3: Remote smoke and launch report

**Files:**
- Add: `docs/experiments/phase4e-mock-smoke-2026-06-14.md`

- [ ] Merge and push if local tests pass.
- [ ] Sync remote using approved git pull or narrow archive sync.
- [ ] Run remote pytest and mock repair smoke.
- [ ] Record the mock smoke.
- [ ] Report a real DeepSeek pilot launch plan before starting.
