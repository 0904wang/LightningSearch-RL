# Env Transition Quality Filter Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic quality manifest tagging and optional exclusion to environment-transition exports.

**Architecture:** Keep quality filtering local to the environment-transition export boundary. A small manifest loader returns ID-scoped quality flags, and `export_env_rollout_transitions` applies it before writing transition, reward, rollout, and summary artifacts.

**Tech Stack:** Python 3.10, pytest, existing `lightningsearch_rl` JSONL export utilities.

---

### Task 1: Export Quality Manifest Support

**Files:**
- Modify: `src/lightningsearch_rl/env_transitions.py`
- Test: `tests/test_env_transitions.py`

- [ ] **Step 1: Write failing tests**
  - Add a test where a manifest tags one rollout ID and verifies metadata is present.
  - Add a test where `exclude_quality_flags={"qa_type_mismatch"}` skips that rollout and reports it in summary.

- [ ] **Step 2: Verify tests fail**
  - Run: `python -m pytest tests/test_env_transitions.py -k quality -v`
  - Expected: fail because manifest arguments and metadata are not implemented.

- [ ] **Step 3: Implement minimal support**
  - Add manifest loading for JSON maps and JSONL rows.
  - Add optional `quality_manifest_path` and `exclude_quality_flags` parameters.
  - Add quality metadata to kept transitions, reward records, and rollouts.
  - Add excluded row diagnostics to `summary.json`.

- [ ] **Step 4: Verify tests pass**
  - Run: `python -m pytest tests/test_env_transitions.py -k quality -v`

### Task 2: CLI Surface and Phase 5Q Manifest

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Modify: `tests/test_cli.py`
- Create: `configs/data_quality/phase5q_known_mismatches.json`

- [ ] **Step 1: Write failing CLI test**
  - Verify `export-env-transitions --quality-manifest ... --exclude-quality-flag qa_type_mismatch` writes a summary with excluded IDs.

- [ ] **Step 2: Verify CLI test fails**
  - Run: `python -m pytest tests/test_cli.py -k quality -v`

- [ ] **Step 3: Implement CLI flags and manifest file**
  - Add `--quality-manifest`.
  - Add repeatable `--exclude-quality-flag`.
  - Add the Phase 5Q known mismatch manifest.

- [ ] **Step 4: Verify targeted and full tests**
  - Run: `python -m pytest tests/test_env_transitions.py tests/test_cli.py -k "quality or export_env_transitions" -v`
  - Run: `python -m pytest`
