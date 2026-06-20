# Phase 5R Scale Rollout Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare Phase 5R 500-rollout data generation and the matching filtered 50-step GRPO config.

**Architecture:** Reuse existing Phase 5Q remote script patterns. Add a 500-example rollout/export script that writes raw and quality-filtered transition artifacts, then add a 4-GPU GRPO launcher pointed at the filtered transition path.

**Tech Stack:** Bash, Python CLI, pytest, verl GRPO, tmux on the approved remote server.

---

### Task 1: Add Phase 5R Config Coverage

**Files:**
- Modify: `tests/test_verl_smoke.py`
- Create: `configs/experiments/phase5r_filtered_env_transition_grpo_4gpu_990x50_softanswer.yaml`

- [x] **Step 1: Write failing test**
  - Test reads the Phase 5R config and verifies 792/198 split, 50 steps, and reward dump path.

- [x] **Step 2: Verify red**
  - Run: `python -m pytest tests/test_verl_smoke.py::test_phase5r_filtered_soft_answer_grpo_config_uses_scaled_split -v`
  - Expected: FAIL because the config does not exist.

- [ ] **Step 3: Add config**
  - Add the Phase 5R 990-transition filtered GRPO config.

- [ ] **Step 4: Verify green**
  - Run the single test, related config tests, then full local pytest.

### Task 2: Add Remote Launchers

**Files:**
- Create: `scripts/remote/phase5r_env_rollout_gold_distractors_500_filtered.sh`
- Create: `scripts/remote/phase5r_filtered_env_transition_grpo_4gpu_990x50_softanswer.sh`

- [ ] **Step 1: Add rollout/export launcher**
  - Produce raw rollout summary, answer diagnostics, raw transition summary, and filtered transition summary.

- [ ] **Step 2: Add GRPO launcher**
  - Follow the Phase 5Q filtered launcher pattern with separate result, checkpoint, log, reward dump, batch diagnostics, and parser outputs.

- [ ] **Step 3: Remote sync and dry-run**
  - Narrow-sync only the new/changed files.
  - Run remote pytest for the Phase 5R config test.
  - Run `train --dry-run --print-command` only after filtered transitions exist.
