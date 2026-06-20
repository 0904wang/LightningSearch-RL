# Phase 5X Variance-Filtered GRPO Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small reward-dump variance filter so Phase 5X GRPO can train on examples whose rollout groups already show nonzero reward differences.

**Architecture:** Add a focused module that reads `reward_dump.jsonl`, computes per-source score ranges for selected stages, and filters an existing env-transition file to selected `source_id`s. Expose it through the existing CLI and use it from a remote launcher before a short GRPO smoke.

**Tech Stack:** Python stdlib JSON/argparse/pathlib, existing LightningSearch-RL CLI, pytest, verl dry-run/training launchers.

---

## Chunk 1: Local Filter Utility

### Task 1: Reward-Dump Variance Filter

**Files:**
- Create: `src/lightningsearch_rl/reward_variance_filter.py`
- Create: `tests/test_reward_variance_filter.py`
- Modify: `src/lightningsearch_rl/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**
  - Test that a reward dump with variable search/answer groups selects only those `source_id`s.
  - Test that output transitions preserve all rows for selected sources and write `summary.json`.
  - Test CLI command `filter-transitions-by-reward-variance`.

- [ ] **Step 2: Verify tests fail before implementation**
  - Run: `python -m pytest tests/test_reward_variance_filter.py tests/test_cli.py -k "variance or filter_transitions" -v`
  - Expected: import or CLI command failure.

- [ ] **Step 3: Implement minimal utility and CLI**
  - Parse `extra_info` whether dict or Python-literal string.
  - Group reward rows by `(stage, source_id)`.
  - Keep groups with `sample_count >= min_samples` and `score_range > min_score_range`.
  - Union selected source IDs across requested stages.
  - Filter transitions by `source_id`, falling back to `id`.
  - Write `transitions.jsonl`, `selected_source_ids.json`, `variance_groups.json`, and `summary.json`.

- [ ] **Step 4: Verify local tests pass**
  - Run: `python -m pytest tests/test_reward_variance_filter.py tests/test_cli.py -k "variance or filter_transitions" -v`
  - Run broader related tests: `python -m pytest tests/test_verl_reward_dump_diagnostics.py tests/test_verl_smoke.py tests/test_cli.py -v`

## Chunk 2: Phase 5X Remote Artifacts

### Task 2: Config and Launchers

**Files:**
- Create: `configs/experiments/phase5x_hard50_env_transition_grpo_4gpu_variance_rankreward.yaml`
- Create: `scripts/remote/phase5x_filter_transitions_from_phase5w_rankreward.sh`
- Create: `scripts/remote/phase5x_hard50_env_transition_grpo_4gpu_variance_rankreward.sh`

- [ ] **Step 1: Add filter launcher**
  - Input transitions: `/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl`
  - Input reward dump: `/data/wzl/LightningSearch-RL/results/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward/reward_dump.jsonl`
  - Output: `/data/wzl/LightningSearch-RL/results/phase5x-env-transitions-variance-rankreward-from-phase5w`

- [ ] **Step 2: Add GRPO config and launcher**
  - Use filtered transitions path.
  - Keep rollout_n=4 and rank reward.
  - Use 4 GPUs only after explicit approval.
  - Set train/val sample counts after the filter output is known.

- [ ] **Step 3: Verify dry run**
  - Narrow sync changed files.
  - Run remote tests.
  - Run filter command.
  - Run `train --dry-run --print-command`.
  - Report exact launch payload and wait for approval before real GRPO launch.
