# Phase 5Y Expanded Variance Probe Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the Phase 5X variance-filtered pool by probing all Phase 5W hard50 transitions with multiple stochastic completions and filtering sources whose rewards vary.

**Architecture:** Add a focused reward-probe module that converts existing env transitions into prompt/reward requests, samples multiple completions per transition, scores them through the existing `verl_reward.compute_score`, and writes a standard `reward_dump.jsonl`. Reuse the existing variance filter to produce an expanded transition pool for a later GRPO run.

**Tech Stack:** Python stdlib JSON/pathlib/os, existing LightningSearch-RL reward function, optional vLLM backend on the remote server, pytest, tmux launchers.

---

## Chunk 1: Local Reward Probe

### Task 1: Probe Requests and Reward Dump

**Files:**
- Create: `src/lightningsearch_rl/reward_probe.py`
- Create: `tests/test_reward_probe.py`
- Modify: `src/lightningsearch_rl/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**
  - Test dry-run request export from transition rows.
  - Test stubbed sampling writes `generations.jsonl`, `reward_dump.jsonl`, and `summary.json`.
  - Test CLI command `probe-reward-variance --dry-run`.

- [ ] **Step 2: Verify tests fail before implementation**
  - Run: `python -m pytest tests/test_reward_probe.py tests/test_cli.py -k "probe_reward or reward_probe" -v`
  - Expected: import or CLI command failure.

- [ ] **Step 3: Implement minimal probe**
  - Convert transition rows into prompt, ground truth, and `extra_info` matching the existing transition GRPO path.
  - Support `--dry-run` without loading a model.
  - Support an injected generator for tests.
  - Use vLLM for real remote sampling when no generator is injected.
  - Score each generated completion with `lightningsearch_rl.verl_reward.compute_score`.
  - Write `probe_requests.jsonl`, `generations.jsonl`, `reward_dump.jsonl`, and `summary.json`.

- [ ] **Step 4: Verify local tests pass**
  - Run targeted tests, then broader reward/filter/CLI tests.

## Chunk 2: Remote Phase 5Y Launcher

### Task 2: Probe and Filter Script

**Files:**
- Create: `scripts/remote/phase5y_reward_probe_variance_pool_from_phase5w.sh`

- [ ] **Step 1: Add launcher**
  - Input transitions: `/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl`
  - Model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
  - Probe output: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6`
  - Filter output: `/data/wzl/LightningSearch-RL/results/phase5y-env-transitions-variance-rankreward-100src`

- [ ] **Step 2: Remote smoke and dry-run**
  - Narrow sync changed files.
  - Run remote targeted tests.
  - Run `probe-reward-variance --dry-run --limit 4 --samples-per-prompt 2`.
  - Run the launcher only after reporting exact GPU/session/log/result paths.

- [ ] **Step 3: Record completed experiment**
  - Write local experiment record under `docs/experiments`.
  - Write remote `EXPERIMENT_RECORD.md` under the probe or filter result directory.
