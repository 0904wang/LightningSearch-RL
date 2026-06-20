# Phase 5G Environment Rollout Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal offline environment rollout inspection path that runs model-generated search actions through BM25 and feeds formatted observations back for answer generation.

**Architecture:** Keep this out of the `verl` trainer for now. Add a focused runner that loads SFT-turn rows, a lexical index, and a model adapter, then writes rollout records plus summary metrics. The CLI only orchestrates paths and arguments.

**Tech Stack:** Python 3.10, pytest, existing `LexicalRetriever`, `SearchEnvironment`, `parse_agent_action`, `transformers` for real generation.

---

## Chunk 1: Core Rollout Runner

### Task 1: Environment rollout data flow

**Files:**
- Create: `src/lightningsearch_rl/environment_rollout.py`
- Test: `tests/test_environment_rollout.py`

- [ ] **Step 1: Write failing tests**

Cover a deterministic fake generator:

```python
def fake_generate(messages, max_new_tokens):
    if len(messages) == 2:
        return "<search>Dr. Elena Voss founded organization award 2021</search>"
    return "<answer>Nobel Peace Prize</answer>"
```

Expected:

- the runner parses the search action
- BM25 returns a formatted `<observation>` as a user message
- the answer generation sees the inserted observation
- output record contains search/action/observation/final answer metadata
- summary reports valid search and answer rates

- [ ] **Step 2: Verify red**

Run:

```bash
python -m pytest tests/test_environment_rollout.py -q
```

Expected: fail because `lightningsearch_rl.environment_rollout` does not exist.

- [ ] **Step 3: Implement minimal runner**

Create:

- `run_environment_rollout(...)`
- `summarize_environment_rollouts(...)`
- small generator protocol/callable type
- JSONL/JSON writers

- [ ] **Step 4: Verify green**

Run:

```bash
python -m pytest tests/test_environment_rollout.py -q
```

Expected: pass.

## Chunk 2: CLI Surface

### Task 2: Add `inspect-env-rollout`

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI dry-run test**

Use a fake local corpus/index and `--dry-run` so the test does not load a real model.

- [ ] **Step 2: Verify red**

Run:

```bash
python -m pytest tests/test_cli.py::test_inspect_env_rollout_cli_dry_run_writes_prompts -q
```

Expected: fail because the CLI command is missing.

- [ ] **Step 3: Implement CLI**

Arguments:

- `--sft`
- `--index`
- `--model`
- `--out-dir`
- `--offset`
- `--limit`
- `--top-k`
- `--max-new-tokens`
- `--dry-run`

- [ ] **Step 4: Verify related tests**

Run:

```bash
python -m pytest tests/test_environment_rollout.py tests/test_cli.py::test_inspect_env_rollout_cli_dry_run_writes_prompts -q
```

Expected: pass.

## Chunk 3: Remote Smoke Preparation

### Task 3: Add remote launcher

**Files:**
- Create: `scripts/remote/phase5g_env_rollout_smoke.sh`

- [ ] **Step 1: Add script**

The script runs:

```bash
python -m lightningsearch_rl.cli inspect-env-rollout \
  --sft /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl \
  --index /data/wzl/LightningSearch-RL/indexes/phase4g-deepseek-titlefix-500-docidfix/index.json \
  --model /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40 \
  --out-dir /data/wzl/LightningSearch-RL/results/phase5g-env-rollout-smoke \
  --offset 0 \
  --limit 3 \
  --top-k 2 \
  --max-new-tokens 64
```

- [ ] **Step 2: Verify script syntax**

Run:

```bash
bash -n scripts/remote/phase5g_env_rollout_smoke.sh
```

Expected: pass.

## Verification

Run before remote sync:

```bash
python -m pytest tests/test_environment_rollout.py tests/test_cli.py::test_inspect_env_rollout_cli_dry_run_writes_prompts -q
python -m pytest -q
```
