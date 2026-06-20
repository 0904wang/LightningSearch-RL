# Turn-Level Agent SFT Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add turn-level tool-use SFT data and a local environment insertion loop so the model learns to emit one action at a time while runtime inserts observations.

**Architecture:** Keep the existing full-trace warmup exporter intact. Add a focused action parser/runtime environment module, then add a separate SFT-turns exporter that writes multi-turn `messages` where `<observation>` appears only as a user/runtime message, never as assistant output.

**Tech Stack:** Python 3.10, pytest, existing `LexicalRetriever`, existing `format_observation`, existing CLI patterns.

---

## Chunk 1: Action Parser And Environment Insertion

### Task 1: Add Single-Action Parser

**Files:**
- Create: `src/lightningsearch_rl/agent_loop.py`
- Create: `tests/test_agent_loop.py`

- [ ] **Step 1: Write failing tests**
  - Valid `<search>query</search>` parses as search.
  - Valid `<answer>text</answer>` parses as answer.
  - `<observation>` from model is invalid.
  - Multiple actions are invalid.
  - `<answer>` inside `<search>` is invalid.

- [ ] **Step 2: Run tests and verify missing module failure**

Run: `python -m pytest tests/test_agent_loop.py -q`

- [ ] **Step 3: Implement minimal parser**

Implement `AgentAction`, `parse_agent_action`, and invalid reasons.

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest tests/test_agent_loop.py -q`

### Task 2: Add Search Environment Observation Insertion

**Files:**
- Modify: `src/lightningsearch_rl/agent_loop.py`
- Modify: `tests/test_agent_loop.py`

- [ ] **Step 1: Write failing environment test**
  - `SearchEnvironment.search_observation(query)` returns formatted `<observation>` from retriever passages.

- [ ] **Step 2: Implement minimal environment wrapper**

Use existing `format_observation`.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_agent_loop.py -q`

## Chunk 2: Turn-Level SFT Export

### Task 3: Add Turn-Level Exporter

**Files:**
- Create: `src/lightningsearch_rl/sft_turns.py`
- Create: `tests/test_sft_turns.py`

- [ ] **Step 1: Write failing exporter test**
  - Export writes `sft_turns.jsonl`, `summary.json`, and `traces.jsonl`.
  - Assistant messages contain only `<search>` or `<answer>`.
  - Observation is a user message.
  - Summary reports zero assistant-generated observations.

- [ ] **Step 2: Implement exporter**

Use gold evidence selected from `gold_doc_ids`.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_sft_turns.py -q`

### Task 4: Add CLI

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI test**
  - `export-sft-turns --examples --index --out-dir` writes artifacts.

- [ ] **Step 2: Wire CLI**

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_cli.py::test_export_sft_turns_cli_writes_turn_level_conversations -q`

## Chunk 3: Verification

- [ ] Run related local tests:

```bash
python -m pytest tests/test_agent_loop.py tests/test_sft_turns.py tests/test_sft_warmup.py tests/test_sft.py tests/test_cli.py tests/test_grpo.py tests/test_verl_reward.py tests/test_verl_smoke.py tests/test_verl_sft_warmup.py -q
```

- [ ] Export a local fixture-scale `sft_turns` artifact.
- [ ] Prepare remote narrow sync and dry-run export only after local tests pass.
