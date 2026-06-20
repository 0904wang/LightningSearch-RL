# SFT Warmup Export Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a gold-evidence SFT warmup exporter that produces non-empty tagged assistant traces for Qwen3-4B format warmup.

**Architecture:** Add a focused exporter in `src/lightningsearch_rl/sft_warmup.py` that consumes existing `examples.jsonl` plus `index.json`, selects gold evidence passages by `gold_doc_ids`, and writes conversation rows with strict system/user/assistant messages. Extend CLI with `export-sft-warmup` and test both API and CLI behavior.

**Tech Stack:** Python, pytest, existing `QAExample`, `LexicalIndex`, JSONL artifacts.

---

### Task 1: Exporter Tests

**Files:**
- Create: `tests/test_sft_warmup.py`

- [ ] Write a failing test that exports one example whose retriever would miss but whose gold evidence exists in the index.
- [ ] Assert `sft_warmup.jsonl`, `summary.json`, and `traces.jsonl` are written.
- [ ] Assert assistant content contains `<think>`, `<observation>`, and non-empty `<answer>Example City</answer>`.
- [ ] Assert summary reports `answer_tag_rate=1.0`, `non_empty_answer_rate=1.0`, and `gold_evidence_coverage=1.0`.

### Task 2: Exporter Implementation

**Files:**
- Create: `src/lightningsearch_rl/sft_warmup.py`

- [ ] Load examples with `load_jsonl_examples`.
- [ ] Load the lexical index as the source of corpus passages.
- [ ] Select passages whose doc IDs match `example.gold_doc_ids`.
- [ ] Build strict messages:
  - system prompt requires exact tags and no extra text outside tags.
  - user prompt is the question.
  - assistant response contains gold evidence observation and the first non-empty gold answer.
- [ ] Write `sft_warmup.jsonl`, `traces.jsonl`, and `summary.json`.

### Task 3: CLI

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Modify: `tests/test_cli.py`

- [ ] Add `export-sft-warmup --examples --index --out-dir`.
- [ ] Add CLI test that confirms artifacts are written.

### Task 4: Verification And Remote Export

- [ ] Run local tests for SFT warmup, SFT, CLI, GRPO, reward, and verl smoke.
- [ ] Narrow-sync changed files to `/data/wzl/LightningSearch-RL/repo`.
- [ ] Run the same tests remotely.
- [ ] Export `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-warmup-gold`.
- [ ] Inspect summary and sample rows.
- [ ] Record the experiment under `docs/experiments`.
