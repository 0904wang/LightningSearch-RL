# Phase 3A SFT Export Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export supervised fine-tuning conversations and audit traces from deterministic search-agent trajectories over prepared examples and a lexical index.

**Architecture:** Add `run_retrieval_episode` to reuse runtime trace logic with an injected retriever, add `formatting.py` for structured assistant content, add `sft.py` for artifact export, and extend `cli.py` with `export-sft`.

**Tech Stack:** Python 3.10+, pytest, JSON / JSONL, standard library only.

---

## File Structure

Create:

- `src/lightningsearch_rl/formatting.py`: format observations and traces into assistant content.
- `src/lightningsearch_rl/sft.py`: export SFT JSONL, traces JSONL, and summary JSON.
- `tests/test_formatting.py`: formatting tests.
- `tests/test_sft.py`: SFT export tests.

Modify:

- `src/lightningsearch_rl/runtime.py`: add `run_retrieval_episode`.
- `src/lightningsearch_rl/cli.py`: add `export-sft`.
- `tests/test_runtime.py`: add injected retriever episode test.
- `tests/test_cli.py`: add export CLI test.
- `README.md`: document Phase 3A workflow.

## Chunk 1: Runtime Support For Shared Retriever

### Task 1: Add `run_retrieval_episode`

**Files:**
- Modify: `src/lightningsearch_rl/runtime.py`
- Modify: `tests/test_runtime.py`

- [ ] **Step 1: Write failing runtime test**

Add:

```python
from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.retrieval import LexicalRetriever


def test_retrieval_episode_uses_injected_shared_retriever(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    convert_hotpot_file(Path("tests/fixtures/hotpot_mixed_raw.jsonl"), corpus, examples, limit=1)
    example = load_jsonl_examples(examples)[0]
    retriever = LexicalRetriever(load_corpus_jsonl(corpus))

    trace = run_retrieval_episode(example, retriever, top_k=2)

    assert trace.final_answer == "Example City"
    assert trace.steps[0].observation[0].doc_id == "hotpot::Alice Smith::0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: FAIL because `run_retrieval_episode` does not exist.

- [ ] **Step 3: Implement runtime helper**

Add `run_retrieval_episode(example, retriever, top_k=5)` by extracting shared logic from `run_rule_based_episode`.

- [ ] **Step 4: Run runtime tests**

Run: `python -m pytest tests/test_runtime.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/runtime.py tests/test_runtime.py
git commit -m "feat: run episodes with shared retrievers"
```

## Chunk 2: Formatting And SFT Export

### Task 2: Add Trace Formatting

**Files:**
- Create: `src/lightningsearch_rl/formatting.py`
- Create: `tests/test_formatting.py`

- [ ] **Step 1: Write failing formatting test**

```python
from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.formatting import format_assistant_trace
from lightningsearch_rl.runtime import run_rule_based_episode


def test_format_assistant_trace_contains_search_observation_and_answer():
    example = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))[0]
    trace = run_rule_based_episode(example, top_k=2)

    content = format_assistant_trace(trace)

    assert "<search>" in content
    assert "<observation>" in content
    assert "<answer>Example City</answer>" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_formatting.py -v`
Expected: FAIL because `formatting` does not exist.

- [ ] **Step 3: Implement formatting**

Format:

```text
<think>I should search for evidence.</think>
<search>...</search>
<observation>
[1] title: text
</observation>
<think>The retrieved evidence supports the answer.</think>
<answer>...</answer>
```

- [ ] **Step 4: Run formatting test**

Run: `python -m pytest tests/test_formatting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/formatting.py tests/test_formatting.py
git commit -m "feat: format search traces for sft"
```

### Task 3: Add SFT Export Module

**Files:**
- Create: `src/lightningsearch_rl/sft.py`
- Create: `tests/test_sft.py`

- [ ] **Step 1: Write failing SFT export test**

```python
import json
from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.index_store import save_lexical_index
from lightningsearch_rl.sft import export_sft


def test_export_sft_writes_conversations_traces_and_summary(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "sft"
    convert_hotpot_file(Path("tests/fixtures/hotpot_mixed_raw.jsonl"), corpus, examples, limit=1)
    save_lexical_index(index, load_corpus_jsonl(corpus))

    summary = export_sft(examples, index, out_dir, top_k=2)

    row = json.loads((out_dir / "sft.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert summary["sft_rows"] == 1
    assert row["messages"][0]["role"] == "system"
    assert "<answer>Example City</answer>" in row["messages"][-1]["content"]
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "summary.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sft.py -v`
Expected: FAIL because `sft` does not exist.

- [ ] **Step 3: Implement SFT export**

Implement `export_sft(examples_path, index_path, out_dir, top_k)`.

Use:

- `load_jsonl_examples`
- `load_lexical_index`
- `run_retrieval_episode`
- `format_assistant_trace`

Write:

- `sft.jsonl`
- `traces.jsonl`
- `summary.json`

- [ ] **Step 4: Run SFT test**

Run: `python -m pytest tests/test_sft.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/sft.py tests/test_sft.py
git commit -m "feat: export sft conversations from traces"
```

## Chunk 3: CLI And Documentation

### Task 4: Add `export-sft` CLI

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Add test that prepares Hotpot mixed fixture, builds index, then calls:

```python
main(["export-sft", "--examples", str(examples), "--index", str(index), "--out-dir", str(out_dir), "--top-k", "2"])
```

Assert `sft.jsonl`, `traces.jsonl`, and `summary.json` exist.

- [ ] **Step 2: Run CLI test**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL because `export-sft` does not exist.

- [ ] **Step 3: Implement CLI command**

Add parser:

- `export-sft`
- `--examples`
- `--index`
- `--out-dir`
- `--top-k`

Call `export_sft`.

- [ ] **Step 4: Run CLI tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Run full tests**

Run: `python -m pytest`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/cli.py tests/test_cli.py
git commit -m "feat: add sft export cli"
```

### Task 5: Update README And Verify

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add:

```bash
python -m lightningsearch_rl.cli export-sft --examples results/phase2b/examples.jsonl --index results/phase2b/index.json --out-dir results/phase3a/sft --top-k 2
```

- [ ] **Step 2: Run final verification**

Run:

```bash
python -m pytest
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_mixed_raw.jsonl --corpus results/phase3a/corpus.jsonl --examples results/phase3a/examples.jsonl --limit 1
python -m lightningsearch_rl.cli build-index --corpus results/phase3a/corpus.jsonl --index results/phase3a/index.json
python -m lightningsearch_rl.cli export-sft --examples results/phase3a/examples.jsonl --index results/phase3a/index.json --out-dir results/phase3a/sft --top-k 2
```

Expected: tests PASS and SFT artifacts exist.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document phase 3a sft export workflow"
```

## Guardrails

- Use @superpowers:test-driven-development for every production behavior.
- Do not add model inference.
- Do not add external dependencies.
- Do not launch remote commands.
- Preserve existing smoke/retrieval baseline commands.
