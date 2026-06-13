# Phase 2B Real-Format Retrieval Baseline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dataset conversion more robust for local HotpotQA / 2Wiki-like files and add a retrieval baseline report command.

**Architecture:** Extend `adapters.py` with normalization helpers and `limit` support, add `baseline.py` for report generation, and extend `cli.py` with `--limit` and `retrieval-baseline`. Keep existing Phase 1 / Phase 2A commands stable.

**Tech Stack:** Python 3.10+, pytest, JSON / JSONL, standard library only.

---

## File Structure

Create:

- `src/lightningsearch_rl/baseline.py`: retrieval baseline report generation.
- `tests/fixtures/hotpot_mixed_raw.jsonl`: local mixed-shape Hotpot-like fixture.
- `tests/fixtures/2wiki_mixed_raw.jsonl`: local mixed-shape 2Wiki-like fixture.
- `tests/test_baseline.py`: baseline report tests.

Modify:

- `src/lightningsearch_rl/adapters.py`: support JSONL, field variants, supporting fact dicts, list-of-dict context, and `limit`.
- `src/lightningsearch_rl/cli.py`: add `--limit` and `retrieval-baseline`.
- `tests/test_adapters.py`: add mixed format and limit tests.
- `tests/test_cli.py`: add Phase 2B CLI test.
- `README.md`: document Phase 2B workflow.

## Chunk 1: Adapter Normalization

### Task 1: Add Mixed Hotpot Fixture And Failing Test

**Files:**
- Create: `tests/fixtures/hotpot_mixed_raw.jsonl`
- Modify: `tests/test_adapters.py`
- Modify: `src/lightningsearch_rl/adapters.py`

- [ ] **Step 1: Create mixed Hotpot JSONL fixture**

```jsonl
{"_id":"hp_mixed_1","question":"Which city is the birthplace of the author of Example Book?","answer":"Example City","context":[["Example Book",["Example Book was written by Alice Smith."]],["Alice Smith",["Alice Smith was born in Example City."]]],"supporting_facts":[{"title":"Alice Smith","sent_id":0}]}
{"id":"hp_mixed_2","question":"Who wrote Example Book?","answers":["Alice Smith"],"context":[{"title":"Example Book","sentences":["Example Book was written by Alice Smith."]},{"title":"Noise","sentences":["Noise text."]}],"supporting_facts":[["Example Book",0]]}
```

- [ ] **Step 2: Add failing adapter test**

Add:

```python
def test_convert_hotpot_file_supports_jsonl_mixed_shapes_and_limit(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"

    convert_hotpot_file(
        Path("tests/fixtures/hotpot_mixed_raw.jsonl"),
        corpus_path,
        examples_path,
        limit=1,
    )

    passages = load_corpus_jsonl(corpus_path)
    examples = load_jsonl_examples(examples_path)

    assert len(examples) == 1
    assert examples[0].id == "hp_mixed_1"
    assert examples[0].answers == ["Example City"]
    assert examples[0].gold_doc_ids == ["hotpot::Alice Smith::0"]
    assert [p.doc_id for p in passages] == ["hotpot::Example Book::0", "hotpot::Alice Smith::0"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: FAIL because `convert_hotpot_file` lacks `limit` and JSONL support.

- [ ] **Step 4: Implement normalization**

In `adapters.py`, add:

- `_load_raw_rows(path)` for `.json` array and `.jsonl`.
- `_row_id(row)`.
- `_answers(row)`.
- `_supporting_fact_pairs(row)`.
- `_context_items(row)` returning `(title, sentences)`.
- optional `limit` param on `convert_hotpot_file`.

- [ ] **Step 5: Run adapter tests**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/adapters.py tests/test_adapters.py tests/fixtures/hotpot_mixed_raw.jsonl
git commit -m "feat: normalize mixed hotpot input formats"
```

### Task 2: Add Mixed 2Wiki Fixture And Failing Test

**Files:**
- Create: `tests/fixtures/2wiki_mixed_raw.jsonl`
- Modify: `tests/test_adapters.py`
- Modify: `src/lightningsearch_rl/adapters.py`

- [ ] **Step 1: Create mixed 2Wiki JSONL fixture**

```jsonl
{"_id":"tw_mixed_1","question":"Which city is the birthplace of the author of Example Book?","answer":"Example City","context":{"Example Book":["Example Book was written by Alice Smith."],"Alice Smith":["Alice Smith was born in Example City."]},"supporting_facts":[{"title":"Alice Smith","sent_id":0}]}
{"id":"tw_mixed_2","question":"Who wrote Example Book?","final_answer":"Alice Smith","context":[{"title":"Example Book","sentences":["Example Book was written by Alice Smith."]}],"supporting_facts":[["Example Book",0]]}
```

- [ ] **Step 2: Add failing 2Wiki test**

Add:

```python
def test_convert_2wiki_file_supports_jsonl_mixed_shapes_and_limit(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"

    convert_2wiki_file(
        Path("tests/fixtures/2wiki_mixed_raw.jsonl"),
        corpus_path,
        examples_path,
        limit=1,
    )

    examples = load_jsonl_examples(examples_path)

    assert len(examples) == 1
    assert examples[0].id == "tw_mixed_1"
    assert examples[0].gold_doc_ids == ["2wiki::Alice Smith::0"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: FAIL because `convert_2wiki_file` lacks `limit` / JSONL support.

- [ ] **Step 4: Extend 2Wiki conversion**

Reuse `_load_raw_rows`, `_row_id`, `_answers`, `_supporting_fact_pairs`, and `_context_items`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/adapters.py tests/test_adapters.py tests/fixtures/2wiki_mixed_raw.jsonl
git commit -m "feat: normalize mixed 2wiki input formats"
```

## Chunk 2: Retrieval Baseline Report

### Task 3: Add Baseline Report Module

**Files:**
- Create: `tests/test_baseline.py`
- Create: `src/lightningsearch_rl/baseline.py`

- [ ] **Step 1: Write failing baseline test**

```python
import json
from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.index_store import save_lexical_index
from lightningsearch_rl.baseline import run_retrieval_baseline


def test_run_retrieval_baseline_writes_report(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    report = tmp_path / "report.json"
    convert_hotpot_file(Path("tests/fixtures/hotpot_mixed_raw.jsonl"), corpus, examples, limit=1)
    save_lexical_index(index, load_corpus_jsonl(corpus))

    run_retrieval_baseline("hotpot", examples, index, report, top_k=2)

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["dataset"] == "hotpot"
    assert payload["top_k"] == 2
    assert payload["metrics"]["recall_at_2"] == 1.0
    assert payload["artifacts"]["examples"].endswith("examples.jsonl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baseline.py -v`
Expected: FAIL because `baseline` does not exist.

- [ ] **Step 3: Implement baseline module**

Create `run_retrieval_baseline(dataset, examples_path, index_path, report_path, top_k)`.

It should:

- Load examples.
- Load lexical index.
- Call `evaluate_retrieval`.
- Write report JSON.
- Return the report dict.

- [ ] **Step 4: Run baseline test**

Run: `python -m pytest tests/test_baseline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/baseline.py tests/test_baseline.py
git commit -m "feat: write retrieval baseline reports"
```

### Task 4: Add CLI Support For Limit And Baseline Report

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI test**

Add test for:

```python
main(["prepare-hotpot", "--raw", "tests/fixtures/hotpot_mixed_raw.jsonl", "--corpus", str(corpus), "--examples", str(examples), "--limit", "1"])
main(["build-index", "--corpus", str(corpus), "--index", str(index)])
main(["retrieval-baseline", "--dataset", "hotpot", "--examples", str(examples), "--index", str(index), "--report", str(report), "--top-k", "2"])
```

Assert report exists and `example_count == 1`.

- [ ] **Step 2: Run CLI test**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL because CLI lacks `--limit` and `retrieval-baseline`.

- [ ] **Step 3: Implement CLI changes**

Add `--limit` to `prepare-hotpot` and `prepare-2wiki`.

Add `retrieval-baseline` subcommand:

- `--dataset`
- `--examples`
- `--index`
- `--report`
- `--top-k`

- [ ] **Step 4: Run CLI tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Run full tests**

Run: `python -m pytest`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/cli.py tests/test_cli.py
git commit -m "feat: add retrieval baseline cli"
```

## Chunk 3: Documentation And Verification

### Task 5: Update README And Verify Commands

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add Phase 2B command sequence:

```bash
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_mixed_raw.jsonl --corpus results/phase2b/corpus.jsonl --examples results/phase2b/examples.jsonl --limit 1
python -m lightningsearch_rl.cli build-index --corpus results/phase2b/corpus.jsonl --index results/phase2b/index.json
python -m lightningsearch_rl.cli retrieval-baseline --dataset hotpot --examples results/phase2b/examples.jsonl --index results/phase2b/index.json --report results/phase2b/baseline_report.json --top-k 2
```

- [ ] **Step 2: Run final verification**

Run:

```bash
python -m pytest
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_mixed_raw.jsonl --corpus results/phase2b/corpus.jsonl --examples results/phase2b/examples.jsonl --limit 1
python -m lightningsearch_rl.cli build-index --corpus results/phase2b/corpus.jsonl --index results/phase2b/index.json
python -m lightningsearch_rl.cli retrieval-baseline --dataset hotpot --examples results/phase2b/examples.jsonl --index results/phase2b/index.json --report results/phase2b/baseline_report.json --top-k 2
```

Expected: tests PASS and report contains `recall_at_2: 1.0`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document phase 2b baseline workflow"
```

## Guardrails

- Use @superpowers:test-driven-development for every production behavior.
- Do not download datasets.
- Do not add dependencies.
- Do not run remote commands.
- Preserve all Phase 1 and Phase 2A commands.
