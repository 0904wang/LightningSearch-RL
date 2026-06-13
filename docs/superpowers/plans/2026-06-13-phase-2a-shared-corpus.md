# Phase 2A Shared Corpus Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend LightningSearch-RL with shared corpus files, tiny HotpotQA / 2Wiki adapters, persisted lexical indexes, and retrieval-only evaluation metrics.

**Architecture:** Keep Phase 1 inline-corpus examples working while adding shared corpus references through focused modules: `corpus.py` for passage IO, `adapters.py` for tiny dataset conversion, `index_store.py` for persisted lexical indexes, and `retrieval_eval.py` for recall metrics. Extend `cli.py` with prepare/build/eval commands that compose these modules.

**Tech Stack:** Python 3.10+, pytest, JSON / JSONL, standard library only.

---

## File Structure

Create:

- `src/lightningsearch_rl/corpus.py`: read/write corpus JSONL, deduplicate by `doc_id`, serialize passages.
- `src/lightningsearch_rl/adapters.py`: convert tiny HotpotQA-like and 2Wiki-like raw files to shared corpus/examples files.
- `src/lightningsearch_rl/index_store.py`: save/load lexical index artifacts.
- `src/lightningsearch_rl/retrieval_eval.py`: compute retrieval-only metrics.
- `tests/fixtures/hotpot_tiny_raw.json`: tiny HotpotQA-like fixture.
- `tests/fixtures/2wiki_tiny_raw.json`: tiny 2Wiki-like fixture.
- `tests/test_corpus.py`
- `tests/test_adapters.py`
- `tests/test_index_store.py`
- `tests/test_retrieval_eval.py`

Modify:

- `src/lightningsearch_rl/data.py`: support prepared examples with `corpus_doc_ids`.
- `src/lightningsearch_rl/cli.py`: add `prepare-hotpot`, `prepare-2wiki`, `build-index`, and `eval-retrieval`.
- `README.md`: document Phase 2A commands.

## Chunk 1: Shared Corpus And Prepared Examples

### Task 1: Add Corpus IO

**Files:**
- Create: `tests/test_corpus.py`
- Create: `src/lightningsearch_rl/corpus.py`

- [ ] **Step 1: Write failing corpus IO test**

```python
from pathlib import Path

from lightningsearch_rl.corpus import load_corpus_jsonl, write_corpus_jsonl
from lightningsearch_rl.data import Passage


def test_write_and_load_corpus_deduplicates_doc_ids(tmp_path):
    path = tmp_path / "corpus.jsonl"
    write_corpus_jsonl(
        path,
        [
            Passage("doc1", "Title", "First text."),
            Passage("doc1", "Title Duplicate", "Second text."),
            Passage("doc2", "Other", "Other text."),
        ],
    )

    passages = load_corpus_jsonl(path)

    assert [p.doc_id for p in passages] == ["doc1", "doc2"]
    assert passages[0].title == "Title"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: FAIL because `lightningsearch_rl.corpus` does not exist.

- [ ] **Step 3: Implement corpus IO**

Create `src/lightningsearch_rl/corpus.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from lightningsearch_rl.data import Passage


def passage_to_dict(passage: Passage) -> dict[str, str]:
    return {"doc_id": passage.doc_id, "title": passage.title, "text": passage.text}


def passage_from_dict(row: dict) -> Passage:
    return Passage(doc_id=row["doc_id"], title=row.get("title", ""), text=row["text"])


def deduplicate_passages(passages: Iterable[Passage]) -> list[Passage]:
    seen: set[str] = set()
    deduped: list[Passage] = []
    for passage in passages:
        if passage.doc_id in seen:
            continue
        seen.add(passage.doc_id)
        deduped.append(passage)
    return deduped


def write_corpus_jsonl(path: Path, passages: Iterable[Passage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for passage in deduplicate_passages(passages):
            handle.write(json.dumps(passage_to_dict(passage), ensure_ascii=False) + "\n")


def load_corpus_jsonl(path: Path) -> list[Passage]:
    passages: list[Passage] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            passages.append(passage_from_dict(json.loads(line)))
    return deduplicate_passages(passages)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/corpus.py tests/test_corpus.py
git commit -m "feat: add shared corpus jsonl io"
```

### Task 2: Extend Data Loader For Shared Corpus References

**Files:**
- Modify: `src/lightningsearch_rl/data.py`
- Create/modify: `tests/test_data.py`

- [ ] **Step 1: Write failing data loader test**

Add to `tests/test_data.py`:

```python
def test_load_jsonl_examples_supports_shared_corpus_doc_ids(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        '{"id":"ex2","question":"Q?","answers":["A"],"gold_doc_ids":["doc1"],"corpus_doc_ids":["doc1","doc2"]}\n',
        encoding="utf-8",
    )

    example = load_jsonl_examples(path)[0]

    assert example.corpus == []
    assert example.corpus_doc_ids == ["doc1", "doc2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL because `QAExample` lacks `corpus_doc_ids`.

- [ ] **Step 3: Implement `corpus_doc_ids`**

Modify `QAExample` to include:

```python
corpus_doc_ids: list[str]
```

In inline examples, default it to passage doc IDs when absent:

```python
corpus = [...]
corpus_doc_ids = list(row.get("corpus_doc_ids", [p.doc_id for p in corpus]))
```

- [ ] **Step 4: Run data tests**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS.

- [ ] **Step 5: Run full tests**

Run: `python -m pytest`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/data.py tests/test_data.py
git commit -m "feat: support shared corpus example references"
```

## Chunk 2: Tiny Dataset Adapters

### Task 3: Add Tiny HotpotQA Adapter

**Files:**
- Create: `tests/fixtures/hotpot_tiny_raw.json`
- Create: `tests/test_adapters.py`
- Create: `src/lightningsearch_rl/adapters.py`

- [ ] **Step 1: Write tiny Hotpot fixture**

Create `tests/fixtures/hotpot_tiny_raw.json`:

```json
[
  {
    "_id": "hp1",
    "question": "Which city is the birthplace of the author of Example Book?",
    "answer": "Example City",
    "context": [
      ["Example Book", ["Example Book was written by Alice Smith."]],
      ["Alice Smith", ["Alice Smith was born in Example City."]],
      ["Noise", ["This passage is unrelated."]]
    ],
    "supporting_facts": [["Alice Smith", 0]]
  }
]
```

- [ ] **Step 2: Write failing adapter test**

```python
from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.data import load_jsonl_examples


def test_convert_hotpot_file_writes_shared_corpus_and_examples(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"

    convert_hotpot_file(
        Path("tests/fixtures/hotpot_tiny_raw.json"),
        corpus_path,
        examples_path,
    )

    passages = load_corpus_jsonl(corpus_path)
    examples = load_jsonl_examples(examples_path)

    assert [p.doc_id for p in passages] == [
        "hotpot::Example Book::0",
        "hotpot::Alice Smith::0",
        "hotpot::Noise::0",
    ]
    assert examples[0].gold_doc_ids == ["hotpot::Alice Smith::0"]
    assert examples[0].corpus_doc_ids == [p.doc_id for p in passages]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: FAIL because `adapters` does not exist.

- [ ] **Step 4: Implement Hotpot conversion**

Implement `convert_hotpot_file(raw_path, corpus_path, examples_path)` in `src/lightningsearch_rl/adapters.py`.

Rules:

- Load raw JSON list.
- Emit one passage per title/sentence.
- Doc ID format: `hotpot::{title}::{sentence_index}`.
- Supporting facts become `gold_doc_ids`.
- Write shared corpus and prepared examples JSONL.

- [ ] **Step 5: Run adapter test**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/adapters.py tests/test_adapters.py tests/fixtures/hotpot_tiny_raw.json
git commit -m "feat: convert tiny hotpot data to shared corpus"
```

### Task 4: Add Tiny 2Wiki Adapter

**Files:**
- Create: `tests/fixtures/2wiki_tiny_raw.json`
- Modify: `tests/test_adapters.py`
- Modify: `src/lightningsearch_rl/adapters.py`

- [ ] **Step 1: Write tiny 2Wiki fixture**

Create `tests/fixtures/2wiki_tiny_raw.json`:

```json
[
  {
    "_id": "tw1",
    "question": "Which city is the birthplace of the author of Example Book?",
    "answer": "Example City",
    "context": {
      "Example Book": ["Example Book was written by Alice Smith."],
      "Alice Smith": ["Alice Smith was born in Example City."],
      "Noise": ["This passage is unrelated."]
    },
    "supporting_facts": [["Alice Smith", 0]]
  }
]
```

- [ ] **Step 2: Write failing adapter test**

Add:

```python
from lightningsearch_rl.adapters import convert_2wiki_file


def test_convert_2wiki_file_writes_shared_corpus_and_examples(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"

    convert_2wiki_file(
        Path("tests/fixtures/2wiki_tiny_raw.json"),
        corpus_path,
        examples_path,
    )

    passages = load_corpus_jsonl(corpus_path)
    examples = load_jsonl_examples(examples_path)

    assert passages[1].doc_id == "2wiki::Alice Smith::0"
    assert examples[0].gold_doc_ids == ["2wiki::Alice Smith::0"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: FAIL because `convert_2wiki_file` does not exist.

- [ ] **Step 4: Implement 2Wiki conversion**

Implement `convert_2wiki_file(raw_path, corpus_path, examples_path)` reusing shared helper functions where reasonable.

- [ ] **Step 5: Run adapter tests**

Run: `python -m pytest tests/test_adapters.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/adapters.py tests/test_adapters.py tests/fixtures/2wiki_tiny_raw.json
git commit -m "feat: convert tiny 2wiki data to shared corpus"
```

## Chunk 3: Index Store And Retrieval Evaluation

### Task 5: Add Lexical Index Persistence

**Files:**
- Create: `tests/test_index_store.py`
- Create: `src/lightningsearch_rl/index_store.py`

- [ ] **Step 1: Write failing index store test**

```python
from lightningsearch_rl.data import Passage
from lightningsearch_rl.index_store import load_lexical_index, save_lexical_index


def test_save_and_load_lexical_index_preserves_search(tmp_path):
    index_path = tmp_path / "index.json"
    passages = [
        Passage("doc1", "Alpha", "Alpha text."),
        Passage("doc2", "Beta", "Beta born city."),
    ]

    save_lexical_index(index_path, passages)
    retriever = load_lexical_index(index_path)

    assert retriever.search("born city", top_k=1)[0].doc_id == "doc2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_index_store.py -v`
Expected: FAIL because `index_store` does not exist.

- [ ] **Step 3: Implement index persistence**

Create `src/lightningsearch_rl/index_store.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from lightningsearch_rl.corpus import passage_from_dict, passage_to_dict
from lightningsearch_rl.data import Passage
from lightningsearch_rl.retrieval import LexicalRetriever


INDEX_VERSION = 1


def save_lexical_index(path: Path, passages: list[Passage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": INDEX_VERSION, "passages": [passage_to_dict(p) for p in passages]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_lexical_index(path: Path) -> LexicalRetriever:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != INDEX_VERSION:
        raise ValueError(f"unsupported index version: {payload.get('version')}")
    return LexicalRetriever([passage_from_dict(row) for row in payload["passages"]])
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_index_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/index_store.py tests/test_index_store.py
git commit -m "feat: persist lexical retrieval indexes"
```

### Task 6: Add Retrieval Evaluation Metrics

**Files:**
- Create: `tests/test_retrieval_eval.py`
- Create: `src/lightningsearch_rl/retrieval_eval.py`

- [ ] **Step 1: Write failing retrieval eval test**

```python
from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.retrieval import LexicalRetriever
from lightningsearch_rl.retrieval_eval import evaluate_retrieval


def test_evaluate_retrieval_reports_recall_at_k(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    examples_path = tmp_path / "examples.jsonl"
    convert_hotpot_file(Path("tests/fixtures/hotpot_tiny_raw.json"), corpus_path, examples_path)

    examples = load_jsonl_examples(examples_path)
    retriever = LexicalRetriever(load_corpus_jsonl(corpus_path))

    metrics = evaluate_retrieval(examples, retriever, top_k=2)

    assert metrics["example_count"] == 1
    assert metrics["recall_at_2"] == 1.0
    assert metrics["evidence_recall_at_2"] == 1.0
    assert metrics["avg_retrieved_docs"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieval_eval.py -v`
Expected: FAIL because `retrieval_eval` does not exist.

- [ ] **Step 3: Implement retrieval eval**

Create `src/lightningsearch_rl/retrieval_eval.py` with `evaluate_retrieval(examples, retriever, top_k)`.

Use `example.question` as the query and compute:

- at least one gold doc retrieved for recall@k.
- fraction of gold docs retrieved for evidence recall.
- average retrieved docs.

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_retrieval_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/retrieval_eval.py tests/test_retrieval_eval.py
git commit -m "feat: evaluate retrieval recall metrics"
```

## Chunk 4: CLI Commands And Documentation

### Task 7: Add Phase 2A CLI Commands

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Create/modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Add a test that calls:

```python
main(["prepare-hotpot", "--raw", "tests/fixtures/hotpot_tiny_raw.json", "--corpus", str(corpus), "--examples", str(examples)])
main(["build-index", "--corpus", str(corpus), "--index", str(index)])
main(["eval-retrieval", "--examples", str(examples), "--index", str(index), "--out", str(metrics), "--top-k", "2"])
```

Assert all files exist and `metrics["recall_at_2"] == 1.0`.

- [ ] **Step 2: Run CLI test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL because commands do not exist.

- [ ] **Step 3: Implement CLI commands**

Add subparsers:

- `prepare-hotpot`
- `prepare-2wiki`
- `build-index`
- `eval-retrieval`

Compose existing modules only; keep `smoke` unchanged.

- [ ] **Step 4: Run CLI test**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Run full tests**

Run: `python -m pytest`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/cli.py tests/test_cli.py
git commit -m "feat: add shared corpus retrieval cli"
```

### Task 8: Update README And Verify

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Document:

```bash
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_tiny_raw.json --corpus results/phase2a/corpus.jsonl --examples results/phase2a/examples.jsonl
python -m lightningsearch_rl.cli build-index --corpus results/phase2a/corpus.jsonl --index results/phase2a/index.json
python -m lightningsearch_rl.cli eval-retrieval --examples results/phase2a/examples.jsonl --index results/phase2a/index.json --out results/phase2a/retrieval_metrics.json --top-k 2
```

- [ ] **Step 2: Run final verification**

Run:

```bash
python -m pytest
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_tiny_raw.json --corpus results/phase2a/corpus.jsonl --examples results/phase2a/examples.jsonl
python -m lightningsearch_rl.cli build-index --corpus results/phase2a/corpus.jsonl --index results/phase2a/index.json
python -m lightningsearch_rl.cli eval-retrieval --examples results/phase2a/examples.jsonl --index results/phase2a/index.json --out results/phase2a/retrieval_metrics.json --top-k 2
```

Expected: tests PASS and metrics contain `recall_at_2: 1.0`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document phase 2a retrieval workflow"
```

## Guardrails

- Use @superpowers:test-driven-development for each production behavior.
- Use @superpowers:verification-before-completion before completion claims.
- Do not download datasets.
- Do not add external dependencies.
- Do not launch remote training.
- Keep Phase 1 `smoke` behavior unchanged.
