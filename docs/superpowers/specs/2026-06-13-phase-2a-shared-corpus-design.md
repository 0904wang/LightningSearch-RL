# Phase 2A Shared Corpus Design

## Context

Phase 1 built a local MVP that can load tiny inline-corpus QA examples, run a deterministic lexical retriever, collect traces, build transitions, compute shaped rewards, and export smoke artifacts.

Phase 2A should extend the data layer without touching model rollout or remote training. The goal is to make the project ready for HotpotQA / 2Wiki-style multi-hop datasets, where examples and corpora should not be duplicated inside every row.

## Goal

Add shared corpus support and tiny dataset adapters:

1. Load `corpus.jsonl` where each row is one passage.
2. Load `examples.jsonl` where each row references `gold_doc_ids` and optionally omits inline `corpus`.
3. Build and save a deterministic lexical index artifact.
4. Load that index artifact for search.
5. Convert tiny HotpotQA-like raw JSON into shared `corpus.jsonl` and `examples.jsonl`.
6. Convert tiny 2Wiki-like raw JSON into shared `corpus.jsonl` and `examples.jsonl`.
7. Evaluate retrieval recall@k and evidence coverage without running an agent policy.

## Scope

In scope:

- Standard-library-only JSON / JSONL parsing.
- Deduplicated passage IDs.
- Deterministic lexical index persistence.
- Tiny raw fixture adapters for HotpotQA-like and 2Wiki-like records.
- CLI commands:
  - `prepare-hotpot`
  - `prepare-2wiki`
  - `build-index`
  - `eval-retrieval`

Out of scope:

- Downloading real datasets.
- Full official HotpotQA / 2Wiki preprocessing.
- FAISS or dense retrieval.
- Qwen rollout.
- SFT export.
- verl integration.
- Remote execution.

## Data Contracts

Shared corpus row:

```json
{"doc_id":"hotpot::Example Book::0","title":"Example Book","text":"Example Book was written by Alice Smith."}
```

Prepared example row:

```json
{
  "id": "hp1",
  "question": "Which city is the birthplace of the author of Example Book?",
  "answers": ["Example City"],
  "gold_doc_ids": ["hotpot::Alice Smith::0"],
  "corpus_doc_ids": ["hotpot::Example Book::0", "hotpot::Alice Smith::0", "hotpot::Noise::0"]
}
```

Phase 1 inline-corpus examples stay supported.

## HotpotQA-Like Tiny Input

The adapter should support this local fixture shape:

```json
{
  "_id": "hp1",
  "question": "...",
  "answer": "Example City",
  "context": [
    ["Example Book", ["Example Book was written by Alice Smith."]],
    ["Alice Smith", ["Alice Smith was born in Example City."]]
  ],
  "supporting_facts": [["Alice Smith", 0]]
}
```

The adapter will emit one passage per sentence and mark supporting facts as gold docs.

## 2Wiki-Like Tiny Input

The adapter should support this local fixture shape:

```json
{
  "_id": "tw1",
  "question": "...",
  "answer": "Example City",
  "context": {
    "Example Book": ["Example Book was written by Alice Smith."],
    "Alice Smith": ["Alice Smith was born in Example City."]
  },
  "supporting_facts": [["Alice Smith", 0]]
}
```

The adapter will emit the same prepared contract as Hotpot.

## Index Contract

The MVP lexical index can be persisted as JSON:

```json
{
  "version": 1,
  "passages": [
    {"doc_id":"...","title":"...","text":"..."}
  ]
}
```

The current `LexicalRetriever` can rebuild its in-memory token statistics from persisted passages. That keeps storage simple and deterministic.

## Retrieval Evaluation

`eval-retrieval` should compute:

- `example_count`
- `recall_at_k`: fraction of examples with at least one gold doc retrieved.
- `evidence_recall_at_k`: average fraction of gold docs retrieved.
- `avg_retrieved_docs`

This is intentionally retrieval-only, separate from agent reward.

## Error Handling

- Empty or missing corpus files should raise clear `ValueError`s.
- Unknown adapter input shapes should raise clear `ValueError`s.
- Example rows without inline corpus and without shared corpus at runtime should fail clearly.
- Duplicate `doc_id`s in a corpus should keep the first passage by default for deterministic behavior.

## Testing

Use TDD for:

- Corpus JSONL load/write and deduplication.
- Prepared examples with shared corpus references.
- Hotpot tiny adapter.
- 2Wiki tiny adapter.
- Index save/load.
- Retrieval metrics.
- CLI artifact creation.

The phase is complete when `python -m pytest` passes and CLI commands can prepare fixtures, build an index, and evaluate retrieval metrics locally.
