# Phase 2B Real-Format Retrieval Baseline Design

## Context

Phase 2A added shared corpus support, tiny HotpotQA / 2Wiki adapters, persisted lexical indexes, and retrieval-only evaluation. The adapters currently support one narrow fixture shape per dataset.

Phase 2B should make the local pipeline closer to real dataset work without downloading datasets or launching remote jobs. The target is robust conversion of common HotpotQA / 2Wiki-like local files and a retrieval baseline report that can be used in experiment notes.

## Goal

Build a local baseline workflow that can:

1. Normalize common HotpotQA / 2Wiki field variants.
2. Convert a limited subset with `--limit N` for quick local experiments.
3. Write a JSON retrieval baseline report containing config, metrics, and paths.
4. Preserve all Phase 1 and Phase 2A behavior.

## Scope

In scope:

- Adapter normalization helpers for local JSON arrays and JSONL files.
- Field variants:
  - id: `_id` or `id`
  - answer: `answer`, `answers`, or `final_answer`
  - context: Hotpot list form, 2Wiki dict form, or list of passage dicts
  - supporting facts: list pairs or dicts with `title` and `sent_id`
- `limit` support in adapters and CLI.
- A retrieval baseline report file with:
  - dataset name
  - top_k
  - example_count
  - metrics
  - artifact paths

Out of scope:

- Downloading HotpotQA or 2Wiki.
- Hugging Face datasets integration.
- Multiprocessing index build.
- FAISS / dense retrieval.
- Agent rollout.
- Remote execution.

## Design

Add focused modules and keep existing module boundaries:

- `adapters.py`: add normalization helpers and optional `limit` parameter to conversion functions.
- `baseline.py`: add `run_retrieval_baseline(...)` that prepares metrics and writes a report.
- `cli.py`: add `--limit` to `prepare-hotpot` / `prepare-2wiki`; add `retrieval-baseline`.

The `retrieval-baseline` command should not redo conversion. It consumes prepared examples and an index, evaluates retrieval, and writes a report:

```json
{
  "dataset": "hotpot",
  "top_k": 2,
  "example_count": 2,
  "metrics": {
    "recall_at_2": 1.0,
    "evidence_recall_at_2": 1.0,
    "avg_retrieved_docs": 2.0
  },
  "artifacts": {
    "examples": "...",
    "index": "...",
    "report": "..."
  }
}
```

## Testing

Use local fixtures only:

- `hotpot_mixed_raw.jsonl`: two records using different field variants.
- `2wiki_mixed_raw.jsonl`: two records using dict/list variants.

Tests should verify:

- JSONL loading works.
- `limit=1` converts only one example and only its local corpus passages.
- supporting fact dicts and pairs both map to gold doc IDs.
- `retrieval-baseline` writes a report with metrics and paths.

## Completion Criteria

- `python -m pytest` passes.
- A local command sequence can prepare a limited mixed Hotpot fixture, build an index, and write a baseline report.
- README documents the Phase 2B command sequence.
