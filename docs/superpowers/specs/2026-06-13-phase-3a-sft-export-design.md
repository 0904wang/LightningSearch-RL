# Phase 3A SFT Export Design

## Context

The project can now prepare local HotpotQA / 2Wiki-like data, build a lexical index, run retrieval baselines, and collect rule-based traces for inline-corpus examples. The next local step is to export supervised fine-tuning data from deterministic search traces.

This phase does not train a model. It produces clean artifacts that later Qwen SFT / rollout code can consume.

## Goal

Add an SFT export workflow that:

1. Runs a deterministic search-agent trace over prepared examples and a shared lexical index.
2. Formats each trace as a structured `think/search/observation/think/answer` assistant message.
3. Writes SFT JSONL rows using a simple conversation schema.
4. Writes trace JSONL and summary JSON for auditability.
5. Keeps Phase 1 / 2 commands unchanged.

## Data Contract

SFT row:

```json
{
  "id": "hp_mixed_1",
  "messages": [
    {"role": "system", "content": "You are a search agent..."},
    {"role": "user", "content": "Question..."},
    {"role": "assistant", "content": "<think>...</think>\n<search>...</search>\n<observation>...</observation>\n<think>...</think>\n<answer>...</answer>"}
  ],
  "metadata": {
    "search_count": 1,
    "gold_doc_ids": ["..."],
    "retrieved_doc_ids": ["..."]
  }
}
```

Trace row:

```json
{
  "question_id": "...",
  "question": "...",
  "steps": [...],
  "final_answer": "...",
  "reward": null,
  "metadata": {...}
}
```

Summary:

```json
{
  "example_count": 1,
  "sft_rows": 1,
  "avg_search_count": 1.0
}
```

## Design

Add modules:

- `formatting.py`: observation and assistant message formatting.
- `sft.py`: export examples with a retriever to `sft.jsonl`, `traces.jsonl`, and `summary.json`.

Extend runtime:

- Add `run_retrieval_episode(example, retriever, top_k)` for shared-corpus examples. It mirrors `run_rule_based_episode` but uses an injected retriever.

Extend CLI:

- `export-sft --examples ... --index ... --out-dir ... --top-k 2`

## Error Handling

- If an example has no gold answer found in retrieved observations, final answer may be empty; export still writes trace for analysis.
- If examples file is empty, write empty artifacts and summary count `0`.
- Keep all output under explicit `--out-dir`.

## Testing

Use existing mixed Hotpot fixture:

- Prepare with limit 1.
- Build index.
- Export SFT.
- Assert assistant content includes `<search>`, `<observation>`, and `<answer>Example City</answer>`.
- Assert trace and summary files exist.

## Completion Criteria

- `python -m pytest` passes.
- Local Phase 3A command writes `sft.jsonl`, `traces.jsonl`, and `summary.json`.
- README documents command sequence.
