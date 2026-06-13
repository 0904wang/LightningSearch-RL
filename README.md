# LightningSearch-RL

Local MVP for a Lightning-style retrieval tool-use Agent RL framework.

Phase 1 validates the local contracts for data loading, offline retrieval, action parsing, trace collection, transition building, shaped rewards, and smoke evaluation. It does not run remote training or call external search APIs.

## Smoke Target

```bash
python -m pip install -e .
python -m pytest
python -m lightningsearch_rl.cli smoke --data tests/fixtures/tiny_multihop.jsonl --out-dir results/smoke
```

Expected smoke artifacts:

- `results/smoke/traces.jsonl`
- `results/smoke/transitions.jsonl`
- `results/smoke/metrics.json`

Expected tiny-fixture metrics:

```json
{
  "answer_em": 1.0,
  "avg_reward": 1.37,
  "avg_search_calls": 1.0,
  "evidence_recall": 1.0
}
```

## Current MVP Scope

Implemented local contracts:

- JSONL multi-hop QA fixture loading.
- Deterministic lexical retrieval.
- Structured action parsing for `<think>`, `<search>`, and `<answer>`.
- Rule-based search-agent episode tracing.
- Trace-to-transition conversion.
- Shaped reward calculation.
- Metric aggregation.
- Local smoke CLI artifact export.

Out of scope for this phase:

- Remote training launch.
- Qwen rollout.
- vLLM serving.
- verl GRPO integration.
- Full HotpotQA / 2Wiki preprocessing.
- FAISS dense retrieval.

## Next Phase Notes

- Add full HotpotQA / 2Wiki adapters.
- Build larger shared corpus indexes.
- Add Qwen rollout policy.
- Export SFT trajectories.
- Add verl-compatible GRPO reward hooks.
- Run the approved remote smoke test under `AGENTS.md`.

## Phase 2A Shared Corpus Workflow

Prepare a tiny HotpotQA-like fixture into shared corpus and examples files:

```bash
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_tiny_raw.json --corpus results/phase2a/corpus.jsonl --examples results/phase2a/examples.jsonl
```

Build a persisted lexical index:

```bash
python -m lightningsearch_rl.cli build-index --corpus results/phase2a/corpus.jsonl --index results/phase2a/index.json
```

Evaluate retrieval-only metrics:

```bash
python -m lightningsearch_rl.cli eval-retrieval --examples results/phase2a/examples.jsonl --index results/phase2a/index.json --out results/phase2a/retrieval_metrics.json --top-k 2
```

Expected tiny-fixture retrieval metrics:

```json
{
  "avg_retrieved_docs": 2.0,
  "evidence_recall_at_2": 1.0,
  "example_count": 1,
  "recall_at_2": 1.0
}
```
