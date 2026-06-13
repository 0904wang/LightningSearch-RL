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

## Phase 2B Retrieval Baseline Workflow

Prepare a mixed-format HotpotQA-like local file with a small limit:

```bash
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_mixed_raw.jsonl --corpus results/phase2b/corpus.jsonl --examples results/phase2b/examples.jsonl --limit 1
```

Build the index:

```bash
python -m lightningsearch_rl.cli build-index --corpus results/phase2b/corpus.jsonl --index results/phase2b/index.json
```

Write a retrieval baseline report:

```bash
python -m lightningsearch_rl.cli retrieval-baseline --dataset hotpot --examples results/phase2b/examples.jsonl --index results/phase2b/index.json --report results/phase2b/baseline_report.json --top-k 2
```

Expected report fields:

- `dataset`
- `top_k`
- `example_count`
- `metrics`
- `artifacts`

## Phase 3A SFT Export Workflow

Prepare a limited HotpotQA-like file and build the shared lexical index:

```bash
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_mixed_raw.jsonl --corpus results/phase3a/corpus.jsonl --examples results/phase3a/examples.jsonl --limit 1
python -m lightningsearch_rl.cli build-index --corpus results/phase3a/corpus.jsonl --index results/phase3a/index.json
```

Export deterministic search-agent trajectories as SFT conversations:

```bash
python -m lightningsearch_rl.cli export-sft --examples results/phase2b/examples.jsonl --index results/phase2b/index.json --out-dir results/phase3a/sft --top-k 2
```

Expected SFT artifacts:

- `results/phase3a/sft/sft.jsonl`
- `results/phase3a/sft/traces.jsonl`
- `results/phase3a/sft/summary.json`

## Phase 3B GRPO Export Workflow

Prepare a limited HotpotQA-like file and build the shared lexical index:

```bash
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_mixed_raw.jsonl --corpus results/phase3b/corpus.jsonl --examples results/phase3b/examples.jsonl --limit 1
python -m lightningsearch_rl.cli build-index --corpus results/phase3b/corpus.jsonl --index results/phase3b/index.json
```

Export rollout, transition, and reward records for later GRPO training:

```bash
python -m lightningsearch_rl.cli export-grpo --examples results/phase3b/examples.jsonl --index results/phase3b/index.json --out-dir results/phase3b/grpo --top-k 2
```

Expected GRPO artifacts:

- `results/phase3b/grpo/rollouts.jsonl`
- `results/phase3b/grpo/transitions.jsonl`
- `results/phase3b/grpo/reward_records.jsonl`
- `results/phase3b/grpo/summary.json`

## Phase 4A Synthetic Data Workflow

Generate HotpotQA-like synthetic raw rows without API usage first:

```bash
python -m lightningsearch_rl.cli synthesize-data --mock --out results/phase4a/synthetic_raw.jsonl --count 10 --topics awards,archives,research --concurrency 50 --seed 0 --summary results/phase4a/synthesis_summary.json
```

Validate rows before converting them into corpus/examples:

```bash
python -m lightningsearch_rl.cli validate-synthetic --raw results/phase4a/synthetic_raw.jsonl --valid results/phase4a/synthetic_valid.jsonl --rejects results/phase4a/synthetic_rejects.jsonl --summary results/phase4a/validation_summary.json
python -m lightningsearch_rl.cli prepare-hotpot --raw results/phase4a/synthetic_valid.jsonl --corpus results/phase4a/corpus.jsonl --examples results/phase4a/examples.jsonl
python -m lightningsearch_rl.cli build-index --corpus results/phase4a/corpus.jsonl --index results/phase4a/index.json
python -m lightningsearch_rl.cli export-grpo --examples results/phase4a/examples.jsonl --index results/phase4a/index.json --out-dir results/phase4a/grpo --top-k 2
```

For real DeepSeek synthesis, set `DEEPSEEK_API_KEY` in the shell environment and
omit `--mock`. Do not put the key in command arguments, config files, or logs.
The default endpoint is `https://api.deepseek.com`, and the default model is
`deepseek-chat`; both are configurable:

```bash
python -m lightningsearch_rl.cli synthesize-data --out results/phase4a/synthetic_raw.jsonl --count 100 --topics awards,archives,research --concurrency 50 --seed 0 --summary results/phase4a/synthesis_summary.json --model deepseek-chat --base-url https://api.deepseek.com
```

Expected Phase 4A artifacts:

- `results/phase4a/synthetic_raw.jsonl`
- `results/phase4a/synthesis_summary.json`
- `results/phase4a/synthetic_valid.jsonl`
- `results/phase4a/synthetic_rejects.jsonl`
- `results/phase4a/validation_summary.json`
- downstream corpus, index, and GRPO export files
