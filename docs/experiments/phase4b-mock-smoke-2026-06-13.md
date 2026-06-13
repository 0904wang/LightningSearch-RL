# Phase 4B Validated Synthetic Mock Smoke

Date: 2026-06-13

## Goal

Verify the new target-valid synthetic data loop on the approved remote workspace
without using the real LLM API.

## Code and Environment

- Commit: `8133ec5`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Results path: `/data/wzl/LightningSearch-RL/results/phase4b-mock-smoke`
- GPU: not used

## Verification

Remote tests:

```text
40 passed in 0.20s
```

Validated synthesis summary:

```json
{
  "api_failed": 0,
  "batch_size": 3,
  "concurrency": 50,
  "generated": 5,
  "max_attempts": 8,
  "reject_count": 0,
  "requested": 5,
  "stopped_reason": "target_valid_reached",
  "target_valid": 5,
  "valid_count": 5
}
```

GRPO summary:

```json
{
  "avg_reward": 1.37,
  "avg_search_count": 1.0,
  "example_count": 5,
  "rollout_count": 5,
  "top_k": 2,
  "transition_count": 10
}
```

## Artifacts

- Raw rows: `/data/wzl/LightningSearch-RL/results/phase4b-mock-smoke/synthetic_raw.jsonl`
- Valid rows: `/data/wzl/LightningSearch-RL/results/phase4b-mock-smoke/synthetic_valid.jsonl`
- Rejects: `/data/wzl/LightningSearch-RL/results/phase4b-mock-smoke/synthetic_rejects.jsonl`
- Corpus: `/data/wzl/LightningSearch-RL/results/phase4b-mock-smoke/corpus.jsonl`
- Examples: `/data/wzl/LightningSearch-RL/results/phase4b-mock-smoke/examples.jsonl`
- Index: `/data/wzl/LightningSearch-RL/results/phase4b-mock-smoke/index.json`
- GRPO artifacts: `/data/wzl/LightningSearch-RL/results/phase4b-mock-smoke/grpo`

## Analysis

The new `synthesize-validated-data` command reaches the target valid count and
feeds the existing corpus, index, and GRPO export path. This clears the dry-run
gate for a real 200-valid-row DeepSeek pilot.
