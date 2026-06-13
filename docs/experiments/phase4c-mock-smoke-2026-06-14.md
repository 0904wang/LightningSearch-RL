# Phase 4C Quality Validator Mock Smoke

Date: 2026-06-14

## Goal

Verify the stricter synthetic data quality controls on the approved remote
workspace without using the real LLM API.

## Code and Environment

- Commit: `0183771`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Results path: `/data/wzl/LightningSearch-RL/results/phase4c-mock-smoke`
- GPU: not used

## Verification

Remote tests:

```text
44 passed in 0.20s
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

## Quality Checks Added

The validator now rejects rows where:

- the answer appears in the question
- the answer equals a context title
- the answer appears in multiple supporting evidence sentences
- non-ASCII text appears in the row

## Artifacts

- Raw rows: `/data/wzl/LightningSearch-RL/results/phase4c-mock-smoke/synthetic_raw.jsonl`
- Valid rows: `/data/wzl/LightningSearch-RL/results/phase4c-mock-smoke/synthetic_valid.jsonl`
- Rejects: `/data/wzl/LightningSearch-RL/results/phase4c-mock-smoke/synthetic_rejects.jsonl`
- GRPO artifacts: `/data/wzl/LightningSearch-RL/results/phase4c-mock-smoke/grpo`

## Next Step

Run a real 200-valid-row pilot and compare against Phase 4B:

- valid rate
- reject reason distribution
- average reward
- answer-as-title count
- answer-in-question count
- answer-in-multiple-supporting-sentences count
