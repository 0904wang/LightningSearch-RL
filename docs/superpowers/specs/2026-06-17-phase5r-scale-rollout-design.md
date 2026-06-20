# Phase 5R Scale Rollout Design

## Goal

Scale the clean Phase 5Q workflow from 200 rollout examples to 500 rollout
examples before running the next GRPO experiment.

## Design

Phase 5R reruns the environment rollout from `offset=0` with `limit=500` using
the same SFT warmup checkpoint, lexical index, gold-distractor candidate pool,
`top_k=8`, `distractor_count=6`, and `max_new_tokens=64`.

The script writes both raw and filtered transition exports:

- raw: `/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500`
- filtered: `/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered`

Filtering uses the existing deterministic quality manifest:

```text
/data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5q_known_mismatches.json
```

The initial GRPO config assumes the expected filtered size is 990 transitions
after removing five known two-step examples from 500 rollouts. It uses an 80/20
split, 792 train rows and 198 validation rows, and increases the run to 50
steps while keeping the known-stable 4-GPU memory settings from Phase 5Q.

## Gates

1. Local config tests must pass.
2. Remote tests and dry-run must pass.
3. The real 500-rollout data prep launch must be reported and approved before
   tmux starts.
4. The 50-step GRPO launch must wait until the filtered export summary confirms
   the actual transition count and diagnostics.
