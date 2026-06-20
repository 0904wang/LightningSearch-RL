# Phase 5X: variance-filtered transitions from Phase 5W

Date: 2026-06-19

## Goal

Create a small GRPO training subset from Phase 5W reward dumps by keeping only source examples whose same-question rollout group had nonzero reward variance.

The purpose is to test whether GRPO learns more consistently when batches contain examples that already produce within-group advantages.

## Launch

Remote session:

```bash
tmux new-session -d -s lightningsearch-20260619-phase5x-filter-transitions -c /data/wzl/LightningSearch-RL/repo "env CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5x_filter_transitions_from_phase5w_rankreward.sh"
```

Repo and environment:

- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- Remote repo state: not a git repository; narrow sync was used
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- GPU env: `CUDA_VISIBLE_DEVICES=7`

Inputs:

- Transitions: `/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward/reward_dump.jsonl`

Outputs:

- Results: `/data/wzl/LightningSearch-RL/results/phase5x-env-transitions-variance-rankreward-from-phase5w`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5x-env-transitions-variance-rankreward-from-phase5w.log`
- Filtered transitions: `/data/wzl/LightningSearch-RL/results/phase5x-env-transitions-variance-rankreward-from-phase5w/transitions.jsonl`
- Selected IDs: `/data/wzl/LightningSearch-RL/results/phase5x-env-transitions-variance-rankreward-from-phase5w/selected_source_ids.json`
- Variance groups: `/data/wzl/LightningSearch-RL/results/phase5x-env-transitions-variance-rankreward-from-phase5w/variance_groups.json`
- Summary: `/data/wzl/LightningSearch-RL/results/phase5x-env-transitions-variance-rankreward-from-phase5w/summary.json`

## Verification

Pre-run tests:

```text
Local full tests: 157 passed, 1 skipped
Remote related tests: 12 passed
Remote full tests: 161 passed
```

Final filter summary:

```text
input_transition_count=978
output_transition_count=16
selected_source_count=8
matched_source_count=8
unmatched_source_count=0
stage_variable_group_counts.answer=7
stage_variable_group_counts.search=1
```

Selected source IDs:

```text
syn-009405
syn-009452
syn-009519
syn-009577
syn-009632
syn-009719
syn-010032
syn-010164
```

Top variance groups:

```text
syn-009405 answer score_range=1.0
syn-009719 answer score_range=0.666667
syn-009632 search score_range=0.6
syn-009452 answer score_range=0.5
syn-009519 answer score_range=0.5
syn-009577 answer score_range=0.5
syn-010032 answer score_range=0.5
syn-010164 answer score_range=0.2
```

GRPO dry-run after filtering:

```text
train_rows=12
val_rows=4
parquet_written=True
```

## Analysis

The filter did exactly what Phase 5W analysis suggested: it isolated the small minority of examples where rollout_n=4 produced actual reward differences. The resulting dataset is intentionally tiny, 16 transition rows from 8 source examples, so it is not suitable for final performance claims.

It is useful as a diagnostic training set. A 50-step GRPO run on this set should show whether nonzero advantage frequency increases when training is concentrated on variable groups. If it still remains low, the bottleneck is likely rollout generation diversity during training rather than offline source selection.

## Next step

Run `phase5x-hard50-env-transition-grpo-4gpu-16x50-variance-rankreward` as a short 4-GPU smoke. Compare `nonzero_adv_count`, reward dump group variance, and answer/search stage score movement against Phase 5W.
