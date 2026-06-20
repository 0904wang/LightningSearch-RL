# Phase 5P Retry1: Soft Answer GRPO 4-GPU Smoke

## Goal

Verify that the soft answer reward can run through a 5-step `verl` GRPO smoke on
the Phase 5K environment-transition dataset, after fixing the inconsistent
`reward_extra_infos` schema that crashed the first Phase 5P attempt.

## Launch

Session:

```text
lightningsearch-20260617-phase5p-softanswer-retry1
```

Command:

```bash
tmux new-session -d -s lightningsearch-20260617-phase5p-softanswer-retry1 "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5p_env_transition_grpo_4gpu_100x5_softanswer_retry1.sh'"
```

Runtime context:

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpus: 0,1,2,5
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
config: configs/experiments/phase5p_env_transition_grpo_4gpu_100x5_softanswer_retry1.yaml
```

Paths:

```text
transitions: /data/wzl/LightningSearch-RL/results/phase5p-env-transitions-soft-answer-from-phase5k/transitions.jsonl
log: /data/wzl/LightningSearch-RL/logs/phase5p-env-transition-grpo-4gpu-100x5-softanswer-retry1.log
results: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer-retry1
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5p-env-transition-grpo-4gpu-100x5-softanswer-retry1
reward_dump: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer-retry1/reward_dump.jsonl
```

## Preflight

Local validation:

```text
tests/test_verl_reward.py tests/test_verl_reward_dump_diagnostics.py: 11 passed
full suite: 131 passed
```

Remote validation:

```text
tests/test_verl_reward.py tests/test_verl_reward_dump_diagnostics.py: 11 passed
full suite: 131 passed
```

Dry-run:

```text
parquet_written: true
train_rows: 80
val_rows: 20
reward_dump_path: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer-retry1/reward_dump.jsonl
```

GPU check before launch:

```text
0: 3506 MiB / 32607 MiB
1: 3505 MiB / 32607 MiB
2: 3507 MiB / 32607 MiB
5: 3493 MiB / 32607 MiB
```

## Result

Status: completed.

Raw completion markers:

```text
Training Progress: 100%|██████████| 5/5
finished_at=2026-06-16T16:13:57+00:00
metrics_summary.completed: true
metrics_summary.final_step: 5
metrics_summary.fatal_marker_count: 0
```

Reward curve:

```text
step 1: critic/rewards/mean = 1.0775001049041748
step 2: critic/rewards/mean = 1.09250009059906
step 3: critic/rewards/mean = 1.0850000381469727
step 4: critic/rewards/mean = 1.0850000381469727
step 5: critic/rewards/mean = 1.0425000190734863
```

Initial validation metrics:

```text
val-core/lightningsearch_rl/reward/mean@1: 1.0600000381469727
val-aux/lightningsearch_rl/score/mean@1: 1.06
val-aux/lightningsearch_rl/answer_reward/mean@1: 0.475
val-aux/lightningsearch_rl/answer_exact_match/mean@1: 0.45
val-aux/lightningsearch_rl/answer_token_f1/mean@1: 0.475
val-aux/lightningsearch_rl/answer_containment_match/mean@1: 0.5
val-aux/lightningsearch_rl/search_reward/mean@1: 0.5
val-aux/lightningsearch_rl/format_reward/mean@1: 1.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.015
val-aux/num_turns/mean: 2.0
```

Latest train metrics:

```text
training/global_step: 5
critic/rewards/mean: 1.0425000190734863
critic/rewards/max: 1.100000023841858
critic/rewards/min: 0.8999999761581421
response/aborted_ratio: 0.0
response_length/mean: 13.0
response_length/max: 22.0
prompt_length/mean: 246.75
num_turns/mean: 2.0
actor/perf/max_memory_allocated_gb: 10.225209712982178
actor/perf/max_memory_reserved_gb: 13.697265625
```

Reward dump summary:

```text
row_count: 40
stage_counts: answer=21, search=19
invalid_action_count: 0
low_score_count: 0
overall score mean: 1.06825
answer score mean: 1.066667
search score mean: 1.07
answer_reward_type_counts: exact=19, containment=2
```

Soft-credit examples:

```text
syn-009154: Vienna Conference Center vs Vienna -> answer_reward=0.5, score=0.6
syn-009020: Golden Quill Award vs Golden Quill -> answer_reward=0.8, score=0.9
```

Batch diagnostics:

```text
train_rows: 80
stage_counts: answer=40, search=40
low_reward_row_count: 2
low_reward_rows:
  syn-009012: expected <answer>Global Health Research Institute</answer>, reward_model_reward=0.1
  syn-009019: expected <answer>Journal of Computational Science</answer>, reward_model_reward=0.1
```

## Warnings

The run logged vLLM shutdown warnings after reaching `Training Progress: 100%`
and before the wrapper wrote `finished_at`:

```text
resource_tracker: process died unexpectedly, relaunching. Some resources might leak.
KeyError: '/psm_c1f65a9e'
KeyError: '/psm_4478281f'
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

The parser recorded these as shutdown warnings, not fatal markers. The important
difference from the first Phase 5P failure is that the previous
`KeyError: 'answer_reward_type'` is gone, all 5 training steps completed, and
post-run diagnostics were written.

## Analysis

The schema fix worked. Returning only stable numeric keys to `verl` avoids mixed
answer/search reward-extra aggregation failures while preserving
`answer_reward_type` in the JSONL diagnostics.

Soft answer reward also fixed the overly harsh exact-match behavior seen in
Phase 5O. The reward dump has no low-score answer cases, and the two expected
label-granularity mismatches receive partial credit instead of falling to the
old `0.1` formatted-wrong-answer score.

This is still a smoke-scale run, not evidence of policy improvement. The reward
curve is stable enough to justify scaling, but the dataset is tiny and the
agent-loop prompts are transition-style supervised states rather than fully
free environment interaction.

## Next Steps

1. Run a larger Phase 5Q GRPO smoke on the same soft-answer transition exporter,
   for example 200 to 500 transitions and 20 to 50 steps.
2. Keep reward dump enabled and compare exact / containment / invalid action
   counts against Phase 5O and Phase 5P retry1.
3. If stable, move from fixed transition GRPO to a fresh environment-rollout
   refresh before scaling dataset size.
