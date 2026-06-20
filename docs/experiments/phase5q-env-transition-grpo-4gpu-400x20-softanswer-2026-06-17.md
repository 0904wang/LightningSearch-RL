# Phase 5Q-B: 400-Transition Soft-Answer GRPO 20-Step Smoke

## Goal

Run the first scaled GRPO smoke beyond Phase 5P:

- 400 exported environment transitions from Phase 5Q-A
- 320 train transitions and 80 validation transitions
- 20 GRPO steps on 4 GPUs
- soft answer reward and reward dump enabled

## Launch

Session:

```text
lightningsearch-20260617-phase5q-grpo-400x20
```

Command:

```bash
tmux new-session -d -s lightningsearch-20260617-phase5q-grpo-400x20 "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5q_env_transition_grpo_4gpu_400x20_softanswer.sh'"
```

Runtime context:

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpus: 0,1,2,5
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
config: configs/experiments/phase5q_env_transition_grpo_4gpu_400x20_softanswer.yaml
```

Paths:

```text
transitions: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/transitions.jsonl
log: /data/wzl/LightningSearch-RL/logs/phase5q-env-transition-grpo-4gpu-400x20-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5q-env-transition-grpo-4gpu-400x20-softanswer
reward_dump: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer/reward_dump.jsonl
```

## Preflight

Data source from Phase 5Q-A:

```text
env_rollouts.jsonl: 200 rows
transitions.jsonl: 400 rows
reward_records.jsonl: 200 rows
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.95
answer_containment_match_rate: 0.975
gold_evidence_recall: 0.9975
avg_total_reward: 1.335583
```

Training dry-run:

```text
parquet_written: true
train_rows: 320
val_rows: 80
total_training_steps: 20
reward_dump_path: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer/reward_dump.jsonl
```

Validation before launch:

```text
local full suite: 132 passed
remote full suite: 132 passed
```

## Result

Status: completed.

Raw completion markers:

```text
Training Progress: 100%|██████████| 20/20
step:20 ... training/global_step:20
finished_at=2026-06-16T16:48:14+00:00
metrics_summary.completed: true
metrics_summary.final_step: 20
metrics_summary.fatal_marker_count: 0
```

Initial validation:

```text
val-core/lightningsearch_rl/reward/mean@1: 1.072500037867576
val-aux/lightningsearch_rl/score/mean@1: 1.0725
val-aux/lightningsearch_rl/answer_reward/mean@1: 0.4875
val-aux/lightningsearch_rl/answer_exact_match/mean@1: 0.4875
val-aux/lightningsearch_rl/answer_token_f1/mean@1: 0.49375
val-aux/lightningsearch_rl/answer_containment_match/mean@1: 0.4875
val-aux/lightningsearch_rl/search_reward/mean@1: 0.5
val-aux/lightningsearch_rl/format_reward/mean@1: 1.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.015
```

Reward curve:

```text
step 1:  1.0775
step 2:  1.0925
step 3:  0.8425
step 4:  1.0925
step 5:  1.0775
step 6:  1.0850
step 7:  1.0850
step 8:  1.0850
step 9:  1.0775
step 10: 1.0700
step 11: 1.0925
step 12: 1.0850
step 13: 1.0925
step 14: 1.0850
step 15: 1.0850
step 16: 1.0850
step 17: 1.0850
step 18: 1.0775
step 19: 1.1000
step 20: 1.0850
```

Latest train metrics:

```text
training/global_step: 20
critic/rewards/mean: 1.0850000381469727
critic/rewards/max: 1.100000023841858
critic/rewards/min: 1.0700000524520874
response/aborted_ratio: 0.0
response_length/mean: 16.5
response_length/max: 27.0
prompt_length/mean: 199.75
num_turns/mean: 2.0
actor/perf/max_memory_allocated_gb: 10.225209712982178
actor/perf/max_memory_reserved_gb: 13.7421875
```

Reward dump summary:

```text
row_count: 160
stage_counts: answer=81, search=79
invalid_action_count: 0
low_score_count: 2
overall score mean: 1.072687
answer score mean: 1.075309
search score mean: 1.07
answer_reward_type_counts: exact=79, none=2
```

Low-score reward dump examples:

```text
syn-009536: <answer>Lakeside University</answer>, answer_reward=0.0, score=0.1
syn-009432: <answer>European Institute of Technology</answer>, answer_reward=0.0, score=0.1
```

Batch diagnostics:

```text
train_rows: 320
low_reward_row_count: 4
low_reward_rows:
  syn-009012: <answer>Global Health Research Institute</answer>, reward_model_reward=0.1
  syn-009019: <answer>Journal of Computational Science</answer>, reward_model_reward=0.1
  syn-009432: <answer>European Institute of Technology</answer>, reward_model_reward=0.1
  syn-009456: <answer>Polar Archives</answer>, reward_model_reward=0.1
```

## Warnings

The run logged shutdown warnings after reaching `Training Progress: 100%`:

```text
RuntimeError: DataLoader worker (pid 130823) is killed by signal: Killed.
resource_tracker: process died unexpectedly, relaunching. Some resources might leak.
KeyError: '/psm_a76d7b88'
KeyError: '/psm_186d5ca5'
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

The log parser classified these as shutdown warnings:

```text
shutdown_warning_count: 11
fatal_marker_count: 0
completed: true
```

This matches the Phase 5P retry behavior: vLLM/Ray teardown is noisy after the
training loop completes, but it did not prevent metrics, batch diagnostics, or
reward dump diagnostics from being written.

## Analysis

The scaled 20-step GRPO smoke is usable. It completes all 20 steps, keeps reward
means stable after the expected low-reward batch at step 3, and does not show
format instability or aborted responses. The reward dump confirms that the
online reward path still uses the fixed numeric schema and no invalid action
spike appears.

The remaining low-score examples are inherited data-quality issues in the
synthetic QA set, not tool-use failures. This is acceptable for a smoke run, but
larger training should either filter these rows or tag them for analysis so
resume-facing metrics are not confused with model/tool failures.

## Next Steps

1. Add a small data filter or split tag for known QA-type mismatches before
   larger runs.
2. Scale to 500 rollout examples and 1000 transitions only after preserving
   reward dump diagnostics.
3. Consider a post-GRPO evaluation pass on held-out env rollout prompts rather
   than judging only training reward.
