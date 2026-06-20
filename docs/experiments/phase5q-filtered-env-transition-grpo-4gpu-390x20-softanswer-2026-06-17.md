# Phase 5Q Filtered 390-Transition Soft-Answer GRPO 20-Step Smoke

## Goal

Run a 4-GPU GRPO sanity check on the filtered Phase 5Q transition set after
removing confirmed `qa_type_mismatch` synthetic rows.

## Launch

Session:

```text
lightningsearch-20260617-phase5q-filtered-grpo-390x20
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260617-phase5q-filtered-grpo-390x20 bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5q_filtered_env_transition_grpo_4gpu_390x20_softanswer.sh'"
```

The first launch attempt failed before creating the tmux session because nested
PowerShell/SSH quotes were malformed:

```text
bash: -c: line 1: unexpected EOF while looking for matching `"'
```

The retry used the tmux positional command form shown above and started
successfully.

Runtime context:

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpus: 0,1,2,5
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
config: configs/experiments/phase5q_filtered_env_transition_grpo_4gpu_390x20_softanswer.yaml
```

Paths:

```text
transitions: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200-filtered/transitions.jsonl
log: /data/wzl/LightningSearch-RL/logs/phase5q-filtered-env-transition-grpo-4gpu-390x20-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5q-filtered-env-transition-grpo-4gpu-390x20-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5q-filtered-env-transition-grpo-4gpu-390x20-softanswer
reward_dump: /data/wzl/LightningSearch-RL/results/phase5q-filtered-env-transition-grpo-4gpu-390x20-softanswer/reward_dump.jsonl
```

## Preflight

Filtered data:

```text
input transitions: 390
train_rows: 312
val_rows: 78
excluded qa_type_mismatch rows: 5
filtered transition summary avg_total_reward: 1.360855
```

Validation before launch:

```text
local full suite: 136 passed
remote config test: 1 passed
remote dry run:
  train_rows: 312
  val_rows: 78
  parquet_written: true
```

## Result

Status: completed.

Raw completion markers:

```text
Training Progress: 100%|██████████| 20/20
training/global_step: 20
finished_at: 2026-06-17T03:27:57+00:00
metrics_summary.completed: true
metrics_summary.final_step: 20
metrics_summary.fatal_marker_count: 0
```

Initial validation:

```text
val-core/lightningsearch_rl/reward/mean@1: 1.0850000381469727
val-aux/lightningsearch_rl/score/mean@1: 1.085
val-aux/lightningsearch_rl/answer_reward/mean@1: 0.5
val-aux/lightningsearch_rl/answer_exact_match/mean@1: 0.5
val-aux/lightningsearch_rl/answer_token_f1/mean@1: 0.5
val-aux/lightningsearch_rl/answer_containment_match/mean@1: 0.5
val-aux/lightningsearch_rl/search_reward/mean@1: 0.5
val-aux/lightningsearch_rl/format_reward/mean@1: 1.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.015
val-aux/num_turns/mean: 2.0
```

Reward curve:

```text
step 1:  1.0850
step 2:  1.0775
step 3:  1.0017
step 4:  1.0850
step 5:  1.1000
step 6:  1.0850
step 7:  1.0700
step 8:  1.0925
step 9:  1.0700
step 10: 1.0850
step 11: 1.0925
step 12: 1.0775
step 13: 1.0850
step 14: 1.0850
step 15: 1.0850
step 16: 1.0925
step 17: 1.0850
step 18: 1.0850
step 19: 1.0850
step 20: 1.0925
```

Latest train metrics:

```text
training/global_step: 20
critic/rewards/mean: 1.09250009059906
critic/rewards/max: 1.100000023841858
critic/rewards/min: 1.0700000524520874
response/aborted_ratio: 0.0
response_length/mean: 13.25
response_length/max: 23.0
prompt_length/mean: 252.25
num_turns/mean: 2.0
actor/perf/max_memory_allocated_gb: 10.225209712982178
actor/perf/max_memory_reserved_gb: 13.744140625
```

Batch diagnostics:

```text
train_rows: 312
stage_counts: answer=156, search=156
low_reward_row_count: 0
precomputed_total_reward mean: 1.358568
precomputed_total_reward min: 0.87
```

Reward dump summary:

```text
row_count: 160
stage_counts: answer=80, search=80
invalid_action_count: 0
low_score_count: 0
overall score mean: 1.082917
overall score min: 0.766667
answer_reward_type_counts: exact=79, containment=1
```

GPU/tmux cleanup:

```text
tmux: no sessions
GPU 0: 3506 MiB
GPU 1: 3505 MiB
GPU 2: 3507 MiB
GPU 5: 3493 MiB
```

## Warnings

The run logged shutdown warnings after reaching `Training Progress: 100%`:

```text
RuntimeError: DataLoader worker ... is killed by signal: Killed.
resource_tracker: process died unexpectedly, relaunching. Some resources might leak.
KeyError: '/psm_...'
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

The parser classified these as shutdown warnings:

```text
shutdown_warning_count: 11
fatal_marker_count: 0
completed: true
```

This matches earlier Phase 5P and Phase 5Q-B behavior: vLLM/Ray teardown is
noisy after the training loop finishes, but metrics, batch diagnostics, and
reward dump diagnostics were written.

## Analysis

The filtered-data GRPO sanity run is stronger than the unfiltered Phase 5Q-B
run for data quality. The most important change is that both diagnostics now
show zero low-quality examples:

- batch diagnostics `low_reward_row_count` dropped to 0;
- reward dump `low_score_count` dropped to 0;
- invalid action count stayed 0;
- no reward-drop alerts were emitted;
- final reward mean reached 1.0925 with min 1.07 at the last step.

Step 3 still dips to 1.0017, but this is not the previous QA type-mismatch
failure mode. The batch diagnostics show no low-reward rows, and the reward dump
only has one containment answer among 80 answer-stage records. This makes the
remaining variance acceptable for a small 20-step smoke.

## Conclusion

The quality-manifest filter is effective and should be kept for larger runs.
The project now has a clean trace-to-transition GRPO path for:

1. deterministic offline env rollout,
2. quality-manifest filtering,
3. transition-level GRPO preparation,
4. reward dump diagnostics,
5. batch diagnostics tied back to source rows.

## Next Step

Scale the same filtered workflow to 500 rollout examples and about 1000
transitions. Keep the same diagnostics enabled, and record the unfiltered vs
filtered summary deltas so the final resume narrative can show a concrete
data-quality improvement rather than only a training metric.
