# Phase 5N: Env Transition GRPO 4-GPU 100x5

## Goal

Scale the successful Phase 5M environment-transition GRPO smoke from `8/4` rows
and `1` training step to `80/20` rows and `5` GRPO training steps. This tests
whether the current low-context vLLM settings remain stable on a larger
transition slice before moving to longer runs.

## Runtime

date: 2026-06-16
session: `lightningsearch-20260616-phase5n-env-transition-grpo-100x5`
remote repo: `/data/wzl/LightningSearch-RL/repo`
remote repo type: narrow-synced working tree, not a git repo
conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
GPU selection: `CUDA_VISIBLE_DEVICES=0,1,2,5`
config: `configs/experiments/phase5n_env_transition_grpo_4gpu_100x5.yaml`
source transitions: `/data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl`
model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`

## Key Settings

```yaml
train_samples: 80
val_samples: 20
train_batch_size: 4
ppo_mini_batch_size: 4
ppo_micro_batch_size_per_gpu: 1
max_prompt_length: 384
max_response_length: 64
rollout_max_model_len: 512
rollout_max_num_batched_tokens: 768
rollout_gpu_memory_utilization: 0.25
total_training_steps: 5
save_freq: -1
test_freq: -1
```

## Launch

```bash
tmux new-session -d -s lightningsearch-20260616-phase5n-env-transition-grpo-100x5 "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5n_env_transition_grpo_4gpu_100x5.sh'"
```

The underlying `verl` command was written to:

```text
/data/wzl/LightningSearch-RL/results/phase5n-env-transition-grpo-4gpu-100x5/launch_command.txt
```

It used the project-local Hugging Face cache and the approved mirror:

```text
HF_HOME=/data/wzl/LightningSearch-RL/.cache/huggingface
HF_ENDPOINT=https://hf-mirror.com
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5n-env-transition-grpo-4gpu-100x5.log
results: /data/wzl/LightningSearch-RL/results/phase5n-env-transition-grpo-4gpu-100x5
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5n-env-transition-grpo-4gpu-100x5
metrics summary: /data/wzl/LightningSearch-RL/results/phase5n-env-transition-grpo-4gpu-100x5/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5n-env-transition-grpo-4gpu-100x5/batch_diagnostics.json
train parquet: /data/wzl/LightningSearch-RL/results/phase5n-env-transition-grpo-4gpu-100x5/data/train.parquet
val parquet: /data/wzl/LightningSearch-RL/results/phase5n-env-transition-grpo-4gpu-100x5/data/val.parquet
```

Note: `save_freq=-1`, so no model checkpoint is expected for this smoke.

## Data

```text
source_type: transitions
train_rows: 80
val_rows: 20
```

The manifest confirms both parquet files were written:

```text
train_parquet_written: true
val_parquet_written: true
```

## Outcome

The run completed all 5 GRPO training steps and wrote `metrics_summary.json`.

```text
started_at=2026-06-16T08:40:11+00:00
finished_at=2026-06-16T08:42:19+00:00
Training Progress: 100%|...| 5/5
completed: true
training_progress_100_seen: true
final_step: 5
fatal_marker_count: 0
shutdown_warning_count: 7
```

No fatal command markers were found:

```text
CalledProcessError: absent
Error executing job: absent
CUDA out of memory: absent
ValueError: To serve: absent
```

## Metrics

Initial validation at step 0:

```text
val-core/lightningsearch_rl/reward/mean@1: 1.0350000370293855
val-aux/lightningsearch_rl/score/mean@1: 1.0350000000000001
val-aux/lightningsearch_rl/answer_reward/mean@1: 0.45
val-aux/lightningsearch_rl/search_reward/mean@1: 0.5
val-aux/lightningsearch_rl/format_reward/mean@1: 1.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.015000000000000003
val-aux/num_turns/mean: 2.0
```

Training reward curve:

```text
step 1: critic/rewards/mean=1.0775001049041748, score=1.0775001049041748
step 2: critic/rewards/mean=1.09250009059906, score=1.09250009059906
step 3: critic/rewards/mean=1.0850000381469727, score=1.0850000381469727
step 4: critic/rewards/mean=1.0850000381469727, score=1.0850000381469727
step 5: critic/rewards/mean=0.5925000309944153, score=0.5925000309944153
```

Final training step:

```text
training/global_step: 5
critic/score/mean: 0.5925000309944153
critic/score/max: 1.100000023841858
critic/score/min: 0.10000000149011612
critic/rewards/mean: 0.5925000309944153
actor/loss: -0.714509129524231
actor/grad_norm: 1.59375
actor/ppo_kl: 0.0
actor/entropy: 0.0302572064101696
response_length/mean: 12.75
response_length/clip_ratio: 0.0
response/aborted_ratio: 0.0
prompt_length/mean: 246.75
prompt_length/clip_ratio: 0.0
num_turns/mean: 2.0
perf/time_per_step: 5.285496520344168
perf/throughput: 49.09661731895389
```

## Validation

Before launch:

```text
local tests: 118 passed in 3.25s
remote tests: 118 passed in 0.77s
remote dry-run: parquet_written=true, train_rows=80, val_rows=20
```

After completion:

```text
tmux sessions: none
GPU 0: 3506 MiB / 32607 MiB
GPU 1: 3505 MiB / 32607 MiB
GPU 2: 3507 MiB / 32607 MiB
GPU 5: 3493 MiB / 32607 MiB
metrics_summary.json: present
fatal markers: none
```

## Warnings

The log contains 7 shutdown warnings, consistent with the earlier Phase 5M
successful smoke. They appeared after training completion and did not include a
fatal command marker. Treat them as runtime shutdown noise for this smoke unless
they recur during checkpoint-saving or longer jobs.

The initial validation metrics are stored in `metrics_summary.json` under
`steps["0"]` as `val-*` fields. The top-level `initial_validation_metrics` field
is currently absent, so a small parser cleanup would make future summaries
clearer.

## Post-run Diagnostics

After the run, `verl_log_parser` was extended to surface validation metrics,
reward curves, shutdown warning examples, and large reward drops. The refreshed
Phase 5N summary reports:

```text
initial_validation_reward: 1.0350000370293855
reward_curve:
  step 1: 1.0775001049041748
  step 2: 1.09250009059906
  step 3: 1.0850000381469727
  step 4: 1.0850000381469727
  step 5: 0.5925000309944153
reward_drop_alerts:
  step 5 vs step 4 delta: -0.4925
shutdown_warning_count: 7
shutdown_warning_examples_count: 5
```

A separate `diagnose-verl-batches` pass inspected the prepared train JSONL under
a contiguous-batch assumption:

```text
batch_count: 20
overall_stage_counts: search=40, answer=40
overall_precomputed_reward_mean: 0.6475
step 5 aligned batch index: 4
step 5 aligned row range: [16, 20)
step 5 aligned stage_counts: search=2, answer=2
step 5 aligned precomputed_reward_mean: 0.685
step 5 aligned low_reward_row_count: 0
step 5 logged_reward_mean: 0.5925000309944153
```

This makes the reward drop less likely to be caused by an obviously low-quality
target transition batch. The likely explanations are generation quality on that
step, actual dataloader shuffling differing from the contiguous-batch
assumption, or missing train-time reward component logging.

## Analysis

This run confirms that the low-context Phase 5M settings scale to a larger
`80/20` transition slice and 5 optimizer updates without the previous vLLM KV
cache failure. Prompt and response clipping stayed at `0.0`, and no aborted
responses were reported.

The important new signal is the late-step reward drop: steps 1-4 are stable
around `1.08`, while step 5 falls to `0.5925`, with minimum reward `0.1`.
Because `ppo_kl` is still `0.0` and clip ratios are `0.0`, this looks more like
batch composition / reward distribution variance than an obvious PPO instability
in this tiny run. Before increasing to a long run, inspect the step-5 sampled
batch or add per-step reward component logging.

## Next Steps

1. Preserve the improved `metrics_summary.json` and `batch_diagnostics.json` as
   the default post-run analysis artifacts.
2. Add train-time reward component logging or response dumping so a future drop
   can be attributed to answer, search, format, or cost reward.
3. Run a controlled repeat with a different seed or fixed batch ordering before
   moving to a longer 50-100 step job.
