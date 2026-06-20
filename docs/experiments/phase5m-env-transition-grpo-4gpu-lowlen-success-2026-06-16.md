# Phase 5M: Env Transition GRPO 4-GPU Lowlen Success

## Goal

Run a tiny 4-GPU `verl` GRPO smoke using Phase 5L environment transition
records. This closes the first end-to-end chain:

```text
real env rollout -> transition/reward export -> verl parquet -> GRPO update
```

## Runtime

date: 2026-06-16
session: `lightningsearch-20260616-phase5m-env-transition-grpo-lowlen`
remote repo: `/data/wzl/LightningSearch-RL/repo`
remote repo type: narrow-synced working tree, not a git repo
conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
GPU selection: `CUDA_VISIBLE_DEVICES=0,1,2,5`
config: `configs/experiments/phase5m_env_transition_grpo_4gpu_lowlen.yaml`
source transitions: `/data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl`
model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`

## Why Lowlen

The first Phase 5M attempt used `rollout_max_model_len=1280` and failed during
vLLM KV cache initialization. The measured prompt lengths for this tiny
transition batch were:

```text
max_prompt_tokens=301
max_prompt_plus_response_64=365
```

The lowlen retry therefore used:

```yaml
max_prompt_length: 384
max_response_length: 64
rollout_max_model_len: 512
rollout_max_num_batched_tokens: 768
rollout_gpu_memory_utilization: 0.25
```

## Launch

```bash
tmux new-session -d -s lightningsearch-20260616-phase5m-env-transition-grpo-lowlen "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5m_env_transition_grpo_4gpu_lowlen.sh'"
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5m-env-transition-grpo-4gpu-lowlen.log
results: /data/wzl/LightningSearch-RL/results/phase5m-env-transition-grpo-4gpu-lowlen
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5m-env-transition-grpo-4gpu-lowlen
train parquet: /data/wzl/LightningSearch-RL/results/phase5m-env-transition-grpo-4gpu-lowlen/data/train.parquet
val parquet: /data/wzl/LightningSearch-RL/results/phase5m-env-transition-grpo-4gpu-lowlen/data/val.parquet
```

Note: `save_freq=-1`, so no model checkpoint is expected for this smoke.

## Data

```text
source_type: transitions
train_rows: 8
val_rows: 4
```

## Outcome

The run completed one GRPO training step and reached `finished_at`.

```text
started_at=2026-06-16T08:24:51+00:00
finished_at=2026-06-16T08:26:37+00:00
Training Progress: 100%|██████████| 1/1
```

No fatal command markers were found:

```text
CalledProcessError: absent
Error executing job: absent
CUDA out of memory: absent
ValueError: To serve: absent
```

## Metrics

Initial validation:

```text
val-core/lightningsearch_rl/reward/mean@1: 0.8350000325590372
val-aux/lightningsearch_rl/score/mean@1: 0.8350000000000001
val-aux/lightningsearch_rl/answer_reward/mean@1: 0.25
val-aux/lightningsearch_rl/search_reward/mean@1: 0.5
val-aux/lightningsearch_rl/format_reward/mean@1: 1.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.015
val-aux/num_turns/mean: 2.0
```

Training step 1:

```text
training/global_step: 1
critic/score/mean: 1.0850000381469727
critic/score/max: 1.100000023841858
critic/score/min: 1.0700000524520874
critic/rewards/mean: 1.0850000381469727
actor/loss: -1.0796763896942139
actor/grad_norm: 0.0284423828125
actor/lr: 1e-06
response_length/mean: 15.5
response_length/max: 22.0
response_length/clip_ratio: 0.0
response/aborted_ratio: 0.0
prompt_length/mean: 191.25
prompt_length/max: 297.0
prompt_length/clip_ratio: 0.0
num_turns/mean: 2.0
actor/perf/max_memory_allocated_gb: 9.381478309631348
actor/perf/max_memory_reserved_gb: 13.560546875
perf/time_per_step: 17.620004686992615
perf/throughput: 11.733822077392881
```

## Warnings

The log contains expected runtime warnings:

- Qwen tokenizer regex warning from Transformers.
- `main_ppo.py` and Ray trainer deprecation warnings.
- vLLM shutdown messages after the completed step:
  - `DataLoader worker ... is killed by signal: Killed`
  - `Engine core proc ... died unexpectedly, shutting down client`
  - `resource_tracker` shared-memory warnings

These appeared after `Training Progress: 100%`, `step:1`, and `finished_at`, and
the shell command did not report `CalledProcessError`. Treat them as shutdown
noise for this smoke, not as a failed training command.

## Validation

Before launch:

```text
local tests: 114 passed in 3.03s
remote lowlen config test: 1 passed in 0.27s
remote dry-run: parquet_written=true, train_rows=8, val_rows=4
```

After completion:

```text
tmux sessions: none
GPU 0: 3506 MiB / 32607 MiB
GPU 1: 3505 MiB / 32607 MiB
GPU 2: 3507 MiB / 32607 MiB
GPU 5: 3493 MiB / 32607 MiB
```

## Analysis

This is the first successful GRPO smoke using environment-derived transition
records. Compared with earlier Phase 5F, it no longer trains from static
gold-answer rollouts or SFT-turn rows only. The data source is now the real model
environment rollout path:

```text
Phase 5K env_rollouts.jsonl
  -> Phase 5L transitions.jsonl / reward_records.jsonl
  -> Phase 5M verl parquet
  -> one GRPO actor update
```

The reward numbers are consistent with mixed search and answer transition
stages: search-stage rows contribute search/format reward, answer-stage rows
contribute answer/format reward.

## Next Steps

1. Preserve this as the milestone that proves the end-to-end training loop.
2. Increase Phase 5M train/val rows from 8/4 to a larger slice, after deciding
   whether to filter the suspicious synthetic labels from Phase 5K.
3. Add a clearer metric export parser for `verl` logs so future runs write a
   compact `metrics_summary.json`.
