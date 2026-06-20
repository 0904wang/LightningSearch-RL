# Phase 5O: Reward Dump GRPO Prelaunch

## Goal

Prepare a repeat of Phase 5N with train-time reward/response dumping enabled.
The run keeps the same `80/20` transition slice and `5` GRPO steps, but writes a
JSONL reward dump from `compute_score` so each scored response can be inspected
by reward stage and component.

## Code Changes

Added optional reward dump support:

```text
env var: LIGHTNINGSEARCH_REWARD_DUMP_PATH
env var: LIGHTNINGSEARCH_REWARD_DUMP_MAX_CHARS
schema: score, answer_reward, search_reward, format_reward, search_cost,
        reward_stage, parsed_action, solution_preview, ground_truth, extra_info
```

Added post-run reward dump diagnostics:

```text
python -m lightningsearch_rl.cli diagnose-reward-dump \
  --dump <reward_dump.jsonl> \
  --out <reward_dump_summary.json>
```

## Config

```text
config: configs/experiments/phase5o_env_transition_grpo_4gpu_100x5_rewarddump.yaml
script: scripts/remote/phase5o_env_transition_grpo_4gpu_100x5_rewarddump.sh
experiment_name: phase5o-env-transition-grpo-4gpu-100x5-rewarddump
source transitions: /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
```

## Data and Training Settings

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
reward_dump_path: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/reward_dump.jsonl
reward_dump_max_chars: 1024
```

## Dry Run

Remote dry-run completed in the approved conda env:

```text
targeted remote tests: 10 passed in 0.34s
parquet_written: true
train_rows: 80
val_rows: 20
summary_reward_dump_path: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/reward_dump.jsonl
summary_reward_dump_max_chars: 1024
manifest_reward_dump_path: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/reward_dump.jsonl
reward_dump_env_in_launch_command: true
```

Dry-run artifacts:

```text
results: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump
train parquet: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/data/train.parquet
val parquet: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/data/val.parquet
checkpoint dir: /data/wzl/LightningSearch-RL/checkpoints/phase5o-env-transition-grpo-4gpu-100x5-rewarddump
```

## Prelaunch GPU Check

```text
GPU 0: 3506 MiB / 32607 MiB
GPU 1: 3505 MiB / 32607 MiB
GPU 2: 3507 MiB / 32607 MiB
GPU 5: 3493 MiB / 32607 MiB
tmux sessions: none
```

GPU 3 and GPU 4 were busy. GPU 7 was mostly free, but this prelaunch keeps the
same 0/1/2/5 selection as Phase 5N for comparability.

## Proposed Launch

Session:

```text
lightningsearch-20260616-phase5o-rewarddump-100x5
```

Command:

```bash
tmux new-session -d -s lightningsearch-20260616-phase5o-rewarddump-100x5 "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5o_env_transition_grpo_4gpu_100x5_rewarddump.sh'"
```

Expected outputs:

```text
log: /data/wzl/LightningSearch-RL/logs/phase5o-env-transition-grpo-4gpu-100x5-rewarddump.log
metrics: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/reward_dump_summary.json
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5o-env-transition-grpo-4gpu-100x5-rewarddump
```

Note: `save_freq=-1`, so no checkpoint is expected.

## Success Criteria

1. `Training Progress: 100%` and `finished_at` are present.
2. No fatal markers: `CalledProcessError`, `Error executing job`, CUDA OOM, KV cache error.
3. `reward_dump.jsonl` is present and non-empty.
4. `reward_dump_summary.json` reports component means by `search` and `answer`
   stage.
5. If reward drops again, compare `reward_drop_alerts`,
   `batch_diagnostics.json`, and `reward_dump_summary.json`.
