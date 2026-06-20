# Phase 5X: variance-filtered GRPO short run, stopped at 3 steps

Date: 2026-06-19

## Goal

Run a 50-step GRPO smoke on the Phase 5X variance-filtered transition set.

## Launch

Remote session:

```bash
tmux new-session -d -s lightningsearch-20260619-phase5x-grpo-variance-50 -c /data/wzl/LightningSearch-RL/repo "env CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5x_hard50_env_transition_grpo_4gpu_16x50_variance_rankreward.sh"
```

Config and paths:

- Config: `configs/experiments/phase5x_hard50_env_transition_grpo_4gpu_16x50_variance_rankreward.yaml`
- Input transitions: `/data/wzl/LightningSearch-RL/results/phase5x-env-transitions-variance-rankreward-from-phase5w/transitions.jsonl`
- Results: `/data/wzl/LightningSearch-RL/results/phase5x-hard50-env-transition-grpo-4gpu-16x50-variance-rankreward`
- Checkpoints: `/data/wzl/LightningSearch-RL/checkpoints/phase5x-hard50-env-transition-grpo-4gpu-16x50-variance-rankreward`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5x-hard50-env-transition-grpo-4gpu-16x50-variance-rankreward.log`
- GPUs: `4,5,6,7`

Dry-run before launch:

```text
train_rows=12
val_rows=4
parquet_written=True
```

## Final status

The run did not reach the intended 50 steps:

```text
completed=False
training_progress_100_seen=False
final_step=3
fatal_marker_count=0
shutdown_warning_count=10
started_at=2026-06-19T13:13:11+00:00
finished_at=2026-06-19T13:15:29+00:00
```

No `global_step_50` checkpoint was written because `save_freq=50` was never reached.

The relevant log lines show that the dataloader had only 3 training batches:

```text
dataset len: 12
Size of train dataloader: 3, Size of val dataloader: 1
Total training steps: 50
Training Progress: 6%|...| 3/50
```

Latest train metrics:

```text
latest_global_step=3
latest_reward_mean=0.5525000095367432
latest_adv_max=0.0
latest_adv_min=0.0
latest_grad_norm=0.0
```

Reward dump diagnostics:

```text
reward_rows=52
search_variable_group_rate=0.166667
answer_variable_group_rate=0.4
batch_count=3
train_rows=12
```

## Root cause

This is a configuration issue, not a reward implementation failure. The filtered training set has only 12 rows and `train_batch_size=4`, so one epoch contains only 3 batches. The config set `total_epochs=1`, which let verl stop after one pass through the dataloader even though `trainer.total_training_steps=50` was set.

The vLLM `EngineCore_DP0 died unexpectedly` messages appeared during teardown after the short run ended. They should not be interpreted as the primary cause of the short run.

## Fix

Create a retry config that keeps:

- `train_rows=12`
- `val_rows=4`
- `total_training_steps=50`
- `save_freq=50`
- rollout_n=4

but changes:

```text
total_epochs=20
```

This gives the tiny dataloader enough passes to reach the requested 50 steps and write `global_step_50`.
