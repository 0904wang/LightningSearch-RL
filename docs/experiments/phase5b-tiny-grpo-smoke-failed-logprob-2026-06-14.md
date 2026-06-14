# Phase 5B Tiny GRPO Smoke Failed Launch - Log-Prob Batch Fix

Date: 2026-06-14

## Goal

Launch the first 1-GPU verl / GRPO tiny smoke after the Phase 5B dry-run
passed.

## Launch Attempt

Approved GPU:

```text
CUDA_VISIBLE_DEVICES=7
```

Session:

```text
lightningsearch-20260614-phase5b-tiny-grpo-smoke
```

Log:

```text
/data/wzl/LightningSearch-RL/logs/phase5b-tiny-grpo-smoke.log
```

The tmux command attempted to launch:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train \
  --config configs/experiments/phase5b_tiny_grpo_smoke.yaml \
  --output-dir /data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke \
  --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke
```

## Result

The run exited quickly. No tmux session remained after inspection, and GPU 7
returned to idle.

Relevant log excerpt:

```text
ValueError: [actor_rollout_ref.rollout] Please set at least one of
'actor_rollout_ref.rollout.log_prob_micro_batch_size' or
'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu'.
```

This was a verl config validation failure before actual training started.

## Root Cause

The launcher set actor PPO micro-batch overrides:

```text
actor_rollout_ref.actor.ppo_mini_batch_size=2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
```

But verl 0.8.0 also requires rollout log-prob micro-batch configuration:

```text
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu
```

The Phase 5B launcher omitted that key.

## Fix

Added the missing override:

```text
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
```

Regression test added:

```text
tests/test_verl_smoke.py
```

Verification after fix:

```text
local pytest: 66 passed
remote pytest: 66 passed
remote dry-run: passed
```

Updated launch command now includes:

```text
'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1'
```

## Status

The failed attempt is recorded. The launcher has been fixed and re-synced to the
remote workspace. A new approval report is required before re-launching because
the exact command changed.
