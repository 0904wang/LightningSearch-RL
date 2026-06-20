# Phase 5Y Hard50 Variance GRPO 4GPU 126x28

Date: 2026-06-20 Asia/Shanghai

## Goal

Run a short 4-GPU GRPO diagnostic on the expanded Phase 5Y variance-filtered transition pool. This follows the Phase 5Y reward probe, which expanded the pool from 8 variable sources to 63 variable sources.

## Launch

Approved launch command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260620-phase5y-grpo-126x28 -c /data/wzl/LightningSearch-RL/repo 'env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5y_hard50_env_transition_grpo_4gpu_126x28_variance_rankreward.sh'"
```

Before launch:

- GPU 0-3 were below the 4000 MiB free threshold.
- Remote dry-run succeeded with `train_rows=112`, `val_rows=14`, and parquet files written.
- Local tests before sync: `160 passed, 1 skipped`.

## Paths

- Repo: `/data/wzl/LightningSearch-RL/repo`
- Env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Config: `/data/wzl/LightningSearch-RL/repo/configs/experiments/phase5y_hard50_env_transition_grpo_4gpu_126x28_variance_rankreward.yaml`
- Input transitions: `/data/wzl/LightningSearch-RL/results/phase5y-env-transitions-variance-rankreward-100src/transitions.jsonl`
- Result dir: `/data/wzl/LightningSearch-RL/results/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward`
- Checkpoint dir: `/data/wzl/LightningSearch-RL/checkpoints/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward`
- Checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward/global_step_28`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward.log`

## Settings

- Model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Train samples: `112`
- Val samples: `14`
- Train batch size: `4`
- Batches / update steps: `28`
- Rollouts per prompt: `4`
- Rollout temperature: `1.2`
- Rollout top-p: `0.95`
- Rollout top-k: `50`
- Max prompt length: `384`
- Max response length: `64`
- Reward: rank-aware search reward plus soft answer reward with answer token F1 threshold `0.5`
- Save frequency: `28`

## Status

The run completed.

- `completed`: `true`
- `training_progress_100_seen`: `true`
- `final_step`: `28`
- `latest_checkpointed_iteration`: `28`
- `fatal_marker_count`: `0`
- `shutdown_warning_count`: `11`
- Started: `2026-06-19T17:26:46+00:00`
- Finished: `2026-06-19T17:34:19+00:00`
- Checkpoint size: about `24G`

Shutdown warnings included DataLoader worker killed and vLLM engine shutdown messages after the 28/28 progress bar and checkpoint save. They are treated as teardown warnings because the final checkpoint and parsed metrics exist.

## Metrics

Initial validation:

```json
{
  "reward_mean": 0.6564285840306964,
  "score_mean": 0.6564285714285714,
  "answer_reward": 0.21428571428571427,
  "search_reward": 0.35714285714285715,
  "evidence_rank_reward": 0.35714285714285715
}
```

Training signal:

```json
{
  "reward_mean_all": 0.587479,
  "reward_last5": 0.476333,
  "nonzero_adv_count": 11,
  "nonzero_grad_count": 11,
  "nonzero_adv_steps": [1, 7, 10, 15, 16, 17, 19, 20, 24, 26, 27],
  "latest_train_reward_mean": 0.8349999785423279,
  "latest_train_grad_norm": 0.0
}
```

Reward dump:

```json
{
  "row_count": 464,
  "stage_counts": {
    "answer": 216,
    "search": 248
  },
  "overall_score_mean": 0.589333,
  "answer_score_mean": 0.570326,
  "search_score_mean": 0.605887,
  "answer_variable_group_rate": 0.264151,
  "search_variable_group_rate": 0.070175,
  "answer_reward_type_counts": {
    "containment": 15,
    "exact": 75,
    "none": 97,
    "token_f1": 29
  }
}
```

Batch diagnostics:

```json
{
  "batch_count": 28,
  "train_rows": 112,
  "train_batch_size": 4,
  "overall_low_reward_row_count": 75
}
```

## Analysis

The run is a successful GRPO smoke/diagnostic on the expanded variance pool. It reached the planned final step and produced a checkpoint.

The learning signal density is better than the earlier broad hard50 run, but it is not uniformly dense. `11/28` steps had nonzero advantage and nonzero gradient. The expanded variance pool is therefore useful, but many batches still collapse to identical rewards within rollout groups.

The search-stage signal improved compared with the probe baseline. In the probe pool, search variable group rate was very low; during this run, search variable group rate in the reward dump was `0.070175`, while answer variable group rate was `0.264151`. This is still answer-heavy, but not completely answer-only.

This run should not be interpreted as final performance improvement yet. There is no post-training heldout eval in this experiment, and the final logged train step had zero advantage. The value is that the training infrastructure now works on a larger, variance-filtered pool and produces measurable policy-gradient signal in a nontrivial subset of batches.

## Next Step

Run an evaluation against the saved `global_step_28` checkpoint and compare it with the SFT warmup model and previous Phase 5U/5X checkpoints. If eval does not improve, the next data-side change should target search-stage diversity rather than simply increasing epochs.
