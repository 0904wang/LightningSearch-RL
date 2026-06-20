# Phase 6D GDPO Warmup From Phase 5Y Retry1

Date: 2026-06-20 Asia/Shanghai

## Goal

Run a short verl GDPO warmup before a follow-up GRPO warm-start experiment. This warmup uses the Phase 5Y variance-filtered hard50 transition slice and optimizes with the GDPO reward manager over `search_reward` and `format_reward`.

## Launch

The first launch on GPUs `4,5,6,7` failed before training because GPU 7 was occupied by another process. This retry used GPUs `0,1,2,3`.

Approved retry command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260620-phase6d-gdpo-warmup-retry1 -c /data/wzl/LightningSearch-RL/repo "env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase6d_gdpo_warmup_from_phase5y.sh"'
```

Before launch:

- Remote repo: `main@be4994e`
- GPU 0-3 memory was below the 4000 MiB free threshold.
- `tmux list-sessions` reported no active tmux server.
- First failed log was preserved as `/data/wzl/LightningSearch-RL/logs/phase6d-gdpo-warmup-from-phase5y-failed-oom.log`.

## Paths

- Repo: `/data/wzl/LightningSearch-RL/repo`
- Env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Config: `/data/wzl/LightningSearch-RL/repo/configs/experiments/phase6d_gdpo_warmup_from_phase5y.yaml`
- Input transitions: `/data/wzl/LightningSearch-RL/results/phase5y-env-transitions-variance-rankreward-100src/transitions.jsonl`
- Result dir: `/data/wzl/LightningSearch-RL/results/phase6d-gdpo-warmup-from-phase5y`
- Checkpoint dir: `/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y`
- Checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/global_step_28`
- Actor checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/global_step_28/actor`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6d-gdpo-warmup-from-phase5y.log`
- Metrics: `/data/wzl/LightningSearch-RL/results/phase6d-gdpo-warmup-from-phase5y/metrics_summary.json`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase6d-gdpo-warmup-from-phase5y/reward_dump.jsonl`
- Reward dump summary: `/data/wzl/LightningSearch-RL/results/phase6d-gdpo-warmup-from-phase5y/reward_dump_summary.json`
- Batch diagnostics: `/data/wzl/LightningSearch-RL/results/phase6d-gdpo-warmup-from-phase5y/batch_diagnostics.json`

## Settings

- Model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Train samples: `112`
- Val samples: `14`
- Train batch size: `4`
- Rollouts per prompt: `4`
- Max prompt length: `384`
- Max response length: `64`
- Rollout temperature / top-p / top-k: `1.2 / 0.95 / 50`
- Advantage estimator: `gdpo`
- GDPO reward keys: `["search_reward", "format_reward"]`
- Reward manager: `gdpo`
- Planned training steps: `28`
- Save frequency: `28`

## Status

The run completed and wrote the planned checkpoint.

- `completed`: `true`
- `training_progress_100_seen`: `true`
- `final_step`: `28`
- `latest_checkpointed_iteration`: `28`
- `fatal_marker_count`: `0`
- `shutdown_warning_count`: `11`
- Started: `2026-06-20T15:04:45+00:00`
- Finished: `2026-06-20T15:12:13+00:00`
- Checkpoint size: about `24G`

Shutdown warnings included DataLoader worker killed and vLLM engine shutdown messages after `Training Progress: 100%` and checkpoint save. They match prior successful short verl runs and are treated as teardown warnings because the checkpoint and parsed metrics exist.

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
  "reward_mean_all": 0.576169,
  "reward_last5": 0.65975,
  "latest_train_reward_mean": 0.8349999785423279,
  "nonzero_adv_count": 3,
  "nonzero_adv_steps": [8, 19, 25],
  "nonzero_grad_count": 3,
  "nonzero_grad_steps": [8, 19, 25],
  "latest_grad_norm": 0.0,
  "gdpo_search_reward_mean_all": 0.28817,
  "gdpo_format_reward_mean_all": 0.962054
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
  "overall_score_mean": 0.580569,
  "search_score_mean": 0.608306,
  "answer_score_mean": 0.548721,
  "search_variable_group_rate": 0.070175,
  "answer_variable_group_rate": 0.339623,
  "invalid_action_count": 0,
  "low_score_count": 149
}
```

Batch diagnostics:

```json
{
  "batch_count": 28,
  "train_rows": 112,
  "train_batch_size": 4,
  "batch_low_reward_row_count": 75
}
```

## Analysis

This is a successful infrastructure warmup: verl accepted the GDPO reward manager, completed 28 steps, produced reward diagnostics, and wrote `global_step_28`.

The training signal is weaker than the earlier Phase 5Y GRPO run. Only `3/28` steps had nonzero advantages and nonzero actor gradient. Phase 5Y GRPO had `11/28` nonzero-advantage steps on the same transition slice. The difference is expected from the chosen GDPO reward keys: `format_reward` is mostly saturated, and `search_reward` group variance remains low.

The reward dump confirms the bottleneck. Search-stage variable group rate is `0.070175`, while answer-stage variable group rate is `0.339623`. Search scores are generally higher on average (`0.608306`) than answer scores (`0.548721`), but most same-question search rollouts still collapse to identical or near-identical reward. Thus GDPO generated some policy-gradient signal, but it is sparse.

This checkpoint is still useful as a warmup candidate because it did move on 3 batches and completed cleanly. The next step should not be a longer GDPO run yet; first evaluate whether the GDPO checkpoint changes deterministic or stochastic hard50 behavior. If it does not move behavior, use it only as a comparison point and continue with GRPO warm-start or better search-diverse data.

## Next Step

Merge `/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/global_step_28/actor` to a Hugging Face checkpoint, then run a heldout hard50 eval comparing:

- Phase 5D SFT baseline
- Phase 5Y GRPO checkpoint
- Phase 6D GDPO warmup checkpoint

If the GDPO checkpoint shows useful policy movement, start Phase 6E GRPO warm-start from the merged GDPO checkpoint. If it does not, Phase 6E should still be run as a controlled comparison, but the resume conclusion should frame GDPO as an attempted preference warmup with sparse advantage signal.
