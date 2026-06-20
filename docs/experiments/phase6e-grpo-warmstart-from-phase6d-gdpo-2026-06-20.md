# Phase 6E GRPO Warmstart From Phase 6D GDPO

## Goal

Run a controlled GRPO warm-start from the Phase 6D GDPO warmup checkpoint. The purpose was to test whether the GDPO warmup checkpoint can provide a better starting point for the same hard50 variance-filtered transition slice than continuing from the Phase 5D SFT checkpoint.

## Code And Sync

- Local commit: `0d705c3 feat: add phase6e grpo warmstart launcher`
- Remote repository state at launch: `be4994e` plus a narrow sync of the Phase 6E config, launcher, and test files.
- Reason for narrow sync: remote `git fetch/pull` to GitHub failed once with `GnuTLS recv error (-110)` and the automatic retry hung in `git remote-https`; the hung fetch process was stopped by exact PID after inspection.
- Local verification before launch: `python -m pytest -q` -> `174 passed, 1 skipped`.
- Remote verification before launch: `PYTHONNOUSERSITE=1 python -m pytest -q` -> `178 passed`.

## Launch

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260620-phase6e-grpo-warmstart -c /data/wzl/LightningSearch-RL/repo "env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase6e_grpo_warmstart_from_phase6d_gdpo.sh"'
```

The launcher first merged the Phase 6D FSDP actor checkpoint:

```bash
PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/global_step_28/actor \
  --target_dir /data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28 \
  --use_cpu_initialization
```

Then it launched verl GRPO with the generated command in `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-from-phase6d-gdpo/launch_command.txt`.

## Paths

- Config: `/data/wzl/LightningSearch-RL/repo/configs/experiments/phase6e_grpo_warmstart_from_phase6d_gdpo.yaml`
- Input transitions: `/data/wzl/LightningSearch-RL/results/phase5y-env-transitions-variance-rankreward-100src/transitions.jsonl`
- Warm-start model: `/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28`
- Result dir: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-from-phase6d-gdpo`
- Checkpoint dir: `/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo`
- Final actor checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/global_step_28/actor`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6e-grpo-warmstart-from-phase6d-gdpo.log`
- Metrics: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-from-phase6d-gdpo/metrics_summary.json`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-from-phase6d-gdpo/reward_dump.jsonl`
- Reward dump summary: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-from-phase6d-gdpo/reward_dump_summary.json`
- Batch diagnostics: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-from-phase6d-gdpo/batch_diagnostics.json`

## Configuration

- Model: Qwen3-4B lineage, initialized from Phase 6D GDPO merged checkpoint.
- Advantage estimator: `grpo`
- Train / val rows: `112 / 14`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Rollouts per prompt: `rollout_n=4`
- Sampling: `temperature=1.2`, `top_p=0.95`, `top_k=50`
- Max prompt / response length: `384 / 64`
- Train batch size: `4`
- Total steps: `28`
- Save frequency: `28`
- Reward dump max chars: `1024`

## Raw Result Summary

```json
{
  "completed": true,
  "final_step": 28,
  "training_progress_100_seen": true,
  "fatal_marker_count": 0,
  "shutdown_warning_count": 11,
  "started_at": "2026-06-20T15:53:38+00:00",
  "finished_at": "2026-06-20T16:01:25+00:00",
  "initial_val_reward_mean": 0.6564285840306964,
  "latest_train_reward_mean": 0.6474999785423279,
  "latest_grad_norm": 9.0,
  "reward_mean_all_steps": 0.567825,
  "reward_mean_last5_steps": 0.62225,
  "nonzero_adv_count": 16,
  "nonzero_adv_steps": [1, 7, 8, 12, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28],
  "nonzero_grad_count": 16,
  "nonzero_grad_steps": [1, 7, 8, 12, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28],
  "reward_dump_row_count": 464,
  "stage_counts": {
    "answer": 216,
    "search": 248
  },
  "overall_score_mean": 0.570357,
  "overall_format_mean": 0.956897,
  "invalid_action_count": 0,
  "batch_low_reward_row_count": 75,
  "search_score_mean": 0.608306,
  "search_variable_group_rate": 0.070175,
  "answer_score_mean": 0.526786,
  "answer_variable_group_rate": 0.358491,
  "latest_checkpoint_iteration": "28"
}
```

## Log Excerpts

Successful completion and checkpoint:

```text
Training Progress: 100%|██████████| 28/28
local_global_step_folder: /data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/global_step_28
finished_at=2026-06-20T16:01:25+00:00
```

Initial validation:

```text
val-core/lightningsearch_rl/reward/mean@1:0.6564285840306964
val-aux/lightningsearch_rl/answer_exact_match/mean@1:0.21428571428571427
val-aux/lightningsearch_rl/search_reward/mean@1:0.35714285714285715
```

Shutdown warnings after checkpoint:

```text
RuntimeError: DataLoader worker ... is killed by signal: Killed.
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

These warnings happened after `Training Progress: 100%`, after `local_global_step_folder`, and after the checkpoint tracker was written, matching the teardown behavior seen in earlier successful short verl runs.

## Comparison To Phase 6D

Phase 6D GDPO warmup:

```json
{
  "reward_mean_all_steps": 0.576169,
  "reward_mean_last5_steps": 0.65975,
  "nonzero_adv_count": 3,
  "nonzero_grad_count": 3,
  "overall_score_mean": 0.580569,
  "answer_score_mean": 0.548721,
  "search_variable_group_rate": 0.070175,
  "answer_variable_group_rate": 0.339623
}
```

Phase 6E GRPO warm-start:

```json
{
  "reward_mean_all_steps": 0.567825,
  "reward_mean_last5_steps": 0.62225,
  "nonzero_adv_count": 16,
  "nonzero_grad_count": 16,
  "overall_score_mean": 0.570357,
  "answer_score_mean": 0.526786,
  "search_variable_group_rate": 0.070175,
  "answer_variable_group_rate": 0.358491
}
```

## Analysis

The run is successful as an infrastructure and optimization-signal experiment. It produced a valid `global_step_28` checkpoint and increased the number of nonzero GRPO update steps from `3/28` in Phase 6D to `16/28` in Phase 6E. This suggests the GRPO stage after GDPO warmup is getting much more usable group-level advantage signal than the GDPO-only stage.

The reward level itself did not improve in this short run. Mean train reward is slightly lower than Phase 6D (`0.567825` vs `0.576169`), and the last-five-step mean is also lower (`0.62225` vs `0.65975`). This is not enough evidence that the checkpoint is worse, because these are stochastic training-batch rewards on the same small hard50 slice, not held-out policy evaluation.

The persistent bottleneck remains search-stage variance. Search-stage variable group rate is still `0.070175`, unchanged from Phase 6D, while answer-stage variable group rate is higher at `0.358491`. That means most search rollouts for the same prompt still collapse to equivalent reward, so GRPO gets more signal mainly from answer-stage behavior rather than from improving the search query policy.

The positive point is that there were no invalid actions in the reward dump, and formatting remained high (`0.956897`). The model is still respecting the action protocol while receiving more GRPO updates.

## Next Steps

1. Merge the Phase 6E `global_step_28/actor` checkpoint to HF format.
2. Run a heldout hard50 deterministic and small stochastic eval comparing:
   - Phase 5D SFT baseline
   - Phase 6D GDPO warmup checkpoint
   - Phase 6E GDPO -> GRPO warm-start checkpoint
3. If Phase 6E moves behavior in eval, keep the GDPO -> GRPO story as a valid resume result.
4. If Phase 6E does not improve eval, shift the main technical contribution to data/reward design: better search-stage variance generation, harder negative query pairs, or source-grouped prompts that force alternative search actions to receive different rewards.
