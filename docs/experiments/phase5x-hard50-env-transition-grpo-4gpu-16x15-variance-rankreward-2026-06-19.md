# Phase 5X: variance-filtered GRPO 15-step diagnostic

Date: 2026-06-19

## Goal

Run a conservative GRPO diagnostic on the Phase 5X variance-filtered transition set.

This run uses only 16 filtered transitions from 8 source examples. It is not intended as a generalization or benchmark run. The goal is to test whether training on examples with known group-level reward variance increases nonzero GRPO advantage and gradient frequency.

## Launch

Remote session:

```bash
tmux new-session -d -s lightningsearch-20260619-phase5x-grpo-variance-15 -c /data/wzl/LightningSearch-RL/repo "env CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5x_hard50_env_transition_grpo_4gpu_16x15_variance_rankreward.sh"
```

Repo and environment:

- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- Remote repo state: not a git repository; narrow sync was used
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- GPUs: `0,1,2,3`
- Model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`

Config and artifacts:

- Config: `configs/experiments/phase5x_hard50_env_transition_grpo_4gpu_16x15_variance_rankreward.yaml`
- Input transitions: `/data/wzl/LightningSearch-RL/results/phase5x-env-transitions-variance-rankreward-from-phase5w/transitions.jsonl`
- Results: `/data/wzl/LightningSearch-RL/results/phase5x-hard50-env-transition-grpo-4gpu-16x15-variance-rankreward`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5x-hard50-env-transition-grpo-4gpu-16x15-variance-rankreward.log`
- Checkpoints: `/data/wzl/LightningSearch-RL/checkpoints/phase5x-hard50-env-transition-grpo-4gpu-16x15-variance-rankreward`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase5x-hard50-env-transition-grpo-4gpu-16x15-variance-rankreward/reward_dump.jsonl`

Key settings:

```text
train_rows=12
val_rows=4
rollout_n=4
total_training_steps=15
total_epochs=5
save_freq=15
train_batch_size=4
answer_token_f1_threshold=0.5
search_reward_top_k=8
```

## Verification

Dry-run before launch:

```text
train_rows=12
val_rows=4
parquet_written=True
trainer.total_training_steps=15
trainer.total_epochs=5
trainer.save_freq=15
```

Final status:

```text
completed=True
training_progress_100_seen=True
final_step=15
fatal_marker_count=0
shutdown_warning_count=11
started_at=2026-06-19T13:33:22+00:00
finished_at=2026-06-19T13:38:11+00:00
```

Checkpoint verification:

```text
/data/wzl/LightningSearch-RL/checkpoints/phase5x-hard50-env-transition-grpo-4gpu-16x15-variance-rankreward/global_step_15 exists
latest_checkpointed_iteration.txt = 15
```

The checkpoint contains FSDP actor model shards and optimizer shards:

```text
actor/model_world_size_4_rank_0.pt ... rank_3.pt
actor/optim_world_size_4_rank_0.pt ... rank_3.pt
actor/fsdp_config.json
global_step_15/data.pt
```

GPU state after completion returned to baseline on the selected cards:

```text
0, 3506 MiB, 32607 MiB
1, 3505 MiB, 32607 MiB
2, 3507 MiB, 32607 MiB
3, 3506 MiB, 32607 MiB
```

Shutdown warnings after completion:

```text
RuntimeError: DataLoader worker ... is killed by signal: Killed.
ERROR ... Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

These are treated as teardown warnings because the run reached 15/15, wrote `global_step_15`, and `fatal_marker_count=0`.

## Metrics

Initial validation metrics:

```text
val reward mean@1=0.6850000116974115
val score mean@1=0.685
val answer_reward mean@1=0.25
val search_reward mean@1=0.35
val evidence_rank_reward mean@1=0.35
```

Training reward and signal:

```text
reward_mean_all=0.472778
reward_last5=0.445167
nonzero_adv_count=11/15
nonzero_adv_steps=2,3,4,5,6,8,9,11,12,13,15
nonzero_grad_count=11/15
nonzero_grad_steps=2,3,4,5,6,8,9,11,12,13,15
```

Reward dump summary:

```text
row_count=244
stage_counts.search=142
stage_counts.answer=102

search.score_mean=0.523521
search.search_reward_mean=0.464085
search.invalid_action_count=0
search.variable_group_count=1
search.group_count=6
search.variable_group_rate=0.166667

answer.score_mean=0.410458
answer.answer_reward_mean=0.310458
answer.invalid_action_count=0
answer.variable_group_count=5
answer.group_count=5
answer.variable_group_rate=1.0
answer_reward_type_counts.exact=8
answer_reward_type_counts.token_f1=42
answer_reward_type_counts.containment=4
answer_reward_type_counts.none=48
```

Batch diagnostics:

```text
batch_count=3
train_rows=12
```

## Analysis

This diagnostic confirms the core hypothesis behind Phase 5X: selecting examples with known rollout reward variance greatly increases GRPO learning signal density. Phase 5W had nonzero advantage and gradient on 7 of 50 steps. This run had nonzero advantage and gradient on 11 of 15 steps.

The result should not be interpreted as benchmark improvement. The dataset is tiny and repeated for 5 epochs, so overfitting is expected. The useful conclusion is narrower: the reward plumbing and variance filtering can produce substantially denser GRPO updates.

The remaining problem is scale. The Phase 5W reward dump yielded only 8 variable source examples from 489 filtered rollouts. A practical next step is to generate or pre-sample a larger pool and keep examples whose sampled rollout groups have nonzero reward variance, rather than training on all examples uniformly.

## Next steps

1. Build a larger variance-filtered pool by pre-sampling multiple rollouts per source before GRPO.
2. Target at least 100 variable sources before a longer training run.
3. Keep the 15-step diagnostic as the quick smoke for reward changes.
4. Use a separate heldout eval before claiming answer-quality improvement.
