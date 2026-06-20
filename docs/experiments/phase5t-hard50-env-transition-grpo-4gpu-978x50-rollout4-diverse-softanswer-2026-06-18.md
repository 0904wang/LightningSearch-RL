# Phase 5T Hard50 GRPO Rollout4 Diverse 50-step Smoke

## Goal

Test whether higher rollout sampling diversity plus a lower token-F1 reward
threshold improves GRPO group signal on the hard50 transition split.

This run is a short smoke, not a checkpoint-producing run.

## Changes Versus Phase 5S Rollout4

```text
rollout_n: 4
rollout_temperature: 1.2
rollout_top_p: 0.95
rollout_top_k: 50
answer_token_f1_threshold: 0.5
total_training_steps: 50
total_epochs: 2
save_freq: -1
```

The `answer_token_f1_threshold` is passed as:

```text
LIGHTNINGSEARCH_ANSWER_TOKEN_F1_THRESHOLD=0.5
```

This gives partial reward to answer-stage outputs with token F1 >= 0.5 instead
of the default >= 0.75.

## Launch

Session:

```text
lightningsearch-20260618-phase5t-hard50-rollout4-diverse-50
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5t-hard50-rollout4-diverse-50 'CUDA_VISIBLE_DEVICES=0,1,2,7 bash /data/wzl/LightningSearch-RL/runs/phase5t_hard50_env_transition_grpo_4gpu_978x50_rollout4_diverse_softanswer.sh' && echo LAUNCHED && tmux list-sessions"
```

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
local source branch/commit: master / 44493db
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpus: 0,1,2,7
started_at: 2026-06-18T09:48:53+00:00
finished_at: 2026-06-18T09:59:03+00:00
```

Prelaunch checks:

```text
local pytest: 148 passed
remote pytest: 148 passed
remote dry-run: train_rows=782, val_rows=196, rollout_n=4, temperature=1.2, top_p=0.95, top_k=50, answer_token_f1_threshold=0.5
GPU selection before launch: 0,1,2,7 all below 4000 MiB
```

## Inputs

```text
config: /data/wzl/LightningSearch-RL/repo/configs/experiments/phase5t_hard50_env_transition_grpo_4gpu_978x50_rollout4_diverse_softanswer.yaml
transitions: /data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/transitions.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
train_rows: 782
val_rows: 196
train_batch_size: 4
rollout_n: 4
total_training_steps: 50
total_epochs: 2
save_freq: -1
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5t-hard50-env-transition-grpo-4gpu-978x50-rollout4-diverse-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5t-hard50-env-transition-grpo-4gpu-978x50-rollout4-diverse-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5t-hard50-env-transition-grpo-4gpu-978x50-rollout4-diverse-softanswer
metrics: /data/wzl/LightningSearch-RL/results/phase5t-hard50-env-transition-grpo-4gpu-978x50-rollout4-diverse-softanswer/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5t-hard50-env-transition-grpo-4gpu-978x50-rollout4-diverse-softanswer/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5t-hard50-env-transition-grpo-4gpu-978x50-rollout4-diverse-softanswer/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5t-hard50-env-transition-grpo-4gpu-978x50-rollout4-diverse-softanswer/reward_dump_summary.json
```

No checkpoint was expected because `save_freq=-1`.

## Status

```text
completed according to parser: true
training_progress_100_seen: true
final_step: 50
fatal_marker_count: 0
shutdown_warning_count: 9
tmux session after run: not present
GPU memory after run: released
```

Final GPU memory after the run:

```text
0: 3506 MiB
1: 3505 MiB
2: 3507 MiB
7: 18 MiB
```

Shutdown-stage warnings appeared after 50/50 progress:

```text
DataLoader worker killed by signal: Killed
resource_tracker process died unexpectedly
```

These occurred after `Training Progress: 100%|...| 50/50`, so they are treated
as teardown warnings for this smoke.

## Metrics

Initial validation:

```text
val reward mean@1: 0.946516075912787
val answer reward mean@1: 0.36151604081632654
val answer exact match mean@1: 0.336734693877551
val answer token F1 mean@1: 0.36501458163265305
val answer containment mean@1: 0.34183673469387754
val search reward mean@1: 0.5
val format reward mean@1: 1.0
val num turns mean: 2.0
```

Train reward curve:

```text
reward points: 50
mean reward, all logged steps: 0.936516
mean reward, first 10 steps: 0.878202
mean reward, last 10 steps: 0.967125
min logged reward: 0.350000 at step 31
max logged reward: 1.092500 at step 7
low-reward steps below 0.7: 7
```

Low-reward steps below 0.7:

```text
4: 0.6162500381469727
8: 0.6475000381469727
24: 0.6475000381469727
30: 0.6475000381469727
31: 0.3500000238418579
32: 0.5924999713897705
38: 0.5850000381469727
```

Rollout4 group signal:

```text
nonzero advantage steps: 6 / 50 = 12.0%
nonzero advantage steps: 4, 8, 9, 24, 30, 41
previous Phase 5S rollout4 effective-195 rate: 14 / 195 = 7.1795%
```

Latest train metrics at step 50:

```text
critic/rewards/mean: 0.8350000381469727
critic/rewards/min: 0.10000000149011612
critic/rewards/max: 1.100000023841858
critic/advantages/mean: 0.0
critic/advantages/min: 0.0
critic/advantages/max: 0.0
actor/grad_norm: 0.0
response/aborted_ratio: 0.0
response_length/clip_ratio: 0.0
```

Reward dump summary:

```text
row_count: 996
stage_counts: search=546, answer=450
overall score mean: 0.938483
overall low_score_count: 122
overall invalid_action_count: 0

search stage:
  row_count: 546
  score mean: 1.063956
  search_reward mean: 0.994505
  low_score_count: 3

answer stage:
  row_count: 450
  score mean: 0.786243
  answer_reward mean: 0.686243
  answer_reward_type_counts: exact=283, containment=1, token_f1=47, none=119
  answer none rate: 26.4444%
```

## Comparison

Compared with the previous Phase 5S hard50 50-step rollout1 smoke:

```text
Phase 5S rollout1 reward mean all: 0.906700
Phase 5T rollout4-diverse reward mean all: 0.936516

Phase 5S rollout1 answer none: 65 / 186 = 34.9462%
Phase 5T rollout4-diverse answer none: 119 / 450 = 26.4444%
```

Compared with the previous Phase 5S hard50 rollout4 effective-195 run:

```text
Phase 5S rollout4 nonzero advantage rate: 14 / 195 = 7.1795%
Phase 5T rollout4-diverse nonzero advantage rate: 6 / 50 = 12.0%

Phase 5S rollout4 answer none rate: 34.7985%
Phase 5T rollout4-diverse answer none rate: 26.4444%
```

## Analysis

The change moved the right metrics in the short smoke. Higher sampling diversity
plus the lower token-F1 threshold increased the fraction of steps with nonzero
advantage and lowered answer-stage `none` rate without increasing invalid
actions. The reward dump also shows 47 answer-stage `token_f1` rewards, which
confirms the shaped reward path is active and creating additional score levels.

This does not prove convergence yet because the run is only 50 steps and has no
checkpoint. It does show that the rollout/reward shaping adjustment is more
informative than the previous rollout4 setting. The next checkpointed run should
use the Phase 5T settings rather than the previous Phase 5S rollout4 config.

## Next Step

Recommended next run:

```text
phase5t hard50 rollout4 diverse, 200 steps
total_epochs: 2
save_freq: 100
same train/val split and model
```

Expected validation:

```text
global_step_100 and global_step_200 checkpoints are produced
nonzero advantage rate stays above the previous 7.18% rollout4 baseline
answer none rate does not regress toward 35%
invalid action count remains 0
```
