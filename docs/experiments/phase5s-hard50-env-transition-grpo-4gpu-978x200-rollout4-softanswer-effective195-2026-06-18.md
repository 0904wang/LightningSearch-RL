# Phase 5S Hard50 GRPO 978x200 Rollout4 Run, Effective 195 Steps

## Goal

Run a longer hard50 GRPO experiment with `rollout_n=4`, so each prompt forms a
4-sample GRPO group instead of the previous `rollout_n=1` setting.

Planned run:

```text
489 rollout examples
978 transitions
782 train rows
196 validation rows
train_batch_size: 4
rollout_n: 4
target total_training_steps: 200
save_freq: 100
```

## Launch

Session:

```text
lightningsearch-20260618-phase5s-hard50-grpo-200-rollout4
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5s-hard50-grpo-200-rollout4 'CUDA_VISIBLE_DEVICES=0,1,2,7 bash /data/wzl/LightningSearch-RL/runs/phase5s_hard50_env_transition_grpo_4gpu_978x200_rollout4_softanswer.sh' && echo LAUNCHED && tmux list-sessions"
```

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
local source branch/commit: master / 44493db
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpus: 0,1,2,7
started_at: 2026-06-18T08:50:58+00:00
finished_at: 2026-06-18T09:25:46+00:00
```

Prelaunch checks:

```text
local tests/test_verl_smoke.py: 27 passed
remote phase5s hard50 config tests: 3 passed
remote dry-run: train_rows=782, val_rows=196, total_training_steps=200, rollout_n=4, save_freq=100
GPU selection before launch: 0,1,2,7 all below 4000 MiB
```

## Inputs

```text
config: /data/wzl/LightningSearch-RL/repo/configs/experiments/phase5s_hard50_env_transition_grpo_4gpu_978x200_rollout4_softanswer.yaml
transitions: /data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/transitions.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
train_rows: 782
val_rows: 196
train_batch_size: 4
rollout_n: 4
total_training_steps: 200
total_epochs: 1
save_freq: 100
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer
metrics: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer/reward_dump_summary.json
```

Checkpoint status:

```text
global_step_100: present
global_step_200: absent
latest_checkpointed_iteration.txt: 100
checkpoint dir size: about 24G
```

## Status

```text
planned total_training_steps: 200
effective final_step: 195
completed according to parser: false
training_progress_100_seen: false
fatal_marker_count: 0
shutdown_warning_count: 10
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

## Metrics

Initial validation:

```text
val reward mean@1: 0.9312099479138851
val answer reward mean@1: 0.34620991326530615
val answer exact match mean@1: 0.34183673469387754
val answer token F1 mean@1: 0.3701166224489796
val answer containment mean@1: 0.3469387755102041
val search reward mean@1: 0.5
val format reward mean@1: 1.0
val num turns mean: 2.0
```

Train reward curve:

```text
reward points: 195
mean reward, all logged steps: 0.902624
mean reward, first 10 steps: 0.835167
mean reward, last 10 steps: 0.970417
min logged reward: 0.342500 at step 73
max logged reward: 1.100000 at step 178
low-reward steps below 0.7: 30
```

Latest train metrics at step 195:

```text
critic/rewards/mean: 1.03000009059906
critic/rewards/min: 0.8999999761581421
critic/rewards/max: 1.100000023841858
critic/advantages/mean: -0.006122284568846226
critic/advantages/min: -1.4999843835830688
critic/advantages/max: 0.499995619058609
actor/grad_norm: 6.3125
response/aborted_ratio: 0.0
response_length/clip_ratio: 0.0
```

Reward dump summary:

```text
row_count: 3316
stage_counts: search=1678, answer=1638
overall score mean: 0.904313
overall low_score_count: 586
overall invalid_action_count: 0
overall format_reward mean: 0.994270

search stage:
  row_count: 1678
  score mean: 1.059511
  search_reward mean: 0.990465
  low_score_count: 16

answer stage:
  row_count: 1638
  score mean: 0.745325
  answer_reward mean: 0.645508
  answer_reward_type_counts: exact=1035, containment=32, token_f1=1, none=570
  answer none rate: 34.7985%
  low_score_count: 570
```

Rollout4 group signal:

```text
train steps with nonzero logged advantage: 14 / 195
```

## Log Excerpts

The run reached step 195:

```text
training/global_step:195
critic/rewards/mean:1.03000009059906
critic/advantages/min:-1.4999843835830688
critic/advantages/max:0.499995619058609
response_length/clip_ratio:0.0
response/aborted_ratio:0.0
```

The only checkpoint saved was step 100:

```text
local_global_step_folder: /data/wzl/LightningSearch-RL/checkpoints/phase5s-hard50-env-transition-grpo-4gpu-978x200-rollout4-softanswer/global_step_100
```

Shutdown-stage warnings appeared after the last logged step:

```text
resource_tracker: process died unexpectedly, relaunching. Some resources might leak.
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
finished_at=2026-06-18T09:25:46+00:00
```

## Root Cause Analysis

The planned 200-step run did not reach 200 steps because the epoch budget was
too small for the requested step count.

The training split has 782 rows and `train_batch_size=4`. With
`total_epochs=1`, verl logs 195 full training steps. This matches:

```text
floor(782 / 4) = 195
```

The final two rows do not form another full training batch, so the run ends
after step 195. This is the same root cause observed in the earlier Phase 5R
effective-195 run.

The vLLM `EngineCore_DP0 died unexpectedly` messages occurred during shutdown,
after the launcher returned from training and printed `finished_at`. The parser
found `fatal_marker_count=0`, so these warnings should not be treated as the
primary reason the run stopped at 195.

## Analysis

This run is a useful rollout4 smoke, but it should not be described as a
successful 200-step training run.

Compared with the earlier hard50 `rollout_n=1` 50-step smoke, rollout4 produced
real nonzero advantage on some steps. The last logged step has nonzero advantage
and nonzero actor grad norm, which confirms the group-comparison path is active.
However, only 14 of 195 logged steps had nonzero advantage, meaning many
4-sample groups still received identical rewards. The likely reason is that the
policy often emits the same short answer for all samples, or the discrete reward
collapses different responses into the same score.

The reward curve is mixed but not bad for hard50:

- all-step mean reward: `0.902624`;
- first-10 mean: `0.835167`;
- last-10 mean: `0.970417`;
- answer none rate: `34.7985%`;
- invalid action count: `0`;
- response clipping and aborted response ratio both stayed at `0.0`.

The main remaining limitation is not data size yet. The immediate issue is
training configuration: `total_epochs=1` prevents the configured 200-step
budget. The second issue is reward diversity within each GRPO group.

## Next Step

Recommended next run:

```text
keep the same hard50 rollout4 data
set total_epochs=2
keep total_training_steps=200
keep save_freq=100
optionally increase sampling diversity after the 200-step run if nonzero-advantage steps remain sparse
```

This should produce both `global_step_100` and `global_step_200` checkpoints if
the run finishes cleanly.
