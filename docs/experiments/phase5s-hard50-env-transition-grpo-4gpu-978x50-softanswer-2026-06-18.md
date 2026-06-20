# Phase 5S Hard50 GRPO 978x50 Smoke

## Goal

Run a short 4-GPU GRPO smoke on the Phase 5S hard50 transition set. This
training set keeps the same 489-example / 978-transition size as Phase 5R, but
uses 50 distractors per example and contains many low-reward hard negatives.

## Launch

Session:

```text
lightningsearch-20260618-phase5s-hard50-grpo-50
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5s-hard50-grpo-50 'CUDA_VISIBLE_DEVICES=0,1,2,7 bash /data/wzl/LightningSearch-RL/runs/phase5s_hard50_env_transition_grpo_4gpu_978x50_softanswer.sh' && echo LAUNCHED && tmux list-sessions"
```

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
local source branch/commit: master / 44493db
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpus: 0,1,2,7
started_at: 2026-06-18T08:22:17+00:00
finished_at: 2026-06-18T08:28:42+00:00
```

Prelaunch checks:

```text
local tests/test_verl_smoke.py: 25 passed
remote phase5s config test: 1 passed
remote rollout dry-run tests: 4 passed
hard50 train dry-run: train_rows=782, val_rows=196, total_training_steps=50
GPU selection before launch: 0,1,2,7 all below 4000 MiB
```

## Inputs

```text
config: /data/wzl/LightningSearch-RL/repo/configs/experiments/phase5s_hard50_env_transition_grpo_4gpu_978x50_softanswer.yaml
transitions: /data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/transitions.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
train_rows: 782
val_rows: 196
train_batch_size: 4
total_training_steps: 50
save_freq: -1
```

The previous Phase 5R 200-step checkpoint directory was deleted before this run:

```text
/data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2
freed size: about 54G
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer
metrics: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer/reward_dump_summary.json
```

No checkpoint was expected because `save_freq=-1`.

## Status

```text
completed according to parser: true
final_step: 50
training_progress_100_seen: true
fatal_marker_count: 0
shutdown_warning_count: 11
tmux session after run: not present
GPU memory after run: released
```

Shutdown-stage warnings appeared after 50/50 progress:

```text
DataLoader worker killed by signal: Killed
resource_tracker process died unexpectedly
EngineCore_DP0 died unexpectedly, shutting down client
```

These occurred after `Training Progress: 100%|...| 50/50` and after
`finished_at`, so they are treated as teardown warnings for this run.

## Metrics

Initial validation:

```text
val reward mean@1: 0.9261079069835191
val answer reward mean@1: 0.3411078724489796
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
mean reward, all logged steps: 0.9067000627517701
mean reward, first 10 steps: 0.8310000598430634
mean reward, last 10 steps: 0.9327500641345978
min logged reward: 0.3500000238418579
max logged reward: 1.09250009059906
low-reward steps below 0.7: 11
```

Low-reward steps below 0.7:

```text
3: 0.5850000381469727
4: 0.5850000381469727
6: 0.5925000309944153
8: 0.5850000381469727
19: 0.5925000309944153
24: 0.5850000381469727
30: 0.5850000381469727
31: 0.3500000238418579
32: 0.5925000309944153
38: 0.5850000381469727
47: 0.5925000309944153
```

Latest train metrics at step 50:

```text
critic/rewards/mean: 0.8350000381469727
critic/rewards/min: 0.10000000149011612
critic/rewards/max: 1.100000023841858
response/aborted_ratio: 0.0
response_length/clip_ratio: 0.0
actor/pg_clipfrac: 0.0
actor/ppo_kl: 0.0
actor/grad_norm: 0.1708984375
```

Reward dump summary:

```text
row_count: 396
stage_counts: search=210, answer=186
overall score mean: 0.916306
overall low_score_count: 66
overall invalid_action_count: 0

search stage:
  row_count: 210
  score mean: 1.064762
  search_reward mean: 0.995238
  low_score_count: 1

answer stage:
  row_count: 186
  score mean: 0.748694
  answer_reward mean: 0.648694
  answer_reward_type_counts: exact=119, containment=2, none=65
  low_score_count: 65
```

## Analysis

This run did what the previous easy Phase 5R training did not: it exposed GRPO
to real low-reward hard-distractor transitions. The reward curve includes 11
low-reward batches below 0.7, and the reward dump contains 65 answer-stage
`none` cases plus one low search-stage case. That confirms the hard50 training
signal is present.

The run is technically stable: it reached 50/50 steps, response clipping stayed
at 0.0, aborted response ratio stayed at 0.0, invalid action count was 0, and
the parser found no fatal markers. Because `save_freq=-1`, there is no model
checkpoint to evaluate from this smoke.

The last-10-step reward mean, `0.9327500641345978`, is higher than the first-10
mean, `0.8310000598430634`, but this 50-step smoke is too short to claim a
reliable learning trend. Treat it as a successful hard-signal GRPO smoke and a
precondition for a longer checkpointed run.

## Next Step

Recommended next step: run a checkpointed hard50 experiment, such as 200 steps
with `save_freq=100`, after confirming disk budget. Then evaluate the resulting
checkpoint on hard50 and possibly global-pool held-out settings.
