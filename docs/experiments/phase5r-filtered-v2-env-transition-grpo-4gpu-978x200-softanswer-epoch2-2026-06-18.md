# Phase 5R Filtered-v2 GRPO 978x200 Epoch2 Run

## Goal

Rerun the Phase 5R filtered-v2 4-GPU GRPO experiment for a clean 200-step
training result after the previous run stopped at 195 effective steps.

The previous run used `total_epochs=1`, which exhausted the 782-row training
split after `floor(782 / 4) = 195` full batches. This run keeps the same data
and target step count, but sets `total_epochs=2`.

Planned run:

```text
489 rollout examples
978 transitions
782 train rows
196 validation rows
target total_training_steps: 200
train_batch_size: 4
total_epochs: 2
```

## Code and Config Change

The launcher command builder was updated so `verl_smoke.py` honors the
configured `total_epochs` instead of always emitting `trainer.total_epochs=1`.

New config:

```text
configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer_epoch2.yaml
```

New remote launcher:

```text
/data/wzl/LightningSearch-RL/runs/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer_epoch2.sh
```

Verification before launch:

```text
local:  python -m pytest tests/test_verl_smoke.py -k "total_epochs or epoch2 or 200step" -> 3 passed
local:  python -m pytest -> 142 passed
remote: python -m pytest tests/test_verl_smoke.py -> 24 passed
remote: python -m pytest -> 142 passed
dry run: command included trainer.total_training_steps=200 and trainer.total_epochs=2
```

## Launch

Session:

```text
lightningsearch-20260618-phase5r-filtered-v2-grpo-978x200-epoch2
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5r-filtered-v2-grpo-978x200-epoch2 'CUDA_VISIBLE_DEVICES=0,1,2,7 bash /data/wzl/LightningSearch-RL/runs/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer_epoch2.sh' && echo LAUNCHED && tmux list-sessions"
```

Selected GPUs:

```text
0,1,2,7
```

Prelaunch GPU memory:

```text
0: 3506 MiB
1: 3505 MiB
2: 3507 MiB
7: 18 MiB
```

## Inputs

```text
repo: /data/wzl/LightningSearch-RL/repo
local source branch/commit: master / 44493db
remote sync method: narrow file sync
config: configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer_epoch2.yaml
transitions: /data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2/transitions.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2.log
results: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2
metrics: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/reward_dump_summary.json
```

Checkpoint status:

```text
global_step_100: present
global_step_200: present
latest checkpointed iteration: 200
```

## Status

```text
started_at: 2026-06-18T06:55:36+00:00
finished_at: 2026-06-18T07:16:10+00:00
planned total_training_steps: 200
effective final_step: 200
completed according to parser: true
training_progress_100_seen: true
fatal_marker_count: 0
shutdown_warning_count: 11
tmux session after run: not present
```

GPU memory was released after the run:

```text
0: 3506 MiB
1: 3505 MiB
2: 3507 MiB
7: 18 MiB
```

## Metrics

Initial validation:

```text
val reward mean@1: 1.0825704971746521
val answer reward mean@1: 0.49757045918367343
val answer exact match mean@1: 0.4897959183673469
val answer token F1 mean@1: 0.49757045918367343
val answer containment mean@1: 0.5
val search reward mean@1: 0.5
val format reward mean@1: 1.0
val num turns mean: 2.0
```

Train reward curve:

```text
reward points: 200
mean reward, all logged steps: 1.0816941183805466
mean reward, first 10 steps: 1.0842500686645509
mean reward, last 10 steps: 1.0850000858306885
mean reward, first 50 steps: 1.0833500671386718
mean reward, last 50 steps: 1.0855000710487366
min logged reward: 0.9525001049041748
max logged reward: 1.1000000238418579
low-reward steps below 1.05: 7
```

Low logged reward steps:

```text
step 53: 0.9525001049041748
step 54: 0.9778573513031006
step 99: 1.001666784286499
step 104: 0.9750000238418579
step 110: 1.0350000858306885
step 142: 1.0225000381469727
step 176: 1.0350000858306885
```

Reward dump summary:

```text
row_count: 996
overall score mean: 1.081867
invalid_action_count: 0
low_score_count: 0
answer stage rows: 495
answer score mean: 1.093877
answer reward mean: 0.993877
answer reward type counts: exact=485, containment=10, none=0
search stage rows: 501
search score mean: 1.07
search reward mean: 1.0
```

Batch diagnostics over prepared train data:

```text
train_rows: 782
train_batch_size: 4
full effective batches per epoch: 195
unused partial rows per epoch: 2
stage_counts: search=391, answer=391
low_reward_row_count: 0
```

## Log Excerpts

The run reached the target training step and wrote the final checkpoint:

```text
Training Progress: 100%|...| 200/200
global_step:200
local_global_step_folder: global_step_200
```

Shutdown-stage warnings appeared after the final step and checkpoint:

```text
DataLoader worker ... is killed by signal: Terminated.
resource_tracker: process died unexpectedly, relaunching. Some resources might leak.
EngineCore_DP0 died unexpectedly, shutting down client.
```

## Analysis

This is a clean 200-step rerun for the current Phase 5R filtered-v2 setup. The
parser reports `completed=true`, `final_step=200`, and `fatal_marker_count=0`.
Both expected checkpoints, `global_step_100` and `global_step_200`, are present.

The shutdown warnings are still present, but they occur after step 200 and after
the `global_step_200` checkpoint save. They should be tracked as teardown noise,
not as a failed training outcome for this run.

Compared with the previous effective-195 run:

```text
mean reward:       1.081694 vs 1.079079
last-50 reward:    1.085500 vs 1.079800
reward dump rows:  996 vs 976
low score count:   0 vs 2
answer reward:     0.993877 vs 0.989601
final checkpoint:  global_step_200 present vs absent
```

The total-epoch fix addressed the real blocker. The result supports using the
`global_step_200` checkpoint as the next candidate for generation inspection or
held-out evaluation.

## Next Step

Recommended next step: evaluate or inspect generations from
`global_step_200` before scaling the dataset again. The immediate checks should
focus on whether the higher train reward translates into better valid
`search -> answer` behavior, not only higher reward-model score.
