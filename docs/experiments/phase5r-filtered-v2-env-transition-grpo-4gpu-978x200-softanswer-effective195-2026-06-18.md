# Phase 5R Filtered-v2 GRPO 978x200 Run, Effective 195 Steps

## Goal

Run the longer 4-GPU GRPO experiment on the cleaned Phase 5R filtered-v2
transition set.

Planned run:

```text
489 rollout examples
978 transitions
782 train rows
196 validation rows
target total_training_steps: 200
train_batch_size: 4
total_epochs: 1
```

## Launch

Session:

```text
lightningsearch-20260618-phase5r-filtered-v2-grpo-978x200
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5r-filtered-v2-grpo-978x200 bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,7 bash /data/wzl/LightningSearch-RL/runs/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer.sh'"
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
remote git metadata: unavailable in /data/wzl/LightningSearch-RL/repo at post-run check
config: configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer.yaml
transitions: /data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2/transitions.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer
metrics: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/reward_dump_summary.json
```

Checkpoint status:

```text
global_step_100: present
global_step_200: absent
latest_checkpointed_iteration.txt: 100
```

## Status

```text
started_at: 2026-06-18T05:15:14+00:00
finished_at: 2026-06-18T05:35:07+00:00
planned total_training_steps: 200
effective final_step: 195
completed according to parser: false
fatal_marker_count: 0
shutdown_warning_count: 10
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
reward points: 195
mean reward, all logged steps: 1.0790794354218702
mean reward, first 10 steps: 1.0842500686645509
mean reward, last 10 steps: 1.0600000858306884
mean reward, first 50 steps: 1.0833500671386718
mean reward, last 50 steps: 1.0798000669479371
min logged reward: 0.8350000381469727
max logged reward: 1.100000023841858
low-reward steps below 1.05: 10
```

Low logged reward steps:

```text
step 53: 0.9525001049041748
step 54: 0.9778573513031006
step 99: 1.001666784286499
step 104: 0.8500000238418579
step 110: 1.0350000858306885
step 120: 1.001666784286499
step 142: 1.0225000381469727
step 157: 1.0350000858306885
step 158: 1.027500033378601
step 193: 0.8350000381469727
```

Reward dump summary:

```text
row_count: 976
overall score mean: 1.07978
invalid_action_count: 0
low_score_count: 2
answer stage rows: 487
answer score mean: 1.089601
answer reward mean: 0.989601
answer reward type counts: exact=474, containment=11, none=2
search stage rows: 489
search score mean: 1.07
search reward mean: 1.0
```

Batch diagnostics over prepared train data:

```text
train_rows: 782
train_batch_size: 4
full effective batches: 195
unused partial rows: 2
stage_counts: search=391, answer=391
low_reward_row_count: 0
reward_model_reward mean: 0.681307
precomputed_total_reward mean: 1.362614
```

Low-score runtime examples:

```text
source_id: syn-009154
reward_stage: answer
score: 0.1
answer_reward: 0.0
parsed_action.valid: true
solution_preview: <answer>Vienne Conference Center</answer>

source_id: syn-009267
reward_stage: answer
score: 0.1
answer_reward: 0.0
parsed_action.valid: true
solution_preview: <answer>Cornith</answer>
```

## Log Excerpts

The run reached step 195:

```text
training/global_step:195
critic/rewards/mean:1.09250009059906
response_length/clip_ratio:0.0
response/aborted_ratio:0.0
```

Shutdown-stage warnings appeared immediately after the last logged step:

```text
resource_tracker: process died unexpectedly, relaunching. Some resources might leak.
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
finished_at=2026-06-18T05:35:07+00:00
```

## Root Cause Analysis

The planned 200-step run did not reach 200 steps because the configured epoch
budget was too small for the requested step count.

The training split has 782 rows and `train_batch_size=4`. With `total_epochs=1`,
that gives `floor(782 / 4) = 195` full training batches. The final 2 rows form a
partial batch and were not enough to produce another training step. Therefore
the observed `final_step=195` is consistent with one full epoch ending before
`trainer.total_training_steps=200`.

The vLLM `EngineCore_DP0 died unexpectedly` messages occurred after step 195 and
after the `python -m lightningsearch_rl.cli train` command returned to the
launcher, because the launcher printed `finished_at` and continued to parse
logs under `set -e`. Treat these as shutdown warnings for this run, not as the
primary reason only 195 steps were logged.

## Analysis

The usable part of the run is stable and broadly consistent with the previous
50-step scale smoke:

- response clipping stayed at `0.0`;
- aborted response ratio stayed at `0.0`;
- invalid runtime tool/action count was `0`;
- search-stage reward remained saturated at `1.07`;
- answer-stage reward stayed high, with only 2 `answer_reward=0` examples in
  the reward dump;
- the last-50-step mean reward, `1.0798000669479371`, was close to the all-step
  mean, `1.0790794354218702`.

The result should not be described as a successful 200-step run. It is better
described as a 195-step one-epoch run on the 782-row train split. It still gives
useful evidence that the filtered-v2 data and reward function remain stable at
roughly 4x the previous 50-step smoke length.

## Next Step

For a clean 200+ step experiment, use one of these changes before relaunching:

```text
Option A: keep the same data, set total_epochs=2, keep total_training_steps=200.
Option B: expand the train split so train_rows >= 800 and keep total_epochs=1.
Option C: set total_training_steps=195 and treat this as the one-epoch budget.
```

The most direct next run is Option A. It should also keep `save_freq=100`, which
will produce `global_step_100` and `global_step_200` checkpoints if the run
finishes cleanly.
