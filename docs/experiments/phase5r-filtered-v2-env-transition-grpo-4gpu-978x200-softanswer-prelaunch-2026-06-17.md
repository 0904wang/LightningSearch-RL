# Phase 5R Filtered-v2 978x200 GRPO Prelaunch

## 50-Step Result Analysis

The prior run completed a 4-GPU, 50-step GRPO smoke on the cleaned 978-transition
filtered-v2 set.

Status:

```text
completed: true
final_step: 50
fatal_marker_count: 0
training_progress_100_seen: true
started_at: 2026-06-17T10:11:53+00:00
finished_at: 2026-06-17T10:18:10+00:00
```

Training stability:

```text
avg reward first 10 steps: 1.08425
avg reward last 10 steps: 1.08275
min reward mean: 1.07
max reward mean: 1.1
avg step time: 5.72938s
avg response length: 17.18
max response clip ratio: 0.0
max aborted ratio: 0.0
max grad norm: 8.0625
```

Runtime reward dump:

```text
rows: 396
overall score mean: 1.080439
invalid actions: 0
low score count: 1
answer exact runtime count: 184
answer containment runtime count: 2
answer none runtime count: 1
answer stage score mean: 1.092106
search stage score mean: 1.07
```

Batch diagnostics:

```text
train rows: 782
stage counts: search=391, answer=391
batch low reward rows: 0
precomputed step reward mean: 0.681307
precomputed total reward mean: 1.362614
```

Comparison to the previous Phase 5Q filtered 390x20 run:

```text
Phase 5Q filtered 390x20:
  final_step: 20
  latest reward mean: 1.09250009059906
  reward dump score mean: 1.082917
  low score count: 0
  invalid action count: 0

Phase 5R filtered-v2 978x50:
  final_step: 50
  latest reward mean: 1.0850000381469727
  reward dump score mean: 1.080439
  low score count: 1
  invalid action count: 0
```

Conclusion: the larger 978-transition run stayed stable. The reward did not
collapse, response clipping stayed at zero, and no invalid actions appeared.
The single runtime low-score answer is a model rollout miss, not a source-data
quality row. A 200-step run is the next reasonable scale step.

## New 200-Step Config

Files added or updated:

```text
configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer.yaml
scripts/remote/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer.sh
tests/test_verl_smoke.py
```

Key settings:

```text
experiment_name: phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer
transitions: /data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2/transitions.jsonl
train_samples: 782
val_samples: 196
total_training_steps: 200
save_freq: 100
test_freq: -1
train_batch_size: 4
ppo_mini_batch_size: 4
n_gpus_per_node: 4
reward_dump: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/reward_dump.jsonl
```

## Verification

Local:

```text
test_phase5r_filtered_v2_soft_answer_grpo_200step_config_saves_checkpoints: passed
phase5r related tests: 4 passed
full local pytest: 140 passed
```

Remote:

```text
new 200-step config test: passed
full remote pytest: 140 passed
bash syntax check: passed
dry-run: succeeded
parquet_written: true
train rows: 782
val rows: 196
```

Remote repo status:

```text
not-a-git-repo
```

This run uses the approved narrow-sync workflow. Hashes were checked for the
new config, the new launcher, the `/runs` launcher copy, and `tests/test_verl_smoke.py`.

## Launch Report

Selected repo:

```text
/data/wzl/LightningSearch-RL/repo
```

Branch / commit:

```text
not available; remote repo is a narrow-synced working tree, not a git checkout
```

Conda env:

```text
/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
```

Selected GPUs:

```text
CUDA_VISIBLE_DEVICES=0,1,2,7
```

Prelaunch GPU memory:

```text
0: 3506 MiB
1: 3505 MiB
2: 3507 MiB
7: 18 MiB
```

tmux session:

```text
lightningsearch-20260617-phase5r-filtered-v2-grpo-978x200
```

Exact launch command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260617-phase5r-filtered-v2-grpo-978x200 bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,7 bash /data/wzl/LightningSearch-RL/runs/phase5r_filtered_v2_env_transition_grpo_4gpu_978x200_softanswer.sh'"
```

Expected outputs:

```text
log: /data/wzl/LightningSearch-RL/logs/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer
metrics: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer/reward_dump_summary.json
```

Expected metrics to inspect:

```text
completion / final_step / fatal_marker_count
critic/rewards/mean curve
response_length/clip_ratio
response/aborted_ratio
invalid_action_count
low_score_count
answer_reward_type_counts
checkpoint directories at save_freq=100
```
