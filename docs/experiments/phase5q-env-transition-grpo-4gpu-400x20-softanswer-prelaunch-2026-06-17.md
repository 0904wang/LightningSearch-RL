# Phase 5Q-B Prelaunch: 400-Transition Soft-Answer GRPO

## Goal

Run a larger GRPO smoke after Phase 5Q-A data expansion:

- 200 environment rollouts
- 400 exported transitions
- 320 train transitions
- 80 validation transitions
- 20 GRPO steps
- reward dump enabled

This is the first scaled run beyond the 100-transition, 5-step Phase 5P retry.

## Data Inputs

Phase 5Q-A artifacts:

```text
rollouts: /data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200/env_rollouts.jsonl
transitions: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/transitions.jsonl
reward_records: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/reward_records.jsonl
```

Phase 5Q-A quality:

```text
env_rollouts.jsonl: 200 rows
transitions.jsonl: 400 rows
reward_records.jsonl: 200 rows
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.95
answer_containment_match_rate: 0.975
answer_token_f1: 0.968583
gold_evidence_recall: 0.9975
avg_total_reward: 1.335583
answer_reward_type_counts: exact=190, containment=5, none=5
```

## Remote Context

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
config: configs/experiments/phase5q_env_transition_grpo_4gpu_400x20_softanswer.yaml
```

## Validation

Local and remote tests after adding Phase 5Q config:

```text
local full suite: 132 passed
remote full suite: 132 passed
```

Training dry-run:

```text
parquet_written: true
source_type: transitions
train_rows: 320
val_rows: 80
train_jsonl: 320 rows
val_jsonl: 80 rows
reward_dump_path: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer/reward_dump.jsonl
```

Dry-run output directory:

```text
/data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer
```

## GPU Check

```text
0: 3506 MiB / 32607 MiB
1: 3505 MiB / 32607 MiB
2: 3507 MiB / 32607 MiB
3: 25985 MiB / 32607 MiB
4: 26101 MiB / 32607 MiB
5: 3493 MiB / 32607 MiB
6: 3505 MiB / 32607 MiB
7: 18 MiB / 32607 MiB
tmux: no active sessions
```

Selected GPUs:

```text
0,1,2,5
```

## Proposed Launch

Session:

```text
lightningsearch-20260617-phase5q-grpo-400x20
```

Command:

```bash
tmux new-session -d -s lightningsearch-20260617-phase5q-grpo-400x20 "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5q_env_transition_grpo_4gpu_400x20_softanswer.sh'"
```

Expected outputs:

```text
log: /data/wzl/LightningSearch-RL/logs/phase5q-env-transition-grpo-4gpu-400x20-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5q-env-transition-grpo-4gpu-400x20-softanswer
metrics: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer/reward_dump_summary.json
```

Metrics to inspect:

```text
critic/rewards/mean curve
answer_reward_type_counts
invalid_action_count
low_score_count
response/aborted_ratio
shutdown warnings vs fatal markers
```

## Success Criteria

1. `Training Progress: 100%` and `finished_at` appear.
2. `metrics_summary.completed` is true and `final_step` is 20.
3. `fatal_marker_count` is 0.
4. `reward_dump_summary.json` exists and has no invalid action spike.
5. Reward dump confirms containment examples remain partial-credit, not hard-zero.
