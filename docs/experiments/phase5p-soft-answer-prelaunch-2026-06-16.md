# Phase 5P: Soft Answer Reward Prelaunch

## Goal

Fix the strict exact-match answer reward issue found in Phase 5O, then prepare a
5-step GRPO repeat using the same `80/20` transition split and reward dump. The
soft answer rule gives partial credit for label-granularity mismatches while
keeping invalid or unrelated answers at zero.

## Reward Rule

Implemented in `answer_metrics.soft_answer_reward` and shared by offline
transition export and online `verl_reward.compute_score`:

```text
exact match: 1.0
containment match: max(0.5, token_f1)
non-containment high token-F1: token_f1 if token_f1 >= 0.75
otherwise: 0.0
```

Examples:

```text
Golden Quill Award vs Golden Quill -> 0.8
Vienna Conference Center vs Vienna -> 0.5
Global Health Research Institute vs Barcelona -> 0.0
```

## Validation

Local tests:

```text
130 passed in 3.36s
```

Remote targeted tests:

```text
14 passed in 0.36s
```

## Transition Re-export

Source:

```text
/data/wzl/LightningSearch-RL/results/phase5k-env-rollout-gold-distractors-50/env_rollouts.jsonl
```

Output:

```text
/data/wzl/LightningSearch-RL/results/phase5p-env-transitions-soft-answer-from-phase5k
```

Summary:

```text
example_count: 50
transition_count: 100
rollout_count: 50
avg_search_credit: 0.27
avg_answer_credit: 1.046
avg_total_reward: 1.316
answer_exact_match_rate: 0.92
answer_containment_match_rate: 0.96
answer_token_f1: 0.946
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
gold_evidence_recall: 1.0
```

Answer reward type counts:

```text
exact: 46
containment: 2
none: 2
```

Soft-credit examples:

```text
syn-009020: Golden Quill Award vs Golden Quill -> answer_reward=0.8, total=1.17
syn-009154: Vienna Conference Center vs Vienna -> answer_reward=0.5, total=0.87
```

## Phase 5P Config

```text
config: configs/experiments/phase5p_env_transition_grpo_4gpu_100x5_softanswer.yaml
script: scripts/remote/phase5p_env_transition_grpo_4gpu_100x5_softanswer.sh
experiment_name: phase5p-env-transition-grpo-4gpu-100x5-softanswer
transitions_path: /data/wzl/LightningSearch-RL/results/phase5p-env-transitions-soft-answer-from-phase5k/transitions.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
```

Training settings:

```yaml
train_samples: 80
val_samples: 20
train_batch_size: 4
ppo_mini_batch_size: 4
ppo_micro_batch_size_per_gpu: 1
max_prompt_length: 384
max_response_length: 64
rollout_max_model_len: 512
rollout_max_num_batched_tokens: 768
rollout_gpu_memory_utilization: 0.25
total_training_steps: 5
save_freq: -1
test_freq: -1
reward_dump_path: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer/reward_dump.jsonl
reward_dump_max_chars: 1024
```

Dry-run:

```text
parquet_written: true
train_rows: 80
val_rows: 20
reward_dump_env_in_launch_command: true
```

## Prelaunch GPU Check

```text
GPU 0: 3506 MiB / 32607 MiB
GPU 1: 3505 MiB / 32607 MiB
GPU 2: 3507 MiB / 32607 MiB
GPU 5: 3493 MiB / 32607 MiB
tmux sessions: none
```

## Proposed Launch

Session:

```text
lightningsearch-20260616-phase5p-softanswer-100x5
```

Command:

```bash
tmux new-session -d -s lightningsearch-20260616-phase5p-softanswer-100x5 "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5p_env_transition_grpo_4gpu_100x5_softanswer.sh'"
```

Expected outputs:

```text
log: /data/wzl/LightningSearch-RL/logs/phase5p-env-transition-grpo-4gpu-100x5-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5p-env-transition-grpo-4gpu-100x5-softanswer
metrics: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer/reward_dump_summary.json
```

Note: `save_freq=-1`, so no checkpoint is expected.

## Success Criteria

1. `Training Progress: 100%` and `finished_at` are present.
2. No fatal markers: `CalledProcessError`, `Error executing job`, CUDA OOM, KV cache error.
3. `reward_dump_summary.json` includes `answer_reward_type_counts`.
4. The low-score answer count should shrink or be easier to explain than Phase 5O.
