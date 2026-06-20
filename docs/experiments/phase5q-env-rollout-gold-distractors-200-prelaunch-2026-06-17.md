# Phase 5Q-A Prelaunch: 200-Example Environment Rollout

## Goal

Scale from the Phase 5K/5P 50-example environment rollout to a 200-example
offline rollout, then export soft-answer transitions for the next GRPO run.

This is intentionally split from GRPO training. Phase 5Q-A creates and checks
data; Phase 5Q-B trains only if the rollout and transition summaries look good.

## Remote Context

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
python: Python 3.10 inside approved conda env
```

Source data:

```text
sft: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
sft rows: 500
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
```

Outputs:

```text
rollout log: /data/wzl/LightningSearch-RL/logs/phase5q-env-rollout-gold-distractors-200.log
rollout results: /data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200
transition results: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200
```

## Local Changes

Added:

```text
scripts/remote/phase5q_env_rollout_gold_distractors_200.sh
scripts/remote/phase5q_env_transition_grpo_4gpu_400x20_softanswer.sh
configs/experiments/phase5q_env_transition_grpo_4gpu_400x20_softanswer.yaml
tests/test_verl_smoke.py::test_phase5q_soft_answer_grpo_config_scales_transition_split
```

Remote checksums:

```text
b576d4c56cd39a2a2e1ee5ba4d7c0f8f4fa616e65ecae1ecb3bdfef0d9b48c42  /data/wzl/LightningSearch-RL/runs/phase5q_env_rollout_gold_distractors_200.sh
cfebf7d2b1c4f0806bb739b2783307c4995e86e71c3e30bee216f5b712e9bab5  /data/wzl/LightningSearch-RL/runs/phase5q_env_transition_grpo_4gpu_400x20_softanswer.sh
742c8745cf0686c57f4f63804743fc25fbe95209394dc963cd2055b86d2caf57  /data/wzl/LightningSearch-RL/repo/configs/experiments/phase5q_env_transition_grpo_4gpu_400x20_softanswer.yaml
```

## Validation

Local:

```text
phase5q config test: 1 passed
script syntax: bash -n passed for both scripts
full suite: 132 passed
```

Remote:

```text
phase5q config test: 1 passed
script syntax: bash -n passed for both scripts
full suite: 132 passed
```

Phase 5Q-A dry-run:

```text
dry_run: true
limit: 200
search_prompt_count: 200
answer_prompt_count: 200
candidate_pool: gold-distractors
distractor_count: 6
top_k: 8
avg_candidate_doc_count: 8.0
gold_evidence_recall: 0.9975
all_gold_evidence_retrieved_rate: 0.995
```

GPU status before launch:

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
disk: /data has 4.4T available
```

## Proposed Launch

Run only the data-generation/export step on 1 GPU:

```text
session: lightningsearch-20260617-phase5q-rollout-200
gpu: 7
```

Command:

```bash
tmux new-session -d -s lightningsearch-20260617-phase5q-rollout-200 "bash -lc 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/runs/phase5q_env_rollout_gold_distractors_200.sh'"
```

The script will run:

```text
inspect-env-rollout --offset 0 --limit 200 --top-k 8 --candidate-pool gold-distractors --distractor-count 6 --max-new-tokens 64
diagnose-rollout-answers
export-env-transitions
```

Expected artifacts:

```text
/data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200/env_rollouts.jsonl
/data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200/summary.json
/data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200/answer_diagnostics.json
/data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/transitions.jsonl
/data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/reward_records.jsonl
/data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/summary.json
```

## Success Criteria

1. `finished_at` appears in the log.
2. `env_rollouts.jsonl` has 200 rows.
3. `transitions.jsonl` has about 400 rows.
4. Valid search/action rates stay close to Phase 5K.
5. `gold_evidence_recall` remains high enough to justify Phase 5Q-B training.
