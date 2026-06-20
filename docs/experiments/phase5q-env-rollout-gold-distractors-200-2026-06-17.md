# Phase 5Q-A: 200-Example Environment Rollout and Transition Export

## Goal

Scale the environment rollout dataset from Phase 5K's 50 examples to 200
examples, using the same offline gold+distractor retrieval setting, then export
soft-answer transitions for a larger GRPO run.

## Launch

Session:

```text
lightningsearch-20260617-phase5q-rollout-200
```

Command:

```bash
tmux new-session -d -s lightningsearch-20260617-phase5q-rollout-200 "bash -lc 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/runs/phase5q_env_rollout_gold_distractors_200.sh'"
```

Runtime context:

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: 7
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
```

Inputs:

```text
sft: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
sft rows: 500
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
candidate_pool: gold-distractors
distractor_count: 6
top_k: 8
offset: 0
limit: 200
max_new_tokens: 64
```

Outputs:

```text
log: /data/wzl/LightningSearch-RL/logs/phase5q-env-rollout-gold-distractors-200.log
rollouts: /data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200/env_rollouts.jsonl
rollout summary: /data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200/summary.json
answer diagnostics: /data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200/answer_diagnostics.json
transitions: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/transitions.jsonl
reward records: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/reward_records.jsonl
transition summary: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200/summary.json
```

## Status

Completed.

Raw completion markers:

```text
started_at=2026-06-16T16:36:34+00:00
finished_at=2026-06-16T16:39:10+00:00
fatal markers: none
tmux: no active sessions after completion
gpu 7 after completion: 18 MiB / 32607 MiB
```

Artifact counts:

```text
env_rollouts.jsonl: 200 rows
transitions.jsonl: 400 rows
reward_records.jsonl: 200 rows
```

## Rollout Metrics

```text
example_count: 200
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.95
answer_containment_match_rate: 0.975
answer_token_f1: 0.968583
gold_evidence_recall: 0.9975
all_gold_evidence_retrieved_rate: 0.995
assistant_observation_rate: 0.0
avg_observation_doc_count: 7.51
```

Answer diagnostics:

```text
suspicious_count: 4
suspicious_adjusted_example_count: 196
suspicious_adjusted_exact_match_rate: 0.969388
```

Suspicious rows:

```text
syn-009012: predicted Global Health Research Institute, gold Barcelona
syn-009019: predicted Journal of Computational Science, gold 2020
syn-009178: predicted University of Copenhagen, gold Copenhagen
syn-009456: predicted Polar Archives, gold Miskatonic University
```

## Transition Metrics

```text
example_count: 200
rollout_count: 200
transition_count: 400
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.95
answer_containment_match_rate: 0.975
answer_token_f1: 0.968583
gold_evidence_recall: 0.9975
avg_search_credit: 0.2695
avg_answer_credit: 1.066083
avg_total_reward: 1.335583
```

Reward type counts:

```text
exact: 190
containment: 5
none: 5
```

Lowest total reward rows:

```text
syn-009536: Lakeside University vs Middlesex University, total=0.27
syn-009012: Global Health Research Institute vs Barcelona, total=0.37
syn-009019: Journal of Computational Science vs 2020, total=0.37
syn-009432: European Institute of Technology vs Geneva, total=0.37
syn-009456: Polar Archives vs Miskatonic University, total=0.37
syn-009154: Vienna Conference Center vs Vienna, total=0.87
syn-009178: University of Copenhagen vs Copenhagen, total=0.87
syn-009245: Nobel Prize in 2023 vs Nobel Prize, total=1.036667
```

## Analysis

The rollout/export step is good enough for the next GRPO smoke. Tool format and
answer format are stable at 1.0 valid action rate, evidence recall stays very
high, and the soft-answer reward converts label-granularity mismatches into
partial credit instead of hard zero.

The remaining low-reward rows are mostly data-quality issues in the synthetic
question-answer pair, not tool-call failures. This matters for scaling: the
reward model will still see a small number of intentionally low-reward answer
transitions, but they are not caused by invalid search or observation parsing.

## Next Step

Proceed to Phase 5Q-B: use the 400 exported transitions for a 20-step 4-GPU
GRPO smoke with reward dump enabled:

```text
config: configs/experiments/phase5q_env_transition_grpo_4gpu_400x20_softanswer.yaml
results: /data/wzl/LightningSearch-RL/results/phase5q-env-transition-grpo-4gpu-400x20-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5q-env-transition-grpo-4gpu-400x20-softanswer
```
