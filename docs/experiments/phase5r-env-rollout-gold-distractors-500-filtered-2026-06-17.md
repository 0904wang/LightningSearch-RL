# Phase 5R: 500-Example Environment Rollout and Filtered Transition Export

## Goal

Scale the Phase 5Q clean workflow from 200 rollout examples to 500 examples,
then export both raw and quality-manifest-filtered environment transitions.

## Launch

Session:

```text
lightningsearch-20260617-phase5r-rollout-500
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260617-phase5r-rollout-500 bash -lc 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/runs/phase5r_env_rollout_gold_distractors_500_filtered.sh'"
```

Runtime context:

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: 7
sft: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
quality_manifest: /data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5q_known_mismatches.json
```

Parameters:

```text
offset: 0
limit: 500
top_k: 8
candidate_pool: gold-distractors
distractor_count: 6
max_new_tokens: 64
```

Artifacts:

```text
rollout results: /data/wzl/LightningSearch-RL/results/phase5r-env-rollout-gold-distractors-500
raw transitions: /data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500
filtered transitions: /data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered
log: /data/wzl/LightningSearch-RL/logs/phase5r-env-rollout-gold-distractors-500-filtered.log
```

## Preflight

```text
local full suite: 137 passed
remote Phase 5R config test: 1 passed
remote rollout dry-run:
  search_prompt_count: 5
  answer_prompt_count: 5
  gold_evidence_recall: 1.0
  avg_candidate_doc_count: 8.0
SFT source rows: 500
GPU 7 before launch: 18 MiB used
```

## Result

Status: completed.

Raw rollout summary:

```text
example_count: 500
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.956
answer_containment_match_rate: 0.978
answer_token_f1: 0.973605
gold_evidence_recall: 0.999
all_gold_evidence_retrieved_rate: 0.998
assistant_observation_rate: 0.0
avg_observation_doc_count: 7.286
```

Raw transition summary:

```text
input_example_count: 500
example_count: 500
transition_count: 1000
avg_total_reward: 1.341071
avg_answer_credit: 1.071271
avg_search_credit: 0.2698
gold_evidence_recall: 0.999
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
```

Filtered transition summary with the Phase 5Q manifest:

```text
input_example_count: 500
example_count: 495
excluded_example_count: 5
excluded_example_ids:
  - syn-009012
  - syn-009019
  - syn-009432
  - syn-009456
  - syn-009536
transition_count: 990
avg_total_reward: 1.351082
avg_answer_credit: 1.081082
avg_search_credit: 0.27
answer_exact_match_rate: 0.965657
answer_containment_match_rate: 0.987879
answer_token_f1: 0.982429
gold_evidence_recall: 1.0
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
```

Line counts:

```text
env_rollouts.jsonl: 500
raw transitions.jsonl: 1000
raw reward_records.jsonl: 500
filtered transitions.jsonl: 990
filtered reward_records.jsonl: 495
filtered rollouts_for_grpo.jsonl: 495
```

## Diagnostics

Answer diagnostics flagged 7 suspicious rollout rows:

```text
syn-009012: question asks for institute, gold is Barcelona
syn-009019: question asks for journal, gold is 2020
syn-009178: question asks for university, gold is Copenhagen; prediction is University of Copenhagen; containment case
syn-009456: prediction Polar Archives, gold Miskatonic University
syn-009947: prediction Lumen Prize, gold Niels Bohr Medal
syn-010326: prediction Greenwood Historical Society, gold Greenfield University
syn-010401: prediction Vance Archive, gold Ashford University
```

The Phase 5Q manifest removed 5 known rows, but 4 suspicious rows remain in the
filtered set:

```text
syn-009178
syn-009947
syn-010326
syn-010401
```

Filtered reward-record low-total rows:

```text
answer_reward_type=none, total=0.37:
  - syn-009857: Global Science Foundation vs Global Science Forum
  - syn-009947: Lumen Prize vs Niels Bohr Medal
  - syn-010022: Oakville vs Riverside
  - syn-010102: 2018 vs 1998
  - syn-010326: Greenwood Historical Society vs Greenfield University
  - syn-010401: Vance Archive vs Ashford University

containment / partial cases below 1.0:
  - syn-009154: Vienna Conference Center vs Vienna, total=0.87
  - syn-009178: University of Copenhagen vs Copenhagen, total=0.87
  - syn-010078: The Nobel Assembly at Karolinska Institutet vs The Nobel Assembly, total=0.941429
```

## Analysis

The 500-rollout expansion succeeded technically: all 500 rollouts completed,
tool-call validity stayed at 1.0, evidence recall is effectively saturated, and
the known Phase 5Q manifest produced exactly the expected 990 filtered
transitions.

However, this is not yet the cleanest dataset for the next GRPO run. Scaling to
500 exposed new low-answer rows that were not present in the 200-example slice.
The most important issue is the six filtered rows with `answer_reward_type=none`
and total reward 0.37. These are not tool-use failures; they are synthetic QA
answer-quality problems or ambiguous gold labels.

The containment rows should remain for now because the soft-answer reward is
designed to support partial aliases. The `none` rows should be filtered before
training if the goal is to preserve the clean Phase 5Q filtered behavior where
batch diagnostics and reward dumps showed zero low-score rows.

## Recommendation

Before running the 50-step GRPO experiment, add a Phase 5R quality manifest for
the six `answer_reward_type=none` low-total rows and re-export the filtered
transitions. Expected cleaned size:

```text
495 examples - 6 additional rows = 489 examples
978 transitions
```

Then update or add a matching 978-transition GRPO config with an 80/20 split.
