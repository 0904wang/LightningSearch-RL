# Phase 5S Hard50 Rollout and Transition Export

## Goal

Build a harder GRPO training set after the hard50 evaluation showed that the
previous 200-step GRPO checkpoint behaved identically to the SFT baseline. This
export uses the same 500 examples, but increases the controlled retrieval pool
to 50 distractors per example so search/evidence quality is no longer saturated.

## Launch

Session:

```text
lightningsearch-20260618-phase5s-hard50-rollout
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5s-hard50-rollout 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/runs/phase5s_env_rollout_gold_distractors_500_hard50_filtered.sh' && echo LAUNCHED && tmux list-sessions"
```

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
local source branch/commit: master / 44493db
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=7
started_at: 2026-06-18T08:12:40+00:00
finished_at: 2026-06-18T08:19:00+00:00
```

Prelaunch checks:

```text
local tests/test_verl_smoke.py: 25 passed
remote phase5s config test: 1 passed
remote rollout dry-run tests: 4 passed
dry-run offset=0 limit=5: gold_evidence_recall=0.5, avg_candidate_doc_count=52.0
GPU 7 before launch: 18 MiB / 32607 MiB
```

## Inputs

```text
sft rows: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
quality_manifest: /data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5r_500_known_mismatches.json
offset: 0
limit: 500
top_k: 8
candidate_pool: gold-distractors
distractor_count: 50
max_new_tokens: 64
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5s-env-rollout-gold-distractors-500-hard50-filtered.log
rollouts: /data/wzl/LightningSearch-RL/results/phase5s-env-rollout-gold-distractors-500-hard50
raw transitions: /data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50
filtered transitions: /data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1
```

Line counts:

```text
filtered transitions.jsonl: 978
filtered reward_records.jsonl: 489
filtered rollouts_for_grpo.jsonl: 489
```

## Metrics

Raw rollout / transition summary:

```text
example_count: 500
transition_count: 1000
valid_search_action_rate: 1.0
valid_answer_action_rate: 0.99
answer_exact_match_rate: 0.62
answer_containment_match_rate: 0.64
answer_token_f1: 0.687671
gold_evidence_recall: 0.818
avg_search_credit: 0.2336
avg_answer_credit: 0.73319
avg_total_reward: 0.96679
```

Filtered-v1 transition summary:

```text
input_example_count: 500
example_count: 489
excluded_example_count: 11
transition_count: 978
valid_search_action_rate: 1.0
valid_answer_action_rate: 0.989775
answer_exact_match_rate: 0.631902
answer_containment_match_rate: 0.650307
answer_token_f1: 0.69946
gold_evidence_recall: 0.819018
avg_search_credit: 0.233804
avg_answer_credit: 0.743753
avg_total_reward: 0.977557
```

Excluded rows:

```text
qa_type_mismatch: 5
answer_none_low_reward: 6
excluded ids:
  syn-009012
  syn-009019
  syn-009432
  syn-009456
  syn-009536
  syn-009857
  syn-009947
  syn-010022
  syn-010102
  syn-010326
  syn-010401
```

Filtered reward distribution:

```text
answer_reward_type_counts:
  exact: 309
  containment: 9
  none: 171
low_total_count: 171
```

## Analysis

This export gives the harder training signal that the previous easy Phase 5R
set lacked. It preserves the same 489-example / 978-transition size after known
quality filtering, but it contains 171 low-total hard negatives. These rows are
not automatically filtered because they are mostly retrieval/ranking failures,
which are exactly the behaviors the next GRPO run should learn from.

Compared with the easy filtered-v2 data, answer exact match dropped from the
high 0.96 range to 0.631902, and evidence recall dropped to 0.819018. Tool-call
format is still stable: search validity is 1.0 and answer validity is about
0.99.

## Next Step

Run the prepared Phase 5S hard50 50-step GRPO smoke:

```text
config: /data/wzl/LightningSearch-RL/repo/configs/experiments/phase5s_hard50_env_transition_grpo_4gpu_978x50_softanswer.yaml
results: /data/wzl/LightningSearch-RL/results/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5s-hard50-env-transition-grpo-4gpu-978x50-softanswer
```

Because this is a 50-step smoke, `save_freq=-1` and no checkpoint is expected.
