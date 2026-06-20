# Phase 5R GRPO global_step_200 Hard50 Evaluation

## Goal

Increase evaluation difficulty after the easy held-out eval saturated. This run
keeps the same SFT-vs-GRPO A/B setup and held-out tail slice, but increases the
controlled retrieval candidate pool from 6 distractors to 50 distractors.

This tests whether the GRPO `global_step_200` checkpoint improves behavior when
retrieval ranking is less forgiving.

## Launch

Session:

```text
lightningsearch-20260618-phase5r-grpo-hard50-eval
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5r-grpo-hard50-eval 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/runs/phase5r_grpo_global200_hard50_eval.sh' && echo LAUNCHED && tmux list-sessions"
```

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
local source branch/commit: master / 44493db
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=7
started_at: 2026-06-18T07:46:25+00:00
finished_at: 2026-06-18T07:49:04+00:00
```

Prelaunch checks:

```text
remote tests: tests/test_environment_rollout.py + inspect-env-rollout CLI dry-run test -> 4 passed
dry-run offset=400 limit=5: gold_evidence_recall=0.8, all_gold_evidence_retrieved_rate=0.6
avg_candidate_doc_count: 52.0
GPU 7 before launch: 18 MiB / 32607 MiB
```

The dry-run confirmed that hard50 is materially harder than the previous
hard6/easy setting, where evidence recall was 1.0.

## Inputs

```text
sft rows: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
sft baseline model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
grpo merged model: /data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/hf_merged_global_step_200
offset: 400
limit: 100
top_k: 8
candidate_pool: gold-distractors
distractor_count: 50
max_new_tokens: 64
```

The GRPO HF checkpoint already existed from the prior held-out eval, so the
launcher skipped merging:

```text
merge skipped: existing HF checkpoint found
merged size: 7.6G
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5r-grpo-global200-hard50-eval.log
results: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-hard50-eval
sft rollouts: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-hard50-eval/sft_baseline/env_rollouts.jsonl
sft summary: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-hard50-eval/sft_baseline/summary.json
sft diagnostics: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-hard50-eval/sft_baseline/answer_diagnostics.json
grpo rollouts: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-hard50-eval/grpo_global_step_200/env_rollouts.jsonl
grpo summary: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-hard50-eval/grpo_global_step_200/summary.json
grpo diagnostics: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-hard50-eval/grpo_global_step_200/answer_diagnostics.json
comparison: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-hard50-eval/comparison_summary.json
diff: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-hard50-eval/diff_summary.json
```

## Metrics

SFT baseline:

```text
example_count: 100
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.66
answer_containment_match_rate: 0.68
answer_token_f1: 0.719429
gold_evidence_recall: 0.855
all_gold_evidence_retrieved_rate: 0.71
assistant_observation_rate: 0.0
avg_observation_doc_count: 8.0
suspicious_count: 17
suspicious_adjusted_exact_match_rate: 0.795181
```

GRPO `global_step_200`:

```text
example_count: 100
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.66
answer_containment_match_rate: 0.68
answer_token_f1: 0.719429
gold_evidence_recall: 0.855
all_gold_evidence_retrieved_rate: 0.71
assistant_observation_rate: 0.0
avg_observation_doc_count: 8.0
suspicious_count: 17
suspicious_adjusted_exact_match_rate: 0.795181
```

Delta, GRPO minus SFT:

```text
answer_exact_match_rate: +0.0
answer_containment_match_rate: +0.0
answer_token_f1: +0.0
valid_search_action_rate: +0.0
valid_answer_action_rate: +0.0
gold_evidence_recall: +0.0
all_gold_evidence_retrieved_rate: +0.0
assistant_observation_rate: +0.0
avg_observation_doc_count: +0.0
```

Diff summary:

```text
changed_count: 0
exact_improvement_count: 0
exact_regression_count: 0
f1_improvement_count: 0
f1_regression_count: 0
```

## Analysis

Hard50 successfully exposed a harder retrieval setting: evidence recall fell
from the previous easy setting's 1.0 to 0.855, and exact match fell from roughly
0.96-0.97 to 0.66. This is the first post-GRPO evaluation where the environment
is no longer saturated.

However, the SFT and GRPO checkpoints produced identical outputs on this
100-example slice. There were no changed answers, no changed search queries, no
exact improvements, and no regressions. That means the 200-step shaped GRPO run
preserved the policy but did not measurably change behavior under this harder
retrieval setup.

The failure mode is retrieval/ranking sensitivity rather than invalid tool use:
both models keep valid search and answer action rates at 1.0, and neither emits
assistant-side observation tags. The 17 suspicious rows are mostly cases where
the prediction matches a retrieved observation title while the gold answer asks
for a different entity/type.

## Conclusion

The current 200-step GRPO checkpoint is stable but behaviorally very close to
the SFT warm-start. For resume/project narrative, this is still useful: it shows
the evaluation can separate easy saturated settings from harder retrieval
settings, and it identifies the next algorithmic bottleneck.

## Next Step

The next useful improvement is not simply longer training on the same
transition set. Better options:

```text
1. Train with hard-distractor rollouts in the transition data, so search actions receive credit under harder retrieval.
2. Add query-quality/evidence-rank reward, not only evidence coverage and answer correctness.
3. Evaluate global-pool retrieval as a diagnostic, but expect lower scores.
4. Generate a new held-out validation set to check whether behavior changes outside this synthetic slice.
```

Recommended next experiment: build/export hard50 environment transitions from
the same 500 examples, then run a short GRPO comparison using those harder
search observations and evidence rewards.
