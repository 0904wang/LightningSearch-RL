# Phase 5U GRPO Hard50 Stochastic Evaluation

Date: 2026-06-18

## Goal

Check whether Phase 5U `global_step_200` changed the model distribution even though deterministic hard50 evaluation produced identical outputs to the SFT baseline.

This small stochastic evaluation compares:

- SFT baseline
- Phase 5U GRPO `global_step_200`

## Code Change

Added eval-only sampling support to `inspect-env-rollout`:

```text
--do-sample
--temperature
--top-p
--sample-top-k
--seed
```

Local and remote verification after the change:

```text
local tests: 151 passed
remote tests: 151 passed
remote stochastic dry-run: succeeded
```

## Launch

Session:

```text
lightningsearch-20260618-phase5u-hard50-stochastic-eval
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5u-hard50-stochastic-eval 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/runs/phase5u_grpo_hard50_stochastic_eval.sh'"
```

Runtime:

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: narrow-sync-working-tree
local source branch/commit: master / 44493db04f0c8eb761c950a9d5322786c78c491e
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=7
started_at: 2026-06-18T14:23:42+00:00
finished_at: 2026-06-18T14:24:25+00:00
```

## Inputs

```text
sft rows: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
sft model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
grpo model: /data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer/hf_merged_global_step_200
offset: 400
limit: 20
top_k: 8
candidate_pool: gold-distractors
distractor_count: 50
max_new_tokens: 64
```

Sampling:

```text
do_sample: true
temperature: 0.7
top_p: 0.9
sample_top_k: 40
seed: 20260618
```

Dry-run check:

```text
gold_evidence_recall=0.8
all_gold_evidence_retrieved_rate=0.6
avg_candidate_doc_count=52.0
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5u-grpo-hard50-stochastic-eval.log
results: /data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-stochastic-eval
sft summary: /data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-stochastic-eval/sft_baseline_seed20260618/summary.json
grpo summary: /data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-stochastic-eval/grpo_global_step_200_seed20260618/summary.json
comparison: /data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-stochastic-eval/comparison_summary.json
```

GPU7 was released after completion:

```text
7, 18 MiB, 32607 MiB
```

## Metrics

SFT baseline:

```text
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.7
answer_containment_match_rate: 0.7
answer_token_f1: 0.725
gold_evidence_recall: 0.875
all_gold_evidence_retrieved_rate: 0.75
assistant_observation_rate: 0.0
avg_observation_doc_count: 8.0
```

Phase 5U GRPO `global_step_200`:

```text
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.7
answer_containment_match_rate: 0.7
answer_token_f1: 0.725
gold_evidence_recall: 0.875
all_gold_evidence_retrieved_rate: 0.75
assistant_observation_rate: 0.0
avg_observation_doc_count: 8.0
```

Delta, GRPO minus SFT:

```text
all tracked metric deltas: 0.0
```

Per-example diff:

```text
changed_answer_count: 0
changed_search_count: 0
exact_improvement_count: 0
exact_regression_count: 0
f1_improvement_count: 0
f1_regression_count: 0
```

## Warnings

No OOM or traceback was observed in the completed run.

Expected warnings:

```text
tokenizer regex warning for local Qwen checkpoint
torch_dtype deprecation warning
```

## Analysis

This stochastic eval did not reveal hidden policy movement. With identical sampling settings and seed, Phase 5U `global_step_200` produced the same search actions and answers as the SFT baseline on the 20-example hard50 slice.

Together with the deterministic hard50 evaluation, this suggests the Phase 5U update is too small to affect actual decoded behavior. The training-side reward/advantage signal improved, but it did not move the policy enough to cross generation boundaries under either greedy decoding or this small stochastic probe.

## Conclusion

Further lengthening the same training recipe is unlikely to be the highest-leverage next step. The next useful change should target the data/reward mechanism rather than simply adding more steps.

Recommended next options:

```text
1. Increase transition diversity with fresh hard50/hard100 rollouts from more examples.
2. Add query-rank or evidence-rank reward so search action quality receives more direct signal.
3. Try a stronger KL/learning-rate/update setting only after confirming reward variance remains healthy.
```
