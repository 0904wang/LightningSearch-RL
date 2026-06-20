# Phase 5U GRPO Hard50 Evaluation

Date: 2026-06-18

## Goal

Evaluate whether the Phase 5U diverse rollout GRPO checkpoints changed real agent-loop behavior on the hard50 held-out setting.

The evaluation compares:

- SFT baseline: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- Phase 5U GRPO `global_step_100`
- Phase 5U GRPO `global_step_200`

## Launch

Session:

```text
lightningsearch-20260618-phase5u-hard50-eval
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5u-hard50-eval 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/runs/phase5u_grpo_hard50_eval.sh'"
```

Runtime:

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: narrow-sync-working-tree
local source branch/commit: master / 44493db04f0c8eb761c950a9d5322786c78c491e
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=7
started_at: 2026-06-18T13:56:08+00:00
finished_at: 2026-06-18T14:00:43+00:00
```

The first two launch attempts failed before creating a tmux session because Windows-to-SSH quoting broke the GPU memory parsing wrapper. The final launch used the direct tmux command above after GPU7 was confirmed free.

## Prelaunch Checks

```text
local eval-related tests: 20 passed
remote eval-related tests: 20 passed
dry-run offset=400 limit=5: gold_evidence_recall=0.8
dry-run all_gold_evidence_retrieved_rate=0.6
dry-run avg_candidate_doc_count=52.0
GPU 7 before launch: 18 MiB / 32607 MiB
```

## Inputs

```text
sft rows: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
phase5u checkpoint root: /data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer
offset: 400
limit: 100
top_k: 8
candidate_pool: gold-distractors
distractor_count: 50
max_new_tokens: 64
```

## Merge Artifacts

The launcher merged both FSDP actor checkpoints into Hugging Face checkpoints:

```text
global_step_100 merged: /data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer/hf_merged_global_step_100
global_step_200 merged: /data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer/hf_merged_global_step_200
merged size each: 7.6G
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5u-grpo-hard50-eval.log
results: /data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-eval
sft summary: /data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-eval/sft_baseline/summary.json
step100 summary: /data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-eval/grpo_global_step_100/summary.json
step200 summary: /data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-eval/grpo_global_step_200/summary.json
comparison: /data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-eval/comparison_summary.json
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
answer_exact_match_rate: 0.66
answer_containment_match_rate: 0.68
answer_token_f1: 0.719429
gold_evidence_recall: 0.855
all_gold_evidence_retrieved_rate: 0.71
assistant_observation_rate: 0.0
avg_observation_doc_count: 8.0
```

GRPO `global_step_100`:

```text
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.66
answer_containment_match_rate: 0.68
answer_token_f1: 0.719429
gold_evidence_recall: 0.855
all_gold_evidence_retrieved_rate: 0.71
assistant_observation_rate: 0.0
avg_observation_doc_count: 8.0
```

GRPO `global_step_200`:

```text
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.66
answer_containment_match_rate: 0.68
answer_token_f1: 0.719429
gold_evidence_recall: 0.855
all_gold_evidence_retrieved_rate: 0.71
assistant_observation_rate: 0.0
avg_observation_doc_count: 8.0
```

Delta versus SFT:

```text
global_step_100: all tracked metric deltas = 0.0
global_step_200: all tracked metric deltas = 0.0
```

Per-example diff:

```text
global_step_100 changed answer count: 0 / 100
global_step_100 changed search count: 0 / 100
global_step_200 changed answer count: 0 / 100
global_step_200 changed search count: 0 / 100
```

## Warnings

Non-fatal warnings:

```text
tokenizer regex warning for local Qwen checkpoint
torch_dtype deprecation warning
generation flags temperature/top_p/top_k ignored under deterministic decoding
```

No CUDA OOM or traceback was observed in the completed run.

## Analysis

The hard50 evaluation remains a useful hard setting: exact match is only 0.66, and evidence recall is 0.855 rather than saturated. However, Phase 5U did not change held-out behavior on this slice. Both GRPO checkpoints produced identical search actions and identical final answers to the SFT baseline.

This means the Phase 5U training reward and nonzero advantage improvements did not translate into observable policy changes under deterministic evaluation. The likely reason is that the model remains very close to the SFT policy under this short GRPO run, and the evaluation uses deterministic decoding, so small policy shifts do not cross the output decision boundary.

## Conclusion

Phase 5U is stable but not behaviorally improved on hard50 held-out. The best current narrative is:

- The pipeline can train, checkpoint, merge, and evaluate GRPO checkpoints end to end.
- Rollout diversity improved training-side advantage signal.
- Held-out hard50 eval shows no behavioral delta yet, so the next improvement should target stronger policy movement or more discriminative data/reward.

## Next Steps

1. Run a small stochastic evaluation for SFT vs Phase 5U step200 to check whether policy probability shifted even though greedy outputs are identical.
2. If stochastic outputs still match, increase data diversity or use harder query-quality rewards before longer GRPO.
3. If stochastic outputs differ, add pass@k / self-consistency style eval metrics to capture non-greedy policy changes.
