# Phase 5K Env Rollout Gold-Distractors 50

Date: 2026-06-16

## Goal

Run a larger real model environment rollout over the controlled retrieval pool:

```text
question -> model <search> -> BM25 over gold+distractors -> observation -> model <answer>
```

This extends Phase 5I from 10 to 50 examples while reporting strict EM, relaxed
containment match, token F1, and suspicious-adjusted EM.

## Runtime

```text
local workspace: D:\resume\Agent RL
local branch/commit: master @ 44493db
remote repo: /data/wzl/LightningSearch-RL/repo
remote repo state: not-a-git-repo, narrow sync
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=6
session: lightningsearch-20260616-phase5k-env-rollout-50
script: /data/wzl/LightningSearch-RL/runs/phase5k_env_rollout_gold_distractors_50.sh
```

Remote log timestamps are UTC:

```text
started_at=2026-06-16T07:04:04+00:00
finished_at=2026-06-16T07:04:49+00:00
```

## Inputs

```text
sft source: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
offset: 0
limit: 50
top_k: 8
candidate_pool: gold-distractors
distractor_count: 6
max_new_tokens: 64
```

Dry-run check before launch:

```text
remote related tests: 3 passed
dry-run gold_evidence_recall: 1.0
dry-run all_gold_evidence_retrieved_rate: 1.0
```

## Launch

```powershell
@'
set -e
tmux new-session -d -s lightningsearch-20260616-phase5k-env-rollout-50 "bash -lc 'CUDA_VISIBLE_DEVICES=6 bash /data/wzl/LightningSearch-RL/runs/phase5k_env_rollout_gold_distractors_50.sh'"
echo launched_phase5k
'@ | ssh user@ssh-22.e6.luyouxia.net -p 29509 "tr -d '\r' | bash -s"
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5k-env-rollout-gold-distractors-50.log
results: /data/wzl/LightningSearch-RL/results/phase5k-env-rollout-gold-distractors-50
rollouts: /data/wzl/LightningSearch-RL/results/phase5k-env-rollout-gold-distractors-50/env_rollouts.jsonl
summary: /data/wzl/LightningSearch-RL/results/phase5k-env-rollout-gold-distractors-50/summary.json
answer diagnostics: /data/wzl/LightningSearch-RL/results/phase5k-env-rollout-gold-distractors-50/answer_diagnostics.json
remote record: /data/wzl/LightningSearch-RL/results/phase5k-env-rollout-gold-distractors-50/EXPERIMENT_RECORD.md
```

## Raw Summary

```json
{
  "all_gold_evidence_retrieved_rate": 1.0,
  "answer_containment_match_rate": 0.96,
  "answer_exact_match_rate": 0.92,
  "answer_token_f1": 0.946,
  "assistant_observation_rate": 0.0,
  "avg_observation_doc_count": 7.56,
  "candidate_pool": "gold-distractors",
  "distractor_count": 6,
  "dry_run": false,
  "example_count": 50,
  "gold_evidence_recall": 1.0,
  "limit": 50,
  "max_new_tokens": 64,
  "offset": 0,
  "top_k": 8,
  "valid_answer_action_rate": 1.0,
  "valid_search_action_rate": 1.0
}
```

Answer diagnostics:

```json
{
  "answer_containment_match_rate": 0.96,
  "answer_exact_match_rate": 0.92,
  "answer_token_f1": 0.946,
  "example_count": 50,
  "suspicious_adjusted_exact_match_rate": 0.958333,
  "suspicious_adjusted_example_count": 48,
  "suspicious_count": 2
}
```

## Non-Exact Rows

```text
syn-009012
question: Which research institute founded in 2021 is led by Dr. Elena Vasquez?
prediction: Global Health Research Institute
gold: Barcelona
type: suspicious gold/question type mismatch

syn-009019
question: Which journal was founded by the editor of the Journal of Applied Mathematics?
prediction: Journal of Computational Science
gold: 2020
type: suspicious gold/question type mismatch

syn-009020
prediction: Golden Quill Award
gold: Golden Quill
type: relaxed containment match, token_f1=0.8

syn-009154
prediction: Vienna Conference Center
gold: Vienna
type: relaxed containment match, token_f1=0.5
```

## Validation

Pre-launch local tests:

```text
python -m pytest --basetemp .pytest-tmp-phase5k-full -q
109 passed in 2.83s
```

Post-run checks:

```text
env_rollouts.jsonl lines: 50
tmux session: no active server after completion
GPU 6 after run: 3505 MiB / 32607 MiB
error scan: no Traceback/RuntimeError/ERROR/Exception matches
```

Warnings in the log:

```text
The tokenizer you are loading ... with an incorrect regex pattern ...
`torch_dtype` is deprecated! Use `dtype` instead!
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k'].
```

These warnings did not stop generation.

## Analysis

Phase 5K is the strongest environment-rollout result so far. The controlled
retrieval pool keeps evidence recall at 1.0 while still including distractors.
The trained model consistently follows the tool-use protocol:

- 50/50 valid search actions
- 50/50 valid answer actions
- no assistant-generated observation tags
- strict EM 0.92
- relaxed containment 0.96
- suspicious-adjusted EM 0.958333

The remaining strict-EM misses are mostly data quality or answer normalization
issues. This supports the project narrative that execution/training separation
and environment-controlled observation insertion are working; the next technical
risk is integrating this rollout style with GRPO rather than proving basic
tool-use behavior.

## Next Steps

- Add a `verl` data-prep path for controlled-pool environment rollouts, or
  export these 50 rollouts into transition/reward records for a small GRPO smoke.
- Keep reporting strict EM, containment, token F1, evidence recall, and
  suspicious-adjusted EM together.
- Consider filtering or repairing suspicious synthetic rows before large-scale
  training.
