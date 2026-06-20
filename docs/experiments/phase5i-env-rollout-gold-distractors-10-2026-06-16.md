# Phase 5I Env Rollout Gold-Distractors 10

Date: 2026-06-16

## Goal

Run a real model environment rollout over the controlled retrieval pool from
Phase 5H:

```text
question -> model <search> -> BM25 over gold+distractors -> observation -> model <answer>
```

This tests whether the Phase 5D SFT checkpoint can use runtime-inserted
observations over a 10-example slice.

## Runtime

```text
local workspace: D:\resume\Agent RL
local branch/commit: master @ 44493db
remote repo: /data/wzl/LightningSearch-RL/repo
remote repo state: not-a-git-repo, narrow sync
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=6
session: lightningsearch-20260616-phase5i-env-rollout-10
script: /data/wzl/LightningSearch-RL/runs/phase5i_env_rollout_gold_distractors_10.sh
```

Remote log timestamps are UTC:

```text
started_at=2026-06-16T06:41:53+00:00
finished_at=2026-06-16T06:42:07+00:00
```

## Inputs

```text
sft source: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
offset: 0
limit: 10
top_k: 8
candidate_pool: gold-distractors
distractor_count: 6
max_new_tokens: 64
```

Dry-run check before launch:

```text
remote related tests: 4 passed
dry-run gold_evidence_recall: 1.0
dry-run all_gold_evidence_retrieved_rate: 1.0
```

## Launch

The first launch command returned zero but did not create a log. A tmux
diagnostic showed the Windows here-string could pass CRLF line endings into the
remote shell. The successful launch stripped carriage returns before `bash -s`:

```powershell
@'
set -e
tmux new-session -d -s lightningsearch-20260616-phase5i-env-rollout-10 "bash -lc 'CUDA_VISIBLE_DEVICES=6 bash /data/wzl/LightningSearch-RL/runs/phase5i_env_rollout_gold_distractors_10.sh'"
echo launched_phase5i
'@ | ssh user@ssh-22.e6.luyouxia.net -p 29509 "tr -d '\r' | bash -s"
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5i-env-rollout-gold-distractors-10.log
results: /data/wzl/LightningSearch-RL/results/phase5i-env-rollout-gold-distractors-10
rollouts: /data/wzl/LightningSearch-RL/results/phase5i-env-rollout-gold-distractors-10/env_rollouts.jsonl
summary: /data/wzl/LightningSearch-RL/results/phase5i-env-rollout-gold-distractors-10/summary.json
remote record: /data/wzl/LightningSearch-RL/results/phase5i-env-rollout-gold-distractors-10/EXPERIMENT_RECORD.md
```

## Raw Summary

```json
{
  "all_gold_evidence_retrieved_rate": 1.0,
  "answer_exact_match_rate": 0.7,
  "assistant_observation_rate": 0.0,
  "avg_observation_doc_count": 7.1,
  "candidate_pool": "gold-distractors",
  "distractor_count": 6,
  "dry_run": false,
  "example_count": 10,
  "gold_evidence_recall": 1.0,
  "limit": 10,
  "max_new_tokens": 64,
  "offset": 0,
  "top_k": 8,
  "valid_answer_action_rate": 1.0,
  "valid_search_action_rate": 1.0
}
```

## Per-Sample Result

Correct exact matches:

```text
syn-009000 -> Nobel Peace Prize
syn-009002 -> Riverstone
syn-009004 -> University of Cambridge
syn-009007 -> Northwood University
syn-009009 -> Edinburgh
syn-009022 -> Oakridge
syn-009025 -> Global Science Foundation
```

Non-exact outputs:

```text
syn-009012 predicted Global Health Research Institute, gold Barcelona
syn-009019 predicted Journal of Computational Science, gold 2020
syn-009020 predicted Golden Quill Award, gold Golden Quill
```

The first two non-exact rows look like dataset/schema quality issues: the
question asks for an institute or journal, while the stored gold answer is a city
or year. The third is likely a strict exact-match normalization issue.

## Validation

Post-run checks:

```text
env_rollouts.jsonl lines: 10
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

Phase 5I is a useful milestone:

- Tool format is stable: search and answer actions were valid for all examples.
- Runtime observation boundary is respected: the model did not emit
  `<observation>`.
- The controlled pool fixes retrieval coverage for this slice.
- The observed answer EM of 0.7 is partly limited by data quality and strict
  answer normalization, not only model behavior.

This supports moving to either a larger controlled-pool rollout or a data quality
cleanup pass before GRPO-on-environment.

## Next Steps

- Add an answer normalization metric that handles suffixes like `Award` and
  reports token F1 alongside exact match.
- Add a dataset consistency diagnostic for question/answer type mismatches.
- Run a `limit=50` controlled-pool model rollout after deciding whether to filter
  suspicious gold answers first.
