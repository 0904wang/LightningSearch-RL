# Phase 5G Environment Rollout Smoke

Date: 2026-06-16

## Goal

Run the first real offline environment insertion smoke:

```text
question -> model <search> -> BM25 observation -> model <answer>
```

This validates the runtime boundary that Phase 5F only simulated with two-stage
offline prompts.

## Runtime

```text
local workspace: D:\resume\Agent RL
local branch/commit: master @ 44493db
remote repo: /data/wzl/LightningSearch-RL/repo
remote repo state: not-a-git-repo, narrow sync
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
python: 3.10.20
gpu: CUDA_VISIBLE_DEVICES=6
session: lightningsearch-20260616-phase5g-env-rollout-smoke
script: /data/wzl/LightningSearch-RL/runs/phase5g_env_rollout_smoke.sh
```

Remote log timestamps are UTC:

```text
started_at=2026-06-15T17:24:06+00:00
finished_at=2026-06-15T17:24:13+00:00
```

## Inputs

```text
sft source: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
offset: 9
limit: 1
top_k: 5
max_new_tokens: 64
sample id: syn-009025
```

The first attempted dry-run window was not suitable:

```text
offset=0 limit=3 top_k=2
gold_evidence_recall=0.0
all_gold_evidence_retrieved_rate=0.0
```

Diagnosis showed the global synthetic index has many repeated entity names and
question-only BM25 often retrieves evidence from other examples. For this smoke,
the selected sample was narrowed to a known recall-positive row:

```text
offset=9 limit=1 top_k=5
dry_run gold_evidence_recall=1.0
dry_run all_gold_evidence_retrieved_rate=1.0
```

## Launch

The stdin launch command was:

```bash
tmux new-session -d -s lightningsearch-20260616-phase5g-env-rollout-smoke "bash -lc 'CUDA_VISIBLE_DEVICES=6 bash /data/wzl/LightningSearch-RL/runs/phase5g_env_rollout_smoke.sh'"
```

As in the Phase 5F response dump, the wrapper command returned a spurious
`unknown command: list-sessions` after launch. The job itself completed and
produced the expected result files.

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5g-env-rollout-smoke.log
results: /data/wzl/LightningSearch-RL/results/phase5g-env-rollout-smoke
rollouts: /data/wzl/LightningSearch-RL/results/phase5g-env-rollout-smoke/env_rollouts.jsonl
summary: /data/wzl/LightningSearch-RL/results/phase5g-env-rollout-smoke/summary.json
remote record: /data/wzl/LightningSearch-RL/results/phase5g-env-rollout-smoke/EXPERIMENT_RECORD.md
```

## Raw Summary

```json
{
  "all_gold_evidence_retrieved_rate": 1.0,
  "answer_exact_match_rate": 1.0,
  "assistant_observation_rate": 0.0,
  "avg_observation_doc_count": 5.0,
  "dry_run": false,
  "example_count": 1,
  "gold_evidence_recall": 1.0,
  "limit": 1,
  "max_new_tokens": 64,
  "offset": 9,
  "top_k": 5,
  "valid_answer_action_rate": 1.0,
  "valid_search_action_rate": 1.0
}
```

## Generated Rollout

```text
id: syn-009025
question: Which organization presented the award won by the author of "The Quantum Horizon"?
search: <search>Which organization presented the award won by the author of "The Quantum Horizon"?</search>
answer: <answer>Global Science Foundation</answer>
gold answer: Global Science Foundation
```

Retrieved observation:

```text
[1] The Quantum Horizon: The Quantum Horizon is a science fiction novel by author Elena Voss.
[2] Knight Award: The Knight Award is presented by the Global Science Foundation.
[3] Stellar Award: The Stellar Award is presented annually by the Global Science Foundation.
[4] Elena Voss: Elena Voss won the Stellar Award for her novel The Quantum Horizon.
[5] Stellar Horizon Prize: The Stellar Horizon Prize is awarded by the Global Science Foundation.
```

Gold evidence:

```text
hotpot::syn-009025::Elena Voss::0
hotpot::syn-009025::Stellar Award::0
```

## Validation

Local tests before remote sync:

```text
python -m pytest -q
105 passed in 3.43s
```

Remote related tests:

```text
tests/test_environment_rollout.py tests/test_cli.py::test_inspect_env_rollout_cli_dry_run_writes_prompts
3 passed in 0.04s
```

Post-run checks:

```text
env_rollouts.jsonl lines: 1
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

Phase 5G confirms the real environment insertion loop works for a recall-positive
sample. The model did not generate `<observation>`, the environment inserted the
observation, and the answer generation produced the exact gold answer.

The important caveat is retrieval quality. With the full 500-example synthetic
index, BM25 over a single question query is heavily confused by repeated entity
names. A scan found:

```text
top2 full gold recall: 8/500
top5 full gold recall: 30/500
top10 full gold recall: 47/500
top20 full gold recall: 68/500
top50 full gold recall: 128/500
top100 full gold recall: 183/500
```

So the next bottleneck is not the runtime loop; it is retrieval environment
design and query supervision.

## Next Steps

- Add a corpus mode for controlled per-example candidate pools: gold documents
  plus distractors, instead of one global index with many duplicate names.
- Add retrieval metrics to environment rollout over larger slices before running
  GRPO against the environment.
- Consider two-step query training where the first hop query retrieves bridge
  evidence and the second hop query targets the answer evidence.
