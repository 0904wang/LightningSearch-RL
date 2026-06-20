# Phase 5H Candidate Pool Dry Run

Date: 2026-06-16

## Goal

Evaluate a controlled retrieval environment for Phase 5G-style env rollouts.
The previous global-index BM25 setting had poor evidence recall because the
synthetic corpus contains many repeated entity names. Phase 5H tests a
per-example candidate pool made from gold evidence plus deterministic
distractors.

## Runtime

```text
local workspace: D:\resume\Agent RL
local branch/commit: master @ 44493db
remote repo: /data/wzl/LightningSearch-RL/repo
remote repo state: not-a-git-repo, narrow sync
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
script: /data/wzl/LightningSearch-RL/runs/phase5h_candidate_pool_dry_run.sh
```

Remote log timestamps are UTC:

```text
started_at=2026-06-16T03:08:39+00:00
finished_at=2026-06-16T03:08:40+00:00
```

## Inputs

```text
sft source: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
model path argument: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
offset: 0
limit: 50
top_k: 8
candidate_pool: gold-distractors
distractor_count: 6
dry_run: true
```

This dry-run did not load the model. It used the gold SFT search action to
construct answer-stage prompts with BM25 observations from the controlled pool.

## Launch

```bash
bash /data/wzl/LightningSearch-RL/runs/phase5h_candidate_pool_dry_run.sh
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5h-candidate-pool-dry-run.log
results: /data/wzl/LightningSearch-RL/results/phase5h-candidate-pool-dry-run
search prompts: /data/wzl/LightningSearch-RL/results/phase5h-candidate-pool-dry-run/search_prompts.jsonl
answer prompts: /data/wzl/LightningSearch-RL/results/phase5h-candidate-pool-dry-run/answer_prompts.jsonl
summary: /data/wzl/LightningSearch-RL/results/phase5h-candidate-pool-dry-run/dry_run_summary.json
remote record: /data/wzl/LightningSearch-RL/results/phase5h-candidate-pool-dry-run/EXPERIMENT_RECORD.md
```

## Raw Summary

```json
{
  "all_gold_evidence_retrieved_rate": 1.0,
  "answer_prompt_count": 50,
  "avg_candidate_doc_count": 8.0,
  "candidate_pool": "gold-distractors",
  "distractor_count": 6,
  "dry_run": true,
  "gold_evidence_recall": 1.0,
  "limit": 50,
  "max_new_tokens": 64,
  "offset": 0,
  "search_prompt_count": 50,
  "top_k": 8
}
```

## Sample Prompt Checks

For `syn-009000`, the controlled candidate pool included:

```text
hotpot::syn-009000::Dr. Elena Voss::0
hotpot::syn-009000::Global Health Initiative::0
six distractor passages
```

The retrieved observation included both gold evidence passages:

```text
[1] Dr. Elena Voss: Dr. Elena Voss founded the Global Health Initiative in 2012.
[2] Global Health Initiative: The Global Health Initiative won the Nobel Peace Prize in 2021.
```

The same pattern held for the first inspected examples `syn-009002` and
`syn-009004`: both gold evidence doc ids were present in the observation.

## Validation

Local tests before remote sync:

```text
python -m pytest -q
106 passed in 4.12s
```

Remote related tests:

```text
tests/test_environment_rollout.py tests/test_cli.py::test_inspect_env_rollout_cli_dry_run_writes_prompts
4 passed in 0.07s
```

Post-run checks:

```text
search_prompts.jsonl lines: 50
answer_prompts.jsonl lines: 50
error scan: no Traceback/RuntimeError/ERROR/Exception matches
tmux session: no active server
```

## Analysis

Phase 5H confirms that controlled candidate pools solve the immediate retrieval
environment issue for small-slice rollout inspection. Compared with the previous
global-index result:

```text
global top-2 first 3 rows: gold_evidence_recall=0.0
gold+distractors top-8 first 50 rows: gold_evidence_recall=1.0
```

This does not mean the agent has learned better search. It means the environment
can now provide useful observations while still containing distractors. That is
the right setting for the next model-generation smoke and later GRPO rollout
experiments.

## Next Steps

- Run a small real model env rollout over this controlled pool, for example
  `limit=10`, `top_k=8`, `distractor_count=6`, on one GPU.
- Add summary metrics over generated search and answer actions:
  valid search, evidence recall, exact match, average observation docs, and
  assistant observation leakage.
- After the 10-sample smoke, decide whether to scale to 50 examples or improve
  query generation first.
