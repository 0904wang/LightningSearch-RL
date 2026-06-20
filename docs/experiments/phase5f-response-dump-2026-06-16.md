# Phase 5F Response Dump

Date: 2026-06-16

## Goal

Dump generated responses from the Phase 5D docid-fixed SFT checkpoint on the same
two-stage prompt shape used by the Phase 5F GRPO smoke. This checks the actual
text behavior behind the Phase 5F aggregate rewards:

- search stage: `system + question -> <search>...</search>`
- answer stage: `system + question + assistant search + observation -> <answer>...</answer>`

## Runtime

```text
local workspace: D:\resume\Agent RL
local branch/commit: master @ 44493db
remote repo: /data/wzl/LightningSearch-RL/repo
remote repo state: not-a-git-repo, narrow sync
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
python: 3.10.20
gpu: CUDA_VISIBLE_DEVICES=6
script: /data/wzl/LightningSearch-RL/runs/phase5f_response_dump.sh
session: lightningsearch-20260616-phase5f-response-dump
```

Remote log timestamps are UTC:

```text
started_at=2026-06-15T17:00:10+00:00
finished_at=2026-06-15T17:00:21+00:00
```

## Inputs

```text
sft source: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
offset: 0
limit: 3
max_new_tokens: 64
modes: search,answer
```

The dry-run generated:

```text
search prompts: 3
answer prompts: 3
ids: syn-009000, syn-009002, syn-009004
```

## Launch

The first PowerShell-quoted launch command failed before creating a session:

```text
bash: -c: line 1: unexpected EOF while looking for matching `"'
```

The successful launch was sent via stdin to avoid local shell quote rewriting:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "bash -s" <<'SH'
set -e
tmux new-session -d -s lightningsearch-20260616-phase5f-response-dump "bash -lc 'CUDA_VISIBLE_DEVICES=6 bash /data/wzl/LightningSearch-RL/runs/phase5f_response_dump.sh'"
tmux list-sessions
SH
```

The wrapper returned a spurious `unknown command: list-sessions` after launch, but
the run itself completed and produced the expected log and result files. A later
direct check showed no active tmux session:

```text
tmux 3.2a
no server running on /tmp/tmux-1000/default
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5f-tiny-grpo-docidfix-two-stage-4gpu-response-dump.log
results: /data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu-response-dump
remote record: /data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu-response-dump/EXPERIMENT_RECORD.md
```

Result files:

```text
answer_prompts.jsonl
dry_run_summary.json
generations.jsonl
search_prompts.jsonl
summary.json
```

## Raw Summary

```json
{
  "answer_prompt_count": 3,
  "by_mode": {
    "answer": {
      "answer_tag_rate": 1.0,
      "avg_new_tokens": 9.666667,
      "eos_rate": 1.0,
      "example_count": 3,
      "gold_answer_mention_rate": 1.0,
      "observation_tag_rate": 0.0,
      "search_tag_rate": 0.0,
      "single_action_rate": 1.0,
      "valid_answer_action_rate": 1.0,
      "valid_search_action_rate": 0.0
    },
    "search": {
      "answer_tag_rate": 0.0,
      "avg_new_tokens": 23.666667,
      "eos_rate": 1.0,
      "example_count": 3,
      "gold_answer_mention_rate": 0.0,
      "observation_tag_rate": 0.0,
      "search_tag_rate": 1.0,
      "single_action_rate": 1.0,
      "valid_answer_action_rate": 0.0,
      "valid_search_action_rate": 1.0
    }
  },
  "dry_run": false,
  "limit": 3,
  "max_new_tokens": 64,
  "overall": {
    "answer_tag_rate": 0.5,
    "avg_new_tokens": 16.666667,
    "eos_rate": 1.0,
    "example_count": 6,
    "gold_answer_mention_rate": 0.5,
    "observation_tag_rate": 0.0,
    "search_tag_rate": 0.5,
    "single_action_rate": 1.0,
    "valid_answer_action_rate": 0.5,
    "valid_search_action_rate": 0.5
  },
  "search_prompt_count": 3
}
```

## Generated Responses

```text
syn-009000 search: <search>Which award did the organization founded by Dr. Elena Voss receive in 2021?</search>
syn-009002 search: <search>Which city is home to the institute where Dr. Elena Voss serves as director?</search>
syn-009004 search: <search>Which university hosted the conference where Dr. Alice Chen presented her award-winning research?</search>

syn-009000 answer: <answer>Nobel Peace Prize</answer>
syn-009002 answer: <answer>Riverstone</answer>
syn-009004 answer: <answer>University of Cambridge</answer>
```

## Validation

Post-run checks:

```text
summary.json exists
generations.jsonl lines: 6
result directory size: 32K
tmux session: no active server after completion
GPU 6 after run: 3505 MiB / 32607 MiB
```

Warnings in the log:

```text
The tokenizer you are loading ... with an incorrect regex pattern ...
`torch_dtype` is deprecated! Use `dtype` instead!
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k'].
```

These warnings did not stop generation and the outputs were valid for this
inspection.

## Analysis

The SFT checkpoint is aligned with the two-stage objective on this tiny slice:

- Search prompts produce exactly one valid `<search>` action.
- Answer prompts produce exactly one valid `<answer>` action.
- The answer stage mentions the gold answer for all three examples.
- No generated response includes `<observation>`, which confirms the runtime
  observation boundary is respected by the inspected checkpoint.

This supports the Phase 5F reward result: the high shaped score is not just a
reward-hook artifact. The model text behavior matches the search/answer stage
contract on the inspected rows.

## Next Steps

- Implement a true offline environment rollout where generated `<search>` is
  executed against BM25 and observations are inserted dynamically.
- Keep the same single-action parser and stage-aware reward contract.
- Add a small end-to-end inspection after environment insertion:
  `question -> generated search -> BM25 observation -> generated answer`.
