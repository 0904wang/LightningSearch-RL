# Phase 5L: Env Rollout Transition Export From Phase 5K

## Goal

Convert the 50 real model environment rollouts from Phase 5K into Lightning-style
training artifacts:

- `transitions.jsonl`: state-action transitions for search and answer turns
- `reward_records.jsonl`: per-example reward components and step credit
- `rollouts_for_grpo.jsonl`: prompt/response/reward rows usable by the existing
  rollout-based `verl` data prep path
- `summary.json`: aggregate metrics for quick comparison

This phase targets the resume claim that agent execution logs can be decoupled
from RL training and adapted into trainable transition records with
evidence-aware credit assignment.

## Runtime

date: 2026-06-16
session: `lightningsearch-20260616-phase5l-env-transitions`
remote workspace: `/data/wzl/LightningSearch-RL`
remote repo: `/data/wzl/LightningSearch-RL/repo`
remote repo type: narrow-synced working tree, not a git repo
local branch/commit at launch: `master @ 44493db`
conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
GPU: none

## Inputs

rollouts:

```text
/data/wzl/LightningSearch-RL/results/phase5k-env-rollout-gold-distractors-50/env_rollouts.jsonl
```

source rollout line count:

```text
50
```

## Command

```bash
tmux new-session -d -s lightningsearch-20260616-phase5l-env-transitions "bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-env-transitions --rollouts /data/wzl/LightningSearch-RL/results/phase5k-env-rollout-gold-distractors-50/env_rollouts.jsonl --out-dir /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5l-env-transitions-from-phase5k.log'"
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5l-env-transitions-from-phase5k.log
results: /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k
transitions: /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl
reward records: /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/reward_records.jsonl
rollouts for GRPO: /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/rollouts_for_grpo.jsonl
summary: /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/summary.json
```

## Summary

```json
{
  "answer_containment_match_rate": 0.96,
  "answer_exact_match_rate": 0.92,
  "answer_token_f1": 0.946,
  "avg_answer_credit": 1.02,
  "avg_search_credit": 0.27,
  "avg_total_reward": 1.29,
  "example_count": 50,
  "gold_evidence_recall": 1.0,
  "rollout_count": 50,
  "transition_count": 100,
  "valid_answer_action_rate": 1.0,
  "valid_search_action_rate": 1.0
}
```

Line counts:

```text
100 /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl
50 /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/reward_records.jsonl
50 /data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/rollouts_for_grpo.jsonl
```

Transition shape check:

```text
transition_action_counts= {'answer': 50, 'search': 50}
terminal_count= 50
reward_min_max= 0.37 1.37
```

First reward record:

```json
{
  "id": "syn-009000",
  "answer_reward": 1.0,
  "evidence_reward": 1.0,
  "format_reward": 1.0,
  "tool_validity_reward": 1.0,
  "search_count": 1,
  "search_cost": 0.03,
  "search_credit": 0.27,
  "answer_credit": 1.1,
  "total": 1.37,
  "valid_search_action": true,
  "valid_answer_action": true,
  "answer_exact_match": true,
  "answer_token_f1": 1.0,
  "answer_containment_match": true,
  "final_answer": "Nobel Peace Prize",
  "gold_answer": "Nobel Peace Prize"
}
```

## Validation

Local tests after implementation:

```text
111 passed in 2.83s
```

Remote targeted tests after narrow sync:

```text
2 passed in 0.07s
```

Remote full tests after narrow sync:

```text
111 passed in 0.72s
```

Monitoring notes:

- First post-launch SSH monitor attempt timed out; a single retry succeeded.
- A first read-only sampling command failed because remote default shell had no
  `python`; the same check passed after activating the approved conda env.
- The tmux session exited normally; `tmux list-sessions` reported no running
  sessions after completion.

## Analysis

The adapter produced exactly two transitions per successful rollout: one search
state-action step and one terminal answer state-action step. The credit split is:

```text
search_credit = 0.2 * evidence_reward + 0.1 * tool_validity_reward - 0.03 * search_count
answer_credit = answer_reward + 0.1 * format_reward
total = search_credit + answer_credit
```

For exact-answer examples, total reward is `1.37`. The minimum reward is `0.37`,
which corresponds to examples where retrieval/tool behavior was valid and
evidence was found, but the strict answer EM failed.

This is the first artifact in the project that directly represents the
Agent-Lightning style separation between runtime traces and training records.
The runtime does not need to know about GRPO or `verl`; it emits environment
rollouts, and this adapter converts them into transition/reward data.

## Next Steps

1. Add a `verl` prep path that can consume `rollouts_for_grpo.jsonl` from
   environment rollouts.
2. Run a tiny 4-GPU GRPO smoke using this Phase 5L export rather than the earlier
   gold-answer/offline rollout rows.
3. If the tiny smoke is stable, scale to a larger generated rollout set after
   filtering or repairing the two suspicious Phase 5K synthetic labels.
