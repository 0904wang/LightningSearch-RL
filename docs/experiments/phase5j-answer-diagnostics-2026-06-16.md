# Phase 5J Answer Diagnostics

Date: 2026-06-16

## Goal

Add answer normalization and rollout-answer diagnostics for Phase 5I results.
The goal is to separate strict exact-match failures from likely data quality
issues and relaxed answer matches.

## Code Changes

```text
src/lightningsearch_rl/answer_metrics.py
src/lightningsearch_rl/rollout_diagnostics.py
src/lightningsearch_rl/environment_rollout.py
src/lightningsearch_rl/cli.py
tests/test_answer_metrics.py
tests/test_rollout_diagnostics.py
tests/test_environment_rollout.py
```

New CLI:

```bash
python -m lightningsearch_rl.cli diagnose-rollout-answers \
  --rollouts <env_rollouts.jsonl> \
  --out <answer_diagnostics.json>
```

## Runtime

```text
local workspace: D:\resume\Agent RL
local branch/commit: master @ 44493db
remote repo: /data/wzl/LightningSearch-RL/repo
remote repo state: not-a-git-repo, narrow sync
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
input: /data/wzl/LightningSearch-RL/results/phase5i-env-rollout-gold-distractors-10/env_rollouts.jsonl
output: /data/wzl/LightningSearch-RL/results/phase5i-env-rollout-gold-distractors-10/answer_diagnostics.json
```

## Validation

Local tests:

```text
python -m pytest --basetemp .pytest-tmp-answerdiag-full -q
109 passed in 2.65s
```

Remote related tests:

```text
tests/test_answer_metrics.py tests/test_rollout_diagnostics.py tests/test_environment_rollout.py
6 passed in 0.05s
```

## Diagnostics Command

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-rollout-answers \
  --rollouts /data/wzl/LightningSearch-RL/results/phase5i-env-rollout-gold-distractors-10/env_rollouts.jsonl \
  --out /data/wzl/LightningSearch-RL/results/phase5i-env-rollout-gold-distractors-10/answer_diagnostics.json
```

## Raw Summary

```json
{
  "answer_containment_match_rate": 0.8,
  "answer_exact_match_rate": 0.7,
  "answer_token_f1": 0.78,
  "example_count": 10,
  "suspicious_count": 2
}
```

Suspicious rows:

```text
syn-009012
question: Which research institute founded in 2021 is led by Dr. Elena Vasquez?
prediction: Global Health Research Institute
gold: Barcelona
reasons: prediction_matches_observation_title, question_gold_type_mismatch

syn-009019
question: Which journal was founded by the editor of the Journal of Applied Mathematics?
prediction: Journal of Computational Science
gold: 2020
reasons: question_gold_type_mismatch
```

The remaining non-exact row, `syn-009020`, is not flagged as suspicious because
it is a relaxed match:

```text
prediction: Golden Quill Award
gold: Golden Quill
```

## Analysis

Phase 5I strict exact match was 0.7. With relaxed containment, it rises to 0.8,
and token F1 is 0.78. The diagnostic supports the earlier manual read: two
errors are likely data/question-answer type mismatches rather than tool-use
failures.

This means the controlled-pool environment is now good enough for larger rollout
inspection, but the dataset should either be filtered or scored with both strict
EM and relaxed/token-F1 metrics.

## Next Steps

- Add a filter or flag for suspicious rows before larger rollout slices.
- Re-run controlled-pool model rollout on 50 examples and report EM,
  containment match, token F1, and suspicious-row-adjusted EM.
- Keep strict EM in the report, but avoid using it alone as the resume-facing
  metric while synthetic labels still contain type mismatches.
