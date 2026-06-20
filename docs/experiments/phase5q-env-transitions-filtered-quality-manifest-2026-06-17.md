# Phase 5Q Filtered Env Transitions with Quality Manifest

## Goal

Re-export Phase 5Q environment rollout transitions after excluding confirmed
synthetic QA type-mismatch rows, preserving the original Phase 5Q artifacts.

## Context

The Phase 5Q-B GRPO run completed, but diagnostics showed several low-reward
answer rows caused by data quality rather than tool-use failures. A deterministic
quality manifest was added at:

```text
configs/data_quality/phase5q_known_mismatches.json
```

The manifest tags these IDs as `qa_type_mismatch`:

```text
syn-009012
syn-009019
syn-009432
syn-009456
syn-009536
```

## Remote Validation Before Export

Remote target tests:

```text
tests/test_env_transitions.py::test_export_env_rollout_transitions_tags_quality_manifest_rows PASSED
tests/test_cli.py::test_export_env_transitions_cli_writes_transition_artifacts PASSED
tests/test_cli.py::test_export_env_transitions_cli_excludes_quality_manifest_rows PASSED
3 passed, 15 deselected
```

Remote full suite:

```text
135 passed in 1.06s
```

## Command

The first export attempt used a relative manifest path and failed because the
approved-path guard requires approved absolute paths for this remote workflow:

```text
ValueError: path is outside approved paths: configs/data_quality/phase5q_known_mismatches.json
```

The retry used the same manifest via an absolute approved path:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "bash -lc 'set -o pipefail && cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-env-transitions --rollouts /data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200/env_rollouts.jsonl --out-dir /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200-filtered --quality-manifest /data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5q_known_mismatches.json --exclude-quality-flag qa_type_mismatch 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5q-env-transitions-soft-answer-from-5q200-filtered.log'"
```

## Artifacts

```text
input rollouts: /data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200/env_rollouts.jsonl
filtered results: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200-filtered
log: /data/wzl/LightningSearch-RL/logs/phase5q-env-transitions-soft-answer-from-5q200-filtered.log
transitions: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200-filtered/transitions.jsonl
reward_records: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200-filtered/reward_records.jsonl
rollouts_for_grpo: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200-filtered/rollouts_for_grpo.jsonl
summary: /data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200-filtered/summary.json
```

## Result Summary

```text
input_example_count: 200
example_count: 195
excluded_example_count: 5
excluded_example_ids:
  - syn-009012
  - syn-009019
  - syn-009432
  - syn-009456
  - syn-009536
excluded_quality_flag_counts: qa_type_mismatch=5
transition_count: 390
reward_records rows: 195
rollouts_for_grpo rows: 195
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.974359
answer_containment_match_rate: 1.0
answer_token_f1: 0.990855
gold_evidence_recall: 1.0
avg_search_credit: 0.27
avg_answer_credit: 1.090855
avg_total_reward: 1.360855
```

Consistency check:

```text
transitions.jsonl rows: 390, excluded_overlap: []
reward_records.jsonl rows: 195, excluded_overlap: []
rollouts_for_grpo.jsonl rows: 195, excluded_overlap: []
min_total: 0.87
answer_reward_type_counts: exact=190, containment=5
low_total_below_1:
  - syn-009154, total=0.87, answer_reward_type=containment
  - syn-009178, total=0.87, answer_reward_type=containment
```

## Analysis

The filter did exactly what was needed for the known Phase 5Q data-quality
problem: the five confirmed `qa_type_mismatch` rows were removed, and none of
those IDs remain in the filtered transitions, reward records, or GRPO rollouts.

The filtered set is materially cleaner than the original Phase 5Q export:

- examples decrease from 200 to 195;
- transitions decrease from 400 to 390;
- valid search and answer action rates remain 1.0;
- gold evidence recall improves to 1.0;
- containment match rate reaches 1.0;
- average total reward increases to 1.360855.

The two remaining lower-total rows, `syn-009154` and `syn-009178`, are
containment cases, not confirmed question-answer type mismatches. They should
remain in the data for now because the soft-answer reward is designed to handle
partial aliases and containment answers.

## Next Step

Use this filtered export for the next 390-transition GRPO sanity check or scale
the rollout to 500 examples / about 1000 transitions with the same quality
manifest enabled. For a larger run, keep both the unfiltered and filtered
summaries so the resume project can report data-quality filtering as a concrete
diagnostic improvement.
