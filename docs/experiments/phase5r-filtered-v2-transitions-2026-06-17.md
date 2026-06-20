# Phase 5R Filtered-v2 Transition Export

## Goal

Remove the six remaining `answer_reward_type=none` rows found after the
500-example Phase 5R rollout, re-export a cleaner transition set, and prepare
the matching 50-step GRPO dry-run artifacts.

## Code Changes

- Added quality manifest:
  `configs/data_quality/phase5r_500_known_mismatches.json`
- Added GRPO config:
  `configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x50_softanswer.yaml`
- Added remote export launcher:
  `scripts/remote/phase5r_export_filtered_v2_transitions.sh`
- Added remote training launcher:
  `scripts/remote/phase5r_filtered_v2_env_transition_grpo_4gpu_978x50_softanswer.sh`
- Updated tests:
  `tests/test_verl_smoke.py`

## Local Verification

```text
python -m pytest tests/test_verl_smoke.py::test_phase5r_quality_manifest_marks_known_rollout_500_mismatches tests/test_verl_smoke.py::test_phase5r_filtered_v2_soft_answer_grpo_config_uses_cleaned_split -v --basetemp .pytest-tmp-phase5r-v2-green
2 passed

python -m pytest tests/test_verl_smoke.py -k "phase5q or phase5r" -v --basetemp .pytest-tmp-phase5r-v2-related
5 passed, 16 deselected

python -m pytest --basetemp .pytest-tmp-phase5r-v2-full
139 passed
```

## Remote Verification

Remote sync was narrow and limited to the changed Phase 5R files. Hashes
matched for:

- `phase5r_500_known_mismatches.json`
- `phase5r_filtered_v2_env_transition_grpo_4gpu_978x50_softanswer.yaml`
- `test_verl_smoke.py`
- `phase5r_export_filtered_v2_transitions.sh`
- `phase5r_filtered_v2_env_transition_grpo_4gpu_978x50_softanswer.sh`

Remote test:

```text
PYTHONNOUSERSITE=1 python -m pytest \
  tests/test_verl_smoke.py::test_phase5r_quality_manifest_marks_known_rollout_500_mismatches \
  tests/test_verl_smoke.py::test_phase5r_filtered_v2_soft_answer_grpo_config_uses_cleaned_split \
  -v --basetemp .pytest-tmp-phase5r-v2-remote

2 passed
```

Remote bash syntax check passed for the repo and `/data/wzl/LightningSearch-RL/runs`
copies of both new scripts.

## Export Command

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "bash /data/wzl/LightningSearch-RL/runs/phase5r_export_filtered_v2_transitions.sh"
```

Inputs:

```text
rollouts: /data/wzl/LightningSearch-RL/results/phase5r-env-rollout-gold-distractors-500/env_rollouts.jsonl
quality_manifest: /data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5r_500_known_mismatches.json
exclude_quality_flags: qa_type_mismatch, answer_none_low_reward
```

Outputs:

```text
filtered-v2 transitions: /data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2
log: /data/wzl/LightningSearch-RL/logs/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2.log
```

## Filtered-v2 Summary

```json
{
  "input_example_count": 500,
  "example_count": 489,
  "excluded_example_count": 11,
  "transition_count": 978,
  "answer_exact_match_rate": 0.977505,
  "answer_containment_match_rate": 1.0,
  "answer_token_f1": 0.99312,
  "gold_evidence_recall": 1.0,
  "valid_search_action_rate": 1.0,
  "valid_answer_action_rate": 1.0,
  "avg_total_reward": 1.36312,
  "avg_search_credit": 0.27,
  "avg_answer_credit": 1.09312,
  "excluded_quality_flag_counts": {
    "answer_none_low_reward": 6,
    "qa_type_mismatch": 5
  }
}
```

Excluded IDs:

```text
syn-009012
syn-009019
syn-009432
syn-009456
syn-009536
syn-009857
syn-009947
syn-010022
syn-010102
syn-010326
syn-010401
```

Line counts:

```text
transitions.jsonl: 978
reward_records.jsonl: 489
rollouts_for_grpo.jsonl: 489
```

Reward type counts after filtering:

```json
{
  "answer_reward_type_counts": {
    "containment": 11,
    "exact": 478
  },
  "none_rows": []
}
```

## GRPO Dry Run

Command:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train \
  --config configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x50_softanswer.yaml \
  --output-dir /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer \
  --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer \
  --dry-run \
  --print-command
```

Dry-run summary:

```text
source_type: transitions
train_rows: 782
val_rows: 196
parquet_written: true
reward_dump_path: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/reward_dump.jsonl
```

Generated training data:

```text
/data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/data/train.jsonl 782
/data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/data/val.jsonl 196
/data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/data/train.parquet
/data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/data/val.parquet
```

## Analysis

Filtered-v2 is materially cleaner than the first Phase 5R filtered export:

- removed all six `answer_reward_type=none` rows;
- kept the evidence and format behavior intact;
- improved answer exact match from `0.965657` to `0.977505`;
- improved containment match from `0.987879` to `1.0`;
- increased average total reward from `1.351082` to `1.36312`.

This is now the preferred input for the next 4-GPU 50-step GRPO run.

## Next Step

Launch one tmux training session after explicit approval:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260617-phase5r-filtered-v2-grpo-978x50 bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,7 bash /data/wzl/LightningSearch-RL/runs/phase5r_filtered_v2_env_transition_grpo_4gpu_978x50_softanswer.sh'"
```

Do not launch until four GPUs are below the project free-memory threshold.
The prelaunch check immediately after dry-run found every GPU above 4000 MiB:

```text
0: 19059 MiB
1: 18967 MiB
2: 18989 MiB
3: 28638 MiB
4: 28774 MiB
5: 18955 MiB
6: 18987 MiB
7: 16429 MiB
```

There were no active `lightningsearch-*` tmux sessions at that check.
