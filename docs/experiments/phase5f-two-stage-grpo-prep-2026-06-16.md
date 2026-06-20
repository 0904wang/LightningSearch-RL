# Phase 5F Two-Stage GRPO Preparation

Date: 2026-06-16

## Goal

Fix the Phase 5E objective mismatch before running another GRPO smoke. Phase 5E
proved the GRPO infrastructure works, but it rewarded only final
`<answer>...</answer>` outputs while the turn-level SFT checkpoint correctly
emits `<search>...</search>` for question-only prompts.

Phase 5F prepares a two-stage GRPO input path:

- search stage: prompt is `system + question`, reward valid single
  `<search>...</search>`
- answer stage: prompt is `system + question + assistant search + runtime
  observation`, reward exact `<answer>...</answer>`

## Changes

- `src/lightningsearch_rl/verl_reward.py`
  - added `reward_stage=search` support
  - valid single search action receives `search_reward=1.0`
  - answer outputs are rejected when search stage is expected
- `src/lightningsearch_rl/verl_smoke.py`
  - config may now use `sft_turns_path` instead of `rollouts_path`
  - `prompt_stages: [search, answer]` expands each SFT-turn row into stage rows
  - existing rollouts-based configs remain supported
- `configs/experiments/phase5f_tiny_grpo_docidfix_two_stage_4gpu.yaml`
  - tiny 4-GPU two-stage GRPO config using docidfix SFT-turns data
  - warm-starts from the Phase 5D docidfix HF checkpoint

## Tests

TDD red checks:

```text
test_compute_score_rewards_valid_search_stage_action failed with score -0.03
test_compute_score_rejects_answer_when_search_stage_expected failed with score 1.07
test_prepare_verl_smoke_builds_two_stage_rows_from_sft_turns failed because rollouts_path was required
```

Green checks:

```text
python -m pytest tests\test_verl_reward.py::test_compute_score_rewards_valid_search_stage_action tests\test_verl_reward.py::test_compute_score_rejects_answer_when_search_stage_expected -q -> 2 passed
python -m pytest tests\test_verl_smoke.py::test_prepare_verl_smoke_builds_two_stage_rows_from_sft_turns -q -> 1 passed
python -m pytest tests\test_verl_smoke.py tests\test_verl_reward.py tests\test_grpo.py tests\test_agent_loop.py -q -> 22 passed
python -m pytest -q -> 102 passed
```

Remote related tests:

```text
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m pytest tests/test_verl_smoke.py tests/test_verl_reward.py tests/test_grpo.py tests/test_agent_loop.py -q

22 passed in 0.34s
```

## Dry Run

Command:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train \
  --config configs/experiments/phase5f_tiny_grpo_docidfix_two_stage_4gpu.yaml \
  --output-dir /data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu \
  --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5f-tiny-grpo-docidfix-two-stage-4gpu \
  --dry-run --print-command
```

Result:

```text
source: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
source rows available: 500
train source rows: 2
val source rows: 1
train stage rows: 4
val stage rows: 2
```

Train rows:

```text
syn-009000::search stage=search roles=system,user ground_truth="" expected=<search>Which award did the organization founded by Dr. Elena Voss receive in 2021?</search>
syn-009000::answer stage=answer roles=system,user,assistant,user ground_truth=Nobel Peace Prize expected=<answer>Nobel Peace Prize</answer>
syn-009002::search stage=search roles=system,user ground_truth="" expected=<search>Which city is home to the institute where Dr. Elena Voss serves as director?</search>
syn-009002::answer stage=answer roles=system,user,assistant,user ground_truth=Riverstone expected=<answer>Riverstone</answer>
```

Dry-run artifacts:

```text
/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu/data/train.jsonl
/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu/data/train.parquet
/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu/data/val.jsonl
/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu/data/val.parquet
/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu/dry_run_summary.json
/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu/launch_command.txt
/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu/manifest.json
```

## Next Step

Launch the Phase 5F tiny 4-GPU GRPO smoke after the standard approval report.
The expected improvement over Phase 5E is that search-stage validation should no
longer be scored as `-0.03` simply because the model emits a valid `<search>`
instead of a final `<answer>`.
