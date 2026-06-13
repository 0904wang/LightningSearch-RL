# Phase 4A Mock Synthetic Data Smoke

Date: 2026-06-13

## Goal

Verify that the Phase 4A synthetic data pipeline runs on the approved remote
workspace without using a real LLM API:

- mock HotpotQA-like raw row generation
- synthetic row validation
- Hotpot-style corpus/example conversion
- lexical index build
- GRPO rollout/transition/reward export

## Code and Environment

- Local commit: `0cbebcb55fd192ce4de62194ba822c9f9d7573a4`
- Remote sync method: approved narrow `git archive` upload and extraction
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Remote archive: `/data/wzl/LightningSearch-RL/runs/lightningsearch-0cbebcb.tar`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Python: remote approved env, `PYTHONNOUSERSITE=1`
- GPU: not used for this mock smoke

## Commands

Remote tests:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m pytest
```

Remote mock smoke:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-data --mock --out /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/synthetic_raw.jsonl --count 5 --topics awards,archives,research --concurrency 50 --seed 100 --summary /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/synthesis_summary.json
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli validate-synthetic --raw /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/synthetic_raw.jsonl --valid /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/synthetic_valid.jsonl --rejects /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/synthetic_rejects.jsonl --summary /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/validation_summary.json
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli prepare-hotpot --raw /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/synthetic_valid.jsonl --corpus /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/corpus.jsonl --examples /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/examples.jsonl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli build-index --corpus /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/corpus.jsonl --index /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/index.json
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-grpo --examples /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/examples.jsonl --index /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/index.json --out-dir /data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/grpo --top-k 2
```

## Raw Result Summary

Remote tests:

```text
34 passed in 0.15s
```

Synthesis summary:

```json
{
  "concurrency": 50,
  "failed": 0,
  "failures": [],
  "requested": 5,
  "skipped_existing": 0,
  "written": 5
}
```

Validation summary:

```json
{
  "raw_count": 5,
  "reject_count": 0,
  "valid_count": 5
}
```

GRPO summary:

```json
{
  "avg_reward": 1.37,
  "avg_search_count": 1.0,
  "example_count": 5,
  "rollout_count": 5,
  "top_k": 2,
  "transition_count": 10
}
```

## Artifacts

- Raw synthetic rows: `/data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/synthetic_raw.jsonl`
- Valid rows: `/data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/synthetic_valid.jsonl`
- Rejects: `/data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/synthetic_rejects.jsonl`
- Corpus: `/data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/corpus.jsonl`
- Examples: `/data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/examples.jsonl`
- Index: `/data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/index.json`
- GRPO rollouts: `/data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/grpo/rollouts.jsonl`
- GRPO transitions: `/data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/grpo/transitions.jsonl`
- GRPO reward records: `/data/wzl/LightningSearch-RL/results/phase4a-mock-smoke/grpo/reward_records.jsonl`

## Analysis

The remote environment can execute the new synthetic pipeline end to end without
network/API dependencies. The generated mock rows pass schema and evidence
validation, then feed the existing adapter, retrieval index, and GRPO export
contracts. This validates the code path needed before running a real DeepSeek
pilot.

## Next Step

Run a small real DeepSeek pilot only after `DEEPSEEK_API_KEY` is present in the
remote shell environment and the exact command is approved. Suggested first real
scope: 50 rows, concurrency 50, same validation/export pipeline, results under
`/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot`.
