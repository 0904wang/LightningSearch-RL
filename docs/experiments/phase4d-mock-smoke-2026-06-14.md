# Phase 4D Mock Smoke - Chain Schema Strict Mode

## Goal

Verify that Phase 4D schema-first synthetic data changes work on the remote environment before any real DeepSeek pilot.

## Code

- Local branch: `master`
- Commit: `180dcba` (`feat: require chain schema for synthetic data`)
- Remote sync: narrow `git archive` sync to `/data/wzl/LightningSearch-RL/repo`
- Remote env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Python mode: `PYTHONNOUSERSITE=1`

## Commands

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m pytest
```

```bash
out=/data/wzl/LightningSearch-RL/results/phase4d-mock-smoke
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-validated-data \
  --mock \
  --raw "$out/raw.jsonl" \
  --valid "$out/valid.jsonl" \
  --rejects "$out/rejects.jsonl" \
  --target-valid 5 \
  --topics awards,archives \
  --concurrency 2 \
  --batch-size 2 \
  --max-attempts 5 \
  --seed 400 \
  --summary "$out/synthesis_summary.json" \
  --require-chain-schema
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli prepare-hotpot \
  --raw "$out/valid.jsonl" \
  --corpus "$out/corpus.jsonl" \
  --examples "$out/examples.jsonl"
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli build-index \
  --corpus "$out/corpus.jsonl" \
  --index "$out/index.json"
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-grpo \
  --examples "$out/examples.jsonl" \
  --index "$out/index.json" \
  --out-dir "$out/grpo" \
  --top-k 2
```

## Results

Remote pytest:

```text
51 passed in 0.20s
```

Synthesis summary:

```json
{
  "api_failed": 0,
  "generated": 5,
  "requested": 5,
  "require_chain_schema": true,
  "reject_count": 0,
  "stopped_reason": "target_valid_reached",
  "target_valid": 5,
  "valid_count": 5
}
```

GRPO export summary:

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

## Analysis

Strict mode is wired through the prompt, validator, CLI, mock generation, Hotpot conversion, lexical index, and GRPO export path. The next real pilot should target 50 valid examples with `--require-chain-schema` to measure whether Phase 4D reduces Phase 4C's answer-as-title rejects and shallow multi-hop artifacts.
