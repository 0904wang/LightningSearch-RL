# Phase 4E Mock Smoke - Targeted Chain Repair

## Goal

Verify that Phase 4E targeted chain-schema repair works on the remote environment before any real DeepSeek pilot.

## Code

- Local branch: `master`
- Commit: `e4fa3b7` (`feat: repair chain schema synthesis rows`)
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
out=/data/wzl/LightningSearch-RL/results/phase4e-mock-smoke
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
  --seed 800 \
  --summary "$out/synthesis_summary.json" \
  --require-chain-schema \
  --repair-chain-schema
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
54 passed in 0.21s
```

Synthesis summary:

```json
{
  "api_failed": 0,
  "generated": 5,
  "requested": 5,
  "require_chain_schema": true,
  "repair_chain_schema": true,
  "repair_attempt_count": 0,
  "repair_success_count": 0,
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

The remote smoke validates CLI wiring for `--repair-chain-schema` and confirms the strict schema path still produces usable Hotpot-style examples and GRPO artifacts. Mock rows are already valid, so repair counters are expected to stay at zero. The real pilot should compare repair attempts and repair successes against Phase 4D's dominant missing-intermediate reject reasons.
