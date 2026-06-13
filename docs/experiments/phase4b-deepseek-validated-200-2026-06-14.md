# Phase 4B DeepSeek Validated 200-Row Pilot

Date: 2026-06-14

## Goal

Run the validated DeepSeek synthetic data loop until 200 valid HotpotQA-like
multi-hop QA rows are available, then export GRPO-ready rollout and transition
artifacts.

## Code and Environment

- Commit: `32959d3`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- tmux session: `lightningsearch-20260614-phase4b-deepseek-validated-200`
- Log path: `/data/wzl/LightningSearch-RL/logs/phase4b-deepseek-validated-200.log`
- Results path: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200`
- GPU: not used
- Model: `deepseek-chat`
- Endpoint: `https://api.deepseek.com`
- Remote log time: UTC (`2026-06-13T16:42:49+00:00` to `2026-06-13T16:43:25+00:00`)

## Command

The API key was passed through the launched session environment only. It was not
written to source files, config files, logs, or result artifacts.

```bash
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-validated-data \
  --raw /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_raw.jsonl \
  --valid /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_valid.jsonl \
  --rejects /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_rejects.jsonl \
  --target-valid 200 \
  --topics awards,archives,research \
  --concurrency 50 \
  --batch-size 50 \
  --max-attempts 320 \
  --seed 2000 \
  --summary /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/validated_summary.json \
  --model deepseek-chat \
  --base-url https://api.deepseek.com
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli prepare-hotpot \
  --raw /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_valid.jsonl \
  --corpus /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/corpus.jsonl \
  --examples /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/examples.jsonl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli build-index \
  --corpus /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/corpus.jsonl \
  --index /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/index.json
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-grpo \
  --examples /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/examples.jsonl \
  --index /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/index.json \
  --out-dir /data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/grpo \
  --top-k 2
```

## Raw Result Summary

Validated synthesis:

```json
{
  "api_failed": 0,
  "batch_size": 50,
  "concurrency": 50,
  "generated": 254,
  "max_attempts": 320,
  "reject_count": 54,
  "requested": 254,
  "stopped_reason": "target_valid_reached",
  "target_valid": 200,
  "valid_count": 200
}
```

Reject reason distribution:

```json
{
  "supporting_facts must cover at least two titles": 27,
  "answer not found in supporting evidence": 27
}
```

GRPO export:

```json
{
  "avg_reward": 0.4795,
  "avg_search_count": 1.0,
  "example_count": 200,
  "rollout_count": 200,
  "top_k": 2,
  "transition_count": 400
}
```

Line counts:

```text
254 synthetic_raw.jsonl
200 synthetic_valid.jsonl
54 synthetic_rejects.jsonl
200 grpo/rollouts.jsonl
400 grpo/transitions.jsonl
```

Secret scan:

```text
NO_SECRET_PATTERN
```

## Artifacts

- Raw rows: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_raw.jsonl`
- Valid rows: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_valid.jsonl`
- Rejects: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/synthetic_rejects.jsonl`
- Corpus: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/corpus.jsonl`
- Examples: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/examples.jsonl`
- Index: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/index.json`
- GRPO rollouts: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/grpo/rollouts.jsonl`
- GRPO transitions: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/grpo/transitions.jsonl`
- GRPO reward records: `/data/wzl/LightningSearch-RL/results/phase4b-deepseek-validated-200/grpo/reward_records.jsonl`

## Analysis

The validated loop reached the target 200 valid rows in 254 generated rows, a
valid rate of about 78.7%. This improves operational usability over the first
50-row pilot because downstream export receives the requested valid sample count
without manual resubmission.

Rejects are now concentrated in two equal categories: same-title supporting
facts and answer/evidence mismatch. The previous "supporting_facts count too
small" failure disappeared, suggesting the tightened prompt helped with minimum
format completeness but not yet with title diversity and answer grounding.

The rule-based GRPO export still has modest average reward (`0.4795`), which is
expected for noisy synthetic examples and a simple lexical/rule rollout. The
artifact set is now large enough to inspect query quality and reward components
before moving into SFT or GRPO training.

## Next Step

Add a lightweight reject repair pass focused on the two remaining failure modes,
or inspect 20 valid/rejected examples manually before changing the prompt again.
Then run retrieval metrics on the 200-row set and compare against the 50-row
pilot.
