# Phase 4A DeepSeek Synthetic Data Pilot

Date: 2026-06-13

## Goal

Run the first real DeepSeek-powered synthetic data pilot for HotpotQA-like
multi-hop QA rows, then validate rows and export GRPO-ready artifacts.

## Code and Environment

- Commit: `d06b18d`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- tmux session: `lightningsearch-20260613-phase4a-deepseek-pilot`
- Log path: `/data/wzl/LightningSearch-RL/logs/phase4a-deepseek-pilot.log`
- Results path: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot`
- GPU: not used
- Model: `deepseek-chat`
- Endpoint: `https://api.deepseek.com`

## Command

The key was provided via `DEEPSEEK_API_KEY` in the launched session environment;
it was not written to source files, config files, or logs.

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-data \
  --out /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthetic_raw.jsonl \
  --count 50 \
  --topics awards,archives,research \
  --concurrency 50 \
  --seed 1000 \
  --summary /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthesis_summary.json \
  --model deepseek-chat \
  --base-url https://api.deepseek.com
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli validate-synthetic \
  --raw /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthetic_raw.jsonl \
  --valid /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthetic_valid.jsonl \
  --rejects /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthetic_rejects.jsonl \
  --summary /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/validation_summary.json
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli prepare-hotpot \
  --raw /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthetic_valid.jsonl \
  --corpus /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/corpus.jsonl \
  --examples /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/examples.jsonl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli build-index \
  --corpus /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/corpus.jsonl \
  --index /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/index.json
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-grpo \
  --examples /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/examples.jsonl \
  --index /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/index.json \
  --out-dir /data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/grpo \
  --top-k 2
```

## Raw Result Summary

Synthesis:

```json
{
  "concurrency": 50,
  "failed": 0,
  "requested": 50,
  "skipped_existing": 0,
  "written": 50
}
```

Validation:

```json
{
  "raw_count": 50,
  "reject_count": 13,
  "valid_count": 37
}
```

Reject reason distribution:

```json
{
  "supporting_facts must cover at least two titles": 7,
  "answer not found in supporting evidence": 4,
  "supporting_facts must contain at least two facts": 2
}
```

GRPO export:

```json
{
  "avg_reward": 0.46009,
  "avg_search_count": 1.0,
  "example_count": 37,
  "rollout_count": 37,
  "top_k": 2,
  "transition_count": 74
}
```

## Artifacts

- Raw synthetic rows: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthetic_raw.jsonl`
- Valid rows: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthetic_valid.jsonl`
- Rejects: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/synthetic_rejects.jsonl`
- Corpus: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/corpus.jsonl`
- Examples: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/examples.jsonl`
- Index: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/index.json`
- GRPO rollouts: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/grpo/rollouts.jsonl`
- GRPO transitions: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/grpo/transitions.jsonl`
- GRPO reward records: `/data/wzl/LightningSearch-RL/results/phase4a-deepseek-pilot/grpo/reward_records.jsonl`

## Analysis

The real DeepSeek synthesis path works end to end. The pass rate was 74%
(`37/50`) under the first prompt. Most rejects are not API failures; they are
schema or evidence quality failures caused by insufficient multi-title
supporting facts or answer/evidence mismatch. This is a useful signal for the
next prompt iteration.

The exported GRPO reward is lower than the deterministic mock smoke because the
rule-based rollout often does not recover exact answers from noisier generated
questions/evidence. That makes this pilot a realistic stress test for reward
shaping, query quality, and evidence-aware filtering.

## Next Step

Improve the synthesis prompt and/or add a repair pass:

- require exactly two supporting fact titles from different context entries
- require the answer string to appear verbatim in the final supporting sentence
- optionally retry or repair only rejected rows until the valid count reaches a
  target budget

After that, run a 200-row pilot and compare valid rate, evidence recall, and
GRPO reward statistics against this 50-row baseline.
