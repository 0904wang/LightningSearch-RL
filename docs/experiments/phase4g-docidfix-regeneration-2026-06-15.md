# Phase 4G DocID-Fix Regeneration

Date: 2026-06-15

## Goal

Regenerate derived artifacts from the existing Phase 4G `valid.jsonl` after
fixing row-scoped passage ids. The raw synthetic data was not regenerated; only
prepared corpus/examples/index and training exports were rebuilt.

## Source

```text
/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/valid.jsonl
```

## Output Directory

```text
/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix
```

## Commands

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
out=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli prepare-hotpot \
  --raw /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/valid.jsonl \
  --corpus "$out/corpus.jsonl" \
  --examples "$out/examples.jsonl"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli build-index \
  --corpus "$out/corpus.jsonl" \
  --index "$out/index.json"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-sft-warmup \
  --examples "$out/examples.jsonl" \
  --index "$out/index.json" \
  --out-dir "$out/sft-warmup-gold"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-sft-turns \
  --examples "$out/examples.jsonl" \
  --index "$out/index.json" \
  --out-dir "$out/sft-turns-gold"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-grpo \
  --examples "$out/examples.jsonl" \
  --index "$out/index.json" \
  --out-dir "$out/grpo-gold-answer" \
  --top-k 5
```

## Artifacts

```text
corpus: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/corpus.jsonl
examples: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/examples.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
sft warmup: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-warmup-gold/sft_warmup.jsonl
sft turns: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
grpo rollouts: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/grpo-gold-answer/rollouts.jsonl
```

## Summary

```text
corpus rows: 2344
examples rows: 500
sft_warmup rows: 500
sft_turns rows: 500
grpo rollouts: 500
grpo transitions: 1000
```

SFT turn summary:

```json
{
  "answer_tag_rate": 1.0,
  "assistant_observation_rate": 0.0,
  "assistant_single_action_rate": 1.0,
  "avg_gold_evidence_count": 2.0,
  "example_count": 500,
  "gold_evidence_coverage": 1.0,
  "non_empty_answer_rate": 1.0,
  "sft_rows": 500
}
```

GRPO summary:

```json
{
  "avg_reward": 0.3732,
  "avg_search_count": 1.0,
  "example_count": 500,
  "rollout_count": 500,
  "top_k": 5,
  "transition_count": 1000
}
```

## Targeted Check

The previous bad held-out row now has row-scoped ids and matching evidence:

```text
id: syn-010502
answer: National Science Foundation
gold_doc_ids:
  hotpot::syn-010502::Dr. Elena Voss::0
  hotpot::syn-010502::Institute for Quantum Materials::0
observation:
  Dr. Elena Voss founded the Institute for Quantum Materials in 2019.
  The Institute for Quantum Materials received a major grant from the National Science Foundation.
```

The exported turn-level target is now:

```text
<answer>National Science Foundation</answer>
```

## Full Audit

Every SFT-turn row was checked to ensure the gold answer appears in the runtime
observation:

```json
{
  "row_count": 500,
  "answer_missing_in_observation_count": 0,
  "bad_ids": []
}
```

## Conclusion

The doc-id collision root cause is fixed in the regenerated derived artifacts.
Future SFT and GRPO runs should use the `phase4g-deepseek-titlefix-500-docidfix`
directory instead of the original Phase 4G derived artifacts.
