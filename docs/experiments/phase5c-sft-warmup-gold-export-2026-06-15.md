# Phase 5C Gold-Evidence SFT Warmup Export

Date: 2026-06-15

## Goal

Create a clean SFT warmup dataset that teaches the model the target `think/search/observation/answer` tag format with non-empty gold answers and gold evidence. This is intended to fix the behavior observed in Phase 5B where Qwen3-4B either stayed in long thinking mode or answered in generic chat style without `<answer>` tags.

## Code Changes

- Added `src/lightningsearch_rl/sft_warmup.py`.
- Added CLI command `export-sft-warmup`.
- Added tests:
  - `tests/test_sft_warmup.py`
  - `tests/test_cli.py::test_export_sft_warmup_cli_writes_gold_evidence_conversations`

The exporter uses:

- `examples.jsonl` for question, gold answer, and `gold_doc_ids`.
- `index.json` as the corpus source.
- Gold passages selected by doc ID, not BM25 ranking.

## Remote Export Command

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-sft-warmup \
  --examples /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/examples.jsonl \
  --index /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/index.json \
  --out-dir /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-warmup-gold
```

## Artifacts

- SFT rows: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-warmup-gold/sft_warmup.jsonl`
- Traces: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-warmup-gold/traces.jsonl`
- Summary: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-warmup-gold/summary.json`

## Validation

Remote summary:

```json
{
  "answer_tag_rate": 1.0,
  "avg_gold_evidence_count": 2.0,
  "example_count": 500,
  "gold_evidence_coverage": 1.0,
  "non_empty_answer_rate": 1.0,
  "sft_rows": 500
}
```

Additional checks:

```text
row_count: 500
empty_answer_rows: 0
missing_answer_tag: 0
missing_observation: 0
missing_gold_evidence: 0
```

Sample row:

```text
<think>I should search for evidence.</think>
<search>Which award did the organization founded by Dr. Elena Voss receive in 2021?</search>
<observation>
[1] Dr. Elena Voss: Dr. Elena Voss founded the Global Health Initiative in 2012.
[2] Global Health Initiative: The Global Health Initiative won the Nobel Peace Prize in 2021.
</observation>
<think>The retrieved evidence supports the answer.</think>
<answer>Nobel Peace Prize</answer>
```

## Tests

Local:

```text
python -m pytest tests\test_sft_warmup.py tests\test_sft.py tests\test_cli.py tests\test_grpo.py tests\test_verl_reward.py tests\test_verl_smoke.py -q
21 passed
```

Remote:

```text
PYTHONNOUSERSITE=1 python -m pytest tests/test_sft_warmup.py tests/test_sft.py tests/test_cli.py tests/test_grpo.py tests/test_verl_reward.py tests/test_verl_smoke.py -q
21 passed
```

## Analysis

This dataset is suitable for format warmup because every assistant target contains the full tag sequence and a non-empty answer. It is intentionally easier than retrieval-agent RL because it uses gold evidence directly. That is the right behavior for warmup: teach the output schema first, then return to rollout / GRPO with the model less likely to spend all tokens in Qwen3 thinking mode or omit `<answer>`.

## Next Step

Prepare a tiny SFT training launcher that consumes `sft_warmup.jsonl`, runs a short Qwen3-4B SFT job, and then repeats generation inspection before returning to GRPO.
