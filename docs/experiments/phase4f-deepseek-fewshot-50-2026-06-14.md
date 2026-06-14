# Phase 4F DeepSeek Few-Shot Pilot 50

## Goal

Test whether a few-shot chain-schema example in the synthesis prompt improves valid rate compared to Phase 4D/4E.

## Code

- GitHub commit: \8d0b696\
- Conda env: \/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl\
- Session: \lightningsearch-20260614-phase4f-deepseek-fewshot-50\
- Log: \/data/wzl/LightningSearch-RL/logs/phase4f-deepseek-fewshot-50.log\
- Results: \/data/wzl/LightningSearch-RL/results/phase4f-deepseek-fewshot-50\

## Command

\\\ash
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-validated-data \
  --raw .../raw.jsonl --valid .../valid.jsonl --rejects .../rejects.jsonl \
  --target-valid 50 --topics awards,archives,research institutes,scientific discoveries,academic journals \
  --concurrency 50 --batch-size 50 --max-attempts 500 --seed 7000 \
  --require-chain-schema --few-shot-chain-schema
\\\

## Result

\\\json
{
  "generated": 212, "requested": 212, "valid_count": 50, "reject_count": 162,
  "few_shot_chain_schema": true, "stopped_reason": "target_valid_reached"
}
\\\

## Reject Reasons

\\\	ext
intermediate entity missing from hop2: 54
intermediate entity missing from hop1: 51
answer equals context title: 31
final answer leaks in hop1: 10
final answer missing from hop2: 7
supporting_facts must cover at least two titles: 7
non-ascii text detected: 1
\\\

## Valid Quality

- answer-as-title: 0
- answer-in-question: 0
- exactly one supporting sentence hits: 50/50

## Comparison

| Phase | Generated | Valid | Rate | Top Reject |
|-------|-----------|-------|------|------------|
| 4D (schema) | 250 | 10 | 4.0% | missing intermediate |
| 4E (repair) | 500 | 33 | 6.6% | missing intermediate |
| **4F (few-shot)** | **212** | **50** | **23.6%** | missing intermediate |

The few-shot example eliminated nearly all non-ASCII failures (35/58 → 1) and dramatically improved valid rate. The answer-equals-title reject rate increased, suggesting the example may need to emphasize title-answer distinction more explicitly in Phase 4G.
