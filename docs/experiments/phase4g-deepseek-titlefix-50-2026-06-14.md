# Phase 4G DeepSeek Title-Fix Pilot 50

## Goal

Validate whether adding explicit answer/context-title separation to the few-shot prompt reduces nswer equals context title rejects and improves valid rate.

## Code

- GitHub commit: 121eb55
- Session: lightningsearch-20260614-phase4g-deepseek-titlefix-50
- Results: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-50
- Log: /data/wzl/LightningSearch-RL/logs/phase4g-deepseek-titlefix-50.log

## Result

`json
{
  "generated": 140,
  "valid_count": 50,
  "reject_count": 90,
  "few_shot_chain_schema": true,
  "stopped_reason": "target_valid_reached"
}
`

## Reject Reasons

`	ext
intermediate entity missing from hop2: 37
answer equals context title: 18
intermediate entity missing from hop1: 18
supporting_facts must cover at least two titles: 9
final answer missing from hop2: 3
non-ascii text detected: 2
final answer leaks in hop1: 2
supporting fact missing from context: 1
`

## Valid Quality

- answer-as-title: 0
- answer-in-question: 0
- exactly one supporting sentence hits: 50/50

## Comparison

| Phase | Generated | Valid | Rate | answer=title rejects |
|-------|-----------|-------|------|----------------------|
| 4F | 212 | 50 | 23.6% | 31 |
| 4G | 140 | 50 | 35.7% | 18 |

Phase 4G improved throughput and reduced title-answer errors. The remaining bottleneck is still intermediate entity exact matching across hop1/hop2.
