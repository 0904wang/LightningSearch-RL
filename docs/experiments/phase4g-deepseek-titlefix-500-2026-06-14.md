# Phase 4G DeepSeek Title-Fix 500 Dataset

## Goal

Generate the first larger strict synthetic dataset for GRPO experiments using the Phase 4G few-shot prompt.

## Code And Environment

- GitHub main at launch: e5d3eeb
- Remote repo: /data/wzl/LightningSearch-RL/repo
- Env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
- Session: lightningsearch-20260614-phase4g-deepseek-titlefix-500
- Results: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500
- Log: /data/wzl/LightningSearch-RL/logs/phase4g-deepseek-titlefix-500.log

## Synthesis Command

`ash
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-validated-data \
  --raw /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/raw.jsonl \
  --valid /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/valid.jsonl \
  --rejects /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/rejects.jsonl \
  --target-valid 500 \
  --topics awards,archives,research\ institutes,scientific\ discoveries,academic\ journals \
  --concurrency 50 \
  --batch-size 50 \
  --max-attempts 2000 \
  --seed 9000 \
  --summary /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/synthesis_summary.json \
  --require-chain-schema \
  --few-shot-chain-schema
`

## Synthesis Result

`json
{
  "generated": 1547,
  "requested": 1547,
  "valid_count": 500,
  "reject_count": 1047,
  "api_failed": 0,
  "few_shot_chain_schema": true,
  "stopped_reason": "target_valid_reached"
}
`

Valid rate: 32.3%.

## Quality Checks

`json
{
  "valid_count": 500,
  "valid_answer_title": 0,
  "valid_answer_question": 0,
  "valid_answer_support_sentence_hits": {"1": 500}
}
`

## Reject Reasons

`	ext
intermediate entity missing from hop2: 353
intermediate entity missing from hop1: 320
answer equals context title: 179
supporting_facts must cover at least two titles: 59
final answer leaks in hop1: 57
final answer missing from hop2: 44
non-ascii text detected: 17
chain_schema does not match supporting_facts: 8
chain_schema final_answer does not match answer: 5
supporting fact missing from context: 5
`

## GRPO Export

`json
{
  "example_count": 500,
  "rollout_count": 500,
  "transition_count": 1000,
  "avg_reward": 0.325,
  "avg_search_count": 1.0,
  "top_k": 2
}
`

Artifacts:

- /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/corpus.jsonl
- /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/examples.jsonl
- /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/index.json
- /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo/rollouts.jsonl
- /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo/transitions.jsonl
- /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo/reward_records.jsonl

## Analysis

This is the first useful training-scale synthetic set for the project. It is still small for final training, but large enough for pipeline-level GRPO smoke tests and reward/debug analysis. The valid set remains strict-clean, while the remaining rejects show that intermediate exact matching remains the main generator weakness.
