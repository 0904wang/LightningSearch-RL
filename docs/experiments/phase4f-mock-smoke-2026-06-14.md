# Phase 4F Few-Shot Mock Smoke

## Goal

Verify that Phase 4F few-shot chain-schema prompt produces valid strict rows end-to-end.

## Environment

- GitHub commit: 9a3b06
- Conda env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
- Remote results: /data/wzl/LightningSearch-RL/results/phase4f-mock-smoke

## Command

--require-chain-schema --few-shot-chain-schema --mock

## Result

5/5 valid, 0 rejects, ew_shot_chain_schema: true in summary.

GRPO export: 5 rollouts, 10 transitions, avg_reward 1.37.

## Local/Remote Tests

- Local pytest: 57 passed
- Remote pytest: 57 passed
