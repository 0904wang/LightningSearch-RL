# Phase 5P Soft Answer GRPO Failed: Reward Extra Info Schema

## Goal

Run the Phase 5P 5-step GRPO smoke with soft answer reward on the 100
environment-transition split from Phase 5K.

## Launch

Session:

```text
lightningsearch-20260616-phase5p-softanswer-100x5
```

Command:

```bash
tmux new-session -d -s lightningsearch-20260616-phase5p-softanswer-100x5 "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5p_env_transition_grpo_4gpu_100x5_softanswer.sh'"
```

Paths:

```text
log: /data/wzl/LightningSearch-RL/logs/phase5p-env-transition-grpo-4gpu-100x5-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5p-env-transition-grpo-4gpu-100x5-softanswer
reward_dump: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer/reward_dump.jsonl
```

## Status

Failed before the first training step. The tmux session exited and no
`metrics_summary.json`, `batch_diagnostics.json`, or `reward_dump_summary.json`
was produced. `reward_dump.jsonl` contains 20 validation rows written before the
crash.

## Raw Error Excerpt

```text
ray.exceptions.RayTaskError(KeyError): ray::AgentLoopWorker.generate_sequences()
  File ".../verl/experimental/agent_loop/agent_loop.py", line 982, in _postprocess
    non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])
  File ".../verl/experimental/agent_loop/agent_loop.py", line 982, in <listcomp>
    non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])
KeyError: 'answer_reward_type'
```

## Root Cause

The online reward function returned inconsistent extra-info keys across stages:

- answer-stage rewards included the string key `answer_reward_type`
- search-stage rewards did not include that key

verl's AgentLoop aggregates every key across all `reward_extra_infos`. Mixed
answer/search batches therefore failed when one item exposed
`answer_reward_type` and another did not.

## Fix

Keep the reward dict returned to verl as a stable numeric schema only:

```text
score
answer_reward
answer_exact_match
answer_token_f1
answer_containment_match
search_reward
format_reward
search_cost
```

Keep string `answer_reward_type` only in the diagnostic reward dump JSONL.

## Validation

Local regression before fix:

```text
2 failed, 6 passed in tests/test_verl_reward.py
```

Local validation after fix:

```text
tests/test_verl_reward.py tests/test_verl_reward_dump_diagnostics.py: 11 passed
full suite: 131 passed
```

## Retry Plan

Use isolated retry paths so the failed `reward_dump.jsonl` is preserved:

```text
config: configs/experiments/phase5p_env_transition_grpo_4gpu_100x5_softanswer_retry1.yaml
script: scripts/remote/phase5p_env_transition_grpo_4gpu_100x5_softanswer_retry1.sh
log: /data/wzl/LightningSearch-RL/logs/phase5p-env-transition-grpo-4gpu-100x5-softanswer-retry1.log
results: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer-retry1
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5p-env-transition-grpo-4gpu-100x5-softanswer-retry1
reward_dump: /data/wzl/LightningSearch-RL/results/phase5p-env-transition-grpo-4gpu-100x5-softanswer-retry1/reward_dump.jsonl
```
