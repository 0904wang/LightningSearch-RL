# Phase 5M: Env Transition GRPO 4-GPU Failed KV Cache Attempt

## Goal

Run the first 4-GPU `verl` GRPO smoke that consumes Phase 5L environment
transition records rather than earlier gold-answer rollout rows.

## Runtime

date: 2026-06-16
session: `lightningsearch-20260616-phase5m-env-transition-grpo`
remote repo: `/data/wzl/LightningSearch-RL/repo`
conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
GPU selection: `CUDA_VISIBLE_DEVICES=0,1,2,5`
config: `configs/experiments/phase5m_env_transition_grpo_4gpu.yaml`

## Inputs

transitions:

```text
/data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl
```

dry-run data:

```text
/data/wzl/LightningSearch-RL/results/phase5m-env-transition-grpo-4gpu/data/train.parquet
/data/wzl/LightningSearch-RL/results/phase5m-env-transition-grpo-4gpu/data/val.parquet
```

Rows:

```text
train_rows: 8
val_rows: 4
```

## Launch Command

```bash
tmux new-session -d -s lightningsearch-20260616-phase5m-env-transition-grpo "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5m_env_transition_grpo_4gpu.sh'"
```

## Failure

The run entered `verl.trainer.main_ppo`, read the train and validation parquet
files, initialized `RayPPOTrainer`, and then failed while starting vLLM rollout
servers.

Root error:

```text
ValueError: To serve at least one request with the models's max seq len (1280),
(0.18 GiB KV cache is needed, which is larger than the available KV cache
memory (0.10 GiB). Based on the available memory, the estimated maximum model
length is 736. Try increasing `gpu_memory_utilization` or decreasing
`max_model_len` when initializing the engine.
```

The tmux session exited after the failure. GPU memory returned to the pre-run
state.

## Root Cause

`phase5m_env_transition_grpo_4gpu.yaml` used:

```yaml
max_prompt_length: 1024
rollout_max_model_len: 1280
rollout_max_num_batched_tokens: 1536
rollout_gpu_memory_utilization: 0.25
```

This was too large for the colocated 4-GPU GRPO setup. The actual Phase 5M
prompt lengths are much shorter:

```text
max_prompt_tokens=301
max_prompt_plus_response_64=365
```

The large vLLM context setting, not the data itself, caused the KV cache failure.

## Prepared Fix

Created a separate low-context retry config rather than overwriting the failed
attempt:

```text
configs/experiments/phase5m_env_transition_grpo_4gpu_lowlen.yaml
scripts/remote/phase5m_env_transition_grpo_4gpu_lowlen.sh
```

Low-context settings:

```yaml
max_prompt_length: 384
max_response_length: 64
rollout_max_model_len: 512
rollout_max_num_batched_tokens: 768
rollout_gpu_memory_utilization: 0.25
```

Validation:

```text
local tests: 114 passed in 3.03s
remote lowlen config test: 1 passed in 0.27s
remote lowlen dry-run: parquet_written=true, train_rows=8, val_rows=4
```

## Next Step

Launch the low-context retry only after user approval because it changes runtime
settings from the previously approved command.
