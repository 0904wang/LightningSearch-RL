# Phase 5B Tiny 4-GPU GRPO Smoke: offload5

Date: 2026-06-15

## Goal

Run the first tiny `verl` GRPO training smoke for LightningSearch-RL on 4 RTX 5090 GPUs with Qwen3-4B, using offline rollout data and the custom `lightningsearch_rl` reward hook.

## Remote Runtime

- Remote workspace: `/data/wzl/LightningSearch-RL`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Sync method: narrow file sync from local workspace, because the remote repo path is not currently a git checkout.
- Local workspace: `D:\resume\Agent RL`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Python: 3.10.20
- Model: `/data/wzl/LightningSearch-RL/models/Qwen/Qwen3-4B`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,7`

## Effective Config

- Config: `configs/experiments/phase5b_tiny_grpo_smoke_4gpu.yaml`
- Rollouts: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo/rollouts.jsonl`
- Train samples: 4
- Validation samples: 1
- Training steps: 1
- `data.train_batch_size`: 4
- `actor_rollout_ref.actor.ppo_mini_batch_size`: 4
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: 1
- `actor_rollout_ref.actor.fsdp_config.model_dtype`: `bfloat16`
- `actor_rollout_ref.actor.fsdp_config.param_offload`: `True`
- `actor_rollout_ref.actor.fsdp_config.optimizer_offload`: `True`
- `actor_rollout_ref.rollout.name`: `vllm`
- `actor_rollout_ref.rollout.gpu_memory_utilization`: 0.25
- `actor_rollout_ref.rollout.max_model_len`: 768
- `actor_rollout_ref.rollout.max_num_batched_tokens`: 1024
- `actor_rollout_ref.rollout.max_num_seqs`: 4
- `actor_rollout_ref.rollout.agent.num_workers`: 4
- `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes`: 512
- `trainer.save_freq`: -1
- `trainer.test_freq`: -1

## Commands

Remote tests:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m pytest tests/test_verl_smoke.py tests/test_verl_reward.py -q
```

Dry run:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train \
  --config configs/experiments/phase5b_tiny_grpo_smoke_4gpu.yaml \
  --output-dir /data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke-4gpu-vllm-offload5 \
  --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke-4gpu-vllm-offload5 \
  --dry-run --print-command
```

Launch:

```bash
tmux new-session -d -s lightningsearch-20260615-phase5b-tiny-grpo-smoke-4gpu-vllm-offload5 "bash -lc 'set -eo pipefail; cd /data/wzl/LightningSearch-RL/repo; source /home/user/anaconda3/etc/profile.d/conda.sh; conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl; CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train --config configs/experiments/phase5b_tiny_grpo_smoke_4gpu.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke-4gpu-vllm-offload5 --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke-4gpu-vllm-offload5 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5b-tiny-grpo-smoke-4gpu-vllm-offload5.log'"
```

## Artifacts

- Log: `/data/wzl/LightningSearch-RL/logs/phase5b-tiny-grpo-smoke-4gpu-vllm-offload5.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke-4gpu-vllm-offload5`
- Checkpoints: `/data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke-4gpu-vllm-offload5`
- No model checkpoint was expected because `trainer.save_freq=-1`.

## Result

Status: completed the 1-step GRPO smoke.

Key raw metrics from the log:

```text
Initial validation:
val-core/lightningsearch_rl/reward/mean@1: 0.9700000286102295
val-aux/lightningsearch_rl/score/mean@1: 0.97
val-aux/lightningsearch_rl/answer_reward/mean@1: 1.0
val-aux/lightningsearch_rl/format_reward/mean@1: 0.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.03
val-aux/num_turns/mean: 2.0

Training:
training/global_step: 1
critic/score/mean: 0.9700000286102295
critic/rewards/mean: 0.9700000286102295
actor/loss: -0.9699990749359131
actor/grad_norm: 6.65625
actor/perf/max_memory_allocated_gb: 9.381478309631348
actor/perf/max_memory_reserved_gb: 13.560546875
response_length/mean: 256.0
response/aborted_ratio: 0.0
timing_s/gen: 3.400666925124824
timing_s/update_actor: 5.244312927592546
timing_s/update_weights: 1.7464691791683435
timing_s/step: 20.30608448619023
perf/throughput: 13.825905244850754
```

The run finished with GPUs released back to idle. The log includes shutdown-time warnings from `torchdata` / multiprocessing `resource_tracker` and vLLM engine client shutdown after the training step had completed:

```text
RuntimeError: DataLoader worker (...) is killed by signal: Killed.
resource_tracker: process died unexpectedly, relaunching. Some resources might leak.
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

These appeared after `Training Progress: 100%|...| 1/1`, `training/global_step:1`, and `Final validation metrics: None`. Treat them as residual cleanup warnings for now, not as evidence that the smoke step failed.

## Failure History and Fixes

- `hf` rollout failed because current `verl` async rollout registry does not support `hf`; switched to `vllm`.
- Initial vLLM default run hung / crashed around vLLM server startup; constrained vLLM with small `max_model_len`, `max_num_batched_tokens`, `max_num_seqs`, eager mode, and disabled chunked prefill / prefix caching.
- A launch script with `set -u` broke conda activation on `SYS_SYSROOT`; launch scripts now use `set -eo pipefail`.
- `lightningsearch_rl` data source needed a custom reward hook; added `src/lightningsearch_rl/verl_reward.py`.
- No-offload vLLM reward run reached validation but OOMed during `wake_up(weights)`; enabled actor bf16, FSDP param/optimizer offload, and reduced update weights bucket to 512MB.
- `gpu_memory_utilization=0.18` was too low for vLLM KV cache; raised to 0.25.
- `train_batch_size=4` with default 8 agent workers failed `DataProto` equal chunk assertion; set `actor_rollout_ref.rollout.agent.num_workers=4`.
- `ppo_mini_batch_size=2` failed because data parallel size is 4; set `ppo_mini_batch_size=4`.

## Analysis

This confirms the minimal end-to-end path works:

1. prepare rollouts into verl parquet rows,
2. launch Qwen3-4B actor with GRPO,
3. start async vLLM rollout servers,
4. run custom reward computation for `lightningsearch_rl`,
5. complete actor update,
6. sync updated weights back to rollout without CUDA OOM.

The current smoke data is still toy-scale and reward is mostly a plumbing validation. The `format_reward` is 0.0 in validation because generated text did not include a valid `<answer>` tag, while exact answer reward is positive due current reward extraction / ground truth setup. This should be revisited before interpreting reward quality.

## Next Steps

- Add an explicit smoke config variant with `train_batch_size`, `ppo_mini_batch_size`, and `agent.num_workers` documented as divisibility constraints.
- Decide whether to keep vLLM cleanup warnings as known benign behavior or reduce dataloader worker count / shutdown complexity if they become noisy.
- Run a slightly larger 4-GPU GRPO smoke with 8 to 16 train samples after confirming generated samples and reward components are meaningful.
- Before scaling, inspect generated responses from the rollout path because `response_length/clip_ratio=1.0` indicates all responses hit the 256-token cap.
