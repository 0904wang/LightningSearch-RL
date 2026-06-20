# Phase 5C 4-GPU SFT Warmup Success

Date: 2026-06-15

## Goal

Run a short full-parameter Qwen3-4B SFT warmup on the gold-evidence
`think/search/observation/answer` dataset. This run retries the failed 1-GPU
attempt with 4 GPUs so FSDP can shard model and optimizer state.

## Command

```bash
tmux new-session -d -s lightningsearch-20260615-sft-warmup-tiny-4gpu "bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train-sft-warmup --config configs/experiments/phase5c_sft_warmup_tiny_4gpu.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny-4gpu --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny-4gpu --print-command 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny-4gpu.log'"
```

## Runtime

```text
config: /data/wzl/LightningSearch-RL/repo/configs/experiments/phase5c_sft_warmup_tiny_4gpu.yaml
session: lightningsearch-20260615-sft-warmup-tiny-4gpu
gpus: CUDA_VISIBLE_DEVICES=0,1,2,7
model: /data/wzl/LightningSearch-RL/models/Qwen/Qwen3-4B
train rows: 480
val rows: 20
global batch: 4
micro batch per GPU: 1
steps: 20
lr: 1e-5
```

## Artifacts

- Log: `/data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny-4gpu.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny-4gpu`
- Checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny-4gpu/global_step_20`
- Checkpoint tracker: `/data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny-4gpu/latest_checkpointed_iteration.txt`

Checkpoint contents:

```text
latest_checkpointed_iteration: 20
checkpoint size: 24G
files: 26
model shards: model_world_size_4_rank_0.pt ... model_world_size_4_rank_3.pt
optimizer shards: optim_world_size_4_rank_0.pt ... optim_world_size_4_rank_3.pt
huggingface tokenizer/config folder saved
```

## Metrics

```text
step:20
train/loss: 0.43445271253585815
train/grad_norm: 18.375
train/lr: 1e-05
train/global_tokens: 421
train/total_tokens(B): 8.34e-06
val/loss: 0.4961507022380829
max_memory_allocated_gb: 10.225692749023438
max_memory_reserved_gb: 14.880859375
```

## Validation

Post-run checks:

```text
tmux: no active session
latest checkpoint: 20
tail check: no OutOfMemory, ChildFailedError, or Traceback
GPU memory released after completion
```

Final GPU state:

```text
0, 3506 MiB, 32607 MiB
1, 3505 MiB, 32607 MiB
2, 3507 MiB, 32607 MiB
3, 25985 MiB, 32607 MiB
4, 26101 MiB, 32607 MiB
5, 3493 MiB, 32607 MiB
6, 3505 MiB, 32607 MiB
7, 18 MiB, 32607 MiB
```

## Analysis

The 4-GPU run resolves the 1-GPU OOM root cause. In the failed run, FSDP fell
back to `NO_SHARD`; here FSDP sharded across four ranks and kept peak allocated
memory around 10.23 GiB per selected GPU. The loss dropped quickly from 6.75 at
step 1 to 0.43 at step 20, and validation loss was 0.496 on the 20-row split.

This checkpoint is suitable for the next phase: inspect generation behavior from
`global_step_20`, then decide whether to run a longer SFT warmup or use it as the
initial policy for the next GRPO smoke.
