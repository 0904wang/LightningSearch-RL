# Phase 5C SFT Warmup Tiny Failed OOM

Date: 2026-06-15

## Goal

Launch a short 1-GPU Qwen3-4B full-parameter SFT warmup with 480 train rows and
20 validation rows from the gold-evidence `sft_warmup.jsonl` dataset.

## Launch

Session:

```text
lightningsearch-20260615-sft-warmup-tiny
```

Command:

```bash
tmux new-session -d -s lightningsearch-20260615-sft-warmup-tiny "bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train-sft-warmup --config configs/experiments/phase5c_sft_warmup_tiny.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny --print-command 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny.log'"
```

## Artifacts

- Log: `/data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny`
- Checkpoint dir: `/data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny`

No checkpoint was produced.

## Status

Failed at the first optimizer step with CUDA OOM. The tmux session exited and
GPU memory was released.

Post-run GPU state:

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

## Log Evidence

```text
Qwen3ForCausalLM contains 4.02B parameters
FSDP is switching to use `NO_SHARD` instead of ShardingStrategy.FULL_SHARD since the world size is 1.
After FSDP, memory allocated (GB): 7.49, memory reserved (GB): 8.47, device memory used/total (GB): 9.59/31.36
Total steps: 20, num_warmup_steps: 0
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 194.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 50.12 MiB is free.
```

The `GPU 0` in the error refers to logical device 0 inside
`CUDA_VISIBLE_DEVICES=7`.

## Analysis

The dataset, tokenizer path, model path, FSDP initialization, and trainer entry
were all valid. The failure happened when Adam initialized optimizer state on the
first training step. With `n_gpus_per_node=1`, FSDP falls back to `NO_SHARD`, so
the full Qwen3-4B parameters, gradients, and optimizer state are effectively
single-device. That does not fit in a 32 GiB 5090 for full-parameter SFT.

## Recommended Retry

Use a 2-GPU full-parameter SFT smoke so FSDP can actually shard parameters and
optimizer state:

- `CUDA_VISIBLE_DEVICES=6,7`
- `n_gpus_per_node=2`
- keep `train_batch_size=2`
- keep `micro_batch_size_per_gpu=1`
- keep `total_training_steps=20`
- use a new result/checkpoint/log path such as `phase5c-sft-warmup-tiny-2gpu`

Alternative: use LoRA on 1 GPU. That is lower memory, but it changes the
downstream checkpoint format and is less directly aligned with the planned GRPO
warm-start path.

## 2-GPU Retry Prepared

Added and dry-ran:

```text
configs/experiments/phase5c_sft_warmup_tiny_2gpu.yaml
```

Remote tests:

```text
27 passed in 0.46s
```

Dry-run output:

```text
/data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny-2gpu
```

Planned retry:

```text
session: lightningsearch-20260615-sft-warmup-tiny-2gpu
gpus: CUDA_VISIBLE_DEVICES=6,7
log: /data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny-2gpu.log
results: /data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny-2gpu
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny-2gpu
```

Planned launch command:

```bash
tmux new-session -d -s lightningsearch-20260615-sft-warmup-tiny-2gpu "bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train-sft-warmup --config configs/experiments/phase5c_sft_warmup_tiny_2gpu.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny-2gpu --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny-2gpu --print-command 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny-2gpu.log'"
```

## 4-GPU Retry Prepared

The user requested a direct 4-GPU retry. Added and dry-ran:

```text
configs/experiments/phase5c_sft_warmup_tiny_4gpu.yaml
```

Remote tests:

```text
28 passed in 0.49s
```

GPU state before launch report:

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

Planned retry:

```text
session: lightningsearch-20260615-sft-warmup-tiny-4gpu
gpus: CUDA_VISIBLE_DEVICES=0,1,2,7
log: /data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny-4gpu.log
results: /data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny-4gpu
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny-4gpu
```

Planned launch command:

```bash
tmux new-session -d -s lightningsearch-20260615-sft-warmup-tiny-4gpu "bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train-sft-warmup --config configs/experiments/phase5c_sft_warmup_tiny_4gpu.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny-4gpu --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny-4gpu --print-command 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny-4gpu.log'"
```
