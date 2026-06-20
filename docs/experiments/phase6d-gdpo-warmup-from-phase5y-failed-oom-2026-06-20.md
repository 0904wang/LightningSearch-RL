# Phase 6D GDPO Warmup Failed OOM

Date: 2026-06-20 Asia/Shanghai

## Goal

Run a short verl GDPO warmup before a follow-up GRPO warm-start experiment. This used the Phase 5Y variance-filtered hard50 transition slice and the Phase 5D SFT checkpoint.

## Launch

Approved launch command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260620-phase6d-gdpo-warmup -c /data/wzl/LightningSearch-RL/repo "env CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase6d_gdpo_warmup_from_phase5y.sh"'
```

## Paths

- Repo: `/data/wzl/LightningSearch-RL/repo`
- Branch / commit: `main`, `ec1ba8c`
- Env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Config: `/data/wzl/LightningSearch-RL/repo/configs/experiments/phase6d_gdpo_warmup_from_phase5y.yaml`
- Input transitions: `/data/wzl/LightningSearch-RL/results/phase5y-env-transitions-variance-rankreward-100src/transitions.jsonl`
- Result dir: `/data/wzl/LightningSearch-RL/results/phase6d-gdpo-warmup-from-phase5y`
- Checkpoint dir: `/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6d-gdpo-warmup-from-phase5y.log`

## Settings

- Model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- GPUs: `CUDA_VISIBLE_DEVICES=4,5,6,7`
- Train samples: `112`
- Val samples: `14`
- Train batch size: `4`
- Rollouts per prompt: `4`
- Max prompt length: `384`
- Max response length: `64`
- Advantage estimator: `gdpo`
- GDPO reward keys: `["search_reward", "format_reward"]`
- Reward manager: `gdpo`
- Planned training steps: `28`
- Save frequency: `28`

## Status

The run failed before the first training step.

- `completed`: `false`
- `checkpoint_written`: `false`
- `reward_dump_written`: `false`
- `tmux_session_after_failure`: none
- Failure type: CUDA OOM during initial FSDP actor weight sync to vLLM.

## Raw Evidence

The log confirmed that verl accepted the GDPO settings and built the train / val data:

```text
[validate_config] All configuration checks passed successfully!
Using dataset class: RLHFDataset
dataset len: 112
Using dataset class: RLHFDataset
dataset len: 14
Size of train dataloader: 28, Size of val dataloader: 1
Total training steps: 28
Training from scratch
```

The failure happened while calling `checkpoint_manager.update_weights` before training:

```text
ray.exceptions.RayTaskError(OutOfMemoryError): ray::WorkerDict.actor_rollout_update_weights()
...
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 742.00 MiB.
GPU 0 has a total capacity of 31.36 GiB of which 301.06 MiB is free.
Process 3573322 has 20.08 GiB memory in use.
Process 3637988 has 8.23 GiB memory in use.
```

Post-failure GPU process check showed GPU 7 was occupied by another job:

```text
7, GPU-4d94047d-7c6b-7c5d-a4b8-d4d1d5fdb201, 20583 MiB, 32607 MiB
GPU-4d94047d-7c6b-7c5d-a4b8-d4d1d5fdb201, 3573322, 20560, /home/user/anaconda3/envs/spt_paper/bin/python
```

No reward dump was written:

```text
reward_dump: missing
```

## Analysis

This failure is primarily a resource selection / contention issue, not evidence that the GDPO configuration is invalid. The run reached verl config validation, loaded the 112 / 14 datasets, initialized FSDP workers, and started vLLM servers. It then failed during the initial actor-to-rollout weight update because one of the selected physical GPUs became occupied by an external process using about 20.5 GiB.

The exact Phase 5Y GRPO run used the same model size, 4 GPUs, `rollout_n=4`, and similar memory parameters on GPUs `0,1,2,3`, and completed 28 steps. The safest retry is therefore to avoid GPU 7 and launch the same Phase 6D config on currently free GPUs `0,1,2,3` after a fresh GPU check.

## Next Step

Retry Phase 6D GDPO warmup on `CUDA_VISIBLE_DEVICES=0,1,2,3`, with the same config and paths, after reporting the exact launch command and receiving approval. If it OOMs again without external GPU contention, the next hypothesis is that GDPO reward-loop placement increases memory pressure; then reduce `rollout_gpu_memory_utilization` from `0.30` to `0.20` or reduce `rollout_max_num_seqs` from `16` to `8`.
