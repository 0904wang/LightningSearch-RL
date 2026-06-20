# Phase 5D DocID-Fixed Turn-Level 4-GPU SFT Warmup Success

Date: 2026-06-15

## Goal

Run a short turn-level SFT warmup on the regenerated doc-id-consistent
synthetic corpus. This is the corrected Phase 5D path: the old turn-level SFT
run used data generated before row-scoped document IDs were enforced, while
this run consumes the `phase4g-deepseek-titlefix-500-docidfix` artifacts.

The training target is still turn-level agent behavior:

- assistant emits one `<search>...</search>` action
- environment/user provides `<observation>...</observation>`
- assistant emits one `<answer>...</answer>` action

## Command

```bash
tmux new-session -d -s lightningsearch-20260615-sft-turns-docidfix-4gpu "bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train-sft-warmup --config configs/experiments/phase5d_sft_turns_docidfix_4gpu.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu --print-command 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5d-sft-turns-docidfix-4gpu.log'"
```

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
repo sync: not-a-git-repo narrow sync from local workspace
local branch: master
local base commit: 44493db
config: /data/wzl/LightningSearch-RL/repo/configs/experiments/phase5d_sft_turns_docidfix_4gpu.yaml
session: lightningsearch-20260615-sft-turns-docidfix-4gpu
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpus: CUDA_VISIBLE_DEVICES=0,1,2,5
model: /data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny-4gpu/hf_merged_global_step_20
sft data: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
train rows: 480
val rows: 20
global batch: 4
micro batch per GPU: 1
max length: 1024
max token len per GPU: 2048
steps: 40
lr: 1e-5
seed: 19
```

GPU `7` was occupied before launch, so the approved 4-GPU set was changed from
`1,2,5,7` to `0,1,2,5`.

## Artifacts

- Log: `/data/wzl/LightningSearch-RL/logs/phase5d-sft-turns-docidfix-4gpu.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu`
- Train parquet: `/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/data/train.parquet`
- Val parquet: `/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/data/val.parquet`
- Checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/global_step_40`
- Checkpoint tracker: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/latest_checkpointed_iteration.txt`
- Remote record: `/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/EXPERIMENT_RECORD.md`

Checkpoint contents:

```text
latest_checkpointed_iteration: 40
checkpoint size: 24G
model shards: model_world_size_4_rank_0.pt ... model_world_size_4_rank_3.pt
optimizer shards: optim_world_size_4_rank_0.pt ... optim_world_size_4_rank_3.pt
extra state shards: extra_state_world_size_4_rank_0.pt ... extra_state_world_size_4_rank_3.pt
dataloader state shards: data_0.pt ... data_3.pt
huggingface tokenizer/config folder saved
```

Result files:

```text
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/data/train.jsonl
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/data/train.parquet
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/data/val.jsonl
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/data/val.parquet
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/dry_run_summary.json
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/launch_command.txt
/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu/manifest.json
```

## Metrics

Final raw log excerpt:

```text
step:40 - perf/max_memory_allocated_gb:10.225693225860596
step:40 - perf/max_memory_reserved_gb:14.703125
step:40 - perf/cpu_memory_used_gb:101.98000717163086
step:40 - train/loss:5.235062781139277e-05
step:40 - train/grad_norm:0.017822265625
step:40 - train/lr:1e-05
step:40 - train/global_tokens:753
step:40 - train/total_tokens(B):2.9561e-05
step:40 - val/loss:0.0033128075301647186
Final validation metrics: {'val/loss': 0.0033128075301647186}
```

The loss is much lower than the previous non-docidfix Phase 5D SFT run
(`val/loss:0.06891687214374542`). This is plausible because the run starts from
the Phase 5C merged checkpoint and trains on a compact, highly templated
two-action turn-level dataset. It should not be treated as evidence of
end-to-end task success until generation inspection confirms behavior.

## Validation

Pre-launch checks:

```text
local tests: python -m pytest tests\test_verl_sft_warmup.py -q -> 8 passed
remote tests: python -m pytest tests/test_verl_sft_warmup.py -q -> 8 passed
dry-run command produced CUDA_VISIBLE_DEVICES=0,1,2,5 and expected verl SFT args
```

Post-run checks:

```text
local full tests: python -m pytest -q -> 96 passed
fatal log keyword scan: 0 matches for Traceback, ChildFailedError, OutOfMemory, CUDA error, RuntimeError, Exception
tmux session: no active lightningsearch-20260615-sft-turns-docidfix-4gpu session after completion
latest checkpoint iteration: 40
checkpoint size: 24G
log size: 41235 bytes
```

Final GPU state after completion:

```text
0, 3506 MiB, 32607 MiB
1, 3505 MiB, 32607 MiB
2, 3507 MiB, 32607 MiB
3, 25985 MiB, 32607 MiB
4, 26101 MiB, 32607 MiB
5, 3493 MiB, 32607 MiB
6, 3505 MiB, 32607 MiB
7, 16159 MiB, 32607 MiB
```

## Warnings

Observed warnings were non-fatal:

```text
MultiTurnSFTDataset apply_chat_template to each turn separately and concat input_ids as a whole sequence.
Set ignore_input_ids_mismatch=True to ignore input_ids mismatch and use the concatenated input_ids as the final input_ids.
The tokenizer ... incorrect regex pattern ... set fix_mistral_regex=True.
Warning: Failed to set NUMA affinity: libnuma.so: cannot open shared object file.
torch_dtype is deprecated; use dtype instead.
```

The config explicitly sets `data.ignore_input_ids_mismatch=True`, matching the
verl warning recommendation for this multi-turn dataset path.

## Analysis

This run completes the corrected turn-level SFT warmup on docid-fixed data. It
uses row-scoped document IDs from the regenerated Phase 4G corpus, so the
observation passages are no longer vulnerable to cross-example title collisions.
The Phase 5D data consistency guard also requires selected gold evidence to
contain the final answer before SFT examples are exported.

The resulting checkpoint is a better candidate than the previous Phase 5D
checkpoint for the next policy initialization, but it is still only an SFT
checkpoint. The next required step is to convert or merge the FSDP checkpoint
into a loadable HF checkpoint, then run generation inspection on the docidfix
turn-level prompts. The inspection should check whether the model:

- emits a single `<search>...</search>` action for question-only prompts
- does not hallucinate `<observation>` blocks as assistant text
- emits `<answer>...</answer>` after observation insertion
- keeps answer text grounded in the provided observation

Only after that inspection should the project move to GRPO initialization from
this checkpoint.
