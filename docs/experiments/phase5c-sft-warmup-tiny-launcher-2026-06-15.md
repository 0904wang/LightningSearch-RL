# Phase 5C SFT Warmup Tiny Launcher

Date: 2026-06-15

## Goal

Prepare a short verl SFT warmup run for Qwen3-4B using the gold-evidence
`sft_warmup.jsonl` produced earlier in Phase 5C. The goal is to teach the model
the strict `think/search/observation/answer` tag format before returning to GRPO.

No training job was launched in this step. This record covers launcher creation,
dry-run artifact generation, and pre-launch validation.

## Code Changes

- Added `src/lightningsearch_rl/verl_sft_warmup.py`.
- Added CLI command `train-sft-warmup`.
- Added config `configs/experiments/phase5c_sft_warmup_tiny.yaml`.
- Added tests in `tests/test_verl_sft_warmup.py`.

## Config

```yaml
experiment_name: phase5c-sft-warmup-tiny
project_name: lightningsearch-rl
sft_path: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/sft-warmup-gold/sft_warmup.jsonl
train_samples: 480
val_samples: 20
model_path: /data/wzl/LightningSearch-RL/models/Qwen/Qwen3-4B
cuda_visible_devices: "7"
n_gpus_per_node: 1
total_training_steps: 20
train_batch_size: 1
micro_batch_size_per_gpu: 1
max_length: 1024
learning_rate: 1.0e-5
```

## Remote Dry Run

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train-sft-warmup \
  --config configs/experiments/phase5c_sft_warmup_tiny.yaml \
  --output-dir /data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny \
  --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny \
  --dry-run \
  --print-command
```

## Artifacts

- Train parquet: `/data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny/data/train.parquet`
- Val parquet: `/data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny/data/val.parquet`
- Manifest: `/data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny/manifest.json`
- Dry-run summary: `/data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny/dry_run_summary.json`
- Launch command: `/data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny/launch_command.txt`
- Planned checkpoint dir: `/data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny`
- Planned log path: `/data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny.log`

## Validation

Local related tests:

```text
26 passed in 1.44s
```

Remote related tests:

```text
26 passed in 0.46s
```

Remote parquet / dataset validation:

```text
summary_train_rows 480
summary_val_rows 20
train_shape (480, 5)
val_shape (20, 5)
columns ['id', 'messages', 'enable_thinking', 'data_source', 'extra_info']
first_answer_tag True
dataset_len 2
input_len 108
loss_tokens 11
```

verl emitted the expected Qwen thinking-template mismatch warning. The run config
sets `data.ignore_input_ids_mismatch=True`, matching verl's own recommendation
for this tokenizer behavior.

## Planned Launch Command

```bash
tmux new-session -d -s lightningsearch-20260615-sft-warmup-tiny "bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train-sft-warmup --config configs/experiments/phase5c_sft_warmup_tiny.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5c-sft-warmup-tiny --print-command 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5c-sft-warmup-tiny.log'"
```

The generated inner verl command is recorded at:

```text
/data/wzl/LightningSearch-RL/results/phase5c-sft-warmup-tiny/launch_command.txt
```

## Next Step

Launch the planned tmux job only after explicit user approval for the exact
session name, GPU, command, log path, result path, and checkpoint path.
