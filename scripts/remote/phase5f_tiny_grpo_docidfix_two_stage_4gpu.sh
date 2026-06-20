#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
CONFIG=configs/experiments/phase5f_tiny_grpo_docidfix_two_stage_4gpu.yaml
OUT=/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu
CKPT=/data/wzl/LightningSearch-RL/checkpoints/phase5f-tiny-grpo-docidfix-two-stage-4gpu
LOG=/data/wzl/LightningSearch-RL/logs/phase5f-tiny-grpo-docidfix-two-stage-4gpu.log

mkdir -p "$(dirname "$LOG")" "$OUT" "$CKPT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "config=$CONFIG"
echo "out=$OUT"
echo "checkpoint=$CKPT"
echo "cuda_visible_devices=0,1,2,5"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

CUDA_VISIBLE_DEVICES=0,1,2,5 PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train \
  --config "$CONFIG" \
  --output-dir "$OUT" \
  --checkpoint-dir "$CKPT" \
  --print-command

echo "finished_at=$(date -Is)"
