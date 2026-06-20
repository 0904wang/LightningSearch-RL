#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
CONFIG=configs/experiments/phase5m_env_transition_grpo_4gpu_lowlen.yaml
OUT=/data/wzl/LightningSearch-RL/results/phase5m-env-transition-grpo-4gpu-lowlen
CKPT=/data/wzl/LightningSearch-RL/checkpoints/phase5m-env-transition-grpo-4gpu-lowlen
LOG=/data/wzl/LightningSearch-RL/logs/phase5m-env-transition-grpo-4gpu-lowlen.log

mkdir -p "$(dirname "$LOG")" "$OUT" "$CKPT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "config=$CONFIG"
echo "out=$OUT"
echo "checkpoint=$CKPT"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train \
  --config "$CONFIG" \
  --output-dir "$OUT" \
  --checkpoint-dir "$CKPT" \
  --print-command

echo "finished_at=$(date -Is)"
