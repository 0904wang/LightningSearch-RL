#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
CONFIG=configs/experiments/phase5n_env_transition_grpo_4gpu_100x5.yaml
OUT=/data/wzl/LightningSearch-RL/results/phase5n-env-transition-grpo-4gpu-100x5
CKPT=/data/wzl/LightningSearch-RL/checkpoints/phase5n-env-transition-grpo-4gpu-100x5
LOG=/data/wzl/LightningSearch-RL/logs/phase5n-env-transition-grpo-4gpu-100x5.log
METRICS=$OUT/metrics_summary.json

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

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli parse-verl-log \
  --log "$LOG" \
  --out "$METRICS"
