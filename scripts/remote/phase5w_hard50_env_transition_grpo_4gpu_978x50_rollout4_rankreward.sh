#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
CONFIG=configs/experiments/phase5w_hard50_env_transition_grpo_4gpu_978x50_rollout4_rankreward.yaml
OUT=/data/wzl/LightningSearch-RL/results/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward
CKPT=/data/wzl/LightningSearch-RL/checkpoints/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward
LOG=/data/wzl/LightningSearch-RL/logs/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward.log
METRICS=$OUT/metrics_summary.json
BATCH_DIAG=$OUT/batch_diagnostics.json
REWARD_DUMP=$OUT/reward_dump.jsonl
REWARD_DUMP_SUMMARY=$OUT/reward_dump_summary.json

mkdir -p "$(dirname "$LOG")" "$OUT" "$CKPT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "config=$CONFIG"
echo "out=$OUT"
echo "checkpoint=$CKPT"
echo "reward_dump=$REWARD_DUMP"
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

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-verl-batches \
  --train-jsonl "$OUT/data/train.jsonl" \
  --metrics-summary "$METRICS" \
  --train-batch-size 4 \
  --out "$BATCH_DIAG"

if test -s "$REWARD_DUMP"; then
  PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-reward-dump \
    --dump "$REWARD_DUMP" \
    --out "$REWARD_DUMP_SUMMARY"
else
  echo "reward_dump_missing_or_empty=$REWARD_DUMP"
fi
