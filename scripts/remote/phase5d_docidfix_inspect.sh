#!/usr/bin/env bash
set -euo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
LOG=/data/wzl/LightningSearch-RL/logs/phase5d-sft-turns-docidfix-4gpu-generation-inspection.log
OUT=/data/wzl/LightningSearch-RL/results/phase5d-sft-turns-docidfix-4gpu-generation-inspection
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "cuda_visible_devices=6"
echo "sft=$SFT"
echo "model=$MODEL"
echo "out=$OUT"

cd "$REPO"
set +u
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"
set -u

CUDA_VISIBLE_DEVICES=6 PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli inspect-generation \
  --sft "$SFT" \
  --model "$MODEL" \
  --out-dir "$OUT" \
  --offset 480 \
  --limit 5 \
  --max-new-tokens 64

echo "finished_at=$(date -Is)"
