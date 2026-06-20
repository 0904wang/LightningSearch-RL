#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
OUT=/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu-response-dump
LOG=/data/wzl/LightningSearch-RL/logs/phase5f-tiny-grpo-docidfix-two-stage-4gpu-response-dump.log
GPU_IDS=${CUDA_VISIBLE_DEVICES:-6}

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "sft=$SFT"
echo "model=$MODEL"
echo "out=$OUT"
echo "cuda_visible_devices=$GPU_IDS"
echo "offset=0"
echo "limit=3"
echo "max_new_tokens=64"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli inspect-generation \
  --sft "$SFT" \
  --model "$MODEL" \
  --out-dir "$OUT" \
  --offset 0 \
  --limit 3 \
  --max-new-tokens 64

if [ -f "$OUT/summary.json" ]; then
  cat "$OUT/summary.json"
fi

echo "finished_at=$(date -Is)"
