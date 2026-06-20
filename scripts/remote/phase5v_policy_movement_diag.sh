#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
SFT_MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
GRPO_MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer/hf_merged_global_step_200
OUT=/data/wzl/LightningSearch-RL/results/phase5v-policy-movement-diag
LOG=/data/wzl/LightningSearch-RL/logs/phase5v-policy-movement-diag.log
GPU_IDS=${CUDA_VISIBLE_DEVICES:-7}

OFFSET=400
LIMIT=20
DEVICE=cuda
DTYPE=bfloat16
TOP_K_TENSORS=30

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "sft=$SFT"
echo "sft_model=$SFT_MODEL"
echo "grpo_model=$GRPO_MODEL"
echo "out=$OUT"
echo "cuda_visible_devices=$GPU_IDS"
echo "offset=$OFFSET"
echo "limit=$LIMIT"
echo "device=$DEVICE"
echo "dtype=$DTYPE"
echo "top_k_tensors=$TOP_K_TENSORS"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-policy-movement \
  --base-model "$SFT_MODEL" \
  --candidate-model "$GRPO_MODEL" \
  --sft "$SFT" \
  --out-dir "$OUT" \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --top-k-tensors "$TOP_K_TENSORS"

echo "== summary =="
cat "$OUT/summary.json"
echo "== parameter diff =="
cat "$OUT/parameter_diff.json"
echo "== logprob comparison =="
cat "$OUT/logprob_comparison.json"
echo "finished_at=$(date -Is)"
