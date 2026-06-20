#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
INDEX=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
OUT=/data/wzl/LightningSearch-RL/results/phase5q-env-rollout-gold-distractors-200
TRANSITIONS_OUT=/data/wzl/LightningSearch-RL/results/phase5q-env-transitions-soft-answer-from-5q200
LOG=/data/wzl/LightningSearch-RL/logs/phase5q-env-rollout-gold-distractors-200.log
GPU_IDS=${CUDA_VISIBLE_DEVICES:-7}

mkdir -p "$(dirname "$LOG")" "$OUT" "$TRANSITIONS_OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "sft=$SFT"
echo "index=$INDEX"
echo "model=$MODEL"
echo "out=$OUT"
echo "transitions_out=$TRANSITIONS_OUT"
echo "cuda_visible_devices=$GPU_IDS"
echo "offset=0"
echo "limit=200"
echo "top_k=8"
echo "candidate_pool=gold-distractors"
echo "distractor_count=6"
echo "max_new_tokens=64"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli inspect-env-rollout \
  --sft "$SFT" \
  --index "$INDEX" \
  --model "$MODEL" \
  --out-dir "$OUT" \
  --offset 0 \
  --limit 200 \
  --top-k 8 \
  --candidate-pool gold-distractors \
  --distractor-count 6 \
  --max-new-tokens 64

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-rollout-answers \
  --rollouts "$OUT/env_rollouts.jsonl" \
  --out "$OUT/answer_diagnostics.json"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-env-transitions \
  --rollouts "$OUT/env_rollouts.jsonl" \
  --out-dir "$TRANSITIONS_OUT"

echo "== rollout summary =="
cat "$OUT/summary.json"
echo "== answer diagnostics =="
cat "$OUT/answer_diagnostics.json"
echo "== transition summary =="
cat "$TRANSITIONS_OUT/summary.json"

echo "finished_at=$(date -Is)"
