#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
INDEX=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
OUT=/data/wzl/LightningSearch-RL/results/phase5r-env-rollout-gold-distractors-500
RAW_TRANSITIONS_OUT=/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500
FILTERED_TRANSITIONS_OUT=/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered
QUALITY_MANIFEST=/data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5q_known_mismatches.json
LOG=/data/wzl/LightningSearch-RL/logs/phase5r-env-rollout-gold-distractors-500-filtered.log
GPU_IDS=${CUDA_VISIBLE_DEVICES:-7}

mkdir -p "$(dirname "$LOG")" "$OUT" "$RAW_TRANSITIONS_OUT" "$FILTERED_TRANSITIONS_OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "sft=$SFT"
echo "index=$INDEX"
echo "model=$MODEL"
echo "out=$OUT"
echo "raw_transitions_out=$RAW_TRANSITIONS_OUT"
echo "filtered_transitions_out=$FILTERED_TRANSITIONS_OUT"
echo "quality_manifest=$QUALITY_MANIFEST"
echo "cuda_visible_devices=$GPU_IDS"
echo "offset=0"
echo "limit=500"
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
  --limit 500 \
  --top-k 8 \
  --candidate-pool gold-distractors \
  --distractor-count 6 \
  --max-new-tokens 64

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-rollout-answers \
  --rollouts "$OUT/env_rollouts.jsonl" \
  --out "$OUT/answer_diagnostics.json"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-env-transitions \
  --rollouts "$OUT/env_rollouts.jsonl" \
  --out-dir "$RAW_TRANSITIONS_OUT"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-env-transitions \
  --rollouts "$OUT/env_rollouts.jsonl" \
  --out-dir "$FILTERED_TRANSITIONS_OUT" \
  --quality-manifest "$QUALITY_MANIFEST" \
  --exclude-quality-flag qa_type_mismatch

echo "== rollout summary =="
cat "$OUT/summary.json"
echo "== answer diagnostics =="
cat "$OUT/answer_diagnostics.json"
echo "== raw transition summary =="
cat "$RAW_TRANSITIONS_OUT/summary.json"
echo "== filtered transition summary =="
cat "$FILTERED_TRANSITIONS_OUT/summary.json"

echo "finished_at=$(date -Is)"
