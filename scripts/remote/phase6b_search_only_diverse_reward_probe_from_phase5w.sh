#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
TRANSITIONS=/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl
MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
OUT=/data/wzl/LightningSearch-RL/results/phase6b-search-only-diverse-probe-rankreward-493x12
PAIRS_OUT=/data/wzl/LightningSearch-RL/results/phase6b-search-vs-search-pairs-rankreward-493x12
LOG=/data/wzl/LightningSearch-RL/logs/phase6b-search-only-diverse-probe-rankreward-493x12.log
REWARD_DUMP=$OUT/reward_dump.jsonl
REWARD_DUMP_SUMMARY=$OUT/reward_dump_summary.json

mkdir -p "$(dirname "$LOG")" "$OUT" "$PAIRS_OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "transitions=$TRANSITIONS"
echo "model=$MODEL"
echo "out=$OUT"
echo "pairs_out=$PAIRS_OUT"
echo "reward_dump=$REWARD_DUMP"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"
set -u

HF_HOME=/data/wzl/LightningSearch-RL/.cache/huggingface \
HF_ENDPOINT=https://hf-mirror.com \
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli probe-reward-variance \
  --transitions "$TRANSITIONS" \
  --model "$MODEL" \
  --out-dir "$OUT" \
  --offset 0 \
  --limit 493 \
  --stage search \
  --search-diversity-prompt \
  --samples-per-prompt 12 \
  --max-new-tokens 64 \
  --search-reward-top-k 8 \
  --backend vllm \
  --batch-size 48 \
  --temperature 1.6 \
  --top-p 0.98 \
  --top-k 80 \
  --seed 20260620 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 768 \
  --tensor-parallel-size 1

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-reward-dump \
  --dump "$REWARD_DUMP" \
  --out "$REWARD_DUMP_SUMMARY"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli build-preference-pairs \
  --probe-requests "$OUT/probe_requests.jsonl" \
  --generations "$OUT/generations.jsonl" \
  --reward-dump "$REWARD_DUMP" \
  --out-dir "$PAIRS_OUT" \
  --stage search \
  --pair-category search_vs_search \
  --min-score-gap 0.05 \
  --min-samples 2 \
  --max-pairs-per-group 4 \
  --val-fraction 0.1 \
  --seed 20260620

echo "== probe summary =="
cat "$OUT/summary.json"

echo "== reward dump summary =="
cat "$REWARD_DUMP_SUMMARY"

echo "== pair summary =="
cat "$PAIRS_OUT/summary.json"

echo "== line counts =="
wc -l "$OUT/probe_requests.jsonl" "$OUT/generations.jsonl" "$REWARD_DUMP" "$PAIRS_OUT/pairs.jsonl" "$PAIRS_OUT/train.jsonl" "$PAIRS_OUT/val.jsonl"

echo "finished_at=$(date -Is)"
