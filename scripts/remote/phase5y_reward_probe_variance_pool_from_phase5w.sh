#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
TRANSITIONS=/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl
MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
OUT=/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6
FILTERED_OUT=/data/wzl/LightningSearch-RL/results/phase5y-env-transitions-variance-rankreward-100src
LOG=/data/wzl/LightningSearch-RL/logs/phase5y-reward-probe-rankreward-978x6.log
REWARD_DUMP=$OUT/reward_dump.jsonl
REWARD_DUMP_SUMMARY=$OUT/reward_dump_summary.json

mkdir -p "$(dirname "$LOG")" "$OUT" "$FILTERED_OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "transitions=$TRANSITIONS"
echo "model=$MODEL"
echo "out=$OUT"
echo "filtered_out=$FILTERED_OUT"
echo "reward_dump=$REWARD_DUMP"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

HF_HOME=/data/wzl/LightningSearch-RL/.cache/huggingface \
HF_ENDPOINT=https://hf-mirror.com \
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli probe-reward-variance \
  --transitions "$TRANSITIONS" \
  --model "$MODEL" \
  --out-dir "$OUT" \
  --offset 0 \
  --limit 978 \
  --samples-per-prompt 6 \
  --max-new-tokens 64 \
  --search-reward-top-k 8 \
  --answer-token-f1-threshold 0.5 \
  --backend vllm \
  --batch-size 64 \
  --temperature 1.3 \
  --top-p 0.95 \
  --top-k 50 \
  --seed 20260619 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 768 \
  --tensor-parallel-size 1

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-reward-dump \
  --dump "$REWARD_DUMP" \
  --out "$REWARD_DUMP_SUMMARY"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli filter-transitions-by-reward-variance \
  --transitions "$TRANSITIONS" \
  --reward-dump "$REWARD_DUMP" \
  --out-dir "$FILTERED_OUT" \
  --stage search \
  --stage answer \
  --min-score-range 0.000000001 \
  --min-samples 2 \
  --max-source-count 100

echo "== probe summary =="
cat "$OUT/summary.json"

echo "== reward dump summary =="
cat "$REWARD_DUMP_SUMMARY"

echo "== filtered summary =="
cat "$FILTERED_OUT/summary.json"

echo "== line counts =="
wc -l "$OUT/probe_requests.jsonl" "$OUT/generations.jsonl" "$REWARD_DUMP" "$FILTERED_OUT/transitions.jsonl"

echo "finished_at=$(date -Is)"
