#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
TRANSITIONS=/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl
OUT=/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w
LOG=/data/wzl/LightningSearch-RL/logs/phase6c-synthetic-search-negatives-rankreward-from-phase5w.log

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "transitions=$TRANSITIONS"
echo "out=$OUT"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"
set -u

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli build-synthetic-search-preferences \
  --transitions "$TRANSITIONS" \
  --out-dir "$OUT" \
  --offset 0 \
  --limit 493 \
  --search-reward-top-k 8 \
  --min-chosen-score 0.5 \
  --min-score-gap 0.05 \
  --max-negatives-per-transition 6 \
  --val-fraction 0.1 \
  --seed 20260620

echo "== synthetic preference summary =="
cat "$OUT/summary.json"

echo "== line counts =="
wc -l "$OUT/candidates.jsonl" "$OUT/reward_dump.jsonl" "$OUT/pairs.jsonl" "$OUT/train.jsonl" "$OUT/val.jsonl"

echo "finished_at=$(date -Is)"
