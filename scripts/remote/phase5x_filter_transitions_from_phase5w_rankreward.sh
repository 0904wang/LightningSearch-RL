#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
TRANSITIONS=/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl
REWARD_DUMP=/data/wzl/LightningSearch-RL/results/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward/reward_dump.jsonl
OUT=/data/wzl/LightningSearch-RL/results/phase5x-env-transitions-variance-rankreward-from-phase5w
LOG=/data/wzl/LightningSearch-RL/logs/phase5x-env-transitions-variance-rankreward-from-phase5w.log

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "transitions=$TRANSITIONS"
echo "reward_dump=$REWARD_DUMP"
echo "out=$OUT"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli filter-transitions-by-reward-variance \
  --transitions "$TRANSITIONS" \
  --reward-dump "$REWARD_DUMP" \
  --out-dir "$OUT" \
  --stage search \
  --stage answer \
  --min-score-range 0.000001 \
  --min-samples 2

echo "== variance filter summary =="
cat "$OUT/summary.json"

echo "== line counts =="
wc -l "$OUT/transitions.jsonl"

echo "finished_at=$(date -Is)"
