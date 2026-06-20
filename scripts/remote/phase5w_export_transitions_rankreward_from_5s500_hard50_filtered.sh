#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
ROLLOUTS=/data/wzl/LightningSearch-RL/results/phase5s-env-rollout-gold-distractors-500-hard50/env_rollouts.jsonl
INDEX=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
OUT=/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1
QUALITY_MANIFEST=/data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5r_500_known_mismatches.json
LOG=/data/wzl/LightningSearch-RL/logs/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1.log

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "rollouts=$ROLLOUTS"
echo "index=$INDEX"
echo "out=$OUT"
echo "quality_manifest=$QUALITY_MANIFEST"
echo "exclude_quality_flags=qa_type_mismatch,answer_none_low_reward"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-env-transitions \
  --rollouts "$ROLLOUTS" \
  --index "$INDEX" \
  --out-dir "$OUT" \
  --quality-manifest "$QUALITY_MANIFEST" \
  --exclude-quality-flag qa_type_mismatch \
  --exclude-quality-flag answer_none_low_reward

echo "== transition summary =="
cat "$OUT/summary.json"

echo "== line counts =="
wc -l "$OUT/transitions.jsonl" "$OUT/reward_records.jsonl" "$OUT/rollouts_for_grpo.jsonl"

echo "== candidate passage check =="
PYTHONNOUSERSITE=1 python - <<'PY'
import json
from pathlib import Path

path = Path("/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl")
counts = []
search_rows = 0
with path.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if row.get("action_type") != "search":
            continue
        search_rows += 1
        counts.append(len(row.get("candidate_passages") or []))
print(json.dumps({
    "search_rows": search_rows,
    "candidate_passage_min": min(counts) if counts else 0,
    "candidate_passage_max": max(counts) if counts else 0,
    "candidate_passage_mean": round(sum(counts) / len(counts), 6) if counts else 0.0,
}, ensure_ascii=False, sort_keys=True))
PY

echo "finished_at=$(date -Is)"
