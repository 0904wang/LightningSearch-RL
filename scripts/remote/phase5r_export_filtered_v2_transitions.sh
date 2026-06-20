#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
ROLLOUTS=/data/wzl/LightningSearch-RL/results/phase5r-env-rollout-gold-distractors-500/env_rollouts.jsonl
OUT=/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2
QUALITY_MANIFEST=/data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5r_500_known_mismatches.json
LOG=/data/wzl/LightningSearch-RL/logs/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2.log

mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "rollouts=$ROLLOUTS"
echo "out=$OUT"
echo "quality_manifest=$QUALITY_MANIFEST"
echo "exclude_quality_flags=qa_type_mismatch,answer_none_low_reward"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-env-transitions \
  --rollouts "$ROLLOUTS" \
  --out-dir "$OUT" \
  --quality-manifest "$QUALITY_MANIFEST" \
  --exclude-quality-flag qa_type_mismatch \
  --exclude-quality-flag answer_none_low_reward

echo "== filtered-v2 transition summary =="
cat "$OUT/summary.json"

echo "== filtered-v2 line counts =="
wc -l "$OUT/transitions.jsonl" "$OUT/reward_records.jsonl" "$OUT/rollouts_for_grpo.jsonl"

echo "== filtered-v2 answer_reward_type counts =="
PYTHONNOUSERSITE=1 python - <<'PY'
import collections
import json
from pathlib import Path

path = Path("/data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2/reward_records.jsonl")
counts = collections.Counter()
low = []
with path.open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        counts[row.get("answer_reward_type", "missing")] += 1
        if row.get("answer_reward_type") == "none":
            low.append(
                {
                    "id": row.get("id"),
                    "final_answer": row.get("final_answer"),
                    "gold_answer": row.get("gold_answer"),
                    "total": row.get("total"),
                }
            )
print(json.dumps({"answer_reward_type_counts": counts, "none_rows": low}, ensure_ascii=False, sort_keys=True))
PY

echo "finished_at=$(date -Is)"
