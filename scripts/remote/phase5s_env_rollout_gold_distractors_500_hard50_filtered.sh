#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
INDEX=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
OUT=/data/wzl/LightningSearch-RL/results/phase5s-env-rollout-gold-distractors-500-hard50
RAW_TRANSITIONS_OUT=/data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50
FILTERED_TRANSITIONS_OUT=/data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1
QUALITY_MANIFEST=/data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5r_500_known_mismatches.json
LOG=/data/wzl/LightningSearch-RL/logs/phase5s-env-rollout-gold-distractors-500-hard50-filtered.log
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
echo "exclude_quality_flags=qa_type_mismatch,answer_none_low_reward"
echo "cuda_visible_devices=$GPU_IDS"
echo "offset=0"
echo "limit=500"
echo "top_k=8"
echo "candidate_pool=gold-distractors"
echo "distractor_count=50"
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
  --distractor-count 50 \
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
  --exclude-quality-flag qa_type_mismatch \
  --exclude-quality-flag answer_none_low_reward

echo "== rollout summary =="
cat "$OUT/summary.json"
echo "== answer diagnostics =="
cat "$OUT/answer_diagnostics.json"
echo "== raw transition summary =="
cat "$RAW_TRANSITIONS_OUT/summary.json"
echo "== filtered transition summary =="
cat "$FILTERED_TRANSITIONS_OUT/summary.json"

echo "== filtered-v1 line counts =="
wc -l "$FILTERED_TRANSITIONS_OUT/transitions.jsonl" "$FILTERED_TRANSITIONS_OUT/reward_records.jsonl" "$FILTERED_TRANSITIONS_OUT/rollouts_for_grpo.jsonl"

echo "== filtered-v1 reward diagnostics =="
PYTHONNOUSERSITE=1 python - <<'PY'
import collections
import json
from pathlib import Path

path = Path("/data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/reward_records.jsonl")
counts = collections.Counter()
low = []
with path.open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        counts[row.get("answer_reward_type", "missing")] += 1
        if float(row.get("total", 0.0)) < 0.7:
            low.append(
                {
                    "id": row.get("id"),
                    "final_answer": row.get("final_answer"),
                    "gold_answer": row.get("gold_answer"),
                    "total": row.get("total"),
                    "answer_reward_type": row.get("answer_reward_type"),
                    "evidence_reward": row.get("evidence_reward"),
                }
            )
print(json.dumps({"answer_reward_type_counts": counts, "low_total_count": len(low), "low_total_examples": low[:20]}, ensure_ascii=False, sort_keys=True))
PY

echo "finished_at=$(date -Is)"
