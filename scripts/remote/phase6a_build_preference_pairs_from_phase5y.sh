#!/usr/bin/env bash
set -eo pipefail

cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
export PYTHONNOUSERSITE=1
set -u

INPUT_DIR=/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6
OUT_DIR=/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y

mkdir -p "$OUT_DIR"

python -m lightningsearch_rl.cli build-preference-pairs \
  --probe-requests "$INPUT_DIR/probe_requests.jsonl" \
  --generations "$INPUT_DIR/generations.jsonl" \
  --reward-dump "$INPUT_DIR/reward_dump.jsonl" \
  --out-dir "$OUT_DIR" \
  --stage search \
  --stage answer \
  --min-score-gap 0.25 \
  --min-samples 2 \
  --max-pairs-per-group 2 \
  --max-answer-pairs 300 \
  --val-fraction 0.1 \
  --seed 20260620

python - <<'PY'
from pathlib import Path
import json

summary_path = Path("/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y/summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print(json.dumps({
    "pair_count": summary["pair_count"],
    "train_count": summary["train_count"],
    "val_count": summary["val_count"],
    "stage_pair_counts": summary["stage_pair_counts"],
    "stage_candidate_pair_counts": summary["stage_candidate_pair_counts"],
}, ensure_ascii=False, indent=2, sort_keys=True))
PY
