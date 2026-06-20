#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
INDEX=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
SFT_MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
GRPO_CKPT=/data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/global_step_200/actor
GRPO_MERGED=/data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/hf_merged_global_step_200
OUT=/data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval
SFT_OUT=$OUT/sft_baseline
GRPO_OUT=$OUT/grpo_global_step_200
LOG=/data/wzl/LightningSearch-RL/logs/phase5r-grpo-global200-heldout-eval.log
GPU_IDS=${CUDA_VISIBLE_DEVICES:-7}

OFFSET=400
LIMIT=100
TOP_K=8
DISTRACTOR_COUNT=6
MAX_NEW_TOKENS=64

mkdir -p "$(dirname "$LOG")" "$OUT" "$SFT_OUT" "$GRPO_OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "sft=$SFT"
echo "index=$INDEX"
echo "sft_model=$SFT_MODEL"
echo "grpo_ckpt=$GRPO_CKPT"
echo "grpo_merged=$GRPO_MERGED"
echo "out=$OUT"
echo "sft_out=$SFT_OUT"
echo "grpo_out=$GRPO_OUT"
echo "cuda_visible_devices=$GPU_IDS"
echo "offset=$OFFSET"
echo "limit=$LIMIT"
echo "top_k=$TOP_K"
echo "candidate_pool=gold-distractors"
echo "distractor_count=$DISTRACTOR_COUNT"
echo "max_new_tokens=$MAX_NEW_TOKENS"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

if ! test -f "$GRPO_MERGED/config.json" || ! find "$GRPO_MERGED" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) | grep -q .; then
  echo "== merge grpo checkpoint =="
  PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$GRPO_CKPT" \
    --target_dir "$GRPO_MERGED" \
    --use_cpu_initialization
else
  echo "== merge skipped: existing HF checkpoint found =="
fi

echo "== merged checkpoint size =="
du -sh "$GRPO_MERGED"

echo "== evaluate sft baseline =="
CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli inspect-env-rollout \
  --sft "$SFT" \
  --index "$INDEX" \
  --model "$SFT_MODEL" \
  --out-dir "$SFT_OUT" \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --top-k "$TOP_K" \
  --candidate-pool gold-distractors \
  --distractor-count "$DISTRACTOR_COUNT" \
  --max-new-tokens "$MAX_NEW_TOKENS"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-rollout-answers \
  --rollouts "$SFT_OUT/env_rollouts.jsonl" \
  --out "$SFT_OUT/answer_diagnostics.json"

echo "== evaluate grpo global_step_200 =="
CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli inspect-env-rollout \
  --sft "$SFT" \
  --index "$INDEX" \
  --model "$GRPO_MERGED" \
  --out-dir "$GRPO_OUT" \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --top-k "$TOP_K" \
  --candidate-pool gold-distractors \
  --distractor-count "$DISTRACTOR_COUNT" \
  --max-new-tokens "$MAX_NEW_TOKENS"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-rollout-answers \
  --rollouts "$GRPO_OUT/env_rollouts.jsonl" \
  --out "$GRPO_OUT/answer_diagnostics.json"

PYTHONNOUSERSITE=1 python - <<'PY'
import json
from pathlib import Path

out = Path("/data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval")
sft = json.loads((out / "sft_baseline" / "summary.json").read_text(encoding="utf-8"))
grpo = json.loads((out / "grpo_global_step_200" / "summary.json").read_text(encoding="utf-8"))
sft_diag = json.loads((out / "sft_baseline" / "answer_diagnostics.json").read_text(encoding="utf-8"))
grpo_diag = json.loads((out / "grpo_global_step_200" / "answer_diagnostics.json").read_text(encoding="utf-8"))
metrics = [
    "valid_search_action_rate",
    "valid_answer_action_rate",
    "answer_exact_match_rate",
    "answer_containment_match_rate",
    "answer_token_f1",
    "gold_evidence_recall",
    "all_gold_evidence_retrieved_rate",
    "assistant_observation_rate",
    "avg_observation_doc_count",
]
comparison = {
    "sft_baseline": {metric: sft.get(metric) for metric in metrics},
    "grpo_global_step_200": {metric: grpo.get(metric) for metric in metrics},
    "delta_grpo_minus_sft": {
        metric: round(float(grpo.get(metric, 0.0)) - float(sft.get(metric, 0.0)), 6)
        for metric in metrics
    },
    "sft_answer_diagnostics": sft_diag,
    "grpo_answer_diagnostics": grpo_diag,
    "paths": {
        "sft_summary": str(out / "sft_baseline" / "summary.json"),
        "grpo_summary": str(out / "grpo_global_step_200" / "summary.json"),
        "sft_rollouts": str(out / "sft_baseline" / "env_rollouts.jsonl"),
        "grpo_rollouts": str(out / "grpo_global_step_200" / "env_rollouts.jsonl"),
    },
}
(out / "comparison_summary.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "== sft summary =="
cat "$SFT_OUT/summary.json"
echo "== grpo summary =="
cat "$GRPO_OUT/summary.json"
echo "== comparison summary =="
cat "$OUT/comparison_summary.json"
echo "finished_at=$(date -Is)"
