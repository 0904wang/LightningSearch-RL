#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
INDEX=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
SFT_MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
PHASE5U_CKPT_ROOT=/data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer
GRPO100_CKPT=$PHASE5U_CKPT_ROOT/global_step_100/actor
GRPO100_MERGED=$PHASE5U_CKPT_ROOT/hf_merged_global_step_100
GRPO200_CKPT=$PHASE5U_CKPT_ROOT/global_step_200/actor
GRPO200_MERGED=$PHASE5U_CKPT_ROOT/hf_merged_global_step_200
OUT=/data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-eval
SFT_OUT=$OUT/sft_baseline
GRPO100_OUT=$OUT/grpo_global_step_100
GRPO200_OUT=$OUT/grpo_global_step_200
LOG=/data/wzl/LightningSearch-RL/logs/phase5u-grpo-hard50-eval.log
GPU_IDS=${CUDA_VISIBLE_DEVICES:-7}

OFFSET=400
LIMIT=100
TOP_K=8
DISTRACTOR_COUNT=50
MAX_NEW_TOKENS=64

mkdir -p "$(dirname "$LOG")" "$OUT" "$SFT_OUT" "$GRPO100_OUT" "$GRPO200_OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "sft=$SFT"
echo "index=$INDEX"
echo "sft_model=$SFT_MODEL"
echo "phase5u_ckpt_root=$PHASE5U_CKPT_ROOT"
echo "grpo100_ckpt=$GRPO100_CKPT"
echo "grpo100_merged=$GRPO100_MERGED"
echo "grpo200_ckpt=$GRPO200_CKPT"
echo "grpo200_merged=$GRPO200_MERGED"
echo "out=$OUT"
echo "sft_out=$SFT_OUT"
echo "grpo100_out=$GRPO100_OUT"
echo "grpo200_out=$GRPO200_OUT"
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

merge_if_needed() {
  local fsdp_dir=$1
  local target_dir=$2
  local label=$3
  if ! test -f "$target_dir/config.json" || ! find "$target_dir" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) | grep -q .; then
    echo "== merge $label checkpoint =="
    PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$fsdp_dir" \
      --target_dir "$target_dir" \
      --use_cpu_initialization
  else
    echo "== merge skipped for $label: existing HF checkpoint found =="
  fi
  echo "== $label merged checkpoint size =="
  du -sh "$target_dir"
}

run_eval() {
  local model=$1
  local out_dir=$2
  local label=$3
  echo "== evaluate $label =="
  CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli inspect-env-rollout \
    --sft "$SFT" \
    --index "$INDEX" \
    --model "$model" \
    --out-dir "$out_dir" \
    --offset "$OFFSET" \
    --limit "$LIMIT" \
    --top-k "$TOP_K" \
    --candidate-pool gold-distractors \
    --distractor-count "$DISTRACTOR_COUNT" \
    --max-new-tokens "$MAX_NEW_TOKENS"

  PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-rollout-answers \
    --rollouts "$out_dir/env_rollouts.jsonl" \
    --out "$out_dir/answer_diagnostics.json"
}

merge_if_needed "$GRPO100_CKPT" "$GRPO100_MERGED" "grpo global_step_100"
merge_if_needed "$GRPO200_CKPT" "$GRPO200_MERGED" "grpo global_step_200"

run_eval "$SFT_MODEL" "$SFT_OUT" "sft baseline"
run_eval "$GRPO100_MERGED" "$GRPO100_OUT" "grpo global_step_100"
run_eval "$GRPO200_MERGED" "$GRPO200_OUT" "grpo global_step_200"

PYTHONNOUSERSITE=1 python - <<'PY'
import json
from pathlib import Path

out = Path("/data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-eval")
models = {
    "sft_baseline": out / "sft_baseline",
    "grpo_global_step_100": out / "grpo_global_step_100",
    "grpo_global_step_200": out / "grpo_global_step_200",
}
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
summaries = {
    name: json.loads((path / "summary.json").read_text(encoding="utf-8"))
    for name, path in models.items()
}
diagnostics = {
    name: json.loads((path / "answer_diagnostics.json").read_text(encoding="utf-8"))
    for name, path in models.items()
}
sft = summaries["sft_baseline"]
comparison = {
    name: {metric: summaries[name].get(metric) for metric in metrics}
    for name in models
}
for name in ("grpo_global_step_100", "grpo_global_step_200"):
    comparison[f"delta_{name}_minus_sft"] = {
        metric: round(float(summaries[name].get(metric, 0.0)) - float(sft.get(metric, 0.0)), 6)
        for metric in metrics
    }
comparison["answer_diagnostics"] = diagnostics
comparison["paths"] = {
    name: {
        "summary": str(path / "summary.json"),
        "rollouts": str(path / "env_rollouts.jsonl"),
        "answer_diagnostics": str(path / "answer_diagnostics.json"),
    }
    for name, path in models.items()
}
(out / "comparison_summary.json").write_text(
    json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "== sft summary =="
cat "$SFT_OUT/summary.json"
echo "== grpo global_step_100 summary =="
cat "$GRPO100_OUT/summary.json"
echo "== grpo global_step_200 summary =="
cat "$GRPO200_OUT/summary.json"
echo "== comparison summary =="
cat "$OUT/comparison_summary.json"
echo "finished_at=$(date -Is)"
