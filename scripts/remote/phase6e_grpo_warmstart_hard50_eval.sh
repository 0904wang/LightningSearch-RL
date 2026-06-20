#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
INDEX=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
SFT_MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
PHASE6D_ROOT=/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y
GDPO_ACTOR=/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/global_step_28/actor
GDPO_MERGED=/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28
PHASE6E_ROOT=/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo
GRPO_ACTOR=/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/global_step_28/actor
GRPO_MERGED=/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/hf_merged_global_step_28
OUT=/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval
SFT_OUT=$OUT/sft_baseline
GDPO_OUT=$OUT/phase6d_gdpo_global_step_28
GRPO_OUT=$OUT/phase6e_grpo_global_step_28
LOG=/data/wzl/LightningSearch-RL/logs/phase6e-grpo-warmstart-hard50-eval.log
GPU_IDS=${CUDA_VISIBLE_DEVICES:-0}

OFFSET=400
LIMIT=100
TOP_K=8
DISTRACTOR_COUNT=50
MAX_NEW_TOKENS=64

mkdir -p "$(dirname "$LOG")" "$OUT" "$SFT_OUT" "$GDPO_OUT" "$GRPO_OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "sft=$SFT"
echo "index=$INDEX"
echo "sft_model=$SFT_MODEL"
echo "phase6d_root=$PHASE6D_ROOT"
echo "phase6d_actor=$GDPO_ACTOR"
echo "phase6d_merged=$GDPO_MERGED"
echo "phase6e_root=$PHASE6E_ROOT"
echo "phase6e_actor=$GRPO_ACTOR"
echo "phase6e_merged=$GRPO_MERGED"
echo "out=$OUT"
echo "sft_out=$SFT_OUT"
echo "gdpo_out=$GDPO_OUT"
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
set -u

ensure_merged_checkpoint() {
  local actor_dir=$1
  local merged_dir=$2
  local label=$3
  if ! test -d "$actor_dir"; then
    echo "missing_${label}_actor=$actor_dir"
    exit 1
  fi
  if ! test -f "$merged_dir/config.json" || ! find "$merged_dir" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) | grep -q .; then
    echo "== merge $label global_step_28 checkpoint =="
    PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "$actor_dir" \
      --target_dir "$merged_dir" \
      --use_cpu_initialization
  else
    echo "== merge skipped for $label global_step_28: existing HF checkpoint found =="
  fi
  echo "== $label merged checkpoint size =="
  du -sh "$merged_dir"
}

ensure_merged_checkpoint "$GDPO_ACTOR" "$GDPO_MERGED" "phase6d_gdpo"
ensure_merged_checkpoint "$GRPO_ACTOR" "$GRPO_MERGED" "phase6e_grpo"

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

run_eval "$SFT_MODEL" "$SFT_OUT" "sft baseline"
run_eval "$GDPO_MERGED" "$GDPO_OUT" "phase6d gdpo warmup"
run_eval "$GRPO_MERGED" "$GRPO_OUT" "phase6e grpo warmstart"

PYTHONNOUSERSITE=1 python - <<'PY'
import json
from pathlib import Path

out = Path("/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval")
model_dirs = {
    "sft_baseline": out / "sft_baseline",
    "phase6d_gdpo_global_step_28": out / "phase6d_gdpo_global_step_28",
    "phase6e_grpo_global_step_28": out / "phase6e_grpo_global_step_28",
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

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def row_id(row: dict) -> str:
    return str(row.get("id") or row.get("question_id") or row.get("metadata", {}).get("id") or "")

def search_query(row: dict) -> str:
    return str((row.get("search_action") or {}).get("query") or "")

summaries = {
    name: load_json(model_dir / "summary.json")
    for name, model_dir in model_dirs.items()
}
diagnostics = {
    name: load_json(model_dir / "answer_diagnostics.json")
    for name, model_dir in model_dirs.items()
}
rollouts = {
    name: {row_id(row): row for row in load_jsonl(model_dir / "env_rollouts.jsonl")}
    for name, model_dir in model_dirs.items()
}

def metric_view(name: str) -> dict:
    return {metric: summaries[name].get(metric) for metric in metrics}

def delta(candidate: str, baseline: str) -> dict:
    return {
        metric: round(float(summaries[candidate].get(metric, 0.0)) - float(summaries[baseline].get(metric, 0.0)), 6)
        for metric in metrics
    }

def diff_pair(candidate: str, baseline: str) -> dict:
    changed_answers = []
    changed_searches = []
    exact_improvements = []
    exact_regressions = []
    f1_improvements = []
    f1_regressions = []
    candidate_rollouts = rollouts[candidate]
    baseline_rollouts = rollouts[baseline]
    for key, candidate_row in candidate_rollouts.items():
        baseline_row = baseline_rollouts.get(key)
        if not baseline_row:
            continue
        if candidate_row.get("final_answer") != baseline_row.get("final_answer"):
            changed_answers.append(key)
        if search_query(candidate_row) != search_query(baseline_row):
            changed_searches.append(key)
        baseline_exact = bool(baseline_row.get("answer_exact_match"))
        candidate_exact = bool(candidate_row.get("answer_exact_match"))
        if candidate_exact and not baseline_exact:
            exact_improvements.append(key)
        if baseline_exact and not candidate_exact:
            exact_regressions.append(key)
        baseline_f1 = float(baseline_row.get("answer_token_f1", 0.0))
        candidate_f1 = float(candidate_row.get("answer_token_f1", 0.0))
        if candidate_f1 > baseline_f1:
            f1_improvements.append(key)
        if candidate_f1 < baseline_f1:
            f1_regressions.append(key)
    return {
        "changed_answer_count": len(changed_answers),
        "changed_search_count": len(changed_searches),
        "changed_answer_ids": changed_answers,
        "changed_search_ids": changed_searches,
        "exact_improvement_count": len(exact_improvements),
        "exact_regression_count": len(exact_regressions),
        "f1_improvement_count": len(f1_improvements),
        "f1_regression_count": len(f1_regressions),
        "exact_improvement_ids": exact_improvements,
        "exact_regression_ids": exact_regressions,
        "f1_improvement_ids": f1_improvements,
        "f1_regression_ids": f1_regressions,
    }

comparison = {
    "sft_baseline": metric_view("sft_baseline"),
    "phase6d_gdpo_global_step_28": metric_view("phase6d_gdpo_global_step_28"),
    "phase6e_grpo_global_step_28": metric_view("phase6e_grpo_global_step_28"),
    "deltas": {
        "phase6d_minus_sft": delta("phase6d_gdpo_global_step_28", "sft_baseline"),
        "phase6e_minus_sft": delta("phase6e_grpo_global_step_28", "sft_baseline"),
        "phase6e_minus_phase6d": delta("phase6e_grpo_global_step_28", "phase6d_gdpo_global_step_28"),
    },
    "diff_summary": {
        "phase6d_vs_sft": diff_pair("phase6d_gdpo_global_step_28", "sft_baseline"),
        "phase6e_vs_sft": diff_pair("phase6e_grpo_global_step_28", "sft_baseline"),
        "phase6e_vs_phase6d": diff_pair("phase6e_grpo_global_step_28", "phase6d_gdpo_global_step_28"),
    },
    "answer_diagnostics": diagnostics,
    "paths": {
        name: {
            "summary": str(model_dir / "summary.json"),
            "rollouts": str(model_dir / "env_rollouts.jsonl"),
            "answer_diagnostics": str(model_dir / "answer_diagnostics.json"),
        }
        for name, model_dir in model_dirs.items()
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
echo "== phase6d gdpo summary =="
cat "$GDPO_OUT/summary.json"
echo "== phase6e grpo summary =="
cat "$GRPO_OUT/summary.json"
echo "== comparison summary =="
cat "$OUT/comparison_summary.json"
echo "finished_at=$(date -Is)"
