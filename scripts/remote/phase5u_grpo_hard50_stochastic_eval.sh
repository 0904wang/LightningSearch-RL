#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
INDEX=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
SFT_MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
PHASE5U_CKPT_ROOT=/data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer
GRPO200_CKPT=$PHASE5U_CKPT_ROOT/global_step_200/actor
GRPO200_MERGED=$PHASE5U_CKPT_ROOT/hf_merged_global_step_200
OUT=/data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-stochastic-eval
SFT_OUT=$OUT/sft_baseline_seed20260618
GRPO_OUT=$OUT/grpo_global_step_200_seed20260618
LOG=/data/wzl/LightningSearch-RL/logs/phase5u-grpo-hard50-stochastic-eval.log
GPU_IDS=${CUDA_VISIBLE_DEVICES:-7}

OFFSET=400
LIMIT=20
TOP_K=8
DISTRACTOR_COUNT=50
MAX_NEW_TOKENS=64
TEMPERATURE=0.7
TOP_P=0.9
SAMPLE_TOP_K=40
SEED=20260618

mkdir -p "$(dirname "$LOG")" "$OUT" "$SFT_OUT" "$GRPO_OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "sft=$SFT"
echo "index=$INDEX"
echo "sft_model=$SFT_MODEL"
echo "grpo200_ckpt=$GRPO200_CKPT"
echo "grpo200_merged=$GRPO200_MERGED"
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
echo "do_sample=true"
echo "temperature=$TEMPERATURE"
echo "top_p=$TOP_P"
echo "sample_top_k=$SAMPLE_TOP_K"
echo "seed=$SEED"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"

if ! test -f "$GRPO200_MERGED/config.json" || ! find "$GRPO200_MERGED" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) | grep -q .; then
  echo "== merge grpo global_step_200 checkpoint =="
  PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$GRPO200_CKPT" \
    --target_dir "$GRPO200_MERGED" \
    --use_cpu_initialization
else
  echo "== merge skipped for grpo global_step_200: existing HF checkpoint found =="
fi

echo "== grpo global_step_200 merged checkpoint size =="
du -sh "$GRPO200_MERGED"

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
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --do-sample \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --sample-top-k "$SAMPLE_TOP_K" \
    --seed "$SEED"

  PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-rollout-answers \
    --rollouts "$out_dir/env_rollouts.jsonl" \
    --out "$out_dir/answer_diagnostics.json"
}

run_eval "$SFT_MODEL" "$SFT_OUT" "sft baseline stochastic"
run_eval "$GRPO200_MERGED" "$GRPO_OUT" "grpo global_step_200 stochastic"

PYTHONNOUSERSITE=1 python - <<'PY'
import json
from pathlib import Path

out = Path("/data/wzl/LightningSearch-RL/results/phase5u-grpo-hard50-stochastic-eval")
sft_dir = out / "sft_baseline_seed20260618"
grpo_dir = out / "grpo_global_step_200_seed20260618"

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

sft_summary = load_json(sft_dir / "summary.json")
grpo_summary = load_json(grpo_dir / "summary.json")
sft_diag = load_json(sft_dir / "answer_diagnostics.json")
grpo_diag = load_json(grpo_dir / "answer_diagnostics.json")
sft_rollouts = {row_id(row): row for row in load_jsonl(sft_dir / "env_rollouts.jsonl")}
grpo_rollouts = {row_id(row): row for row in load_jsonl(grpo_dir / "env_rollouts.jsonl")}

changed_answers = []
changed_searches = []
exact_improvements = []
exact_regressions = []
f1_improvements = []
f1_regressions = []
for key, grpo_row in grpo_rollouts.items():
    sft_row = sft_rollouts.get(key)
    if not sft_row:
        continue
    if grpo_row.get("final_answer") != sft_row.get("final_answer"):
        changed_answers.append(key)
    if search_query(grpo_row) != search_query(sft_row):
        changed_searches.append(key)
    sft_exact = bool(sft_row.get("answer_exact_match"))
    grpo_exact = bool(grpo_row.get("answer_exact_match"))
    if grpo_exact and not sft_exact:
        exact_improvements.append(key)
    if sft_exact and not grpo_exact:
        exact_regressions.append(key)
    sft_f1 = float(sft_row.get("answer_token_f1", 0.0))
    grpo_f1 = float(grpo_row.get("answer_token_f1", 0.0))
    if grpo_f1 > sft_f1:
        f1_improvements.append(key)
    if grpo_f1 < sft_f1:
        f1_regressions.append(key)

comparison = {
    "sft_baseline": {metric: sft_summary.get(metric) for metric in metrics},
    "grpo_global_step_200": {metric: grpo_summary.get(metric) for metric in metrics},
    "delta_grpo_minus_sft": {
        metric: round(float(grpo_summary.get(metric, 0.0)) - float(sft_summary.get(metric, 0.0)), 6)
        for metric in metrics
    },
    "diff_summary": {
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
    },
    "sft_answer_diagnostics": sft_diag,
    "grpo_answer_diagnostics": grpo_diag,
    "sampling": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "sample_top_k": 40,
        "seed": 20260618,
    },
    "paths": {
        "sft_summary": str(sft_dir / "summary.json"),
        "grpo_summary": str(grpo_dir / "summary.json"),
        "sft_rollouts": str(sft_dir / "env_rollouts.jsonl"),
        "grpo_rollouts": str(grpo_dir / "env_rollouts.jsonl"),
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
echo "== grpo global_step_200 summary =="
cat "$GRPO_OUT/summary.json"
echo "== comparison summary =="
cat "$OUT/comparison_summary.json"
echo "finished_at=$(date -Is)"
