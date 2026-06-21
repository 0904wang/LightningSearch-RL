#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
SFT=/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
SFT_MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
GDPO_MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28
GRPO_MODEL=/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/hf_merged_global_step_28
OUT=/data/wzl/LightningSearch-RL/results/phase6f-policy-movement-diag
SFT_VS_6D_OUT=$OUT/sft_vs_phase6d_gdpo_global_step_28
SFT_VS_6E_OUT=$OUT/sft_vs_phase6e_grpo_global_step_28
GDPO_VS_GRPO_OUT=$OUT/phase6d_gdpo_vs_phase6e_grpo
LOG=/data/wzl/LightningSearch-RL/logs/phase6f-policy-movement-diag.log
GPU_IDS=${CUDA_VISIBLE_DEVICES:-7}

OFFSET=400
LIMIT=20
DEVICE=cuda
DTYPE=bfloat16
TOP_K_TENSORS=30

mkdir -p "$(dirname "$LOG")" "$OUT" "$SFT_VS_6D_OUT" "$SFT_VS_6E_OUT" "$GDPO_VS_GRPO_OUT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "sft=$SFT"
echo "sft_model=$SFT_MODEL"
echo "gdpo_model=$GDPO_MODEL"
echo "grpo_model=$GRPO_MODEL"
echo "out=$OUT"
echo "sft_vs_6d_out=$SFT_VS_6D_OUT"
echo "sft_vs_6e_out=$SFT_VS_6E_OUT"
echo "gdpo_vs_grpo_out=$GDPO_VS_GRPO_OUT"
echo "cuda_visible_devices=$GPU_IDS"
echo "offset=$OFFSET"
echo "limit=$LIMIT"
echo "device=$DEVICE"
echo "dtype=$DTYPE"
echo "top_k_tensors=$TOP_K_TENSORS"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"
set -u

run_diag() {
  local base_model=$1
  local candidate_model=$2
  local out_dir=$3
  local label=$4
  echo "== diagnose $label =="
  CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-policy-movement \
    --base-model "$base_model" \
    --candidate-model "$candidate_model" \
    --sft "$SFT" \
    --out-dir "$out_dir" \
    --offset "$OFFSET" \
    --limit "$LIMIT" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --top-k-tensors "$TOP_K_TENSORS"
}

run_diag "$SFT_MODEL" "$GDPO_MODEL" "$SFT_VS_6D_OUT" "sft vs phase6d gdpo"
run_diag "$SFT_MODEL" "$GRPO_MODEL" "$SFT_VS_6E_OUT" "sft vs phase6e grpo"
run_diag "$GDPO_MODEL" "$GRPO_MODEL" "$GDPO_VS_GRPO_OUT" "phase6d gdpo vs phase6e grpo"

PYTHONNOUSERSITE=1 python - <<'PY'
import json
from pathlib import Path

root = Path("/data/wzl/LightningSearch-RL/results/phase6f-policy-movement-diag")
comparisons = {
    "sft_vs_phase6d_gdpo_global_step_28": root / "sft_vs_phase6d_gdpo_global_step_28",
    "sft_vs_phase6e_grpo_global_step_28": root / "sft_vs_phase6e_grpo_global_step_28",
    "phase6d_gdpo_vs_phase6e_grpo": root / "phase6d_gdpo_vs_phase6e_grpo",
}

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def compact_parameter_diff(report: dict) -> dict:
    return {
        "compared_tensors": report.get("compared_tensors"),
        "changed_tensors": report.get("changed_tensors"),
        "unchanged_tensors": report.get("unchanged_tensors"),
        "relative_l2_diff": report.get("relative_l2_diff"),
        "mean_abs_diff": report.get("mean_abs_diff"),
        "max_abs_diff": report.get("max_abs_diff"),
        "top_tensor_changes": report.get("top_tensor_changes", [])[:5],
    }

def compact_logprob(report: dict) -> dict:
    return {
        "compared_records": report.get("compared_records"),
        "base_mean_logprob": report.get("base_mean_logprob"),
        "candidate_mean_logprob": report.get("candidate_mean_logprob"),
        "delta_mean_logprob": report.get("delta_mean_logprob"),
        "by_stage": report.get("by_stage", {}),
    }

summary = {
    "offset": 400,
    "limit": 20,
    "prompt_count_per_comparison": 40,
    "comparisons": {},
}
for name, path in comparisons.items():
    parameter_diff = load_json(path / "parameter_diff.json")
    logprob_comparison = load_json(path / "logprob_comparison.json")
    summary["comparisons"][name] = {
        "summary": str(path / "summary.json"),
        "parameter_diff": str(path / "parameter_diff.json"),
        "logprob_comparison": str(path / "logprob_comparison.json"),
        "compact_parameter_diff": compact_parameter_diff(parameter_diff),
        "compact_logprob_comparison": compact_logprob(logprob_comparison),
    }

(root / "comparison_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "== comparison summary =="
cat "$OUT/comparison_summary.json"
echo "finished_at=$(date -Is)"
