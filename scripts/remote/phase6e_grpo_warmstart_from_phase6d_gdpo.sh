#!/usr/bin/env bash
set -eo pipefail

REPO=/data/wzl/LightningSearch-RL/repo
ENV=/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
CONFIG=configs/experiments/phase6e_grpo_warmstart_from_phase6d_gdpo.yaml
OUT=/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-from-phase6d-gdpo
CKPT=/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo
LOG=/data/wzl/LightningSearch-RL/logs/phase6e-grpo-warmstart-from-phase6d-gdpo.log
METRICS=$OUT/metrics_summary.json
BATCH_DIAG=$OUT/batch_diagnostics.json
REWARD_DUMP=$OUT/reward_dump.jsonl
REWARD_DUMP_SUMMARY=$OUT/reward_dump_summary.json
PHASE6D_CKPT_ROOT=/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y
GDPO_ACTOR=/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/global_step_28/actor
GDPO_MERGED=/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28

mkdir -p "$(dirname "$LOG")" "$OUT" "$CKPT"
exec > >(tee "$LOG") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "repo=$REPO"
echo "env=$ENV"
echo "config=$CONFIG"
echo "out=$OUT"
echo "checkpoint=$CKPT"
echo "phase6d_actor=$GDPO_ACTOR"
echo "phase6d_merged=$GDPO_MERGED"
echo "reward_dump=$REWARD_DUMP"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

cd "$REPO"
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV"
set -u

if ! test -d "$GDPO_ACTOR"; then
  echo "missing_phase6d_actor=$GDPO_ACTOR"
  exit 1
fi

if ! test -f "$GDPO_MERGED/config.json" || ! find "$GDPO_MERGED" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) | grep -q .; then
  echo "== merge phase6d gdpo global_step_28 checkpoint =="
  PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$GDPO_ACTOR" \
    --target_dir "$GDPO_MERGED" \
    --use_cpu_initialization
else
  echo "== merge skipped for phase6d gdpo global_step_28: existing HF checkpoint found =="
fi

echo "== phase6d gdpo merged checkpoint size =="
du -sh "$GDPO_MERGED"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train \
  --config "$CONFIG" \
  --output-dir "$OUT" \
  --checkpoint-dir "$CKPT" \
  --print-command

echo "finished_at=$(date -Is)"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli parse-verl-log \
  --log "$LOG" \
  --out "$METRICS"

PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-verl-batches \
  --train-jsonl "$OUT/data/train.jsonl" \
  --metrics-summary "$METRICS" \
  --train-batch-size 4 \
  --out "$BATCH_DIAG"

if test -s "$REWARD_DUMP"; then
  PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli diagnose-reward-dump \
    --dump "$REWARD_DUMP" \
    --out "$REWARD_DUMP_SUMMARY"
else
  echo "reward_dump_missing_or_empty=$REWARD_DUMP"
fi
