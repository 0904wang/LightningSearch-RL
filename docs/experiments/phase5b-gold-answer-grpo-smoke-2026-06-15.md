# Phase 5B Gold-Answer GRPO Smoke

Date: 2026-06-15

## Goal

Fix the GRPO rollout ground-truth bug, regenerate rollout data with non-empty gold answers, and rerun the 4-GPU tiny `verl` GRPO smoke to verify reward behavior is no longer falsely positive.

## Root Cause

The original Phase 4G GRPO export wrote `metadata.answer` from the rule-based retrieval trace's `final_answer`. When BM25 top-2 missed the gold answer, `trace.final_answer` became an empty string. `verl_smoke` then copied that empty string into `reward_model.ground_truth`, and the reward hook treated an output without `<answer>` as an exact match against empty ground truth.

Fixes:

- `src/lightningsearch_rl/grpo.py`: rollout `metadata.answer` now uses the first non-empty gold answer from `example.answers`.
- `src/lightningsearch_rl/verl_reward.py`: `answer_reward` is only positive when an `<answer>` tag exists and `ground_truth` is non-empty.

## Data Export

Command:

```bash
cd /data/wzl/LightningSearch-RL/repo
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli export-grpo \
  --examples /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/examples.jsonl \
  --index /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/index.json \
  --out-dir /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo-gold-answer \
  --top-k 2
```

Artifacts:

- Rollouts: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo-gold-answer/rollouts.jsonl`
- Transitions: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo-gold-answer/transitions.jsonl`
- Reward records: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo-gold-answer/reward_records.jsonl`
- Summary: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500/grpo-gold-answer/summary.json`

Export checks:

```text
rollout_count: 500
empty_metadata_answer: 0
empty_response_answer_tag: 459
answer_reward mean in export reward records: 0.082
evidence_reward mean: 0.365
format_reward mean: 1.0
tool_validity_reward mean: 1.0
avg_reward: 0.325
```

The new metadata is fixed, but the rule-based retrieval trajectories still leave many response answer tags empty because BM25 top-2 often misses the answer.

## Smoke Run

Remote runtime:

- Remote workspace: `/data/wzl/LightningSearch-RL`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Sync method: narrow file sync from `D:\resume\Agent RL`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Model: `/data/wzl/LightningSearch-RL/models/Qwen/Qwen3-4B`
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,7`
- Session: `lightningsearch-20260615-phase5b-tiny-grpo-smoke-4gpu-vllm-gold-answer`

Launch command:

```bash
tmux new-session -d -s lightningsearch-20260615-phase5b-tiny-grpo-smoke-4gpu-vllm-gold-answer "bash -lc 'set -eo pipefail; cd /data/wzl/LightningSearch-RL/repo; source /home/user/anaconda3/etc/profile.d/conda.sh; conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl; CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli train --config configs/experiments/phase5b_tiny_grpo_smoke_4gpu.yaml --output-dir /data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke-4gpu-vllm-gold-answer --checkpoint-dir /data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke-4gpu-vllm-gold-answer 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase5b-tiny-grpo-smoke-4gpu-vllm-gold-answer.log'"
```

Artifacts:

- Log: `/data/wzl/LightningSearch-RL/logs/phase5b-tiny-grpo-smoke-4gpu-vllm-gold-answer.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase5b-tiny-grpo-smoke-4gpu-vllm-gold-answer`
- Checkpoints: `/data/wzl/LightningSearch-RL/checkpoints/phase5b-tiny-grpo-smoke-4gpu-vllm-gold-answer`

Effective training config:

- Train samples: 4
- Validation samples: 1
- Total training steps: 1
- `data.train_batch_size`: 4
- `actor_rollout_ref.actor.ppo_mini_batch_size`: 4
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`: 1
- `actor_rollout_ref.rollout.agent.num_workers`: 4
- `actor_rollout_ref.rollout.gpu_memory_utilization`: 0.25
- `actor_rollout_ref.rollout.max_model_len`: 768
- `actor_rollout_ref.rollout.max_num_batched_tokens`: 1024
- `actor_rollout_ref.rollout.max_num_seqs`: 4
- `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes`: 512
- `actor_rollout_ref.actor.fsdp_config.model_dtype`: `bfloat16`
- `actor_rollout_ref.actor.fsdp_config.param_offload`: `True`
- `actor_rollout_ref.actor.fsdp_config.optimizer_offload`: `True`

## Result

Status: completed 1-step GRPO smoke.

Key log metrics:

```text
Initial validation:
val-core/lightningsearch_rl/reward/mean@1: -0.029999999329447746
val-aux/lightningsearch_rl/score/mean@1: -0.03
val-aux/lightningsearch_rl/answer_reward/mean@1: 0.0
val-aux/lightningsearch_rl/format_reward/mean@1: 0.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.03
val-aux/num_turns/mean: 2.0

Training:
training/global_step: 1
critic/score/mean: -0.029999999329447746
critic/rewards/mean: -0.029999999329447746
actor/loss: 0.029999971389770508
actor/grad_norm: 0.216796875
actor/perf/max_memory_allocated_gb: 9.381478309631348
actor/perf/max_memory_reserved_gb: 13.560546875
response_length/mean: 256.0
response_length/clip_ratio: 1.0
response/aborted_ratio: 0.0
timing_s/gen: 3.647714480292052
timing_s/update_actor: 4.8409229828976095
timing_s/update_weights: 1.7906881291419268
timing_s/step: 20.430301097687334
perf/throughput: 13.741843483245594
```

The run again emitted shutdown-time DataLoader/resource tracker/vLLM engine warnings after the training step completed. GPUs returned to idle after completion.

## Analysis

This smoke is more honest than the previous `offload5` run. The earlier positive validation reward was caused by empty ground truth, not model quality. With gold answers fixed, the tiny smoke now shows the model is not producing valid `<answer>` output under the current prompt / rollout setup. The persistent `response_length/clip_ratio=1.0` also indicates generated responses hit the 256-token cap.

The next bottleneck is no longer infrastructure. It is behavior formatting and answer extraction:

- The model is likely continuing free-form generation or long thinking instead of emitting `<answer>...</answer>`.
- The current train data contains many empty rule-based answer tags because retrieval misses the answer.
- RL from sparse negative reward at this point is unlikely to improve quickly without better SFT warmup traces, stronger format reward, or a shorter/stricter prompt.

## Next Steps

- Log or dump generated validation responses directly so we can inspect what Qwen3-4B is producing before reward.
- Build a small SFT warmup set from gold evidence traces where `<answer>` is always non-empty.
- Consider reducing `max_response_length` to 128 for smoke runs after inspecting generations.
- Improve rollout prompt/template to explicitly require exactly one final `<answer>` tag.
