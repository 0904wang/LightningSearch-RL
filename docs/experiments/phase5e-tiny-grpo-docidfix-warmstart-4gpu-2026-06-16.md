# Phase 5E Tiny GRPO DocID-Fixed Warm-Start 4-GPU Smoke

Date: 2026-06-16

## Goal

Run a one-step 4-GPU `verl` GRPO smoke initialized from the docidfix Phase 5D
turn-level SFT checkpoint. This checks whether the newly inspected
`<search>`/`<answer>` format checkpoint can enter the GRPO training loop with
vLLM rollout, custom reward, and FSDP actor update.

## Code and Config Changes

- Added `configs/experiments/phase5e_tiny_grpo_docidfix_warmstart_4gpu.yaml`.
- Added optional `agent_system_prompt` support in `src/lightningsearch_rl/verl_smoke.py`.
- Added tests in `tests/test_verl_smoke.py` to verify the system prompt and the
  docidfix warm-start checkpoint path.
- Added remote launcher:
  `scripts/remote/phase5e_tiny_grpo_docidfix_warmstart_4gpu.sh`.

TDD check:

```text
test_prepare_verl_smoke_can_prepend_agent_system_prompt failed before implementation:
expected system,user prompt, got user-only prompt

after implementation:
python -m pytest tests\test_verl_smoke.py::test_prepare_verl_smoke_can_prepend_agent_system_prompt -q -> 1 passed
python -m pytest tests\test_verl_smoke.py tests\test_verl_reward.py tests\test_grpo.py -q -> 12 passed
python -m pytest -q -> 98 passed
remote related tests -> 12 passed
```

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
remote repo state: not-a-git-repo narrow sync from local workspace
local branch: master
local base commit: 44493db
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
config: /data/wzl/LightningSearch-RL/repo/configs/experiments/phase5e_tiny_grpo_docidfix_warmstart_4gpu.yaml
session: lightningsearch-20260616-phase5e-tiny-grpo-docidfix-warmstart-4gpu
gpus: CUDA_VISIBLE_DEVICES=0,1,2,5
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
rollouts: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/grpo-gold-answer/rollouts.jsonl
train rows: 4
val rows: 1
train batch size: 4
ppo mini batch size: 4
ppo micro batch size per GPU: 1
max prompt length: 512
max response length: 128
total training steps: 1
save_freq: -1
test_freq: -1
```

The remote log timestamps are UTC:

```text
started_at=2026-06-15T16:28:03+00:00
finished_at=2026-06-15T16:29:48+00:00
```

## Launch

Remote launcher:

```bash
/data/wzl/LightningSearch-RL/runs/phase5e_tiny_grpo_docidfix_warmstart_4gpu.sh
```

tmux command:

```bash
tmux new-session -d -s lightningsearch-20260616-phase5e-tiny-grpo-docidfix-warmstart-4gpu \
  "bash /data/wzl/LightningSearch-RL/runs/phase5e_tiny_grpo_docidfix_warmstart_4gpu.sh"
```

Inner `verl` command was recorded at:

```text
/data/wzl/LightningSearch-RL/results/phase5e-tiny-grpo-docidfix-warmstart-4gpu/launch_command.txt
```

Key overrides:

```text
actor_rollout_ref.model.path=/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.n=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.25
actor_rollout_ref.rollout.max_model_len=768
actor_rollout_ref.rollout.max_num_batched_tokens=1024
actor_rollout_ref.rollout.max_num_seqs=4
actor_rollout_ref.rollout.agent.num_workers=4
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
algorithm.adv_estimator=grpo
reward.custom_reward_function.path=src/lightningsearch_rl/verl_reward.py
reward.custom_reward_function.name=compute_score
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5e-tiny-grpo-docidfix-warmstart-4gpu.log
results: /data/wzl/LightningSearch-RL/results/phase5e-tiny-grpo-docidfix-warmstart-4gpu
checkpoint dir: /data/wzl/LightningSearch-RL/checkpoints/phase5e-tiny-grpo-docidfix-warmstart-4gpu
remote record: /data/wzl/LightningSearch-RL/results/phase5e-tiny-grpo-docidfix-warmstart-4gpu/EXPERIMENT_RECORD.md
```

Result files:

```text
dry_run_summary.json
launch_command.txt
manifest.json
data/train.jsonl
data/train.parquet
data/val.jsonl
data/val.parquet
```

No model checkpoint was expected because `save_freq=-1`.

## Metrics

Initial validation:

```text
val-core/lightningsearch_rl/reward/mean@1: -0.029999999329447746
val-aux/lightningsearch_rl/score/mean@1: -0.03
val-aux/lightningsearch_rl/answer_reward/mean@1: 0.0
val-aux/lightningsearch_rl/format_reward/mean@1: 0.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.03
val-aux/num_turns/min: 2
val-aux/num_turns/max: 2
val-aux/num_turns/mean: 2.0
```

Training step:

```text
training/global_step: 1
actor/loss: 0.029999971389770508
actor/pg_loss: 0.02999997278675437
actor/grad_norm: 0.00011777877807617188
actor/lr: 1e-06
actor/perf/max_memory_allocated_gb: 9.381476402282715
actor/perf/max_memory_reserved_gb: 13.564453125
critic/score/mean: -0.029999999329447746
critic/rewards/mean: -0.029999999329447746
response_length/mean: 22.75
response_length/max: 26.0
response_length/min: 20.0
response_length/clip_ratio: 0.0
response/aborted_ratio: 0.0
prompt_length/mean: 77.75
prompt_length/clip_ratio: 0.0
num_turns/mean: 2.0
timing_s/gen: 0.7967330291867256
timing_s/update_actor: 5.000708352774382
timing_s/update_weights: 1.7103027049452066
timing_s/step: 17.619405342731625
perf/throughput: 5.703938245649043
```

## Validation

Post-run checks:

```text
tmux session: no active session after completion
GPU 0: 3506 MiB / 32607 MiB
GPU 1: 3505 MiB / 32607 MiB
GPU 2: 3507 MiB / 32607 MiB
GPU 5: 3493 MiB / 32607 MiB
log lines: 961
train rows: 4
val rows: 1
training/global_step: 1 present
finished_at present
Final validation metrics: None
```

The broad `Traceback|RuntimeError|ERROR|Exception` log scan found 11 matches.
They occurred during shutdown after the training progress reached `1/1` and
after the `step:1` metrics were printed:

```text
Exception ignored in: _StatefulMultiProcessingDataLoaderIter.__del__
RuntimeError: DataLoader worker (...) is killed by signal: Killed.
resource_tracker: process died unexpectedly, relaunching. Some resources might leak.
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

This matches the cleanup noise seen in earlier Phase 5B GRPO smoke runs. It
should remain tracked as a known shutdown warning, not hidden.

## Analysis

The infrastructure objective succeeded: the docidfix SFT checkpoint can be used
as the initial actor for a 4-GPU `verl` GRPO smoke, vLLM rollout starts, custom
reward executes, and one actor update completes without CUDA OOM. The most
important improvement over the pre-SFT GRPO smoke is response length:

```text
previous Phase 5B gold-answer smoke response_length/clip_ratio: 1.0
Phase 5E warm-start response_length/clip_ratio: 0.0
Phase 5E response_length/mean: 22.75
```

However, the reward is still negative:

```text
answer_reward: 0.0
format_reward: 0.0
score: -0.03
```

This is now a training-objective mismatch rather than a launch problem. The
Phase 5D turn-level checkpoint is trained to emit `<search>` first for a
question-only prompt, while the current GRPO reward hook only rewards a final
`<answer>...</answer>` in the generated solution string. The logged
`tool_calls/mean:0.0` also indicates that this run is not yet executing the
offline retrieval environment inside rollout.

## Next Steps

Before scaling beyond tiny smoke:

- dump generated rollout responses for the Phase 5E prompt to confirm whether
  they are valid `<search>` actions
- adapt GRPO rollout/reward to the two-stage agent loop, where first-turn
  `<search>` is valid and answer reward is assigned after environment
  observation insertion
- or train/evaluate an answer-stage-only GRPO smoke using prompts that already
  include runtime `<observation>` if the goal is to test final-answer reward
- keep `max_response_length=128` or lower for smoke runs because warm-start
  already avoids the previous 256-token clipping failure
