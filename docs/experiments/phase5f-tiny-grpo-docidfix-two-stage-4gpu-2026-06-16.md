# Phase 5F Tiny Two-Stage GRPO DocID-Fixed 4-GPU Smoke

Date: 2026-06-16

## Goal

Run a one-step 4-GPU `verl` GRPO smoke using the Phase 5F two-stage objective.
This follows the Phase 5E result where infrastructure worked but question-only
prompts were rewarded as if they should produce final `<answer>` tags. Phase 5F
uses SFT-turns rows and evaluates two stages:

- search stage: `system + question -> <search>...</search>`
- answer stage: `system + question + assistant search + observation -> <answer>...</answer>`

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
remote repo state: not-a-git-repo narrow sync from local workspace
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
config: /data/wzl/LightningSearch-RL/repo/configs/experiments/phase5f_tiny_grpo_docidfix_two_stage_4gpu.yaml
session: lightningsearch-20260616-phase5f-tiny-grpo-docidfix-two-stage-4gpu
gpus: CUDA_VISIBLE_DEVICES=0,1,2,5
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
sft turns: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
train source rows: 2
val source rows: 1
train stage rows: 4
val stage rows: 2
max response length: 64
total training steps: 1
save_freq: -1
test_freq: -1
```

Remote log timestamps are UTC:

```text
started_at=2026-06-15T16:48:17+00:00
finished_at=2026-06-15T16:50:02+00:00
```

## Launch

```bash
tmux new-session -d -s lightningsearch-20260616-phase5f-tiny-grpo-docidfix-two-stage-4gpu \
  "bash /data/wzl/LightningSearch-RL/runs/phase5f_tiny_grpo_docidfix_two_stage_4gpu.sh"
```

Inner `verl` command:

```text
/data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu/launch_command.txt
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5f-tiny-grpo-docidfix-two-stage-4gpu.log
results: /data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu
checkpoint dir: /data/wzl/LightningSearch-RL/checkpoints/phase5f-tiny-grpo-docidfix-two-stage-4gpu
remote record: /data/wzl/LightningSearch-RL/results/phase5f-tiny-grpo-docidfix-two-stage-4gpu/EXPERIMENT_RECORD.md
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
val-core/lightningsearch_rl/reward/mean@1: 1.0850000381469727
val-aux/lightningsearch_rl/score/mean@1: 1.085
val-aux/lightningsearch_rl/answer_reward/mean@1: 0.5
val-aux/lightningsearch_rl/search_reward/mean@1: 0.5
val-aux/lightningsearch_rl/format_reward/mean@1: 1.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.015
val-aux/num_turns/mean: 2.0
```

Training step:

```text
training/global_step: 1
critic/score/mean: 1.0850000381469727
critic/rewards/mean: 1.0850000381469727
actor/loss: -1.0783812999725342
actor/pg_loss: -1.0783813297748566
actor/grad_norm: 0.019775390625
actor/lr: 1e-06
actor/perf/max_memory_allocated_gb: 9.381477355957031
actor/perf/max_memory_reserved_gb: 13.59765625
response_length/mean: 17.0
response_length/max: 26.0
response_length/min: 9.0
response_length/clip_ratio: 0.0
response/aborted_ratio: 0.0
prompt_length/mean: 133.25
prompt_length/clip_ratio: 0.0
num_turns/mean: 2.0
timing_s/gen: 0.813267519697547
timing_s/update_actor: 5.002230037003756
timing_s/update_weights: 1.7816048292443156
timing_s/step: 18.177448104135692
perf/throughput: 8.265736705132744
```

## Validation

Post-run checks:

```text
tmux session: no active session after completion
GPU 0: 3506 MiB / 32607 MiB
GPU 1: 3505 MiB / 32607 MiB
GPU 2: 3507 MiB / 32607 MiB
GPU 5: 3493 MiB / 32607 MiB
log lines: 960
train rows: 4
val rows: 2
training/global_step: 1 present
finished_at present
Final validation metrics: None
```

The broad `Traceback|RuntimeError|ERROR|Exception` scan found 11 matches. They
occurred after `Training Progress: 100%`, after the `step:1` metrics, and during
shutdown:

```text
Exception ignored in: _StatefulMultiProcessingDataLoaderIter.__del__
RuntimeError: DataLoader worker (...) is killed by signal: Killed.
resource_tracker: process died unexpectedly, relaunching. Some resources might leak.
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

This is the same cleanup warning pattern observed in Phase 5B and Phase 5E.

## Analysis

Phase 5F fixes the immediate objective mismatch from Phase 5E. The initial
validation reward changed from the Phase 5E negative score:

```text
Phase 5E val score: -0.03
Phase 5F val score: 1.085
```

The split reward confirms both stages are being evaluated:

```text
search_reward mean: 0.5
answer_reward mean: 0.5
format_reward mean: 1.0
```

The rollout is still not a full retrieval-environment RL loop:

```text
tool_calls/mean: 0.0
```

That is expected for this two-stage offline prompt smoke. It validates that the
SFT checkpoint and reward hook are aligned before adding real environment
insertion into rollout.

## Next Steps

- Dump the generated Phase 5F validation/train responses to verify the exact
  search and answer texts, not only aggregate rewards.
- Add a real offline environment rollout path where `<search>` triggers BM25
  observation insertion and the same trajectory receives stage-aware credit.
- Keep runs tiny until shutdown warnings are either accepted as known benign
  behavior or reduced by worker/shutdown configuration changes.
