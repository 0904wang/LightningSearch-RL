# Phase 5R Filtered-v2 978-Transition GRPO 50-Step Run

## Goal

Run a 4-GPU GRPO training smoke on the cleaned Phase 5R filtered-v2 transition
set:

```text
489 rollout examples
978 transitions
782 train rows
196 validation rows
50 training steps
```

## Launch

Session:

```text
lightningsearch-20260617-phase5r-filtered-v2-grpo-978x50
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260617-phase5r-filtered-v2-grpo-978x50 bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,7 bash /data/wzl/LightningSearch-RL/runs/phase5r_filtered_v2_env_transition_grpo_4gpu_978x50_softanswer.sh'"
```

Selected GPUs:

```text
0,1,2,7
```

Prelaunch GPU memory:

```text
0: 3506 MiB
1: 3505 MiB
2: 3507 MiB
7: 18 MiB
```

## Inputs

```text
repo: /data/wzl/LightningSearch-RL/repo
config: configs/experiments/phase5r_filtered_v2_env_transition_grpo_4gpu_978x50_softanswer.yaml
transitions: /data/wzl/LightningSearch-RL/results/phase5r-env-transitions-soft-answer-from-5r500-filtered-v2/transitions.jsonl
model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer.log
results: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer
metrics: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5r-filtered-v2-env-transition-grpo-4gpu-978x50-softanswer/reward_dump_summary.json
```

No model checkpoint was expected because `save_freq: -1`.

## Status

```text
started_at: 2026-06-17T10:11:53+00:00
finished_at: 2026-06-17T10:18:10+00:00
completed: true
final_step: 50
training_progress_100_seen: true
fatal_marker_count: 0
shutdown_warning_count: 11
```

GPU memory was released after completion:

```text
0: 3506 MiB
1: 3505 MiB
2: 3507 MiB
7: 18 MiB
```

No `lightningsearch-*` tmux session remained after completion.

## Metrics

Initial validation:

```text
val reward mean@1: 1.0774684562442862
val answer reward mean@1: 0.49246841836734695
val answer exact match mean@1: 0.4846938775510204
val answer token F1 mean@1: 0.4939261428571429
val answer containment mean@1: 0.49489795918367346
val search reward mean@1: 0.5
val format reward mean@1: 1.0
val num turns mean: 2.0
```

Latest train metrics at step 50:

```text
critic/rewards/mean: 1.0850000381469727
critic/score/mean: 1.0850000381469727
response_length/clip_ratio: 0.0
response/aborted_ratio: 0.0
num_turns/mean: 2.0
```

Batch diagnostics over train data:

```text
train_rows: 782
stage_counts: search=391, answer=391
low_reward_row_count: 0
reward_model_reward mean: 0.681307
precomputed_total_reward mean: 1.362614
```

Reward dump summary:

```text
row_count: 396
overall score mean: 1.080439
invalid_action_count: 0
low_score_count: 1
answer_reward_type_counts: exact=184, containment=2, none=1
search stage rows: 209
answer stage rows: 187
```

Low-score runtime example:

```text
source_id: syn-010327
reward_stage: answer
score: 0.1
answer_reward: 0.0
parsed_action.valid: true
solution_preview: <answer>National Endowment for the Humanities</answer>
```

## Warnings

The log includes tokenizer regex warnings from the local merged checkpoint and
`torch-c-dlpack-ext` optional-extension warnings. These also appeared in earlier
successful runs.

After the progress bar reached `50/50`, the log recorded shutdown-time
DataLoader/resource-tracker/vLLM engine messages, including:

```text
RuntimeError: DataLoader worker ... is killed by signal: Killed.
Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

The parser reported `fatal_marker_count: 0`, `completed: true`, and all
post-run artifacts were generated. Treat these as shutdown cleanup warnings for
this run, not as a failed training job.

## Analysis

This run validates the cleaned Phase 5R transition path at a larger scale than
the previous Phase 5Q 390-transition run. The important checks held:

- 4-GPU verl/GRPO launched from the SFT warm-start checkpoint.
- All 50 configured steps completed.
- Response clipping stayed at `0.0`.
- Runtime invalid action count in reward dump was `0`.
- Batch diagnostics found no low-reward rows in the prepared train split.
- The cleaned source data avoided the repeated low-quality `answer_none` rows
  that motivated filtered-v2.

The runtime reward dump still has one generated answer with `answer_reward=0`.
That is a model rollout error during training, not a known source-data quality
row. It is useful evidence for the next experiment: the model is mostly stable
under the shaped reward but not saturated.

## Next Step

Use this as the successful 50-step scale smoke. The next experiment should be
either:

1. a longer filtered-v2 run, such as 200-500 steps, with the same data and
   reward diagnostics; or
2. another data expansion pass beyond 500 rollout examples before increasing
   training length.

For resume value, the strongest immediate comparison is:

```text
Phase 5Q filtered 390x20 -> Phase 5R filtered-v2 978x50
```

This shows both data-quality filtering and stable larger-scale GRPO execution.
