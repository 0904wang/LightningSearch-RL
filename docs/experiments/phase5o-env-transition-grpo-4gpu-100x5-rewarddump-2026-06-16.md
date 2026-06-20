# Phase 5O: Env Transition GRPO 4-GPU Reward Dump

## Goal

Repeat Phase 5N with the same `80/20` transition slice and `5` GRPO steps, but
enable train-time reward dumping from `compute_score`. The goal is to explain
the late-step reward drop by inspecting actual scored model responses and reward
components.

## Runtime

date: 2026-06-16
session: `lightningsearch-20260616-phase5o-rewarddump-100x5`
remote repo: `/data/wzl/LightningSearch-RL/repo`
remote repo type: narrow-synced working tree, not a git repo
conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
GPU selection: `CUDA_VISIBLE_DEVICES=0,1,2,5`
config: `configs/experiments/phase5o_env_transition_grpo_4gpu_100x5_rewarddump.yaml`
source transitions: `/data/wzl/LightningSearch-RL/results/phase5l-env-transitions-from-phase5k/transitions.jsonl`
model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`

## Launch

```bash
tmux new-session -d -s lightningsearch-20260616-phase5o-rewarddump-100x5 "bash -lc 'CUDA_VISIBLE_DEVICES=0,1,2,5 bash /data/wzl/LightningSearch-RL/runs/phase5o_env_transition_grpo_4gpu_100x5_rewarddump.sh'"
```

The launch command included:

```text
LIGHTNINGSEARCH_REWARD_DUMP_PATH=/data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/reward_dump.jsonl
LIGHTNINGSEARCH_REWARD_DUMP_MAX_CHARS=1024
```

## Artifacts

```text
log: /data/wzl/LightningSearch-RL/logs/phase5o-env-transition-grpo-4gpu-100x5-rewarddump.log
results: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump
checkpoints: /data/wzl/LightningSearch-RL/checkpoints/phase5o-env-transition-grpo-4gpu-100x5-rewarddump
metrics summary: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/metrics_summary.json
batch diagnostics: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/batch_diagnostics.json
reward dump: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/reward_dump.jsonl
reward dump summary: /data/wzl/LightningSearch-RL/results/phase5o-env-transition-grpo-4gpu-100x5-rewarddump/reward_dump_summary.json
```

Note: `save_freq=-1`, so no model checkpoint is expected.

## Data

```text
source_type: transitions
train_rows: 80
val_rows: 20
reward_dump_rows: 40
```

The reward dump contains both validation and train scoring calls:

```text
answer rows: 21
search rows: 19
```

## Outcome

The run completed all 5 GRPO training steps.

```text
started_at=2026-06-16T10:03:04+00:00
finished_at=2026-06-16T10:05:14+00:00
Training Progress: 100%|...| 5/5
completed: true
final_step: 5
fatal_marker_count: 0
shutdown_warning_count: 11
```

No fatal command markers were found:

```text
CalledProcessError: absent
Error executing job: absent
CUDA out of memory: absent
ValueError: To serve: absent
```

## Metrics

Initial validation:

```text
val-core/lightningsearch_rl/reward/mean@1: 1.0350000370293855
val-aux/lightningsearch_rl/score/mean@1: 1.0350000000000001
val-aux/lightningsearch_rl/answer_reward/mean@1: 0.45
val-aux/lightningsearch_rl/search_reward/mean@1: 0.5
val-aux/lightningsearch_rl/format_reward/mean@1: 1.0
val-aux/lightningsearch_rl/search_cost/mean@1: 0.015000000000000003
val-aux/num_turns/mean: 2.0
```

Training reward curve:

```text
step 1: critic/rewards/mean=1.0775001049041748
step 2: critic/rewards/mean=1.09250009059906
step 3: critic/rewards/mean=1.0850000381469727
step 4: critic/rewards/mean=1.0850000381469727
step 5: critic/rewards/mean=0.8424999713897705
```

Final training step:

```text
training/global_step: 5
critic/score/mean: 0.8424999713897705
critic/score/max: 1.100000023841858
critic/score/min: 0.10000000149011612
critic/rewards/mean: 0.8424999713897705
actor/loss: -0.8757684230804443
actor/grad_norm: 0.0791015625
actor/ppo_kl: 0.0
actor/entropy: 0.0016538179479539394
response_length/mean: 13.0
response_length/clip_ratio: 0.0
response/aborted_ratio: 0.0
prompt_length/mean: 246.75
prompt_length/clip_ratio: 0.0
num_turns/mean: 2.0
perf/time_per_step: 5.094446463976055
perf/throughput: 50.98689363736552
```

Unlike Phase 5N, the parser did not flag a large reward drop because the
step-4 to step-5 delta was about `-0.2425`, just under the configured `0.25`
alert threshold.

## Batch Diagnostics

The contiguous-batch diagnostic still maps step 5 to rows `[16, 20)`:

```text
step 5 aligned batch index: 4
step 5 aligned row range: [16, 20)
step 5 aligned stage_counts: search=2, answer=2
step 5 aligned precomputed_reward_mean: 0.685
step 5 logged_reward_mean: 0.8424999713897705
step 5 low_reward_row_count: 0
```

Overall prepared train rows:

```text
search rows: 40
answer rows: 40
precomputed_step_reward mean: 0.6475
precomputed_total_reward mean: 1.295
low_reward_row_count: 3
```

The three low-reward prepared train rows are all answer rows with label
granularity mismatches:

```text
syn-009012: expected Global Health Research Institute, precomputed_total_reward=0.37
syn-009019: expected Journal of Computational Science, precomputed_total_reward=0.37
syn-009020: expected Golden Quill Award, precomputed_total_reward=0.37
```

## Reward Dump Diagnostics

Reward dump component summary:

```text
row_count: 40
overall score mean: 1.03575
overall invalid_action_count: 0
overall low_score_count: 2

search:
  row_count: 19
  score mean: 1.07
  search_reward mean: 1.0
  format_reward mean: 1.0
  invalid_action_count: 0
  low_score_count: 0

answer:
  row_count: 21
  score mean: 1.004762
  answer_reward mean: 0.904762
  format_reward mean: 1.0
  invalid_action_count: 0
  low_score_count: 2
```

The two low-score scored responses were:

```text
id: syn-009154:1:answer
gold answer: Vienna
model answer: Vienna Conference Center
score: 0.1

id: syn-009020:1:answer
gold answer: Golden Quill
model answer: Golden Quill Award
score: 0.1
```

Both are valid `<answer>` outputs with `format_reward=1.0` and
`answer_reward=0.0`. No invalid search or answer action was found in the reward
dump. This points to answer normalization / label granularity as a real issue in
the current synthetic data and strict reward, not a tool-use format failure.

## Validation

Before launch:

```text
local tests: 126 passed in 3.27s
remote targeted tests: 10 passed in 0.30s
remote dry-run: parquet_written=true, train_rows=80, val_rows=20
reward_dump_path recorded in summary and manifest: true
```

After completion:

```text
tmux sessions: none
GPU 0: 3506 MiB / 32607 MiB
GPU 1: 3505 MiB / 32607 MiB
GPU 2: 3507 MiB / 32607 MiB
GPU 5: 3493 MiB / 32607 MiB
metrics_summary.json: present
batch_diagnostics.json: present
reward_dump.jsonl: present
reward_dump_summary.json: present
fatal markers: none
```

## Analysis

Phase 5O confirms the reward-drop investigation path works. The run reproduced
the same stable first four steps from Phase 5N, while step 5 improved from
`0.5925` to `0.8425`. The reward dump shows tool calls are not the culprit:
search actions are all valid, answer formatting is valid, and the only low
scores are answer exact-match failures caused by acceptable but more specific
answers.

This is important for the project narrative: the framework now exposes the
difference between policy/tool failure and reward/data-label failure. For the
resume project, this supports a concrete evidence-aware debugging story.

## Next Steps

1. Add answer normalization aliases or token-F1/containment credit for answer
   stage rewards so `Golden Quill Award` vs `Golden Quill` and venue/city
   mismatches are not treated as equally bad as hallucinations.
2. Re-export Phase 5L transitions with improved answer reward components, then
   repeat the 5-step GRPO smoke.
3. After the strict-reward issue is reduced, run a longer 50-100 step job with
   reward dump enabled on a small sample budget.
