# Phase 5U: hard50 env-transition GRPO rollout4 diverse soft-answer, 200 steps

Date: 2026-06-18

## Goal

Run a longer checkpointed GRPO training smoke after Phase 5T improved group-level signal with rollout sampling diversity.

This run keeps the Phase 5T settings:

- rollout_n = 4
- rollout_temperature = 1.2
- rollout_top_p = 0.95
- rollout_top_k = 50
- answer_token_f1_threshold = 0.5
- hard50 filtered environment transitions

The main changes from Phase 5T are:

- total_training_steps: 50 -> 200
- save_freq: -1 -> 100

## Launch

Remote session:

```bash
tmux new-session -d -s lightningsearch-20260618-phase5u-hard50-rollout4-diverse-200 'CUDA_VISIBLE_DEVICES=0,1,2,7 bash /data/wzl/LightningSearch-RL/runs/phase5u_hard50_env_transition_grpo_4gpu_978x200_rollout4_diverse_softanswer.sh'
```

Repo and environment:

- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- Local branch when prepared: `master`
- Local base commit when prepared: `44493db04f0c8eb761c950a9d5322786c78c491e`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- GPUs: `0,1,2,7`

Config and outputs:

- Config: `configs/experiments/phase5u_hard50_env_transition_grpo_4gpu_978x200_rollout4_diverse_softanswer.yaml`
- Launcher: `/data/wzl/LightningSearch-RL/runs/phase5u_hard50_env_transition_grpo_4gpu_978x200_rollout4_diverse_softanswer.sh`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer`
- Checkpoints: `/data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer`
- Data: `/data/wzl/LightningSearch-RL/results/phase5s-env-transitions-soft-answer-from-5s500-hard50-filtered-v1/transitions.jsonl`

## Pre-run verification

- Local tests: `149 passed`
- Remote tests: first SSH connection attempt timed out before running tests; automatic retry passed with `149 passed`
- Remote dry run succeeded:
  - `train_rows=782`
  - `val_rows=196`
  - `answer_token_f1_threshold=0.5`
  - command contained rollout4 sampling settings, `trainer.total_training_steps=200`, and `trainer.save_freq=100`

## Final status

Training reached `200/200` and the launcher wrote post-run diagnostics.

Raw status from `metrics_summary.json`:

```text
completed=True
training_progress_100_seen=True
final_step=200
fatal_marker_count=0
shutdown_warning_count=11
started_at=2026-06-18T12:36:26+00:00
finished_at=2026-06-18T13:11:30+00:00
```

GPU state after completion returned to baseline:

```text
0, 3506 MiB, 32607 MiB
1, 3505 MiB, 32607 MiB
2, 3507 MiB, 32607 MiB
7, 18 MiB, 32607 MiB
```

Only the unrelated `tea_extract` tmux session remained after training.

## Checkpoints

Checkpoint tracker:

```text
latest_checkpointed_iteration=200
checkpoint_dirs=global_step_100,global_step_200
```

Checkpoint root:

```text
/data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer
```

Total checkpoint size observed:

```text
47G
```

## Metrics

Reward curve:

```text
reward_points=200
reward_mean_all=0.925184
reward_first10=0.880702
reward_last10=1.03625
reward_min=0.3425
reward_max=1.1
low_reward_count_lt_0_7=22
low_reward_steps_lt_0_7=4,8,24,30,31,32,38,51,71,73,80,98,125,131,132,154,164,167,171,175,183,186
```

Group-level advantage signal:

```text
nonzero_adv_steps_count=27
nonzero_adv_rate=0.135
nonzero_adv_steps=4,8,14,20,24,35,41,47,51,64,68,73,76,81,88,91,106,108,112,128,149,157,170,175,177,189,193
```

Reward dump:

```text
reward_dump_rows=3396
overall_score_mean=0.926415
overall_low_score_count=473
overall_invalid_action_count=0
```

Answer-stage rewards:

```text
answer_rows=1670
answer_score_mean=0.787896
answer_reward_mean=0.688015
answer_type_counts={'containment': 32, 'exact': 1065, 'none': 458, 'token_f1': 115}
answer_none_rate=0.274251
```

Search-stage rewards:

```text
search_rows=1726
search_score_mean=1.06044
search_reward_mean=0.991309
search_low_score_count=15
```

Batch diagnostics:

```text
batch_train_rows=782
batch_count=196
```

## Comparison

Compared with Phase 5T 50-step diverse smoke:

- Phase 5T nonzero advantage rate: `6/50 = 12.0%`
- Phase 5U nonzero advantage rate: `27/200 = 13.5%`
- Phase 5T answer none rate: `119/450 = 26.44%`
- Phase 5U answer none rate: `458/1670 = 27.43%`
- Phase 5T reward mean: `0.936516`
- Phase 5U reward mean: `0.925184`
- Phase 5T last-10 reward mean: `0.967125`
- Phase 5U last-10 reward mean: `1.03625`

Compared with the previous Phase 5S rollout4 line:

- Phase 5S nonzero advantage rate: `14/195 = 7.18%`
- Phase 5U nonzero advantage rate: `27/200 = 13.5%`

## Warnings

The log contains shutdown/cleanup warnings after the progress bar reached `200/200`, including:

```text
RuntimeError: DataLoader worker is killed by signal: Killed
KeyError: '/psm_*'
```

The parser reported `fatal_marker_count=0`, and the launcher still produced metrics, reward dump diagnostics, batch diagnostics, and checkpoints at global steps 100 and 200. Treat this as a post-training teardown warning, not as evidence that the 200-step training body failed.

## Analysis

The rollout4 diverse setup continued to produce more useful GRPO group-level signal than the previous sparse rollout4 setting. The nonzero advantage rate increased from the earlier Phase 5S baseline of 7.18% to 13.5%.

The 200-step run did not clearly improve average reward over the 50-step Phase 5T smoke, but the last-10 reward average rose to 1.03625. Answer none rate stayed close to Phase 5T, so the longer run did not obviously degrade answer behavior in this reward dump.

The main value of this run is that it produced usable checkpoints at steps 100 and 200. The next decision should be based on held-out evaluation, not training reward alone.

## Next steps

1. Evaluate `global_step_100` and `global_step_200` on the hard50 held-out environment.
2. Compare against the SFT warmup checkpoint and the previous Phase 5R/5S GRPO checkpoints.
3. If held-out metrics improve, keep Phase 5U as the current best GRPO line.
4. If held-out metrics do not improve, keep the diverse rollout/reward setting but increase data diversity before extending training further.
