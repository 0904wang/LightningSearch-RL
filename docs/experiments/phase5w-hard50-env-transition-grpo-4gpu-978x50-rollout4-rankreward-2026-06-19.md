# Phase 5W: hard50 env-transition GRPO rollout4 rank reward, 50 steps

Date: 2026-06-19

## Goal

Test whether replacing the previous search-stage reward with a dynamic evidence rank reward gives GRPO more useful group-level advantage signal.

Phase 5W changes the reward path so each generated `<search>...</search>` query is evaluated against the per-example candidate passage pool. The reward rebuilds a small lexical retriever from `candidate_passages`, ranks the candidate passages by the generated query, and rewards search actions when gold evidence appears near the top.

Main settings:

- rollout_n: 4
- rollout_temperature: 1.2
- rollout_top_p: 0.95
- rollout_top_k: 50
- search_reward_top_k: 8
- answer_token_f1_threshold: 0.5
- total_training_steps: 50
- train_batch_size: 4
- n_gpus_per_node: 4

## Code and environment

- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- Local branch when prepared: `master`
- Local base commit when prepared: `44493db04f0c8eb761c950a9d5322786c78c491e`
- Sync method: narrow file sync to the approved remote workspace
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- GPUs: `4,5,6,7`
- Model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`

The local worktree had ongoing uncommitted project files when this run was prepared, so the commit above is a base reference, not a complete reproducibility snapshot.

## Transition export

Remote session:

```bash
tmux new-session -d -s lightningsearch-20260619-phase5w-export-transitions 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5w_export_transitions_rankreward_from_5s500_hard50_filtered.sh'
```

Launcher:

```bash
/data/wzl/LightningSearch-RL/repo/scripts/remote/phase5w_export_transitions_rankreward_from_5s500_hard50_filtered.sh
```

Inputs and outputs:

- Rollouts: `/data/wzl/LightningSearch-RL/results/phase5s-env-rollout-gold-distractors-500-hard50/env_rollouts.jsonl`
- Index: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json`
- Quality manifest: `/data/wzl/LightningSearch-RL/repo/configs/data_quality/phase5r_500_known_mismatches.json`
- Transition output: `/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1`
- Transition log: `/data/wzl/LightningSearch-RL/logs/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1.log`

Raw transition summary:

```text
input_example_count=500
excluded_example_count=11
example_count=489
transition_count=978
rollout_count=489
avg_candidate_passage_count=52.0
avg_total_reward=0.977557
avg_search_credit=0.233804
avg_answer_credit=0.743753
valid_search_action_rate=1.0
valid_answer_action_rate=0.989775
answer_exact_match_rate=0.631902
answer_containment_match_rate=0.650307
answer_token_f1=0.69946
gold_evidence_recall=0.819018
```

Candidate passage check:

```text
search_rows=493
candidate_passage_min=52
candidate_passage_max=52
candidate_passage_mean=52.0
```

`search_rows` is slightly larger than `example_count` because a few later-step actions are still classified as search-type actions.

## GRPO launch

Remote session:

```bash
tmux new-session -d -s lightningsearch-20260619-phase5w-grpo-rankreward-50 'CUDA_VISIBLE_DEVICES=4,5,6,7 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5w_hard50_env_transition_grpo_4gpu_978x50_rollout4_rankreward.sh'
```

Launcher:

```bash
/data/wzl/LightningSearch-RL/repo/scripts/remote/phase5w_hard50_env_transition_grpo_4gpu_978x50_rollout4_rankreward.sh
```

Config and artifacts:

- Config: `configs/experiments/phase5w_hard50_env_transition_grpo_4gpu_978x50_rollout4_rankreward.yaml`
- Results: `/data/wzl/LightningSearch-RL/results/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward.log`
- Checkpoints: `/data/wzl/LightningSearch-RL/checkpoints/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward/reward_dump.jsonl`

Dry run before launch wrote:

```text
train_rows=782
val_rows=196
```

## Final status

Fresh verification on 2026-06-19 confirmed:

```text
completed=True
training_progress_100_seen=True
final_step=50
fatal_marker_count=0
shutdown_warning_count=2
started_at=2026-06-19T12:26:22+00:00
finished_at=2026-06-19T12:38:40+00:00
```

Checkpoint verification:

```text
/data/wzl/LightningSearch-RL/checkpoints/phase5w-hard50-env-transition-grpo-4gpu-978x50-rollout4-rankreward/global_step_50 exists
```

GPU state after completion returned to baseline for the selected cards:

```text
4, 3494 MiB, 32607 MiB
5, 3493 MiB, 32607 MiB
6, 3505 MiB, 32607 MiB
7, 18 MiB, 32607 MiB
```

There were two shutdown warnings after the progress bar reached 100 percent and after the checkpoint was written:

```text
RuntimeError: DataLoader worker (pid 3626676) is killed by signal: Killed.
ERROR 06-19 12:38:35 [core_client.py:600] Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

These are recorded as teardown warnings because `fatal_marker_count=0`, the run reached `50/50`, and `global_step_50` exists.

## Metrics

Initial validation metrics:

```text
val reward mean@1=0.8092711559895959
val score mean@1=0.8092711428571429
val answer_reward mean@1=0.3666180816326531
val search_reward mean@1=0.35765306122448975
val evidence_rank_reward mean@1=0.35765306122448975
val retrieved_gold_count mean@1=0.8520408163265306
val format_reward mean@1=1.0
```

Training reward curve:

```text
train_steps=50
reward_mean_all=0.736099
reward_last10=0.76025
reward_min=0.251667
reward_max=1.0675
nonzero_adv_count=7
nonzero_grad_count=7
nonzero_adv_steps=4,8,24,30,35,41,47
nonzero_grad_steps=4,8,24,30,35,41,47
```

Reward dump summary:

```text
row_count=996
stage_counts.search=546
stage_counts.answer=450
overall_score_mean=0.750498

search.score_mean=0.722564
search.search_reward_mean=0.653297
search.evidence_rank_reward_mean=0.653297
search.format_reward_mean=0.992674
search.invalid_action_count=0
search.variable_group_count=1
search.group_count=111
search.variable_group_rate=0.009009

answer.score_mean=0.784392
answer.answer_reward_mean=0.684392
answer.format_reward_mean=1.0
answer.invalid_action_count=0
answer.answer_reward_type_counts.exact=279
answer.answer_reward_type_counts.token_f1=44
answer.answer_reward_type_counts.containment=7
answer.answer_reward_type_counts.none=120
answer.variable_group_count=7
answer.group_count=88
answer.variable_group_rate=0.079545
```

Batch diagnostics:

```text
batch_count=196
train_rows=782
low_reward_row_count=286
alignment_assumption=contiguous train_jsonl order; actual verl dataloader shuffling may differ
```

## Analysis

This run is not a failed experiment. It validates that the dynamic search reward path works end to end inside verl GRPO: generated search queries are scored with `evidence_rank_reward`, reward dumps contain the new component, and the run produced a valid checkpoint.

It also confirms that the training signal is still sparse. `rollout_n=4` plus rank reward produced nonzero advantage and gradient on 7 of 50 steps, which is better than the zero-movement behavior seen in the Phase 5U policy-movement diagnosis. However, search-stage group score variance is only 1 of 111 groups, or 0.9 percent. Answer-stage variance is higher at 7 of 88 groups, or 8.0 percent, but still sparse.

The main bottleneck is no longer infrastructure or reward plumbing. The bottleneck is that most same-question rollout groups still generate equivalent actions and receive identical or near-identical rewards. GRPO can only learn on the minority of groups where rollout samples differ enough for the reward model to create relative advantages.

## Next steps

Recommended next step is a targeted data and sampling pass before a much longer run:

1. Build a variance-filtered GRPO set by pre-sampling rollout groups and keeping examples whose search or answer rewards differ across samples.
2. Add more fine-grained search reward terms, such as query entity coverage and relation-term coverage, so semantically better queries can receive partial credit even when both queries retrieve gold evidence.
3. Run a short 50-step verification on the variance-filtered set.
4. If nonzero advantage frequency improves materially, run a 200-step Phase 5W-long job.

A direct 200-step continuation is possible, but based on this 50-step result it would likely spend most updates on zero-advantage batches unless the training set is filtered or the reward is made more sensitive.
