# Phase 5Y Reward Probe Variance Pool

Date: 2026-06-19

## Goal

Expand the Phase 5X variance-filtered training pool before running longer GRPO. Instead of adding epochs to the 8-source Phase 5X pool, probe all Phase 5W hard50 transitions with multiple stochastic samples and keep sources whose reward varies within a group.

## Command

Preflight and dry-run passed before launch:

- Local tests: `160 passed, 1 skipped`
- Remote related tests: `36 passed`
- Remote full tests: `164 passed`
- Dry-run: 4 real transitions, 2 samples per prompt, 8 expected reward rows

Successful launch:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260619-phase5y-reward-probe -c /data/wzl/LightningSearch-RL/repo 'env CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5y_reward_probe_variance_pool_from_phase5w.sh'"
```

The first PowerShell launch attempt failed due to local quote escaping before any remote session was created. The command above is the successful equivalent launch.

## Paths

- Repo: `/data/wzl/LightningSearch-RL/repo`
- Env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- Input transitions: `/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl`
- Model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- Probe results: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6`
- Filtered transitions: `/data/wzl/LightningSearch-RL/results/phase5y-env-transitions-variance-rankreward-100src`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5y-reward-probe-rankreward-978x6.log`

## Settings

- GPU: `CUDA_VISIBLE_DEVICES=4`
- Probe backend: `vllm`
- Transitions probed: `978`
- Source count: `489`
- Samples per prompt: `6`
- Max new tokens: `64`
- Temperature: `1.3`
- Top-p: `0.95`
- Top-k: `50`
- Answer token F1 threshold: `0.5`
- Search reward top-k: `8`
- Variance filter max source count: `100`

## Raw Summary

Probe:

```json
{
  "selected_transition_count": 978,
  "source_count": 489,
  "stage_counts": {
    "answer": 485,
    "search": 493
  },
  "samples_per_prompt": 6,
  "generated_sample_count": 5868,
  "reward_dump_count": 5868
}
```

Reward dump:

```json
{
  "row_count": 5868,
  "stage_counts": {
    "answer": 2910,
    "search": 2958
  },
  "overall_score_mean": 0.765693,
  "answer_score_mean": 0.779666,
  "search_score_mean": 0.751947,
  "answer_variable_group_rate": 0.121649,
  "search_variable_group_rate": 0.00818,
  "answer_reward_type_counts": {
    "containment": 44,
    "exact": 1844,
    "none": 824,
    "token_f1": 198
  }
}
```

Filtered pool:

```json
{
  "input_transition_count": 978,
  "output_transition_count": 126,
  "selected_source_count": 63,
  "matched_source_count": 63,
  "unmatched_source_count": 0,
  "stage_variable_group_counts": {
    "answer": 59,
    "search": 4
  }
}
```

Line counts:

```text
978  probe_requests.jsonl
5868 generations.jsonl
5868 reward_dump.jsonl
126  filtered transitions.jsonl
```

## Log Notes

The run started at `2026-06-19T14:14:12+00:00` and finished at `2026-06-19T14:15:00+00:00`.

The log includes a tokenizer regex warning from the Hugging Face tokenizer loader, but the model resolves as `Qwen3ForCausalLM`, generation completed, reward dump was written, and the variance filter finished.

## Analysis

This is a useful improvement over Phase 5X. The variance-filtered pool expanded from 8 selected sources / 16 transitions to 63 selected sources / 126 transitions. That is large enough for a short GRPO diagnostic without repeating the same tiny group for many epochs.

The signal is still mostly answer-stage variance: 59 answer groups versus only 4 search groups. That means the next GRPO run will mostly train answer formatting/correctness behavior, with limited direct search-query advantage signal. It is still a better next step than training longer on the 8-source Phase 5X pool.

The search-stage variance rate is only `0.00818`, so if the next GRPO run shows weak search policy movement, the follow-up should make search prompts more exploratory or increase search-stage sampling diversity, rather than simply adding more epochs.

## Next Step

Prepare a Phase 5Y GRPO diagnostic using `/data/wzl/LightningSearch-RL/results/phase5y-env-transitions-variance-rankreward-100src/transitions.jsonl`. A conservative first config is all 126 rows with 4-GPU GRPO, `rollout_n=4`, `train_batch_size=4`, one epoch or a capped short step count, and checkpoint at the final step.
