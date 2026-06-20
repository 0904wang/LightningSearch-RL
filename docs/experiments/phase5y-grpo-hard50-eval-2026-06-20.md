# Phase 5Y GRPO Hard50 Eval

Date: 2026-06-20

## Goal

Evaluate the Phase 5Y variance-filtered rank-reward GRPO checkpoint on the hard50 held-out slice and compare deterministic behavior against the Phase 5D turn-level SFT baseline.

## Command

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260620-phase5y-hard50-eval -c /data/wzl/LightningSearch-RL/repo 'env CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5y_grpo_hard50_eval.sh'"
```

## Runtime

- Repo: `/data/wzl/LightningSearch-RL/repo`
- Branch: `main`
- Commit: `de038e0745c4d83e5e8bbd3b0a03c5390ae86e93`
- Env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- GPU: `CUDA_VISIBLE_DEVICES=7`
- Started: `2026-06-20T01:17:14+00:00`
- Finished: `2026-06-20T01:20:12+00:00`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5y-grpo-hard50-eval.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-eval`

## Inputs

- Eval SFT data: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl`
- Index: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json`
- SFT baseline model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- GRPO FSDP checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward/global_step_28/actor`
- Merged GRPO HF checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward/hf_merged_global_step_28`

## Eval Settings

- Offset: `400`
- Limit: `100`
- Candidate pool: `gold-distractors`
- Distractor count: `50`
- Top-k: `8`
- Max new tokens: `64`
- Sampling: deterministic

## Results

| Metric | SFT baseline | GRPO step28 | Delta |
|---|---:|---:|---:|
| Valid search action rate | 1.000 | 1.000 | 0.000 |
| Valid answer action rate | 1.000 | 1.000 | 0.000 |
| Answer exact match | 0.660 | 0.660 | 0.000 |
| Answer containment match | 0.680 | 0.680 | 0.000 |
| Answer token F1 | 0.719429 | 0.719429 | 0.000000 |
| Gold evidence recall | 0.855 | 0.855 | 0.000 |
| All gold evidence retrieved rate | 0.710 | 0.710 | 0.000 |
| Avg observation doc count | 8.000 | 8.000 | 0.000 |

Diff summary:

- Changed answers: `0`
- Changed search queries: `0`
- Exact improvements: `0`
- Exact regressions: `0`
- F1 improvements: `0`
- F1 regressions: `0`

Diagnostics:

- SFT suspicious rows: `17 / 100`
- GRPO suspicious rows: `17 / 100`
- Suspicious-adjusted exact match: `0.795181` for both models

## Log Notes

- FSDP actor merge succeeded and wrote a 7.6G HF checkpoint.
- Both SFT and GRPO eval loaded 2 HF shards successfully.
- The tokenizer emitted a known Mistral regex warning from the tokenizer loader path. This warning did not stop evaluation.
- Generation emitted a warning that `temperature`, `top_p`, and `top_k` may be ignored under deterministic settings.

## Analysis

This eval is a negative movement result. The Phase 5Y GRPO checkpoint did not change deterministic hard50 behavior relative to the SFT baseline: answer strings, search queries, and all measured metrics are identical across 100 held-out examples.

The most likely explanation is that the 28-step variance-filtered run produced some nonzero advantage updates during training, but the resulting policy movement was too small to alter greedy decoding. This is consistent with the earlier Phase 5Y training diagnostics: only 11 of 28 steps had nonzero advantage / grad signal, and many rollout groups still collapsed to identical rewards.

The practical conclusion is that Phase 5Y confirmed the infrastructure works but did not produce an eval-level policy improvement. Next work should focus on increasing train-time behavioral pressure before scaling epochs:

- Use stochastic eval to check whether the policy distribution shifted even when greedy output is unchanged.
- Add a policy-movement diagnostic against SFT for Phase 5Y step28.
- Improve reward variance further, especially search-stage variance, or move to a pairwise preference / DPO-style objective for generated search and answer alternatives.
- Consider more training data only after there is measurable movement in sampled behavior or logprob diagnostics.

## Artifacts

- Comparison: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-eval/comparison_summary.json`
- SFT summary: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-eval/sft_baseline/summary.json`
- GRPO summary: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-eval/grpo_global_step_28/summary.json`
- SFT rollouts: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-eval/sft_baseline/env_rollouts.jsonl`
- GRPO rollouts: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-eval/grpo_global_step_28/env_rollouts.jsonl`
