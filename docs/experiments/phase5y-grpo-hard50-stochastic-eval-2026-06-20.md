# Phase 5Y GRPO Hard50 Stochastic Eval

Date: 2026-06-20

## Goal

Check whether Phase 5Y variance-filtered rank-reward GRPO changed sampled behavior even though deterministic hard50 evaluation showed no greedy-policy movement.

## Command

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260620-phase5y-stochastic-eval -c /data/wzl/LightningSearch-RL/repo 'env CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5y_grpo_hard50_stochastic_eval.sh'"
```

## Runtime

- Repo: `/data/wzl/LightningSearch-RL/repo`
- Branch: `main`
- Commit: `38a743b9d92456e06cee4bca4dd40658b51a0485`
- Env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- GPU: `CUDA_VISIBLE_DEVICES=7`
- Started: `2026-06-20T01:52:19+00:00`
- Finished: `2026-06-20T01:54:09+00:00`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5y-grpo-hard50-stochastic-eval.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-stochastic-eval`

## Inputs

- Eval SFT data: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl`
- Index: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json`
- SFT baseline model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- GRPO checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward/global_step_28/actor`
- Merged GRPO HF checkpoint: `/data/wzl/LightningSearch-RL/checkpoints/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward/hf_merged_global_step_28`

## Eval Settings

- Offset: `400`
- Limit: `20`
- Candidate pool: `gold-distractors`
- Distractor count: `50`
- Top-k: `8`
- Max new tokens: `64`
- Sampling: `do_sample=true`
- Temperature: `0.7`
- Top-p: `0.9`
- Sample top-k: `40`
- Seed: `20260620`

## Results

| Metric | SFT baseline | GRPO step28 | Delta |
|---|---:|---:|---:|
| Valid search action rate | 1.000 | 1.000 | 0.000 |
| Valid answer action rate | 1.000 | 1.000 | 0.000 |
| Answer exact match | 0.700 | 0.700 | 0.000 |
| Answer containment match | 0.700 | 0.700 | 0.000 |
| Answer token F1 | 0.725 | 0.725 | 0.000 |
| Gold evidence recall | 0.875 | 0.875 | 0.000 |
| All gold evidence retrieved rate | 0.750 | 0.750 | 0.000 |
| Avg observation doc count | 8.000 | 8.000 | 0.000 |

Diff summary:

- Changed answers: `0`
- Changed search queries: `0`
- Exact improvements: `0`
- Exact regressions: `0`
- F1 improvements: `0`
- F1 regressions: `0`

Diagnostics:

- SFT suspicious rows: `3 / 20`
- GRPO suspicious rows: `3 / 20`
- Suspicious-adjusted exact match: `0.823529` for both models

## Log Notes

- The script skipped checkpoint merge because `hf_merged_global_step_28` already existed from deterministic eval.
- Both SFT and GRPO models loaded successfully.
- The tokenizer emitted the known Mistral regex warning from the tokenizer loader path.
- The job ended normally and no `lightningsearch` / `inspect-env-rollout` process remained afterward.
- GPU memory after the job was from unrelated remote workloads, primarily `spt_paper`, not from this eval.

## Analysis

This stochastic eval confirms the deterministic hard50 result: Phase 5Y step28 did not move observable behavior against the SFT baseline on this slice. Even with sampling enabled, the SFT and GRPO rollouts had identical final answers and identical search queries across all 20 evaluated examples.

The result strongly suggests that Phase 5Y's nonzero advantage steps were not enough to produce practical policy movement. The issue is not just greedy decoding hiding a small distributional shift; sampled decoding at `temperature=0.7`, `top_p=0.9`, and `sample_top_k=40` also remained unchanged.

Next work should not simply extend the same 28-step recipe. The better next step is a policy-movement / logprob diagnostic on Phase 5Y step28, followed by reward or objective changes if logprob movement is also near zero. If logprob movement exists but sampled behavior does not change, the training signal is too weak around decision boundaries; if logprob movement is near zero, the update itself is ineffective.

## Artifacts

- Comparison: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-stochastic-eval/comparison_summary.json`
- SFT summary: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-stochastic-eval/sft_baseline_seed20260620/summary.json`
- GRPO summary: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-stochastic-eval/grpo_global_step_28_seed20260620/summary.json`
- SFT rollouts: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-stochastic-eval/sft_baseline_seed20260620/env_rollouts.jsonl`
- GRPO rollouts: `/data/wzl/LightningSearch-RL/results/phase5y-grpo-hard50-stochastic-eval/grpo_global_step_28_seed20260620/env_rollouts.jsonl`
