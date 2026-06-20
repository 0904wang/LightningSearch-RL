# Phase 5Y Policy Movement Diagnostic

Date: 2026-06-20

## Goal

Diagnose whether the Phase 5Y variance-filtered rank-reward GRPO checkpoint moved the policy distribution relative to the Phase 5D SFT baseline, after deterministic and stochastic hard50 evaluation both showed unchanged behavior.

## Command

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260620-phase5y-policy-movement -c /data/wzl/LightningSearch-RL/repo 'env CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5y_policy_movement_diag.sh'"
```

## Runtime

- Repo: `/data/wzl/LightningSearch-RL/repo`
- Branch: `main`
- Commit before launch: `26b6ec92961f85a00d4f07b1dc9a770e20bb2eeb`
- Env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- GPU: `CUDA_VISIBLE_DEVICES=7`
- Started: `2026-06-20T02:08:12+00:00`
- Finished: `2026-06-20T02:11:46+00:00`
- Log: `/data/wzl/LightningSearch-RL/logs/phase5y-policy-movement-diag.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag`

## Inputs

- SFT target data: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl`
- Base model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- Candidate model: `/data/wzl/LightningSearch-RL/checkpoints/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward/hf_merged_global_step_28`

## Diagnostic Settings

- Offset: `400`
- Limit: `20`
- Prompt count: `40`
- Stage counts: `answer=20`, `search=20`
- Device: `cuda`
- Dtype: `bfloat16`
- Top tensor changes: `30`

## Results

Parameter movement:

| Metric | Value |
|---|---:|
| Compared tensors | 398 |
| Changed tensors | 313 |
| Unchanged tensors | 85 |
| Missing tensors | 0 |
| Extra tensors | 0 |
| Relative L2 diff | 2.1334952670171848e-05 |
| Mean absolute diff | 5.3766e-08 |
| Max absolute diff | 2.193450927734375e-05 |

Logprob movement:

| Metric | Value |
|---|---:|
| Compared records | 40 |
| Base mean logprob | -3.9865798641045324e-05 |
| Candidate mean logprob | -3.796018858983024e-05 |
| Delta mean logprob | 1.9056100512150864e-06 |

Stage-level logprob movement:

| Stage | Rows | Mean delta logprob | Improved | Regressed | Unchanged |
|---|---:|---:|---:|---:|---:|
| answer | 20 | 5.947770762622828e-06 | 13 | 6 | 1 |
| search | 20 | -1.3056187140957302e-07 | 8 | 11 | 1 |

Largest tensor changes were still tiny. The top relative changes were around `3e-05` to `4.5e-05`, mostly in early MLP tensors and several attention key projection tensors.

## Analysis

This diagnostic confirms that Phase 5Y did update the model, but the movement was extremely small. The overall parameter relative L2 difference is only about `2.13e-5`, and the mean target-action logprob changed by only `1.9e-6`.

The stage split is important:

- Answer targets moved slightly positive on average.
- Search targets were effectively flat and slightly negative on average.

This explains the earlier deterministic and stochastic evals:

- The policy moved too little to change greedy outputs.
- Sampling at `temperature=0.7`, `top_p=0.9`, and `sample_top_k=40` also showed no changed answers or search queries.
- Search policy did not receive a useful improvement signal, which is especially damaging for a retrieval tool-use agent.

Decision-plan classification:

- This is closest to Case B / weak Case C in the Phase 5Z decision plan: measurable but tiny parameter movement, almost no useful target-action logprob movement, and no behavior change.

## Conclusion

Do not extend the same Phase 5Y GRPO recipe as-is. It is unlikely that simply increasing epochs will solve the core issue, because search-stage signal is not improving and behavior did not move under deterministic or stochastic decoding.

Recommended next step:

1. Build pairwise preference data from reward-probe or rollout alternatives with clear chosen/rejected gaps.
2. Prefer pairs where search query or final answer actually differs.
3. Train a small DPO/SimPO-style preference warmup or adapter smoke.
4. Re-evaluate policy movement and sampled behavior before another GRPO run.

## Artifacts

- Summary: `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag/summary.json`
- Parameter diff: `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag/parameter_diff.json`
- Base logprobs: `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag/base_logprobs.json`
- Candidate logprobs: `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag/candidate_logprobs.json`
- Logprob comparison: `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag/logprob_comparison.json`
- Stage prompts: `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag/stage_prompts.jsonl`
