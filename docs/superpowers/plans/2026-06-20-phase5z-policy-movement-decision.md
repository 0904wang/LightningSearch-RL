# Phase 5Z Policy Movement Decision Plan

Date: 2026-06-20

## Context

Phase 5Y variance-filtered rank-reward GRPO completed and produced a valid checkpoint, but both deterministic and stochastic hard50 evaluations showed no observable behavior change against the Phase 5D SFT baseline:

- Deterministic eval: `changed_answer_count=0`, `changed_search_count=0`
- Stochastic eval: `changed_answer_count=0`, `changed_search_count=0`
- All answer and retrieval metric deltas were `0.0`

The next question is whether GRPO changed the model distribution in a small but behavior-invisible way, or whether the update was effectively negligible.

## Diagnostic

Run Phase 5Y policy movement diagnostic:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260620-phase5y-policy-movement -c /data/wzl/LightningSearch-RL/repo 'env CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase5y_policy_movement_diag.sh'"
```

It compares:

- Base model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- Candidate model: `/data/wzl/LightningSearch-RL/checkpoints/phase5y-hard50-env-transition-grpo-4gpu-126x28-variance-rankreward/hf_merged_global_step_28`
- SFT target prompts: offset `400`, limit `20`, yielding `40` search/answer target prompts

Expected artifacts:

- `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag/parameter_diff.json`
- `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag/logprob_comparison.json`
- `/data/wzl/LightningSearch-RL/results/phase5y-policy-movement-diag/summary.json`

## Decision Rules

### Case A: Near-zero parameter and logprob movement

Evidence pattern:

- `relative_l2_diff` is extremely small.
- `delta_mean_logprob` is near zero.
- Search and answer stage deltas are both near zero.

Interpretation:

GRPO updates were effectively too weak. The current reward/advantage setup is not producing useful pressure.

Next step:

- Do not simply increase epochs.
- Rework training signal first:
  - increase per-group reward variance,
  - make query quality rewards less sparse,
  - add explicit negative rewards for wrong title-copy answers,
  - or build pairwise preference data from rollout alternatives and train DPO/SimPO-style before another GRPO run.

### Case B: Parameter movement exists but logprob movement is near zero

Evidence pattern:

- `relative_l2_diff` is measurable.
- Target action `delta_mean_logprob` is near zero.

Interpretation:

Weights moved, but not in directions that affect search/answer target actions.

Next step:

- Improve credit assignment and reward localization.
- Separate search-action rewards and answer-action rewards more explicitly.
- Prefer transition subsets where the sampled action differs and reward rank is unambiguous.

### Case C: Logprob movement exists but behavior is unchanged

Evidence pattern:

- `delta_mean_logprob` is nontrivial.
- Deterministic and stochastic hard50 outputs remain unchanged.

Interpretation:

The model distribution moved, but not enough to cross decoding decision boundaries.

Next step:

- Run a longer but still controlled training job on the same filtered data.
- Lower KL pressure or increase effective update strength carefully.
- Add a larger stochastic eval with multiple seeds before scaling to a full run.

### Case D: Search-stage and answer-stage movement diverge

Evidence pattern:

- Search `mean_delta_logprob` and answer `mean_delta_logprob` have different signs or magnitudes.

Interpretation:

The current reward composition may favor one stage while leaving the other unchanged or worse.

Next step:

- Split training/evaluation into stage-specific experiments:
  - search-only query policy improvement,
  - answer-only evidence-grounded answer improvement,
  - then recompose into full agent trajectories.

## Current Recommendation

If Phase 5Y policy movement is near-zero, move to a small pairwise preference path instead of another longer GRPO attempt:

1. Reuse Phase 5Y reward-probe samples to construct chosen/rejected action pairs within the same source question.
2. Prefer pairs with different search queries or different final answers and a clear reward gap.
3. Train a small DPO/SimPO-style adapter or full-model smoke run.
4. Re-evaluate deterministic and stochastic hard50 behavior before returning to GRPO.
