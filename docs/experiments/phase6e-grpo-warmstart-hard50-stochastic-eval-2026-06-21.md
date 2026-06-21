# Phase 6E GRPO Warmstart Hard50 Stochastic Eval

## Goal

Evaluate whether the Phase 6D GDPO warmup checkpoint or the Phase 6E GDPO -> GRPO warm-start checkpoint changes sampled hard50 rollout behavior compared with the Phase 5D SFT baseline.

This run was added after the deterministic hard50 eval showed byte-level identical search actions and answers for SFT, Phase 6D, and Phase 6E. The stochastic eval uses sampling to check whether smaller policy movement appears away from greedy decoding.

Compared checkpoints:

- SFT baseline: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- Phase 6D GDPO: `/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28`
- Phase 6E GRPO warm-start: `/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/hf_merged_global_step_28`

## Code And Sync

- Local launcher commit: `2a46c57 feat: add phase6e stochastic eval launcher`
- Local verification before remote sync: `python -m pytest -q` -> `176 passed, 1 skipped`
- Remote GitHub sync partially failed because the existing remote tree had narrow-sync modifications and the pull hit TLS/network issues.
- Fallback sync: narrow sync of:
  - `scripts/remote/phase6e_grpo_warmstart_hard50_stochastic_eval.sh`
  - `tests/test_remote_launchers.py`
- Remote verification after narrow sync: `PYTHONNOUSERSITE=1 python -m pytest -q` -> `180 passed`
- Remote script checks:
  - `bash -n scripts/remote/phase6e_grpo_warmstart_hard50_stochastic_eval.sh`
  - 2-example stochastic `inspect-env-rollout --dry-run`

## Launch

Approved launch command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260621-phase6e-stochastic-eval -c /data/wzl/LightningSearch-RL/repo "env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase6e_grpo_warmstart_hard50_stochastic_eval.sh"'
```

Runtime:

- Remote start: `2026-06-21T04:13:16+00:00`
- Remote finish: `2026-06-21T04:14:22+00:00`
- Asia/Shanghai calendar date: `2026-06-21`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- tmux session: `lightningsearch-20260621-phase6e-stochastic-eval`

## Paths

- Log: `/data/wzl/LightningSearch-RL/logs/phase6e-grpo-warmstart-hard50-stochastic-eval.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-stochastic-eval`
- Comparison: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-stochastic-eval/comparison_summary.json`
- SFT summary: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-stochastic-eval/sft_baseline_seed20260621/summary.json`
- SFT rollouts: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-stochastic-eval/sft_baseline_seed20260621/env_rollouts.jsonl`
- Phase 6D summary: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-stochastic-eval/phase6d_gdpo_global_step_28_seed20260621/summary.json`
- Phase 6D rollouts: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-stochastic-eval/phase6d_gdpo_global_step_28_seed20260621/env_rollouts.jsonl`
- Phase 6E summary: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-stochastic-eval/phase6e_grpo_global_step_28_seed20260621/summary.json`
- Phase 6E rollouts: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-stochastic-eval/phase6e_grpo_global_step_28_seed20260621/env_rollouts.jsonl`

## Configuration

- Offset: `400`
- Limit: `20`
- Retrieval top-k: `8`
- Candidate pool: `gold-distractors`
- Distractor count: `50`
- Max new tokens: `64`
- Decoding: `do_sample=true`, `temperature=0.7`, `top_p=0.9`, `sample_top_k=40`
- Seed: `20260621`

Both merged checkpoints already existed and were reused:

```text
== merge skipped for phase6d_gdpo global_step_28: existing HF checkpoint found ==
== merge skipped for phase6e_grpo global_step_28: existing HF checkpoint found ==
```

## Raw Result Summary

All three models produced identical stochastic hard50 metrics on this 20-example slice:

```json
{
  "sft_baseline": {
    "valid_search_action_rate": 1.0,
    "valid_answer_action_rate": 1.0,
    "answer_exact_match_rate": 0.7,
    "answer_containment_match_rate": 0.7,
    "answer_token_f1": 0.725,
    "gold_evidence_recall": 0.875,
    "all_gold_evidence_retrieved_rate": 0.75,
    "assistant_observation_rate": 0.0,
    "avg_observation_doc_count": 8.0
  },
  "phase6d_gdpo_global_step_28": {
    "valid_search_action_rate": 1.0,
    "valid_answer_action_rate": 1.0,
    "answer_exact_match_rate": 0.7,
    "answer_containment_match_rate": 0.7,
    "answer_token_f1": 0.725,
    "gold_evidence_recall": 0.875,
    "all_gold_evidence_retrieved_rate": 0.75,
    "assistant_observation_rate": 0.0,
    "avg_observation_doc_count": 8.0
  },
  "phase6e_grpo_global_step_28": {
    "valid_search_action_rate": 1.0,
    "valid_answer_action_rate": 1.0,
    "answer_exact_match_rate": 0.7,
    "answer_containment_match_rate": 0.7,
    "answer_token_f1": 0.725,
    "gold_evidence_recall": 0.875,
    "all_gold_evidence_retrieved_rate": 0.75,
    "assistant_observation_rate": 0.0,
    "avg_observation_doc_count": 8.0
  }
}
```

All metric deltas are zero:

```json
{
  "phase6d_minus_sft": {
    "answer_exact_match_rate": 0.0,
    "answer_token_f1": 0.0,
    "gold_evidence_recall": 0.0
  },
  "phase6e_minus_sft": {
    "answer_exact_match_rate": 0.0,
    "answer_token_f1": 0.0,
    "gold_evidence_recall": 0.0
  },
  "phase6e_minus_phase6d": {
    "answer_exact_match_rate": 0.0,
    "answer_token_f1": 0.0,
    "gold_evidence_recall": 0.0
  }
}
```

Behavior-level diffs are also zero:

```json
{
  "phase6d_vs_sft": {
    "changed_answer_count": 0,
    "changed_search_count": 0,
    "exact_improvement_count": 0,
    "exact_regression_count": 0
  },
  "phase6e_vs_sft": {
    "changed_answer_count": 0,
    "changed_search_count": 0,
    "exact_improvement_count": 0,
    "exact_regression_count": 0
  },
  "phase6e_vs_phase6d": {
    "changed_answer_count": 0,
    "changed_search_count": 0,
    "exact_improvement_count": 0,
    "exact_regression_count": 0
  }
}
```

Answer diagnostics were identical:

```json
{
  "example_count": 20,
  "answer_exact_match_rate": 0.7,
  "answer_containment_match_rate": 0.7,
  "answer_token_f1": 0.725,
  "suspicious_count": 3,
  "suspicious_adjusted_example_count": 17,
  "suspicious_adjusted_exact_match_rate": 0.823529
}
```

The three suspicious rows were:

- `syn-010228`: predicted `Antarctic Research Center`, gold `Oak Ridge National Laboratory`
- `syn-010262`: predicted `Institute for Advanced Studies`, gold `National Science Foundation`
- `syn-010276`: predicted `Finch Archive`, gold `Midwest Historical Society`

## Log Notes

The log completed normally:

```text
== evaluate sft baseline stochastic ==
== evaluate phase6d gdpo warmup stochastic ==
== evaluate phase6e grpo warmstart stochastic ==
finished_at=2026-06-21T04:14:22+00:00
```

Common warnings appeared:

- Qwen tokenizer regex warning from Hugging Face tokenizers.
- CUDA SDPA attention implementation warning.

These did not stop the run, and all expected summaries and rollout files were written.

## Analysis

The stochastic eval gives the same conclusion as the deterministic hard50 eval: Phase 6D and Phase 6E did not change observable rollout behavior relative to the SFT baseline. This is stronger than "no metric gain" because the sampled search queries and final answers also stayed unchanged under the comparison script.

The result suggests the current GDPO -> GRPO line is producing optimizer activity but not enough policy movement to alter generation. Phase 6E did improve training-side signal compared with Phase 6D (`16/28` nonzero advantage / gradient steps versus `3/28`), but the learned delta is not visible in heldout rollouts under either greedy decoding or this first sampled setting.

Likely causes:

- The GRPO run is too short and conservative: `28` optimizer steps at low LR after GDPO.
- Search-stage reward variance is still weak, so the model has little reason to move search-query logits.
- Evaluation prompts are stable SFT-style prompts, and the adapter/model may need explicit logprob diagnostics to detect small movement.
- Some hard50 rows still contain label or question/gold mismatch issues, visible in the suspicious diagnostics.

This means extending Phase 6E directly is not justified yet. The next useful step is not simply more GRPO epochs; it is to measure whether any logprob movement exists and then improve the preference / reward data that should drive query changes.

## Next Steps

1. Add policy-movement diagnostics for heldout search and answer prompts:
   - SFT vs Phase 6D logprob delta,
   - SFT vs Phase 6E logprob delta,
   - Phase 6D vs Phase 6E logprob delta,
   - KL or top-token distribution changes on search-action prompts.
2. If logprob movement is near zero, stop extending this checkpoint line and build stronger search-query supervision:
   - higher-variance search groups,
   - harder negative search actions,
   - search-vs-search preferences with explicit evidence recall reward gaps.
3. If logprob movement exists but decoding remains unchanged, try a small targeted GRPO run with stronger search-action reward weight and evaluate with multiple stochastic seeds.
