# Phase 6E GRPO Warmstart Hard50 Eval

## Goal

Evaluate whether the Phase 6D GDPO warmup checkpoint or the Phase 6E GDPO -> GRPO warm-start checkpoint changes deterministic hard50 rollout behavior compared with the Phase 5D SFT baseline.

This run compares:

- SFT baseline: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- Phase 6D GDPO: `/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28`
- Phase 6E GRPO warm-start: `/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/hf_merged_global_step_28`

## Code And Sync

- Local launcher commit: `9aaf84e feat: add phase6e hard50 eval launcher`
- Local verification before remote sync: `python -m pytest -q` -> `175 passed, 1 skipped`
- Remote GitHub sync failed twice:
  - `HTTP/2 stream 1 was not closed cleanly before end of the underlying stream`
  - SSH connection timeout
- Fallback sync: narrow sync of:
  - `scripts/remote/phase6e_grpo_warmstart_hard50_eval.sh`
  - `tests/test_remote_launchers.py`
- Remote verification after narrow sync: `PYTHONNOUSERSITE=1 python -m pytest -q` -> `179 passed`
- Remote script checks:
  - `bash -n scripts/remote/phase6e_grpo_warmstart_hard50_eval.sh`
  - `inspect-env-rollout --dry-run` on 2 SFT examples

## Launch

Approved launch command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260621-phase6e-hard50-eval -c /data/wzl/LightningSearch-RL/repo "env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase6e_grpo_warmstart_hard50_eval.sh"'
```

Runtime:

- Remote start: `2026-06-20T16:15:04+00:00`
- Remote finish: `2026-06-20T16:19:41+00:00`
- Asia/Shanghai calendar date: `2026-06-21`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- tmux session: `lightningsearch-20260621-phase6e-hard50-eval`

## Paths

- Log: `/data/wzl/LightningSearch-RL/logs/phase6e-grpo-warmstart-hard50-eval.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval`
- Comparison: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval/comparison_summary.json`
- SFT summary: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval/sft_baseline/summary.json`
- SFT rollouts: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval/sft_baseline/env_rollouts.jsonl`
- Phase 6D summary: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval/phase6d_gdpo_global_step_28/summary.json`
- Phase 6D rollouts: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval/phase6d_gdpo_global_step_28/env_rollouts.jsonl`
- Phase 6E summary: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval/phase6e_grpo_global_step_28/summary.json`
- Phase 6E rollouts: `/data/wzl/LightningSearch-RL/results/phase6e-grpo-warmstart-hard50-eval/phase6e_grpo_global_step_28/env_rollouts.jsonl`

## Configuration

- Offset: `400`
- Limit: `100`
- Retrieval top-k: `8`
- Candidate pool: `gold-distractors`
- Distractor count: `50`
- Max new tokens: `64`
- Decoding: deterministic default; no `--do-sample`

The launcher merged the Phase 6E FSDP actor checkpoint before evaluation:

```bash
PYTHONNOUSERSITE=1 python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/global_step_28/actor \
  --target_dir /data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/hf_merged_global_step_28 \
  --use_cpu_initialization
```

The Phase 6D merged checkpoint already existed and was reused.

## Raw Result Summary

All three models produced identical deterministic hard50 metrics:

```json
{
  "sft_baseline": {
    "valid_search_action_rate": 1.0,
    "valid_answer_action_rate": 1.0,
    "answer_exact_match_rate": 0.66,
    "answer_containment_match_rate": 0.68,
    "answer_token_f1": 0.719429,
    "gold_evidence_recall": 0.855,
    "all_gold_evidence_retrieved_rate": 0.71,
    "assistant_observation_rate": 0.0,
    "avg_observation_doc_count": 8.0
  },
  "phase6d_gdpo_global_step_28": {
    "valid_search_action_rate": 1.0,
    "valid_answer_action_rate": 1.0,
    "answer_exact_match_rate": 0.66,
    "answer_containment_match_rate": 0.68,
    "answer_token_f1": 0.719429,
    "gold_evidence_recall": 0.855,
    "all_gold_evidence_retrieved_rate": 0.71,
    "assistant_observation_rate": 0.0,
    "avg_observation_doc_count": 8.0
  },
  "phase6e_grpo_global_step_28": {
    "valid_search_action_rate": 1.0,
    "valid_answer_action_rate": 1.0,
    "answer_exact_match_rate": 0.66,
    "answer_containment_match_rate": 0.68,
    "answer_token_f1": 0.719429,
    "gold_evidence_recall": 0.855,
    "all_gold_evidence_retrieved_rate": 0.71,
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

Answer diagnostics:

```json
{
  "example_count": 100,
  "answer_exact_match_rate": 0.66,
  "answer_containment_match_rate": 0.68,
  "answer_token_f1": 0.719429,
  "suspicious_count": 17,
  "suspicious_adjusted_example_count": 83,
  "suspicious_adjusted_exact_match_rate": 0.795181
}
```

## Log Notes

The log completed normally:

```text
== evaluate sft baseline ==
== evaluate phase6d gdpo warmup ==
== evaluate phase6e grpo warmstart ==
finished_at=2026-06-20T16:19:41+00:00
```

Common warnings appeared:

- Qwen tokenizer regex warning from Hugging Face tokenizers.
- Generation flags warning for deterministic generation.

These did not stop the run, and all expected summaries and rollout files were written.

## Analysis

This evaluation shows that the Phase 6D and Phase 6E checkpoints did not change greedy rollout behavior on the deterministic hard50 slice. The result is stronger than just "no metric gain": the generated search actions and final answers were byte-level unchanged under the comparison logic for all 100 examples.

This does not contradict the Phase 6E training diagnostics. Phase 6E increased nonzero GRPO update steps from `3/28` in Phase 6D to `16/28`, so optimizer signal existed. However, the updates were not large enough, not targeted enough, or not directionally aligned enough to move the model's deterministic argmax actions.

The likely reasons are:

- The run is short: only `28` optimizer steps at `lr=1e-6`.
- Search-stage reward variance remains low: Phase 6E had search variable group rate `0.070175`.
- Deterministic decoding is insensitive to small logit shifts; policy movement may only appear under stochastic sampling or logprob comparison.
- The hard50 eval set may be dominated by stable answer templates learned during SFT.

The resume framing should not claim eval improvement from Phase 6D/6E. It can claim:

- GDPO -> GRPO integration completed end to end.
- GRPO after GDPO generated more nonzero advantage / gradient steps.
- Deterministic heldout eval revealed no behavioral movement, leading to diagnosis that stronger search-stage variance and policy-movement diagnostics are needed.

## Next Steps

1. Run a small stochastic eval for SFT vs Phase 6D vs Phase 6E using the same 20-example style as Phase 5Y stochastic eval. This checks whether policy movement appears when sampling exposes logit changes.
2. Run policy movement diagnostics for Phase 6E vs SFT and Phase 6D using heldout stage prompts: parameter diff, chosen-action logprob deltas, and KL on search / answer prompts.
3. If stochastic eval and logprob diagnostics still show no movement, do not extend Phase 6E. Instead, improve data and reward:
   - higher-search-variance transition filtering,
   - more search-vs-search preference pairs,
   - harder negative query candidates,
   - potentially higher LR or more steps only after confirming logprob movement.
