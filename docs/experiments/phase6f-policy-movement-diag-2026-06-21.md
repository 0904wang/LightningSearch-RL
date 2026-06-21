# Phase 6F Policy Movement Diagnostics

## Goal

Diagnose whether the Phase 6D GDPO warmup checkpoint and Phase 6E GDPO -> GRPO warm-start checkpoint actually moved policy logits relative to the Phase 5D SFT baseline.

This run was launched after deterministic and stochastic hard50 rollout evals showed no observable behavior change: `changed_search_count=0` and `changed_answer_count=0` for SFT vs Phase 6D vs Phase 6E. Phase 6F checks whether this was because decoding was insensitive to small shifts, or because the policy barely moved.

Compared checkpoints:

- SFT baseline: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- Phase 6D GDPO: `/data/wzl/LightningSearch-RL/checkpoints/phase6d-gdpo-warmup-from-phase5y/hf_merged_global_step_28`
- Phase 6E GRPO warm-start: `/data/wzl/LightningSearch-RL/checkpoints/phase6e-grpo-warmstart-from-phase6d-gdpo/hf_merged_global_step_28`

## Code And Sync

- Local launcher commit: `4519b79 feat: add phase6f policy movement launcher`
- Local verification:
  - Target RED test before implementation failed because `scripts/remote/phase6f_policy_movement_diag.sh` did not exist.
  - Target GREEN test: `python -m pytest tests/test_remote_launchers.py::test_phase6f_policy_movement_launcher_compares_sft_gdpo_and_grpo_warmstart -q -p no:cacheprovider --basetemp .pytest-tmp-phase6f-green` -> `1 passed`
  - Launcher test file: `7 passed`
  - Full local suite: `177 passed, 1 skipped`
- Remote repo state was still dirty from prior narrow sync work and behind GitHub `origin/main`, so this run used narrow sync for:
  - `scripts/remote/phase6f_policy_movement_diag.sh`
  - `tests/test_remote_launchers.py`
- Remote verification:
  - `bash -n scripts/remote/phase6f_policy_movement_diag.sh`
  - Target remote launcher test -> `1 passed`
  - Full remote suite -> `181 passed`
- Dry-run:
  - `offset=400`, `limit=2`
  - wrote `4` prompts: `2` search and `2` answer

## Launch

Approved launch command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260621-phase6f-policy-movement -c /data/wzl/LightningSearch-RL/repo "env CUDA_VISIBLE_DEVICES=7 PYTHONNOUSERSITE=1 bash /data/wzl/LightningSearch-RL/repo/scripts/remote/phase6f_policy_movement_diag.sh"'
```

Runtime:

- Remote start: `2026-06-21T04:56:27+00:00`
- Remote finish: `2026-06-21T04:57:45+00:00`
- Asia/Shanghai calendar date: `2026-06-21`
- GPU: `CUDA_VISIBLE_DEVICES=7`
- tmux session: `lightningsearch-20260621-phase6f-policy-movement`

## Paths

- Log: `/data/wzl/LightningSearch-RL/logs/phase6f-policy-movement-diag.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase6f-policy-movement-diag`
- Comparison summary: `/data/wzl/LightningSearch-RL/results/phase6f-policy-movement-diag/comparison_summary.json`
- SFT vs 6D: `/data/wzl/LightningSearch-RL/results/phase6f-policy-movement-diag/sft_vs_phase6d_gdpo_global_step_28`
- SFT vs 6E: `/data/wzl/LightningSearch-RL/results/phase6f-policy-movement-diag/sft_vs_phase6e_grpo_global_step_28`
- 6D vs 6E: `/data/wzl/LightningSearch-RL/results/phase6f-policy-movement-diag/phase6d_gdpo_vs_phase6e_grpo`

Each comparison directory contains:

- `stage_prompts.jsonl`
- `parameter_diff.json`
- `base_logprobs.json`
- `candidate_logprobs.json`
- `logprob_comparison.json`
- `summary.json`

## Configuration

- SFT turns: `/data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl`
- Offset: `400`
- Limit: `20`
- Prompt count per comparison: `40`
- Stages: `20` search prompts and `20` answer prompts
- Device: `cuda`
- dtype: `bfloat16`
- Top tensor changes retained: `30`

## Raw Result Summary

### SFT -> Phase 6D GDPO

```json
{
  "parameter_diff": {
    "compared_tensors": 398,
    "changed_tensors": 307,
    "unchanged_tensors": 91,
    "relative_l2_diff": 8.357560475136979e-06,
    "mean_abs_diff": 1.312e-08,
    "max_abs_diff": 9.059906005859375e-06
  },
  "logprob_comparison": {
    "compared_records": 40,
    "delta_mean_logprob": 9.293372233466294e-07,
    "search": {
      "row_count": 20,
      "mean_delta_logprob": -4.4275774232499785e-07,
      "improved_count": 5,
      "regressed_count": 15,
      "unchanged_count": 0
    },
    "answer": {
      "row_count": 20,
      "mean_delta_logprob": 4.0121407148329334e-06,
      "improved_count": 12,
      "regressed_count": 8,
      "unchanged_count": 0
    }
  }
}
```

### SFT -> Phase 6E GRPO Warm-Start

```json
{
  "parameter_diff": {
    "compared_tensors": 398,
    "changed_tensors": 313,
    "unchanged_tensors": 85,
    "relative_l2_diff": 2.283720109333364e-05,
    "mean_abs_diff": 5.5098e-08,
    "max_abs_diff": 3.24249267578125e-05
  },
  "logprob_comparison": {
    "compared_records": 40,
    "delta_mean_logprob": 3.3796879110068338e-06,
    "search": {
      "row_count": 20,
      "mean_delta_logprob": -4.832005277076902e-07,
      "improved_count": 5,
      "regressed_count": 15,
      "unchanged_count": 0
    },
    "answer": {
      "row_count": 20,
      "mean_delta_logprob": 1.2388389486121351e-05,
      "improved_count": 16,
      "regressed_count": 4,
      "unchanged_count": 0
    }
  }
}
```

### Phase 6D GDPO -> Phase 6E GRPO Warm-Start

```json
{
  "parameter_diff": {
    "compared_tensors": 398,
    "changed_tensors": 313,
    "unchanged_tensors": 85,
    "relative_l2_diff": 2.0617502289752145e-05,
    "mean_abs_diff": 5.1868e-08,
    "max_abs_diff": 2.4318695068359375e-05
  },
  "logprob_comparison": {
    "compared_records": 40,
    "delta_mean_logprob": 2.4503506876602044e-06,
    "search": {
      "row_count": 20,
      "mean_delta_logprob": -4.04427853826924e-08,
      "improved_count": 10,
      "regressed_count": 10,
      "unchanged_count": 0
    },
    "answer": {
      "row_count": 20,
      "mean_delta_logprob": 8.37624877128842e-06,
      "improved_count": 12,
      "regressed_count": 7,
      "unchanged_count": 1
    }
  }
}
```

## Log Notes

The run completed normally:

```text
== diagnose sft vs phase6d gdpo ==
== diagnose sft vs phase6e grpo ==
== diagnose phase6d gdpo vs phase6e grpo ==
finished_at=2026-06-21T04:57:45+00:00
```

Warnings:

- Qwen tokenizer regex warning from Hugging Face tokenizers.
- `torch_dtype` deprecation warning.

These warnings did not stop the run, and all expected output files were written.

## Analysis

Phase 6F confirms that the checkpoints did move numerically, but the movement is very small. Parameter relative L2 moved from `8.36e-06` after GDPO to `2.28e-05` after GDPO -> GRPO. This is real movement, but it is not large enough to change decoded search or answer strings in the previous heldout rollout evals.

The most important result is stage asymmetry:

- Search target logprob did not improve. SFT -> 6E search mean delta is slightly negative at `-4.83e-07`, with only `5/20` search prompts improved and `15/20` regressed.
- Answer target logprob improved more clearly. SFT -> 6E answer mean delta is `1.24e-05`, with `16/20` answer prompts improved.
- 6D -> 6E barely changes search target logprob (`-4.04e-08`) but improves answer target logprob (`8.38e-06`).

This explains the Phase 6E evaluation result. The GRPO warm-start did create optimizer signal and shifted weights, but the useful movement is concentrated more on answer targets than search-action targets. Since the prior hard50 rollout already used stable SFT-style one-search trajectories, this small answer-side movement did not change decoded behavior or answer correctness.

The current GDPO -> GRPO line should not be extended directly. More steps on the same transition slice may keep increasing answer-side confidence while leaving the search policy mostly unchanged. The bottleneck is not simply training length; it is search-stage credit assignment and search-query reward variance.

## Next Steps

1. Build a search-focused high-variance preference set:
   - generate multiple candidate search queries per source prompt,
   - score each query with evidence recall / rank reward,
   - keep only groups where chosen vs rejected query score gaps are large,
   - prefer search-only pairs or search-stage transitions.
2. Add or reuse diagnostics that report search-specific preference quality before training:
   - unique query count per source,
   - mean score gap,
   - chosen evidence recall,
   - rejected evidence recall,
   - title-trap / distractor hit rate.
3. Run a small search-focused GRPO or GDPO/GRPO experiment only after the data has enough search-stage variance:
   - `rollout_n=4` or `8`,
   - stronger `search_reward` / `evidence_rank_reward`,
   - lower answer reward influence,
   - evaluate with changed search count, evidence recall, and Phase6F-style logprob diagnostics.
