# Phase 5V Policy Movement Diagnostic

Date: 2026-06-18

## Goal

Diagnose why Phase 5U changed neither deterministic nor stochastic held-out outputs. This run checks whether the Phase 5U `global_step_200` checkpoint moved at all relative to the SFT baseline.

The diagnostic compares:

- Base model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- Candidate model: `/data/wzl/LightningSearch-RL/checkpoints/phase5u-hard50-env-transition-grpo-4gpu-978x200-rollout4-diverse-softanswer/hf_merged_global_step_200`

## Code Change

Added `diagnose-policy-movement` CLI with:

- safetensors parameter diff
- search/answer gold target logprob comparison
- dry-run prompt manifest

Verification:

```text
local tests: 151 passed, 1 skipped
remote related tests: 20 passed
remote full tests: 155 passed
dry-run prompt_count=40, search=20, answer=20
```

## Launch

Session:

```text
lightningsearch-20260618-phase5v-policy-movement-diag
```

Command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5v-policy-movement-diag 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/runs/phase5v_policy_movement_diag.sh'"
```

Runtime:

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: narrow-sync-working-tree
local source branch/commit: master / 44493db04f0c8eb761c950a9d5322786c78c491e
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=7
started_at: 2026-06-18T14:44:25+00:00
finished_at: 2026-06-18T14:44:51+00:00
```

## Inputs

```text
sft rows: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
offset: 400
limit: 20
stage prompts: 40 total, 20 search and 20 answer
dtype: bfloat16
device: cuda
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5v-policy-movement-diag.log
results: /data/wzl/LightningSearch-RL/results/phase5v-policy-movement-diag
summary: /data/wzl/LightningSearch-RL/results/phase5v-policy-movement-diag/summary.json
parameter diff: /data/wzl/LightningSearch-RL/results/phase5v-policy-movement-diag/parameter_diff.json
logprob comparison: /data/wzl/LightningSearch-RL/results/phase5v-policy-movement-diag/logprob_comparison.json
```

GPU7 was released after completion:

```text
7, 18 MiB, 32607 MiB
```

## Parameter Diff

```text
base_tensor_count: 398
candidate_tensor_count: 398
compared_tensors: 398
changed_tensors: 308
unchanged_tensors: 90
total_elements: 4022468096
l2_diff: 0.062136646963914466
base_l2: 1603.2802795474138
relative_l2_diff: 3.875594788794812e-05
mean_abs_diff: 6.8948e-08
max_abs_diff: 6.687641143798828e-05
```

Top tensor changes by relative L2:

```text
model.layers.1.mlp.down_proj.weight: relative_l2_diff=7.5104e-05, max_abs_diff=5.5075e-05
model.layers.1.mlp.up_proj.weight: relative_l2_diff=7.4598e-05, max_abs_diff=5.2989e-05
model.layers.33.self_attn.k_proj.weight: relative_l2_diff=7.3059e-05, max_abs_diff=6.1274e-05
model.layers.34.self_attn.k_proj.weight: relative_l2_diff=6.2145e-05, max_abs_diff=5.5790e-05
model.layers.2.mlp.down_proj.weight: relative_l2_diff=6.1142e-05, max_abs_diff=5.1022e-05
```

## Gold Target Logprob Diff

Overall:

```text
compared_records: 40
base_mean_logprob: -3.9865798641045324e-05
candidate_mean_logprob: -3.520758212451289e-05
delta_mean_logprob: 4.658216516532434e-06
```

By stage:

```text
answer:
  row_count: 20
  mean_delta_logprob: 1.4326137107590595e-05
  improved_count: 14
  regressed_count: 6
  unchanged_count: 0

search:
  row_count: 20
  mean_delta_logprob: 3.7503327381380057e-07
  improved_count: 10
  regressed_count: 10
  unchanged_count: 0
```

## Analysis

The checkpoint is not byte-identical to the SFT baseline. Parameters changed in 308 of 398 tensors. However, the movement is extremely small: the global relative L2 parameter difference is only `3.88e-05`, with mean absolute difference around `6.9e-08`.

The logprob diagnostic agrees with the evaluation results. Phase 5U slightly increases gold answer target logprob on average, especially in the answer stage, but the shift is tiny. Search-stage logprob is almost unchanged and evenly split between improvements and regressions.

This explains why deterministic and stochastic held-out outputs did not change: the policy moved, but not enough to cross decoding boundaries. It also suggests the main bottleneck is not a broken checkpoint save or eval path. The update exists but is too weak and not specifically concentrated on search behavior.

## Conclusion

Phase 5U produced real but very small policy movement. The next improvement should target stronger search-action learning signal, not just longer training.

Recommended next experiment:

```text
Phase 5W: query/evidence-rank reward plus transition filtering for groups with nonzero reward variance.
```

Specific changes:

```text
1. Add query/evidence-rank reward for search-stage actions:
   top1 gold evidence hit > top3 > top8 > miss.
2. Export sampled candidate search actions per question.
3. Keep only groups where rollout_n candidates produce different search rewards.
4. Run a short 50-step GRPO smoke and require:
   nonzero_adv_rate meaningfully above 13.5%,
   actor/grad_norm nonzero on more steps,
   held-out search/query diff no longer 0.
```
