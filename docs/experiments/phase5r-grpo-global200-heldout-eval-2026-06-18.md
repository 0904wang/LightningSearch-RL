# Phase 5R GRPO global_step_200 Held-out Evaluation

## Goal

Evaluate whether the Phase 5R GRPO `global_step_200` checkpoint improves real
agent-loop behavior, not only training reward. The evaluation compares the SFT
warm-start checkpoint against the merged GRPO checkpoint on the same held-out
tail slice in the controlled offline gold+distractor retrieval environment.

## Launch

Session:

```text
lightningsearch-20260618-phase5r-grpo-gs200-eval
```

Final successful command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260618-phase5r-grpo-gs200-eval 'CUDA_VISIBLE_DEVICES=7 bash /data/wzl/LightningSearch-RL/runs/phase5r_grpo_global200_heldout_eval.sh' && echo LAUNCHED_RETRY && tmux list-sessions"
```

The first launch failed before evaluation because the merge input pointed to
`global_step_200`, while this GRPO checkpoint stores actor FSDP shards and
Hugging Face metadata under `global_step_200/actor`. The launcher was corrected
to pass `--local_dir .../global_step_200/actor`.

## Runtime

```text
repo: /data/wzl/LightningSearch-RL/repo
repo state: not-a-git-repository, narrow-sync-working-tree
local source branch/commit: master / 44493db
env: /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl
gpu: CUDA_VISIBLE_DEVICES=7
started_at: 2026-06-18T07:35:22+00:00
finished_at: 2026-06-18T07:38:18+00:00
```

Prelaunch checks:

```text
remote tests: tests/test_environment_rollout.py + inspect-env-rollout CLI dry-run test -> 4 passed
dry-run offset=400 limit=5: gold_evidence_recall=1.0, all_gold_evidence_retrieved_rate=1.0
GPU 7 before launch: 18 MiB / 32607 MiB
```

## Inputs

```text
sft rows: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/sft-turns-gold/sft_turns.jsonl
index: /data/wzl/LightningSearch-RL/results/phase4g-deepseek-titlefix-500-docidfix/index.json
sft baseline model: /data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40
grpo fsdp checkpoint: /data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/global_step_200/actor
grpo merged model: /data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/hf_merged_global_step_200
offset: 400
limit: 100
top_k: 8
candidate_pool: gold-distractors
distractor_count: 6
max_new_tokens: 64
```

Merge artifact:

```text
merged size: 7.6G
target: /data/wzl/LightningSearch-RL/checkpoints/phase5r-filtered-v2-env-transition-grpo-4gpu-978x200-softanswer-epoch2/hf_merged_global_step_200
files: model-00001-of-00002.safetensors, model-00002-of-00002.safetensors, config/tokenizer files
```

## Outputs

```text
log: /data/wzl/LightningSearch-RL/logs/phase5r-grpo-global200-heldout-eval.log
results: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval
sft rollouts: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval/sft_baseline/env_rollouts.jsonl
sft summary: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval/sft_baseline/summary.json
sft diagnostics: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval/sft_baseline/answer_diagnostics.json
grpo rollouts: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval/grpo_global_step_200/env_rollouts.jsonl
grpo summary: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval/grpo_global_step_200/summary.json
grpo diagnostics: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval/grpo_global_step_200/answer_diagnostics.json
comparison: /data/wzl/LightningSearch-RL/results/phase5r-grpo-global200-heldout-eval/comparison_summary.json
```

## Metrics

SFT baseline:

```text
example_count: 100
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.96
answer_containment_match_rate: 0.98
answer_token_f1: 0.975238
gold_evidence_recall: 1.0
all_gold_evidence_retrieved_rate: 1.0
assistant_observation_rate: 0.0
avg_observation_doc_count: 7.36
suspicious_count: 2
suspicious_adjusted_exact_match_rate: 0.979592
```

GRPO `global_step_200`:

```text
example_count: 100
valid_search_action_rate: 1.0
valid_answer_action_rate: 1.0
answer_exact_match_rate: 0.97
answer_containment_match_rate: 0.98
answer_token_f1: 0.978571
gold_evidence_recall: 1.0
all_gold_evidence_retrieved_rate: 1.0
assistant_observation_rate: 0.0
avg_observation_doc_count: 7.36
suspicious_count: 2
suspicious_adjusted_exact_match_rate: 0.989796
```

Delta, GRPO minus SFT:

```text
answer_exact_match_rate: +0.01
answer_token_f1: +0.003333
valid_search_action_rate: +0.0
valid_answer_action_rate: +0.0
answer_containment_match_rate: +0.0
gold_evidence_recall: +0.0
all_gold_evidence_retrieved_rate: +0.0
assistant_observation_rate: +0.0
avg_observation_doc_count: +0.0
```

## Per-example Difference

Only one example changed between the SFT and GRPO outputs:

```text
id: syn-010322
question/search query: Which city is home to the institute founded by Dr. Evelyn Reed?
gold: Cambridge
sft answer: Cambridge, Massachusetts
grpo answer: Cambridge
sft exact: false
grpo exact: true
sft containment: true
grpo containment: true
sft token_f1: 0.666667
grpo token_f1: 1.0
```

There were no exact-match regressions.

The two suspicious rows are unchanged in both models and look like synthetic QA
gold-label issues rather than tool-use failures:

```text
syn-010326: prediction Greenwood Historical Society, gold Greenfield University
syn-010401: prediction Vance Archive, gold Ashford University
```

Both predictions match retrieved observation titles.

## Warnings

Non-fatal warnings:

```text
tokenizer regex warning for the local merged Qwen checkpoint
torch_dtype deprecation warning
generation flags temperature/top_p/top_k ignored under deterministic decoding
```

No CUDA OOM or traceback occurred in the successful run. GPU 7 was released
after completion.

## Analysis

This evaluation is positive but modest. GRPO did not change search behavior on
the 100-example held-out tail slice: both SFT and GRPO had perfect valid search
and answer action rates, perfect gold evidence recall, and zero assistant-side
observation generation. The improvement is concentrated in answer formatting /
normalization: one containment-correct but exact-wrong answer was shortened from
`Cambridge, Massachusetts` to the exact gold answer `Cambridge`.

This supports the claim that the 200-step shaped GRPO checkpoint preserves the
tool-use contract while slightly improving answer exactness. It does not yet
show a large retrieval-policy improvement because the controlled gold+distractor
environment is already saturated for both models.

## Next Step

For stronger evidence, evaluate on a harder setting where search quality can
actually differ:

```text
1. same models, global candidate pool instead of gold-distractors;
2. or larger distractor_count, such as 20 or 50;
3. or a non-overlapping synthetic validation set generated after Phase 5R.
```

The immediate low-risk next experiment is a 100-example hard-distractor eval
with `candidate_pool=gold-distractors` and a larger `distractor_count`.
