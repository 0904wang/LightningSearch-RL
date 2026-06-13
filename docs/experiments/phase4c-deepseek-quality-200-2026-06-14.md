# Phase 4C DeepSeek Quality-Controlled Pilot

Date: 2026-06-14

## Goal

Run the stricter Phase 4C synthetic validator with a 200 valid-row target and
compare data quality against Phase 4B.

## Code and Environment

- Commit: `fa5ff83`
- Remote repo path: `/data/wzl/LightningSearch-RL/repo`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- tmux session: `lightningsearch-20260614-phase4c-deepseek-quality-200`
- Log path: `/data/wzl/LightningSearch-RL/logs/phase4c-deepseek-quality-200.log`
- Results path: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200`
- GPU: not used
- Model: `deepseek-chat`
- Endpoint: `https://api.deepseek.com`
- Remote log time: UTC (`2026-06-13T17:41:10+00:00` to `2026-06-13T17:41:59+00:00`)

## Command

The API key was passed through the launched session environment only. It was not
written to source files, config files, logs, or result artifacts.

```bash
PYTHONNOUSERSITE=1 python -m lightningsearch_rl.cli synthesize-validated-data \
  --raw /data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/synthetic_raw.jsonl \
  --valid /data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/synthetic_valid.jsonl \
  --rejects /data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/synthetic_rejects.jsonl \
  --target-valid 200 \
  --topics awards,archives,research \
  --concurrency 50 \
  --batch-size 50 \
  --max-attempts 500 \
  --seed 3000 \
  --summary /data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/validated_summary.json \
  --model deepseek-chat \
  --base-url https://api.deepseek.com
```

The run then prepared corpus/examples, built a lexical index, and exported GRPO
artifacts from the valid rows.

## Raw Result Summary

Validated synthesis:

```json
{
  "api_failed": 0,
  "batch_size": 50,
  "concurrency": 50,
  "generated": 500,
  "max_attempts": 500,
  "reject_count": 314,
  "requested": 500,
  "stopped_reason": "max_attempts_reached",
  "target_valid": 200,
  "valid_count": 186
}
```

GRPO export:

```json
{
  "avg_reward": 0.34043,
  "avg_search_count": 1.0,
  "example_count": 186,
  "rollout_count": 186,
  "top_k": 2,
  "transition_count": 372
}
```

Line counts:

```text
500 synthetic_raw.jsonl
186 synthetic_valid.jsonl
314 synthetic_rejects.jsonl
186 grpo/rollouts.jsonl
372 grpo/transitions.jsonl
```

Secret scan:

```text
NO_SECRET_PATTERN
```

## Reject Reasons

```json
{
  "answer equals context title": 221,
  "non-ascii text detected": 39,
  "answer appears in multiple supporting sentences": 19,
  "answer not found in supporting evidence": 16,
  "supporting_facts must cover at least two titles": 13,
  "answer appears in question": 3,
  "supporting fact missing from context": 3
}
```

## Valid-Set Quality Checks

The stricter validator achieved the intended quality filters on accepted rows:

```json
{
  "valid_count": 186,
  "valid_answer_in_question": 0,
  "valid_answer_is_context_title": 0,
  "valid_answer_support_sentence_hits": {"1": 186},
  "valid_non_ascii": 0
}
```

## Artifacts

- Raw rows: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/synthetic_raw.jsonl`
- Valid rows: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/synthetic_valid.jsonl`
- Rejects: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/synthetic_rejects.jsonl`
- Corpus: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/corpus.jsonl`
- Examples: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/examples.jsonl`
- Index: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/index.json`
- GRPO rollouts: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/grpo/rollouts.jsonl`
- GRPO transitions: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/grpo/transitions.jsonl`
- GRPO reward records: `/data/wzl/LightningSearch-RL/results/phase4c-deepseek-quality-200/grpo/reward_records.jsonl`

## Analysis

Phase 4C improved accepted-row quality but reduced throughput. The run stopped
at `max_attempts_reached` with 186 valid rows out of 500 generated rows, so the
valid rate was 37.2%. This is much lower than Phase 4B's 78.7%, but the accepted
rows now satisfy the quality controls that were explicitly missing from Phase
4B: no answer in the question, no answer-as-title, exactly one supporting
sentence containing the answer, and no non-ASCII artifacts.

The dominant reject reason is `answer equals context title` (`221/314`). That
means the model still likes to make entity-name answers that are also page
titles, despite the prompt. The next prompt should push final answers toward
attributes, dates, institutions, awards, locations, or numeric facts while using
page titles only as intermediate entities.

Average GRPO reward dropped from Phase 4B (`0.4795`) to Phase 4C (`0.34043`).
This is expected because accepted rows are harder and less likely to be solved
by the current rule-based lexical rollout. For RL training, this may be better
data, but for bootstrapping SFT it may need a stronger trajectory generator or a
mixed curriculum with Phase 4B rows.

## Next Step

Do not simply raise `max_attempts` yet. First update the prompt with explicit
answer-type constraints:

- final answer must not be a page/entity title
- prefer answer types: year, award, city, institution department, method name,
  measured value, or short attribute
- page titles should be intermediate entities, not final answers

Then run a smaller 50-valid pilot to check whether valid rate improves before
running another 200-row job.
