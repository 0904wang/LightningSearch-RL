# Phase 6A-v2 Search-vs-Search Preference Pairs From Phase 5Y

Date: 2026-06-20

## Goal

Filter the Phase 5Y reward-probe alternatives to direct search-query preference pairs only. Unlike Phase 6A, this run keeps only `search_vs_search` pairs so that the resulting data can measure actual query-quality preference rather than tool-format correction.

## Code

- Commit: `60f91388d98afbc9b115fd2a83c6bec2afae1f28`
- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- Branch: `main`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`

## Launch

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260620-phase6a-v2-searchpairs -c /data/wzl/LightningSearch-RL/repo "bash scripts/remote/phase6a_v2_build_search_vs_search_pairs_from_phase5y.sh 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y.log"'
```

The job completed quickly; the tmux session exited after writing artifacts.

## Inputs

- Probe requests: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/probe_requests.jsonl`
- Generations: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/generations.jsonl`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/reward_dump.jsonl`

Input counts:

- requests: `978`
- generations: `5868`
- reward rows: `5868`

## Selection Config

- stage: `search`
- pair category: `search_vs_search`
- `min_score_gap=0.25`
- `min_samples=2`
- `max_pairs_per_group=4`
- `val_fraction=0.1`
- `seed=20260620`

## Outputs

- Result dir: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y`
- Pairs: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y/pairs.jsonl`
- Train: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y/train.jsonl`
- Val: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y/val.jsonl`
- Summary: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y/summary.json`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y.log`

The log file was zero bytes; the authoritative artifact is `summary.json`.

## Raw Summary

```json
{
  "candidate_pair_count": 0,
  "generation_count": 5868,
  "pair_categories": [
    "search_vs_search"
  ],
  "pair_category_counts": {},
  "pair_count": 0,
  "request_count": 978,
  "reward_dump_count": 5868,
  "skipped_group_count": 489,
  "stage_candidate_pair_counts": {},
  "stage_pair_counts": {},
  "stages": [
    "search"
  ],
  "train_count": 0,
  "unfiltered_candidate_pair_count": 7,
  "unfiltered_pair_category_counts": {
    "search_vs_answer": 7
  },
  "val_count": 0
}
```

Line counts:

```text
0 pairs.jsonl
0 train.jsonl
0 val.jsonl
```

## Additional Diagnostic

Search-stage generation diagnostic:

```json
{
  "action_type_counts": {
    "answer": 13,
    "search": 2945
  },
  "groups_by_gap_threshold": {
    "0.01": 0,
    "0.05": 0,
    "0.1": 0,
    "0.25": 0
  },
  "groups_with_at_least_2_unique_valid_search": 0,
  "groups_with_positive_search_vs_search_gap": 0,
  "search_generation_count": 2958,
  "search_request_count": 493,
  "top_search_vs_search_gaps": []
}
```

## Analysis

Phase 6A-v2 is a successful negative diagnostic. It proves that Phase 5Y does not contain direct query-quality preference pairs.

The key finding is not merely that the `0.25` reward gap is too strict. The deeper issue is that, after normalization, every search-stage prompt had at most one unique valid search query across its sampled outputs. There are no valid-search-vs-valid-search alternatives even at tiny gap thresholds such as `0.01`.

This explains why Phase 5Y GRPO barely moved search behavior:

- Search-stage rollout samples were nearly deterministic at the action-text level.
- Reward variance in Phase 5Y came mostly from answer behavior or wrong-stage action format mistakes.
- The available search preference signal is format-level (`search_vs_answer`), not query-quality-level.

Therefore a DPO/SimPO warmup on Phase 6A pairs would mostly teach "emit `<search>` instead of `<answer>` during search turns" and "emit correct answer instead of wrong answer during answer turns." It would not meaningfully improve query construction.

## Conclusion

Do not train a search-query preference warmup on Phase 6A-v2 because it contains zero usable `search_vs_search` pairs.

The next step should create fresh search-stage diversity rather than re-filtering Phase 5Y.

## Recommended Next Step

Run a search-only reward probe designed specifically to produce query alternatives:

- Input: Phase 5Y or Phase 5W search transitions only.
- Prompt: force exactly one `<search>...</search>` action and discourage copying the same query.
- Sampling: increase `samples_per_prompt` to `8` or `12`.
- Temperature: try `1.4` to `1.6`.
- top-p/top-k: keep nucleus sampling open enough to vary query wording.
- Reward: rank-aware evidence retrieval reward.
- Output: build preference pairs with `pair_category=search_vs_search`.

If that still produces zero or very few distinct query alternatives, change the prompt/schema to require query rewrites from a sampled strategy label, or synthesize explicit bad-query negatives from entity dropout/generic-query corruption rather than relying only on model sampling.
