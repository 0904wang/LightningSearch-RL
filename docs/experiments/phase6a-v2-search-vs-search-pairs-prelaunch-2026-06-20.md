# Phase 6A-v2 Search-vs-Search Preference Pair Build Prelaunch

Date: 2026-06-20

## Goal

Build a cleaner search-query preference subset from the Phase 5Y reward-probe alternatives.

Phase 6A produced `44` pairs, but only `6` were search-stage pairs, and spot checks showed the search pairs were mostly valid `<search>` actions beating wrong-stage `<answer>` actions. This v2 run filters to valid-search-vs-valid-search pairs only, so it measures whether Phase 5Y already contains enough direct query-quality preference signal.

## Inputs

- Probe requests: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/probe_requests.jsonl`
- Generations: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/generations.jsonl`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/reward_dump.jsonl`

## Output

- Result dir: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y`
- Pairs: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y/pairs.jsonl`
- Train: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y/train.jsonl`
- Val: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y/val.jsonl`
- Summary: `/data/wzl/LightningSearch-RL/results/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y/summary.json`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y.log`

## Selection Rules

- `stage=search`
- `pair_category=search_vs_search`
- `min_score_gap=0.25`
- `min_samples=2`
- `max_pairs_per_group=4`
- `val_fraction=0.1`
- `seed=20260620`

## Launcher

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260620-phase6a-v2-searchpairs -c /data/wzl/LightningSearch-RL/repo "bash scripts/remote/phase6a_v2_build_search_vs_search_pairs_from_phase5y.sh 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase6a-v2-search-vs-search-pairs-rankreward-from-phase5y.log"'
```

## Success Criteria

- `summary.json` exists.
- `pair_category_counts.search_vs_search > 0`.
- Pair rows contain only `chosen_action_type=search` and `rejected_action_type=search`.
- The summary reports both filtered and unfiltered pair category counts.

## Decision Rule

If this run yields too few search-vs-search pairs, do not run preference warmup yet. Instead, launch a search-only reward probe with more samples per prompt and possibly higher sampling temperature to create direct query-quality alternatives.
