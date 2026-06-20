# Phase 6A Preference Pair Build Prelaunch

Date: 2026-06-20

## Goal

Build a preference dataset from Phase 5Y reward-probe alternatives before attempting another GRPO run. The immediate goal is to convert same-prompt sampled actions into chosen/rejected pairs with clear reward gaps, especially for search-stage actions.

This addresses the Phase 5Y finding that GRPO produced only tiny policy movement and almost no search-action logprob movement.

## Inputs

- Probe requests: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/probe_requests.jsonl`
- Generations: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/generations.jsonl`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/reward_dump.jsonl`
- Repo: `/data/wzl/LightningSearch-RL/repo`
- Branch: `main`
- Local implementation branch: `codex/phase6a-preference-pairs`

## Output

- Result dir: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y`
- Main pairs: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y/pairs.jsonl`
- Train split: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y/train.jsonl`
- Val split: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y/val.jsonl`
- Summary: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y/summary.json`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6a-preference-pairs-rankreward-from-phase5y.log`

## Selection Rules

- Group by `request_index` from the reward probe.
- Keep `search` and `answer` stages.
- Deduplicate equivalent actions after parsing `<search>` / `<answer>` tags.
- Require `score_gap >= 0.25`.
- Require at least 2 unique sampled actions per group.
- Keep at most 2 pairs per prompt group.
- Cap answer-stage pairs at 300 to avoid another answer-heavy preference dataset.
- Do not cap search-stage pairs.
- Use `val_fraction=0.1`.
- Use `seed=20260620`.

## Launcher

Remote script:

```bash
/data/wzl/LightningSearch-RL/repo/scripts/remote/phase6a_build_preference_pairs_from_phase5y.sh
```

Expected tmux launch after sync and smoke:

```bash
tmux new-session -d -s lightningsearch-20260620-phase6a-preference-pairs \
  "bash -lc 'cd /data/wzl/LightningSearch-RL/repo && source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate /data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl && PYTHONNOUSERSITE=1 bash scripts/remote/phase6a_build_preference_pairs_from_phase5y.sh 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase6a-preference-pairs-rankreward-from-phase5y.log'"
```

## Success Criteria

- `summary.json` exists.
- `pair_count > 0`.
- `stage_pair_counts.search > 0`.
- `train.jsonl` and `val.jsonl` are non-empty unless the total pair count is too small for a validation split.
- Pair rows include `prompt`, `chosen`, `rejected`, scores, reward components, source ids, transition ids, and stage.

## Next Step

If Phase 6A yields meaningful search pairs, run a small preference warmup experiment before returning to GRPO. If it remains answer-heavy or search pairs are near-zero, improve query-level reward and/or generate a harder search-specific probe before training.
