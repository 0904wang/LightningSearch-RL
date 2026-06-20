# Phase 6A Preference Pairs From Phase 5Y Reward Probe

Date: 2026-06-20

## Goal

Convert Phase 5Y reward-probe alternatives into chosen/rejected preference pairs before attempting another GRPO run. This is intended to create clearer local supervision than sparse GRPO advantage, especially for search-stage actions.

## Code

- Local/GitHub commit with builder: `ecdc31a07aea2bc9e8282e9dd96066d749d91e2f`
- Launcher fix commit: `3a9b5f5b5e8752c27f7205525949f4b1e66228bb`
- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- Branch: `main`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`

## Launch

Initial tmux launch failed because the Phase 6A launcher used `set -u` before `conda activate`. The remote conda activation hook referenced an unset `SYS_SYSROOT` variable:

```text
/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl/etc/conda/activate.d/activate-gcc_linux-64.sh: line 107: SYS_SYSROOT: unbound variable
```

Fix:

- Use `set -eo pipefail` before conda activation.
- Enable `set -u` only after `conda activate`.
- Added regression test: `tests/test_remote_launchers.py`.

Final launch:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260620-phase6a-preference-pairs -c /data/wzl/LightningSearch-RL/repo "bash scripts/remote/phase6a_build_preference_pairs_from_phase5y.sh 2>&1 | tee /data/wzl/LightningSearch-RL/logs/phase6a-preference-pairs-rankreward-from-phase5y.log"'
```

## Inputs

- Probe requests: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/probe_requests.jsonl`
- Generations: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/generations.jsonl`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase5y-reward-probe-rankreward-978x6/reward_dump.jsonl`

Input counts:

- requests: `978`
- generations: `5868`
- reward rows: `5868`

## Selection Config

- stages: `search`, `answer`
- `min_score_gap=0.25`
- `min_samples=2`
- `max_pairs_per_group=2`
- `max_answer_pairs=300`
- `max_search_pairs=None`
- `val_fraction=0.1`
- `seed=20260620`

## Outputs

- Result dir: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y`
- Pairs: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y/pairs.jsonl`
- Train: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y/train.jsonl`
- Val: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y/val.jsonl`
- Summary: `/data/wzl/LightningSearch-RL/results/phase6a-preference-pairs-rankreward-from-phase5y/summary.json`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6a-preference-pairs-rankreward-from-phase5y.log`

The log file was created but remained zero bytes; the authoritative result is `summary.json`.

## Raw Summary

```json
{
  "candidate_pair_count": 44,
  "generation_count": 5868,
  "pair_count": 44,
  "request_count": 978,
  "reward_dump_count": 5868,
  "skipped_group_count": 897,
  "stage_candidate_pair_counts": {
    "answer": 38,
    "search": 6
  },
  "stage_pair_counts": {
    "answer": 38,
    "search": 6
  },
  "train_count": 40,
  "val_count": 4
}
```

Line counts:

```text
44 pairs.jsonl
40 train.jsonl
4 val.jsonl
```

Score-gap spot check:

```text
search count 6, gap_min 0.5, gap_mean 0.566667, gap_max 0.6
answer count 38, gap_min 0.333333, gap_mean 0.842231, gap_max 1.1
```

Example search pair:

```json
{
  "pair_id": "syn-009009:1:search:search:9:1>0",
  "chosen_score": 0.47,
  "rejected_score": -0.03,
  "chosen": "<search>In which city was the editor of the Journal of Quantum Computing born?</search>",
  "rejected": "<answer>New York</answer>"
}
```

Example answer pair:

```json
{
  "pair_id": "syn-009022:1:answer:answer:13:0>2",
  "chosen_score": 1.1,
  "rejected_score": 0.1,
  "chosen": "<answer>Oakridge</answer>",
  "rejected": "<answer>Boston</answer>"
}
```

## Analysis

Phase 6A succeeded as a data-construction smoke: the pipeline can now build preference pairs from reward-probe alternatives and preserve prompt, chosen/rejected actions, scores, reward components, source ids, transition ids, and stage.

However, the output is much smaller than desired:

- Only `44 / 978` prompt groups yielded a qualifying pair.
- Only `6` pairs are search-stage pairs.
- `897` groups were skipped, mostly because sampled actions were not both unique and reward-separated enough.

The search pairs are useful but not yet the ideal signal. The sampled search example shows a valid `<search>` chosen over an invalid stage action such as `<answer>...`, so part of the search preference signal is format/stage correctness rather than nuanced query quality. This can help stabilize tool-use format, but it is unlikely to strongly improve query ranking behavior by itself.

The answer pairs are higher-quality and larger in count, but using them directly for preference warmup risks repeating the Phase 5Y issue: the model may move on answer tokens while search policy remains mostly unchanged.

## Conclusion

Phase 6A is a successful infrastructure step, not yet a sufficient training dataset. The next training run should not use this dataset naively as a balanced preference warmup, because it is still answer-heavy and the search pairs are too few.

## Recommended Next Step

Build Phase 6A-v2 with stricter stage-compatible pair construction:

- For search-stage pairs, require both chosen and rejected parsed actions to be valid `search`.
- Add a separate `search-format` preference set only if we intentionally want format repair.
- Lower or stage-specialize the search gap threshold if needed, but report query-rank metrics separately.
- Increase search alternatives by running a search-only reward probe with higher sampling temperature and possibly `samples_per_prompt=8`.
- Add diagnostics for pair categories:
  - valid search vs valid search,
  - valid search vs invalid action,
  - answer vs answer,
  - answer vs invalid action.

Only after enough valid-search-vs-valid-search pairs exist should we run DPO/SimPO warmup aimed at search query policy.
