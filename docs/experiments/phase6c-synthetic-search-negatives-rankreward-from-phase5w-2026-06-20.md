# Phase 6C Synthetic Search Negatives Rank-Reward From Phase 5W

## Goal

Build direct `search_vs_search` preference data without relying on stochastic model sampling diversity. Phase 6B produced only 27 direct search-query pairs from 493 search prompts. Phase 6C instead takes each search transition, keeps the original search query as the chosen action when it scores above a threshold, and creates deterministic bad-query negatives with the same rank-aware search reward.

## Launch

- Status: completed
- Started: `2026-06-20T09:55:46+00:00`
- Finished: `2026-06-20T09:55:50+00:00`
- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- Branch: `main`
- Commit: `7591c016292f4d58590c2723655cc6ad2675176c`
- Environment: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`
- GPU selection: no GPU intended; launched with `env CUDA_VISIBLE_DEVICES=`
- tmux session: `lightningsearch-20260620-phase6c-synthetic-negatives`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6c-synthetic-search-negatives-rankreward-from-phase5w.log`
- Results: `/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w`

Exact launch command:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 "tmux new-session -d -s lightningsearch-20260620-phase6c-synthetic-negatives -c /data/wzl/LightningSearch-RL/repo env CUDA_VISIBLE_DEVICES= bash scripts/remote/phase6c_synthetic_search_negatives_from_phase5w.sh"
```

## Inputs And Settings

- Transitions: `/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl`
- Input transition count: `978`
- Search transition count: `493`
- Selected search transitions: `493`
- Offset / limit: `0 / 493`
- Search reward top-k: `8`
- Min chosen score: `0.5`
- Min score gap: `0.05`
- Max negatives per transition: `6`
- Val fraction: `0.1`
- Seed: `20260620`

## Artifacts

- Candidates: `/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w/candidates.jsonl`
- Reward dump: `/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w/reward_dump.jsonl`
- Pairs: `/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w/pairs.jsonl`
- Train: `/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w/train.jsonl`
- Val: `/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w/val.jsonl`
- Summary: `/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w/summary.json`

Line counts:

```text
4055 candidates.jsonl
4055 reward_dump.jsonl
2258 pairs.jsonl
2032 train.jsonl
 226 val.jsonl
```

## Raw Summary

```json
{
  "candidate_corruption_type_counts": {
    "answer_only": 446,
    "chosen": 493,
    "distractor_title": 446,
    "entity_dropout": 446,
    "generic": 446,
    "gold_title_only": 886,
    "question_keywords": 446,
    "relation_dropout": 446
  },
  "candidate_count": 4055,
  "input_transition_count": 978,
  "limit": 493,
  "max_negatives_per_transition": 6,
  "min_chosen_score": 0.5,
  "min_score_gap": 0.05,
  "offset": 0,
  "pair_category_counts": {
    "search_vs_search": 2258
  },
  "pair_corruption_type_counts": {
    "answer_only": 356,
    "distractor_title": 439,
    "entity_dropout": 229,
    "generic": 446,
    "gold_title_only": 308,
    "question_keywords": 320,
    "relation_dropout": 160
  },
  "pair_count": 2258,
  "search_reward_top_k": 8,
  "search_transition_count": 493,
  "seed": 20260620,
  "selected_transition_count": 493,
  "skip_reason_counts": {
    "chosen_score_below_min": 47
  },
  "stage_pair_counts": {
    "search": 2258
  },
  "train_count": 2032,
  "val_count": 226,
  "val_fraction": 0.1
}
```

## Sample Pair

Chosen:

```text
<search>Which award did the organization founded by Dr. Elena Voss receive in 2021?</search>
```

Rejected:

```text
<search>Institute for Advanced Studies</search>
```

Reward details:

```text
chosen_score=0.57
rejected_score=0.07
score_gap=0.50
rejected_corruption_type=distractor_title
chosen_search_reward=0.5
rejected_search_reward=0.0
```

## Verification

Before launch:

- Local full test suite on main: `170 passed, 1 skipped`
- Remote targeted smoke: `4 passed`
- Remote full test suite: `174 passed`
- Remote preflight: tmux available, conda hook works, `/data` has about `4.3T` free, no active tmux session before launch

After launch:

- `tmux list-sessions` returned no active session, consistent with completed short CPU job.
- Log contains `finished_at=2026-06-20T09:55:50+00:00`.
- Result directory size: `53M`.
- GPU memory unchanged, confirming this was not a GPU job.

## Analysis

Phase 6C succeeded as a data-construction step. It expanded direct query-quality supervision from Phase 6B's 27 stochastic `search_vs_search` pairs to 2258 deterministic synthetic `search_vs_search` pairs. The train/val split is large enough for a small preference-warmup smoke, and the pair categories are clean: every pair is `search_vs_search`, so this dataset targets query quality rather than search-vs-answer format repair.

The main caveat is that the rejected queries are synthetic corruptions, not model-native failure modes. This makes the signal dense and controllable, but it may over-teach obvious distinctions such as precise query versus generic query or distractor title. The corruption mix is reasonably varied: generic, distractor title, answer-only, gold-title-only, question keywords, entity dropout, and relation dropout all appear in the selected pairs.

The chosen threshold filtered out 47 search transitions whose original query scored below `0.5`. This is useful because it prevents weak rollout queries from becoming chosen examples, but it also means the preference data is biased toward already-reasonable search actions.

## Conclusion

Use Phase 6C for the next preference-warmup smoke before another GRPO attempt. It is materially stronger than Phase 6A/6B for search policy movement because it provides thousands of same-state direct query comparisons.

Recommended next step:

1. Export Phase 6C train/val to the preference trainer format.
2. Run a tiny DPO/SimPO-style preference warmup smoke on Qwen3-4B or LoRA if full preference training is too heavy.
3. Evaluate policy movement specifically on search-stage prompts before returning to GRPO.
4. If preference warmup moves search logprobs in the desired direction, run a small GRPO job initialized from that warmed model.
