# Phase 6B Search-Only Diverse Reward Probe

Date: 2026-06-20

## Goal

Create direct search-query preference data after Phase 6A-v2 found zero `search_vs_search` pairs in the Phase 5Y reward probe. This experiment changes the probe input distribution instead of training:

- filter to search-stage transitions only,
- add a search-probe diversity instruction,
- increase sampling to 12 outputs per prompt,
- use higher-temperature vLLM sampling,
- score with rank-aware retrieval reward,
- build `search_vs_search` chosen/rejected pairs.

## Code

- Commit: `5e754edf0f52a25c7031ede5d4c7dc7dd0c8ae94`
- Remote repo: `/data/wzl/LightningSearch-RL/repo`
- Branch: `main`
- Conda env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`

Remote GitHub fetch failed twice before launch due TLS/HTTP2 connection termination, so the remote repo was updated via a local git bundle under `/data/wzl/LightningSearch-RL/runs/sync/phase6b-main.bundle` using `git pull --ff-only`.

## Launch

The original nested-shell tmux launch returned success but exited immediately without logs. A direct tmux argv launch succeeded:

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260620-phase6b-search-diverse-probe -c /data/wzl/LightningSearch-RL/repo env CUDA_VISIBLE_DEVICES=7 bash scripts/remote/phase6b_search_only_diverse_reward_probe_from_phase5w.sh'
```

GPU:

- `CUDA_VISIBLE_DEVICES=7`
- Peak observed memory: about `15531 MiB / 32607 MiB`

## Inputs

- Transitions: `/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl`
- Model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`

## Probe Config

- stage filter: `search`
- input transitions: `978`
- filtered search transitions: `493`
- selected transitions: `493`
- source count: `489`
- samples per prompt: `12`
- expected reward rows: `5916`
- max new tokens: `64`
- search reward top-k: `8`
- search diversity prompt: enabled
- backend: `vllm`
- batch size: `48`
- temperature: `1.6`
- top-p: `0.98`
- top-k: `80`
- seed: `20260620`
- tensor parallel size: `1`
- gpu memory utilization: `0.45`
- max model length: `768`

## Outputs

- Probe result dir: `/data/wzl/LightningSearch-RL/results/phase6b-search-only-diverse-probe-rankreward-493x12`
- Pair result dir: `/data/wzl/LightningSearch-RL/results/phase6b-search-vs-search-pairs-rankreward-493x12`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6b-search-only-diverse-probe-rankreward-493x12.log`

Line counts:

```text
493 probe_requests.jsonl
5916 generations.jsonl
5916 reward_dump.jsonl
27 pairs.jsonl
24 train.jsonl
3 val.jsonl
```

## Raw Probe Summary

```json
{
  "dry_run": false,
  "input_transition_count": 978,
  "filtered_transition_count": 493,
  "selected_transition_count": 493,
  "source_count": 489,
  "stage_counts": {
    "search": 493
  },
  "samples_per_prompt": 12,
  "expected_reward_rows": 5916,
  "generated_sample_count": 5916,
  "reward_dump_count": 5916,
  "search_diversity_prompt": true,
  "temperature": 1.6,
  "top_p": 0.98,
  "top_k": 80
}
```

Reward dump summary highlights:

```json
{
  "row_count": 5916,
  "stage_counts": {
    "search": 5916
  },
  "score": {
    "min": -0.03,
    "mean": 0.754424,
    "max": 0.97
  },
  "search_reward": {
    "min": 0.0,
    "mean": 0.68461,
    "max": 0.9
  },
  "invalid_action_count": 0,
  "low_score_count": 562,
  "source_id_variable_groups": {
    "variable_group_count": 26,
    "variable_group_rate": 0.05317
  }
}
```

## Pair Summary

```json
{
  "request_count": 493,
  "generation_count": 5916,
  "reward_dump_count": 5916,
  "unfiltered_candidate_pair_count": 31,
  "unfiltered_pair_category_counts": {
    "search_vs_answer": 4,
    "search_vs_search": 27
  },
  "candidate_pair_count": 27,
  "pair_count": 27,
  "pair_category_counts": {
    "search_vs_search": 27
  },
  "stage_pair_counts": {
    "search": 27
  },
  "train_count": 24,
  "val_count": 3,
  "min_score_gap": 0.05
}
```

## Additional Diagnostic

```json
{
  "action_type_counts": {
    "answer": 11,
    "search": 5905
  },
  "groups_with_at_least_2_unique_valid_search": 92,
  "groups_with_positive_search_vs_search_gap": 23,
  "groups_by_gap_threshold": {
    "0.01": 23,
    "0.05": 23,
    "0.1": 21,
    "0.25": 12
  },
  "pair_count": 27,
  "pair_gap_min": 0.1,
  "pair_gap_mean": 0.225926,
  "pair_gap_max": 0.4
}
```

Example pair:

```json
{
  "pair_id": "syn-009034:0:search:search:11:11>0",
  "chosen_score": 0.72,
  "rejected_score": 0.47,
  "score_gap": 0.25,
  "chosen": "<search>Which university publishes the journal edited by Dr. Elena Voss in 2018?</search>",
  "rejected": "<search>Which university publishes the journal that Dr. Elena Voss edited in 2018?</search>"
}
```

## Analysis

Phase 6B is a successful probe. The diversity prompt plus higher-temperature sampling changed the search-stage behavior enough to create direct query-quality alternatives:

- Phase 6A-v2: `0` search-vs-search pairs.
- Phase 6B: `27` search-vs-search pairs.

The improvement is real, but the data is still small. Only `92 / 493` prompt groups produced at least two unique valid search queries, and only `23` groups had positive search-query reward gaps. This is enough for a tiny preference-warmup smoke test, but not enough for a meaningful full DPO/SimPO run.

The sampled pairs show many chosen/rejected queries are close paraphrases. That is expected because the SFT model strongly anchors on one canonical query form. The rank-aware reward can still distinguish some paraphrases because BM25 scoring is sensitive to relation words and entity placement, but the signal is narrow.

The probe also solved the previous format problem: `5905 / 5916` generations were valid search actions and only `11` were answer actions. The remaining search preference signal is therefore mostly query-level rather than format-level.

## Conclusion

Phase 6B establishes that search-query preference data can be produced by changing the probe prompt and sampling settings, but the current dataset is too small for a serious training run.

## Recommended Next Step

Run a Phase 6B-v2 scale-up before training:

- keep search-only and diversity prompt,
- increase sampled prompts beyond `493` if more search transitions are available,
- keep `samples_per_prompt=12` or try `16`,
- preserve `temperature=1.6`,
- add synthetic query corruptions to increase negative diversity:
  - entity dropout,
  - relation dropout,
  - generic query replacement,
  - bridge-entity-only query,
  - answer-entity-only query.

If synthetic corruption is implemented, pair each generated good query against deterministic bad queries and score both through the same rank-aware reward. That should yield many more `search_vs_search` pairs than stochastic sampling alone.
