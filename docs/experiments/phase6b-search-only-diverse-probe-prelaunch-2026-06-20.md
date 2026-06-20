# Phase 6B Search-Only Diverse Reward Probe Prelaunch

Date: 2026-06-20

## Goal

Generate direct query-quality alternatives for search-stage actions. Phase 6A-v2 showed that Phase 5Y had zero `search_vs_search` pairs because each search prompt produced at most one unique valid search query. Phase 6B changes the probe input distribution rather than training:

- keep only search transitions,
- add a search-probe instruction to the prompt,
- increase sampling count and temperature,
- score with rank-aware search reward,
- immediately build `search_vs_search` preference pairs.

## Inputs

- Transitions: `/data/wzl/LightningSearch-RL/results/phase5w-env-transitions-rankreward-from-5s500-hard50-filtered-v1/transitions.jsonl`
- Model: `/data/wzl/LightningSearch-RL/checkpoints/phase5d-sft-turns-docidfix-4gpu/hf_merged_global_step_40`
- Repo: `/data/wzl/LightningSearch-RL/repo`
- Env: `/data/wzl/LightningSearch-RL/.conda-envs/lightningsearch-rl`

## Probe Config

- stage filter: `search`
- limit after filtering: `493`
- search diversity prompt: enabled
- samples per prompt: `12`
- max new tokens: `64`
- search reward top-k: `8`
- backend: `vllm`
- batch size: `48`
- temperature: `1.6`
- top-p: `0.98`
- top-k: `80`
- seed: `20260620`
- tensor parallel: `1`
- GPU memory utilization: `0.45`
- max model length: `768`

## Pair Build Config

- stage: `search`
- pair category: `search_vs_search`
- min score gap: `0.05`
- min samples: `2`
- max pairs per group: `4`
- val fraction: `0.1`
- seed: `20260620`

## Outputs

- Probe result dir: `/data/wzl/LightningSearch-RL/results/phase6b-search-only-diverse-probe-rankreward-493x12`
- Pair result dir: `/data/wzl/LightningSearch-RL/results/phase6b-search-vs-search-pairs-rankreward-493x12`
- Log: `/data/wzl/LightningSearch-RL/logs/phase6b-search-only-diverse-probe-rankreward-493x12.log`

Expected artifacts:

- `$OUT/probe_requests.jsonl`
- `$OUT/generations.jsonl`
- `$OUT/reward_dump.jsonl`
- `$OUT/reward_dump_summary.json`
- `$OUT/summary.json`
- `$PAIRS_OUT/pairs.jsonl`
- `$PAIRS_OUT/train.jsonl`
- `$PAIRS_OUT/val.jsonl`
- `$PAIRS_OUT/summary.json`

## Launcher

```bash
ssh user@ssh-22.e6.luyouxia.net -p 29509 'tmux new-session -d -s lightningsearch-20260620-phase6b-search-diverse-probe -c /data/wzl/LightningSearch-RL/repo "CUDA_VISIBLE_DEVICES=7 bash scripts/remote/phase6b_search_only_diverse_reward_probe_from_phase5w.sh"'
```

## Success Criteria

- Probe completes without vLLM/runtime errors.
- `generated_sample_count = selected_transition_count * 12`.
- `pair_category_counts.search_vs_search > 0`.
- Search generation diagnostic shows at least some prompt groups with multiple unique valid search queries.

## Decision Rule

If Phase 6B yields enough `search_vs_search` pairs, use them for a small preference warmup. If it still yields zero or very few direct query pairs, switch to synthetic bad-query negatives through entity dropout, relation dropout, and generic-query corruption instead of relying on model sampling.
