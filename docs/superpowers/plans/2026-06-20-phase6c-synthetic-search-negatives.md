# Phase 6C Synthetic Search Negatives Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic Phase 6C data-construction path that turns search transitions into search-vs-search preference pairs by corrupting good search queries into controlled bad-query negatives.

**Architecture:** Add a focused synthetic preference builder that reads exported environment transitions, scores the original search action and synthetic bad search actions with the existing rank-aware `compute_score`, then writes `pairs.jsonl`, `train.jsonl`, `val.jsonl`, `candidates.jsonl`, `reward_dump.jsonl`, and `summary.json`. Expose it through the CLI and a remote launcher, keeping the Phase 6A/6B reward-probe pipeline unchanged.

**Tech Stack:** Python 3.10+, pytest, existing `lightningsearch_rl.verl_reward.compute_score`, existing transition JSONL schema, bash/tmux remote launcher.

---

## Chunk 1: Local Builder And CLI

### Task 1: RED tests for synthetic search preference pairs

**Files:**
- Create: `tests/test_synthetic_search_preferences.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_remote_launchers.py`

- [ ] Write a failing unit test that builds two search transitions with gold/distractor passages and expects direct `search_vs_search` pairs from synthetic bad queries.
- [ ] Write a failing CLI test for `build-synthetic-search-preferences`.
- [ ] Write a failing remote launcher test that verifies conda activation happens before `set -u`.
- [ ] Run the targeted tests and confirm they fail because the new module/command/script is missing.

### Task 2: Implement builder

**Files:**
- Create: `src/lightningsearch_rl/synthetic_search_preferences.py`

- [ ] Load transition JSONL and filter `action_type == search`.
- [ ] Extract the chosen query from `row["query"]` or parsed `row["action"]`.
- [ ] Generate deterministic negative queries: generic query, distractor-title query, gold-title-only partial query, answer-only query, entity-dropout query, and relation-dropout query.
- [ ] Score chosen and rejected actions through `compute_score` with `reward_stage=search`, `candidate_passages`, `gold_doc_ids`, and `search_reward_top_k`.
- [ ] Keep only pairs where `chosen_score >= min_chosen_score` and `chosen_score - rejected_score >= min_score_gap`.
- [ ] Write pair artifacts and a summary with counts by corruption type and skip reason.

### Task 3: Wire CLI and launcher

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Create: `scripts/remote/phase6c_synthetic_search_negatives_from_phase5w.sh`

- [ ] Add CLI arguments for transitions path, output dir, offset, limit, top-k, min chosen score, min score gap, max negatives per transition, val fraction, and seed.
- [ ] Add a remote launcher that runs inside the approved conda env and writes under `/data/wzl/LightningSearch-RL/results/phase6c-synthetic-search-negatives-rankreward-from-phase5w`.
- [ ] Keep `set -u` after conda activation.

### Task 4: Verification and integration

**Files:**
- Tests and docs only unless failures require focused fixes.

- [ ] Run targeted tests for synthetic builder, CLI, remote launchers, reward probe, and preference pairs.
- [ ] Run the full pytest suite.
- [ ] Commit and push the branch.
- [ ] Sync remote safely, run remote smoke tests for the new CLI, then prepare the Phase 6C real-run approval payload.
