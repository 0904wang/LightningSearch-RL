# Phase 3B GRPO Rollout / Reward Export Design

## Goal

Build a local GRPO data interface layer for LightningSearch-RL. The feature converts prepared examples and a persisted lexical index into rollout, transition, reward, and summary artifacts that a later verl/GRPO trainer can consume.

This phase strengthens the project's central claim: agent execution is decoupled from training. The agent runtime produces traces; the training interface consumes those traces and exports normalized records without knowing how the agent is implemented.

## Non-Goals

- Do not run Qwen inference.
- Do not start verl, vLLM, or any remote training process.
- Do not add GPU dependencies.
- Do not implement policy updates.
- Do not add new external Python dependencies.

## Inputs

The export should use artifacts already produced by previous phases:

- `examples.jsonl`: prepared QA examples from HotpotQA-like or 2Wiki-like adapters.
- `index.json`: persisted lexical index built from the shared corpus.
- `top_k`: retrieval budget for deterministic rollout generation.

## Outputs

`export_grpo(examples_path, index_path, out_dir, top_k)` writes four files:

- `rollouts.jsonl`
- `transitions.jsonl`
- `reward_records.jsonl`
- `summary.json`

All output files are written under `out_dir`, which is created if missing.

## Rollout Record Schema

Each row in `rollouts.jsonl` represents one episode-level training sample:

```json
{
  "id": "hp_mixed_1",
  "prompt": "Which city is the birthplace of the author of Example Book?",
  "response": "<think>...</think>\n<search>...</search>\n<observation>...</observation>\n<answer>Example City</answer>",
  "reward": 1.37,
  "metadata": {
    "answer": "Example City",
    "search_count": 1,
    "gold_doc_ids": ["hotpot::Alice Smith::0"],
    "retrieved_doc_ids": ["hotpot::Alice Smith::0", "hotpot::Example Book::0"]
  }
}
```

The `prompt` is the user question. The `response` uses the Phase 3A structured assistant trace format. The `reward` is the shaped reward total from the existing reward module.

## Reward Record Schema

Each row in `reward_records.jsonl` stores the reward breakdown for one episode:

```json
{
  "id": "hp_mixed_1",
  "total": 1.37,
  "answer_reward": 1.0,
  "evidence_reward": 1.0,
  "format_reward": 1.0,
  "tool_validity_reward": 1.0,
  "search_count": 1,
  "search_cost": 0.03
}
```

The field names should mirror the existing reward result fields where possible. The existing reward dataclass exposes `search_count`; `search_cost` is a derived artifact field computed as `0.03 * search_count` for easier reward-cost analysis.

## Transition Schema

`transitions.jsonl` reuses the existing transition builder:

- one row per state/action boundary;
- search steps keep observations;
- terminal answer steps carry the episode reward.

This phase does not implement per-step credit assignment. It only preserves the transition boundary and terminal reward. Evidence-aware credit assignment can be added in a later phase without changing the rollout schema.

## Module Design

Create `src/lightningsearch_rl/grpo.py` with:

- `export_grpo(examples_path: Path, index_path: Path, out_dir: Path, top_k: int = 5) -> dict`
- private helpers for rollout rows, reward rows, retrieved doc id extraction, and JSONL writing.

Implementation flow:

1. Load examples with `load_jsonl_examples`.
2. Load retriever with `load_lexical_index`.
3. For each example, run `run_retrieval_episode(example, retriever, top_k=top_k)`.
4. Compute reward with `compute_reward`.
5. Rebuild the trace with `reward` and reward metadata attached.
6. Format the assistant response with `format_assistant_trace`.
7. Build rollout row, transition rows, and reward row.
8. Write all artifacts and return summary.

The summary should include:

- `example_count`
- `rollout_count`
- `transition_count`
- `avg_reward`
- `avg_search_count`
- `top_k`
- artifact paths

## CLI Design

Extend `src/lightningsearch_rl/cli.py` with:

```bash
python -m lightningsearch_rl.cli export-grpo \
  --examples results/phase3a/examples.jsonl \
  --index results/phase3a/index.json \
  --out-dir results/phase3b/grpo \
  --top-k 2
```

The command returns exit code `0` after writing artifacts. It should follow the style of existing CLI commands and should not print large payloads.

## Testing

Add `tests/test_grpo.py`:

- prepare a tiny HotpotQA-like mixed fixture;
- build an index;
- call `export_grpo`;
- assert all four artifacts exist;
- assert the first rollout has `prompt`, `response`, `reward`, and `metadata`;
- assert the reward record contains total reward and reward breakdown fields;
- assert summary counts are correct.

Extend `tests/test_cli.py`:

- prepare data;
- build index;
- call `export-grpo`;
- assert `rollouts.jsonl`, `transitions.jsonl`, `reward_records.jsonl`, and `summary.json` exist.

Final verification:

```bash
python -m pytest
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_mixed_raw.jsonl --corpus results/phase3b/corpus.jsonl --examples results/phase3b/examples.jsonl --limit 1
python -m lightningsearch_rl.cli build-index --corpus results/phase3b/corpus.jsonl --index results/phase3b/index.json
python -m lightningsearch_rl.cli export-grpo --examples results/phase3b/examples.jsonl --index results/phase3b/index.json --out-dir results/phase3b/grpo --top-k 2
```

## Risks

- The current rollout is deterministic and rule-based, so it is a data-interface smoke target rather than a learned policy rollout.
- Reward field names must be checked against the existing reward dataclass before implementation.
- The transition export carries terminal reward only; shaped intermediate credit assignment remains future work.
