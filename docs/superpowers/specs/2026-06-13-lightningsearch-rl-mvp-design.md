# LightningSearch-RL MVP Design

## Context

`D:\resume\Agent RL` is a new local project for building a resume-ready retrieval tool-use Agent RL framework. The current repository contains remote experiment rules in `AGENTS.md`, but no implementation code yet.

The first phase should produce a local, testable MVP rather than launching remote training immediately. This keeps the trace schema, retrieval environment, reward logic, and evaluation interface stable before using remote GPUs for SFT or GRPO.

## Goal

Build a local Python MVP for `LightningSearch-RL` that can run a tiny offline multi-hop QA search-agent loop end to end:

1. Load a small JSONL dataset.
2. Build or load an offline BM25-style retrieval index.
3. Parse and validate `think/search/observe/answer` actions.
4. Execute a deterministic search-agent baseline.
5. Collect traces using a stable schema.
6. Convert traces into trainable transition records.
7. Compute answer, evidence, format, tool-validity, and search-cost rewards.
8. Export evaluation metrics and artifacts for later remote SFT / GRPO work.

This phase does not include real model training, remote launch, Qwen inference, vLLM, or verl integration. Those are second-phase tasks after the local interfaces are tested.

## Recommended Approach

Use a small package named `lightningsearch_rl` with focused modules:

- `data`: JSONL examples and tiny sample dataset loading.
- `retrieval`: deterministic lexical BM25-like retriever with no heavyweight dependency in MVP.
- `actions`: tag parsing and action validation.
- `runtime`: agent loop, tool execution, and trace collection.
- `transitions`: trace-to-transition adapter.
- `rewards`: shaped reward components and total reward.
- `eval`: metrics aggregation.
- `cli`: smoke command for local and remote preflight use.

The MVP should prefer deterministic components. A rule-based baseline agent is acceptable because the goal is to validate the framework contract, not the final RL policy.

## Architecture

```text
JSONL Dataset
      |
      v
Dataset Loader
      |
      v
Lexical Retriever  <--- Corpus passages
      |
      v
Agent Runtime
rule policy -> think/search/observe/answer loop
      |
      v
Trace Collector
question, steps, observations, metadata
      |
      v
Transition Builder
trace -> state/action/reward transitions
      |
      v
Reward + Evaluation
answer EM/F1, evidence recall, tool validity, search cost
      |
      v
Artifacts
traces.jsonl, transitions.jsonl, metrics.json
```

## Data Contract

MVP input should be JSONL. Each line is one example:

```json
{
  "id": "ex1",
  "question": "Which city is the birthplace of the author of Example Book?",
  "answers": ["Example City"],
  "gold_doc_ids": ["doc_author", "doc_city"],
  "corpus": [
    {"doc_id": "doc_book", "title": "Example Book", "text": "Example Book was written by Alice Smith."},
    {"doc_id": "doc_author", "title": "Alice Smith", "text": "Alice Smith was born in Example City."},
    {"doc_id": "doc_city", "title": "Example City", "text": "Example City is a coastal city."}
  ]
}
```

This intentionally keeps corpus-per-example support for tiny smoke tests. Later phases can add shared corpus indexes and HotpotQA / 2Wiki adapters.

## Trace Contract

The trace schema should be serializable to JSON:

```json
{
  "question_id": "ex1",
  "question": "...",
  "steps": [
    {
      "state": "question + history",
      "action": "<search>Example Book author birthplace</search>",
      "action_type": "search",
      "query": "Example Book author birthplace",
      "observation": [{"doc_id": "doc_book", "text": "..."}],
      "valid_tool_call": true,
      "terminal": false
    },
    {
      "state": "question + history + observation",
      "action": "<answer>Example City</answer>",
      "action_type": "answer",
      "terminal": true
    }
  ],
  "final_answer": "Example City",
  "reward": 1.17,
  "metadata": {"search_count": 1}
}
```

## Reward Design

The MVP should implement deterministic, testable components:

- `answer_reward`: exact match over normalized answers, `1.0` or `0.0`.
- `format_reward`: valid action tags and valid terminal answer, `1.0` or `0.0`.
- `tool_validity_reward`: non-empty search query and no duplicate searches, `1.0` or partial.
- `evidence_reward`: fraction of gold document ids retrieved in observations.
- `cost_penalty`: `0.03 * search_count`.

Default total:

```text
R = answer_reward
  + 0.2 * evidence_reward
  + 0.1 * format_reward
  + 0.1 * tool_validity_reward
  - 0.03 * search_count
```

## CLI Contract

The first CLI command should be:

```bash
python -m lightningsearch_rl.cli smoke --data tests/fixtures/tiny_multihop.jsonl --out-dir results/smoke
```

Expected outputs:

```text
results/smoke/traces.jsonl
results/smoke/transitions.jsonl
results/smoke/metrics.json
```

The remote `AGENTS.md` dry-run command can later call the same `smoke` command with a remote config or dataset path.

## Testing Strategy

Use TDD for implementation:

- Parser tests for tag extraction and invalid actions.
- Retriever tests for deterministic ranking and top-k behavior.
- Runtime tests for trace shape and terminal answer handling.
- Reward tests for answer, evidence, tool validity, and cost components.
- Transition tests for preserving state/action/observation boundaries.
- CLI smoke test for writing all expected artifacts.

The MVP is complete when `python -m pytest` passes locally and the smoke command creates valid artifacts from the tiny fixture.

## Out Of Scope For Phase 1

- Real HotpotQA / 2Wiki full preprocessing.
- FAISS dense retrieval.
- Qwen inference.
- vLLM serving.
- SFT training.
- GRPO training.
- Remote launch.
- TensorBoard dashboards.
- Resume PDF editing.

## Phase 2 Preview

After the local MVP:

1. Add HotpotQA / 2Wiki adapters and shared corpus indexing.
2. Add Qwen policy rollout through vLLM or transformers.
3. Add SFT trajectory export.
4. Add verl-compatible GRPO data / reward hooks.
5. Run remote smoke tests under `AGENTS.md`.
6. Launch approved 1-GPU then 2-4-GPU experiments.

