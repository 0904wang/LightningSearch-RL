# Phase 3B GRPO Rollout / Reward Export Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export deterministic search-agent rollouts, transitions, reward records, and summary artifacts for later verl/GRPO training.

**Architecture:** Add a focused `grpo.py` data-interface module that reuses existing prepared examples, persisted lexical index loading, deterministic runtime traces, shaped rewards, trace formatting, and transition building. Extend the CLI with `export-grpo` and document the local Phase 3B workflow.

**Tech Stack:** Python 3.10+, pytest, JSON / JSONL, standard library only.

---

## File Structure

Create:

- `src/lightningsearch_rl/grpo.py`: export GRPO-compatible rollout, transition, reward, and summary artifacts.
- `tests/test_grpo.py`: module-level tests for `export_grpo`.

Modify:

- `src/lightningsearch_rl/cli.py`: add `export-grpo`.
- `tests/test_cli.py`: add CLI pipeline coverage.
- `README.md`: document Phase 3B workflow and artifacts.

Do not modify:

- `src/lightningsearch_rl/rewards.py`: use the existing `compute_reward` contract.
- `src/lightningsearch_rl/transitions.py`: use the existing `build_transitions` contract.
- Remote experiment files or AGENTS instructions.

## Chunk 1: GRPO Export Module

### Task 1: Add `export_grpo`

**Files:**
- Create: `src/lightningsearch_rl/grpo.py`
- Create: `tests/test_grpo.py`

- [ ] **Step 1: Write the failing GRPO export test**

Create `tests/test_grpo.py`:

```python
import json
from pathlib import Path

from lightningsearch_rl.adapters import convert_hotpot_file
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.grpo import export_grpo
from lightningsearch_rl.index_store import save_lexical_index


def test_export_grpo_writes_rollouts_transitions_rewards_and_summary(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "grpo"
    convert_hotpot_file(Path("tests/fixtures/hotpot_mixed_raw.jsonl"), corpus, examples, limit=1)
    save_lexical_index(index, load_corpus_jsonl(corpus))

    summary = export_grpo(examples, index, out_dir, top_k=2)

    rollout = json.loads((out_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    reward = json.loads(
        (out_dir / "reward_records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    transition = json.loads(
        (out_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )

    assert summary["rollout_count"] == 1
    assert summary["transition_count"] == 2
    assert summary["avg_reward"] == 1.37
    assert rollout["prompt"] == "Which city is the birthplace of the author of Example Book?"
    assert "<answer>Example City</answer>" in rollout["response"]
    assert rollout["reward"] == 1.37
    assert rollout["metadata"]["answer"] == "Example City"
    assert rollout["metadata"]["search_count"] == 1
    assert rollout["metadata"]["gold_doc_ids"] == ["hotpot::Alice Smith::0"]
    assert rollout["metadata"]["retrieved_doc_ids"] == [
        "hotpot::Alice Smith::0",
        "hotpot::Example Book::0",
    ]
    assert reward["total"] == 1.37
    assert reward["answer_reward"] == 1.0
    assert reward["evidence_reward"] == 1.0
    assert reward["format_reward"] == 1.0
    assert reward["tool_validity_reward"] == 1.0
    assert reward["search_count"] == 1
    assert reward["search_cost"] == 0.03
    assert transition["terminal"] is True
    assert transition["reward"] == 1.37
    assert (out_dir / "summary.json").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_grpo.py -v
```

Expected: FAIL during import because `lightningsearch_rl.grpo` does not exist.

- [ ] **Step 3: Implement `src/lightningsearch_rl/grpo.py`**

Implement:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightningsearch_rl.data import QAExample, load_jsonl_examples
from lightningsearch_rl.formatting import format_assistant_trace
from lightningsearch_rl.index_store import load_lexical_index
from lightningsearch_rl.rewards import RewardBreakdown, compute_reward
from lightningsearch_rl.runtime import EpisodeTrace, run_retrieval_episode
from lightningsearch_rl.transitions import build_transitions


SEARCH_COST = 0.03


def export_grpo(examples_path: Path, index_path: Path, out_dir: Path, top_k: int = 5) -> dict[str, Any]:
    examples = load_jsonl_examples(examples_path)
    retriever = load_lexical_index(index_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollouts: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []
    for example in examples:
        trace = run_retrieval_episode(example, retriever, top_k=top_k)
        reward = compute_reward(example, trace)
        rewarded_trace = EpisodeTrace(
            question_id=trace.question_id,
            question=trace.question,
            steps=trace.steps,
            final_answer=trace.final_answer,
            reward=reward.total,
            metadata={**trace.metadata, "reward": reward.total},
        )
        rollouts.append(_build_rollout_row(example, rewarded_trace))
        transitions.extend(transition.to_dict() for transition in build_transitions(rewarded_trace))
        reward_records.append(_build_reward_row(example.id, reward))

    _write_jsonl(out_dir / "rollouts.jsonl", rollouts)
    _write_jsonl(out_dir / "transitions.jsonl", transitions)
    _write_jsonl(out_dir / "reward_records.jsonl", reward_records)
    summary = {
        "example_count": len(examples),
        "rollout_count": len(rollouts),
        "transition_count": len(transitions),
        "avg_reward": _average(row["reward"] for row in rollouts),
        "avg_search_count": _average(row["metadata"]["search_count"] for row in rollouts),
        "top_k": top_k,
        "artifacts": {
            "rollouts": str(out_dir / "rollouts.jsonl"),
            "transitions": str(out_dir / "transitions.jsonl"),
            "reward_records": str(out_dir / "reward_records.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _build_rollout_row(example: QAExample, trace: EpisodeTrace) -> dict[str, Any]:
    return {
        "id": example.id,
        "prompt": example.question,
        "response": format_assistant_trace(trace),
        "reward": trace.reward,
        "metadata": {
            "answer": trace.final_answer,
            "search_count": trace.metadata["search_count"],
            "gold_doc_ids": example.gold_doc_ids,
            "retrieved_doc_ids": _retrieved_doc_ids(trace),
        },
    }


def _build_reward_row(example_id: str, reward: RewardBreakdown) -> dict[str, Any]:
    return {
        **reward.to_dict(),
        "id": example_id,
        "search_cost": round(SEARCH_COST * reward.search_count, 6),
    }


def _retrieved_doc_ids(trace: EpisodeTrace) -> list[str]:
    return [
        passage.doc_id
        for step in trace.steps
        if step.action_type == "search" and step.observation
        for passage in step.observation
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _average(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(float(value) for value in items) / len(items), 6)
```

- [ ] **Step 4: Run the GRPO module test**

Run:

```bash
python -m pytest tests/test_grpo.py -v
```

Expected: PASS.

- [ ] **Step 5: Run related regression tests**

Run:

```bash
python -m pytest tests/test_sft.py tests/test_rewards.py tests/test_transitions.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/grpo.py tests/test_grpo.py
git commit -m "feat: export grpo rollout records"
```

## Chunk 2: CLI And Documentation

### Task 2: Add `export-grpo` CLI

**Files:**
- Modify: `src/lightningsearch_rl/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI test**

Add this test to `tests/test_cli.py`:

```python
def test_export_grpo_cli_writes_rollouts_transitions_rewards_and_summary(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    examples = tmp_path / "examples.jsonl"
    index = tmp_path / "index.json"
    out_dir = tmp_path / "grpo"

    assert (
        main(
            [
                "prepare-hotpot",
                "--raw",
                "tests/fixtures/hotpot_mixed_raw.jsonl",
                "--corpus",
                str(corpus),
                "--examples",
                str(examples),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert main(["build-index", "--corpus", str(corpus), "--index", str(index)]) == 0
    assert (
        main(
            [
                "export-grpo",
                "--examples",
                str(examples),
                "--index",
                str(index),
                "--out-dir",
                str(out_dir),
                "--top-k",
                "2",
            ]
        )
        == 0
    )

    assert (out_dir / "rollouts.jsonl").exists()
    assert (out_dir / "transitions.jsonl").exists()
    assert (out_dir / "reward_records.jsonl").exists()
    assert (out_dir / "summary.json").exists()
```

- [ ] **Step 2: Run CLI tests to verify the new test fails**

Run:

```bash
python -m pytest tests/test_cli.py -v
```

Expected: FAIL because `export-grpo` is not an argparse subcommand.

- [ ] **Step 3: Implement CLI support**

Modify `src/lightningsearch_rl/cli.py`:

1. Add import:

```python
from lightningsearch_rl.grpo import export_grpo
```

2. Add parser near `export-sft`:

```python
export_grpo_parser = subparsers.add_parser("export-grpo")
export_grpo_parser.add_argument("--examples", required=True)
export_grpo_parser.add_argument("--index", required=True)
export_grpo_parser.add_argument("--out-dir", required=True)
export_grpo_parser.add_argument("--top-k", type=int, default=5)
```

3. Add dispatch:

```python
if args.command == "export-grpo":
    export_grpo(
        Path(args.examples),
        Path(args.index),
        Path(args.out_dir),
        top_k=args.top_k,
    )
    return 0
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
python -m pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full tests**

Run:

```bash
python -m pytest
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/cli.py tests/test_cli.py
git commit -m "feat: add grpo export cli"
```

### Task 3: Document Phase 3B workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Append a section:

````markdown
## Phase 3B GRPO Export Workflow

Prepare a limited HotpotQA-like file and build the shared lexical index:

```bash
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_mixed_raw.jsonl --corpus results/phase3b/corpus.jsonl --examples results/phase3b/examples.jsonl --limit 1
python -m lightningsearch_rl.cli build-index --corpus results/phase3b/corpus.jsonl --index results/phase3b/index.json
```

Export rollout, transition, and reward records for later GRPO training:

```bash
python -m lightningsearch_rl.cli export-grpo --examples results/phase3b/examples.jsonl --index results/phase3b/index.json --out-dir results/phase3b/grpo --top-k 2
```

Expected GRPO artifacts:

- `results/phase3b/grpo/rollouts.jsonl`
- `results/phase3b/grpo/transitions.jsonl`
- `results/phase3b/grpo/reward_records.jsonl`
- `results/phase3b/grpo/summary.json`
````

- [ ] **Step 2: Run final verification**

Run:

```bash
python -m pytest
python -m lightningsearch_rl.cli prepare-hotpot --raw tests/fixtures/hotpot_mixed_raw.jsonl --corpus results/phase3b/corpus.jsonl --examples results/phase3b/examples.jsonl --limit 1
python -m lightningsearch_rl.cli build-index --corpus results/phase3b/corpus.jsonl --index results/phase3b/index.json
python -m lightningsearch_rl.cli export-grpo --examples results/phase3b/examples.jsonl --index results/phase3b/index.json --out-dir results/phase3b/grpo --top-k 2
```

Expected: tests PASS and all four GRPO artifacts exist.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document phase 3b grpo export workflow"
```

## Guardrails

- Use @superpowers:test-driven-development for every production behavior.
- Use @superpowers:using-git-worktrees before implementation.
- Use @superpowers:verification-before-completion before claiming completion.
- Do not add model inference.
- Do not add external dependencies.
- Do not launch remote commands.
- Preserve existing `smoke`, `retrieval-baseline`, and `export-sft` commands.
