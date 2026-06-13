# LightningSearch-RL MVP Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, testable MVP for a Lightning-style search-agent RL framework that runs tiny offline multi-hop QA examples end to end and exports traces, transitions, rewards, and metrics.

**Architecture:** Implement a small Python package with isolated modules for data loading, retrieval, action parsing, runtime tracing, transition building, reward calculation, evaluation, and a smoke CLI. Use deterministic fixtures and a rule-based baseline policy first so the Agent RL interfaces are stable before adding Qwen / verl / remote training.

**Tech Stack:** Python 3.10+, pytest, standard-library dataclasses / JSON, optional `pyproject.toml` packaging, no heavyweight retrieval or training dependencies in Phase 1.

---

## File Structure

Create the following files:

- `pyproject.toml`: package metadata, pytest config, Python version, console/package discovery.
- `README.md`: local MVP purpose, setup, smoke command, remote-training boundary.
- `src/lightningsearch_rl/__init__.py`: package version marker.
- `src/lightningsearch_rl/data.py`: JSONL dataset loader and `QAExample` / `Passage` dataclasses.
- `src/lightningsearch_rl/actions.py`: parser for `<think>`, `<search>`, `<answer>` tags and action validation.
- `src/lightningsearch_rl/retrieval.py`: deterministic lexical retriever for MVP.
- `src/lightningsearch_rl/runtime.py`: tool loop, baseline policy, trace dataclasses, trace serialization.
- `src/lightningsearch_rl/transitions.py`: trace-to-transition adapter.
- `src/lightningsearch_rl/rewards.py`: reward components and total reward.
- `src/lightningsearch_rl/eval.py`: metrics aggregation.
- `src/lightningsearch_rl/cli.py`: `smoke` command.
- `tests/fixtures/tiny_multihop.jsonl`: tiny deterministic dataset.
- `tests/test_data.py`: loader tests.
- `tests/test_actions.py`: parser tests.
- `tests/test_retrieval.py`: retriever tests.
- `tests/test_runtime.py`: trace/runtime tests.
- `tests/test_transitions.py`: transition adapter tests.
- `tests/test_rewards.py`: reward tests.
- `tests/test_eval.py`: metric tests.
- `tests/test_cli.py`: smoke CLI artifact test.

Keep generated outputs under ignored directories such as `results/`, `logs/`, `checkpoints/`, `runs/`, `data/`, and `indexes/`.

## Chunk 1: Repository Baseline

### Task 1: Add Python Project Metadata

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/lightningsearch_rl/__init__.py`

- [ ] **Step 1: Write minimal package metadata**

Add `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "lightningsearch-rl"
version = "0.1.0"
description = "Local MVP for retrieval tool-use Agent RL traces, rewards, and evaluation."
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Add package marker**

Create `src/lightningsearch_rl/__init__.py`:

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Add README**

Create `README.md` with:

```markdown
# LightningSearch-RL

Local MVP for a Lightning-style retrieval tool-use Agent RL framework.

Phase 1 validates the local contracts for data loading, offline retrieval, action parsing, trace collection, transition building, shaped rewards, and smoke evaluation. It does not run remote training or call external search APIs.

## Smoke Target

```bash
python -m pytest
python -m lightningsearch_rl.cli smoke --data tests/fixtures/tiny_multihop.jsonl --out-dir results/smoke
```
```

- [ ] **Step 4: Verify baseline metadata**

Run:

```bash
python -m pytest
```

Expected: pytest starts and reports no tests collected or passes existing tests.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md src/lightningsearch_rl/__init__.py .gitignore AGENTS.md docs/superpowers/specs/2026-06-13-lightningsearch-rl-mvp-design.md docs/superpowers/plans/2026-06-13-lightningsearch-rl-mvp.md
git commit -m "chore: initialize lightningsearch rl planning"
```

## Chunk 2: Data And Action Contracts

### Task 2: Implement Dataset Loader With TDD

**Files:**
- Create: `tests/fixtures/tiny_multihop.jsonl`
- Create: `tests/test_data.py`
- Create: `src/lightningsearch_rl/data.py`

- [ ] **Step 1: Write the fixture**

Create `tests/fixtures/tiny_multihop.jsonl`:

```jsonl
{"id":"ex1","question":"Which city is the birthplace of the author of Example Book?","answers":["Example City"],"gold_doc_ids":["doc_author"],"corpus":[{"doc_id":"doc_book","title":"Example Book","text":"Example Book was written by Alice Smith."},{"doc_id":"doc_author","title":"Alice Smith","text":"Alice Smith was born in Example City."},{"doc_id":"doc_noise","title":"Noise","text":"This passage is unrelated."}]}
```

- [ ] **Step 2: Write the failing loader test**

Create `tests/test_data.py`:

```python
from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples


def test_load_jsonl_examples_parses_fixture():
    examples = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))

    assert len(examples) == 1
    example = examples[0]
    assert example.id == "ex1"
    assert example.answers == ["Example City"]
    assert example.gold_doc_ids == ["doc_author"]
    assert example.corpus[0].doc_id == "doc_book"
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_data.py -v
```

Expected: FAIL because `lightningsearch_rl.data` does not exist.

- [ ] **Step 4: Implement minimal loader**

Create `src/lightningsearch_rl/data.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    answers: list[str]
    gold_doc_ids: list[str]
    corpus: list[Passage]


def load_jsonl_examples(path: Path) -> list[QAExample]:
    examples: list[QAExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                QAExample(
                    id=row["id"],
                    question=row["question"],
                    answers=list(row["answers"]),
                    gold_doc_ids=list(row.get("gold_doc_ids", [])),
                    corpus=[
                        Passage(
                            doc_id=item["doc_id"],
                            title=item.get("title", ""),
                            text=item["text"],
                        )
                        for item in row.get("corpus", [])
                    ],
                )
            )
    return examples
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_data.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lightningsearch_rl/data.py tests/test_data.py tests/fixtures/tiny_multihop.jsonl
git commit -m "feat: load jsonl qa examples"
```

### Task 3: Implement Action Parser With TDD

**Files:**
- Create: `tests/test_actions.py`
- Create: `src/lightningsearch_rl/actions.py`

- [ ] **Step 1: Write failing parser tests**

Create `tests/test_actions.py`:

```python
from lightningsearch_rl.actions import parse_action


def test_parse_search_action_extracts_query():
    action = parse_action("<search>Example Book author</search>")

    assert action.type == "search"
    assert action.content == "Example Book author"
    assert action.valid is True


def test_parse_answer_action_extracts_answer():
    action = parse_action("<answer>Example City</answer>")

    assert action.type == "answer"
    assert action.content == "Example City"
    assert action.valid is True


def test_parse_empty_search_is_invalid():
    action = parse_action("<search>   </search>")

    assert action.type == "search"
    assert action.valid is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_actions.py -v
```

Expected: FAIL because `lightningsearch_rl.actions` does not exist.

- [ ] **Step 3: Implement minimal parser**

Create `src/lightningsearch_rl/actions.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedAction:
    type: str
    content: str
    valid: bool
    raw: str


_ACTION_RE = re.compile(r"^\s*<(think|search|answer)>(.*?)</\1>\s*$", re.DOTALL)


def parse_action(raw: str) -> ParsedAction:
    match = _ACTION_RE.match(raw)
    if not match:
        return ParsedAction(type="invalid", content="", valid=False, raw=raw)
    action_type, content = match.group(1), match.group(2).strip()
    return ParsedAction(
        type=action_type,
        content=content,
        valid=bool(content),
        raw=raw,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_actions.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/actions.py tests/test_actions.py
git commit -m "feat: parse structured agent actions"
```

## Chunk 3: Retrieval, Runtime, And Trace Adapter

### Task 4: Implement Lexical Retriever With TDD

**Files:**
- Create: `tests/test_retrieval.py`
- Create: `src/lightningsearch_rl/retrieval.py`

- [ ] **Step 1: Write failing retriever test**

Create `tests/test_retrieval.py`:

```python
from lightningsearch_rl.data import Passage
from lightningsearch_rl.retrieval import LexicalRetriever


def test_search_ranks_matching_passage_first():
    retriever = LexicalRetriever(
        [
            Passage("doc_book", "Example Book", "Example Book was written by Alice Smith."),
            Passage("doc_author", "Alice Smith", "Alice Smith was born in Example City."),
            Passage("doc_noise", "Noise", "Unrelated text."),
        ]
    )

    results = retriever.search("Alice Smith born city", top_k=2)

    assert [result.doc_id for result in results] == ["doc_author", "doc_book"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_retrieval.py -v
```

Expected: FAIL because `LexicalRetriever` does not exist.

- [ ] **Step 3: Implement deterministic lexical retriever**

Create `src/lightningsearch_rl/retrieval.py`:

```python
from __future__ import annotations

import math
import re

from lightningsearch_rl.data import Passage


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class LexicalRetriever:
    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self._tokens = [tokenize(f"{p.title} {p.text}") for p in passages]
        self._doc_freq: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

    def search(self, query: str, top_k: int = 5) -> list[Passage]:
        query_tokens = tokenize(query)
        scored = [
            (self._score(query_tokens, doc_tokens), index, passage)
            for index, (doc_tokens, passage) in enumerate(zip(self._tokens, self.passages))
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [passage for score, _, passage in scored[:top_k] if score > 0]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        doc_len = len(doc_tokens)
        counts: dict[str, int] = {}
        for token in doc_tokens:
            counts[token] = counts.get(token, 0) + 1
        score = 0.0
        num_docs = max(len(self.passages), 1)
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(token, 0)
            idf = math.log((num_docs + 1) / (df + 1)) + 1.0
            score += idf * tf / doc_len
        return score
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_retrieval.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/retrieval.py tests/test_retrieval.py
git commit -m "feat: add deterministic lexical retriever"
```

### Task 5: Implement Runtime Trace Collection With TDD

**Files:**
- Create: `tests/test_runtime.py`
- Create: `src/lightningsearch_rl/runtime.py`

- [ ] **Step 1: Write failing runtime test**

Create `tests/test_runtime.py`:

```python
from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.runtime import run_rule_based_episode


def test_rule_based_episode_collects_search_and_answer_trace():
    example = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))[0]

    trace = run_rule_based_episode(example, top_k=2)

    assert trace.question_id == "ex1"
    assert trace.final_answer == "Example City"
    assert [step.action_type for step in trace.steps] == ["search", "answer"]
    assert trace.steps[0].observation[0].doc_id == "doc_author"
    assert trace.metadata["search_count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_runtime.py -v
```

Expected: FAIL because `runtime` does not exist.

- [ ] **Step 3: Implement minimal runtime**

Create `src/lightningsearch_rl/runtime.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass

from lightningsearch_rl.data import Passage, QAExample
from lightningsearch_rl.retrieval import LexicalRetriever


@dataclass(frozen=True)
class TraceStep:
    state: str
    action: str
    action_type: str
    query: str | None = None
    observation: list[Passage] | None = None
    valid_tool_call: bool = True
    terminal: bool = False


@dataclass(frozen=True)
class EpisodeTrace:
    question_id: str
    question: str
    steps: list[TraceStep]
    final_answer: str
    reward: float | None
    metadata: dict[str, int | str | float]

    def to_dict(self) -> dict:
        return asdict(self)


def run_rule_based_episode(example: QAExample, top_k: int = 5) -> EpisodeTrace:
    retriever = LexicalRetriever(example.corpus)
    query = example.question
    observation = retriever.search(query, top_k=top_k)
    final_answer = _answer_from_observation(example, observation)
    steps = [
        TraceStep(
            state=example.question,
            action=f"<search>{query}</search>",
            action_type="search",
            query=query,
            observation=observation,
            valid_tool_call=bool(query.strip()),
            terminal=False,
        ),
        TraceStep(
            state=f"{example.question}\nObservation: {observation}",
            action=f"<answer>{final_answer}</answer>",
            action_type="answer",
            terminal=True,
        ),
    ]
    return EpisodeTrace(
        question_id=example.id,
        question=example.question,
        steps=steps,
        final_answer=final_answer,
        reward=None,
        metadata={"search_count": 1},
    )


def _answer_from_observation(example: QAExample, observation: list[Passage]) -> str:
    observed_text = " ".join(p.text for p in observation).lower()
    for answer in example.answers:
        if answer.lower() in observed_text:
            return answer
    return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_runtime.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/runtime.py tests/test_runtime.py
git commit -m "feat: collect rule based episode traces"
```

### Task 6: Implement Trace-To-Transition Adapter With TDD

**Files:**
- Create: `tests/test_transitions.py`
- Create: `src/lightningsearch_rl/transitions.py`

- [ ] **Step 1: Write failing transition test**

Create `tests/test_transitions.py`:

```python
from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.runtime import run_rule_based_episode
from lightningsearch_rl.transitions import build_transitions


def test_build_transitions_preserves_state_action_boundaries():
    example = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))[0]
    trace = run_rule_based_episode(example, top_k=2)

    transitions = build_transitions(trace)

    assert len(transitions) == 2
    assert transitions[0].state == trace.steps[0].state
    assert transitions[0].action == trace.steps[0].action
    assert transitions[0].terminal is False
    assert transitions[1].terminal is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_transitions.py -v
```

Expected: FAIL because `transitions` does not exist.

- [ ] **Step 3: Implement minimal adapter**

Create `src/lightningsearch_rl/transitions.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass

from lightningsearch_rl.runtime import EpisodeTrace


@dataclass(frozen=True)
class Transition:
    question_id: str
    step_index: int
    state: str
    action: str
    action_type: str
    observation: list[dict] | None
    terminal: bool
    reward: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def build_transitions(trace: EpisodeTrace) -> list[Transition]:
    transitions: list[Transition] = []
    for index, step in enumerate(trace.steps):
        transitions.append(
            Transition(
                question_id=trace.question_id,
                step_index=index,
                state=step.state,
                action=step.action,
                action_type=step.action_type,
                observation=[asdict(p) for p in step.observation] if step.observation else None,
                terminal=step.terminal,
                reward=trace.reward if step.terminal else None,
            )
        )
    return transitions
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_transitions.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/transitions.py tests/test_transitions.py
git commit -m "feat: convert traces to transitions"
```

## Chunk 4: Rewards, Evaluation, And CLI

### Task 7: Implement Reward Calculation With TDD

**Files:**
- Create: `tests/test_rewards.py`
- Create: `src/lightningsearch_rl/rewards.py`

- [ ] **Step 1: Write failing reward tests**

Create `tests/test_rewards.py`:

```python
from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.rewards import compute_reward
from lightningsearch_rl.runtime import run_rule_based_episode


def test_compute_reward_combines_answer_evidence_format_tool_and_cost():
    example = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))[0]
    trace = run_rule_based_episode(example, top_k=2)

    reward = compute_reward(example, trace)

    assert reward.answer_reward == 1.0
    assert reward.evidence_reward == 1.0
    assert reward.format_reward == 1.0
    assert reward.tool_validity_reward == 1.0
    assert reward.search_count == 1
    assert reward.total == 1.37
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_rewards.py -v
```

Expected: FAIL because `rewards` does not exist.

- [ ] **Step 3: Implement reward logic**

Create `src/lightningsearch_rl/rewards.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import re

from lightningsearch_rl.data import QAExample
from lightningsearch_rl.runtime import EpisodeTrace


@dataclass(frozen=True)
class RewardBreakdown:
    answer_reward: float
    evidence_reward: float
    format_reward: float
    tool_validity_reward: float
    search_count: int
    total: float

    def to_dict(self) -> dict:
        return {
            "answer_reward": self.answer_reward,
            "evidence_reward": self.evidence_reward,
            "format_reward": self.format_reward,
            "tool_validity_reward": self.tool_validity_reward,
            "search_count": self.search_count,
            "total": self.total,
        }


def compute_reward(example: QAExample, trace: EpisodeTrace) -> RewardBreakdown:
    answer_reward = 1.0 if _normalize(trace.final_answer) in {_normalize(a) for a in example.answers} else 0.0
    retrieved_doc_ids = {
        passage.doc_id
        for step in trace.steps
        for passage in (step.observation or [])
    }
    evidence_reward = (
        len(set(example.gold_doc_ids) & retrieved_doc_ids) / len(example.gold_doc_ids)
        if example.gold_doc_ids
        else 0.0
    )
    format_reward = 1.0 if trace.steps and trace.steps[-1].terminal and trace.steps[-1].action_type == "answer" else 0.0
    search_queries = [step.query for step in trace.steps if step.action_type == "search"]
    non_empty = all(query and query.strip() for query in search_queries)
    no_duplicates = len([q.lower() for q in search_queries if q]) == len({q.lower() for q in search_queries if q})
    tool_validity_reward = 1.0 if non_empty and no_duplicates else 0.0
    search_count = len(search_queries)
    total = answer_reward + 0.2 * evidence_reward + 0.1 * format_reward + 0.1 * tool_validity_reward - 0.03 * search_count
    return RewardBreakdown(
        answer_reward=answer_reward,
        evidence_reward=evidence_reward,
        format_reward=format_reward,
        tool_validity_reward=tool_validity_reward,
        search_count=search_count,
        total=round(total, 6),
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_rewards.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/rewards.py tests/test_rewards.py
git commit -m "feat: compute shaped agent rewards"
```

### Task 8: Implement Evaluation Metrics With TDD

**Files:**
- Create: `tests/test_eval.py`
- Create: `src/lightningsearch_rl/eval.py`

- [ ] **Step 1: Write failing metric test**

Create `tests/test_eval.py`:

```python
from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.eval import evaluate_traces
from lightningsearch_rl.runtime import run_rule_based_episode


def test_evaluate_traces_reports_core_metrics():
    example = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))[0]
    trace = run_rule_based_episode(example, top_k=2)

    metrics = evaluate_traces([example], [trace])

    assert metrics["answer_em"] == 1.0
    assert metrics["evidence_recall"] == 1.0
    assert metrics["avg_search_calls"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_eval.py -v
```

Expected: FAIL because `eval` does not exist.

- [ ] **Step 3: Implement metric aggregation**

Create `src/lightningsearch_rl/eval.py`:

```python
from __future__ import annotations

from lightningsearch_rl.data import QAExample
from lightningsearch_rl.rewards import compute_reward
from lightningsearch_rl.runtime import EpisodeTrace


def evaluate_traces(examples: list[QAExample], traces: list[EpisodeTrace]) -> dict[str, float]:
    if len(examples) != len(traces):
        raise ValueError("examples and traces must have the same length")
    if not examples:
        return {"answer_em": 0.0, "evidence_recall": 0.0, "avg_search_calls": 0.0, "avg_reward": 0.0}
    rewards = [compute_reward(example, trace) for example, trace in zip(examples, traces)]
    count = len(rewards)
    return {
        "answer_em": sum(r.answer_reward for r in rewards) / count,
        "evidence_recall": sum(r.evidence_reward for r in rewards) / count,
        "avg_search_calls": sum(r.search_count for r in rewards) / count,
        "avg_reward": sum(r.total for r in rewards) / count,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_eval.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightningsearch_rl/eval.py tests/test_eval.py
git commit -m "feat: aggregate search agent metrics"
```

### Task 9: Implement Smoke CLI With TDD

**Files:**
- Create: `tests/test_cli.py`
- Create: `src/lightningsearch_rl/cli.py`

- [ ] **Step 1: Write failing CLI test**

Create `tests/test_cli.py`:

```python
import json
from pathlib import Path

from lightningsearch_rl.cli import main


def test_smoke_cli_writes_artifacts(tmp_path):
    out_dir = tmp_path / "smoke"

    exit_code = main([
        "smoke",
        "--data",
        "tests/fixtures/tiny_multihop.jsonl",
        "--out-dir",
        str(out_dir),
    ])

    assert exit_code == 0
    assert (out_dir / "traces.jsonl").exists()
    assert (out_dir / "transitions.jsonl").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["answer_em"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_cli.py -v
```

Expected: FAIL because `cli` does not exist.

- [ ] **Step 3: Implement smoke CLI**

Create `src/lightningsearch_rl/cli.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.eval import evaluate_traces
from lightningsearch_rl.rewards import compute_reward
from lightningsearch_rl.runtime import run_rule_based_episode
from lightningsearch_rl.transitions import build_transitions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lightningsearch-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--data", required=True)
    smoke.add_argument("--out-dir", required=True)
    smoke.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        return _run_smoke(Path(args.data), Path(args.out_dir), args.top_k)
    raise ValueError(f"unknown command: {args.command}")


def _run_smoke(data_path: Path, out_dir: Path, top_k: int) -> int:
    examples = load_jsonl_examples(data_path)
    traces = []
    transitions = []
    for example in examples:
        trace = run_rule_based_episode(example, top_k=top_k)
        reward = compute_reward(example, trace)
        trace = trace.__class__(
            question_id=trace.question_id,
            question=trace.question,
            steps=trace.steps,
            final_answer=trace.final_answer,
            reward=reward.total,
            metadata={**trace.metadata, "reward": reward.total},
        )
        traces.append(trace)
        transitions.extend(build_transitions(trace))
    metrics = evaluate_traces(examples, traces)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "traces.jsonl", [trace.to_dict() for trace in traces])
    _write_jsonl(out_dir / "transitions.jsonl", [transition.to_dict() for transition in transitions])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run:

```bash
python -m pytest
```

Expected: all tests PASS.

- [ ] **Step 6: Run local smoke command**

Run:

```bash
python -m lightningsearch_rl.cli smoke --data tests/fixtures/tiny_multihop.jsonl --out-dir results/smoke
```

Expected:

- `results/smoke/traces.jsonl` exists.
- `results/smoke/transitions.jsonl` exists.
- `results/smoke/metrics.json` contains `answer_em: 1.0`.

- [ ] **Step 7: Commit**

```bash
git add src/lightningsearch_rl/cli.py tests/test_cli.py
git commit -m "feat: add local smoke evaluation cli"
```

## Chunk 5: Final Verification And Handoff

### Task 10: Verify MVP Plan Completion

**Files:**
- Modify: `README.md`
- Optional create: `docs/experiments/.gitkeep`

- [ ] **Step 1: Update README with observed commands**

After implementation, add the exact passing commands and artifact locations to `README.md`.

- [ ] **Step 2: Run full verification**

Run:

```bash
python -m pytest
python -m lightningsearch_rl.cli smoke --data tests/fixtures/tiny_multihop.jsonl --out-dir results/smoke
git status --short
```

Expected:

- pytest reports all tests passed.
- smoke command exits `0`.
- `git status --short` shows only expected ignored generated outputs, or is clean after commit.

- [ ] **Step 3: Commit final documentation**

```bash
git add README.md docs/experiments/.gitkeep
git commit -m "docs: document local mvp smoke workflow"
```

- [ ] **Step 4: Prepare next-phase notes**

Add follow-up issues or notes for:

- HotpotQA / 2Wiki adapter.
- Shared corpus index.
- Qwen rollout policy.
- SFT trajectory export.
- verl GRPO integration.
- Remote smoke test under `AGENTS.md`.

## Implementation Guardrails

- Use @superpowers:test-driven-development for every production module.
- Use @superpowers:verification-before-completion before claiming a task is complete.
- Do not start remote training in Phase 1.
- Do not add Qwen, vLLM, FAISS, or verl dependencies until the local MVP tests pass.
- Do not write generated data or results into git.
- Keep every module small and focused.

