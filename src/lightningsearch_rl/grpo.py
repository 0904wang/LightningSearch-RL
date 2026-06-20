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
            "answer": _gold_answer(example),
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


def _gold_answer(example: QAExample) -> str:
    for answer in example.answers:
        if answer.strip():
            return answer
    return ""


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _average(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(float(value) for value in items) / len(items), 6)
