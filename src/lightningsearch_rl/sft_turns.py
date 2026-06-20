from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from lightningsearch_rl.agent_loop import parse_agent_action
from lightningsearch_rl.data import Passage, QAExample, load_jsonl_examples
from lightningsearch_rl.formatting import format_observation
from lightningsearch_rl.index_store import load_lexical_index
from lightningsearch_rl.runtime import EpisodeTrace, TraceStep


SYSTEM_PROMPT = (
    "You are a search agent for multi-hop QA. Output exactly one action per turn: "
    "<search>...</search> or <answer>...</answer>. Never output <observation>; "
    "observations are provided by the environment. Do not put <answer> inside <search>. "
    "Use only provided observations when answering."
)


def export_sft_turns(examples_path: Path, index_path: Path, out_dir: Path) -> dict[str, Any]:
    examples = load_jsonl_examples(examples_path)
    retriever = load_lexical_index(index_path)
    passages_by_id = {passage.doc_id: passage for passage in retriever.passages}
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    coverage_hits = 0
    for example in examples:
        gold_passages = [passages_by_id[doc_id] for doc_id in example.gold_doc_ids if doc_id in passages_by_id]
        answer = _gold_answer(example)
        _ensure_answer_in_gold_evidence(example, gold_passages, answer)
        trace = _build_turn_trace(example, gold_passages, answer)
        rows.append(_build_sft_turn_row(example, trace, gold_passages, answer))
        traces.append(trace.to_dict())
        if example.gold_doc_ids and len(gold_passages) == len(example.gold_doc_ids):
            coverage_hits += 1

    _write_jsonl(out_dir / "sft_turns.jsonl", rows)
    _write_jsonl(out_dir / "traces.jsonl", traces)
    summary = {
        "example_count": len(examples),
        "sft_rows": len(rows),
        "answer_tag_rate": _rate("<answer>" in row["messages"][-1]["content"] for row in rows),
        "assistant_observation_rate": _rate(
            any(message["role"] == "assistant" and "<observation>" in message["content"] for message in row["messages"])
            for row in rows
        ),
        "assistant_single_action_rate": _rate(_assistant_messages_are_single_actions(row["messages"]) for row in rows),
        "non_empty_answer_rate": _rate(bool(row["metadata"]["answer"]) for row in rows),
        "gold_evidence_coverage": round(coverage_hits / len(examples), 6) if examples else 0.0,
        "avg_gold_evidence_count": _average(len(row["metadata"]["gold_evidence_doc_ids"]) for row in rows),
        "artifacts": {
            "sft_turns": str(out_dir / "sft_turns.jsonl"),
            "traces": str(out_dir / "traces.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _build_turn_trace(example: QAExample, gold_passages: list[Passage], answer: str) -> EpisodeTrace:
    search_action = f"<search>{example.question}</search>"
    return EpisodeTrace(
        question_id=example.id,
        question=example.question,
        steps=[
            TraceStep(
                state=example.question,
                action=search_action,
                action_type="search",
                query=example.question,
                observation=gold_passages,
                valid_tool_call=True,
                terminal=False,
            ),
            TraceStep(
                state=f"{example.question}\n{format_observation(gold_passages)}",
                action=f"<answer>{answer}</answer>",
                action_type="answer",
                terminal=True,
            ),
        ],
        final_answer=answer,
        reward=None,
        metadata={"search_count": 1},
    )


def _build_sft_turn_row(
    example: QAExample,
    trace: EpisodeTrace,
    gold_passages: list[Passage],
    answer: str,
) -> dict[str, Any]:
    search_step = trace.steps[0]
    answer_step = trace.steps[-1]
    return {
        "id": example.id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example.question},
            {"role": "assistant", "content": search_step.action},
            {"role": "user", "content": format_observation(gold_passages)},
            {"role": "assistant", "content": answer_step.action},
        ],
        "metadata": {
            "answer": answer,
            "search_count": trace.metadata["search_count"],
            "gold_doc_ids": example.gold_doc_ids,
            "gold_evidence_doc_ids": [passage.doc_id for passage in gold_passages],
            "turn_style": "single_action_with_runtime_observation",
        },
    }


def _assistant_messages_are_single_actions(messages: list[dict[str, str]]) -> bool:
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    return all(parse_agent_action(message["content"]).valid for message in assistant_messages)


def _gold_answer(example: QAExample) -> str:
    for answer in example.answers:
        if answer.strip():
            return answer
    return ""


def _ensure_answer_in_gold_evidence(example: QAExample, gold_passages: list[Passage], answer: str) -> None:
    normalized_answer = _normalize_text(answer)
    normalized_evidence = _normalize_text(" ".join(passage.text for passage in gold_passages))
    if normalized_answer and normalized_answer not in normalized_evidence:
        raise ValueError(f"answer not found in gold evidence for example {example.id}: {answer}")


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rate(values: Any) -> float:
    items = list(values)
    return round(sum(1 for value in items if value) / len(items), 6) if items else 0.0


def _average(values: Any) -> float:
    items = list(values)
    return round(sum(float(value) for value in items) / len(items), 6) if items else 0.0
