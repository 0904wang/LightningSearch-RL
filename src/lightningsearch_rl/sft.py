from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightningsearch_rl.data import QAExample, load_jsonl_examples
from lightningsearch_rl.formatting import format_assistant_trace
from lightningsearch_rl.index_store import load_lexical_index
from lightningsearch_rl.runtime import EpisodeTrace, run_retrieval_episode


SYSTEM_PROMPT = (
    "You are a search agent. Use <think>, <search>, <observation>, and <answer> tags "
    "to answer the question with retrieved evidence."
)


def export_sft(examples_path: Path, index_path: Path, out_dir: Path, top_k: int = 5) -> dict[str, Any]:
    examples = load_jsonl_examples(examples_path)
    retriever = load_lexical_index(index_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for example in examples:
        trace = run_retrieval_episode(example, retriever, top_k=top_k)
        sft_rows.append(_build_sft_row(example, trace))
        traces.append(trace.to_dict())

    _write_jsonl(out_dir / "sft.jsonl", sft_rows)
    _write_jsonl(out_dir / "traces.jsonl", traces)
    summary = {
        "example_count": len(examples),
        "sft_rows": len(sft_rows),
        "top_k": top_k,
        "avg_search_count": _average_search_count(traces),
        "artifacts": {
            "sft": str(out_dir / "sft.jsonl"),
            "traces": str(out_dir / "traces.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _build_sft_row(example: QAExample, trace: EpisodeTrace) -> dict[str, Any]:
    retrieved_doc_ids = [
        passage.doc_id
        for step in trace.steps
        if step.action_type == "search" and step.observation
        for passage in step.observation
    ]
    return {
        "id": example.id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example.question},
            {"role": "assistant", "content": format_assistant_trace(trace)},
        ],
        "metadata": {
            "search_count": trace.metadata["search_count"],
            "gold_doc_ids": example.gold_doc_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _average_search_count(traces: list[dict[str, Any]]) -> float:
    if not traces:
        return 0.0
    return sum(int(trace["metadata"].get("search_count", 0)) for trace in traces) / len(traces)
