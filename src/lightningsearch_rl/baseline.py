from __future__ import annotations

import json
from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.index_store import load_lexical_index
from lightningsearch_rl.retrieval_eval import evaluate_retrieval


def run_retrieval_baseline(
    dataset: str,
    examples_path: Path,
    index_path: Path,
    report_path: Path,
    top_k: int,
) -> dict:
    examples = load_jsonl_examples(examples_path)
    metrics = evaluate_retrieval(examples, load_lexical_index(index_path), top_k=top_k)
    report = {
        "dataset": dataset,
        "top_k": top_k,
        "example_count": len(examples),
        "metrics": metrics,
        "artifacts": {
            "examples": str(examples_path),
            "index": str(index_path),
            "report": str(report_path),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
