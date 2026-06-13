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
    _write_jsonl(
        out_dir / "transitions.jsonl",
        [transition.to_dict() for transition in transitions],
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
