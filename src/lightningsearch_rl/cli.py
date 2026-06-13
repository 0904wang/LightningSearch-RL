from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lightningsearch_rl.adapters import convert_2wiki_file, convert_hotpot_file
from lightningsearch_rl.baseline import run_retrieval_baseline
from lightningsearch_rl.corpus import load_corpus_jsonl
from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.eval import evaluate_traces
from lightningsearch_rl.grpo import export_grpo
from lightningsearch_rl.index_store import load_lexical_index, save_lexical_index
from lightningsearch_rl.rewards import compute_reward
from lightningsearch_rl.retrieval_eval import evaluate_retrieval
from lightningsearch_rl.runtime import run_rule_based_episode
from lightningsearch_rl.sft import export_sft
from lightningsearch_rl.transitions import build_transitions


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lightningsearch-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--data", required=True)
    smoke.add_argument("--out-dir", required=True)
    smoke.add_argument("--top-k", type=int, default=2)
    prepare_hotpot = subparsers.add_parser("prepare-hotpot")
    prepare_hotpot.add_argument("--raw", required=True)
    prepare_hotpot.add_argument("--corpus", required=True)
    prepare_hotpot.add_argument("--examples", required=True)
    prepare_hotpot.add_argument("--limit", type=int, default=None)
    prepare_2wiki = subparsers.add_parser("prepare-2wiki")
    prepare_2wiki.add_argument("--raw", required=True)
    prepare_2wiki.add_argument("--corpus", required=True)
    prepare_2wiki.add_argument("--examples", required=True)
    prepare_2wiki.add_argument("--limit", type=int, default=None)
    build_index = subparsers.add_parser("build-index")
    build_index.add_argument("--corpus", required=True)
    build_index.add_argument("--index", required=True)
    eval_retrieval = subparsers.add_parser("eval-retrieval")
    eval_retrieval.add_argument("--examples", required=True)
    eval_retrieval.add_argument("--index", required=True)
    eval_retrieval.add_argument("--out", required=True)
    eval_retrieval.add_argument("--top-k", type=int, default=5)
    retrieval_baseline = subparsers.add_parser("retrieval-baseline")
    retrieval_baseline.add_argument("--dataset", required=True)
    retrieval_baseline.add_argument("--examples", required=True)
    retrieval_baseline.add_argument("--index", required=True)
    retrieval_baseline.add_argument("--report", required=True)
    retrieval_baseline.add_argument("--top-k", type=int, default=5)
    export_sft_parser = subparsers.add_parser("export-sft")
    export_sft_parser.add_argument("--examples", required=True)
    export_sft_parser.add_argument("--index", required=True)
    export_sft_parser.add_argument("--out-dir", required=True)
    export_sft_parser.add_argument("--top-k", type=int, default=5)
    export_grpo_parser = subparsers.add_parser("export-grpo")
    export_grpo_parser.add_argument("--examples", required=True)
    export_grpo_parser.add_argument("--index", required=True)
    export_grpo_parser.add_argument("--out-dir", required=True)
    export_grpo_parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        return _run_smoke(Path(args.data), Path(args.out_dir), args.top_k)
    if args.command == "prepare-hotpot":
        convert_hotpot_file(
            Path(args.raw),
            Path(args.corpus),
            Path(args.examples),
            limit=args.limit,
        )
        return 0
    if args.command == "prepare-2wiki":
        convert_2wiki_file(
            Path(args.raw),
            Path(args.corpus),
            Path(args.examples),
            limit=args.limit,
        )
        return 0
    if args.command == "build-index":
        save_lexical_index(Path(args.index), load_corpus_jsonl(Path(args.corpus)))
        return 0
    if args.command == "eval-retrieval":
        metrics = evaluate_retrieval(
            load_jsonl_examples(Path(args.examples)),
            load_lexical_index(Path(args.index)),
            args.top_k,
        )
        _write_json(Path(args.out), metrics)
        return 0
    if args.command == "retrieval-baseline":
        run_retrieval_baseline(
            args.dataset,
            Path(args.examples),
            Path(args.index),
            Path(args.report),
            args.top_k,
        )
        return 0
    if args.command == "export-sft":
        export_sft(
            Path(args.examples),
            Path(args.index),
            Path(args.out_dir),
            top_k=args.top_k,
        )
        return 0
    if args.command == "export-grpo":
        export_grpo(
            Path(args.examples),
            Path(args.index),
            Path(args.out_dir),
            top_k=args.top_k,
        )
        return 0
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


def _write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
