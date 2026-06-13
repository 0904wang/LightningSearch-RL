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
