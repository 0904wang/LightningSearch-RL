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
