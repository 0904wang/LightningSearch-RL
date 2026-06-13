from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.formatting import format_assistant_trace
from lightningsearch_rl.runtime import run_rule_based_episode


def test_format_assistant_trace_contains_search_observation_and_answer():
    example = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))[0]
    trace = run_rule_based_episode(example, top_k=2)

    content = format_assistant_trace(trace)

    assert "<search>" in content
    assert "<observation>" in content
    assert "<answer>Example City</answer>" in content
