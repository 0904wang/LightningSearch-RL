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
