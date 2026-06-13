from pathlib import Path

from lightningsearch_rl.data import load_jsonl_examples
from lightningsearch_rl.rewards import compute_reward
from lightningsearch_rl.runtime import run_rule_based_episode


def test_compute_reward_combines_answer_evidence_format_tool_and_cost():
    example = load_jsonl_examples(Path("tests/fixtures/tiny_multihop.jsonl"))[0]
    trace = run_rule_based_episode(example, top_k=2)

    reward = compute_reward(example, trace)

    assert reward.answer_reward == 1.0
    assert reward.evidence_reward == 1.0
    assert reward.format_reward == 1.0
    assert reward.tool_validity_reward == 1.0
    assert reward.search_count == 1
    assert reward.total == 1.37
