from __future__ import annotations

from lightningsearch_rl.data import QAExample
from lightningsearch_rl.rewards import compute_reward
from lightningsearch_rl.runtime import EpisodeTrace


def evaluate_traces(examples: list[QAExample], traces: list[EpisodeTrace]) -> dict[str, float]:
    if len(examples) != len(traces):
        raise ValueError("examples and traces must have the same length")
    if not examples:
        return {
            "answer_em": 0.0,
            "evidence_recall": 0.0,
            "avg_search_calls": 0.0,
            "avg_reward": 0.0,
        }
    rewards = [compute_reward(example, trace) for example, trace in zip(examples, traces)]
    count = len(rewards)
    return {
        "answer_em": sum(r.answer_reward for r in rewards) / count,
        "evidence_recall": sum(r.evidence_reward for r in rewards) / count,
        "avg_search_calls": sum(r.search_count for r in rewards) / count,
        "avg_reward": sum(r.total for r in rewards) / count,
    }
