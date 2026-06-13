from __future__ import annotations

from dataclasses import dataclass
import re

from lightningsearch_rl.data import QAExample
from lightningsearch_rl.runtime import EpisodeTrace


@dataclass(frozen=True)
class RewardBreakdown:
    answer_reward: float
    evidence_reward: float
    format_reward: float
    tool_validity_reward: float
    search_count: int
    total: float

    def to_dict(self) -> dict:
        return {
            "answer_reward": self.answer_reward,
            "evidence_reward": self.evidence_reward,
            "format_reward": self.format_reward,
            "tool_validity_reward": self.tool_validity_reward,
            "search_count": self.search_count,
            "total": self.total,
        }


def compute_reward(example: QAExample, trace: EpisodeTrace) -> RewardBreakdown:
    normalized_answers = {_normalize(answer) for answer in example.answers}
    answer_reward = 1.0 if _normalize(trace.final_answer) in normalized_answers else 0.0
    retrieved_doc_ids = {
        passage.doc_id
        for step in trace.steps
        for passage in (step.observation or [])
    }
    evidence_reward = (
        len(set(example.gold_doc_ids) & retrieved_doc_ids) / len(example.gold_doc_ids)
        if example.gold_doc_ids
        else 0.0
    )
    format_reward = (
        1.0
        if trace.steps and trace.steps[-1].terminal and trace.steps[-1].action_type == "answer"
        else 0.0
    )
    search_queries = [step.query for step in trace.steps if step.action_type == "search"]
    non_empty = all(query and query.strip() for query in search_queries)
    lowered_queries = [query.lower() for query in search_queries if query]
    no_duplicates = len(lowered_queries) == len(set(lowered_queries))
    tool_validity_reward = 1.0 if non_empty and no_duplicates else 0.0
    search_count = len(search_queries)
    total = (
        answer_reward
        + 0.2 * evidence_reward
        + 0.1 * format_reward
        + 0.1 * tool_validity_reward
        - 0.03 * search_count
    )
    return RewardBreakdown(
        answer_reward=answer_reward,
        evidence_reward=evidence_reward,
        format_reward=format_reward,
        tool_validity_reward=tool_validity_reward,
        search_count=search_count,
        total=round(total, 6),
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
