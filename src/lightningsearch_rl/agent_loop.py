from __future__ import annotations

import re
from dataclasses import dataclass

from lightningsearch_rl.formatting import format_observation
from lightningsearch_rl.retrieval import LexicalRetriever


_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass(frozen=True)
class AgentAction:
    type: str
    valid: bool
    query: str | None = None
    answer: str | None = None
    reason: str | None = None


def parse_agent_action(text: str) -> AgentAction:
    if "<observation>" in text or "</observation>" in text:
        return AgentAction(type="invalid", valid=False, reason="model_generated_observation")

    search_matches = _SEARCH_RE.findall(text)
    answer_matches = _ANSWER_RE.findall(text)
    if len(search_matches) + len(answer_matches) > 1:
        if search_matches and "<answer>" in search_matches[0]:
            return AgentAction(type="invalid", valid=False, reason="nested_answer_in_search")
        return AgentAction(type="invalid", valid=False, reason="multiple_actions")

    if search_matches:
        raw_query = search_matches[0]
        if "<answer>" in raw_query or "</answer>" in raw_query:
            return AgentAction(type="invalid", valid=False, reason="nested_answer_in_search")
        query = raw_query.strip()
        if not query:
            return AgentAction(type="invalid", valid=False, reason="empty_search")
        return AgentAction(type="search", valid=True, query=query)

    if answer_matches:
        answer = answer_matches[0].strip()
        if not answer:
            return AgentAction(type="invalid", valid=False, reason="empty_answer")
        return AgentAction(type="answer", valid=True, answer=answer)

    return AgentAction(type="invalid", valid=False, reason="no_valid_action")


class SearchEnvironment:
    def __init__(self, retriever: LexicalRetriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k

    def search_observation(self, query: str) -> str:
        return format_observation(self.retriever.search(query, top_k=self.top_k))
