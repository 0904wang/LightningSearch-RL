from __future__ import annotations

from dataclasses import asdict, dataclass

from lightningsearch_rl.data import Passage, QAExample
from lightningsearch_rl.query import build_search_query
from lightningsearch_rl.retrieval import LexicalRetriever


@dataclass(frozen=True)
class TraceStep:
    state: str
    action: str
    action_type: str
    query: str | None = None
    observation: list[Passage] | None = None
    valid_tool_call: bool = True
    terminal: bool = False


@dataclass(frozen=True)
class EpisodeTrace:
    question_id: str
    question: str
    steps: list[TraceStep]
    final_answer: str
    reward: float | None
    metadata: dict[str, int | str | float]

    def to_dict(self) -> dict:
        return asdict(self)


def run_rule_based_episode(example: QAExample, top_k: int = 5) -> EpisodeTrace:
    retriever = LexicalRetriever(example.corpus)
    return run_retrieval_episode(example, retriever, top_k=top_k)


def run_retrieval_episode(example: QAExample, retriever: LexicalRetriever, top_k: int = 5) -> EpisodeTrace:
    query = build_search_query(example.question)
    observation = retriever.search(query, top_k=top_k)
    final_answer = _answer_from_observation(example, observation)
    steps = [
        TraceStep(
            state=example.question,
            action=f"<search>{query}</search>",
            action_type="search",
            query=query,
            observation=observation,
            valid_tool_call=bool(query.strip()),
            terminal=False,
        ),
        TraceStep(
            state=f"{example.question}\nObservation: {observation}",
            action=f"<answer>{final_answer}</answer>",
            action_type="answer",
            terminal=True,
        ),
    ]
    return EpisodeTrace(
        question_id=example.id,
        question=example.question,
        steps=steps,
        final_answer=final_answer,
        reward=None,
        metadata={"search_count": 1},
    )


def _answer_from_observation(example: QAExample, observation: list[Passage]) -> str:
    observed_text = " ".join(p.text for p in observation).lower()
    for answer in example.answers:
        if answer.lower() in observed_text:
            return answer
    return ""

