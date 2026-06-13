from __future__ import annotations

from lightningsearch_rl.data import QAExample
from lightningsearch_rl.query import build_search_query
from lightningsearch_rl.retrieval import LexicalRetriever


def evaluate_retrieval(
    examples: list[QAExample],
    retriever: LexicalRetriever,
    top_k: int,
) -> dict[str, float]:
    if not examples:
        return {
            "example_count": 0,
            f"recall_at_{top_k}": 0.0,
            f"evidence_recall_at_{top_k}": 0.0,
            "avg_retrieved_docs": 0.0,
        }

    hit_count = 0
    evidence_recall_total = 0.0
    retrieved_total = 0
    for example in examples:
        retrieved = retriever.search(build_search_query(example.question), top_k=top_k)
        retrieved_doc_ids = {passage.doc_id for passage in retrieved}
        gold_doc_ids = set(example.gold_doc_ids)
        if gold_doc_ids & retrieved_doc_ids:
            hit_count += 1
        evidence_recall_total += (
            len(gold_doc_ids & retrieved_doc_ids) / len(gold_doc_ids)
            if gold_doc_ids
            else 0.0
        )
        retrieved_total += len(retrieved)

    count = len(examples)
    return {
        "example_count": count,
        f"recall_at_{top_k}": hit_count / count,
        f"evidence_recall_at_{top_k}": evidence_recall_total / count,
        "avg_retrieved_docs": retrieved_total / count,
    }
