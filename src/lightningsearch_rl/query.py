from __future__ import annotations


def build_search_query(question: str) -> str:
    if "birthplace" in question.lower():
        return f"{question} born city"
    return question
