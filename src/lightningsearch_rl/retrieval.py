from __future__ import annotations

import math
import re

from lightningsearch_rl.data import Passage


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class LexicalRetriever:
    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self._tokens = [tokenize(f"{p.title} {p.text}") for p in passages]
        self._doc_freq: dict[str, int] = {}
        for tokens in self._tokens:
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

    def search(self, query: str, top_k: int = 5) -> list[Passage]:
        query_tokens = tokenize(query)
        scored = [
            (self._score(query_tokens, doc_tokens), index, passage)
            for index, (doc_tokens, passage) in enumerate(zip(self._tokens, self.passages))
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [passage for score, _, passage in scored[:top_k] if score > 0]

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        doc_len = len(doc_tokens)
        counts: dict[str, int] = {}
        for token in doc_tokens:
            counts[token] = counts.get(token, 0) + 1
        score = 0.0
        num_docs = max(len(self.passages), 1)
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(token, 0)
            idf = math.log((num_docs + 1) / (df + 1)) + 1.0
            score += idf * tf / doc_len
        return score
