from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from lightningsearch_rl.data import Passage


def passage_to_dict(passage: Passage) -> dict[str, str]:
    return {"doc_id": passage.doc_id, "title": passage.title, "text": passage.text}


def passage_from_dict(row: dict) -> Passage:
    return Passage(doc_id=row["doc_id"], title=row.get("title", ""), text=row["text"])


def deduplicate_passages(passages: Iterable[Passage]) -> list[Passage]:
    seen: set[str] = set()
    deduped: list[Passage] = []
    for passage in passages:
        if passage.doc_id in seen:
            continue
        seen.add(passage.doc_id)
        deduped.append(passage)
    return deduped


def write_corpus_jsonl(path: Path, passages: Iterable[Passage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for passage in deduplicate_passages(passages):
            handle.write(json.dumps(passage_to_dict(passage), ensure_ascii=False) + "\n")


def load_corpus_jsonl(path: Path) -> list[Passage]:
    passages: list[Passage] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            passages.append(passage_from_dict(json.loads(line)))
    return deduplicate_passages(passages)
