from __future__ import annotations

import json
from pathlib import Path

from lightningsearch_rl.corpus import passage_from_dict, passage_to_dict
from lightningsearch_rl.data import Passage
from lightningsearch_rl.retrieval import LexicalRetriever


INDEX_VERSION = 1


def save_lexical_index(path: Path, passages: list[Passage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": INDEX_VERSION,
        "passages": [passage_to_dict(passage) for passage in passages],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_lexical_index(path: Path) -> LexicalRetriever:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != INDEX_VERSION:
        raise ValueError(f"unsupported index version: {payload.get('version')}")
    return LexicalRetriever([passage_from_dict(row) for row in payload["passages"]])
