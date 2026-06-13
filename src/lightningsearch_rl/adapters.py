from __future__ import annotations

import json
from pathlib import Path

from lightningsearch_rl.corpus import write_corpus_jsonl
from lightningsearch_rl.data import Passage


def convert_hotpot_file(raw_path: Path, corpus_path: Path, examples_path: Path) -> None:
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    passages: list[Passage] = []
    examples: list[dict] = []
    for row in rows:
        row_passages = _passages_from_hotpot_context(row["context"])
        passages.extend(row_passages)
        corpus_doc_ids = [passage.doc_id for passage in row_passages]
        gold_doc_ids = [
            _doc_id("hotpot", title, sentence_index)
            for title, sentence_index in row.get("supporting_facts", [])
        ]
        examples.append(
            {
                "id": row["_id"],
                "question": row["question"],
                "answers": [row["answer"]],
                "gold_doc_ids": gold_doc_ids,
                "corpus_doc_ids": corpus_doc_ids,
            }
        )
    write_corpus_jsonl(corpus_path, passages)
    _write_jsonl(examples_path, examples)


def _passages_from_hotpot_context(context: list) -> list[Passage]:
    passages: list[Passage] = []
    for title, sentences in context:
        for sentence_index, sentence in enumerate(sentences):
            passages.append(
                Passage(
                    doc_id=_doc_id("hotpot", title, sentence_index),
                    title=title,
                    text=sentence,
                )
            )
    return passages


def _doc_id(prefix: str, title: str, sentence_index: int) -> str:
    return f"{prefix}::{title}::{sentence_index}"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
