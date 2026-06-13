from __future__ import annotations

import json
from pathlib import Path

from lightningsearch_rl.corpus import write_corpus_jsonl
from lightningsearch_rl.data import Passage


def convert_hotpot_file(
    raw_path: Path,
    corpus_path: Path,
    examples_path: Path,
    limit: int | None = None,
) -> None:
    rows = _limit_rows(_load_raw_rows(raw_path), limit)
    passages: list[Passage] = []
    examples: list[dict] = []
    for row in rows:
        row_passages = _passages_from_context("hotpot", row["context"])
        passages.extend(row_passages)
        corpus_doc_ids = [passage.doc_id for passage in row_passages]
        gold_doc_ids = [
            _doc_id("hotpot", title, sentence_index)
            for title, sentence_index in _supporting_fact_pairs(row)
        ]
        examples.append(
            {
                "id": _row_id(row),
                "question": row["question"],
                "answers": _answers(row),
                "gold_doc_ids": gold_doc_ids,
                "corpus_doc_ids": corpus_doc_ids,
            }
        )
    write_corpus_jsonl(corpus_path, passages)
    _write_jsonl(examples_path, examples)


def convert_2wiki_file(
    raw_path: Path,
    corpus_path: Path,
    examples_path: Path,
    limit: int | None = None,
) -> None:
    rows = _limit_rows(_load_raw_rows(raw_path), limit)
    passages: list[Passage] = []
    examples: list[dict] = []
    for row in rows:
        row_passages = _passages_from_context("2wiki", row["context"])
        passages.extend(row_passages)
        corpus_doc_ids = [passage.doc_id for passage in row_passages]
        gold_doc_ids = [
            _doc_id("2wiki", title, sentence_index)
            for title, sentence_index in _supporting_fact_pairs(row)
        ]
        examples.append(
            {
                "id": _row_id(row),
                "question": row["question"],
                "answers": _answers(row),
                "gold_doc_ids": gold_doc_ids,
                "corpus_doc_ids": corpus_doc_ids,
            }
        )
    write_corpus_jsonl(corpus_path, passages)
    _write_jsonl(examples_path, examples)


def _load_raw_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError("raw dataset file must contain a JSON array or JSONL rows")


def _limit_rows(rows: list[dict], limit: int | None) -> list[dict]:
    return rows[:limit] if limit is not None else rows


def _row_id(row: dict) -> str:
    return row.get("_id") or row["id"]


def _answers(row: dict) -> list[str]:
    if "answers" in row:
        answers = row["answers"]
        return answers if isinstance(answers, list) else [answers]
    if "answer" in row:
        return [row["answer"]]
    if "final_answer" in row:
        return [row["final_answer"]]
    raise ValueError("row is missing answer, answers, or final_answer")


def _supporting_fact_pairs(row: dict) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for item in row.get("supporting_facts", []):
        if isinstance(item, dict):
            pairs.append((item["title"], int(item.get("sent_id", item.get("sent_idx", 0)))))
        else:
            title, sentence_index = item
            pairs.append((title, int(sentence_index)))
    return pairs


def _passages_from_context(prefix: str, context) -> list[Passage]:
    if isinstance(context, dict):
        return _passages_from_mapping_context(prefix, context)
    if isinstance(context, list):
        return _passages_from_list_context(prefix, context)
    raise ValueError("context must be a mapping or list")


def _passages_from_list_context(prefix: str, context: list) -> list[Passage]:
    passages: list[Passage] = []
    for item in context:
        if isinstance(item, dict):
            title = item["title"]
            sentences = item.get("sentences") or item.get("text") or []
            if isinstance(sentences, str):
                sentences = [sentences]
        else:
            title, sentences = item
        for sentence_index, sentence in enumerate(sentences):
            passages.append(
                Passage(
                    doc_id=_doc_id(prefix, title, sentence_index),
                    title=title,
                    text=sentence,
                )
            )
    return passages

def _passages_from_mapping_context(prefix: str, context: dict[str, list[str]]) -> list[Passage]:
    passages: list[Passage] = []
    for title, sentences in context.items():
        if isinstance(sentences, str):
            sentences = [sentences]
        for sentence_index, sentence in enumerate(sentences):
            passages.append(
                Passage(
                    doc_id=_doc_id(prefix, title, sentence_index),
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
