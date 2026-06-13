from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    answers: list[str]
    gold_doc_ids: list[str]
    corpus: list[Passage]


def load_jsonl_examples(path: Path) -> list[QAExample]:
    examples: list[QAExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            examples.append(
                QAExample(
                    id=row["id"],
                    question=row["question"],
                    answers=list(row["answers"]),
                    gold_doc_ids=list(row.get("gold_doc_ids", [])),
                    corpus=[
                        Passage(
                            doc_id=item["doc_id"],
                            title=item.get("title", ""),
                            text=item["text"],
                        )
                        for item in row.get("corpus", [])
                    ],
                )
            )
    return examples
