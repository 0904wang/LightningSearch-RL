from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class _SynthesisTask:
    request_id: str
    topic: str


class DeepSeekClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for real synthesis")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 1200,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"DeepSeek request failed with HTTP {exc.code}: {error_body}") from exc

        content = response_payload["choices"][0]["message"]["content"]
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek response content was not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("DeepSeek response JSON must be an object")
        return decoded


def build_synthesis_prompt(request_id: str, topic: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You create synthetic HotpotQA-style multi-hop QA data for a retrieval "
                "agent. Return only one valid json object and no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create one synthetic multi-hop QA example as json. "
                f"Use id {request_id}. Topic: {topic}. "
                "The json object must contain: id, question, answer, context, "
                "supporting_facts. context must be a list of [title, sentences] pairs, "
                "where sentences is a list of short strings. supporting_facts must be a "
                "list of [title, sentence_index] pairs. Include at least two evidence "
                "titles, include distractor context, and make the final answer appear "
                "verbatim in one supporting evidence sentence."
            ),
        },
    ]


def validate_synthetic_row(row: dict) -> ValidationResult:
    row_id = str(row.get("id") or row.get("_id") or "").strip()
    if not row_id:
        return ValidationResult(False, "missing id")
    if not str(row.get("question", "")).strip():
        return ValidationResult(False, "missing question")
    try:
        answers = _answers(row)
    except ValueError as exc:
        return ValidationResult(False, str(exc))
    if not any(answer.strip() for answer in answers):
        return ValidationResult(False, "missing answer text")

    try:
        sentence_by_key = _context_sentence_map(row)
    except ValueError as exc:
        return ValidationResult(False, str(exc))
    if not sentence_by_key:
        return ValidationResult(False, "context has no sentences")

    try:
        supporting_pairs = _supporting_fact_pairs(row)
    except (KeyError, TypeError, ValueError) as exc:
        return ValidationResult(False, f"invalid supporting_facts: {exc}")
    if len(supporting_pairs) < 2:
        return ValidationResult(False, "supporting_facts must contain at least two facts")
    if len({title for title, _ in supporting_pairs}) < 2:
        return ValidationResult(False, "supporting_facts must cover at least two titles")

    evidence_sentences: list[str] = []
    for title, sentence_index in supporting_pairs:
        sentence = sentence_by_key.get((title, sentence_index))
        if sentence is None:
            return ValidationResult(
                False,
                f"supporting fact missing from context: {title}:{sentence_index}",
            )
        evidence_sentences.append(sentence)

    normalized_answers = [_normalize_text(answer) for answer in answers if answer.strip()]
    normalized_evidence = _normalize_text(" ".join(evidence_sentences))
    if not any(answer in normalized_evidence for answer in normalized_answers):
        return ValidationResult(False, "answer not found in supporting evidence")
    return ValidationResult(True)


def validate_synthetic_file(
    raw_path: Path,
    valid_path: Path,
    rejects_path: Path,
) -> dict[str, int]:
    raw_rows = _load_rows(raw_path)
    valid_rows: list[dict] = []
    rejected_rows: list[dict] = []
    for row in raw_rows:
        result = validate_synthetic_row(row)
        if result.valid:
            valid_rows.append(row)
        else:
            rejected_rows.append(
                {
                    "id": str(row.get("id") or row.get("_id") or ""),
                    "reason": result.reason,
                    "row": row,
                }
            )
    _write_jsonl(valid_path, valid_rows)
    _write_jsonl(rejects_path, rejected_rows)
    return {
        "raw_count": len(raw_rows),
        "valid_count": len(valid_rows),
        "reject_count": len(rejected_rows),
    }


def synthesize_file(
    out_path: Path,
    count: int,
    topics: list[str],
    client,
    concurrency: int = 50,
    seed: int = 0,
    temperature: float = 0.8,
    max_tokens: int = 1200,
    retries: int = 3,
) -> dict[str, Any]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if retries < 1:
        raise ValueError("retries must be at least 1")
    if not topics:
        raise ValueError("at least one topic is required")

    existing_ids = _existing_ids(out_path)
    tasks = [
        _SynthesisTask(request_id=f"syn-{seed + index:06d}", topic=topics[index % len(topics)])
        for index in range(count)
        if f"syn-{seed + index:06d}" not in existing_ids
    ]
    rows: list[dict] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for result in executor.map(
            lambda task: _synthesize_one(
                task,
                client,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
            ),
            tasks,
        ):
            if "row" in result:
                rows.append(result["row"])
            else:
                failures.append(result["error"])

    if rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "requested": len(tasks),
        "written": len(rows),
        "skipped_existing": count - len(tasks),
        "failed": len(failures),
        "concurrency": concurrency,
        "out": str(out_path),
        "failures": failures,
    }


def mock_synthetic_row(request_id: str, topic: str) -> dict:
    topic_slug = re.sub(r"[^A-Za-z0-9]+", " ", topic).strip() or "General Topic"
    answer = f"{topic_slug} Archive"
    first_title = f"{topic_slug} Researcher"
    second_title = f"{topic_slug} Institute"
    return {
        "id": request_id,
        "question": f"Which archive stores the institute that hosted {first_title}?",
        "answer": answer,
        "context": [
            [
                first_title,
                [f"{first_title} was hosted by {second_title} during the pilot study."],
            ],
            [
                second_title,
                [f"{second_title} keeps its experiment notes in {answer}."],
            ],
            [
                f"{topic_slug} Distractor",
                [f"{topic_slug} Distractor describes an unrelated venue."],
            ],
        ],
        "supporting_facts": [[first_title, 0], [second_title, 0]],
    }


def _synthesize_one(
    task: _SynthesisTask,
    client,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    messages = build_synthesis_prompt(task.request_id, task.topic)
    last_error = ""
    for attempt in range(retries):
        try:
            row = client.complete_json(messages, temperature=temperature, max_tokens=max_tokens)
            if not isinstance(row, dict):
                raise ValueError("client returned a non-object JSON value")
            row["id"] = task.request_id
            return {"row": row}
        except Exception as exc:  # pragma: no cover - exercised by real API failures
            last_error = _redact_secret(str(exc))
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 8))
    return {"error": {"id": task.request_id, "reason": last_error}}


def _load_rows(path: Path) -> list[dict]:
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
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("raw synthetic file must contain a JSON object, JSON array, or JSONL")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("id") or row.get("_id") or "") for row in _load_rows(path)}


def _answers(row: dict) -> list[str]:
    if "answers" in row:
        answers = row["answers"]
        if isinstance(answers, list):
            return [str(answer) for answer in answers]
        return [str(answers)]
    if "answer" in row:
        return [str(row["answer"])]
    if "final_answer" in row:
        return [str(row["final_answer"])]
    raise ValueError("missing answer, answers, or final_answer")


def _context_sentence_map(row: dict) -> dict[tuple[str, int], str]:
    context = row.get("context")
    if context is None:
        raise ValueError("missing context")
    sentence_by_key: dict[tuple[str, int], str] = {}
    for title, sentences in _context_items(context):
        for sentence_index, sentence in enumerate(sentences):
            if not str(sentence).strip():
                continue
            sentence_by_key[(title, sentence_index)] = str(sentence)
    return sentence_by_key


def _context_items(context) -> list[tuple[str, list[str]]]:
    if isinstance(context, dict):
        return [
            (str(title), _sentences_from_value(sentences))
            for title, sentences in context.items()
        ]
    if not isinstance(context, list):
        raise ValueError("context must be a mapping or list")
    items: list[tuple[str, list[str]]] = []
    for item in context:
        if isinstance(item, dict):
            title = str(item.get("title", "")).strip()
            sentences = item.get("sentences", item.get("text", []))
        else:
            try:
                title, sentences = item
            except ValueError as exc:
                raise ValueError("context list items must be dicts or [title, sentences] pairs") from exc
            title = str(title).strip()
        if not title:
            raise ValueError("context item is missing title")
        items.append((title, _sentences_from_value(sentences)))
    return items


def _sentences_from_value(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(sentence) for sentence in value]
    raise ValueError("context sentences must be a string or list")


def _supporting_fact_pairs(row: dict) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for item in row.get("supporting_facts", []):
        if isinstance(item, dict):
            title = item["title"]
            sentence_index = item.get("sent_id", item.get("sent_idx", 0))
        else:
            title, sentence_index = item
        pairs.append((str(title), int(sentence_index)))
    return pairs


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _redact_secret(message: str) -> str:
    redacted = re.sub(r"Bearer\s+[^'\"\s]+", "Bearer [REDACTED]", message)
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        redacted = redacted.replace(key, "[REDACTED]")
    return redacted
