from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import mean


def diagnose_dataset(valid_path: Path, grpo_dir: Path | None = None) -> dict:
    rows = _load_jsonl(valid_path)
    answer_type_counts = Counter()
    answer_lengths = []
    question_counts = Counter()
    answer_counts = Counter()
    context_title_counts = Counter()
    answer_equals_context_title = 0
    answer_in_question = 0
    answer_support_sentence_hits = Counter()
    supporting_title_pair_counts = Counter()

    for row in rows:
        answer = str(row.get("answer", row.get("final_answer", ""))).strip()
        normalized_answer = _normalize(answer)
        answer_lengths.append(len(answer.split()))
        question = str(row.get("question", ""))
        question_counts[_normalize(question)] += 1
        answer_counts[normalized_answer] += 1
        chain_schema = row.get("chain_schema") if isinstance(row.get("chain_schema"), dict) else {}
        answer_type = str(chain_schema.get("answer_type", "<missing>")).strip() or "<missing>"
        answer_type_counts[answer_type] += 1

        title_to_sentences = {}
        titles = []
        for item in row.get("context", []):
            if isinstance(item, list) and len(item) >= 2:
                title = str(item[0])
                titles.append(_normalize(title))
                title_to_sentences[title] = [str(sentence) for sentence in item[1]]
        context_title_counts.update(titles)
        if normalized_answer in set(titles):
            answer_equals_context_title += 1
        if normalized_answer and normalized_answer in _normalize(question):
            answer_in_question += 1

        hits = 0
        supporting_titles = []
        for title, index in row.get("supporting_facts", []):
            title = str(title)
            supporting_titles.append(title)
            sentences = title_to_sentences.get(title, [])
            try:
                sentence = sentences[int(index)]
            except (IndexError, TypeError, ValueError):
                continue
            if normalized_answer and normalized_answer in _normalize(sentence):
                hits += 1
        answer_support_sentence_hits[str(hits)] += 1
        if len(supporting_titles) >= 2:
            supporting_title_pair_counts[tuple(supporting_titles[:2])] += 1

    reward = _load_reward_summary(grpo_dir) if grpo_dir else _empty_reward_summary()
    grpo_summary = _load_json(grpo_dir / "summary.json") if grpo_dir and (grpo_dir / "summary.json").exists() else {}
    return {
        "row_count": len(rows),
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "answer_length": _numeric_summary(answer_lengths),
        "quality": {
            "answer_equals_context_title": answer_equals_context_title,
            "answer_in_question": answer_in_question,
            "answer_support_sentence_hits": dict(sorted(answer_support_sentence_hits.items())),
        },
        "duplicates": {
            "duplicate_questions": sum(count - 1 for count in question_counts.values() if count > 1),
            "duplicate_answers": sum(count - 1 for count in answer_counts.values() if count > 1),
            "duplicate_context_titles": sum(count - 1 for count in context_title_counts.values() if count > 1),
            "duplicate_supporting_title_pairs": sum(
                count - 1 for count in supporting_title_pair_counts.values() if count > 1
            ),
        },
        "reward": reward,
        "grpo_summary": grpo_summary,
    }


def _load_reward_summary(grpo_dir: Path) -> dict:
    path = grpo_dir / "reward_records.jsonl"
    if not path.exists():
        return _empty_reward_summary()
    rewards = []
    for row in _load_jsonl(path):
        reward = row.get("reward", {})
        if isinstance(reward, dict) and "total" in reward:
            rewards.append(float(reward["total"]))
        elif "total_reward" in row:
            rewards.append(float(row["total_reward"]))
    summary = _numeric_summary(rewards)
    summary["count"] = len(rewards)
    return summary


def _empty_reward_summary() -> dict:
    return {"count": 0, "min": None, "max": None, "avg": None}


def _numeric_summary(values: list[float | int]) -> dict:
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {
        "min": min(values),
        "max": max(values),
        "avg": round(mean(values), 6),
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
