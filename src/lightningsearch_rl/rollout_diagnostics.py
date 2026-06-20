from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from lightningsearch_rl.answer_metrics import normalize_answer, score_answer


_EXPECTED_KEYWORDS = {
    "institute": ("institute", "center", "centre", "lab", "laboratory"),
    "journal": ("journal",),
    "university": ("university", "college"),
    "award": ("award", "prize", "medal"),
}


def diagnose_rollout_answers(rollouts_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(rollouts_path)
    records = [_diagnose_row(row) for row in rows]
    non_suspicious = [record for record in records if not record["suspicious"]]
    return {
        "example_count": len(records),
        "answer_exact_match_rate": _rate(record["exact_match"] for record in records),
        "answer_token_f1": _average(record["token_f1"] for record in records),
        "answer_containment_match_rate": _rate(record["containment_match"] for record in records),
        "suspicious_count": sum(1 for record in records if record["suspicious"]),
        "suspicious_adjusted_example_count": len(non_suspicious),
        "suspicious_adjusted_exact_match_rate": _rate(record["exact_match"] for record in non_suspicious),
        "suspicious_rows": [
            record for record in records if record["suspicious"]
        ],
    }


def _diagnose_row(row: dict[str, Any]) -> dict[str, Any]:
    prediction = str(row.get("final_answer", "")).strip()
    gold = str(row.get("gold_answer", "")).strip()
    question = str(row.get("question", ""))
    observation = str(row.get("observation", ""))
    score = score_answer(prediction, gold)
    expected_type = _expected_answer_type(question)
    observation_titles = _observation_titles(observation)
    prediction_matches_title = normalize_answer(prediction) in {
        normalize_answer(title) for title in observation_titles
    }
    reasons = []
    if not score["exact_match"] and prediction_matches_title:
        reasons.append("prediction_matches_observation_title")
    if (
        not score["exact_match"]
        and expected_type in {"institute", "journal", "university"}
        and not _answer_matches_expected_type(gold, expected_type)
        and _answer_matches_expected_type(prediction, expected_type)
    ):
        reasons.append("question_gold_type_mismatch")
    return {
        "id": row.get("id"),
        "question": question,
        "prediction": prediction,
        "gold_answer": gold,
        "exact_match": score["exact_match"],
        "token_f1": score["token_f1"],
        "containment_match": score["containment_match"],
        "expected_answer_type": expected_type,
        "reasons": reasons,
        "suspicious": bool(reasons),
    }


def _expected_answer_type(question: str) -> str | None:
    normalized = normalize_answer(question)
    for expected, keywords in _EXPECTED_KEYWORDS.items():
        if any(keyword in normalized.split() for keyword in keywords):
            return expected
    return None


def _answer_matches_expected_type(answer: str, expected_type: str) -> bool:
    normalized_tokens = set(normalize_answer(answer).split())
    return any(keyword in normalized_tokens for keyword in _EXPECTED_KEYWORDS.get(expected_type, ()))


def _observation_titles(observation: str) -> list[str]:
    titles = []
    for line in observation.splitlines():
        match = re.match(r"\[\d+\]\s+([^:]+):", line.strip())
        if match:
            titles.append(match.group(1).strip())
    return titles


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _rate(values: Any) -> float:
    items = list(values)
    return round(sum(1 for value in items if value) / len(items), 6) if items else 0.0


def _average(values: Any) -> float:
    items = list(values)
    return round(sum(float(value) for value in items) / len(items), 6) if items else 0.0
