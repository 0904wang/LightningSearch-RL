from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


COMPONENT_KEYS = (
    "score",
    "answer_reward",
    "search_reward",
    "evidence_rank_reward",
    "format_reward",
    "search_cost",
)


def diagnose_reward_dump(dump_path: Path, *, low_score_threshold: float = 0.5) -> dict[str, Any]:
    rows = _load_jsonl(dump_path)
    by_stage = {
        stage: _summarize_rows(stage_rows, low_score_threshold)
        for stage, stage_rows in _group_by_stage(rows).items()
    }
    return {
        "dump_path": str(dump_path),
        "row_count": len(rows),
        "stage_counts": {stage: summary["row_count"] for stage, summary in by_stage.items()},
        "overall": _summarize_rows(rows, low_score_threshold),
        "by_stage": by_stage,
        "low_score_examples": _low_score_examples(rows, low_score_threshold),
    }


def write_reward_dump_diagnostics(
    dump_path: Path,
    out_path: Path,
    *,
    low_score_threshold: float = 0.5,
) -> dict[str, Any]:
    report = diagnose_reward_dump(dump_path, low_score_threshold=low_score_threshold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _summarize_rows(rows: list[dict[str, Any]], low_score_threshold: float) -> dict[str, Any]:
    parsed_actions = [_dict(row.get("parsed_action")) for row in rows]
    summary = {
        "row_count": len(rows),
        "invalid_action_count": sum(1 for action in parsed_actions if not action.get("valid")),
        "low_score_count": sum(1 for row in rows if _float(row.get("score")) < low_score_threshold),
        "answer_reward_type_counts": _answer_reward_type_counts(rows),
        "group_score_variance": _group_score_variance(rows),
    }
    for key in COMPONENT_KEYS:
        summary[key] = _numeric_summary([_float(row.get(key)) for row in rows])
    return summary


def _group_by_stage(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stage = str(row.get("reward_stage") or "unknown")
        grouped.setdefault(stage, []).append(row)
    return dict(sorted(grouped.items()))


def _group_score_variance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        extra = _dict(row.get("extra_info"))
        source_id = str(extra.get("source_id") or extra.get("id") or "").strip()
        if not source_id:
            continue
        groups.setdefault(source_id, []).append(_float(row.get("score")))
    multi_sample_groups = {source_id: scores for source_id, scores in groups.items() if len(scores) > 1}
    ranges = [
        {
            "source_id": source_id,
            "count": len(scores),
            "score_min": round(min(scores), 6),
            "score_max": round(max(scores), 6),
            "score_range": round(max(scores) - min(scores), 6),
        }
        for source_id, scores in multi_sample_groups.items()
    ]
    variable_groups = [row for row in ranges if row["score_range"] > 1e-9]
    variable_groups.sort(key=lambda row: (-float(row["score_range"]), str(row["source_id"])))
    return {
        "group_count": len(multi_sample_groups),
        "variable_group_count": len(variable_groups),
        "variable_group_rate": round(len(variable_groups) / len(multi_sample_groups), 6)
        if multi_sample_groups
        else 0.0,
        "avg_score_range": round(sum(row["score_range"] for row in ranges) / len(ranges), 6)
        if ranges
        else 0.0,
        "top_variable_groups": variable_groups[:20],
    }


def _answer_reward_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reward_type = row.get("answer_reward_type")
        if reward_type is None:
            continue
        key = str(reward_type)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _low_score_examples(rows: list[dict[str, Any]], low_score_threshold: float) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        score = _float(row.get("score"))
        if score >= low_score_threshold:
            continue
        extra = _dict(row.get("extra_info"))
        action = _dict(row.get("parsed_action"))
        examples.append(
            {
                "id": extra.get("id"),
                "source_id": extra.get("source_id"),
                "reward_stage": row.get("reward_stage"),
                "score": score,
                "answer_reward": _float(row.get("answer_reward")),
                "search_reward": _float(row.get("search_reward")),
                "format_reward": _float(row.get("format_reward")),
                "search_cost": _float(row.get("search_cost")),
                "parsed_action": action,
                "solution_preview": str(row.get("solution_preview", ""))[:240],
            }
        )
    return examples[:20]


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
