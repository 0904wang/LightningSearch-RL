from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_STAGES = ("search", "answer")


def filter_transitions_by_reward_variance(
    *,
    transitions_path: Path,
    reward_dump_path: Path,
    out_dir: Path,
    stages: Sequence[str] = DEFAULT_STAGES,
    min_score_range: float = 1e-9,
    min_samples: int = 2,
    max_source_count: int | None = None,
) -> dict[str, Any]:
    reward_rows = _load_jsonl(reward_dump_path)
    selected_stages = tuple(stages) if stages else DEFAULT_STAGES
    variable_groups = _variable_groups(
        reward_rows,
        stages=selected_stages,
        min_score_range=min_score_range,
        min_samples=min_samples,
    )
    ranked_source_ids = _ranked_source_ids(variable_groups)
    if max_source_count is not None:
        ranked_source_ids = ranked_source_ids[:max_source_count]
    selected_source_ids = sorted(ranked_source_ids)
    selected_source_set = set(selected_source_ids)

    transition_rows = _load_jsonl(transitions_path)
    filtered_rows = [
        row for row in transition_rows if _source_id_from_transition(row) in selected_source_set
    ]
    matched_source_ids = sorted({_source_id_from_transition(row) for row in filtered_rows})
    unmatched_source_ids = sorted(selected_source_set - set(matched_source_ids))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "transitions.jsonl", filtered_rows)
    _write_json(out_dir / "selected_source_ids.json", selected_source_ids)
    _write_json(out_dir / "variance_groups.json", variable_groups)

    summary = {
        "transitions_path": str(transitions_path),
        "reward_dump_path": str(reward_dump_path),
        "out_dir": str(out_dir),
        "stages": list(selected_stages),
        "min_score_range": min_score_range,
        "min_samples": min_samples,
        "max_source_count": max_source_count,
        "input_transition_count": len(transition_rows),
        "output_transition_count": len(filtered_rows),
        "selected_source_count": len(selected_source_ids),
        "matched_source_count": len(matched_source_ids),
        "unmatched_source_count": len(unmatched_source_ids),
        "selected_source_ids": selected_source_ids,
        "unmatched_source_ids": unmatched_source_ids,
        "stage_variable_group_counts": _stage_counts(variable_groups),
        "top_variable_groups": variable_groups[:20],
        "artifacts": {
            "transitions": str(out_dir / "transitions.jsonl"),
            "selected_source_ids": str(out_dir / "selected_source_ids.json"),
            "variance_groups": str(out_dir / "variance_groups.json"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _variable_groups(
    rows: list[dict[str, Any]],
    *,
    stages: Sequence[str],
    min_score_range: float,
    min_samples: int,
) -> list[dict[str, Any]]:
    stage_set = {str(stage) for stage in stages}
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        stage = str(row.get("reward_stage") or "unknown")
        if stage not in stage_set:
            continue
        extra = _dict(row.get("extra_info"))
        source_id = str(extra.get("source_id") or extra.get("id") or "").strip()
        if not source_id:
            continue
        grouped.setdefault((stage, source_id), []).append(_float(row.get("score")))

    variable_groups = []
    for (stage, source_id), scores in grouped.items():
        if len(scores) < min_samples:
            continue
        score_min = min(scores)
        score_max = max(scores)
        score_range = score_max - score_min
        if score_range <= min_score_range:
            continue
        variable_groups.append(
            {
                "stage": stage,
                "source_id": source_id,
                "sample_count": len(scores),
                "score_min": round(score_min, 6),
                "score_max": round(score_max, 6),
                "score_range": round(score_range, 6),
            }
        )
    variable_groups.sort(
        key=lambda row: (
            -float(row["score_range"]),
            str(row["stage"]),
            str(row["source_id"]),
        )
    )
    return variable_groups


def _ranked_source_ids(variable_groups: list[dict[str, Any]]) -> list[str]:
    ranked = []
    seen = set()
    for group in variable_groups:
        source_id = str(group["source_id"])
        if source_id in seen:
            continue
        seen.add(source_id)
        ranked.append(source_id)
    return ranked


def _stage_counts(variable_groups: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in variable_groups:
        stage = str(group["stage"])
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _source_id_from_transition(row: dict[str, Any]) -> str:
    source_id = row.get("source_id") or row.get("id")
    if source_id:
        return str(source_id)
    transition_id = str(row.get("transition_id", ""))
    return transition_id.split(":", 1)[0] if transition_id else ""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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
