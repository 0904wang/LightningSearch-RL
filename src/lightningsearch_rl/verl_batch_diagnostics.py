from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any


LOW_REWARD_THRESHOLD = 0.2


def diagnose_verl_training_batches(
    train_jsonl_path: Path,
    *,
    metrics_summary_path: Path | None = None,
    train_batch_size: int,
    low_reward_threshold: float = LOW_REWARD_THRESHOLD,
) -> dict[str, Any]:
    if train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    rows = _load_jsonl(train_jsonl_path)
    batches = [
        _summarize_batch(
            index,
            rows[index * train_batch_size : (index + 1) * train_batch_size],
            low_reward_threshold,
            start_index=index * train_batch_size,
        )
        for index in range(ceil(len(rows) / train_batch_size))
    ]
    metrics_summary = _load_json(metrics_summary_path) if metrics_summary_path else {}
    return {
        "train_jsonl": str(train_jsonl_path),
        "metrics_summary": str(metrics_summary_path) if metrics_summary_path else None,
        "alignment_assumption": "contiguous train_jsonl order; actual verl dataloader shuffling may differ",
        "train_rows": len(rows),
        "train_batch_size": train_batch_size,
        "batch_count": len(batches),
        "overall": _summarize_rows(rows, low_reward_threshold),
        "batches": batches,
        "step_alignment": _align_steps(batches, metrics_summary),
    }


def write_verl_batch_diagnostics(
    train_jsonl_path: Path,
    out_path: Path,
    *,
    metrics_summary_path: Path | None = None,
    train_batch_size: int,
    low_reward_threshold: float = LOW_REWARD_THRESHOLD,
) -> dict[str, Any]:
    report = diagnose_verl_training_batches(
        train_jsonl_path,
        metrics_summary_path=metrics_summary_path,
        train_batch_size=train_batch_size,
        low_reward_threshold=low_reward_threshold,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _summarize_batch(
    index: int,
    rows: list[dict[str, Any]],
    low_reward_threshold: float,
    *,
    start_index: int,
) -> dict[str, Any]:
    summary = _summarize_rows(rows, low_reward_threshold, start_index=start_index)
    summary.update(
        {
            "batch_index": index,
            "row_start": summary["row_start"] if rows else index,
            "row_end_exclusive": summary["row_end_exclusive"] if rows else index,
        }
    )
    return summary


def _summarize_rows(rows: list[dict[str, Any]], low_reward_threshold: float, *, start_index: int = 0) -> dict[str, Any]:
    indexed = list(enumerate(rows, start=start_index))
    rewards = [_reward_model_reward(row) for row in rows]
    precomputed_step_rewards = [_extra_float(row, "precomputed_step_reward") for row in rows]
    precomputed_total_rewards = [_extra_float(row, "precomputed_total_reward") for row in rows]
    low_reward_rows = [
        _row_brief(index, row)
        for index, row in indexed
        if _reward_model_reward(row) < low_reward_threshold
    ]
    return {
        "row_count": len(rows),
        "row_start": indexed[0][0] if indexed else 0,
        "row_end_exclusive": indexed[-1][0] + 1 if indexed else 0,
        "stage_counts": _stage_counts(rows),
        "reward_model_reward": _numeric_summary(rewards),
        "precomputed_step_reward": _numeric_summary(precomputed_step_rewards),
        "precomputed_total_reward": _numeric_summary(precomputed_total_rewards),
        "low_reward_row_count": len(low_reward_rows),
        "low_reward_rows": low_reward_rows[:10],
    }


def _align_steps(batches: list[dict[str, Any]], metrics_summary: dict[str, Any]) -> list[dict[str, Any]]:
    steps = metrics_summary.get("train_steps") or {
        step_id: metrics
        for step_id, metrics in metrics_summary.get("steps", {}).items()
        if isinstance(metrics, dict) and "training/global_step" in metrics
    }
    aligned = []
    for step_id in sorted(steps, key=lambda value: int(value)):
        if not batches:
            break
        step = int(step_id)
        batch = batches[(step - 1) % len(batches)]
        metrics = steps[step_id]
        aligned.append(
            {
                "step": step,
                "batch_index": batch["batch_index"],
                "row_start": batch["row_start"],
                "row_end_exclusive": batch["row_end_exclusive"],
                "logged_reward_mean": metrics.get("critic/rewards/mean"),
                "precomputed_reward_mean": batch["reward_model_reward"]["mean"],
                "stage_counts": batch["stage_counts"],
                "low_reward_row_count": batch["low_reward_row_count"],
            }
        )
    return aligned


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(_extra(row).get("reward_stage") or "unknown")
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(sum(values) / len(values), 6),
    }


def _row_brief(index: int, row: dict[str, Any]) -> dict[str, Any]:
    extra = _extra(row)
    expected_action = str(extra.get("expected_action", ""))
    return {
        "index": index,
        "id": extra.get("id"),
        "source_id": extra.get("source_id"),
        "reward_stage": extra.get("reward_stage"),
        "reward_model_reward": _reward_model_reward(row),
        "precomputed_total_reward": _extra_float(row, "precomputed_total_reward"),
        "expected_action": expected_action[:160],
    }


def _reward_model_reward(row: dict[str, Any]) -> float:
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        return 0.0
    return _float(reward_model.get("reward"))


def _extra_float(row: dict[str, Any], key: str) -> float:
    return _float(_extra(row).get(key))


def _extra(row: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra_info")
    return extra if isinstance(extra, dict) else {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
