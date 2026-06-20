from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")
FATAL_MARKERS = {
    "called_process_error": ("CalledProcessError",),
    "error_executing_job": ("Error executing job",),
    "cuda_oom": ("CUDA out of memory", "OutOfMemory"),
    "kv_cache_or_model_len": ("ValueError: To serve", "available KV cache memory"),
}
SHUTDOWN_WARNING_MARKERS = (
    "DataLoader worker",
    "Engine core proc",
    "resource_tracker",
    "KeyError: '/psm_",
)
REWARD_DROP_THRESHOLD = 0.25


def parse_verl_log(log_path: Path) -> dict[str, Any]:
    lines = [_clean_line(line) for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    steps: dict[str, dict[str, float]] = {}
    fatal_markers: dict[str, int] = {}
    shutdown_warning_count = 0
    shutdown_warning_examples: list[str] = []
    started_at = None
    finished_at = None
    training_progress_100_seen = False

    for line in lines:
        if line.startswith("started_at="):
            started_at = line.split("=", 1)[1].strip()
        if line.startswith("finished_at="):
            finished_at = line.split("=", 1)[1].strip()
        if "Training Progress: 100%" in line:
            training_progress_100_seen = True
        for marker_name, patterns in FATAL_MARKERS.items():
            if any(pattern in line for pattern in patterns):
                fatal_markers[marker_name] = fatal_markers.get(marker_name, 0) + 1
        if any(pattern in line for pattern in SHUTDOWN_WARNING_MARKERS):
            shutdown_warning_count += 1
            if len(shutdown_warning_examples) < 5:
                shutdown_warning_examples.append(line)
        match = STEP_RE.search(line)
        if match:
            step_id = match.group(1)
            steps[step_id] = _parse_metric_tail(match.group(2))

    validation_steps = _filter_validation_steps(steps)
    train_steps = _filter_train_steps(steps)
    numeric_step_ids = sorted(int(step_id) for step_id in train_steps or steps)
    final_step = numeric_step_ids[-1] if numeric_step_ids else None
    latest_train_metrics = train_steps.get(str(final_step), {}) if final_step is not None else {}
    initial_validation_metrics = _initial_validation_metrics(validation_steps)
    reward_curve = _build_reward_curve(train_steps)
    fatal_marker_count = sum(fatal_markers.values())
    return {
        "log_path": str(log_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "completed": bool(finished_at and training_progress_100_seen and fatal_marker_count == 0),
        "training_progress_100_seen": training_progress_100_seen,
        "fatal_marker_count": fatal_marker_count,
        "fatal_markers": fatal_markers,
        "shutdown_warning_count": shutdown_warning_count,
        "shutdown_warning_examples": shutdown_warning_examples,
        "final_step": final_step,
        "steps": steps,
        "validation_steps": validation_steps,
        "initial_validation_metrics": initial_validation_metrics,
        "train_steps": train_steps,
        "reward_curve": reward_curve,
        "reward_drop_alerts": _build_reward_drop_alerts(reward_curve),
        "latest_train_metrics": latest_train_metrics,
    }


def write_verl_log_summary(log_path: Path, out_path: Path) -> dict[str, Any]:
    summary = parse_verl_log(log_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _clean_line(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def _parse_metric_tail(value: str) -> dict[str, float]:
    metrics = {}
    for part in value.split(" - "):
        if ":" not in part:
            continue
        key, raw_value = part.rsplit(":", 1)
        key = key.strip()
        raw_value = raw_value.strip().rstrip(",")
        try:
            metrics[key] = float(raw_value)
        except ValueError:
            continue
    return metrics


def _filter_validation_steps(steps: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        step_id: metrics
        for step_id, metrics in steps.items()
        if any(key.startswith("val-") for key in metrics)
    }


def _filter_train_steps(steps: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        step_id: metrics
        for step_id, metrics in steps.items()
        if "training/global_step" in metrics or any(key.startswith("critic/") for key in metrics)
    }


def _initial_validation_metrics(validation_steps: dict[str, dict[str, float]]) -> dict[str, float]:
    if not validation_steps:
        return {}
    first_step = min(validation_steps, key=lambda step_id: int(step_id))
    return validation_steps[first_step]


def _build_reward_curve(train_steps: dict[str, dict[str, float]]) -> list[dict[str, float | int]]:
    curve = []
    for step_id in sorted(train_steps, key=lambda item: int(item)):
        metrics = train_steps[step_id]
        if "critic/rewards/mean" not in metrics and "critic/score/mean" not in metrics:
            continue
        row: dict[str, float | int] = {"step": int(step_id)}
        if "critic/rewards/mean" in metrics:
            row["critic/rewards/mean"] = metrics["critic/rewards/mean"]
        if "critic/score/mean" in metrics:
            row["critic/score/mean"] = metrics["critic/score/mean"]
        curve.append(row)
    return curve


def _build_reward_drop_alerts(reward_curve: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    alerts = []
    previous = None
    for row in reward_curve:
        reward_mean = row.get("critic/rewards/mean")
        if not isinstance(reward_mean, float):
            previous = row
            continue
        if previous is not None:
            previous_reward = previous.get("critic/rewards/mean")
            if isinstance(previous_reward, float):
                delta = round(reward_mean - previous_reward, 6)
                if delta <= -REWARD_DROP_THRESHOLD:
                    alerts.append(
                        {
                            "step": int(row["step"]),
                            "previous_step": int(previous["step"]),
                            "previous_reward_mean": previous_reward,
                            "reward_mean": reward_mean,
                            "delta": delta,
                        }
                    )
        previous = row
    return alerts
