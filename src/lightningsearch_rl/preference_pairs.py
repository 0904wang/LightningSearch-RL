from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

from lightningsearch_rl.agent_loop import parse_agent_action


DEFAULT_STAGES = ("search", "answer")
LOCAL_ROOT_MARKER = "Agent RL"
REMOTE_ROOTS = ("/data/wzl/LightningSearch-RL", "/home/user/wzl/LightningSearch-RL")


def build_preference_pairs(
    *,
    probe_requests_path: Path,
    generations_path: Path,
    reward_dump_path: Path,
    out_dir: Path,
    stages: Sequence[str] = DEFAULT_STAGES,
    min_score_gap: float = 0.25,
    min_samples: int = 2,
    max_pairs_per_group: int = 1,
    max_search_pairs: int | None = None,
    max_answer_pairs: int | None = None,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    _ensure_approved_path(out_dir)
    if min_score_gap < 0:
        raise ValueError("min_score_gap must be >= 0")
    if min_samples < 2:
        raise ValueError("min_samples must be >= 2")
    if max_pairs_per_group < 1:
        raise ValueError("max_pairs_per_group must be >= 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")

    selected_stages = tuple(stages) if stages else DEFAULT_STAGES
    stage_set = {str(stage).strip().lower() for stage in selected_stages}
    requests = _load_jsonl(probe_requests_path)
    generations = _load_jsonl(generations_path)
    reward_rows = _load_jsonl(reward_dump_path)
    request_by_index = {_int(row.get("request_index")): row for row in requests}
    reward_by_key = {_reward_key(row): row for row in reward_rows}

    grouped_generations: dict[int, list[dict[str, Any]]] = {}
    for row in generations:
        grouped_generations.setdefault(_int(row.get("request_index")), []).append(row)

    candidate_pairs = []
    skipped_groups = 0
    for request_index in sorted(grouped_generations):
        request = request_by_index.get(request_index)
        if request is None:
            skipped_groups += 1
            continue
        request_extra = _dict(request.get("extra_info"))
        stage = str(request_extra.get("reward_stage") or "").strip().lower()
        if not stage:
            stage = _stage_from_generations(grouped_generations[request_index])
        if stage not in stage_set:
            continue

        candidates = _unique_candidates(
            request=request,
            generations=grouped_generations[request_index],
            reward_by_key=reward_by_key,
        )
        if len(candidates) < min_samples:
            skipped_groups += 1
            continue
        candidate_pairs.extend(
            _pairs_for_group(
                request=request,
                request_index=request_index,
                stage=stage,
                candidates=candidates,
                min_score_gap=min_score_gap,
                max_pairs_per_group=max_pairs_per_group,
            )
        )

    selected_pairs = _apply_stage_caps(
        candidate_pairs,
        max_search_pairs=max_search_pairs,
        max_answer_pairs=max_answer_pairs,
    )
    train_pairs, val_pairs = _split_train_val(selected_pairs, val_fraction=val_fraction, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "pairs.jsonl", selected_pairs)
    _write_jsonl(out_dir / "train.jsonl", train_pairs)
    _write_jsonl(out_dir / "val.jsonl", val_pairs)
    summary = {
        "probe_requests_path": str(probe_requests_path),
        "generations_path": str(generations_path),
        "reward_dump_path": str(reward_dump_path),
        "out_dir": str(out_dir),
        "stages": list(selected_stages),
        "min_score_gap": min_score_gap,
        "min_samples": min_samples,
        "max_pairs_per_group": max_pairs_per_group,
        "max_search_pairs": max_search_pairs,
        "max_answer_pairs": max_answer_pairs,
        "val_fraction": val_fraction,
        "seed": seed,
        "request_count": len(requests),
        "generation_count": len(generations),
        "reward_dump_count": len(reward_rows),
        "candidate_pair_count": len(candidate_pairs),
        "pair_count": len(selected_pairs),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "skipped_group_count": skipped_groups,
        "stage_candidate_pair_counts": _stage_counts(candidate_pairs),
        "stage_pair_counts": _stage_counts(selected_pairs),
        "artifacts": {
            "pairs": str(out_dir / "pairs.jsonl"),
            "train": str(out_dir / "train.jsonl"),
            "val": str(out_dir / "val.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _unique_candidates(
    *,
    request: dict[str, Any],
    generations: list[dict[str, Any]],
    reward_by_key: dict[tuple[int, int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    best_by_action: dict[str, dict[str, Any]] = {}
    for generation in generations:
        solution = str(generation.get("solution") or "")
        action_key = _normalized_action_key(solution)
        candidate = _candidate_from_generation(
            request=request,
            generation=generation,
            reward_row=reward_by_key.get(_generation_reward_key(generation)),
            action_key=action_key,
        )
        current = best_by_action.get(action_key)
        if current is None or _candidate_sort_key(candidate) < _candidate_sort_key(current):
            best_by_action[action_key] = candidate
    candidates = list(best_by_action.values())
    candidates.sort(key=_candidate_sort_key)
    return candidates


def _candidate_from_generation(
    *,
    request: dict[str, Any],
    generation: dict[str, Any],
    reward_row: dict[str, Any] | None,
    action_key: str,
) -> dict[str, Any]:
    reward_payload = _reward_payload(generation, reward_row)
    request_extra = _dict(request.get("extra_info"))
    return {
        "request_index": _int(generation.get("request_index")),
        "sample_index": _int(generation.get("sample_index")),
        "source_id": str(generation.get("source_id") or request_extra.get("source_id") or ""),
        "transition_id": str(generation.get("id") or request_extra.get("id") or ""),
        "reward_stage": str(generation.get("reward_stage") or request_extra.get("reward_stage") or "").strip().lower(),
        "solution": str(generation.get("solution") or ""),
        "score": round(_float(reward_payload.get("score", generation.get("score"))), 6),
        "reward": reward_payload,
        "action_key": action_key,
    }


def _pairs_for_group(
    *,
    request: dict[str, Any],
    request_index: int,
    stage: str,
    candidates: list[dict[str, Any]],
    min_score_gap: float,
    max_pairs_per_group: int,
) -> list[dict[str, Any]]:
    request_extra = _dict(request.get("extra_info"))
    group_pairs = []
    for chosen in candidates:
        for rejected in reversed(candidates):
            if chosen["sample_index"] == rejected["sample_index"]:
                continue
            score_gap = round(chosen["score"] - rejected["score"], 6)
            if score_gap < min_score_gap:
                continue
            group_pairs.append(
                {
                    "pair_id": (
                        f"{chosen['transition_id']}:{stage}:{request_index}:"
                        f"{chosen['sample_index']}>{rejected['sample_index']}"
                    ),
                    "source_id": chosen["source_id"],
                    "transition_id": chosen["transition_id"],
                    "request_index": request_index,
                    "reward_stage": stage,
                    "prompt": request.get("prompt", []),
                    "ground_truth": request.get("ground_truth", ""),
                    "chosen": chosen["solution"],
                    "rejected": rejected["solution"],
                    "chosen_score": chosen["score"],
                    "rejected_score": rejected["score"],
                    "score_gap": score_gap,
                    "chosen_sample_index": chosen["sample_index"],
                    "rejected_sample_index": rejected["sample_index"],
                    "chosen_reward": chosen["reward"],
                    "rejected_reward": rejected["reward"],
                    "expected_action": request_extra.get("expected_action", ""),
                    "gold_doc_ids": _list(request_extra.get("gold_doc_ids") or request_extra.get("gold_evidence_doc_ids")),
                }
            )
    group_pairs.sort(
        key=lambda row: (
            -float(row["score_gap"]),
            -float(row["chosen_score"]),
            float(row["rejected_score"]),
            int(row["chosen_sample_index"]),
            int(row["rejected_sample_index"]),
        )
    )
    return group_pairs[:max_pairs_per_group]


def _apply_stage_caps(
    pairs: list[dict[str, Any]],
    *,
    max_search_pairs: int | None,
    max_answer_pairs: int | None,
) -> list[dict[str, Any]]:
    caps = {"search": max_search_pairs, "answer": max_answer_pairs}
    selected = []
    counts: dict[str, int] = {}
    for pair in pairs:
        stage = str(pair.get("reward_stage") or "")
        cap = caps.get(stage)
        current = counts.get(stage, 0)
        if cap is not None and current >= cap:
            continue
        selected.append(pair)
        counts[stage] = current + 1
    return selected


def _split_train_val(
    pairs: list[dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not pairs or val_fraction <= 0:
        return list(pairs), []
    val_count = int(round(len(pairs) * val_fraction))
    if val_count <= 0:
        return list(pairs), []
    val_count = min(len(pairs) - 1, val_count)
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    val_ids = {row["pair_id"] for row in shuffled[:val_count]}
    train = [row for row in pairs if row["pair_id"] not in val_ids]
    val = [row for row in pairs if row["pair_id"] in val_ids]
    return train, val


def _reward_payload(generation: dict[str, Any], reward_row: dict[str, Any] | None) -> dict[str, Any]:
    if reward_row is None:
        reward = _dict(generation.get("reward"))
        if "score" not in reward:
            reward["score"] = _float(generation.get("score"))
        return reward
    keys = (
        "score",
        "answer_reward",
        "answer_exact_match",
        "answer_token_f1",
        "answer_containment_match",
        "search_reward",
        "evidence_rank_reward",
        "gold_top_rank",
        "retrieved_gold_count",
        "format_reward",
        "search_cost",
        "answer_reward_type",
    )
    return {key: reward_row.get(key) for key in keys if key in reward_row}


def _reward_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    extra = _dict(row.get("extra_info"))
    return (
        _int(extra.get("index")),
        _int(extra.get("probe_sample_index")),
        str(extra.get("id") or "").strip(),
        str(row.get("reward_stage") or extra.get("reward_stage") or "").strip().lower(),
    )


def _generation_reward_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        _int(row.get("request_index")),
        _int(row.get("sample_index")),
        str(row.get("id") or "").strip(),
        str(row.get("reward_stage") or "").strip().lower(),
    )


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int]:
    return (-float(candidate["score"]), int(candidate["sample_index"]))


def _normalized_action_key(solution: str) -> str:
    action = parse_agent_action(solution)
    if action.valid and action.type == "search":
        return "search:" + _normalize_text(action.query)
    if action.valid and action.type == "answer":
        return "answer:" + _normalize_text(action.answer)
    return "invalid:" + _normalize_text(solution)


def _stage_from_generations(generations: list[dict[str, Any]]) -> str:
    for row in generations:
        stage = str(row.get("reward_stage") or "").strip().lower()
        if stage:
            return stage
    return "unknown"


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("reward_stage") or "unknown")
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _ensure_approved_path(path: Path) -> None:
    normalized = str(path).replace("\\", "/")
    if any(normalized.startswith(root) for root in REMOTE_ROOTS):
        return
    if LOCAL_ROOT_MARKER in normalized:
        return
    raise ValueError(f"path is outside approved paths: {path}")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())
