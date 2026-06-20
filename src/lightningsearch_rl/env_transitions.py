from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lightningsearch_rl.corpus import passage_to_dict
from lightningsearch_rl.answer_metrics import soft_answer_reward
from lightningsearch_rl.index_store import load_lexical_index


SEARCH_COST = 0.03
LOCAL_ROOT_MARKER = "Agent RL"
REMOTE_ROOTS = ("/data/wzl/LightningSearch-RL", "/home/user/wzl/LightningSearch-RL")


def export_env_rollout_transitions(
    rollouts_path: Path,
    out_dir: Path,
    quality_manifest_path: Path | None = None,
    exclude_quality_flags: set[str] | None = None,
    index_path: Path | None = None,
) -> dict[str, Any]:
    _ensure_approved_path(out_dir)
    rows = _load_jsonl(rollouts_path)
    passage_lookup = _load_passage_lookup(index_path)
    rows = [_row_with_candidate_passages(row, passage_lookup) for row in rows]
    quality_manifest = _load_quality_manifest(quality_manifest_path)
    rows = [_row_with_quality(row, quality_manifest) for row in rows]
    excluded_flags = exclude_quality_flags or set()
    kept_rows = []
    excluded_rows = []
    for row in rows:
        if excluded_flags.intersection(set(_list(row.get("quality_flags", [])))):
            excluded_rows.append(row)
        else:
            kept_rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)

    reward_records = [_build_reward_record(row) for row in kept_rows]
    transitions = [
        transition
        for row, reward in zip(kept_rows, reward_records)
        for transition in _build_transitions(row, reward)
    ]
    rollouts_for_grpo = [
        _build_rollout_for_grpo(row, reward) for row, reward in zip(kept_rows, reward_records)
    ]

    _write_jsonl(out_dir / "transitions.jsonl", transitions)
    _write_jsonl(out_dir / "reward_records.jsonl", reward_records)
    _write_jsonl(out_dir / "rollouts_for_grpo.jsonl", rollouts_for_grpo)
    summary = {
        "input_example_count": len(rows),
        "example_count": len(kept_rows),
        "excluded_example_count": len(excluded_rows),
        "excluded_example_ids": [str(row.get("id", "")) for row in excluded_rows],
        "quality_flag_counts": _count_quality_flags(rows),
        "excluded_quality_flag_counts": _count_quality_flags(excluded_rows),
        "rollout_count": len(rollouts_for_grpo),
        "transition_count": len(transitions),
        "avg_total_reward": _average(record["total"] for record in reward_records),
        "avg_search_credit": _average(record["search_credit"] for record in reward_records),
        "avg_answer_credit": _average(record["answer_credit"] for record in reward_records),
        "avg_candidate_passage_count": _average(len(_list(row.get("candidate_passages", []))) for row in kept_rows),
        "valid_search_action_rate": _rate(record["valid_search_action"] for record in reward_records),
        "valid_answer_action_rate": _rate(record["valid_answer_action"] for record in reward_records),
        "answer_exact_match_rate": _rate(record["answer_exact_match"] for record in reward_records),
        "answer_containment_match_rate": _rate(record["answer_containment_match"] for record in reward_records),
        "answer_token_f1": _average(record["answer_token_f1"] for record in reward_records),
        "gold_evidence_recall": _average(record["evidence_reward"] for record in reward_records),
        "artifacts": {
            "transitions": str(out_dir / "transitions.jsonl"),
            "reward_records": str(out_dir / "reward_records.jsonl"),
            "rollouts_for_grpo": str(out_dir / "rollouts_for_grpo.jsonl"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _build_reward_record(row: dict[str, Any]) -> dict[str, Any]:
    answer_score = _answer_score(row)
    valid_search = _valid_search_action(row)
    valid_answer = _valid_answer_action(row)
    search_count = 1 if valid_search else 0
    answer_reward = float(answer_score["answer_reward"])
    evidence_reward = _float(row.get("gold_evidence_recall", 0.0))
    format_reward = 1.0 if valid_search and valid_answer else 0.0
    tool_validity_reward = 1.0 if valid_search else 0.0
    search_cost = round(SEARCH_COST * search_count, 6)
    search_credit = round(0.2 * evidence_reward + 0.1 * tool_validity_reward - search_cost, 6)
    answer_credit = round(answer_reward + 0.1 * format_reward, 6)
    total = round(search_credit + answer_credit, 6)
    return {
        "id": row.get("id"),
        "answer_reward": answer_reward,
        "answer_reward_type": answer_score["answer_reward_type"],
        "evidence_reward": evidence_reward,
        "format_reward": format_reward,
        "tool_validity_reward": tool_validity_reward,
        "search_count": search_count,
        "search_cost": search_cost,
        "search_credit": search_credit,
        "answer_credit": answer_credit,
        "total": total,
        "valid_search_action": valid_search,
        "valid_answer_action": valid_answer,
        "answer_exact_match": bool(answer_score["exact_match"]),
        "answer_token_f1": answer_score["token_f1"],
        "answer_containment_match": bool(answer_score["containment_match"]),
        "final_answer": str(row.get("final_answer", "")),
        "gold_answer": str(row.get("gold_answer", "")),
        "gold_evidence_doc_ids": _list(row.get("gold_evidence_doc_ids", [])),
        "observation_doc_ids": _list(row.get("observation_doc_ids", [])),
        "candidate_passages": _list(row.get("candidate_passages", [])),
        "quality_flags": _list(row.get("quality_flags", [])),
        "quality_notes": _list(row.get("quality_notes", [])),
    }


def _build_transitions(row: dict[str, Any], reward: dict[str, Any]) -> list[dict[str, Any]]:
    row_id = str(row.get("id", ""))
    search_action = _dict(row.get("search_action"))
    answer_action = _dict(row.get("answer_action"))
    has_answer_step = bool(str(row.get("answer_generated", "")).strip()) or bool(row.get("answer_messages"))
    transitions = [
        {
            "id": row.get("id"),
            "transition_id": f"{row_id}:0:{search_action.get('type', 'invalid')}",
            "step_index": 0,
            "state": _messages_to_text(_list(row.get("search_messages", []))),
            "state_messages": _list(row.get("search_messages", [])),
            "action": str(row.get("search_generated", "")),
            "action_type": str(search_action.get("type", "invalid")),
            "valid_action": bool(search_action.get("valid")),
            "query": search_action.get("query"),
            "observation": row.get("observation"),
            "observation_doc_ids": _list(row.get("observation_doc_ids", [])),
            "gold_evidence_doc_ids": _list(row.get("gold_evidence_doc_ids", [])),
            "candidate_passages": _list(row.get("candidate_passages", [])),
            "terminal": not has_answer_step,
            "reward": reward["search_credit"],
            "reward_components": {
                "evidence_reward": reward["evidence_reward"],
                "tool_validity_reward": reward["tool_validity_reward"],
                "search_cost": reward["search_cost"],
                "step_credit": reward["search_credit"],
            },
            "metadata": _transition_metadata(row, reward),
        }
    ]
    if has_answer_step:
        transitions.append(
            {
                "id": row.get("id"),
                "transition_id": f"{row_id}:1:{answer_action.get('type', 'invalid')}",
                "step_index": 1,
                "state": _messages_to_text(_list(row.get("answer_messages", []))),
                "state_messages": _list(row.get("answer_messages", [])),
                "action": str(row.get("answer_generated", "")),
                "action_type": str(answer_action.get("type", "invalid")),
                "valid_action": bool(answer_action.get("valid")),
                "answer": answer_action.get("answer"),
                "observation": None,
                "observation_doc_ids": _list(row.get("observation_doc_ids", [])),
                "gold_evidence_doc_ids": _list(row.get("gold_evidence_doc_ids", [])),
                "candidate_passages": _list(row.get("candidate_passages", [])),
                "terminal": True,
                "reward": reward["answer_credit"],
                "reward_components": {
                    "answer_reward": reward["answer_reward"],
                    "answer_reward_type": reward["answer_reward_type"],
                    "format_reward": reward["format_reward"],
                    "step_credit": reward["answer_credit"],
                },
                "metadata": _transition_metadata(row, reward),
            }
        )
    return transitions


def _build_rollout_for_grpo(row: dict[str, Any], reward: dict[str, Any]) -> dict[str, Any]:
    response = "\n".join(
        part
        for part in [
            str(row.get("search_generated", "")).strip(),
            str(row.get("observation", "")).strip(),
            str(row.get("answer_generated", "")).strip(),
        ]
        if part
    )
    return {
        "id": row.get("id"),
        "prompt": str(row.get("question", "")),
        "response": response,
        "reward": reward["total"],
        "metadata": {
            "answer": str(row.get("gold_answer", "")),
            "final_answer": str(row.get("final_answer", "")),
            "search_count": reward["search_count"],
            "gold_doc_ids": _list(row.get("gold_evidence_doc_ids", [])),
            "retrieved_doc_ids": _list(row.get("observation_doc_ids", [])),
            "candidate_doc_ids": _list(row.get("candidate_doc_ids", [])),
            "candidate_passages": _list(row.get("candidate_passages", [])),
            "candidate_pool": row.get("candidate_pool"),
            "quality_flags": _list(row.get("quality_flags", [])),
            "quality_notes": _list(row.get("quality_notes", [])),
            "reward": reward["total"],
        },
    }


def _transition_metadata(row: dict[str, Any], reward: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "final_answer": row.get("final_answer"),
        "candidate_doc_ids": _list(row.get("candidate_doc_ids", [])),
        "candidate_passages": _list(row.get("candidate_passages", [])),
        "candidate_pool": row.get("candidate_pool"),
        "answer_exact_match": reward["answer_exact_match"],
        "answer_token_f1": reward["answer_token_f1"],
        "answer_containment_match": reward["answer_containment_match"],
        "answer_reward_type": reward["answer_reward_type"],
        "quality_flags": _list(row.get("quality_flags", [])),
        "quality_notes": _list(row.get("quality_notes", [])),
        "total_reward": reward["total"],
    }


def _load_quality_manifest(path: Path | None) -> dict[str, dict[str, list[str]]]:
    if path is None:
        return {}
    _ensure_approved_path(path)
    if path.suffix == ".jsonl":
        rows = _load_jsonl(path)
        return _normalize_quality_manifest_rows(rows)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _normalize_quality_manifest_rows(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            return _normalize_quality_manifest_rows(payload["rows"])
        return {
            str(row_id): _normalize_quality_entry(entry)
            for row_id, entry in payload.items()
            if str(row_id).strip()
        }
    raise ValueError(f"unsupported quality manifest format: {path}")


def _load_passage_lookup(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    _ensure_approved_path(path)
    retriever = load_lexical_index(path)
    return {passage.doc_id: passage_to_dict(passage) for passage in retriever.passages}


def _row_with_candidate_passages(
    row: dict[str, Any],
    passage_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if _list(row.get("candidate_passages", [])) or not passage_lookup:
        return row
    passages = [
        passage_lookup[doc_id]
        for doc_id in _strings(row.get("candidate_doc_ids", []))
        if doc_id in passage_lookup
    ]
    if not passages:
        return row
    return {**row, "candidate_passages": passages}


def _normalize_quality_manifest_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    manifest = {}
    for row in rows:
        row_id = str(row.get("id") or row.get("example_id") or "").strip()
        if row_id:
            manifest[row_id] = _normalize_quality_entry(row)
    return manifest


def _normalize_quality_entry(entry: Any) -> dict[str, list[str]]:
    if isinstance(entry, str):
        return {"flags": [entry], "notes": []}
    if not isinstance(entry, dict):
        return {"flags": [], "notes": []}
    flags = _strings(entry.get("flags") or entry.get("quality_flags"))
    notes = _strings(entry.get("notes") or entry.get("quality_notes") or entry.get("note"))
    reason = entry.get("reason")
    if isinstance(reason, str) and reason.strip():
        notes.append(reason.strip())
    return {"flags": sorted(set(flags)), "notes": notes}


def _row_with_quality(
    row: dict[str, Any],
    manifest: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    quality = manifest.get(str(row.get("id", "")), {"flags": [], "notes": []})
    return {
        **row,
        "quality_flags": quality["flags"],
        "quality_notes": quality["notes"],
    }


def _count_quality_flags(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for flag in _list(row.get("quality_flags", [])):
            flag_name = str(flag)
            counts[flag_name] = counts.get(flag_name, 0) + 1
    return dict(sorted(counts.items()))


def _answer_score(row: dict[str, Any]) -> dict[str, float | bool | str]:
    if {"answer_exact_match", "answer_token_f1", "answer_containment_match"} <= set(row):
        token_score = _float(row.get("answer_token_f1", 0.0))
        exact = bool(row.get("answer_exact_match"))
        containment = bool(row.get("answer_containment_match"))
        if exact:
            answer_reward = 1.0
            reward_type = "exact"
        elif containment:
            answer_reward = max(0.5, token_score)
            reward_type = "containment"
        elif token_score >= 0.75:
            answer_reward = token_score
            reward_type = "token_f1"
        else:
            answer_reward = 0.0
            reward_type = "none"
        return {
            "exact_match": bool(row.get("answer_exact_match")),
            "token_f1": token_score,
            "containment_match": bool(row.get("answer_containment_match")),
            "answer_reward": round(answer_reward, 6),
            "answer_reward_type": reward_type,
        }
    return soft_answer_reward(str(row.get("final_answer", "")), str(row.get("gold_answer", "")))


def _valid_search_action(row: dict[str, Any]) -> bool:
    action = _dict(row.get("search_action"))
    query = str(action.get("query") or "").strip()
    generated = str(row.get("search_generated", ""))
    return (
        bool(action.get("valid"))
        and action.get("type") == "search"
        and bool(query)
        and "<observation>" not in generated
        and "</observation>" not in generated
    )


def _valid_answer_action(row: dict[str, Any]) -> bool:
    action = _dict(row.get("answer_action"))
    answer = str(action.get("answer") or "").strip()
    generated = str(row.get("answer_generated", ""))
    return (
        bool(action.get("valid"))
        and action.get("type") == "answer"
        and bool(answer)
        and "<observation>" not in generated
        and "</observation>" not in generated
    )


def _messages_to_text(messages: list[Any]) -> str:
    parts = []
    for message in messages:
        if isinstance(message, dict):
            parts.append(f"{message.get('role', '')}: {message.get('content', '')}")
    return "\n".join(parts)


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rate(values: Any) -> float:
    items = list(values)
    return round(sum(1 for value in items if value) / len(items), 6) if items else 0.0


def _average(values: Any) -> float:
    items = list(values)
    return round(sum(float(value) for value in items) / len(items), 6) if items else 0.0
