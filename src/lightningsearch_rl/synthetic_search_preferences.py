from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterator, Sequence

from lightningsearch_rl.agent_loop import parse_agent_action
from lightningsearch_rl.verl_reward import compute_score


LOCAL_ROOT_MARKER = "Agent RL"
REMOTE_ROOTS = ("/data/wzl/LightningSearch-RL", "/home/user/wzl/LightningSearch-RL")


def build_synthetic_search_preferences(
    *,
    transitions_path: Path,
    out_dir: Path,
    offset: int = 0,
    limit: int | None = None,
    search_reward_top_k: int = 8,
    min_chosen_score: float = 0.5,
    min_score_gap: float = 0.05,
    max_negatives_per_transition: int = 4,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    _ensure_approved_path(out_dir)
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    if search_reward_top_k < 1:
        raise ValueError("search_reward_top_k must be >= 1")
    if min_chosen_score < 0:
        raise ValueError("min_chosen_score must be >= 0")
    if min_score_gap < 0:
        raise ValueError("min_score_gap must be >= 0")
    if max_negatives_per_transition < 1:
        raise ValueError("max_negatives_per_transition must be >= 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")

    transition_rows = _load_jsonl(transitions_path)
    search_rows = [row for row in transition_rows if _action_type(row) == "search"]
    selected_rows = search_rows[offset:] if limit is None else search_rows[offset : offset + limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    reward_dump_path = out_dir / "reward_dump.jsonl"
    reward_dump_path.write_text("", encoding="utf-8")
    candidates: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    skip_reason_counts: dict[str, int] = {}

    with _temporary_reward_env(reward_dump_path):
        for request_index, row in enumerate(selected_rows, start=offset):
            transition_id = _transition_id(row, request_index)
            source_id = _source_id(row, transition_id)
            chosen_query = _chosen_query(row)
            if not chosen_query:
                _increment(skip_reason_counts, "missing_chosen_query")
                continue
            extra_info = _extra_info(
                row,
                request_index=request_index,
                transition_id=transition_id,
                source_id=source_id,
                search_reward_top_k=search_reward_top_k,
            )
            chosen = _score_candidate(
                row=row,
                request_index=request_index,
                candidate_index=0,
                source_id=source_id,
                transition_id=transition_id,
                query=chosen_query,
                corruption_type="chosen",
                is_chosen=True,
                extra_info=extra_info,
            )
            candidates.append(chosen)
            if chosen["score"] < min_chosen_score:
                _increment(skip_reason_counts, "chosen_score_below_min")
                continue

            rejected_candidates = []
            for candidate_index, negative in enumerate(
                _negative_queries(row, chosen_query),
                start=1,
            ):
                rejected = _score_candidate(
                    row=row,
                    request_index=request_index,
                    candidate_index=candidate_index,
                    source_id=source_id,
                    transition_id=transition_id,
                    query=negative["query"],
                    corruption_type=negative["corruption_type"],
                    is_chosen=False,
                    extra_info=extra_info,
                )
                candidates.append(rejected)
                score_gap = round(chosen["score"] - rejected["score"], 6)
                if score_gap >= min_score_gap:
                    rejected_candidates.append((score_gap, rejected))
            if not rejected_candidates:
                _increment(skip_reason_counts, "no_negative_above_gap")
                continue
            rejected_candidates.sort(
                key=lambda item: (
                    -float(item[0]),
                    float(item[1]["score"]),
                    str(item[1]["corruption_type"]),
                    int(item[1]["candidate_index"]),
                )
            )
            for score_gap, rejected in rejected_candidates[:max_negatives_per_transition]:
                pairs.append(
                    _pair_from_candidates(
                        row=row,
                        request_index=request_index,
                        chosen=chosen,
                        rejected=rejected,
                        score_gap=score_gap,
                    )
                )

    train_pairs, val_pairs = _split_train_val(pairs, val_fraction=val_fraction, seed=seed)
    _write_jsonl(out_dir / "candidates.jsonl", candidates)
    _write_jsonl(out_dir / "pairs.jsonl", pairs)
    _write_jsonl(out_dir / "train.jsonl", train_pairs)
    _write_jsonl(out_dir / "val.jsonl", val_pairs)
    summary = {
        "transitions_path": str(transitions_path),
        "out_dir": str(out_dir),
        "offset": offset,
        "limit": limit,
        "search_reward_top_k": search_reward_top_k,
        "min_chosen_score": min_chosen_score,
        "min_score_gap": min_score_gap,
        "max_negatives_per_transition": max_negatives_per_transition,
        "val_fraction": val_fraction,
        "seed": seed,
        "input_transition_count": len(transition_rows),
        "search_transition_count": len(search_rows),
        "selected_transition_count": len(selected_rows),
        "candidate_count": len(candidates),
        "pair_count": len(pairs),
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        "candidate_corruption_type_counts": _corruption_counts(candidates),
        "pair_corruption_type_counts": _rejected_corruption_counts(pairs),
        "pair_category_counts": _pair_category_counts(pairs),
        "stage_pair_counts": _stage_counts(pairs),
        "artifacts": {
            "candidates": str(out_dir / "candidates.jsonl"),
            "pairs": str(out_dir / "pairs.jsonl"),
            "train": str(out_dir / "train.jsonl"),
            "val": str(out_dir / "val.jsonl"),
            "reward_dump": str(reward_dump_path),
            "summary": str(out_dir / "summary.json"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _score_candidate(
    *,
    row: dict[str, Any],
    request_index: int,
    candidate_index: int,
    source_id: str,
    transition_id: str,
    query: str,
    corruption_type: str,
    is_chosen: bool,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    solution = _search_action(query)
    candidate_extra = {
        **extra_info,
        "probe_sample_index": candidate_index,
        "synthetic_corruption_type": corruption_type,
    }
    reward = compute_score(
        data_source="lightningsearch_rl",
        solution_str=solution,
        ground_truth="",
        extra_info=candidate_extra,
    )
    action = parse_agent_action(solution)
    return {
        "candidate_id": f"{transition_id}:search:{request_index}:{candidate_index}:{corruption_type}",
        "source_id": source_id,
        "transition_id": transition_id,
        "request_index": request_index,
        "candidate_index": candidate_index,
        "reward_stage": "search",
        "solution": solution,
        "query": query,
        "score": round(float(reward["score"]), 6),
        "reward": reward,
        "action_type": action.type if action.valid else "invalid",
        "action_valid": action.valid,
        "action_reason": action.reason,
        "corruption_type": corruption_type,
        "is_chosen": is_chosen,
        "expected_action": str(row.get("action", "")),
    }


def _pair_from_candidates(
    *,
    row: dict[str, Any],
    request_index: int,
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    score_gap: float,
) -> dict[str, Any]:
    metadata = _dict(row.get("metadata"))
    return {
        "pair_id": (
            f"{chosen['transition_id']}:search:{request_index}:"
            f"{chosen['candidate_index']}>{rejected['candidate_index']}:{rejected['corruption_type']}"
        ),
        "source_id": chosen["source_id"],
        "transition_id": chosen["transition_id"],
        "request_index": request_index,
        "reward_stage": "search",
        "prompt": _list(row.get("state_messages")),
        "ground_truth": "",
        "chosen": chosen["solution"],
        "rejected": rejected["solution"],
        "chosen_score": chosen["score"],
        "rejected_score": rejected["score"],
        "score_gap": round(score_gap, 6),
        "pair_category": "search_vs_search",
        "chosen_action_type": chosen["action_type"],
        "rejected_action_type": rejected["action_type"],
        "chosen_action_valid": chosen["action_valid"],
        "rejected_action_valid": rejected["action_valid"],
        "chosen_sample_index": chosen["candidate_index"],
        "rejected_sample_index": rejected["candidate_index"],
        "chosen_reward": chosen["reward"],
        "rejected_reward": rejected["reward"],
        "chosen_query": chosen["query"],
        "rejected_query": rejected["query"],
        "rejected_corruption_type": rejected["corruption_type"],
        "expected_action": str(row.get("action", "")),
        "gold_doc_ids": _gold_doc_ids(row),
        "gold_answer": str(metadata.get("gold_answer") or ""),
    }


def _negative_queries(row: dict[str, Any], chosen_query: str) -> list[dict[str, str]]:
    metadata = _dict(row.get("metadata"))
    candidate_passages = _candidate_passages(row)
    gold_ids = set(_gold_doc_ids(row))
    gold_titles = [
        str(passage.get("title") or "").strip()
        for passage in candidate_passages
        if str(passage.get("doc_id") or "").strip() in gold_ids and str(passage.get("title") or "").strip()
    ]
    distractor_title = next(
        (
            str(passage.get("title") or "").strip()
            for passage in candidate_passages
            if str(passage.get("doc_id") or "").strip() not in gold_ids
            and str(passage.get("title") or "").strip()
        ),
        "",
    )
    answer = str(metadata.get("gold_answer") or row.get("gold_answer") or "").strip()
    question = _question(row)

    raw_candidates = [
        ("generic", "related evidence"),
        ("distractor_title", distractor_title),
        ("gold_title_only", gold_titles[0] if gold_titles else ""),
        ("gold_title_only", gold_titles[1] if len(gold_titles) > 1 else ""),
        ("answer_only", answer),
        ("entity_dropout", _drop_phrase(chosen_query, [*gold_titles, answer])),
        ("relation_dropout", _entity_like_query(chosen_query)),
        ("question_keywords", _question_keywords(question)),
    ]
    negatives = []
    seen = {_normalize_text(chosen_query), ""}
    for corruption_type, query in raw_candidates:
        normalized = _normalize_text(query)
        if normalized in seen:
            continue
        seen.add(normalized)
        negatives.append({"corruption_type": corruption_type, "query": _clean_query(query)})
    return negatives


def _drop_phrase(query: str, phrases: Sequence[str]) -> str:
    cleaned = query
    for phrase in sorted((p for p in phrases if p), key=len, reverse=True):
        phrase_tokens = re.escape(phrase)
        next_value = re.sub(phrase_tokens, " ", cleaned, flags=re.IGNORECASE)
        if _normalize_text(next_value) != _normalize_text(cleaned):
            cleaned = next_value
            break
    if _normalize_text(cleaned) == _normalize_text(query):
        tokens = query.split()
        if len(tokens) > 2:
            cleaned = " ".join(tokens[:-2])
    return _clean_query(cleaned)


def _entity_like_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    kept = [
        token
        for token in tokens
        if token[:1].isupper() or token.isdigit() or len(token) > 8
    ]
    if len(kept) >= 2:
        return _clean_query(" ".join(kept))
    return _clean_query(" ".join(tokens[: max(1, len(tokens) // 2)]))


def _question_keywords(question: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", question)
        if token.lower() not in {"which", "what", "when", "where", "who", "did", "the", "by", "in", "of", "a", "an"}
    ]
    return _clean_query(" ".join(tokens[:4]))


def _chosen_query(row: dict[str, Any]) -> str:
    query = str(row.get("query") or "").strip()
    if query:
        return _clean_query(query)
    action = parse_agent_action(str(row.get("action") or ""))
    if action.valid and action.type == "search" and action.query:
        return _clean_query(action.query)
    return ""


def _extra_info(
    row: dict[str, Any],
    *,
    request_index: int,
    transition_id: str,
    source_id: str,
    search_reward_top_k: int,
) -> dict[str, Any]:
    metadata = _dict(row.get("metadata"))
    return {
        "id": transition_id,
        "source_id": source_id,
        "index": request_index,
        "answer": metadata.get("gold_answer", ""),
        "search_count": 1,
        "gold_doc_ids": _gold_doc_ids(row),
        "retrieved_doc_ids": _list(row.get("observation_doc_ids")),
        "candidate_passages": _candidate_passages(row),
        "search_reward_top_k": search_reward_top_k,
        "reward_stage": "search",
        "expected_action": str(row.get("action", "")),
        "precomputed_step_reward": _float(row.get("reward")),
        "precomputed_total_reward": _float(metadata.get("total_reward")),
    }


def _candidate_passages(row: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = _dict(row.get("metadata"))
    passages = _list(row.get("candidate_passages"))
    if passages:
        return [passage for passage in passages if isinstance(passage, dict)]
    return [passage for passage in _list(metadata.get("candidate_passages")) if isinstance(passage, dict)]


def _gold_doc_ids(row: dict[str, Any]) -> list[str]:
    metadata = _dict(row.get("metadata"))
    return _strings(
        row.get("gold_evidence_doc_ids")
        or row.get("gold_doc_ids")
        or metadata.get("gold_evidence_doc_ids")
        or metadata.get("gold_doc_ids")
    )


def _question(row: dict[str, Any]) -> str:
    metadata = _dict(row.get("metadata"))
    question = str(metadata.get("question") or row.get("question") or "").strip()
    if question:
        return question
    for message in reversed(_list(row.get("state_messages"))):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


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


@contextmanager
def _temporary_reward_env(reward_dump_path: Path) -> Iterator[None]:
    updates = {
        "LIGHTNINGSEARCH_REWARD_DUMP_PATH": str(reward_dump_path),
        "LIGHTNINGSEARCH_REWARD_DUMP_MAX_CHARS": "1024",
    }
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _search_action(query: str) -> str:
    return f"<search>{_clean_query(query)}</search>"


def _clean_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _action_type(row: dict[str, Any]) -> str:
    action_type = str(row.get("action_type", "")).strip().lower() or "answer"
    return action_type if action_type in {"search", "answer"} else "answer"


def _transition_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("transition_id") or f"{row.get('id', 'transition')}:{index}:search")


def _source_id(row: dict[str, Any], transition_id: str) -> str:
    return str(row.get("id") or transition_id.split(":", 1)[0])


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str(row.get("reward_stage") or "unknown")
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _pair_category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("pair_category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _corruption_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        corruption_type = str(row.get("corruption_type") or "unknown")
        counts[corruption_type] = counts.get(corruption_type, 0) + 1
    return dict(sorted(counts.items()))


def _rejected_corruption_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        corruption_type = str(row.get("rejected_corruption_type") or "unknown")
        counts[corruption_type] = counts.get(corruption_type, 0) + 1
    return dict(sorted(counts.items()))


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


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
